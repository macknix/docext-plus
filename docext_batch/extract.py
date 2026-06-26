"""Per-document field extraction and per-directory CSV assembly.

We call docext's prompt builder + client directly (rather than going through the
Gradio app) so each document is one self-contained VLM request, and the optional
confidence pass is a second request.

Output is one CSV per job in **long format**: one row per (file, entity), with
columns ``filename``, ``entity_type``, ``value`` and (when confidence is enabled)
``confidence``, plus an ``error`` column.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path

import json_repair
import pandas as pd
from loguru import logger

from docext.core.client import sync_request
from docext.core.confidence import get_fields_confidence_score_messages_binary
from docext.core.confidence import get_fields_confidence_score_messages_numeric
from docext.core.prompts import get_fields_messages

from docext_batch.config import Config
from docext_batch.config import Field
from docext_batch.config import Job
from docext_batch.images import list_documents
from docext_batch.images import tiff_to_jpeg


def _normalise(name: str) -> str:
    """Match docext's key normalisation (it lowercases/underscores field names).

    docext builds the prompt and the expected JSON keys from normalised names, so
    we must read the model's response back under the same keys.
    """
    return name.replace(" ", "_").lower()


def _call(messages: list[dict], model_name: str, max_tokens: int, fmt: dict) -> str:
    """Send one request to the VLM and return the raw response text."""
    raw = sync_request(messages, model_name, max_tokens=max_tokens, format=fmt)
    return raw["choices"][0]["message"]["content"]


def _parse(content: str) -> dict:
    """Parse the model's (possibly messy) JSON reply into a dict."""
    parsed = json_repair.loads(content)
    # The model occasionally wraps a single document's answer in a list.
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else {}
    return parsed if isinstance(parsed, dict) else {}


def _confidence_pass(
    messages: list[dict],
    answer: str,
    names: list[str],
    model_name: str,
    max_tokens: int,
    mode: str,
) -> dict:
    """Run docext's confidence pass over a prior answer ("binary" or "numeric")."""
    if mode == "numeric":
        conf_messages = get_fields_confidence_score_messages_numeric(messages, answer, names)
        props = {_normalise(n): {"type": "string"} for n in names}
    else:  # binary
        conf_messages = get_fields_confidence_score_messages_binary(messages, answer, names)
        props = {_normalise(n): {"type": "string", "enum": ["High", "Low"]} for n in names}
    content = _call(
        conf_messages, model_name, max_tokens, {"type": "object", "properties": props},
    )
    return _parse(content)


def extract_fields(
    image_path: Path,
    model_name: str,
    fields: list[Field],
    max_tokens: int,
    confidence: str = "none",
) -> dict[str, dict]:
    """Extract all fields from a single prepared (JPEG) image.

    ``confidence`` is one of "none", "binary" (High/Low) or "numeric" (0-100).
    Returns ``{field_name: {"value": str, "confidence": str | None}}`` keyed by
    the original field names from the config.
    """
    names = [f.name for f in fields]
    descriptions = [f.description for f in fields]
    want_conf = confidence != "none"

    messages = get_fields_messages(names, descriptions, [str(image_path)])
    answer = _call(
        messages,
        model_name,
        max_tokens,
        {"type": "object", "properties": {_normalise(n): {"type": "string"} for n in names}},
    )
    extracted = _parse(answer)

    confidences = (
        _confidence_pass(messages, answer, names, model_name, max_tokens, confidence)
        if want_conf
        else {}
    )

    result: dict[str, dict] = {}
    for name in names:
        key = _normalise(name)
        result[name] = {
            "value": extracted.get(key, ""),
            # Extraction keys are normalised; the confidence pass keys by the
            # original field name - try both.
            "confidence": confidences.get(name, confidences.get(key)) if want_conf else None,
        }
    return result


def _rows_for_doc(
    filename: str,
    fields: list[Field],
    values: dict[str, dict],
    error: str,
    want_conf: bool,
) -> list[dict]:
    """Build the long-format rows (one per field) for a single document."""
    rows = []
    for f in fields:
        row = {"filename": filename, "entity_type": f.name, "value": values[f.name]["value"]}
        if want_conf:
            row["confidence"] = values[f.name]["confidence"]
        row["error"] = error
        rows.append(row)
    return rows


def _process_one(doc: Path, work_dir: Path, cfg: Config, job: Job) -> list[dict]:
    """Convert and extract one document. Errors become rows, not a crash."""
    fields = job.fields
    want_conf = cfg.extraction.confidence != "none"
    # Per-job rotation overrides the global extraction.rotate default.
    rotate = job.rotate if job.rotate is not None else cfg.extraction.rotate
    try:
        jpeg = tiff_to_jpeg(
            doc,
            work_dir,
            max_size=cfg.extraction.max_img_size,
            rotate=rotate,
        )
        values = extract_fields(
            jpeg,
            cfg.model.name,
            fields,
            max_tokens=cfg.extraction.max_tokens,
            confidence=cfg.extraction.confidence,
        )
        return _rows_for_doc(doc.name, fields, values, error="", want_conf=want_conf)
    except Exception as exc:  # noqa: BLE001 - keep the batch going
        logger.exception(f"Failed to process {doc.name}: {exc}")
        empty = {f.name: {"value": "", "confidence": ""} for f in fields}
        return _rows_for_doc(doc.name, fields, empty, error=str(exc), want_conf=want_conf)


def run_job(job: Job, cfg: Config, work_root: Path) -> Path:
    """Process every document in a job's input directory and write its CSV."""
    docs = list_documents(job.input_dir, cfg.extraction.input_extensions)
    logger.info(f"[{job.name}] {len(docs)} document(s) in {job.input_dir}")
    if not docs:
        logger.warning(f"[{job.name}] no documents found; writing empty CSV.")

    work_dir = work_root / job.name
    rows: list[dict] = []
    if docs:
        with ThreadPoolExecutor(max_workers=max(1, cfg.extraction.concurrency)) as pool:
            futures = {
                pool.submit(_process_one, doc, work_dir, cfg, job): doc
                for doc in docs
            }
            for i, fut in enumerate(as_completed(futures), start=1):
                doc = futures[fut]
                rows.extend(fut.result())
                logger.info(f"[{job.name}] {i}/{len(docs)} done ({doc.name})")

    # Long-format columns: filename, entity_type, value, [confidence], error.
    columns = ["filename", "entity_type", "value"]
    if cfg.extraction.confidence != "none":
        columns.append("confidence")
    columns.append("error")

    df = pd.DataFrame(rows, columns=columns)
    if not df.empty:
        # Stable sort by filename keeps each document's fields in declared order
        # (a file's rows are produced contiguously and in order).
        df = df.sort_values("filename", kind="stable").reset_index(drop=True)

    job.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(job.output_csv, index=False)
    logger.info(f"[{job.name}] wrote {len(df)} row(s) -> {job.output_csv}")
    return job.output_csv
