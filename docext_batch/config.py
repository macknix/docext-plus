"""Configuration loading and validation.

The whole batch run is driven by a single YAML file. The important idea is the
``jobs`` list: each job points at one input directory and declares its own set
of fields to extract, so different document sets can have completely different
extraction requirements in one run.
"""
from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from dataclasses import field as dc_field
from pathlib import Path

import yaml


class ConfigError(ValueError):
    """Raised when the config file is missing required values or malformed."""


@dataclass
class Field:
    """A single entity/field to extract from a document."""

    name: str
    description: str = ""


# Default server port per backend, used when server_port is left unset.
_DEFAULT_PORTS = {"vllm": 8000, "ollama": 11434}


@dataclass
class ModelConfig:
    # The model name *selects the backend* via its prefix:
    #   "hosted_vllm/<hf_repo>"  -> local vLLM server (GPU / TRE)
    #   "ollama/<model[:tag]>"   -> a running Ollama server (great for Mac/CPU)
    #   anything else            -> a vendor-hosted / external endpoint
    name: str = "hosted_vllm/Qwen/Qwen2.5-VL-7B-Instruct-AWQ"
    server_host: str = "127.0.0.1"
    # Leave as None to use the backend default (8000 vLLM, 11434 Ollama).
    server_port: int | None = None
    # Only meaningful for the vLLM backend: start vLLM inside the container.
    start_server: bool = True
    # vLLM server tuning (ignored by other backends).
    max_model_len: int = 15000
    gpu_memory_utilization: float = 0.9
    max_num_imgs: int = 5
    server_start_timeout: int = 600
    dtype: str = "bfloat16"
    # Ollama only: how long Ollama keeps the model resident. "-1" pins it in
    # memory for the whole run so it is loaded exactly once (no reload between
    # documents). Use e.g. "5m" to let Ollama unload it after the run.
    ollama_keep_alive: str = "-1"

    @property
    def backend(self) -> str:
        if self.name.startswith("hosted_vllm/"):
            return "vllm"
        if self.name.startswith("ollama/"):
            return "ollama"
        return "external"

    @property
    def port(self) -> int:
        if self.server_port is not None:
            return self.server_port
        return _DEFAULT_PORTS.get(self.backend, 8000)

    @property
    def url(self) -> str:
        """Base URL the docext client expects in VLM_MODEL_URL."""
        base = f"http://{self.server_host}:{self.port}"
        # docext appends /v1 only for vLLM (OpenAI-compatible); Ollama uses the root.
        return f"{base}/v1" if self.backend == "vllm" else base


@dataclass
class ExtractionConfig:
    max_tokens: int = 4000
    # Maximum length of the longest image side after conversion to JPEG.
    max_img_size: int = 2048
    # Number of documents to extract concurrently (vLLM batches them server-side).
    concurrency: int = 4
    # Per-field confidence/score from docext's second pass:
    #   "none"    - skip it (one VLM call per document)
    #   "binary"  - High/Low
    #   "numeric" - a 0-100 score
    # ("binary"/"numeric" roughly double the number of VLM calls.)
    confidence: str = "none"
    # File extensions treated as input documents (lower-case, with dot).
    input_extensions: tuple[str, ...] = (".tif", ".tiff")
    # Pillow decompression-bomb guard. High-DPI scans are legitimately large, so
    # we raise Pillow's default (~89 MP). Pillow still errors above 2x this value;
    # set to null to disable the check entirely.
    max_pixels: int | None = 300_000_000
    # Rotate every page CLOCKWISE by this many degrees before extraction (e.g. 90,
    # 180, 270 for misoriented scans). Arbitrary angles are allowed; the canvas is
    # expanded and any new corners are filled white. 0 = no rotation.
    rotate: float = 0.0


@dataclass
class Job:
    name: str
    input_dir: Path
    output_csv: Path
    fields: list[Field] = dc_field(default_factory=list)
    # Per-job rotation (degrees clockwise). None = inherit extraction.rotate.
    rotate: float | None = None


@dataclass
class Config:
    model: ModelConfig
    extraction: ExtractionConfig
    jobs: list[Job]


def _build_fields(raw_fields: list, job_name: str) -> list[Field]:
    if not raw_fields:
        raise ConfigError(f"Job '{job_name}' has no fields defined.")
    fields: list[Field] = []
    seen: set[str] = set()
    for entry in raw_fields:
        if isinstance(entry, str):
            name, description = entry, ""
        elif isinstance(entry, dict):
            name = entry.get("name")
            description = entry.get("description", "")
        else:
            raise ConfigError(
                f"Job '{job_name}': each field must be a string or a mapping, "
                f"got {type(entry).__name__}.",
            )
        if not name:
            raise ConfigError(f"Job '{job_name}': a field is missing its 'name'.")
        if name in seen:
            raise ConfigError(f"Job '{job_name}': duplicate field name '{name}'.")
        seen.add(name)
        fields.append(Field(name=name, description=description or ""))
    return fields


def _reject_unknown_keys(raw: dict, dataclass_type, section: str) -> None:
    """Fail with a clear error if ``raw`` contains keys the dataclass doesn't define."""
    known = {f.name for f in dataclasses.fields(dataclass_type)}
    unknown = set(raw) - known
    if unknown:
        raise ConfigError(f"Unknown {section} option(s): {', '.join(sorted(unknown))}")


def _parse_model(raw: dict) -> ModelConfig:
    model_raw = dict(raw)
    # Accept legacy key names so older config files keep working.
    aliases = {
        "vllm_host": "server_host",
        "vllm_port": "server_port",
        "start_vllm": "start_server",
        "vllm_start_timeout": "server_start_timeout",
    }
    for old, new in aliases.items():
        if old in model_raw and new not in model_raw:
            model_raw[new] = model_raw.pop(old)

    _reject_unknown_keys(model_raw, ModelConfig, "model")
    model = ModelConfig(**model_raw)
    if model.dtype not in ("bfloat16", "float16"):
        raise ConfigError("model.dtype must be 'bfloat16' or 'float16'.")
    return model


def _parse_extraction(raw: dict) -> ExtractionConfig:
    extraction_raw = dict(raw)
    if "input_extensions" in extraction_raw:
        extraction_raw["input_extensions"] = tuple(
            e.lower() if e.startswith(".") else f".{e.lower()}"
            for e in extraction_raw["input_extensions"]
        )
    # Legacy alias: include_confidence: true/false -> confidence: binary/none.
    if "include_confidence" in extraction_raw and "confidence" not in extraction_raw:
        extraction_raw["confidence"] = (
            "binary" if extraction_raw.pop("include_confidence") else "none"
        )
    else:
        extraction_raw.pop("include_confidence", None)

    _reject_unknown_keys(extraction_raw, ExtractionConfig, "extraction")
    extraction = ExtractionConfig(**extraction_raw)
    if extraction.confidence not in ("none", "binary", "numeric"):
        raise ConfigError("extraction.confidence must be 'none', 'binary', or 'numeric'.")
    return extraction


def load_config(path: str | os.PathLike) -> Config:
    """Load and validate the batch config from a YAML file."""
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"Config file not found: {path}")

    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ConfigError("Top-level config must be a mapping.")

    model = _parse_model(raw.get("model") or {})
    extraction = _parse_extraction(raw.get("extraction") or {})

    raw_jobs = raw.get("jobs") or []
    if not raw_jobs:
        raise ConfigError("No jobs defined in config. Add at least one entry under 'jobs'.")

    jobs: list[Job] = []
    seen_names: set[str] = set()
    for i, raw_job in enumerate(raw_jobs):
        name = raw_job.get("name") or f"job_{i}"
        if name in seen_names:
            raise ConfigError(f"Duplicate job name '{name}'.")
        seen_names.add(name)

        input_dir = raw_job.get("input_dir")
        if not input_dir:
            raise ConfigError(f"Job '{name}' is missing 'input_dir'.")

        output_csv = raw_job.get("output_csv") or f"{name}.csv"
        jobs.append(
            Job(
                name=name,
                input_dir=Path(input_dir),
                output_csv=Path(output_csv),
                fields=_build_fields(raw_job.get("fields"), name),
                rotate=raw_job.get("rotate"),  # None -> inherit extraction.rotate
            ),
        )

    return Config(model=model, extraction=extraction, jobs=jobs)
