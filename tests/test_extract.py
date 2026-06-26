"""Tests for extraction key-normalisation and long-format CSV assembly.

The VLM call (``docext_batch.extract._call``) is monkeypatched so these tests run
without a model server.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from docext_batch import extract
from docext_batch.config import ExtractionConfig
from docext_batch.config import Field
from docext_batch.config import Job
from docext_batch.config import ModelConfig


def _config(confidence="none", concurrency=1):
    cfg = type("C", (), {})()
    cfg.model = ModelConfig(name="ollama/m")
    cfg.extraction = ExtractionConfig(confidence=confidence, concurrency=concurrency)
    return cfg


def test_normalise():
    assert extract._normalise("Place of Residence") == "place_of_residence"


def test_extract_fields_maps_normalised_keys(monkeypatch):
    fields = [Field("Place of Residence", "where"), Field("NHS Number")]

    # The extraction reply is keyed by normalised names (as docext prompts ask).
    def fake_call(messages, model_name, max_tokens, fmt):
        return json.dumps({"place_of_residence": "12 High St", "nhs_number": "123"})

    monkeypatch.setattr(extract, "_call", fake_call)
    result = extract.extract_fields("img.jpg", "ollama/m", fields, max_tokens=100)

    assert result["Place of Residence"]["value"] == "12 High St"
    assert result["NHS Number"]["value"] == "123"
    assert result["Place of Residence"]["confidence"] is None  # confidence="none"


def test_extract_fields_numeric_confidence(monkeypatch):
    fields = [Field("Alpha")]
    calls = []

    def fake_call(messages, model_name, max_tokens, fmt):
        calls.append(fmt)
        # First call = extraction, second = confidence pass.
        return json.dumps({"alpha": "v"} if len(calls) == 1 else {"alpha": "95"})

    monkeypatch.setattr(extract, "_call", fake_call)
    result = extract.extract_fields("img.jpg", "ollama/m", fields, max_tokens=100, confidence="numeric")

    assert result["Alpha"]["value"] == "v"
    assert result["Alpha"]["confidence"] == "95"
    assert len(calls) == 2  # extraction + one confidence pass


def test_run_job_long_format(tmp_path, monkeypatch):
    from PIL import Image

    indir = tmp_path / "in"
    indir.mkdir()
    for name in ["b.tiff", "a.tiff"]:
        Image.new("RGB", (8, 8)).save(indir / name)

    monkeypatch.setattr(extract, "tiff_to_jpeg", lambda src, dst, **kw: src)
    monkeypatch.setattr(
        extract,
        "extract_fields",
        lambda image_path, model_name, fields, max_tokens, confidence="none": {
            f.name: {"value": f"{f.name}_v", "confidence": "90"} for f in fields
        },
    )

    out = tmp_path / "out.csv"
    job = Job("j", indir, out, [Field("Alpha"), Field("Beta Two")])
    extract.run_job(job, _config(confidence="numeric", concurrency=2), tmp_path / "work")

    df = pd.read_csv(out)
    assert list(df.columns) == ["filename", "entity_type", "value", "confidence", "error"]
    assert len(df) == 4  # 2 files x 2 fields
    assert df["filename"].tolist() == ["a.tiff", "a.tiff", "b.tiff", "b.tiff"]  # stable sort
    assert df["entity_type"].tolist()[:2] == ["Alpha", "Beta Two"]  # field order kept


def test_run_job_omits_confidence_column_when_none(tmp_path, monkeypatch):
    from PIL import Image

    indir = tmp_path / "in"
    indir.mkdir()
    Image.new("RGB", (8, 8)).save(indir / "doc.tiff")

    monkeypatch.setattr(extract, "tiff_to_jpeg", lambda src, dst, **kw: src)
    monkeypatch.setattr(
        extract,
        "extract_fields",
        lambda *a, **k: {"Alpha": {"value": "v", "confidence": None}},
    )

    out = tmp_path / "out.csv"
    extract.run_job(Job("j", indir, out, [Field("Alpha")]), _config(), tmp_path / "work")

    df = pd.read_csv(out)
    assert "confidence" not in df.columns


@pytest.mark.parametrize(
    "job_rotate,default_rotate,expected",
    [
        (90, 0, 90),      # per-job overrides the default
        (None, 180, 180), # no per-job value -> inherits extraction.rotate
        (0, 270, 0),      # explicit 0 on the job overrides a non-zero default
    ],
)
def test_run_job_uses_per_job_rotation(tmp_path, monkeypatch, job_rotate, default_rotate, expected):
    from PIL import Image

    indir = tmp_path / "in"
    indir.mkdir()
    Image.new("RGB", (8, 8)).save(indir / "doc.tiff")

    captured = {}

    def fake_tiff(src, dst, max_size=2048, rotate=0):
        captured["rotate"] = rotate
        return src

    monkeypatch.setattr(extract, "tiff_to_jpeg", fake_tiff)
    monkeypatch.setattr(
        extract, "extract_fields",
        lambda *a, **k: {"Alpha": {"value": "v", "confidence": None}},
    )

    cfg = _config()
    cfg.extraction.rotate = default_rotate
    job = Job("j", indir, tmp_path / "out.csv", [Field("Alpha")], rotate=job_rotate)
    extract.run_job(job, cfg, tmp_path / "work")

    assert captured["rotate"] == expected


def test_run_job_records_errors(tmp_path, monkeypatch):
    from PIL import Image

    indir = tmp_path / "in"
    indir.mkdir()
    Image.new("RGB", (8, 8)).save(indir / "doc.tiff")

    monkeypatch.setattr(extract, "tiff_to_jpeg", lambda src, dst, **kw: src)

    def boom(*a, **k):
        raise RuntimeError("model exploded")

    monkeypatch.setattr(extract, "extract_fields", boom)

    out = tmp_path / "out.csv"
    extract.run_job(Job("j", indir, out, [Field("Alpha")]), _config(), tmp_path / "work")

    df = pd.read_csv(out)
    assert df.loc[0, "error"] == "model exploded"
    assert pd.isna(df.loc[0, "value"]) or df.loc[0, "value"] == ""
