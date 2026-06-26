"""Tests for config loading, validation, and backend inference."""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from docext_batch.config import ConfigError
from docext_batch.config import ModelConfig
from docext_batch.config import load_config


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(textwrap.dedent(body))
    return path


def test_backend_inferred_from_name():
    assert ModelConfig(name="hosted_vllm/Qwen/X").backend == "vllm"
    assert ModelConfig(name="ollama/qwen2.5vl").backend == "ollama"
    assert ModelConfig(name="gpt-4o").backend == "external"


def test_default_ports_and_url():
    vllm = ModelConfig(name="hosted_vllm/X", server_host="h")
    assert vllm.port == 8000
    assert vllm.url == "http://h:8000/v1"  # vLLM gets the /v1 suffix

    ollama = ModelConfig(name="ollama/m", server_host="h")
    assert ollama.port == 11434
    assert ollama.url == "http://h:11434"  # Ollama uses the root

    assert ModelConfig(name="ollama/m", server_port=9999).port == 9999


def test_load_minimal(tmp_path):
    cfg_path = _write(
        tmp_path,
        """
        jobs:
          - name: j
            input_dir: /data/in
            fields: [alpha]
        """,
    )
    cfg = load_config(cfg_path)
    assert len(cfg.jobs) == 1
    assert cfg.jobs[0].fields[0].name == "alpha"
    assert cfg.jobs[0].output_csv == Path("j.csv")  # defaulted from job name


def test_fields_string_and_mapping(tmp_path):
    cfg = load_config(
        _write(
            tmp_path,
            """
            jobs:
              - name: j
                input_dir: /data/in
                fields:
                  - bare_field
                  - {name: rich_field, description: a description}
            """,
        ),
    )
    fields = cfg.jobs[0].fields
    assert (fields[0].name, fields[0].description) == ("bare_field", "")
    assert (fields[1].name, fields[1].description) == ("rich_field", "a description")


def test_legacy_model_aliases(tmp_path):
    cfg = load_config(
        _write(
            tmp_path,
            """
            model:
              name: ollama/m
              vllm_host: legacyhost
              vllm_port: 1234
              start_vllm: false
            jobs:
              - {name: j, input_dir: /data/in, fields: [a]}
            """,
        ),
    )
    assert cfg.model.server_host == "legacyhost"
    assert cfg.model.server_port == 1234
    assert cfg.model.start_server is False


@pytest.mark.parametrize(
    "value,expected",
    [("true", "binary"), ("false", "none")],
)
def test_legacy_include_confidence(tmp_path, value, expected):
    cfg = load_config(
        _write(
            tmp_path,
            f"""
            extraction: {{include_confidence: {value}}}
            jobs:
              - {{name: j, input_dir: /data/in, fields: [a]}}
            """,
        ),
    )
    assert cfg.extraction.confidence == expected


def test_input_extensions_normalised(tmp_path):
    cfg = load_config(
        _write(
            tmp_path,
            """
            extraction: {input_extensions: [TIFF, .tif]}
            jobs:
              - {name: j, input_dir: /data/in, fields: [a]}
            """,
        ),
    )
    assert cfg.extraction.input_extensions == (".tiff", ".tif")


@pytest.mark.parametrize(
    "body,message",
    [
        ("jobs: []", "No jobs"),
        ("model: {nonsense: 1}\njobs: [{name: j, input_dir: x, fields: [a]}]", "Unknown model"),
        ("extraction: {nonsense: 1}\njobs: [{name: j, input_dir: x, fields: [a]}]", "Unknown extraction"),
        ("extraction: {confidence: bogus}\njobs: [{name: j, input_dir: x, fields: [a]}]", "confidence"),
        ("model: {dtype: float64}\njobs: [{name: j, input_dir: x, fields: [a]}]", "dtype"),
        ("jobs: [{name: j, input_dir: x}]", "no fields"),
        ("jobs: [{name: j, fields: [a]}]", "input_dir"),
        ("jobs: [{name: j, input_dir: x, fields: [a, a]}]", "duplicate field"),
    ],
)
def test_invalid_configs_rejected(tmp_path, body, message):
    with pytest.raises(ConfigError) as exc:
        load_config(_write(tmp_path, body))
    assert message.lower() in str(exc.value).lower()
