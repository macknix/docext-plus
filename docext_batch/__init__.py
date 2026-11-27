"""Batch document field extraction built on NanoNets/docext.

Runs a vision-language model (vLLM or Ollama) over directories of single-page
TIFF documents and writes one long-format CSV per directory, with one row per
(file, entity).
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
