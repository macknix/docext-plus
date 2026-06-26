"""Batch document field extraction built on NanoNets/docext.

Runs a vision-language model (served locally with vLLM) over directories of
single-page TIFF documents and writes one CSV per directory: rows are file
names, columns are the extracted entities/fields.
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"
