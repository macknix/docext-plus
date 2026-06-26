"""Test fixtures.

The extraction module imports docext and json_repair, which are heavy/GPU-only
and not installed in a plain test environment. We stub just enough of them here
so the package imports and the pure-Python logic (config, image prep, CSV
assembly, key normalisation) can be tested anywhere.
"""
from __future__ import annotations

import json
import sys
import types


def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# json_repair: fall back to the stdlib JSON parser (fine for valid JSON in tests).
try:  # pragma: no cover - depends on environment
    import json_repair  # noqa: F401
except ImportError:
    _stub_module("json_repair").loads = json.loads

# docext.core.*: provide importable placeholders. Tests that exercise extraction
# monkeypatch docext_batch.extract._call, so these are only used as stand-ins.
for _name in (
    "docext",
    "docext.core",
    "docext.core.client",
    "docext.core.confidence",
    "docext.core.prompts",
):
    if _name not in sys.modules:
        try:  # pragma: no cover - prefer the real module if present
            __import__(_name)
        except ImportError:
            _stub_module(_name)


def _default(mod_name: str, attr: str, fn) -> None:
    mod = sys.modules[mod_name]
    if not hasattr(mod, attr):
        setattr(mod, attr, fn)


_default(
    "docext.core.client",
    "sync_request",
    lambda *a, **k: {"choices": [{"message": {"content": "{}"}}]},
)
_default("docext.core.prompts", "get_fields_messages", lambda names, desc, paths: [])
_default("docext.core.confidence", "get_fields_confidence_score_messages_binary", lambda m, a, n: m)
_default("docext.core.confidence", "get_fields_confidence_score_messages_numeric", lambda m, a, n: m)
