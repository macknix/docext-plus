"""CLI entrypoint: start the model, run every job, write CSVs.

    python -m docext_batch.run --config config.yaml
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

from loguru import logger

from docext_batch.config import ConfigError
from docext_batch.config import load_config
from docext_batch.extract import run_job
from docext_batch.images import set_max_image_pixels
from docext_batch.server import ModelServer

# Chatty third-party loggers we quiet unless --verbose is given. These fire on
# every request and are the main source of running-log noise.
_NOISY_LOGGERS = ("LiteLLM", "litellm", "httpx", "httpcore", "urllib3", "transformers", "vllm")


def configure_logging(verbose: bool, log_level: str | None = None) -> None:
    """Set up logging.

    Default (not verbose): our own INFO logs only — config + model startup
    (initialisation) and per-file progress. Third-party libraries and the vLLM
    subprocess are quieted to WARNING so the "running" phase stays clean.

    Verbose: everything at DEBUG, including the vLLM / litellm / transformers
    internals. ``log_level`` (if given) overrides and is treated as advanced.
    """
    if log_level:
        level = log_level.upper()
        verbose = level == "DEBUG"
    else:
        level = "DEBUG" if verbose else "INFO"

    logger.remove()
    fmt = (
        "<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | "
        + ("<cyan>{name}:{function}:{line}</cyan> - " if verbose else "")
        + "<level>{message}</level>"
    )
    logger.add(sys.stderr, level=level, format=fmt)

    # Quiet (or unleash) the noisy libraries and the vLLM subprocess.
    third_party_level = logging.INFO if verbose else logging.WARNING
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(third_party_level)
    os.environ["VLLM_LOGGING_LEVEL"] = "INFO" if verbose else "WARNING"
    os.environ["TRANSFORMERS_VERBOSITY"] = "info" if verbose else "error"
    try:  # litellm has its own debug switch
        import litellm

        litellm.suppress_debug_info = not verbose
    except Exception:  # noqa: BLE001 - litellm is optional at import time here
        pass


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="docext_batch",
        description="Batch field extraction from TIFF documents using docext + vLLM.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        help="Path to the YAML config file (default: config.yaml).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override model.name from the config (selects the backend by prefix, "
        "e.g. 'ollama/qwen2.5vl' or 'hosted_vllm/Qwen/Qwen2.5-VL-7B-Instruct-AWQ').",
    )
    parser.add_argument(
        "--server-host",
        default=None,
        help="Override model.server_host (e.g. 'host.docker.internal' on Mac+Docker).",
    )
    parser.add_argument(
        "--server-port",
        type=int,
        default=None,
        help="Override model.server_port.",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Directory for intermediate JPEGs (default: a temp directory).",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Show full debug logging (vLLM/litellm/transformers internals). "
        "Default shows only initialisation logs and per-file progress.",
    )
    parser.add_argument(
        "--log-level",
        default=None,
        help="Advanced: set the log level explicitly (DEBUG/INFO/WARNING/ERROR); "
        "overrides --verbose.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    configure_logging(verbose=args.verbose, log_level=args.log_level)

    try:
        cfg = load_config(args.config)
    except ConfigError as exc:
        logger.error(f"Invalid config: {exc}")
        return 2

    # CLI overrides win over the config file (handy for switching models per run).
    if args.model:
        cfg.model.name = args.model
    if args.server_host:
        cfg.model.server_host = args.server_host
    if args.server_port is not None:
        cfg.model.server_port = args.server_port

    # Apply Pillow's decompression-bomb limit before any image is opened.
    set_max_image_pixels(cfg.extraction.max_pixels)

    logger.info(
        f"Loaded {len(cfg.jobs)} job(s); model='{cfg.model.name}', "
        f"backend='{cfg.model.backend}', start_server={cfg.model.start_server}",
    )

    if args.work_dir:
        work_root = Path(args.work_dir)
        work_root.mkdir(parents=True, exist_ok=True)
        tmp_ctx = None
    else:
        tmp_ctx = tempfile.TemporaryDirectory(prefix="docext_batch_")
        work_root = Path(tmp_ctx.name)

    failures = 0
    try:
        with ModelServer(cfg.model):
            for job in cfg.jobs:
                try:
                    run_job(job, cfg, work_root)
                except Exception as exc:  # noqa: BLE001 - one job failing shouldn't sink the rest
                    failures += 1
                    logger.exception(f"Job '{job.name}' failed: {exc}")
    finally:
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    if failures:
        logger.error(f"{failures} job(s) failed.")
        return 1
    logger.info("All jobs completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
