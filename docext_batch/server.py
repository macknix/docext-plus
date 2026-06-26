"""Model-server lifecycle, one class for every backend.

The backend is selected by the model name prefix (see ModelConfig.backend):

* ``vllm``     - start a local vLLM server (GPU / TRE) via docext's own
                 ``VLLMServer``, or reuse one that is already running.
* ``ollama``   - connect to a running Ollama server (ideal for Mac/CPU dev).
                 Ollama runs as its own service; we never launch it.
* ``external`` - a vendor-hosted or already-running endpoint; we only point
                 docext at it.

In all cases docext's client discovers the endpoint through the
``VLM_MODEL_URL`` environment variable, which we set here.
"""
from __future__ import annotations

import inspect
import os

import requests
from loguru import logger

from docext_batch.config import ModelConfig


def _http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        return requests.get(url, timeout=timeout).status_code == 200
    except requests.RequestException:
        return False


def _vllm_healthy(host: str, port: int) -> bool:
    return _http_ok(f"http://{host}:{port}/v1/models")


def _ollama_healthy(host: str, port: int) -> bool:
    return _http_ok(f"http://{host}:{port}/api/tags")


class ModelServer:
    """Prepare the model endpoint for a run and tear it down afterwards."""

    def __init__(self, model: ModelConfig):
        self.model = model
        self._server = None  # docext VLLMServer, only for the vllm backend
        self._started = False

    def __enter__(self) -> "ModelServer":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        # Idempotent: the model is loaded into memory exactly once per run. The
        # server is started here, before the job loop, and reused for every job
        # and every document (each extraction is just an HTTP request to it).
        if self._started:
            logger.debug("ModelServer.start() already called; reusing loaded model.")
            return

        model = self.model
        os.environ["VLM_MODEL_URL"] = model.url
        logger.info(f"Backend '{model.backend}', VLM_MODEL_URL={model.url}")

        if model.backend == "vllm":
            self._start_vllm()
        elif model.backend == "ollama":
            self._start_ollama()
        else:
            logger.info(f"Using external model endpoint for '{model.name}'.")

        self._started = True
        logger.info("Model loaded once; it will be reused for all jobs/documents.")

    def _start_vllm(self) -> None:
        model = self.model
        if _vllm_healthy(model.server_host, model.port):
            logger.info(
                f"Reusing vLLM server already running at "
                f"{model.server_host}:{model.port}.",
            )
            return
        if not model.start_server:
            raise RuntimeError(
                f"No vLLM server reachable at {model.server_host}:{model.port} "
                "and model.start_server is false.",
            )

        # Imported lazily: docext.core.vllm pulls in the heavy serving stack and
        # is only available in the GPU image.
        from docext.core.vllm import VLLMServer

        # VLLMServer's signature varies across docext versions (e.g. `dtype` was
        # added after 0.1.11). Pass only the arguments the installed version
        # accepts; anything it doesn't know about falls back to its own default.
        candidate_kwargs = {
            "model_name": model.name,
            "host": model.server_host,
            "port": model.port,
            "max_model_len": model.max_model_len,
            "gpu_memory_utilization": model.gpu_memory_utilization,
            "max_num_imgs": model.max_num_imgs,
            "vllm_start_timeout": model.server_start_timeout,
            "dtype": model.dtype,
        }
        supported = set(inspect.signature(VLLMServer.__init__).parameters)
        kwargs = {k: v for k, v in candidate_kwargs.items() if k in supported}
        dropped = sorted(set(candidate_kwargs) - set(kwargs))
        if dropped:
            logger.debug(
                f"This docext's VLLMServer doesn't accept {dropped}; using its defaults.",
            )

        logger.info(f"Starting vLLM server for '{model.name}'...")
        self._server = VLLMServer(**kwargs)
        self._server.run_in_background()  # blocks until healthy
        logger.info("vLLM server is ready.")

    def _start_ollama(self) -> None:
        model = self.model
        if not _ollama_healthy(model.server_host, model.port):
            raise RuntimeError(
                f"No Ollama server at {model.server_host}:{model.port}. "
                "Install Ollama, start it, and `ollama pull` the model first "
                "(e.g. `ollama pull qwen2.5vl`). On Mac+Docker, set "
                "model.server_host to 'host.docker.internal'.",
            )
        logger.info(f"Ollama server reachable at {model.server_host}:{model.port}.")

        # Warm-load the model once and pin it in memory for the run, so it is not
        # unloaded/reloaded between documents (Ollama unloads after keep_alive).
        # Best-effort: if this fails the model still loads lazily on the first
        # real request, so we warn rather than abort.
        ollama_name = model.name.split("/", 1)[1]  # strip the "ollama/" prefix
        # Ollama wants keep_alive as an integer (seconds, or -1 = forever) or a
        # duration string like "5m" - NOT a numeric string like "-1".
        keep_alive = model.ollama_keep_alive
        try:
            keep_alive = int(keep_alive)
        except (TypeError, ValueError):
            pass  # leave as a duration string, e.g. "5m"

        logger.info(f"Warm-loading '{ollama_name}' into memory (keep_alive={keep_alive})...")
        try:
            resp = requests.post(
                f"http://{model.server_host}:{model.port}/api/generate",
                json={"model": ollama_name, "keep_alive": keep_alive},
                timeout=model.server_start_timeout,
            )
            resp.raise_for_status()
            logger.info(f"'{ollama_name}' loaded into memory.")
        except requests.RequestException as exc:
            logger.warning(
                f"Warm-load of '{ollama_name}' failed ({exc}); it will load on the "
                f"first request instead. If extraction then fails, check the model "
                f"is pulled: `ollama pull {ollama_name}`.",
            )

    def stop(self) -> None:
        if self._server is not None:
            self._server.stop_server()
            self._server = None
