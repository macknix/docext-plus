# ner-ocr-docext

Batch field extraction from scanned documents, built on
[NanoNets/docext](https://github.com/NanoNets/docext). A vision-language model
(Qwen2.5-VL by default) reads each page directly, with no separate OCR step.

You give it directories of **single-page TIFFs** and, for each directory, the
fields to extract. It writes **one long-format CSV per directory**, one row per
(file, field):

```
filename,entity_type,value,confidence,error
doc001.tiff,invoice_number,INV-2024-001,98,
doc001.tiff,invoice_date,2024-03-14,95,
```

`confidence` appears only when enabled. `error` is blank unless that document
failed, in which case `value` is empty.

The same code runs in two places. The backend is chosen by the model-name prefix:

| | Backend | Image | Model prefix |
|---|---|---|---|
| Local testing (Mac) | Ollama (CPU/Metal) | `Dockerfile.dev` (multi-arch) | `ollama/` |
| Production (TRE) | vLLM (CUDA GPU) | `Dockerfile` (amd64 only) | `hosted_vllm/` |

Any other prefix (e.g. `gpt-4o`) is treated as an external endpoint.

## How it works

```
config.yaml ─▶ start model server once ─▶ for each job, for each TIFF:
               TIFF ─▶ rotate/downscale ─▶ JPEG ─▶ docext prompt ─▶ JSON ─▶ rows ─▶ <job>.csv
```

- **One model load per run.** The server starts before the job loop and every
  document is an HTTP request to it. Put all directories in one config and run
  one container.
- **TIFF → JPEG.** docext sends images as `image/jpeg` data URLs, so pages are
  converted first (first frame, RGB, optional rotation, longest side ≤ `max_img_size`).
- **Failures are per document.** An exception becomes an `error` row; the run continues.

## Local testing (Mac, Ollama)

vLLM has no macOS build, so local runs use Ollama on the host (fastest, uses Metal):

```bash
brew install ollama && ollama serve &
ollama pull qwen2.5vl

docker build -f Dockerfile.dev -t ner-ocr-docext:dev .
docker run --rm \
  -v "$PWD/config.dev.yaml":/app/config.yaml \
  -v "$PWD/data":/data \
  ner-ocr-docext:dev --config /app/config.yaml --server-host host.docker.internal
```

Put TIFFs under `data/input/sample/`; CSVs appear in `data/output/`.

To run everything in Docker instead (CPU only, slower):

```bash
docker compose -f docker-compose.dev.yml up -d ollama
docker compose -f docker-compose.dev.yml exec ollama ollama pull qwen2.5vl
docker compose -f docker-compose.dev.yml run --rm app
```

## Production (TRE, vLLM on GPU)

The TRE is offline and x86_64. Build the image and download models on a
networked machine, then move both in as tarballs.

### 1. Check the GPU driver

Run `nvidia-smi` in the TRE. The image uses CUDA 12.4
(`nvidia/cuda:12.4.1-runtime-ubuntu22.04`), which needs a driver reporting
CUDA ≥ 12.4. For a different driver, rebuild with `--build-arg CUDA_IMAGE=...`
using a CUDA version no newer than the driver's.

### 2. Download models and build the image (networked machine)

```bash
pip install huggingface_hub
./scripts/download_model.sh Qwen/Qwen2.5-VL-7B-Instruct-AWQ   # repeat for more models
./scripts/export_model.sh                                     # -> models.tar.gz
./scripts/export_image.sh                                     # -> ner-ocr-docext-amd64.tar.gz
```

`export_image.sh` builds for `linux/amd64` (emulated on Apple Silicon) and saves
a gzipped tarball. To bake one model into the image so nothing needs mounting:

```bash
BAKE_MODEL=true HUGGING_FACE_HUB_TOKEN=<token> \
  MODEL_REPO=Qwen/Qwen2.5-VL-7B-Instruct-AWQ ./scripts/export_image.sh
```

### 3. Load in the TRE

```bash
tar -xzf models.tar.gz                        # -> ./models/hf_cache/hub/models--Qwen--...
podman load -i ner-ocr-docext-amd64.tar.gz    # or docker load -i ...
podman images                                 # expect localhost/ner-ocr-docext:amd64
```

If `podman images` is empty after `found incomplete layer ... deleting it`, the
load ran out of disk. Allow about 25 GB free during the load (tarball plus the
~13 GB unpacked image).

### 4. Run

```bash
docker run --rm --gpus all --shm-size=16g --security-opt label=disable \
  -v "$PWD/models/hf_cache":/models/hf_cache \
  -v "$PWD/config.yaml":/app/config.yaml \
  -v "$PWD/data":/data \
  ner-ocr-docext:amd64 --config /app/config.yaml
```

| Flag | Why |
|---|---|
| `--gpus all` | vLLM needs a CUDA GPU. Use `--gpus '"device=0"'` to pin one. |
| `--shm-size=16g` | vLLM needs shared memory; raise it on `bus error`. |
| `--security-opt label=disable` | Lets SELinux hosts (RHEL/Rocky) read bind mounts. Alternatively append `:z` to each mount. Drop it if the host has no SELinux. |
| `-v .../hf_cache` | The model cache. Omit if the model is baked in. |

The container runs with `HF_HUB_OFFLINE=1` and telemetry disabled, so it makes
no outbound calls. Switch between cached models with `--model`.

**GPU memory.** The default 7B AWQ model needs about 16 GB. If short, lower
`model.max_model_len`, `extraction.max_img_size` or
`model.gpu_memory_utilization`, or pick a smaller model.

**Multiple GPUs.** Run one container per GPU (`--gpus '"device=N"'`), each with
its own config covering a subset of the jobs. Never run two containers on one
GPU: each loads its own copy of the weights. Tensor parallelism isn't supported
because docext 0.1.11's launcher doesn't expose it.

### Image notes

The `Dockerfile` is a two-stage build on the CUDA runtime base: a builder stage
pip-installs `docext==0.1.11` (vLLM 0.8.3, PyTorch 2.6) into a venv, which is
copied into a clean runtime stage. Two choices matter if you change versions:

- **`transformers>=4.51.1,<4.53.0` is pinned.** docext 0.1.11 doesn't pin it,
  and transformers 5.x breaks vLLM 0.8.3 (`Could not import module 'ProcessorMixin'`).
- **No flash-attn.** vLLM falls back to xformers/SDPA and runs with
  `--enforce-eager`, so it isn't needed and saves about 2 GB.

Check a build with:

```bash
docker run --rm --entrypoint python ner-ocr-docext:amd64 \
  -c "import torch, transformers; print(torch.version.cuda, transformers.__version__)"
# expect: 12.4 4.52.x
```

## Configuration

Copy `config.example.yaml` to `config.yaml`. `config.dev.yaml` is a ready-made
Ollama config.

| Section | Controls |
|---|---|
| `model` | Model name (selects backend), server host/port, vLLM tuning |
| `extraction` | Image size, concurrency, confidence, input extensions, default rotation |
| `jobs[]` | Per directory: `input_dir`, `output_csv`, `fields`, optional `rotate` |

```yaml
jobs:
  - name: invoices
    input_dir: /data/input/invoices
    output_csv: /data/output/invoices.csv
    rotate: 90                  # optional, degrees clockwise; overrides extraction.rotate
    fields:
      - name: invoice_number
        description: "The unique invoice number"   # descriptions guide the model
      - supplier_name                              # bare string also works
```

CLI flags override the config: `--model`, `--server-host`, `--server-port`,
`--work-dir`, `--verbose` (full third-party debug logs) and `--log-level`.

**Confidence** (`extraction.confidence`): `none` (default, 1 VLM call per
document), `binary` (`High`/`Low`) or `numeric` (0–100). The last two roughly
double the number of calls.

**Rotation.** Right angles are lossless; other angles expand the canvas with
white corners. Preview an angle with `notebooks/rotation_test.ipynb`.

Older key names (`vllm_host`, `vllm_port`, `start_vllm`, `vllm_start_timeout`,
`include_confidence`) are still accepted.

### Performance

The GPU is the bottleneck. Watch throughput and `nvidia-smi` while tuning.

| Setting | Effect |
|---|---|
| `extraction.concurrency` | Documents in flight; vLLM batches them. Main throughput lever; try 8–32. |
| `extraction.max_img_size` | Image tokens scale with resolution². Lower is much faster but small text may become unreadable. |
| `extraction.confidence` | `none` is about 2× faster than `binary`/`numeric`. |
| `model.gpu_memory_utilization`, `model.max_model_len` | Trade KV-cache room (concurrency) against OOM risk. |

## Post-processing

`notebooks/postprocess.ipynb` turns the long CSVs into one wide CSV per dataset
with cleaned address, date and serial-number columns, and flags mismatches
between the serial number in the file name and the extracted one. Edit its
`DATASETS` list and mapping dictionaries for your data.

## Development

```
docext_batch/
├── config.py    # YAML loading and validation
├── images.py    # document discovery, TIFF → JPEG
├── server.py    # model endpoint lifecycle (vLLM / Ollama / external)
├── extract.py   # per-document extraction, per-job CSV
└── run.py       # CLI entrypoint
scripts/         # model download, model/image export for the TRE
notebooks/       # rotation preview, post-processing
tests/           # pytest; docext is stubbed, no GPU needed
```

```bash
pip install pytest pyyaml pillow pandas
pytest
```

## Limitations

- Only the first frame of a multi-page TIFF is used.
- Quality depends on the model and field descriptions. Be specific, and give a
  format where it matters (e.g. "date as YYYY-MM-DD").
- The Ollama path is for development. Use vLLM on a GPU for production results.
- Never commit documents, outputs or model weights: `data/`, `models/`, CSVs and
  TIFFs are gitignored.
