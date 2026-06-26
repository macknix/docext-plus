# ner-ocr-docext

Batch document field extraction built on [NanoNets/docext](https://github.com/NanoNets/docext).

A vision-language model (Qwen2.5-VL by default) reads your documents directly —
there is no separate OCR step. You point it at one or more directories of
**single-page TIFFs**, declare which **entities/fields** to extract for each
directory, and it writes **one CSV per directory** in long format — one row per
(file, entity):

```
filename,entity_type,value,confidence,error
doc001.tiff,invoice_number,INV-2024-001,98,
doc001.tiff,invoice_date,2024-03-14,95,
doc002.tiff,invoice_number,INV-2024-002,90,
doc002.tiff,invoice_date,2024-03-15,72,
```

- `filename` — the document
- `entity_type` — the field you asked for
- `value` — the extracted value
- `confidence` — model confidence/score (column present only when enabled; see
  [Configuration](#configuration))
- `error` — blank unless that document failed (then `value` is empty)

It runs two ways from the same code:

| | Backend | Where | Image | Architecture |
|---|---|---|---|---|
| **Local testing** | Ollama (CPU/Metal) | your Mac | `Dockerfile.dev` | native (ARM64 *and* amd64) — **runs on the Mac** |
| **Production** | vLLM (GPU) | the TRE | `Dockerfile` | amd64/x86_64 only — **built on the Mac, runs in the TRE** |

The backend is chosen by the model-name prefix, so the same config/code works in
both places — you just point at a different model.

> **Architecture note (important).** Your Mac is ARM64 (Apple Silicon); the TRE is
> x86_64. The dev image is multi-arch and runs natively on the Mac. The GPU image
> is built on a CUDA base image, which is **amd64-only**, so on the Mac
> you build it with `--platform linux/amd64` (emulated) purely to **export** it —
> it cannot run on the Mac (no NVIDIA GPU). See
> [Production on GPU / in the TRE](#production-on-gpu--in-the-tre-vllm).

---

## How it works

```
config.yaml ─▶ docext_batch.run ─┬─▶ prepare model endpoint (vLLM or Ollama)
                                 │
              for each job:      ▼
   TIFF ─▶ JPEG ─▶ VLM prompt (docext) ─▶ JSON ─▶ row ─▶ <job>.csv
```

1. **One model, many jobs.** Each *job* in the config is a directory plus its own
   field list, so different document sets get different extraction schemas in one
   run.
2. **TIFF → JPEG.** docext encodes images as `image/jpeg` data URLs, so TIFFs are
   converted (first frame, flattened to RGB, downscaled to `max_img_size`) before
   being sent — required, as raw TIFF bytes aren't decodable by the model.
3. **Per-document request.** Each file is one self-contained VLM request, parsed
   into rows. Failures are caught per file and recorded in the `error` column.

It reuses docext's own prompt builders, client, and vLLM launcher, so model
behaviour matches upstream docext.

### The model is loaded into memory once

The model server is started **once**, before the job loop, and reused for every
job and every document — each extraction is just an HTTP request to it. The
weights are never reloaded mid-run.

- **vLLM (TRE).** A single `vllm serve` process loads the weights into GPU memory
  once at startup and stays resident for the whole run. This is the important
  case for the TRE — and the reason to put **all your directories as `jobs` in one
  config and do a single `docker run`**. Running one container per directory would
  reload the model each time; one container processes every directory off a single
  load.
- **Ollama (dev).** The model is warm-loaded once at startup and pinned with
  `keep_alive` (`-1` by default) so it isn't unloaded between documents. For an
  absolute guarantee across long idle gaps, also start Ollama with
  `OLLAMA_KEEP_ALIVE=-1`.

Startup logs `Model loaded once; it will be reused for all jobs/documents.` so you
can confirm it.

---

## Selecting / switching models

The model name's **prefix selects the backend**:

| Prefix | Backend | Example |
|---|---|---|
| `hosted_vllm/` | local vLLM (GPU) | `hosted_vllm/Qwen/Qwen2.5-VL-7B-Instruct-AWQ` |
| `ollama/` | Ollama (Mac/CPU) | `ollama/qwen2.5vl` |
| *(other)* | vendor/external endpoint | `gpt-4o` |

Set it in `config.yaml` under `model.name`, or override per run **without editing
the config**:

```bash
... --model hosted_vllm/Qwen/Qwen2.5-VL-32B-Instruct-AWQ
... --model ollama/llama3.2-vision
```

`--server-host` and `--server-port` can likewise be overridden on the CLI.

---

## Local testing on your Mac (Ollama)

vLLM is CUDA-only with no macOS wheels, so locally we use **Ollama**. The Mac
image (`Dockerfile.dev`) installs docext *without* its heavy GPU deps.

### Recommended: Ollama on the host (uses Metal, fastest)

```bash
# 1. Install + start Ollama, then pull a vision model (stored in ~/.ollama)
brew install ollama        # or download the app from ollama.com
ollama serve &             # the desktop app starts this automatically
ollama pull qwen2.5vl      # or: ollama pull llama3.2-vision

# 2. Put some TIFFs under data/input/sample/ and build the dev image
docker build -f Dockerfile.dev -t ner-ocr-docext:dev .

# 3. Run, pointing at Ollama on the host
docker run --rm \
  -v "$PWD/config.dev.yaml":/app/config.yaml \
  -v "$PWD/data":/data \
  ner-ocr-docext:dev \
  --config /app/config.yaml \
  --model ollama/qwen2.5vl \
  --server-host host.docker.internal
```

Results appear in `data/output/`. `config.dev.yaml` is a ready-made Ollama config.

### Alternative: everything in Docker (zero host setup, CPU-only)

```bash
docker compose -f docker-compose.dev.yml up -d ollama
docker compose -f docker-compose.dev.yml exec ollama ollama pull qwen2.5vl
docker compose -f docker-compose.dev.yml run --rm app
```

(Ollama-in-Docker on a Mac can't use Metal, so it's slower than the host option.)

---

## Production on GPU / in the TRE (vLLM)

### CUDA / GPU driver compatibility

Check the TRE GPU first:

```bash
nvidia-smi      # note "CUDA Version" (max the driver supports) and "Driver Version"
```

The container's CUDA runtime must be **≤** the driver's CUDA version. The default
`Dockerfile` is built **`FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04`** and
pip-installs `docext` (→ vLLM 0.8.3 → **PyTorch 2.6, CUDA 12.4**) — an exact match
to a driver reporting CUDA 12.4 (e.g. driver 550.x). Override the base with
`--build-arg CUDA_IMAGE=...` if your driver differs.

Verify the built image before shipping it:

```bash
docker run --rm --entrypoint python ner-ocr-docext:amd64 \
  -c "import torch; print('torch CUDA:', torch.version.cuda)"   # expect 12.4
```

> Don't bump the base to a *newer* CUDA (e.g. a 12.6 image) to "match" a 12.4
> driver — that's backwards. The runtime must be ≤ the driver's CUDA.

### Image size

A GPU inference image is inherently large — the bulk is CUDA + PyTorch + vLLM,
which is irreducible (the model weights are **not** in the image; they're mounted).
The `Dockerfile` is built to be as small as practical (~5–6 GB): a CUDA-runtime
base, `pip install docext` with **no flash-attn**, and build tools confined to a
builder stage (see [How the Dockerfile was built](#how-the-dockerfile-was-built)).
Check the real size after building:

```bash
docker images ner-ocr-docext:amd64        # SIZE column = uncompressed on-disk size
```

### Models: download, tar, move, untar, mount

The TRE is network-isolated, so the model is prepared on a **networked machine**
and moved in as a tarball. The image reads models from `/models/hf_cache`
(`HF_HOME`) and runs offline (`HF_HUB_OFFLINE=1`), so once the cache is mounted
you can switch models with `--model` without rebuilding.

**1. On a networked machine — download the model(s) and tar the cache:**

```bash
pip install huggingface_hub
./scripts/download_model.sh Qwen/Qwen2.5-VL-7B-Instruct-AWQ
./scripts/download_model.sh Qwen/Qwen2.5-VL-32B-Instruct-AWQ   # add as many as you like

./scripts/export_model.sh                 # -> models.tar.gz  (the whole models/ tree)
```

<details><summary>…or tar it by hand</summary>

```bash
tar -czf models.tar.gz models/            # run from the project root
```
</details>

**2. Move `models.tar.gz` into the TRE, then extract it there:**

```bash
tar -xzf models.tar.gz                     # recreates ./models/hf_cache/...
ls models/hf_cache/hub                      # sanity check: you should see models--Qwen--...
```

That `./models/hf_cache` directory is what you mount in the run command below.

### Build on your Mac, export, load in the TRE

Because the Mac is ARM64 and the TRE is x86_64, build for `linux/amd64` and ship a
gzipped tarball. **Build on the Mac (or any networked machine), never in the TRE** —
the build needs network (to pull the CUDA base and pip packages); only the *run*
must be offline. The helper script does build + `docker save` + gzip in one step
(and shows a progress bar if `pv` is installed):

```bash
./scripts/export_image.sh                       # -> ner-ocr-docext-amd64.tar.gz

# Fully self-contained variant (model baked in, nothing to mount in the TRE):
BAKE_MODEL=true HUGGING_FACE_HUB_TOKEN=hf_xxx \
  MODEL_REPO=Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
  ./scripts/export_image.sh
```

Equivalent manual commands, if you prefer not to use the script:

```bash
docker build --platform linux/amd64 -t ner-ocr-docext:amd64 .
docker save ner-ocr-docext:amd64 | gzip > ner-ocr-docext-amd64.tar.gz
# (on a native x86_64 build host you can drop --platform)
```

```bash
# Inside the TRE (see "Load the image in the TRE" below):
podman load -i ner-ocr-docext-amd64.tar.gz
```

### Load the image in the TRE

The tarball is gzip-compressed; `podman`/`docker load` auto-detect that, so you do
**not** need to gunzip it first:

```bash
podman load -i ner-ocr-docext-amd64.tar.gz       # docker load -i ... if the TRE uses Docker

# Equivalent explicit pipe form:
gunzip -c ner-ocr-docext-amd64.tar.gz | podman load
```

Then confirm it actually persisted:

```bash
podman images        # expect: localhost/ner-ocr-docext:amd64  (Podman prefixes local images with localhost/)
```

> **If `podman images` is empty / you saw `found incomplete layer ... deleting it`:**
> the load ran out of disk and the partial layers were pruned. The unpacked image
> is ~13–14 GB on disk; during load the gzipped tarball (~9–11 GB) and the unpacked
> image briefly coexist, so allow ~25 GB free on the storage filesystem, settling
> to ~13–14 GB once you delete the tarball. Check `df -h ~` and `podman system df`,
> free space (or raise your quota), then reload.

### Run (in the TRE)

Many TREs run on SELinux-enforcing hosts (RHEL/Rocky), which **block bind-mounted
volumes** from being read inside the container unless you either disable SELinux
labelling for the container or relabel the mounts. The run command below includes
`--security-opt label=disable` for that reason — drop it if your TRE doesn't use
SELinux.

```bash
docker run --rm \
  --gpus all \
  --shm-size=16g \
  --security-opt label=disable \
  -v "$PWD/models/hf_cache":/models/hf_cache \
  -v "$PWD/config.yaml":/app/config.yaml \
  -v "$PWD/data":/data \
  ner-ocr-docext:amd64 --config /app/config.yaml
```

Flag by flag:

| Flag | Why |
|---|---|
| `--gpus all` | vLLM requires a CUDA GPU (CPU is **not** supported). Use `--gpus '"device=0"'` to pin one GPU. |
| `--shm-size=16g` | vLLM needs shared memory; raise it if you see `SHM`/`bus error` crashes. |
| `--security-opt label=disable` | Lets the container read the SELinux-labelled bind mounts on a locked-down TRE host. |
| `-v .../models/hf_cache:/models/hf_cache` | The model cache you untarred. Omit if the model is baked into the image. |
| `-v .../config.yaml:/app/config.yaml` | Your job config. |
| `-v .../data:/data` | Input TIFFs + output CSVs; paths must match `input_dir`/`output_csv` in the config. |

> **SELinux alternative.** If your TRE forbids `--security-opt label=disable`, append
> `:z` (shared) or `:Z` (private) to each mount instead so Docker relabels them,
> e.g. `-v "$PWD/data":/data:z`. Use one approach or the other, not both.

> **Run once for everything.** Put all your directories as `jobs` in the one
> `config.yaml` and do a single `docker run` — the model is then loaded into GPU
> memory once and reused across every directory (see
> [the model is loaded once](#the-model-is-loaded-into-memory-once)).

### Fully self-contained image (no mounts at all)

For a locked-down TRE you can bake one model into the image instead of mounting a
cache — the easy way is the export script's `BAKE_MODEL` path shown above. Doing
it by hand needs **buildx** (for `--secret`) and, on Apple Silicon, `--platform`:

```bash
export HUGGING_FACE_HUB_TOKEN=hf_xxx
DOCKER_BUILDKIT=1 docker buildx build --platform linux/amd64 --load \
  --secret id=hf_token,env=HUGGING_FACE_HUB_TOKEN \
  --build-arg BAKE_MODEL=true \
  --build-arg MODEL_REPO=Qwen/Qwen2.5-VL-7B-Instruct-AWQ \
  -t ner-ocr-docext:offline .
```

With a baked model you move just one file into the TRE (the image tarball, no
`models.tar.gz`) and the run command drops the model mount:

```bash
docker run --rm --gpus all --shm-size=16g --security-opt label=disable \
  -v "$PWD/config.yaml":/app/config.yaml \
  -v "$PWD/data":/data \
  ner-ocr-docext:offline --config /app/config.yaml
```

### Suggested data layout

```
data/
├── input/
│   ├── invoices/          # *.tiff   (job "invoices")
│   └── patient_records/   # *.tiff   (job "patient_records")
└── output/                # CSVs land here
models/
└── hf_cache/              # mounted HF model cache
```

### GPU memory

The default `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` needs ~16 GB. If tight, lower
`model.max_model_len`, `extraction.max_img_size`, and
`model.gpu_memory_utilization`, or pick a smaller model with `--model`.

---

## How the Dockerfile was built

The GPU image (`Dockerfile`) is hand-rolled on an official NVIDIA CUDA base rather
than a prebuilt all-in-one image, to keep it small and to match the TRE's CUDA
version exactly. Here's the reasoning behind each decision — useful if you ever
need to bump versions or debug a build.

### 1. Base image: `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
- **Why CUDA 12.4:** the TRE GPU driver (`nvidia-smi`) reports CUDA 12.4. A
  container's CUDA runtime must be **≤** the driver's version, so 12.4 is an exact
  match — no reliance on best-effort minor-version forward compatibility.
- **Why `-runtime` (not `-devel` or `-base`):** `-runtime` ships the CUDA runtime
  libraries vLLM/PyTorch need, without the multi-GB compiler toolchain in
  `-devel`. (`-base` is smaller still but omits cuDNN; `-runtime` is the safe
  choice. Override with `--build-arg CUDA_IMAGE=...` to experiment.)

### 2. Install the stack with pip, pinned for reproducibility
`pip install docext==0.1.11` pulls the whole serving stack — **vLLM 0.8.3 → PyTorch
2.6 (CUDA 12.4 wheels)**, transformers, litellm, pandas, Pillow, etc. Two pins
matter:
- **`docext==0.1.11`** — fixes the docext (and therefore vLLM) version.
- **`transformers>=4.51.1,<4.53.0`** — *critical.* docext 0.1.11's PyPI metadata
  doesn't pin transformers and vLLM 0.8.3 leaves its upper bound open, so a fresh
  build pulls **transformers 5.x**, which breaks vLLM with a
  `Could not import module 'ProcessorMixin'` error at model load. Pinning to
  docext's own tested range forces a compatible 4.52.x (and the matching
  `tokenizers 0.21.x`).

### 3. No flash-attn
flash-attn is a large (~2 GB), separately-compiled package. It is **not** in
docext's requirements (some upstream images add it on top). We omit it: vLLM falls
back to xformers / torch SDPA, and we already serve with `--enforce-eager`, so
there's no functional loss for this workload.

### 4. Multi-stage build (build tools don't ship)
- **Builder stage:** installs `build-essential` + Python 3.11, creates a venv at
  `/opt/venv`, and pip-installs everything into it.
- **Runtime stage:** the same CUDA base + Python 3.11 + a few runtime libs
  (`poppler-utils`, `libgl1`), then `COPY --from=builder /opt/venv /opt/venv`.
- Net effect: compilers and pip caches stay in the throwaway builder; the shipped
  image carries only the runtime libraries and the venv.

### 5. Offline / TRE hardening
- `HF_HOME=/models/hf_cache` so the model is read from a **mounted cache** (or
  baked in with `--build-arg BAKE_MODEL=true`).
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `HF_HUB_DISABLE_TELEMETRY=1`,
  `GRADIO_ANALYTICS_ENABLED=False` — **no outbound network calls at run time**.
- The build itself needs network (CUDA base + pip), so it's done on a networked
  machine and the finished image is shipped as a tarball; only the *run* is offline.

### Result
~5–6 GB (vs ~13 GB for an all-in-one prebuilt base), GPU-capable on a CUDA-12.4
driver, and fully offline at run time. Verify a build with:

```bash
docker images ner-ocr-docext:amd64                                   # size
docker run --rm --entrypoint python ner-ocr-docext:amd64 \
  -c "import torch, transformers; print(torch.version.cuda, transformers.__version__)"
# expect: 12.4 4.52.x
```

---

## Configuration

Copy `config.example.yaml` to `config.yaml` and edit it.

| Section | Controls |
|---|---|
| `model` | Model name (selects backend), server host/port, GPU/vLLM tuning |
| `extraction` | Image size, parallelism, optional confidence column, input extensions |
| `jobs[]` | Per-directory: `input_dir`, `output_csv`, and the `fields` to extract |

A field is a mapping with a `description` (recommended — it guides the model) or
a bare string:

```yaml
jobs:
  - name: invoices
    input_dir: /data/input/invoices
    output_csv: /data/output/invoices.csv
    fields:
      - name: invoice_number
        description: "The unique invoice number/identifier"
      - supplier_name            # bare string, no description
```

`extraction.confidence` adds a `confidence` column with a per-entity score:

| value | column shows | cost |
|---|---|---|
| `none` (default) | *(no confidence column)* | 1 VLM call/doc |
| `binary` | `High` / `Low` | ~2 VLM calls/doc |
| `numeric` | a `0`–`100` score | ~2 VLM calls/doc |

(The legacy `include_confidence: true/false` key still works, mapping to
`binary`/`none`.)

Legacy `model` keys (`vllm_host`, `vllm_port`, `start_vllm`,
`vllm_start_timeout`) are still accepted as aliases.

### Rotating pages

If your scans are misoriented, rotate pages before extraction (degrees
**clockwise**). Rotation is **per job**, so each directory can have its own
orientation; `extraction.rotate` is just the default for jobs that don't set their
own:

```yaml
extraction:
  rotate: 0           # default for all jobs

jobs:
  - name: invoices
    input_dir: /data/input/invoices
    # ... no rotate -> uses the default (0)
  - name: patient_records
    input_dir: /data/input/patient_records
    rotate: 90        # this directory's scans are sideways -> overrides the default
    # ...
```

Right-angle values (90/180/270) are lossless reorientations. Arbitrary angles are
allowed too — the canvas is expanded and the new corners are filled white. The
rotation is applied to the whole page before the image is downscaled and sent to
the model. Use the `notebooks/rotation_test.ipynb` notebook to preview an angle
before committing it to the config.

---

## Performance tuning

The GPU is the bottleneck. These config knobs trade speed against accuracy or
memory — tune them while watching throughput (docs/min) and GPU memory
(`nvidia-smi`).

| Setting | Effect | Trade-off |
|---|---|---|
| `extraction.concurrency` | Documents sent to vLLM at once; vLLM batches them on the GPU. **The main throughput lever.** | Higher = better GPU utilisation, up to saturation; beyond that, requests just queue (no harm). Try 8–32. |
| `extraction.max_img_size` | Longest image side. VLM cost scales with image tokens (≈ resolution²), so this is the **biggest speed lever**. | Lower = much faster + less memory, but small text/handwriting may become unreadable. |
| `extraction.confidence` | `none` = 1 VLM call/doc; `binary`/`numeric` = 2 calls/doc. | `none` is ~2× faster. Only enable confidence if you need the score. |
| `extraction.max_tokens` | Cap on generated tokens per field/doc. | Lower is faster if your answers are short; too low truncates long values. |
| `model.gpu_memory_utilization` | Fraction of GPU RAM vLLM uses for the KV cache. | Higher = more room for concurrent sequences (more throughput); too high risks OOM with other processes. |
| `model.max_model_len` | vLLM context length. | Lower frees memory for more concurrency; must still fit prompt + image tokens + output. |
| `model.name` | Model size/quantisation. | A smaller or AWQ-quantised model (e.g. `...-7B-Instruct-AWQ`) is faster and lighter than a full 32B; larger models are slower but may extract better. |

Notes:
- vLLM is started with `--enforce-eager` (by docext), which disables CUDA-graph
  capture. That trades a little speed for lower memory and faster startup; it is
  not configurable through this wrapper without patching docext.
- The model loads **once** per run (see below), so per-document cost is pure
  inference — there's no reload overhead to optimise away.

## Running in parallel

**Yes — within a single run.** Set `extraction.concurrency`. The wrapper submits
that many documents to the one vLLM server concurrently, and vLLM's continuous
batching runs them together on the GPU. This is the intended way to parallelise:

```yaml
extraction:
  concurrency: 16        # 16 documents in flight at once
```

That's it — one container, one model load, many concurrent requests. Raise
`concurrency` until `nvidia-smi` shows the GPU saturated or memory is full; higher
values past that point simply queue.

### Don't run multiple containers on the same GPU

Each container starts its **own** vLLM server and loads the model again, so two
containers on one GPU means two copies of the weights in VRAM (likely OOM) plus
contention — strictly worse than one container with higher `concurrency`. The
whole design is *load once, batch many*, so scale with `concurrency`, not with
more containers.

### Multiple GPUs: one container per GPU (data-parallel)

If you have several GPUs, run **one container per GPU**, each pinned to a different
device and given a different subset of jobs (split your directories across configs):

```bash
# GPU 0 — first half of the directories
docker run --rm --gpus '"device=0"' --shm-size=16g --security-opt label=disable \
  -v "$PWD/models/hf_cache":/models/hf_cache \
  -v "$PWD/config.gpu0.yaml":/app/config.yaml \
  -v "$PWD/data":/data \
  ner-ocr-docext:amd64 --config /app/config.yaml &

# GPU 1 — second half
docker run --rm --gpus '"device=1"' --shm-size=16g --security-opt label=disable \
  -v "$PWD/models/hf_cache":/models/hf_cache \
  -v "$PWD/config.gpu1.yaml":/app/config.yaml \
  -v "$PWD/data":/data \
  ner-ocr-docext:amd64 --config /app/config.yaml &
wait
```

Each loads the model once on its **own** GPU (no contention), and each writes its
own job CSVs. This is the supported way to use more than one GPU — the wrapper
doesn't drive vLLM tensor-parallelism (multi-GPU for a *single* model), since
docext 0.1.11's launcher doesn't expose `--tensor-parallel-size`.

---

## Logging

By default the output is quiet: you see the **initialisation logs** (config,
model startup) and **per-file progress** (`[job] 3/10 done (file.tiff)`), but not
the chatty per-request internals of vLLM / litellm / transformers.

```bash
# default — init logs + per-file progress only
... --config /app/config.yaml

# full debug firehose (vLLM/litellm/transformers internals, file:line, etc.)
... --config /app/config.yaml --verbose

# advanced: set the level explicitly (overrides --verbose)
... --config /app/config.yaml --log-level WARNING
```

`--verbose` is the toggle; use it when diagnosing a problem and leave it off for
normal runs.

---

## Project layout

```
docext_batch/
├── config.py     # YAML config loading + validation (dataclasses)
├── images.py     # document discovery + TIFF→JPEG conversion
├── server.py     # model endpoint lifecycle (vLLM / Ollama / external)
├── extract.py    # per-document extraction + per-job CSV assembly
└── run.py        # CLI entrypoint: prepare model, run jobs, write CSVs
Dockerfile             # GPU/TRE image — CUDA 12.4 base, pip-install docext, no flash-attn; optional model bake
Dockerfile.dev         # Mac/CPU image (docext --no-deps + Ollama)
docker-compose.dev.yml # all-in-Docker local stack (Ollama + app)
config.example.yaml    # production (vLLM) config template
config.dev.yaml        # local (Ollama) config
scripts/
├── download_model.sh  # pre-download a model into models/hf_cache (offline/TRE)
├── export_model.sh    # tar models/ -> models.tar.gz for transport into the TRE
└── export_image.sh    # build amd64 GPU image + save as a tarball for the TRE
tests/                 # pytest suite (no GPU/model needed; docext is stubbed)
```

### Running the tests

The tests cover config validation, image prep, key normalisation and CSV
assembly. They stub out docext, so no GPU or model server is required:

```bash
pip install pytest pyyaml pillow pandas
pytest
```

## Notes & limitations

- Inputs are expected to be **single-page** TIFFs; only the first frame of a
  multi-frame TIFF is used.
- Extraction quality depends on the model and your field descriptions — write
  clear, specific descriptions (with a normalisation hint where it matters,
  e.g. "date as YYYY-MM-DD").
- Blank `error` = success; a non-empty value is that document's exception
  message (its field columns will be empty).
- The Mac/Ollama path is for **development and validation**; use a vLLM model on
  GPU for production-quality results.
