# MuseTalk Separate TensorRT Environment Plan

## Purpose

This document defines the next branch for TensorRT work after the current
`/content/py310` environment proved insufficient for a trustworthy full VAE
TensorRT path.

The goal is:

- keep the current stable MuseTalk server environment untouched
- create a separate TensorRT-focused experiment environment
- use that environment only for export and benchmark work first
- only consider moving the full server there if the benchmark materially improves

## Why A Separate Environment

The current environment already proved three things:

- the repo-side TensorRT integration work is mostly in place
- a VAE engine artifact could be exported once
- but the current stack still failed at trustworthy runtime activation and then
  failed a stricter full compile because of unsupported operators

So the next problem is no longer “missing code wiring.”
It is “the current backend stack is the wrong place to keep forcing this.”

## Best Practical Choice On This Machine

This machine does **not** currently have Docker available.

That means the best practical next move is:

- a **separate Python 3.10 venv**
- on the same machine
- with a **newer coherent Torch / Torch-TensorRT / TensorRT stack**
- while leaving `/content/py310` untouched

If Docker or another container runtime becomes available later, a containerized
TensorRT environment would be even cleaner. But for now, a second venv is the
best practical option.

## Environment Layout

Recommended paths:

- current stable env:
  - `/content/py310`
- new TRT experiment env:
  - `/content/py310_trt_exp`
- shared repo:
  - `/content/MuseTalk`
- separate TRT artifact directories:
  - `/content/MuseTalk/models/tensorrt_altenv`
  - `/content/MuseTalk/models/tensorrt_altenv_bs32`

Using a separate artifact directory matters because we do **not** want to mix:

- current-environment TensorRT artifacts
- alternate-environment TensorRT artifacts

## Current Attempt Status

The separate TensorRT environment is now real and technically validated, but
the current active broad-batch TRT VAE artifact is **not yet functionally
trustworthy** for visual output.

Current confirmed state:

- `/content/py310_trt_exp` exists and imports the backend stack successfully
- installed backend family in that env:
  - `torch==2.5.1`
  - `torchvision==0.20.1`
  - `torchaudio==2.5.1`
  - `torch-tensorrt==2.5.0`
  - `tensorrt==10.3.0`
- the repo-side exporter needed two compatibility fixes for this newer stack:
  - normalize old `dynamo_compile` references to `dynamo`
  - pass example tensor inputs into `torch_tensorrt.save(...)`
- a first successful VAE export was then produced in the alternate artifact dir:
  - `/content/MuseTalk/models/tensorrt_altenv/vae_decoder_trt.ts`
  - `/content/MuseTalk/models/tensorrt_altenv/vae_decoder_trt_meta.json`
  - final successful profile:
    - `--batch-sizes 1,4,8`
  - metadata:
    - batch range `[1, 8]`
    - opt batch `4`
  - saved engine size:
    - about `137.5 MB`
- a newer broader VAE export was then produced for HLS-like batch shapes:
  - `/content/MuseTalk/models/tensorrt_altenv_bs32/vae_decoder_trt.ts`
  - `/content/MuseTalk/models/tensorrt_altenv_bs32/vae_decoder_trt_meta.json`
  - successful profile:
    - `--batch-sizes 4,8,16,32,48`
  - metadata:
    - batch range `[4, 48]`
    - opt batch `16`
  - saved engine size:
    - about `132 MB`
- runtime loading is also validated in `/content/py310_trt_exp`
  with fallback disabled
- a direct decode smoke test succeeded on `cuda:0`
- a broader isolated benchmark also now exists:
  - `/content/MuseTalk/benchmark_pipeline_trt_vae_bs32.json`
  - batch set `[4, 8, 16, 32, 48]`
  - best throughput `61.3 fps` at `batch_size=32`
- broad PyTorch comparison already exists:
  - `/content/MuseTalk/benchmark_pipeline_results.json`
  - best throughput `51.1 fps` at `batch_size=32`

Important current limit:

- `/content/py310` is still **not** TensorRT-ready
  - `torch_tensorrt` import fails there
  - `tensorrt` import fails there
- `/content/py310_trt_exp` now starts the HLS `api_server.py` path successfully
  with TRT VAE active
  - WebRTC is still disabled there because `aiortc` is not installed
  - avatar preparation is still not the validated path there because the
    `mmpose` stack has not been brought over
  - the currently validated HLS runtime shape is:
    - eager UNet
    - TRT VAE
  - a later server run with compiled UNet + TRT VAE failed on the first HLS
    generation batch during CUDA graph capture with:
    - `CUDNN_STATUS_INTERNAL_ERROR_DEVICE_ALLOCATION_FAILED`
    - `CUDA error: operation failed due to a previous error during capture`

Practical meaning:

- the alternate env is now good enough for:
  - export
  - runtime loader validation
  - isolated model-path benchmarking
  - HLS `api_server.py` startup with TRT active
- a first backend-active HLS load test now also exists:
  - `/content/MuseTalk/load_test_report.json`
  - eager UNet + TRT VAE
  - `concurrency=1`
  - `batch_size=4`
  - `avg_time_to_live_ready_s=3.015`
  - `avg_segment_interval_s=0.196`
  - `max_segment_interval_s=1.512`
- the alternate env now also has a real measured result:
  - the broader VAE-only TRT branch improves the isolated benchmark by roughly
    `+20%`
  - but it still does **not** get close to the `96 fps` target
- the current active broad-batch TRT artifact is now also known to be
  **visually wrong**
  - HLS `/wall` output shows the talking-face ROI replaced by a flat gray mask
  - direct A/B decode checks show the TRT face output collapses into a narrow
    mid-gray band before blend
  - cached/avatar latent decode:
    - PyTorch range `0..250`, mean `63.2`
    - TRT range `104..136`, mean `120.2`
    - MAE about `87.1`
  - real UNet-predicted latent decode:
    - PyTorch range `0..236`, mean `116.6`
    - TRT range `93..150`, mean `120.3`
    - MAE about `53.3`
  - export-wrapper control test in pure PyTorch still matches the normal VAE
    path closely
    - MAE about `5.3e-05`
    - max abs about `0.0022`
- it is **not** yet proven for:
  - multi-stream HLS `load_test.py`
  - WebRTC in the TRT env

Important operational caveat:

- the backend-active HLS load test exposed ffmpeg encoder failure on
  `h264_nvenc`
- segments were produced only because the code fell back to CPU `libx264`
- observed errors included:
  - `OpenEncodeSessionEx failed: unsupported device (2)`
  - `No capable devices found`
  - repeated `Broken pipe` retries
- so the next concurrency validation step should either fix NVENC in this env
  or deliberately treat the next load tests as CPU-encode-fallback runs

Important correctness caveat:

- the current active broad-batch TRT artifact in
  `models/tensorrt_altenv_bs32/vae_decoder_trt.ts` should now be treated as
  **performance-only / untrusted**
- the gray-mask regression is not explained by prepared avatar assets,
  `get_image_blending(...)`, or the wrapper math in
  `scripts/tensorrt_export.py`
- the current strongest evidence is now:
  - the fault is already present in the **in-memory compiled TRT module**
  - the bug is therefore not primarily a save/load-only problem
- direct post-patch validation still fails against the active artifact:
  - command:
    - `python scripts/validate_vae_backend.py --avatar-id test_avatar --trt-dir ./models/tensorrt_altenv_bs32`
  - PyTorch output range: `0.0..0.9814`, mean `0.2485`
  - TRT output range: `0.3989..0.5337`, mean `0.4714`
  - MAE: about `0.3408`
- the repo now has two new correctness guardrails:
  - `scripts/tensorrt_export.py` can run post-export VAE validation and record
    the result in `vae_decoder_trt_meta.json`
  - `scripts/trt_runtime.py` can refuse unvalidated or known-bad artifacts when
    `MUSETALK_TRT_REQUIRE_VALIDATION=1`
- an exact-batch FP16 retry has now also been completed:
  - output dir: `models/tensorrt_fp16_bs4`
  - batch range: `[4, 4]`
  - save format: `torchscript`
  - compile: passed
  - save: passed
  - validation: failed with `mae=0.340751`
- practical meaning:
  - the visual regression is not just a wide dynamic-shape engine issue
  - even the exact `batch_size=4` FP16 TRT VAE artifact still collapses toward
    the same gray output band
  - the current evidence therefore points more strongly at the FP16 TRT VAE
    compile/runtime behavior on this stack
- an in-memory compiled-module correctness check is now also complete:
  - script:
    - `scripts/validate_vae_trt_inmemory.py`
  - batch size:
    - `4`
  - precision:
    - `fp16`
  - PyTorch output range:
    - `0.0..0.9814`, mean `0.2485`
  - TRT in-memory output range:
    - `0.3989..0.5342`, mean `0.4714`
  - MAE:
    - `0.3407516`
  - practical meaning:
    - the corruption is already present before any save/load boundary
- a decoder stage-inspection pass is now also complete:
  - script:
    - `scripts/inspect_vae_trt_stages.py`
  - first bad stage:
    - `decoder_mid_block`
  - stages that still match exactly:
    - `scale_post_quant`
    - `decoder_conv_in`
  - first divergent stage metrics:
    - `decoder_mid_block`
    - MAE `0.4712`
    - max abs `6.0391`
  - final normalization stage:
    - still matches exactly when given the same pre-normalized tensor
  - practical meaning:
    - the bug localizes to the decoder core, not the final output clamp/scale

Current serialization-path findings on this stack:

- loadable TorchScript-style broad artifact:
  - `models/tensorrt_altenv_bs32/vae_decoder_trt.ts`
  - loads, benchmarks faster, but produces gray-mask output
- `exported_program` non-retrace save attempt:
  - `models/tensorrt_altenv_ep_bs32_v3/vae_decoder_trt.ts`
  - saved successfully
  - failed reload with symbolic-shape deserialization errors (`KeyError: s0`)
- `exported_program` retrace save attempt:
  - `models/tensorrt_altenv_ep_bs32_v4`
  - failed during save because TRT engine state export tripped:
    - `NotImplementedError: '__len__' is not implemented for __torch__.torch.classes.tensorrt.Engine`
- exact-batch FP16 TorchScript save attempt:
  - `models/tensorrt_fp16_bs4/vae_decoder_trt.ts`
  - compiled and saved successfully
  - failed post-export validation with:
    - `RuntimeError: Post-export VAE validation failed: mae=0.340751 > max_mae=0.050000`

Recommended next branch inside this alternate environment:

1. keep the saved monolithic TRT VAE artifacts marked visually untrusted
2. use the new runtime-only stagewise backend as the active repair branch
3. keep future experiments on exact scheduler buckets instead of wide dynamic
   profiles
4. treat `batch_size=4` as the first repaired bucket and first HLS visual
   checkpoint
5. validate wider stagewise correctness next, then measure speed
6. only widen HLS validation after stagewise TRT is both correct and fast

New stagewise runtime update in this alternate env:

- `scripts/trt_runtime.py` now includes backend `trt_stagewise`
- it compiles exact-batch decoder stages in memory and caches them by batch size
- it uses `native_group_norm` PyTorch fallback during stage compilation
- first full validation result:
  - script:
    - `scripts/validate_vae_backend.py --backend trt_stagewise`
  - batch:
    - `4`
  - report:
    - `tmp/vae_stagewise_backend_validation_bs4/report.json`
  - metrics:
    - MAE `0.0005082`
    - max abs `0.0097656`
  - practical meaning:
    - the gray-mask collapse is no longer present in this stagewise path at
      `batch_size=4`
- a later real HLS `/wall` run in `/content/py310_trt_exp` now also appears
  visually repaired on this path:
  - real mouth output is visible again instead of the gray ROI slab
  - this was observed on the stagewise `batch_size=4` server configuration
  - so the alternate env now has both:
    - metric validation at `batch_size=4`
    - and a first user-visible wall validation at the same bucket
- a later stagewise HLS `load_test.py` run in this same env now also produced
  the first encouraging `concurrency=8` result:
  - `batch_size=4`
  - `playback_fps=24`
  - `musetalk_fps=12`
  - `hold_seconds=30`
  - `completed=8`
  - `failed=0`
  - `avg_time_to_live_ready_s=1.631`
  - `avg_segment_interval_s=1.769`
  - `max_segment_interval_s=2.524`
  - GPU average util about `83.76%`, peak about `96%`
  - GPU memory stayed around `6742 MB`
  - practical meaning:
    - the repaired stagewise path is no longer just a synthetic decode fix
    - it now has a promising real `concurrency=8` HLS result too
    - but it still remains slightly above the repo's throttling threshold
- later Threadripper HLS runs in this same alternate env then exposed the next
  real bottleneck:
  - the GPU became underfed by the host pipeline
  - one representative poor `concurrency=8` run showed:
    - `avg_time_to_live_ready_s=3.469`
    - `avg_segment_interval_s=3.622`
    - `max_segment_interval_s=6.137`
    - `avg GPU util ~= 37.2%`
  - practical meaning:
    - the alternate TRT environment is no longer blocked mainly by VAE decode
      correctness at `batch_size=4`
    - the next implementation branch inside this env should be host-side HLS
      pipeline refactoring, not another blind CPU-thread-cap experiment
    - the highest-value next refactors are:
      - parallelize shared HLS prep in `scripts/hls_gpu_scheduler.py`
      - reduce avatar cache-miss cost in `scripts/api_avatar.py`
      - replace per-chunk `ffmpeg` spawn with a persistent encode path
      - refactor compose to use adjacent CPU cores more effectively
      - consolidate older direct live-serving paths in `api_server.py`
- the first host-side multithreaded refactor slice has now landed in this same
  alternate env:
  - batched audio feature extraction / Whisper segment encode
  - vectorized prompt construction
  - concurrent HLS prep subtasks
  - parallel avatar cache-miss frame / mask loading
- first measured `concurrency=8` result after that slice:
  - `avg_time_to_live_ready_s=1.760`
  - `avg_segment_interval_s=1.733`
  - `max_segment_interval_s=2.535`
  - `avg GPU util ~= 82.06%`
  - `completed=8/8`
  - practical meaning:
    - the host-side refactor branch is validated inside the alternate TRT env
    - the severe CPU-starved regression is largely recovered
    - the path is still slightly throttled, so persistent encode / compose
      follow-on work is still justified
- later March 24 ramp results in this same env now show:
  - `concurrency=6`
    - `avg_time_to_live_ready_s=1.342`
    - `avg_segment_interval_s=1.294`
    - `max_segment_interval_s=2.032`
    - `avg GPU util ~= 83.84%`
    - practical meaning:
      - this is the first practical realtime milestone for the repaired
        stagewise branch on this box
      - the warning threshold is only missed by about `32ms`
  - `concurrency=7`
    - `avg_time_to_live_ready_s=1.508`
    - `avg_segment_interval_s=1.516`
    - `max_segment_interval_s=2.530`
  - repeated `concurrency=8`
    - `avg_time_to_live_ready_s=1.569-1.760`
    - `avg_segment_interval_s=1.733-1.736`
    - `max_segment_interval_s=2.524-2.535`
    - practical meaning:
      - `7` and `8` are now in the same batch-4 saturation band
      - the remaining work is mainly about shrinking tail jitter

## Core Principle

Do **not** piece together the new environment one package at a time the way the
current environment ended up being assembled.

Instead:

- choose one coherent PyTorch + CUDA + Torch-TensorRT family
- choose the matching TensorRT Python/runtime packages for that family
- install them together in the new venv

The key mistake to avoid is mixing:

- older Torch
- newer TensorRT bits
- or export/runtime stacks that are only accidentally compatible

One more mistake to avoid:

- do **not** run an unpinned `pip install torch-tensorrt tensorrt`

That now proved to be dangerous because pip selected:

- `torch-tensorrt 2.10.0`
- a newer `torch` family
- `tensorrt-cu13`

which is outside the intended compatibility family for this branch.

## Phase 0: Historical Cleanup Guardrail

This was the guardrail that mattered during the first failed setup attempt.
It is no longer the active blocker on the current machine, but it should still
be followed if the alternate env ever needs to be recreated from scratch.

Recommended commands:

```bash
deactivate 2>/dev/null || true
rm -rf /root/.cache/pip
rm -rf /content/py310_trt_exp
df -h /content
```

Why:

- the first install attempt failed with `OSError: [Errno 28] No space left on device`
- Torch + CUDA wheels are large enough that low free space will produce noisy,
  partially installed states
- clearing the pip cache first is the fastest way to recover several gigabytes

## Phase 1: Create The New Venv

```bash
python3.10 -m venv /content/py310_trt_exp
source /content/py310_trt_exp/bin/activate
python -m pip install --upgrade pip setuptools wheel
```

## Phase 2: Install A Coherent New Torch/TRT Stack

Use the official compatibility guidance for the chosen stack and install the
core packages together:

- `torch`
- `torchvision`
- `torchaudio`
- `torch-tensorrt`
- `tensorrt` or the matching NVIDIA runtime packages

Guidance for this phase:

- stay on Python `3.10`
- prefer one CUDA family consistently across the stack
- do **not** mix the old `/content/py310` Torch stack into this new venv
- do **not** use unpinned `torch-tensorrt`
- verify importability immediately after install

### Current Confirmed Stack

The alternate env is already created and working on the current machine.

- Python `3.10`
- NVIDIA driver `580.95.05`
- GPU `NVIDIA GeForce RTX 3090`
- free space on `/content`: about `72G`

Current confirmed working stack in `/content/py310_trt_exp`:

- `torch==2.5.1`
- `torchvision==0.20.1`
- `torchaudio==2.5.1`
- `torch-tensorrt==2.5.0`
- `tensorrt==10.3.0`

If the env ever needs to be recreated from scratch, use the same pinned
`cu121` / `torch-tensorrt 2.5.x` family:

Recommended commands:

```bash
source /content/py310_trt_exp/bin/activate

python -m pip install --no-cache-dir \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install --no-cache-dir \
  --extra-index-url https://pypi.nvidia.com \
  torch-tensorrt==2.5.0
```

Important notes:

- `--no-cache-dir` is still a good default for large wheel installs
- the earlier unpinned install attempted to move to `torch-tensorrt 2.10.0`
  and `tensorrt-cu13`, which is not the planned stack
- the earlier disk-space blocker is historical, not the current machine state

Minimum validation:

```bash
python - <<'PY'
import torch
import torch_tensorrt
import tensorrt
print("torch", torch.__version__)
print("torch_tensorrt", torch_tensorrt.__version__)
print("tensorrt", tensorrt.__version__)
print("Engine class", getattr(torch.classes.tensorrt, "Engine"))
PY
```

If that last line fails, stop there. The environment is not ready.

If the import fails with `No module named 'torch'`, that means the new torch
install never completed successfully. That is not a TensorRT compatibility
signal by itself; it is just an incomplete environment.

## Phase 3: Install Only The Dependencies Needed For The Current Goal

The current branch has already shown that it is better to stage this env in
layers instead of immediately cloning the full stable server environment.

What is already installed in `/content/py310_trt_exp` beyond the backend stack:

- `opencv-python`
- `diffusers==0.30.2`
- `transformers==4.39.2`
- `accelerate==0.28.0`
- `huggingface_hub==0.30.2`
- `einops==0.8.1`
- `safetensors==0.7.0`
- `pillow==11.3.0`
- `numpy==1.23.5`

Why this matters:

- that minimal set is enough for:
  - exporter runs
  - runtime loader validation
  - isolated benchmark validation
- that set alone is **not** enough for:
  - `api_server.py`
  - WebRTC/HLS serving

Current validated HLS/api_server dependency set:

- `fastapi==0.135.1`
- `uvicorn==0.42.0`
- `aiohttp==3.13.3`
- `soundfile==0.12.1`
- `librosa==0.11.0`
- `imageio[ffmpeg]==2.37.3`
- `omegaconf==2.3.0`
- `ffmpeg-python==0.2.0`
- `aiofiles==24.1.0`
- `av==17.0.0`

Important packaging lesson:

- keep `numpy==1.23.5`
- do **not** pull unpinned UI packages such as `gradio` into the TRT env for
  HLS validation
- the earlier drift to `numpy 2.2.6`, `gradio 6.9.0`, and
  `huggingface_hub 1.7.2` broke OpenCV imports and the `transformers` family

Current easiest reproduction path:

```bash
cd /content/MuseTalk
scripts/setup_trt_experiment_env.sh --clean --install-server-deps
```

Current rule:

- do **not** install the full repo requirements first
- do **not** install unpinned `gradio` / `moviepy` into the TRT env unless you
  explicitly need a different entrypoint than HLS `api_server.py`

## Phase 4: Keep TRT Artifacts Separate

When exporting in the new venv, do **not** overwrite the current TensorRT
artifact path first.

Use:

```bash
export MUSETALK_TRT_DIR=./models/tensorrt_altenv_bs32
```

That lets us compare:

- current-environment artifacts
- alternate-environment artifacts
- and keep the currently active broad-batch validation artifact separate from
  the earlier small-batch smoke artifact

without ambiguity.

## Phase 5: Export First, Then Server Validation

The first job of the new environment is:

- export the VAE TensorRT engine

The export step should be isolated first.

Recommended command shape:

```bash
cd /content/MuseTalk
export MUSETALK_TRT_DIR=./models/tensorrt_altenv_bs32
python scripts/tensorrt_export.py \
  --components vae \
  --batch-sizes 4,8,16,32,48
```

Important current note:

- the older profile `4,8,16,24,32,40,48` was too aggressive for the RTX
  3090 builder path in this env and repeatedly ran into builder-memory pressure
- the later profile `4,8,16,32,48` successfully built and is now the active
  broad-batch artifact for validation

## Phase 6: Runtime Loader Validation

Before benchmarking, validate that the engine can really load in the new env
with fallback disabled.

```bash
cd /content/MuseTalk
MUSETALK_TRT_ENABLED=1 \
MUSETALK_VAE_BACKEND=trt \
MUSETALK_TRT_FALLBACK=0 \
MUSETALK_TRT_DIR=./models/tensorrt_altenv_bs32 \
python - <<'PY'
from scripts.trt_runtime import load_vae_trt_decoder
backend = load_vae_trt_decoder()
print("backend", type(backend).__name__, getattr(backend, "name", None))
PY
```

If this fails, do **not** run the benchmark yet.

Current status:

- this validation now **passes** in `/content/py310_trt_exp`
- `load_vae_trt_decoder(...)` successfully loads the saved engine
- the active broad-batch engine metadata is `[4, 48]` with opt batch `16`
- a direct decode test has succeeded there

## Phase 7: Benchmark Before Any Server Migration

Only after load succeeds, run the isolated benchmark.

Current recommended first benchmark:

```bash
cd /content/MuseTalk
MUSETALK_TRT_ENABLED=1 \
MUSETALK_VAE_BACKEND=trt \
MUSETALK_TRT_FALLBACK=0 \
MUSETALK_TRT_DIR=./models/tensorrt_altenv_bs32 \
MUSETALK_COMPILE_VAE=0 \
python scripts/benchmark_pipeline.py \
  --batch-sizes 4,8,16,32,48 \
  --warmup 5 \
  --iters 20 \
  --output-json benchmark_pipeline_trt_vae_bs32.json
```

This is the real gate for whether the alternate environment is worth pursuing.

Why this shape:

- it matches the active broad-batch engine's supported range
- it directly answers whether TRT helps at the HLS-like combined batch sizes
- it is the right gate before API-server migration

Additional current benchmark result:

- TRT VAE benchmark:
  - `batch_size=4`: `53.4 fps`
  - `batch_size=8`: `57.5 fps`
  - `batch_size=16`: `60.2 fps`
  - `batch_size=32`: `61.3 fps`
  - `batch_size=48`: `61.2 fps`
- broad PyTorch benchmark:
  - `batch_size=4`: `45.8 fps`
  - `batch_size=8`: `48.7 fps`
  - `batch_size=16`: `50.5 fps`
  - `batch_size=32`: `51.1 fps`
  - `batch_size=48`: `50.9 fps`

Interpretation:

- the backend is truly active and materially faster across the broad `4..48`
  range
- the best broad throughput gain is about `+20.0%` at `batch_size=32`
- but the model-path ceiling is still only about `7.7 fps/stream` at
  `8` concurrent
- that is still far below the `8 x 12 fps` target

## Success Criteria

### Technical Success

- the engine exports cleanly
- the runtime loader activates the backend with fallback disabled
- benchmark logs show:
  - `VAE decode backend: tensorrt`

### Performance Success

At least one of these should happen:

- best total throughput rises materially above the matched PyTorch baseline
- VAE full-path latency drops materially versus matched PyTorch baseline

If neither happens, then the alternate environment is not worth migrating.

Current measured outcome:

- technical success: **yes**
- isolated model-path performance win: **yes, material**
- HLS `api_server.py` startup in `/content/py310_trt_exp`: **yes**
- visual correctness of the active broad-batch TRT artifact: **no**
- strong enough to claim the final `8 x 12 fps` goal is solved: **no**

Important comparison note:

- compare against the PyTorch path on the **same** broad batch set first
  - `4,8,16,32,48`
- the older small-batch `1,2,4,8` run remains useful as an early smoke
  milestone, but it is no longer the main decision driver

## Decision Gate

Current decision from the measured benchmark and the later correctness check:

1. keep using the working TRT HLS server startup in `/content/py310_trt_exp`
2. treat `models/tensorrt_altenv_bs32/vae_decoder_trt.ts` as performance-only
   until correctness is restored
3. do **not** use the current active TRT artifact for lip-sync validation
4. investigate a correctness-safe backend path before more migration work

If the alternate environment still fails or does not improve the benchmark:

- stop pushing the TensorRT VAE branch
- pivot to the next backend option, such as ONNX Runtime

## What Not To Do

- do not modify the stable `/content/py310` environment further for TensorRT
- do not overwrite the stable artifact path with alternate-environment outputs
- do not migrate the full API server before the new environment wins the benchmark

## Bottom Line

The next clean branch is now:

1. keep `/content/py310` unchanged as the stable PyTorch server env
2. use `/content/py310_trt_exp` only for TRT export/runtime/benchmark work
3. treat the broader TRT VAE result as a performance datapoint, not a valid
   visual-output milestone
4. treat the active broad-batch TRT artifact as untrusted until the gray-mask
   regression is fixed
5. restore correctness before more TRT-backed HLS load-test interpretation
6. if needed after that, move to the next larger acceleration step
