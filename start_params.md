# Start Params

This file is now the source of truth for the current MuseTalk TRT-stagewise HLS
server path, especially for Vast.ai-style deployments.

The server no longer targets the old `/content/py310` venv by default. The
current working runtime is:

- repo: `/workspace/MuseTalk`
- venv: `/workspace/.venvs/musetalk_trt_stagewise`
- VAE backend: `trt_stagewise`
- launcher: [`scripts/run_trt_stagewise_server.sh`](./scripts/run_trt_stagewise_server.sh)
- fresh-node setup: [`scripts/setup_trt_stagewise_server_env.sh`](./scripts/setup_trt_stagewise_server_env.sh)
- Vast on-start wrapper: [`scripts/vast_onstart.sh`](./scripts/vast_onstart.sh)
- Vast server control helper: [`scripts/vast_server_ctl.sh`](./scripts/vast_server_ctl.sh)

The shell launchers are path-neutral now. If the repo lives under
`/workspace/MuseTalk`, their defaults resolve to `/workspace/.venvs/...`. If
the repo lives under `/content/MuseTalk`, they still fall back to `/content`.

## Validated Live State

The current single-venv path has now been verified end-to-end on a CUDA 12.1
node:

- toolkit: CUDA `12.1`
- GPU class tested live: RTX `3090`
- shared venv:
  - server/runtime imports passed:
    - `api_server`
    - `torch`
    - `torch_tensorrt`
    - `tensorrt`
  - avatar-prep imports passed:
    - `mmcv`
    - `mmcv._ext`
    - `mmengine`
    - `mmdet`
    - `mmpose`
- avatar preparation was re-tested successfully after the S3FD face-detector
  weight was added to bootstrap

Current package-side validation on that node:

- `torch==2.5.1+cu121`
- `torch_tensorrt==2.5.0`
- `tensorrt==10.3.0`
- `mmcv==2.1.0`
- `mmengine==0.10.4`
- `mmdet==3.2.0`
- `mmpose==1.3.1`

## Fresh Server Setup

Create the current TRT-stagewise server env from scratch:

```bash
cd /workspace/MuseTalk
bash scripts/setup_trt_stagewise_server_env.sh --clean
```

What that does:

- installs the required system packages for the current server path
- recreates `/workspace/.venvs/musetalk_trt_stagewise`
- installs the pinned PyTorch + Torch-TensorRT stack
- installs the pinned HLS/api_server dependencies
- includes the current WebRTC runtime deps used by `api_server.py`
- downloads and validates the current model weights
- runs a final import smoke test against `api_server.py`

Important note:

- avatar-preparation deps (`mmpose/mmcv/mmdet/mmengine`) are **not** part of the
  default server bootstrap anymore
- this keeps autoscaled inference workers on the stable inference-only path
- if you need a single CUDA 12.1 node to support both avatar preparation and
  TRT inference in the same venv, opt in with:

```bash
cd /workspace/MuseTalk
bash scripts/setup_trt_stagewise_server_env.sh --clean --full-stack
```

or in the Vast wrapper:

```bash
SETUP_FULL_STACK=1 bash scripts/vast_onstart.sh
```

Compatibility note:

- `--install-avatar-prep-deps` and `SETUP_INSTALL_AVATAR_PREP_DEPS=1` still
  work
- `--full-stack` and `SETUP_FULL_STACK=1` are the preferred forms because they
  describe the intended one-venv outcome directly
- full-stack avatar preparation requires full `mmcv`; `mmcv-lite` is not
  enough because the preprocessing path imports `mmcv._ext`
- avatar prep now validates `mmcv._ext`, so a "successful" full-stack install
  really means the compiled MMCV ops are present
- if `scripts/vast_onstart.sh` needs to bootstrap an already-existing target
  venv, it now recreates that venv cleanly instead of attempting an unsupported
  in-place upgrade
- `download_weights.sh` now also stages the S3FD face-detector weight at
  `models/face_detection/s3fd.pth`, so avatar prep should not need an external
  runtime download on the first request

## Startup Scripts

### Stable Baseline

```bash
cd /workspace/MuseTalk
bash scripts/run_trt_stagewise_server.sh --profile baseline
```

This short command is the intended canonical launcher. The script itself sets
the current baseline runtime env vars internally.

This is the safe TRT-stagewise baseline:

- `HLS_SCHEDULER_MAX_BATCH=4`
- `HLS_SCHEDULER_FIXED_BATCH_SIZES=4`
- `HLS_SCHEDULER_STARTUP_SLICE_SIZE=4`
- `MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=4`
- `MUSETALK_TRT_STAGEWISE_PRECISION=fp16`
- worker pools:
  - `HLS_PREP_WORKERS=8`
  - `HLS_COMPOSE_WORKERS=8`
  - `HLS_ENCODE_WORKERS=8`
- encoder:
  - `HLS_CHUNK_VIDEO_ENCODER=libx264`
  - `HLS_CHUNK_ENCODER_PRESET=ultrafast`
  - `HLS_CHUNK_ENCODER_CRF=28`

### Current Throughput Branch

```bash
cd /workspace/MuseTalk
bash scripts/run_trt_stagewise_server.sh --profile throughput_record
```

This is now GPU-aware. On 24GB RTX 3090-class cards it preserves the widened
branch that produced the current best average throughput at `concurrency=8`:

- `HLS_SCHEDULER_MAX_BATCH=16`
- `HLS_SCHEDULER_FIXED_BATCH_SIZES=8,16`
- `HLS_SCHEDULER_STARTUP_SLICE_SIZE=4`
- default warmup:
  - `MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8,16`
- VAE stagewise precision:
  - `MUSETALK_TRT_STAGEWISE_PRECISION=fp16`
- worker pools remain:
  - `HLS_PREP_WORKERS=8`
  - `HLS_COMPOSE_WORKERS=8`
  - `HLS_ENCODE_WORKERS=8`

Important caveat:

- warming `4,8,16` together previously OOM'd on the RTX 3090
- that is why the launcher defaults to warming `8,16` on this profile
- scheduler fixed buckets must stay aligned with warmed TRT batches
- do not leave `4` in `HLS_SCHEDULER_FIXED_BATCH_SIZES` unless batch `4` is
  also warmed; otherwise tiny tail batches can trigger a live batch-4 TRT
  compile and stall HLS playback

On 32GB V100-class cards, the same profile now defaults to:

- `GPU_TOTAL_MEMORY_GB=32`
- `GPU_RESERVED_MEMORY_GB=8`
- `HLS_SCHEDULER_MAX_BATCH=32`
- `HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16,32`
- `MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=4,8,16,32`

Manual env vars still win. See `docs/gpu_vram_budgeting.md` for the full VRAM
class table.

### VAE Decoder INT8 Experiment Flags

The INT8 VAE decoder path is experimental and disabled by default. It is scoped
to the existing `trt_stagewise` backend and should only be enabled after a real
calibration corpus has been captured from live `pred_latents`. The working
runtime path is now `MUSETALK_TRT_STAGEWISE_INT8_FRONTEND=onnx_qdq`: ModelOpt
exports Q/DQ ONNX for selected stages, TensorRT builds `.plan` engines, and the
runtime executes those engines from PyTorch tensor data pointers.

ModelOpt setup required for `onnx_qdq` INT8:

```bash
bash scripts/setup_trt_stagewise_server_env.sh --install-modelopt
```

The setup script pins `nvidia-modelopt==0.23.2` for this `torch==2.5.1+cu121`
runtime; do not install the latest unpinned ModelOpt into this venv because the
latest package line currently tries to pull a newer Torch/CUDA family.

Capture calibration batches from the shared GPU scheduler:

```bash
MUSETALK_VAE_CALIBRATION_CAPTURE=1 \
MUSETALK_VAE_CALIBRATION_DIR=./calibration/vae_decoder \
MUSETALK_VAE_CALIBRATION_MAX_BATCHES=128 \
bash scripts/run_trt_stagewise_server.sh --profile throughput_record
```

Enable a mixed INT8/FP16 stagewise experiment:

```bash
MUSETALK_TRT_STAGEWISE_PRECISION=int8_mixed \
MUSETALK_TRT_STAGEWISE_INT8_FRONTEND=onnx_qdq \
MUSETALK_TRT_STAGEWISE_INT8_STAGES=decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2 \
MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_DIR=./calibration/vae_decoder \
MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_ALGO=minmax \
MUSETALK_TRT_STAGEWISE_INT8_CACHE_DIR=./models/tensorrt/stagewise_int8_onnx_qdq_cache \
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8,16 \
MUSETALK_TRT_STAGEWISE_WORKSPACE_GB=2 \
HLS_SCHEDULER_MAX_BATCH=16 \
HLS_SCHEDULER_FIXED_BATCH_SIZES=8,16 \
bash scripts/run_trt_stagewise_server.sh --profile throughput_record
```

Current INT8 validation note from 2026-05-26:

- The original two-stage shape above was the first validated live-server INT8
  experiment.
- API logs must include `VAE decode backend active:
  tensorrt_stagewise_int8_mixed` or `backend=tensorrt_stagewise_int8_mixed` for
  generated requests before a run should be treated as an INT8 result.
- In the first batch-8 `/generate` test, the two-stage INT8 VAE decoder reduced
  VAE decode time from about `0.0988s` to `0.0883s`, but end-to-end generation
  improved only about `2.1%` sequentially and was about `3.5%` slower in a
  4-job concurrent test.
- After expanding to five safe INT8 stages, VAE decode time fell to about
  `0.0765s` in `/generate` and about `0.066s` during the latest WebRTC ramp.
- The batch-16 stagewise context is now fixed for the five-stage INT8 path.
  Use `8,16` for throughput experiments when its roughly `17.9 GB` resident
  footprint is acceptable; keep batch `8` as the conservative fallback.

Expanded live INT8 stage list after the one-stage probes:

```text
MUSETALK_TRT_STAGEWISE_INT8_STAGES=decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2
```

Do not add `decoder_up_block_3` or `decoder_postprocess` to the live stage list
yet. Both built successfully with `onnx_qdq`, but the saved comparison crops had
visible color/texture shifts and much higher MAE.

Important caveats:

- The FP16 stagewise path remains the default production path.
- The live API now runs the tested `onnx_qdq` INT8 frontend. The live-serving
  guard remains for the old `torchscript_ptq` frontend because that path crashed
  TensorRT calibration on VAE up-blocks.
- `MUSETALK_TRT_STAGEWISE_PRECISION=int8_mixed` now requires an explicit
  `MUSETALK_TRT_STAGEWISE_INT8_STAGES` list. There is no safe default INT8
  stage set yet.
- `MUSETALK_TRT_STAGEWISE_INT8_ALLOW_UNSAFE_STAGES=1` only matters for
  `torchscript_ptq`; it is not required for the working `onnx_qdq` path.
- The INT8 path uses TensorRT calibration caches per selected stage and exact
  batch size. Delete the directory pointed to by
  `MUSETALK_TRT_STAGEWISE_INT8_CACHE_DIR` if you need to force a fresh
  calibration build.
- Scheduler fixed buckets and `MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES` must stay
  aligned so live traffic does not compile a new quantized batch shape.
- A failed INT8 experiment can be rolled back by setting
  `MUSETALK_TRT_STAGEWISE_PRECISION=fp16`.

Current INT8 debug command, added after the 2026-05-26 calibration failure.
This is for isolated diagnostics only, not a known-good live-server config:

```bash
cd /workspace/MuseTalk
/workspace/.venvs/musetalk_trt_stagewise/bin/python \
  scripts/experiment_vae_decoder_int8.py \
  --stage decoder_up_block_1 \
  --frontend onnx_qdq \
  --batch-size 8 \
  --calibration-batches 8 \
  --calibration-dir ./calibration/vae_decoder \
  --enabled-precisions int8 \
  --min-block-size 1 \
  --output-dir ./tmp/vae_decoder_int8_experiment
```

Run that with the API server stopped, or on another GPU. The live FP16 API keeps
enough VRAM allocated that isolated INT8 compile probes can OOM while the server
is running. If the command succeeds and writes non-empty calibration cache files,
and passes direct image comparison against PyTorch FP16, then the live-serving
guard can be revisited. If it fails, keep the API on
`MUSETALK_TRT_STAGEWISE_PRECISION=fp16` and test a different offline
quantization path before trying multiple INT8 stages. The script writes
`report.json` for both successful comparisons and failed TensorRT builds.

Additional INT8 diagnostic env vars:

```text
MUSETALK_TRT_STAGEWISE_INT8_ENABLED_PRECISIONS=int8
MUSETALK_TRT_STAGEWISE_INT8_FRONTEND=onnx_qdq
MUSETALK_TRT_STAGEWISE_INT8_MIN_BLOCK_SIZE=1
MUSETALK_TRT_STAGEWISE_INT8_REQUIRE_FULL_COMPILATION=0
MUSETALK_TRT_STAGEWISE_INT8_REQUIRE_CALIBRATION_CACHE=1
MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_FORMAT=tensor
```

Set `MUSETALK_TRT_STAGEWISE_INT8_REQUIRE_FULL_COMPILATION=1` only when you want
TensorRT to fail early if the selected stage cannot fully compile.

2026-05-26 experiment result: the live API can run VAE INT8 via `onnx_qdq`.
The current running/proven five-stage server shape is:

```text
MUSETALK_TRT_STAGEWISE_PRECISION=int8_mixed
MUSETALK_TRT_STAGEWISE_INT8_FRONTEND=onnx_qdq
MUSETALK_TRT_STAGEWISE_INT8_STAGES=decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2
MUSETALK_TRT_STAGEWISE_INT8_CACHE_DIR=./models/tensorrt/stagewise_int8_onnx_qdq_cache
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8,16
HLS_SCHEDULER_MAX_BATCH=16
HLS_SCHEDULER_FIXED_BATCH_SIZES=8,16
```

Batch `16` is now enabled in the current live INT8 WebRTC shape as of
2026-05-27. The previous apparent failure was caused by the local relay env file
pinning the live server to `HLS_SCHEDULER_MAX_BATCH=8`,
`HLS_SCHEDULER_FIXED_BATCH_SIZES=8`, and
`MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8`. Updating
`.env.webrtc-turn.local` to `16`, `8,16`, and `8,16` allowed the same five safe
INT8 decoder stages to warm batch `16` successfully:

- `Stagewise TRT batch=8 ready in 23.03s`
- `Stagewise TRT batch=16 ready in 129.11s`
- `Stagewise TRT warmup complete (batches=[8, 16], total=152.14s)`
- active backend: `tensorrt_stagewise_int8_mixed`
- `/stats` scheduler cap: `max_combined_batch_size=16`

No separate INT8 weights are required for batch `16`; the INT8 calibration and
selected decoder stages are shared. The batch-specific work is TensorRT
engine/profile/context warmup for the larger shape.

Validated isolated `onnx_qdq` results at batch `8`:

- `decoder_pre`: MAE `0.0021`, max abs `0.074`.
- `decoder_mid_block`: MAE `0.0011`, max abs `0.034`.
- `decoder_up_block_0`: MAE `0.0015`, max abs `0.065`.
- `decoder_up_block_1`: MAE `0.0019`, max abs `0.050`.
- `decoder_up_block_2`: MAE `0.003323`, max abs `0.083008`.

Do not promote these stages yet:

- `decoder_up_block_3`: MAE `0.019000`, max abs `0.099609`, visible
  color/texture harshness.
- `decoder_postprocess`: MAE `0.019470`, max abs `0.096191`, visible
  color/texture shift.

Latest five-stage INT8 WebRTC validation on 2026-05-26:

- command family: `load_test_webrtc.py`
- base URL: `http://127.0.0.1:8000`
- avatar: `test_avatar_2`
- audio: `data/audio/ai-assistant.mpga`
- request shape: `20/20 fps`, request `batch_size=8`
- ramp: `4,5,6,8`
- server WebRTC encoder: `h264_nvenc`
- server relay policy and local turnserver were active
- report:
  `tmp/load_tests/load_test_webrtc_3090_int8_5stage_20_20_4_5_6_8streams_batch8_relay_20260526.json`

| Streams | Completed | Avg frame interval | Approx aggregate FPS | Avg live-ready | Max frame interval |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | `4/4` | `0.069s` | `58.0` | `3.196s` | `0.310s` |
| 5 | `5/5` | `0.086s` | `58.1` | `4.213s` | `0.484s` |
| 6 | `6/6` | `0.106s` | `56.6` | `4.877s` | `0.810s` |
| 8 | `8/8` | `0.143s` | `55.9` | `6.661s` | `1.098s` |

This is better than the closest saved RTX 3090 WebRTC diagnostic reference
(`45.5-47.6` aggregate FPS), but it is not a clean FP16-vs-INT8 A/B because the
saved reference used a different avatar, `libx264`, and `8,16` buckets.

Latest five-stage INT8 WebRTC `8,16` bucket validation on 2026-05-27:

- command family: `load_test_webrtc.py`
- base URL: `http://127.0.0.1:8000`
- avatar: `test_avatar_2`, warmed into cache in `3.38s`
- audio: `data/audio/ai-assistant.mpga`
- request shape: `20/20 fps`, request `batch_size=8`
- server buckets: `HLS_SCHEDULER_MAX_BATCH=16`,
  `HLS_SCHEDULER_FIXED_BATCH_SIZES=8,16`,
  `MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8,16`
- server WebRTC encoder: `h264_nvenc`
- report:
  `tmp/load_tests/load_test_webrtc_3090_int8_5stage_20_20_4_5_6_8streams_8_16_buckets_batch8_relay_20260527.json`
- corrected C5 rerun:
  `tmp/load_tests/load_test_webrtc_3090_int8_5stage_20_20_5streams_8_16_buckets_batch8_relay_rerun_20260527.json`

| Streams | Completed | Avg frame interval | Approx aggregate FPS | Delta vs batch-8-only | Avg live-ready | Max frame interval | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | `4/4` | `0.062s` | `64.5` | `+11.3%` | `4.681s` | `0.555s` | `17873 MB` |
| 5 | `5/5` | `0.077s` | `64.9` | `+11.7%` | `3.975s` | `0.854s` | `17895 MB` |
| 6 | `6/6` | `0.092s` | `65.2` | `+15.2%` | `4.809s` | `1.111s` | `17885 MB` |
| 8 | `8/8` | `0.128s` | `62.5` | `+11.7%` | `6.058s` | `1.905s` | `17895 MB` |

The first C5 point in the combined ramp was a malformed WebRTC connection
measurement (`0/5` peers ready). The single-stage C5 rerun above completed
cleanly and is the valid C5 number. Server logs confirm the larger bucket was
actually used during the runs, for example `GPU batch timing ... actual=16
padded=16`. The cost is residency: peak VRAM rose from about `8.3 GB` in the
batch-8-only run to about `17.9 GB` with `8,16` resident.

Latest five-stage INT8 HLS validation on 2026-05-26:

- command family: `load_test.py`
- base URL: `http://127.0.0.1:8000`
- avatar: `test_avatar_2`
- audio: `data/audio/ai-assistant.mpga`
- request shape: `20/20 fps`, request `batch_size=8`
- ramp: `4,5,6,8`
- HLS encoder: `libx264`
- report:
  `tmp/load_tests/load_test_hls_3090_int8_5stage_20_20_4_5_6_8streams_batch8_test_avatar_2_20260526.json`

| Streams | Completed | Avg segment interval | Approx generated FPS | Avg live-ready | Max segment interval |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | `4/4` | `1.091s` | `73.3` | `1.895s` | `1.555s` |
| 5 | `5/5` | `1.397s` | `71.6` | `1.921s` | `2.537s` |
| 6 | `6/6` | `1.703s` | `70.5` | `2.187s` | `3.077s` |
| 8 | `8/8` | `2.278s` | `70.2` | `2.468s` | `4.610s` |

Closest saved C4/C5 HLS references were `1.188s` and `1.477s` average segment
interval, so this is a modest directional improvement. It is not a clean
FP16-vs-INT8 A/B because the live worker returned `404` for the old
`test_avatar` HLS shape and the successful run used `test_avatar_2`.

Avatar portability note from 2026-05-27:

- The live INT8 `.plan` files are stage/batch scoped, not avatar scoped.
- Current calibration data was captured from `test_avatar_2` `pred_latents`, so
  broaden the calibration corpus before treating this as fully production
  representative.
- Runtime compatibility was validated with a second prepared avatar:
  `int8_avatar_probe_ai`, created from `data/video/ai_test_default_moving_vid.mp4`.
- `/generate` request `gen_89a80670` completed successfully for that avatar and
  logs confirmed `backend=tensorrt_stagewise_int8_mixed`.
- Output:
  `results/v15/avatars/int8_avatar_probe_ai/vid_output/int8_avatar_probe_ai_smoke.mp4.mp4`.

### WebRTC Throughput Expansion Plan

Update from 2026-05-27: after the five-stage VAE decoder INT8 pass, the next
throughput work is no longer "quantize more VAE at any cost." The live WebRTC
cycle is now split between the VAE decoder and MuseTalk UNet.

Latest batch-8 WebRTC timing at the 8-stream point:

```text
avg_gpu_batch ~= 0.1217s
avg_unet      ~= 0.0554s
avg_vae       ~= 0.0650s
avg_compose   ~= 0.0612s
aggregate FPS ~= 55.9
```

Concurrency target math:

| Target | Generated FPS needed | Current gap |
| --- | ---: | ---: |
| `4 x 20 fps` | `80` | `1.43x` |
| `6 x 20 fps` | `120` | `2.15x` |
| `8 x 20 fps` | `160` | `2.86x` |

Operational conclusion:

- The current VAE INT8 path is valid and visually good enough for continued use.
- The rejected VAE stages, `decoder_up_block_3` and `decoder_postprocess`, should
  stay out of the live INT8 list.
- The next model target is the MuseTalk UNet, because it now consumes nearly as
  much of the GPU generation turn as VAE decode.
- Batch `16` is now recovered for the five safe INT8 VAE stages. The `8,16`
  server profile raised WebRTC aggregate FPS by about `11-15%`, but increased
  peak residency to about `17.9 GB`, so it is a useful throughput profile rather
  than a full concurrency solution.

Next execution order:

1. Keep the current five-stage VAE INT8 server as the visual baseline.
2. Add scheduler-side UNet capture for real `latent_batch`, `audio_feature_batch`,
   `timesteps`, and FP16 `pred_latents`.
3. Build a UNet correctness harness before any live server switch.
4. Try exact-batch FP16 TensorRT or ONNX UNet at batch `8`.
5. Only after FP16 UNet backend correctness passes, test mixed INT8/FP16 UNet.
6. Keep the validated VAE `8,16` profile in the benchmark matrix while tracking
   VRAM residency and tail frame intervals.
7. Re-run WebRTC load tests at C4/C6/C8 and require both quality and aggregate FPS
   gains before changing the default start profile.

Useful first targets:

- reduce `avg_unet` from about `0.055s` toward `0.040s` at batch `8`
- keep VAE around or below the current `0.065s` batch-8 timing
- get C4 WebRTC close to `80` aggregate FPS before claiming strict `4 x 20 fps`
- move C6 toward `120` aggregate FPS before advertising easy higher concurrency

### Plain-English Pipeline Map

For the current WebRTC path, think of each live stream as asking the GPU to make
one mouth-synced face crop per frame:

- WebRTC is the live browser video pipe. It is how the generated frames reach the
  wall without waiting for a completed MP4.
- The shared scheduler is the traffic controller. It gathers frame work from
  several sessions and sends it to the GPU in buckets like `8` or `16` frames.
- Audio features are the mouth-movement clues extracted from the audio.
- UNet is the mouth-motion predictor. It takes the current avatar latent plus
  audio features and predicts the next mouth/face latent.
- The VAE decoder is the image painter. It turns UNet's small latent tensor back
  into an RGB face crop.
- Compose is the paste step. It puts the generated face crop back into the avatar
  frame.
- Frame callback/WebRTC handoff is the delivery step. It pushes composed frames
  to the live video track.
- TensorRT is the optimized runtime. FP16 is half precision; INT8 is smaller and
  potentially faster, but only safe when calibration and visual checks pass.
- Calibration capture means saving real tensors from live traffic so a new
  backend can be compared against known-good PyTorch output.

### 2026-05-29 UNet Backend Implementation Update

The repo now has the first correctness-first UNet backend plumbing:

- `scripts/hls_gpu_scheduler.py` can capture real UNet inputs/outputs with:

```text
MUSETALK_UNET_CALIBRATION_CAPTURE=1
MUSETALK_UNET_CALIBRATION_DIR=./calibration/unet
MUSETALK_UNET_CALIBRATION_MAX_BATCHES=64
```

- Captured files are `unet_io_*.pt` and include `latent_batch`,
  `audio_feature_batch`, `timesteps`, FP16 `pred_latents`, actual/padded batch
  sizes, and avatar/request metadata.
- `scripts/validate_unet_backend.py` validates either PyTorch or TensorRT UNet
  candidates against those captures.
- `scripts/tensorrt_export.py --components unet` can now use those captures as
  TensorRT export example tensors via `--unet-capture-dir`, then run post-export
  capture validation via `--validate-unet-capture-dir`.
- `scripts/trt_runtime.py` and `scripts/avatar_manager_parallel.py` now include
  an opt-in UNet TensorRT runtime path. It stays off until explicitly enabled:

```text
MUSETALK_UNET_BACKEND=trt
MUSETALK_TRT_UNET_ENABLED=1
MUSETALK_TRT_UNET_PATH=./models/tensorrt/unet_trt.ts
MUSETALK_TRT_UNET_META_PATH=./models/tensorrt/unet_trt_meta.json
```

Operational guardrail:

- `scripts/trt_runtime.py` now refuses to silently run a requested TensorRT
  backend on CPU when `MUSETALK_TRT_FALLBACK=0`.
- On 2026-05-29, after a restart attempt for UNet capture, this worker's CUDA
  driver API returned `CUDA_ERROR_UNKNOWN` / `cuInit 999` even though
  `nvidia-smi` still saw the RTX 3090. In that state, do not run WebRTC
  throughput tests because the server cannot be the INT8 GPU baseline.
- If this appears again, verify with:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python - <<'PY'
import ctypes, torch
print("torch_cuda", torch.cuda.is_available())
print("cuInit", ctypes.CDLL("libcuda.so.1").cuInit(0))
PY
```

  A container/instance restart is the likely fix when `cuInit` returns `999`.

Next concrete run order:

1. Restart once with UNet capture enabled while keeping the current five-stage
   VAE INT8 `8,16` profile.
2. Run a short representative WebRTC wall session to write `./calibration/unet`
   captures.
3. Validate the saved PyTorch reference path:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/validate_unet_backend.py \
  --capture-dir ./calibration/unet \
  --backend pytorch \
  --limit 8 \
  --report-path tmp/unet_pytorch_reference_validation.json
```

4. Export an FP16 UNet TensorRT candidate for the same buckets:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/tensorrt_export.py \
  --components unet \
  --batch-sizes 8,16 \
  --output-dir ./models/tensorrt \
  --precision fp16 \
  --save-format exported_program \
  --unet-capture-dir ./calibration/unet \
  --validate-unet-capture-dir ./calibration/unet \
  --validate-unet-limit 8 \
  --validate-unet-report-path tmp/unet_trt_fp16_validation.json \
  --require-valid-unet
```

5. If a separate validation run is needed, validate the TensorRT candidate
   against the same captures:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/validate_unet_backend.py \
  --capture-dir ./calibration/unet \
  --backend trt \
  --trt-path ./models/tensorrt/unet_trt.ts \
  --limit 8 \
  --report-path tmp/unet_trt_fp16_validation.json
```

6. Only if validation is clean, enable the UNet TRT runtime for one lipsync smoke
   test, then WebRTC C4/C6/C8 load tests.

Historical failed `torchscript_ptq` result:
`decoder_up_block_0` and `decoder_up_block_1` crashed TensorRT PTQ calibration
with CUDA illegal memory access at batch `8`; `decoder_up_block_1` also failed
with group norm and upsample forced to PyTorch, batch `1`, and list-format
calibrator input. `decoder_mid_block` failed in pure INT8 because TensorRT still
assigned FP16 to at least one layer/output; with `fp16,int8`, calibration failed
with no scaling factors detected. `decoder_pre` and `decoder_postprocess` could
write calibration caches, but their decoded outputs were visually unusable
(`decoder_postprocess` collapsed to constant `0.5`). A ModelOpt fake-quant QDQ
prototype also had high stage error and did not export cleanly through the
current Torch-TensorRT Dynamo path. The bad experimental caches were archived under
`tmp/vae_decoder_int8_experiment_cache_snapshot/` and removed from the live
TensorRT cache directory.

### WebRTC TURN Relay On Vast

For public WebRTC wall testing on Vast.ai, prefer the TCP-mapped local TURN
listener path. The current working shape is:

```text
WEBRTC_RELAY_ENABLED=1
WEBRTC_TURN_AUTOSTART=1
TURN_LISTEN_PORT=1455
TURN_PUBLIC_TRANSPORT=tcp
TURN_PUBLIC_PORT=$VAST_TCP_PORT_1455
WEBRTC_USE_LOCAL_TURN=1
WEBRTC_ICE_TRANSPORT_POLICY=relay
```

The 2026-05-26 working public values were:

```text
API:  http://194.228.55.129:37331
TURN: turn:194.228.55.129:37187?transport=tcp
```

Those values are host-specific. On a fresh Vast boot, use the current
`PUBLIC_IPADDR` and `VAST_TCP_PORT_1455` mapping, or let
`scripts/run_webrtc_relay_api_server.sh` and
`scripts/run_turnserver_tcp_relay.sh` auto-detect them. Do not commit the local
`.env.webrtc-turn.local`; it contains credentials.

## Matching Load Tests

Activate the TRT-stagewise venv first:

```bash
cd /workspace/MuseTalk
source /workspace/.venvs/musetalk_trt_stagewise/bin/activate
```

For the stable baseline at `concurrency=8`:

```bash
python load_test.py \
  --base-url http://127.0.0.1:8000 \
  --avatar-id test_avatar \
  --audio-file ./data/audio/ai-assistant.mpga \
  --concurrency 8 \
  --segment-duration 1.0 \
  --playback-fps 30 \
  --musetalk-fps 15 \
  --batch-size 4
```

For the widened throughput branch at `concurrency=8`:

```bash
python load_test.py \
  --base-url http://127.0.0.1:8000 \
  --avatar-id test_avatar \
  --audio-file ./data/audio/ai-assistant.mpga \
  --concurrency 8 \
  --segment-duration 1.0 \
  --playback-fps 30 \
  --musetalk-fps 15 \
  --batch-size 8
```

Optional more realistic staggered arrival test:

```bash
python load_test.py \
  --base-url http://127.0.0.1:8000 \
  --avatar-id test_avatar \
  --audio-file ./data/audio/ai-assistant.mpga \
  --concurrency 8 \
  --segment-duration 1.0 \
  --playback-fps 30 \
  --musetalk-fps 15 \
  --batch-size 8 \
  --stagger-seconds 0.5
```

## Current Record

Current best average-throughput result on the widened-batch branch:

- server profile:
  - `throughput_record`
- request `batch_size=8`
- `concurrency=8`
- `avg_time_to_live_ready_s=2.197`
- `avg_segment_interval_s=1.408`
- `max_segment_interval_s=2.525`
- `wall_time_s=26.4`
- `avg_gpu_util_pct=85.21`
- `avg_gpu_memory_used_mb=23922`

Important comparison on the same server branch:

- request `batch_size=4`
- `avg_segment_interval_s=1.423`
- `max_segment_interval_s=2.046`

Practical meaning:

- request `batch_size=8` is the current best average-throughput point
- request `batch_size=4` on the same branch still has the better tail latency

## Hosted 8-Stream Validation

Latest hosted validation on May 10, 2026, using the active `throughput_record`
server on port `8000` and the widened load-test command above:

- request `batch_size=8`
- `concurrency=8`
- `segment_duration=1.0`
- `playback_fps=30`
- `musetalk_fps=15`
- `completed=8`
- `failed=0`
- `avg_time_to_live_ready_s=2.265`
- `avg_segment_interval_s=1.779`
- `max_segment_interval_s=2.546`
- `wall_time_s=33.2`
- `avg_gpu_util_pct=83.59`
- `peak_gpu_util_pct=100.0`
- `avg_gpu_memory_used_mb=23983.8`
- `peak_gpu_memory_used_mb=23984.0`

Interpretation:

- the server can host eight simultaneous HLS streams on this profile
- average segment cadence stayed under the `2.0s` practical threshold
- the strict load tester still warns because one tail interval exceeded `2.0s`
- this should be treated as a functional 8-stream hosted pass with remaining
  tail-jitter risk, not as a clean no-warning realtime pass

## 30/30 FPS Validation

Additional hosted validation on May 10, 2026 tested `musetalk_fps=30` with
`playback_fps=30`. This removes the cheaper half-rate generation path used by
the normal `15/30` stream shape.

Single stream on the temporary `4,8` profile:

- request `batch_size=8`
- `concurrency=1`
- `completed=1`
- `failed=0`
- `avg_time_to_live_ready_s=1.513`
- `avg_segment_interval_s=0.476`
- `max_segment_interval_s=0.512`
- `wall_time_s=9.2`
- `peak_gpu_util_pct=100.0`
- `peak_gpu_memory_used_mb=13856.0`

Eight simultaneous streams on the temporary `4,8` profile:

- request `batch_size=8`
- `concurrency=8`
- `completed=8`
- `failed=0`
- `avg_time_to_live_ready_s=3.270`
- `avg_segment_interval_s=3.826`
- `max_segment_interval_s=5.574`
- `wall_time_s=69.0`
- `peak_gpu_util_pct=100.0`
- `peak_gpu_memory_used_mb=13856.0`

Eight simultaneous streams on the current `8,16` throughput profile:

- request `batch_size=8`
- `concurrency=8`
- `completed=8`
- `failed=0`
- `avg_time_to_live_ready_s=5.032`
- `avg_segment_interval_s=3.567`
- `max_segment_interval_s=6.088`
- `wall_time_s=66.6`
- `peak_gpu_util_pct=100.0`
- `peak_gpu_memory_used_mb=23920.0`

Interpretation:

- `30/30` is viable for a single stream on this host
- at `concurrency=8`, both `4,8` and `8,16` throttle badly against the `2.0s`
  segment-interval threshold
- `8,16` improved average segment cadence by `0.259s` and wall time by `2.4s`
  versus `4,8`
- `8,16` worsened average live-ready by `1.762s`, worsened max segment interval
  by `0.514s`, and raised peak memory by about `10064MB`
- practical conclusion: keep `15/30` for the 8-stream target; reserve `30/30`
  for low-concurrency quality experiments

## 24/24 FPS 3-Stream Validation

Hosted validation on May 10, 2026 tested `musetalk_fps=24` with
`playback_fps=24` on the current `8,16` throughput profile:

- request `batch_size=8`
- `concurrency=3`
- `segment_duration=1.0`
- `completed=3`
- `failed=0`
- `avg_time_to_live_ready_s=1.848`
- `avg_segment_interval_s=1.060`
- `max_segment_interval_s=1.527`
- `wall_time_s=19.9`
- `avg_gpu_util_pct=82.33`
- `peak_gpu_util_pct=100.0`
- `avg_gpu_memory_used_mb=23922.0`
- `peak_gpu_memory_used_mb=23922.0`

Interpretation:

- `24/24` at three concurrent streams passed cleanly on the current `8,16`
  profile
- max segment interval stayed below the `2.0s` throttle threshold
- this is much healthier than the `30/30` 8-stream stress result, but it is not
  an 8-stream capacity claim

## 20/20 FPS 4-5 Stream Validation

Hosted validation on May 10, 2026 tested `musetalk_fps=20` with
`playback_fps=20` on the current `8,16` throughput profile.

Five simultaneous streams:

- request `batch_size=8`
- `concurrency=5`
- `segment_duration=1.0`
- `completed=5`
- `failed=0`
- `avg_time_to_live_ready_s=2.014`
- `avg_segment_interval_s=1.477`
- `max_segment_interval_s=2.550`
- `wall_time_s=27.5`
- `avg_gpu_util_pct=78.07`
- `peak_gpu_util_pct=100.0`
- `avg_gpu_memory_used_mb=23922.0`
- `peak_gpu_memory_used_mb=23922.0`

Four simultaneous streams:

- request `batch_size=8`
- `concurrency=4`
- `segment_duration=1.0`
- `completed=4`
- `failed=0`
- `avg_time_to_live_ready_s=1.889`
- `avg_segment_interval_s=1.188`
- `max_segment_interval_s=2.041`
- `wall_time_s=22.4`
- `avg_gpu_util_pct=77.04`
- `peak_gpu_util_pct=100.0`
- `avg_gpu_memory_used_mb=23922.0`
- `peak_gpu_memory_used_mb=23922.0`

Interpretation:

- both `20/20` runs completed without failed sessions
- `concurrency=5` had good average cadence but failed the strict tail threshold
  by `0.550s`
- `concurrency=4` was much closer, exceeding the `2.0s` threshold by only
  `0.041s`
- practical conclusion: `20/20` looks usable around four streams on this host,
  but it is still not a clean no-warning profile under burst-start load

## Notes

- the launcher uses the exact venv python path instead of relying on whichever
  `python` is active in the shell
- this avoids the mixed-venv state seen earlier where `VIRTUAL_ENV` and the
  actual interpreter did not match
- if startup logs show values like `compose_workers=10` or `encode_workers=10`
  instead of the documented baseline worker counts, a shell env override is
  still active and the launcher is preserving it
- `scripts/vast_server_ctl.sh` starts the server in the background with `nohup`
  and writes logs under `/workspace/logs/musetalk`
- the Vast boot/control wrappers now default to `throughput_record` when
  `PROFILE` is unset; the direct foreground launcher still defaults to
  `baseline`
- normal on-start does not currently prove public WebRTC reachability by itself.
  The boot path is `scripts/vast_onstart.sh` -> `scripts/vast_server_ctl.sh start`;
  without relay env, it starts `scripts/run_trt_stagewise_server.sh` with
  STUN/direct ICE. The saved successful WebRTC load-test reports used
  `ice_transport_policy=all`, so those runs relied on direct/local ICE working,
  not on a hidden auto-started TURN server.
- the external Vast bootstrap script reclones `origin/main` into
  `/workspace/MuseTalk` before calling `scripts/vast_onstart.sh`; local
  uncommitted startup fixes are discarded on a fresh boot unless they are pushed.
  That bootstrap currently passes setup/profile/port settings, not WebRTC
  relay/TURN settings.
- for public WebRTC wall playback, preserve the relay launch path during
  restarts. Copy `.env.webrtc-turn.local.example` to `.env.webrtc-turn.local`,
  fill the current TURN public mapping and password, then restart with:

```bash
cd /workspace/MuseTalk
PROFILE=throughput_record bash scripts/vast_server_ctl.sh restart
```

  The ignored env file can set `WEBRTC_RELAY_ENABLED=1` so the control helper
  launches `scripts/run_webrtc_relay_api_server.sh`; with
  `WEBRTC_TURN_AUTOSTART=1`, it also starts local coturn through
  `scripts/run_turnserver_tcp_relay.sh` before the API. If using managed TURN,
  leave autostart off and set `WEBRTC_TURN_URLS`,
  `WEBRTC_SERVER_TURN_URLS`, `WEBRTC_TURN_USER`, and `WEBRTC_TURN_PASS`.
- to watch live logs on Vast:

```bash
tail -f /workspace/logs/musetalk/api_server_8000.log
```

- to run in the foreground for debugging:

```bash
cd /workspace/MuseTalk
bash scripts/vast_server_ctl.sh stop
PROFILE=baseline PORT=8000 bash scripts/run_trt_stagewise_server.sh
```

- NVENC is still not the default because raw ffmpeg NVENC session open fails on
  the current host/runtime; `libx264` remains the reliable baseline
- the avatar-prep path still prints several upstream warnings today, including
  `torch.load(..., weights_only=False)` and some MMDetection/MMEngine
  deprecation warnings; those warnings were observed during a successful
  end-to-end avatar preparation and are not currently treated as blockers
- if a failed avatar-prep attempt leaves partial files on disk, retry with
  `force_recreate=true`

## Related Docs

- [`current_start_param_reference.md`](./current_start_param_reference.md)
- [`current_tensorrt_environment_plan.md`](./current_tensorrt_environment_plan.md)
- [`current_model_backend_findings.md`](./current_model_backend_findings.md)
- [`docs/vast_ai_boot.md`](./docs/vast_ai_boot.md)
- [`runbook_hls_load_test.md`](./runbook_hls_load_test.md)
