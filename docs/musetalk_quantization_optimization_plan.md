# MuseTalk Quantization And Backend Optimization Plan

Date: 2026-05-25

Latest WebRTC load-test synthesis:
`docs/webrtc_load_test_findings_2026-06-07.md` summarizes the RTX 5000 Ada
FP16-vs-INT8 runs, the `4,8,16` bucket follow-up, and the missing TRT UNet
artifact needed for the next RTX 5000 Ada optimized-path test. The 2026-06-08
RTX 6000 Ada follow-up now has both a five-stage VAE INT8 + PyTorch UNet
baseline and a validated five-stage VAE INT8 + TRT UNet split8 rerun.

## Goal

Improve MuseTalk throughput by applying quantization and backend acceleration to
the parts of the pipeline where lower precision neural inference can actually
reduce end-to-end generation time.

The plan is intentionally staged. The safest path is not to quantize every model
at once. The right path is:

1. Measure the current end-to-end baseline.
2. Quantize the VAE decoder first.
3. Validate visual quality.
4. Quantize or compile the UNet second.
5. Only target Whisper if audio preparation is a measured bottleneck.
6. Keep composition and encoding optimization separate from model quantization.

## Current Starting Point

Based on the current docs and runtime notes:

- The app already uses FP16 for important model paths.
- The working TensorRT path is focused on VAE decode.
- The preferred VAE acceleration path is stagewise TensorRT, not one monolithic
  exported VAE graph.
- Existing benchmarks suggest TensorRT VAE decode gives roughly a 20 percent
  gain over PyTorch FP16 in the tested setup.
- That gain is useful, but insufficient by itself for a large end-to-end
  throughput jump because the remaining cost is split across UNet, VAE decode,
  composition, encoding, memory movement, and scheduler overhead.

## 2026-05-26 Implementation Update

The VAE quantization branch should now be implemented against the repaired
runtime backend, not the older serialized monolithic TensorRT artifact path.

Current repo-level decision:

- Keep `MUSETALK_VAE_BACKEND=trt_stagewise` as the VAE acceleration base.
- Keep the monolithic TensorRT VAE artifacts marked performance-only /
  untrusted because they produced gray collapsed face output.
- Add calibration capture at the live scheduler boundary immediately after
  UNet produces `pred_latents` and before VAE decode.
- Use real captured `pred_latents` for INT8 calibration. Cached avatar latents
  remain useful for smoke validation, but they are not sufficient calibration
  data by themselves.
- Add mixed INT8/FP16 as an opt-in stagewise precision policy. The default
  production path remains FP16 stagewise TensorRT.
- Keep normalization, output postprocess, and any fragile stages in FP16 or
  PyTorch until direct decode and blended-video validation proves otherwise.
- Require exact scheduler bucket warmup for any quantized stagewise batch size.
  Scheduler fixed buckets must stay aligned with warmed backend buckets.

Current target runtime switches:

```text
MUSETALK_VAE_BACKEND=trt_stagewise
MUSETALK_TRT_STAGEWISE_PRECISION=fp16

MUSETALK_VAE_BACKEND=trt_stagewise
MUSETALK_TRT_STAGEWISE_PRECISION=int8_mixed
MUSETALK_TRT_STAGEWISE_INT8_STAGES=decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2
MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_DIR=./calibration/vae_decoder
MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_ALGO=minmax
MUSETALK_TRT_STAGEWISE_INT8_CACHE_DIR=./models/tensorrt/stagewise_int8_onnx_qdq_cache
```

Current calibration capture switches:

```text
MUSETALK_VAE_CALIBRATION_CAPTURE=1
MUSETALK_VAE_CALIBRATION_DIR=./calibration/vae_decoder
MUSETALK_VAE_CALIBRATION_MAX_BATCHES=128
```

Implementation correction from the local prototype:

- ModelOpt Q/DQ calibration works in the compatible pinned package line.
- The initial Torch-TensorRT TorchScript PTQ path is no longer the live INT8
  path. It hit no-scaling-factor failures, CUDA illegal memory access, or bad
  decoded output quality.
- The implemented runtime INT8 experiment uses ModelOpt Q/DQ ONNX export plus
  TensorRT Python `.plan` engines for selected stagewise decoder modules. It
  still uses the real captured `pred_latents` corpus for calibration.
- ONNX/QDQ `.onnx` and `.plan` artifacts are written per stage and exact batch
  size so repeat warmups can reuse the built TensorRT engines when enabled.

Important optional dependency correction from the local dry-run:

- Do not install latest unpinned `nvidia-modelopt` into the serving venv.
- Latest `nvidia-modelopt` currently wants a newer Torch/CUDA family than the
  pinned MuseTalk TRT stack.
- Use the setup script's pinned `nvidia-modelopt==0.23.2` default for this
  `torch==2.5.1+cu121` environment for offline QDQ experiments only, unless a
  newer compatible version is tested in a separate venv first.

The first safe experiment is:

1. Capture representative VAE decoder latents from the FP16 stagewise server.
2. Validate FP16 stagewise output against PyTorch on those captured latents.
3. Quantize only one or two conv-heavy up-block stages.
4. Validate direct decoder output and blended video.
5. Expand INT8 stage coverage only if visual quality remains stable.

## 2026-05-26 Live INT8 Experiment Context

Current user goal: experiment with VAE decoder INT8 first because VAE decode is
the active throughput bottleneck. The base/reference video work is already in
place, including `test_avatar_2` prepared from `chatgpt_moving_vid.mp4` and a
saved WebRTC talking reference under `docs/reference_videos/`.

Current live service state:

- The API service has been restarted in INT8 mixed mode for the VAE decoder
  experiment.
- Current intended live precision is
  `MUSETALK_TRT_STAGEWISE_PRECISION=int8_mixed` with
  `MUSETALK_TRT_STAGEWISE_INT8_FRONTEND=onnx_qdq`.
- Current live INT8 stages are
  `decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2`.
- The scheduler is intentionally capped to batch `8` while batch `16` remains a
  separate TensorRT context-creation problem.
- The working public WebRTC wall uses TURN-over-TCP relay when TURN env is
  available. That fixed the black screen/ICE issue but is unrelated to INT8
  decoder correctness.

Calibration data state:

- `calibration/vae_decoder` contains 37 captured `.pt` files from real
  `pred_latents`.
- The corpus includes mostly batch-16 tensors plus one batch-8 tensor.
- Verified tensor shape is `(B, 4, 32, 32)` with dtype `torch.float16`.
- These are representative enough for the first build experiment, but more
  audio diversity is still recommended before trusting visual quality broadly.

Representative audio guidance:

- Do use real utterances with varied mouth movement: quiet/loud, fast/slow,
  vowels, closed-mouth consonants, wide-mouth phonemes, short pauses, and at
  least a few seconds of continuous speech.
- The point of running FP16 first is to capture the real UNet output distribution
  that the VAE decoder will see. INT8 calibration learns activation ranges from
  those tensors; random latents or cached avatar latents are not enough.
- The count matters because more batches cover more activation range. Eight
  captured batches is a small first build set, not a final quality corpus.

INT8 failure history:

1. Initial INT8 stagewise run failed because Torch-TensorRT could not freeze
   `Int64`/`Float64` constants. The runtime now passes
   `truncate_long_and_double=True`.
2. The next TorchScript PTQ INT8 run failed with TensorRT reporting calibration completed with
   no scaling factors. That means the selected stage/build did not produce usable
   INT8 calibration scales; it is not a WebRTC failure.
3. TorchScript PTQ also hit CUDA illegal memory access on the VAE up-blocks.
4. The working fix is a different frontend: ModelOpt Q/DQ ONNX export, followed
   by a TensorRT Python engine that executes from PyTorch tensor data pointers.
   This avoids the TorchScript PTQ calibrator crash path.

Current runtime changes for the next experiment:

- Selected INT8 stages can use `MUSETALK_TRT_STAGEWISE_INT8_MIN_BLOCK_SIZE=1`
  so TensorRT does not discard small convertible partitions.
- Selected INT8 stages default to
  `MUSETALK_TRT_STAGEWISE_INT8_ENABLED_PRECISIONS=int8` so the experiment proves
  an INT8 path instead of silently choosing FP16 kernels.
- `MUSETALK_TRT_STAGEWISE_INT8_REQUIRE_FULL_COMPILATION=1` can be used as a
  stricter diagnostic if a stage silently falls back.
- `MUSETALK_TRT_STAGEWISE_INT8_FRONTEND=onnx_qdq` is the current working INT8
  frontend. `torchscript_ptq` is retained only as a diagnostic fallback.
- The runtime now fails loudly when PTQ returns without writing a non-empty
  calibration cache, unless
  `MUSETALK_TRT_STAGEWISE_INT8_REQUIRE_CALIBRATION_CACHE=0` is set.
- `MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_FORMAT=list` is available only as a
  calibrator wiring diagnostic; default remains `tensor`.
- `scripts/experiment_vae_decoder_int8.py` now writes `report.json` even when
  TensorRT build/calibration fails, so each failed stage probe leaves structured
  settings, error text, and traceback instead of only terminal logs.
- `MUSETALK_TRT_STAGEWISE_PRECISION=int8_mixed` now requires an explicit
  `MUSETALK_TRT_STAGEWISE_INT8_STAGES` list. There is no safe implicit default
  stage set yet.
- The live-serving guard applies to the old `torchscript_ptq` frontend. The
  `onnx_qdq` frontend is allowed in the API after direct image comparison.

The diagnostic entry point used for these tests is:

```bash
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

Change `--stage` for each candidate. The example above is a diagnostic command,
not a known-good live-server configuration.

Run this only after stopping the live FP16 server or on a separate GPU. The live
server occupies enough VRAM that even a tiny isolated Torch-TensorRT PTQ probe
can OOM while it is running. If the experiment fails, restart the FP16 server and
keep the WebRTC validation path stable.

Actual isolated results from 2026-05-26 after stopping the API:

| INT8 stage | Precision setting | Result | Quality |
| --- | --- | --- | --- |
| `decoder_pre` | `onnx_qdq`, `int8` | Built and ran through TensorRT Q/DQ engine | passed, MAE `0.0021`, max abs `0.074` |
| `decoder_mid_block` | `onnx_qdq`, `int8` | Built and ran through TensorRT Q/DQ engine | passed, MAE `0.0011`, max abs `0.034` |
| `decoder_up_block_0` | `onnx_qdq`, `int8` | Built and ran through TensorRT Q/DQ engine | passed, MAE `0.0015`, max abs `0.065` |
| `decoder_up_block_1` | `onnx_qdq`, `int8` | Built and ran through TensorRT Q/DQ engine | passed, MAE `0.0019`, max abs `0.050` |
| `decoder_mid_block` | `int8` | TensorRT build failed because FP16 was assigned to at least one layer/output while FP16 was not enabled | failed build |
| `decoder_mid_block` | `fp16,int8` | TensorRT calibration failed with no scaling factors detected | failed build |
| `decoder_up_block_1` | `int8` | TensorRT calibration hit CUDA illegal memory access; no cache written | failed build |
| `decoder_up_block_1` | `fp16,int8` | Same CUDA illegal memory access; no cache written | failed build |
| `decoder_up_block_1` | `int8`, `group_norm` kept on PyTorch | Same CUDA illegal memory access; no cache written | failed build |
| `decoder_up_block_1` | `int8`, `group_norm` + `upsample_nearest2d` kept on PyTorch | Same CUDA illegal memory access; no cache written | failed build |
| `decoder_up_block_1` | `int8`, batch `1` | Same CUDA illegal memory access; no cache written | failed build |
| `decoder_up_block_1` | `int8`, calibrator returns `[tensor]` instead of `tensor` | Same CUDA illegal memory access; no cache written | failed build |
| `decoder_up_block_0` | `int8` | Same CUDA illegal memory access; no cache written | failed build |
| `decoder_pre` | `int8` | Built in `91.30s`; cache size `574` bytes | bad, MAE `0.213`, output range compressed |
| `decoder_postprocess` | `int8` | Failed because the stage required FP16 but FP16 was not enabled | failed build |
| `decoder_postprocess` | `fp16,int8` | Built in `163.06s`; cache size `1133` bytes | bad, constant `0.5` output, MAE `0.210` |
| `decoder_up_block_1` | ModelOpt fake-quant QDQ prototype | Fake-quant stage already had very high error; Torch-TensorRT Dynamo export failed on ModelOpt quantized modules | not viable in current venv |

Conclusion: the Torch-TensorRT TorchScript PTQ path can invoke the calibrator
and write caches, but it is not ready to enable in the API. The ModelOpt
ONNX/QDQ path is the working path for VAE INT8 on this stack. As of this update,
the live API has booted successfully with
`MUSETALK_TRT_STAGEWISE_INT8_STAGES=decoder_up_block_0,decoder_up_block_1`,
`MUSETALK_TRT_STAGEWISE_INT8_FRONTEND=onnx_qdq`, batch-8 warmup, and the
scheduler capped to batch `8`.

Root cause assessment: this is not a WebRTC problem, not missing calibration
data, and not just a need for more warmup runs. The captured calibration tensors
load with the right shape and dtype, and the calibrator path can write caches
for simpler stages. The failure boundary is TensorRT PTQ calibration/build for
Diffusers VAE decoder stages in the current `torch==2.5.1+cu121`,
Torch-TensorRT `2.5.0`, TensorRT `10.3` stack. Keeping individual ops on
PyTorch and changing calibrator input shape did not move the failure, which
points to the frontend/build path rather than the WebRTC relay or the scheduler.

The bad experimental cache files were copied to
`tmp/vae_decoder_int8_experiment_cache_snapshot/` and removed from the live
`models/tensorrt/stagewise_int8_calibration_cache/` directory so a later server
run cannot reuse them accidentally.

Recommended next INT8 direction:

1. Keep `onnx_qdq` as the API INT8 frontend.
2. Measure WebRTC/HLS throughput with the promoted five-stage INT8 set at
   batch `8`.
3. Use the now-validated `8,16` WebRTC serving profile for higher-throughput
   experiments, but track the VRAM cost separately from model speedup. The
   batch-16 context does not need separate INT8 weights; it needs the live env
   to warm the batch-16 TensorRT shape.
4. Test `decoder_up_block_2` and `decoder_up_block_3` one at a time; only add
   them to the API if direct image comparison remains within tolerance.
5. Keep `torchscript_ptq` behind the live-serving guard because that frontend is
   the source of the CUDA illegal memory access.

## 2026-05-26 INT8 Smoke And Load-Test Results

Chat context captured in this update:

- The user first asked for a VAE decoder quantization plan after a base
  reference video had already been created.
- The relevant plan and pipeline docs were reviewed and updated around a
  VAE-first strategy.
- The first INT8 attempts failed under Torch-TensorRT TorchScript PTQ, including
  no scaling factors, CUDA illegal memory access, and bad output quality on
  simpler stages.
- The implementation was redirected to ModelOpt Q/DQ ONNX export plus TensorRT
  Python engines, which is the current working INT8 path.
- `test_avatar_2` was prepared from `chatgpt_moving_vid.mp4`.
- WebRTC black-screen investigation showed a separate ICE/TURN/runtime issue,
  not an INT8 decoder issue. The startup script path can run without manually
  starting a TURN server when the relay env exists.
- A simple lipsync smoke video using `data/audio/yongen.wav` was generated at:
  `results/v15/avatars/test_avatar_2/vid_output/int8_lipsync_smoke_test.mp4`.
- The user visually reviewed the INT8 smoke output and reported that lipsync
  quality looked very good, with audio still to be heard/reviewed separately.
- `ffprobe` confirms the smoke MP4 contains both tracks: H.264 video
  (`7.96s`) and AAC audio (`8.00s`).

Initial two-stage live INT8 environment for the smoke test. Later sections
supersede this for the five-stage load-test server:

```text
MUSETALK_VAE_BACKEND=trt_stagewise
MUSETALK_TRT_STAGEWISE_PRECISION=int8_mixed
MUSETALK_TRT_STAGEWISE_INT8_FRONTEND=onnx_qdq
MUSETALK_TRT_STAGEWISE_INT8_STAGES=decoder_up_block_0,decoder_up_block_1
MUSETALK_TRT_FALLBACK=0
HLS_SCHEDULER_MAX_BATCH=8
HLS_SCHEDULER_FIXED_BATCH_SIZES=8
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8
```

The exact smoke-test request was confirmed by the API log:

```text
VAE decode timing backend=tensorrt_stagewise_int8_mixed calls=25 avg_tensor=0.0856s avg_post=0.0035s avg_total=0.0891s max_total=0.0945s
```

Load-test setup:

- Avatar: `test_avatar_2`
- Source avatar video: `data/video/chatgpt_moving_vid.mp4`
- Audio: `data/audio/yongen.wav`
- Audio duration: about `8s`
- Request path: `/generate`
- Batch cap: `8`
- Bench artifacts:
  - `tmp/load_tests/int8_onnx_qdq_generate_bench.json`
  - `tmp/load_tests/fp16_stagewise_generate_bench.json`
  - `tmp/load_tests/int8_onnx_qdq_generate_c4_bench.json`
  - `tmp/load_tests/fp16_stagewise_generate_c4_bench.json`

Measured result:

| Test | FP16 stagewise | INT8 ONNX/QDQ mixed | Result |
| --- | ---: | ---: | --- |
| VAE decode avg total, batch 8 | `0.0988s` | `0.0883s` | INT8 `~10.6%` lower, `~1.12x` faster |
| Sequential `/generate`, 3 runs | `14.408s` avg | `14.100s` avg | INT8 `~2.1%` faster |
| Concurrent `/generate`, 4 jobs | `55.400s` stage wall | `57.322s` stage wall | INT8 `~3.5%` slower |
| Concurrent jobs/min | `4.332` | `4.187` | INT8 `~3.3%` lower |

Conclusion:

- INT8 is genuinely running for the selected VAE decoder stages.
- The VAE decoder itself is faster on the batch-8 server, and the later `8,16`
  profile adds another measured WebRTC throughput gain.
- The end-to-end speedup is still moderate rather than massive because this is a
  hybrid VAE-only INT8 path and the UNet/compose loop remains expensive.
- The next throughput work should focus on UNet backend acceleration,
  VAE/stage-boundary overhead, and composition timing under WebRTC load.

Why the end-to-end speedup is small:

- This is a hybrid experiment, not full-model INT8.
- The live API now uses five safe INT8 VAE decoder stages, but
  `decoder_up_block_3` and `decoder_postprocess` remain FP16 because direct
  comparison showed visible regressions.
- The MuseTalk UNet is still not quantized, and it is still a major per-frame
  generation cost.
- The VAE encoder is not the right live-throughput target for cached avatars;
  it runs during avatar preparation, not every streamed/generated frame.
- The measured INT8 saving was about `0.0105s` per VAE decode call. On the
  8-second smoke audio with about `25` decode calls, that is only about
  `0.26s` of possible request-level saving before scheduler and media overhead.
- The recovered `8,16` bucket profile improves WebRTC aggregate FPS by about
  `11-15%`, but TensorRT ONNX/QDQ stage bridge, CPU composition, and media
  handoff still hide a large part of the decoder-only gain.

Updated quantization direction:

1. Keep the current WebRTC/API path on the validated five-stage INT8 decoder:
   `decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2`.
2. Keep `decoder_up_block_3` and `decoder_postprocess` out of the live INT8
   list unless a new quantization method passes direct visual comparison.
3. Use the recovered `8,16` batch profile for throughput experiments where
   `~17.9 GB` resident VRAM is acceptable.
4. Expand the calibration corpus across multiple avatars and speech patterns
   before calling the current VAE INT8 path production-representative.
5. Start the second major branch: stable UNet backend acceleration first, then
   UNet mixed INT8/FP16 calibration. That is the likely path to a larger
   end-to-end speedup than VAE decoder-only quantization.

## 2026-05-26 Expanded VAE Decoder INT8 Results

The VAE decoder INT8 coverage was expanded and tested one stage at a time with
the `onnx_qdq` frontend at batch `8`.

One-stage isolated results:

| Stage | MAE vs PyTorch FP16 | Max abs | Decision |
| --- | ---: | ---: | --- |
| `decoder_pre` | `0.002805` | `0.092041` | promote |
| `decoder_mid_block` | `0.001287` | `0.043945` | promote |
| `decoder_up_block_2` | `0.003323` | `0.083008` | promote |
| `decoder_up_block_3` | `0.019000` | `0.099609` | do not promote yet |
| `decoder_postprocess` | `0.019470` | `0.096191` | do not promote |

Visual review:

- `decoder_pre`, `decoder_mid_block`, and `decoder_up_block_2` looked close to
  the PyTorch reference in the saved comparison crops.
- `decoder_up_block_3` introduced visible color/texture harshness.
- `decoder_postprocess` introduced the clearest visible color/texture shift and
  should remain FP16/PyTorch for now.

Cumulative tests:

| Stage set | MAE | Max abs | Decision |
| --- | ---: | ---: | --- |
| `decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2` | `0.005037` | `0.128662` | promoted live |
| same set plus `decoder_up_block_3` | `0.019373` | `0.151855` | rejected for now |

The live API was restarted with the promoted five-stage set:

```text
MUSETALK_TRT_STAGEWISE_PRECISION=int8_mixed
MUSETALK_TRT_STAGEWISE_INT8_FRONTEND=onnx_qdq
MUSETALK_TRT_STAGEWISE_INT8_STAGES=decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2
HLS_SCHEDULER_MAX_BATCH=8
HLS_SCHEDULER_FIXED_BATCH_SIZES=8
WEBRTC_ICE_TRANSPORT_POLICY=relay
```

Expanded five-stage live timing:

| Runtime | VAE decode avg total | Sequential `/generate` avg | C4 stage wall | C4 jobs/min |
| --- | ---: | ---: | ---: | ---: |
| FP16 stagewise | `0.0988s` | `14.408s` | `55.400s` | `4.332` |
| INT8 two-stage | `0.0883s` | `14.100s` | `57.322s` | `4.187` |
| INT8 five-stage | `0.0765s` | `14.099s` | `56.329s` | `4.261` |

WebRTC load test on the live five-stage INT8 server:

- Date: 2026-05-26.
- Hardware: RTX 3090-class 24 GB node.
- Server precision: `MUSETALK_TRT_STAGEWISE_PRECISION=int8_mixed`.
- INT8 frontend: `onnx_qdq`.
- INT8 stages:
  `decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2`.
- Scheduler buckets: batch `8` only.
- WebRTC shape: `playback_fps=20`, `musetalk_fps=20`, request
  `batch_size=8`.
- Avatar: `test_avatar_2`.
- Audio: `data/audio/ai-assistant.mpga`, about `17.76s`.
- Server WebRTC encoder: `h264_nvenc`.
- TURN relay service was running and the API was launched with relay policy,
  but the automated load client connected locally to `127.0.0.1`; treat this
  as a server-side WebRTC load test, not a full public-browser TURN test.
- Report:
  `tmp/load_tests/load_test_webrtc_3090_int8_5stage_20_20_4_5_6_8streams_batch8_relay_20260526.json`.
- Detailed report:
  `tmp/load_tests/load_test_webrtc_3090_int8_5stage_20_20_4_5_6_8streams_batch8_relay_20260526_detailed.json`.

| Streams | Completed | Avg frame interval | Approx per-stream FPS | Approx aggregate FPS | Avg live-ready | Max frame interval | Avg GPU util | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | `4/4` | `0.069s` | `14.5` | `58.0` | `3.196s` | `0.310s` | `50.06%` | `8303 MB` |
| 5 | `5/5` | `0.086s` | `11.6` | `58.1` | `4.213s` | `0.484s` | `51.48%` | `8313 MB` |
| 6 | `6/6` | `0.106s` | `9.4` | `56.6` | `4.877s` | `0.810s` | `52.64%` | `8315 MB` |
| 8 | `8/8` | `0.143s` | `7.0` | `55.9` | `6.661s` | `1.098s` | `57.55%` | `8315 MB` |

Comparison caveat:

- The closest saved RTX 3090 WebRTC diagnostic reference was
  `load_test_webrtc_report_20_20_4_5_6_8streams_8_16_diagnostics_libx264.json`.
  It used `20/20`, request `batch_size=8`, and completed the same
  `4,5,6,8` ramp at about `45.5-47.6` aggregate FPS.
- The current five-stage INT8 run reached about `55.9-58.1` aggregate FPS,
  roughly `22-28%` higher than that saved 3090 diagnostic reference.
- Do not treat the full delta as pure INT8 speedup yet. The current run uses
  `test_avatar_2`, `h264_nvenc`, batch-8-only warmup, and a same-host WebRTC
  load client, while the older reference used `test_avatar`, `libx264`, and an
  `8,16` profile.
- A clean FP16-vs-INT8 WebRTC A/B still needs the same avatar, same encoder,
  same batch buckets, same TURN/local path, and back-to-back restarts.

Server-side scheduler logs during the 8-stream stage showed the model path was
still the limiting loop:

| Metric from last 8 WebRTC streams | Average |
| --- | ---: |
| `avg_gpu_batch` | `0.1217s` |
| `avg_unet` | `0.0554s` |
| `avg_vae` | `0.0650s` |
| `avg_compose` | `0.0612s` |
| `avg_callback` | `0.0051s` |

### 2026-05-27 INT8 `8,16` WebRTC Load Test

The live server was restarted with the same five safe ONNX/QDQ decoder stages,
but with batch `16` enabled in the relay/startup env:

- `HLS_SCHEDULER_MAX_BATCH=16`
- `HLS_SCHEDULER_FIXED_BATCH_SIZES=8,16`
- `MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8,16`
- active backend: `tensorrt_stagewise_int8_mixed`
- startup proof: batch `8` ready in `23.03s`, batch `16` ready in `129.11s`,
  total stagewise warmup `152.14s`
- scheduler proof: `/stats` reported `max_combined_batch_size=16`

No new INT8 calibration set or avatar-specific INT8 weights were required.
Batch `16` reuses the existing calibration/scales and selected INT8 stages; the
additional requirement is a TensorRT engine/profile/context that accepts the
larger batch shape.

Load-test shape:

- command family: `load_test_webrtc.py`
- avatar: `test_avatar_2`, warmed into cache before the run
- audio: `data/audio/ai-assistant.mpga`
- WebRTC request: `playback_fps=20`, `musetalk_fps=20`, request `batch_size=8`
- server WebRTC encoder: `h264_nvenc`
- same-host WebRTC client against `http://127.0.0.1:8000`
- report:
  `tmp/load_tests/load_test_webrtc_3090_int8_5stage_20_20_4_5_6_8streams_8_16_buckets_batch8_relay_20260527.json`
- corrected C5 rerun:
  `tmp/load_tests/load_test_webrtc_3090_int8_5stage_20_20_5streams_8_16_buckets_batch8_relay_rerun_20260527.json`

Results compared with the previous batch-8-only INT8 WebRTC run:

| Streams | Batch-8 agg FPS | `8,16` agg FPS | Delta | `8,16` avg frame interval | Avg live-ready | Max frame interval | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | `58.0` | `64.5` | `+11.3%` | `0.062s` | `4.681s` | `0.555s` | `17873 MB` |
| 5 | `58.1` | `64.9` | `+11.7%` | `0.077s` | `3.975s` | `0.854s` | `17895 MB` |
| 6 | `56.6` | `65.2` | `+15.2%` | `0.092s` | `4.809s` | `1.111s` | `17885 MB` |
| 8 | `55.9` | `62.5` | `+11.7%` | `0.128s` | `6.058s` | `1.905s` | `17895 MB` |

The first C5 row in the multi-stage ramp was invalid because no peer
connections became ready before the load harness fired; the single-stage C5
rerun is the value in the table. Server logs confirmed real batch-16 turns,
for example `GPU batch timing ... actual=16 padded=16`.

Operational read:

- `8,16` improves aggregate WebRTC FPS by about `11-15%` on this RTX 3090.
- Peak resident VRAM rises from the batch-8-only `~8.3 GB` test footprint to
  `~17.9 GB`.
- The larger bucket still does not meet strict `20 fps` per stream beyond low
  concurrency: C4 would need `80` aggregate FPS, and C8 would need `160`.
- Latest scheduler timing during batch-16 turns is roughly
  `avg_gpu_batch=0.20-0.21s`, `avg_unet=0.07-0.08s`,
  `avg_vae=0.128-0.129s`, and `avg_compose=0.064-0.073s`, so the next large
  throughput gain still needs UNet/backend work in addition to VAE batching.

### 2026-06-07 RTX 5000 Ada INT8 `8,16` WebRTC Load Test

The previous RTX 5000 Ada WebRTC ramp on this node was accidentally run with
the FP16 stagewise backend, not INT8. The server was restarted with the safe
five-stage ONNX/QDQ mixed INT8 decoder and the same `8,16` scheduler buckets:

- GPU: `NVIDIA RTX 5000 Ada Generation`, about `32 GB` VRAM visible.
- active backend: `tensorrt_stagewise_int8_mixed`
- INT8 stages:
  `decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2`
- scheduler proof: `/stats` reported `max_combined_batch_size=16`.
- startup proof: batch `8` ready in `157.66s`, batch `16` ready in `88.86s`,
  total stagewise warmup `246.53s`.

Load-test shape:

- command family: `load_test_webrtc.py`
- avatar: `test_avatar_2`
- audio: `data/audio/ai-assistant.mpga`
- WebRTC request: `playback_fps=20`, `musetalk_fps=20`, request `batch_size=8`
- same-host WebRTC client against `http://127.0.0.1:8000`
- FP16 report:
  `tmp/load_tests/load_test_webrtc_rtx5000ada_20_20_1_2_3_4_5_6_8streams_8_16_20260607.json`
- INT8 report:
  `tmp/load_tests/load_test_webrtc_rtx5000ada_int8_20_20_1_2_3_4_5_6_8streams_8_16_20260607.json`
- INT8 detailed report:
  `tmp/load_tests/load_test_webrtc_rtx5000ada_int8_20_20_1_2_3_4_5_6_8streams_8_16_20260607_detailed.json`

Result:

| Streams | FP16 agg FPS | INT8 agg FPS | INT8 avg interval | INT8 strict video stalls | INT8 peak VRAM | Read |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `19.6` | `19.6` | `0.051s` | `0` | `17636 MB` | clean |
| 2 | `39.2` | `39.2` | `0.051s` | `0` | `17636 MB` | clean |
| 3 | `58.8` | `58.8` | `0.051s` | `0` | `17646 MB` | clean |
| 4 | `61.5` | `71.4` | `0.056s` | `44` | `17646 MB` | higher throughput, not strict 20 fps |
| 5 | `61.0` | `70.4` | `0.071s` | `115` | `17668 MB` | saturated |
| 6 | `61.2` | `71.4` | `0.084s` | `144` | `17668 MB` | saturated |
| 8 | `60.6` | `71.4` | `0.112s` | `205` | `17668 MB` | saturated |

Operational read:

- INT8 raises the RTX 5000 Ada aggregate WebRTC plateau from about `60-61 fps`
  to about `70-71 fps`, roughly a `15-17%` gain at saturated concurrency.
- Strict smooth `20 fps` capacity remains `3` concurrent WebRTC streams because
  C4 records video stalls and an average interval above the `0.050s` target.
- INT8 materially reduces resident VRAM versus the FP16 run:
  FP16 peaked around `24.4 GB`, while INT8 peaked around `17.7 GB`.
- The remaining cap is still the shared generation loop rather than WebRTC
  signaling: VAE INT8 helps, but the UNet, stage handoff, composition, and
  playout pacing still prevent C4 from reaching the required `80` aggregate fps.

Follow-up `4,8,16` bucket warmup on the same RTX 5000 Ada node:

- Reason: the `8,16` INT8 run had much lower VRAM than FP16, so there was room to
  test whether an additional warmed batch-4 engine reduced padding/tail latency.
- Active backend: `tensorrt_stagewise_int8_mixed`.
- VAE postprocess: `MUSETALK_VAE_FAST_POSTPROCESS` defaults on in
  `musetalk/models/vae.py` and was not disabled for this run.
- UNet backend: PyTorch UNet. This was not the prior `tensorrt_unet_multi`
  split8 path because the batch-8 UNet TRT artifact is not present on this node.
- Scheduler buckets: `4,8,16`, `max_combined_batch_size=16`.
- Startup proof: batch `4` ready in `114.81s`, batch `8` ready in `21.25s`,
  batch `16` ready in `28.17s`, total stagewise warmup `164.23s`.
- Report:
  `tmp/load_tests/load_test_webrtc_rtx5000ada_int8_20_20_1_2_3_4_5_6_8streams_4_8_16_20260607.json`
- Detailed report:
  `tmp/load_tests/load_test_webrtc_rtx5000ada_int8_20_20_1_2_3_4_5_6_8streams_4_8_16_20260607_detailed.json`

| Streams | Completed | Agg FPS | Avg interval | Max interval | Strict video stalls | Peak VRAM | Read |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `1/1` | `19.6` | `0.051s` | `0.094s` | `0` | `19752 MB` | clean |
| 2 | `2/2` | `39.2` | `0.051s` | `0.118s` | `0` | `19752 MB` | strict-clean |
| 3 | `3/3` | `58.8` | `0.051s` | `0.089s` | `0` | `19762 MB` | clean |
| 4 | `4/4` | `71.4` | `0.056s` | `0.400s` | `41` | `19776 MB` | not strict 20 fps |
| 5 | `5/5` | `70.4` | `0.071s` | `0.757s` | `114` | `19776 MB` | saturated |
| 6 | `6/6` | `72.3` | `0.083s` | `0.969s` | `134` | `19776 MB` | saturated |
| 8 | `8/8` | `64.5` | `0.124s` | `30.080s` | `204` | `19786 MB` | saturated with one large tail stall |

Read from the bucket experiment:

- Adding batch `4` consumed about another `2.1 GB` of resident VRAM versus the
  `8,16` INT8 profile, reaching about `19.8 GB` peak, still below the available
  `32 GB`.
- It did not improve strict 20 fps capacity. C1-C3 remained clean; C4 and above
  still recorded strict video stalls.
- Aggregate throughput was essentially unchanged at C4/C5, slightly better at
  C6 within run noise, and worse at C8 because of one long tail stall.
- The practical bottleneck is compute/tail latency in the generation path, not
  VRAM. Extra warmed buckets are useful only if they let the scheduler avoid
  meaningful padding without adding engine overhead.

Current optimized-path caveat from the markdowns:

- The faster measured path documented on 2026-05-29 was five-stage VAE INT8 plus
  static batch-8 TensorRT UNet split runtime, enabled with:
  `MUSETALK_UNET_BACKEND=trt`,
  `MUSETALK_TRT_UNET_ENABLED=1`,
  `MUSETALK_TRT_UNET_PATHS=8:models/tensorrt_unet_static_bs8_20260529/unet_trt.ts`,
  and `MUSETALK_TRT_FALLBACK=0`.
- On the RTX 3090 notes, that path improved WebRTC aggregate FPS by about
  `7-10%` over VAE INT8 plus PyTorch UNet at C4/C6/C8.
- This RTX 5000 Ada workspace currently lacks that UNet TRT artifact and the
  saved capture directory; an artifact check found only `models/musetalkV15/unet.pth`.
- The first 2026-06-08 RTX 6000 Ada run also lacked the validated UNet
  TensorRT artifact. Its server log showed `UNet backend: PyTorch`, so that
  result is a VAE INT8 baseline.
- The second 2026-06-08 RTX 6000 Ada run regenerated real batch-8 UNet
  captures, exported and validated
  `models/tensorrt_unet_static_bs8_rtx6000ada_20260608/unet_trt.ts`, enabled
  `UNet backend active: tensorrt_unet_multi`, and reran the WebRTC ramp.
- On RTX 6000 Ada, VAE INT8 + TRT UNet split8 improved the saturated WebRTC
  plateau from about `108-111 fps` to about `112-114 fps`, but strict smooth
  `20 fps` capacity remained `4` concurrent sessions.
- A same-backend RTX 6000 Ada `4,8,16` bucket rerun did not materially improve
  FPS versus `8,16`; it mainly raised peak VRAM from about `19.6 GB` to
  `21.7 GB`. Keep `8,16` as the optimized serving baseline.

HLS session load test on the same five-stage INT8 server:

- Date: 2026-05-26.
- Command family: `load_test.py`.
- Avatar: `test_avatar_2`.
- Audio: `data/audio/ai-assistant.mpga`.
- Request shape: `playback_fps=20`, `musetalk_fps=20`, request
  `batch_size=8`.
- Scheduler buckets: batch `8` only.
- HLS encoder: `libx264`.
- Report:
  `tmp/load_tests/load_test_hls_3090_int8_5stage_20_20_4_5_6_8streams_batch8_test_avatar_2_20260526.json`.
- Detailed report:
  `tmp/load_tests/load_test_hls_3090_int8_5stage_20_20_4_5_6_8streams_batch8_test_avatar_2_20260526_detailed.json`.

| Streams | Completed | Avg segment interval | Approx generated FPS | Avg live-ready | Max segment interval | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | `4/4` | `1.091s` | `73.3` | `1.895s` | `1.555s` | `8325 MB` |
| 5 | `5/5` | `1.397s` | `71.6` | `1.921s` | `2.537s` | `8335 MB` |
| 6 | `6/6` | `1.703s` | `70.5` | `2.187s` | `3.077s` | `8345 MB` |
| 8 | `8/8` | `2.278s` | `70.2` | `2.468s` | `4.610s` | `8367 MB` |

Closest saved HLS references:

- `load_test_report_20_20_4streams_8_16.json`: C4, `test_avatar`,
  FP16-oriented `8,16` profile, avg segment interval `1.188s`.
- `load_test_report_20_20_5streams_8_16.json`: C5, `test_avatar`,
  FP16-oriented `8,16` profile, avg segment interval `1.477s`.

The current INT8 run was modestly better at the comparable C4/C5 points:

| Streams | Saved reference | Current five-stage INT8 | Direction |
| ---: | ---: | ---: | ---: |
| 4 | `1.188s`, about `67.3` generated FPS | `1.091s`, about `73.3` generated FPS | `~8.2%` lower interval, `~8.9%` higher generated FPS |
| 5 | `1.477s`, about `67.7` generated FPS | `1.397s`, about `71.6` generated FPS | `~5.4%` lower interval, `~5.7%` higher generated FPS |

Caveat:

- An initial attempt to rerun the old `test_avatar` HLS shape on the live worker
  returned `404` for every session, so the successful current run used
  `test_avatar_2`.
- The comparison is therefore directional, not a clean FP16-vs-INT8 A/B.
- HLS segment cadence is also not the same metric as WebRTC frame receive
  cadence. The "generated FPS" column is estimated as
  `streams * musetalk_fps / avg_segment_interval`.

Server-side HLS logs across this run averaged:

| Metric | Average across 23 HLS sessions |
| --- | ---: |
| `avg_gpu_batch` | `0.1134s` |
| `avg_unet` | `0.0472s` |
| `avg_vae` | `0.0648s` |
| `avg_compose` | `0.0575s` |
| `avg_encode` | `0.1777s` |

This explains why HLS gains remain small: after VAE INT8, HLS still pays a
substantial per-chunk libx264 encode cost, while WebRTC does not use the HLS
segment encoder path.

Avatar portability validation:

- The INT8 artifacts are VAE decoder stage engines, not avatar-specific assets.
  The live cache names are stage and batch scoped, for example
  `decoder_pre_bs8_minmax_onnx_qdq.plan` and
  `decoder_up_block_2_bs8_minmax_onnx_qdq.plan`.
- The calibration corpus was captured from real `pred_latents` generated during
  `test_avatar_2` WebRTC requests. The payload metadata records
  `avatar_id=test_avatar_2`, so the current corpus is not multi-avatar yet.
- That calibration source affects how representative the INT8 activation ranges
  are; it does not bake `test_avatar_2` frames, masks, coordinates, or avatar
  latents into the TensorRT engines.
- A second avatar, `int8_avatar_probe_ai`, was prepared from
  `data/video/ai_test_default_moving_vid.mp4` on 2026-05-27.
- A `/generate` smoke request for that second avatar completed successfully:
  `gen_89a80670`.
- Output:
  `results/v15/avatars/int8_avatar_probe_ai/vid_output/int8_avatar_probe_ai_smoke.mp4.mp4`.
- `ffprobe` confirmed the output contains H.264 video at `512x512` for
  `7.95s` plus AAC audio for `8.00s`.
- API logs during that request confirmed
  `VAE decode timing backend=tensorrt_stagewise_int8_mixed`.

Conclusion from this validation:

- The current INT8 decoder should run for any prepared `v15` avatar that uses
  the same MuseTalk/VAE decode path and an already-warmed supported batch size.
- "Any avatar" is a runtime compatibility statement, not a universal visual
  quality guarantee. For production confidence, the calibration corpus should be
  expanded beyond `test_avatar_2` to include multiple avatars, face sizes,
  lighting conditions, and speech patterns.

Conclusion:

- Expanding INT8 coverage improved VAE decode materially:
  - about `22.6%` lower VAE decode time than FP16 stagewise
  - about `13.4%` lower VAE decode time than the two-stage INT8 config
- End-to-end `/generate` throughput still does not move much because the VAE
  decoder is no longer enough of the total request wall time. The UNet,
  scheduler, composition, encoding/muxing, request polling granularity, and
  serving overhead still dominate.
- The later batch `16` recovery improved WebRTC aggregate FPS, but the next
  likely major throughput branch is still UNet backend acceleration or UNet
  mixed INT8/FP16.
- The current five-stage INT8 WebRTC run is a real improvement over the saved
  RTX 3090 WebRTC diagnostic reference, but it still does not deliver strict
  `20 fps` per stream beyond low concurrency. At 8 streams, the server is closer
  to `7 fps` per stream, so larger gains still require batch-16 recovery,
  reducing the VAE/postprocess loop further, and accelerating UNet.
- The HLS session load path shows a small directional gain at the comparable
  C4/C5 points, not a massive throughput jump. HLS now appears limited by the
  combined GPU model pass plus libx264 segment encoding and queueing.

## 2026-05-27 UNet And VAE Throughput Plan For WebRTC

The next throughput push should treat the VAE decoder and MuseTalk UNet as a
single shared GPU generation cycle, because WebRTC concurrency is capped by how
quickly that loop can produce generated frames.

Current live WebRTC reference on the RTX 3090-class node:

| Metric | Current value |
| --- | ---: |
| request shape | `20/20 fps`, `batch_size=8` |
| scheduler bucket | `8` only |
| aggregate WebRTC FPS at 8 streams | about `55.9` |
| per-stream FPS at 8 streams | about `7.0` |
| `avg_gpu_batch` at 8 streams | about `0.1217s` |
| `avg_unet` at 8 streams | about `0.0554s` |
| `avg_vae` at 8 streams | about `0.0650s` |
| `avg_compose` at 8 streams | about `0.0612s` |

Capacity math:

| Target | Required generated FPS | Gap vs current `55.9` FPS |
| --- | ---: | ---: |
| `4 x 20 fps` strict realtime | `80` | `1.43x` |
| `6 x 20 fps` strict realtime | `120` | `2.15x` |
| `8 x 20 fps` strict realtime | `160` | `2.86x` |
| `10 x 20 fps` strict realtime | `200` | `3.58x` |

This means another small VAE-only improvement will not make higher WebRTC
concurrency feel easy. To make `6-8` concurrent streams smooth at `20 fps`, the
system needs a multi-part speedup: UNet acceleration, VAE batch/boundary work,
batch-16 recovery where memory allows, and eventually compose optimization if the
model loop improves enough.

Plain-English component map:

- WebRTC is the live pipe to the browser.
- The scheduler batches work from multiple streams so the GPU can process more
  frames per turn.
- UNet predicts the mouth/face latent from the avatar latent and audio features.
- The VAE decoder paints that latent back into a face crop.
- Compose pastes the generated crop into the avatar frame.
- TensorRT is the optimized runtime; FP16 is the safer fast path, INT8 is the
  smaller/faster path that needs calibration and visual gates.
- Calibration capture saves real tensors from the live path so optimized
  backends can be tested against known-good PyTorch outputs.

### Component 1: VAE Decoder

Current status:

- The five-stage ONNX/QDQ INT8 decoder is the validated live VAE path.
- The promoted stages are
  `decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2`.
- `decoder_up_block_3` and `decoder_postprocess` should remain out of the live
  INT8 set because direct comparison showed visible color/texture regressions.
- VAE decode still averages about `0.065s` per batch-8 WebRTC turn, so it remains
  the largest single model component, but it is no longer the only meaningful
  bottleneck.

Highest-ROI VAE work:

1. Keep the `8,16` VAE serving profile in the WebRTC A/B matrix now that exact
   batch `16` warms successfully.
2. Track its VRAM residency and tail frame intervals separately from average
   aggregate FPS.
3. Keep the rejected VAE stages in FP16/PyTorch unless a different quantization
   method proves visual safety.
4. Expand the calibration corpus across multiple avatars and speech patterns
   before calling the current VAE INT8 path production-representative.
5. Profile VAE boundary overhead separately: TensorRT stage bridge, tensor dtype
   casts, CPU/NumPy postprocess, and crop handoff into composition.

Expected impact:

- VAE-only work can likely improve the current model cycle, but the remaining
  safe VAE surface is limited.
- The main VAE upside now comes from better batch residency and less boundary
  overhead, not from blindly adding `decoder_up_block_3` or `decoder_postprocess`
  to INT8.

### Component 2: MuseTalk UNet

Current status:

- UNet is now close to the VAE decoder in the WebRTC GPU cycle:
  `0.0554s` UNet vs `0.0650s` VAE at the latest 8-stream point.
- It runs every generated batch and directly controls mouth motion, so it is the
  next major model quantization/backend target.
- The repo already has a UNet TensorRT export wrapper in `scripts/tensorrt_export.py`,
  but the live server does not yet have a validated UNet TensorRT runtime path.
- Previous `torch.compile`/CUDA graph attempts are not enough here; one live run
  failed during CUDA graph capture. The next branch should prefer explicit
  TensorRT/ONNX-style backend validation rather than relying on compile-only
  acceleration.

Required UNet calibration capture:

```text
latent_batch             # shape: (B, 8, 32, 32)
audio_feature_batch      # shape: (B, 50, 384)
timesteps                # usually tensor([0])
pred_latents reference   # FP16 PyTorch UNet output
avatar_id
audio_id or request_id
batch size
```

UNet implementation order:

1. Add a scheduler-side UNet input/output capture flag next to the existing VAE
   `pred_latents` calibration capture.
2. Build a correctness harness that compares PyTorch FP16 UNet output against a
   candidate backend on real captured batches.
3. Attempt exact-batch FP16 TensorRT UNet at batch `8` first.
4. Wire the UNet backend behind an opt-in runtime flag only after the isolated
   harness passes.
5. Run a lipsync smoke test through the current five-stage VAE INT8 decoder.
6. Add mixed INT8 only after FP16 backend correctness and live runtime stability
   are proven.
7. Start UNet INT8 with convolution/residual-heavy blocks; keep attention,
   normalization, timestep embedding, and output-sensitive layers in FP16 until
   stage-level validation says otherwise.

2026-05-29 implementation status:

- Step 1 is implemented in `scripts/hls_gpu_scheduler.py` behind
  `MUSETALK_UNET_CALIBRATION_CAPTURE=1`.
- Step 2 is implemented as `scripts/validate_unet_backend.py`. It can compare
  saved scheduler references against either the PyTorch UNet path or a serialized
  TensorRT UNet candidate.
- Step 3 is wired into `scripts/tensorrt_export.py`: `--unet-capture-dir` uses
  captured WebRTC tensors for UNet export examples, and
  `--validate-unet-capture-dir` runs post-export validation against those same
  captures.
- Step 4 has initial opt-in runtime plumbing in `scripts/trt_runtime.py` and
  `scripts/avatar_manager_parallel.py`, guarded by `MUSETALK_UNET_BACKEND=trt`
  or `MUSETALK_TRT_UNET_ENABLED=1`.
- The opt-in runtime must remain disabled until a real UNet TRT artifact passes
  validation on captured scheduler batches.

Useful commands for the next run:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/validate_unet_backend.py \
  --capture-dir ./calibration/unet \
  --backend pytorch \
  --limit 8 \
  --report-path tmp/unet_pytorch_reference_validation.json

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

/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/validate_unet_backend.py \
  --capture-dir ./calibration/unet \
  --backend trt \
  --trt-path ./models/tensorrt/unet_trt.ts \
  --limit 8 \
  --report-path tmp/unet_trt_fp16_validation.json
```

UNet quality gates:

- Predicted latent MAE and max error must stay bounded on real scheduler inputs.
- Decoded face crops through the current VAE INT8 path must remain visually close
  to the FP16 UNet reference.
- Full WebRTC lipsync must show no new mouth-shape drift, temporal jitter,
  identity drift, or teeth/tongue artifacts.

UNet performance gate:

- A UNet backend branch should reduce `avg_unet` enough to move aggregate WebRTC
  FPS, not merely improve an isolated synthetic benchmark.
- A useful first target is `avg_unet <= 0.040s` at batch `8`.
- For comfortable `6-8` WebRTC streams at `20 fps`, UNet acceleration must be
  combined with VAE/batch improvements; UNet alone is not expected to close a
  `2.86x` gap.

### Updated Execution Order

1. Keep the current five-stage VAE INT8 server as the visual baseline.
2. Add UNet real-input calibration capture and a UNet correctness harness.
3. Build FP16 UNet TensorRT or ONNX backend for exact batch `8`.
4. Validate UNet backend output, then run one lipsync smoke and one WebRTC C4/C6
   load test.
5. Keep the validated VAE `8,16` serving profile available for throughput
   experiments, but monitor VRAM residency and tail frame intervals.
6. Only after FP16 UNet backend is stable, add UNet mixed INT8/FP16 one region at
   a time.
7. If the combined model cycle drops enough that `avg_compose` becomes the
   limiter, begin the separate WebRTC compose optimization branch.

Success criteria for the next milestone:

| Milestone | Required result |
| --- | --- |
| C4 WebRTC | aggregate FPS near `80`, stable audio/video, no visual regression |
| C6 WebRTC | aggregate FPS near `120`, acceptable startup latency |
| C8 WebRTC | clear movement toward `160` aggregate FPS; if below target, quantify remaining gap |
| quality | lip sync and mouth detail comparable to current five-stage VAE INT8 output |

## Optimization Principles

1. **Optimize measured bottlenecks only**
   - Do not quantize a model just because it exists.
   - Quantize where the profiler shows meaningful time.

2. **Use real calibration data**
   - Random tensors are not enough for INT8 calibration.
   - Calibration should use real MuseTalk latents, real audio features, and real
     batch sizes from actual jobs.

3. **Prefer mixed precision over full INT8 at first**
   - Keep fragile or quality-sensitive stages in FP16.
   - Use INT8 only where visual and sync quality remain stable.

4. **Validate end-to-end, not only isolated model fps**
   - A faster VAE benchmark is useful.
   - The real success metric is generated video throughput and latency.

5. **Separate model optimization from media pipeline optimization**
   - TensorRT and quantization help UNet/VAE/Whisper.
   - They do not directly fix CPU composition or ffmpeg encode time.

## Candidate Ranking

| Priority | Target | Why | Main risk | Recommendation |
| --- | --- | --- | --- | --- |
| 1 | VAE decoder | Known bottleneck and already has TensorRT stagewise backend | Visual artifacts | Start here |
| 2 | MuseTalk UNet | Core per-frame generation model | Lip-sync drift and latent errors | Do after VAE |
| 3 | Whisper | Audio prep model | Audio alignment degradation | Only if prep is bottleneck |
| 4 | VAE encoder | Avatar prep only for cached avatars | Prep quality issues | Not a live-throughput priority |
| 5 | Face detection/parsing | Avatar prep only | Bad crops/masks | Avoid for now |
| 6 | Positional encoding | Too small | Little or no speedup | Do not prioritize |
| 7 | SyncNet | Not core live inference | No runtime benefit | Do not prioritize |

## Phase 0: Establish A Baseline

Before changing precision, capture a clean baseline.

Required measurements:

- End-to-end fps for generated output.
- First-frame or first-segment latency.
- UNet time per batch.
- VAE decode time per batch.
- Whisper/audio preparation time.
- Tensor H2D/D2H transfer time if available.
- CPU composition time.
- ffmpeg encode/mux time.
- Queueing and scheduler wait time.

Recommended batch coverage:

```text
BS1, BS2, BS4, BS8, BS16, BS32
```

Recommended test clips:

- Short audio, 3 to 5 seconds.
- Medium audio, 15 to 30 seconds.
- Longer audio, 60 seconds.
- At least two avatars with different face sizes and motion.

Baseline output artifacts to save:

```text
baseline timing logs
baseline generated videos
baseline generated face crops
baseline UNet outputs if practical
baseline VAE decoder outputs
```

Acceptance for moving forward:

- Reproducible timings across multiple runs.
- Clear percentage split between UNet, VAE decode, compose, encode, and audio
  prep.
- A saved visual reference set for quality comparison.

## Phase 1: Capture Calibration Data

INT8 quantization needs representative tensors.

For VAE decoder calibration, capture:

```text
pred_latents
batch size
avatar id or avatar type
frame index
output face crop reference
```

For UNet calibration, capture:

```text
latent_batch
audio_feature_batch
timesteps
positional encoding output if separate
UNet predicted latents
batch size
```

For Whisper calibration, only if needed, capture:

```text
audio feature inputs
Whisper encoder hidden states
audio duration
language or phonetic variety
```

Calibration corpus requirements:

- Use real avatars and real audio.
- Include different speakers.
- Include quiet, loud, fast, and slow speech.
- Include different mouth motion intensity.
- Include the actual fixed batch sizes used by the scheduler.

Do not calibrate only on random tensors. Random tensors may produce a fast
engine that is inaccurate on real MuseTalk distributions.

## Phase 2: VAE Decoder Mixed INT8/FP16

This is the first quantization target.

Why:

- VAE decode is a known hot path.
- TensorRT stagewise infrastructure already exists.
- VAE decode runs for every generated frame batch.
- Reducing VAE time directly improves the GPU generation loop.

Recommended implementation strategy:

1. Keep the current stagewise TensorRT structure.
2. Add INT8 calibration for selected decoder stages.
3. Keep fragile normalization or output-sensitive stages in FP16 at first.
4. Compare each stage against the FP16 baseline.
5. Promote more stages to INT8 only when quality remains stable.

Suggested precision order:

```text
current: FP16 PyTorch / FP16 TensorRT stagewise
step 1: INT8 internal conv-heavy stages, FP16 boundaries
step 2: mixed INT8/FP16 with normalization kept FP16
step 3: broader INT8 only if visual quality remains stable
```

Validation checks:

- Side-by-side generated face crops.
- Full-frame blended output.
- Mouth-region comparison.
- Color and skin texture stability.
- Flicker across consecutive frames.
- Edge blending around the mask.
- End-to-end HLS/WebRTC playback.

Suggested metrics:

```text
VAE isolated latency
end-to-end fps
mean absolute pixel error on generated crop
SSIM or perceptual similarity if available
manual visual review
```

Quality gate:

- No obvious mouth artifacts.
- No obvious color shift.
- No frame-to-frame flicker introduced by quantization.
- No unstable mask boundary artifacts.

Performance gate:

- VAE decode should improve meaningfully beyond the current FP16 TensorRT
  stagewise result.
- End-to-end fps should improve, not just isolated VAE fps.

## Phase 3: UNet TensorRT Or Mixed INT8/FP16

The UNet is the second major target.

Why:

- It is the main audio-conditioned latent generator.
- It runs per generated batch.
- After VAE decode improves, UNet may become the next dominant GPU cost.

Recommended strategy:

1. First attempt stable TensorRT/compiled FP16 UNet if current runtime supports
   it reliably.
2. If FP16 compilation is stable, add INT8 calibration for selected blocks.
3. Keep attention, normalization, and sensitive output areas in FP16 until
   proven safe.
4. Validate predicted latents and final generated video.

Calibration inputs:

```text
latent_batch
audio_feature_batch
timesteps
```

Validation outputs:

```text
UNet predicted latents
VAE decoded face crops
final blended video frames
```

Risk areas:

- Audio-lip alignment.
- Mouth openness.
- Teeth and tongue detail.
- Jitter across frames.
- Identity preservation.
- Latent distribution shift that only becomes visible after VAE decode.

Quality gate:

- Lip-sync must look equivalent to baseline.
- Mouth shapes must remain stable across fast phoneme changes.
- No increased jitter in the mouth region.
- No systematic identity drift.

Performance gate:

- UNet isolated latency improves.
- End-to-end generation improves after accounting for scheduler and VAE time.
- No new first-request compilation penalty in production paths.

## Phase 4: Whisper Quantization Only If Needed

Whisper should not be the first quantization target unless profiling shows audio
preparation is limiting throughput or first-response latency.

Useful when:

- Many short audio requests arrive concurrently.
- First-token or first-segment latency matters more than steady-state fps.
- Audio preprocessing dominates queue time.
- GPU time is available but audio prep is blocking job scheduling.

Risks:

- Worse phoneme representation.
- Degraded speech timing.
- Subtle lip-sync errors downstream.

Recommendation:

- Leave Whisper FP16 unless the baseline proves it is a bottleneck.
- If optimized, compare Whisper feature outputs and final video sync, not only
  Whisper encoder speed.

## Phase 5: Non-model Bottlenecks

Quantization will not fix every bottleneck. Once UNet and VAE are faster, these
areas may become dominant:

- CPU composition.
- Generated crop resize and blend.
- CPU/GPU tensor copies.
- ffmpeg encode and mux.
- HLS segment creation.
- WebRTC frame pacing.
- Scheduler queueing.

Potential non-quantization work:

- Move composition to GPU if CPU blend becomes dominant.
- Reduce unnecessary CPU/GPU transfers.
- Keep tensors in GPU memory until absolutely necessary.
- Batch compose work more efficiently.
- Tune ffmpeg settings for latency and throughput.
- Ensure NVENC is used when available and beneficial.
- Avoid per-request cold starts and repeated avatar loads.

These should be tracked separately from quantization experiments so the impact
of each change is clear.

## Acceptance Criteria

An optimization should only be considered successful if it passes both speed and
quality gates.

Speed criteria:

- Isolated target model latency improves.
- End-to-end generation fps improves.
- First playable output latency does not regress.
- Throughput under concurrent sessions improves or remains stable.
- No major new memory pressure or engine build delay appears during live use.

Quality criteria:

- Lip-sync remains comparable to FP16 baseline.
- No obvious visual artifacts.
- No new flicker.
- No systematic color shift.
- No identity degradation.
- No mask-edge instability.

Operational criteria:

- Engines are warmed before live traffic.
- Batch sizes match scheduler behavior.
- Fallback to FP16/PyTorch remains available.
- Logs identify which backend and precision were used.
- A bad quantized engine can be disabled quickly through config.

## Recommended Experiment Order

1. Re-run current FP16 PyTorch and TensorRT stagewise baselines.
2. Add calibration capture for VAE `pred_latents`.
3. Build a mixed INT8/FP16 VAE decoder experiment.
4. Compare VAE output against FP16 baseline.
5. Measure end-to-end fps and first-output latency.
6. If VAE quality is stable, expand INT8 coverage stage by stage.
7. Profile again to see whether UNet is now the main GPU bottleneck.
8. Attempt stable FP16 UNet TensorRT/compiled execution.
9. Add UNet INT8 only after FP16 UNet backend is stable.
10. Revisit Whisper only if audio prep remains material.

## Rollback Plan

Every quantization experiment should have a fast rollback path.

Recommended runtime switches:

```text
VAE_BACKEND=pyTorch_fp16
VAE_BACKEND=trt_stagewise_fp16
VAE_BACKEND=trt_stagewise_int8_mixed

UNET_BACKEND=pyTorch_fp16
UNET_BACKEND=trt_fp16
UNET_BACKEND=trt_int8_mixed
```

The exact variable names can follow the existing project naming style, but the
principle is important: precision and backend should be configurable without
code edits.

## Final Recommendation

The best next optimization is not full-pipeline quantization. It is targeted
mixed precision:

1. VAE decoder stagewise TensorRT INT8/FP16 first.
2. UNet TensorRT or mixed INT8/FP16 second.
3. Whisper only if audio preparation is measured as a bottleneck.
4. Composition and encoding handled as separate pipeline optimization tracks.

This gives the highest chance of real throughput improvement without damaging
the visual quality that makes the generated talking head usable.

## 2026-05-29 Current Chat Context

The active user goal is WebRTC concurrency throughput, not HLS segment latency:
roughly `80`, `120`, and `160` aggregate generated fps for `4`, `6`, and `8`
concurrent `20 fps` WebRTC streams.

Important clarification from this chat: `scripts/hls_gpu_scheduler.py` is still
the right implementation point because WebRTC uses the shared GPU scheduler for
the `UNet -> VAE decode -> compose` generation loop. The recent scheduler edits
do not directly increase fps; they make the next UNet TensorRT experiment
measurable and safe.

Current status:

- Five-stage VAE INT8 remains the baseline.
- UNet is the next backend target because it is now close to VAE in the measured
  WebRTC GPU turn.
- PyTorch UNet captures are the correctness reference. They contain real
  `latent_batch`, `audio_feature_batch`, `timesteps`, and PyTorch FP16
  `pred_latents`.
- FP16 TensorRT UNet export uses the existing model graph and weights; captures
  are for real example tensors and validation. Future INT8 UNet will use the
  same capture corpus for calibration.
- Implemented: UNet capture, UNet validation harness, capture-aware UNet export
  flags, opt-in UNet TRT runtime, and strict no-silent-fallback guards.
- Measured on RTX 6000 Ada: real batch-8 UNet TensorRT export, validation, and
  WebRTC speedup. The result is incremental, about `2-4%` aggregate FPS versus
  VAE INT8 + PyTorch UNet on that node.
- Still pending on RTX 5000 Ada: local UNet TensorRT export and WebRTC rerun.

Infrastructure blocker:

```text
nvidia-smi sees RTX 3090
GPU memory used = 0 MiB
torch.cuda.is_available() = False
libcuda.cuInit(0) = 999
```

This is below PyTorch/TensorRT/MuseTalk. Restarting only `api_server.py` is not
expected to fix it; the Vast container/instance likely needs a restart. The API
and TURN server were intentionally left stopped to avoid fake CPU/PyTorch
fallback tests.

Resume sequence after Vast restart:

1. Verify CUDA returns `torch_cuda True` and `cuInit 0`.
2. Start the current VAE INT8 `8,16` profile with
   `MUSETALK_UNET_CALIBRATION_CAPTURE=1`.
3. Run a short representative WebRTC wall session to write `./calibration/unet`.
4. Export and validate FP16 UNet TensorRT with:

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

5. Disable capture, enable the opt-in UNet TRT runtime, run one lipsync smoke,
   then run WebRTC C4/C6/C8 load tests.

## 2026-05-29 VAE Postprocess Optimization Update

The CUDA blocker above was resolved after moving back to a healthy worker. The
current live stack is now:

- five-stage VAE INT8 decoder with `8,16` warmed buckets
- static batch-8 TensorRT UNet runtime, splitting larger scheduler batches into
  validated batch-8 calls
- TURN relay path restored for public WebRTC wall tests

Implemented and tested the first VAE postprocess optimization:

- `musetalk/models/vae.py` now defaults `MUSETALK_VAE_FAST_POSTPROCESS=1`
- decoded tensors are scaled, rounded, clamped, converted to `uint8`, and
  flipped RGB-to-BGR on GPU before the CPU copy
- `MUSETALK_VAE_FAST_POSTPROCESS=0` keeps the old NumPy postprocess path as a
  rollback switch

Validation:

- random-tensor equivalence with the old path: exact match, `max_diff=0`
- postprocess-only microbenchmark:
  - batch `8`: `11.97ms` -> `0.27ms`
  - batch `16`: `35.07ms` -> `0.44ms`
- live VAE timing before:
  `avg_tensor=0.1245s avg_post=0.0069s avg_total=0.1314s`
- live VAE timing after:
  `avg_tensor=0.1254s avg_post=0.0009s avg_total=0.1263s`

WebRTC C4/C6/C8 after the fast postprocess path:

| Streams | Completed | Aggregate FPS | Max frame interval |
| ---: | ---: | ---: | ---: |
| `4` | `4/4` | `71.4` | `0.263s` |
| `6` | `6/6` | `72.3` | `0.997s` |
| `8` | `8/8` | `72.1` | `1.552s` |

The optimization is worth keeping because it removes almost all CPU/NumPy
postprocess overhead, but it is not a major throughput unlock by itself. The
VAE tensor decode is still about `125ms` per live decode call, so the next VAE
work should target tensor decode reduction, VAE stage scheduling/overlap, or a
larger output-path architecture change rather than more CPU postprocess tuning.

### 2026-05-31 9:16 Avatar Shape Check

Prepared and load-tested a portrait idle/base video to check whether moving
from the usual `1024 x 1024` source shape to `720 x 1280` creates a WebRTC
throughput or VRAM regression.

Tested avatar:

- avatar id: `portrait_9x16_720p_20260531`
- source: `assets/ee3102596e4ee7b432d81d2fccde54b6.mp4`
- source shape: `720 x 1280`, `24 fps`, `10.04s`
- prepared cycle frames: `482`
- prepared artifact size: about `536 MB`
- warmed avatar cache memory: about `2198.6 MB`

WebRTC load test on the current RTX 3090 `8,16` warmed-bucket path:

| Streams | Completed | Avg frame interval | Approx per-stream FPS | Avg live-ready | Max frame interval | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4` | `4/4` | `0.054s` | `18.5` | `3.720s` | `0.354s` | `19611 MB` |
| `5` | `5/5` | `0.068s` | `14.7` | `3.270s` | `0.699s` | `19613 MB` |
| `6` | `6/6` | `0.080s` | `12.5` | `4.051s` | `0.884s` | `19621 MB` |
| `8` | `8/8` | `0.108s` | `9.3` | `6.408s` | `1.429s` | `19623 MB` |

Report files:

- `tmp/load_tests/load_test_webrtc_3090_portrait_9x16_720p_20_20_4_5_6_8streams_8_16_20260531.json`
- `tmp/load_tests/load_test_webrtc_3090_portrait_9x16_720p_20_20_4_5_6_8streams_8_16_20260531_detailed.json`

Result: no CUDA OOM and no failed WebRTC sessions. Compared with the closest
May 29 fast-postprocess baseline, average frame intervals were slightly better
and peak VRAM was only about `25-30 MB` higher, which is effectively runtime
noise at this footprint. The remaining caveat is smoothness: `8` streams
complete, but cadence is still below strict `20 fps` per stream.

## 2026-05-29 VAE Stage Timing Update

Added opt-in stage-level timing to `scripts/trt_runtime.py`:

```text
MUSETALK_TRT_STAGEWISE_STAGE_TIMING=1
MUSETALK_TRT_STAGEWISE_STAGE_TIMING_SYNC=1
MUSETALK_TRT_STAGEWISE_STAGE_TIMING_LOG_INTERVAL=10
```

This is profiling-only because synchronized timing forces a CUDA synchronize
after every stage. It should stay disabled for normal WebRTC wall testing.

Measured with the live five-stage INT8 VAE decoder, `8,16` buckets, static
batch-8 TensorRT UNet split runtime, and a short C4 WebRTC run:

| Stage | Avg time | Share |
| --- | ---: | ---: |
| `decoder_pre` | `0.0004s` | `0.3%` |
| `decoder_mid_block` | `0.0035s` | `2.8%` |
| `decoder_up_block_0` | `0.0051s` | `4.1%` |
| `decoder_up_block_1` | `0.0193s` | `15.6%` |
| `decoder_up_block_2` | `0.0312s` | `25.2%` |
| `decoder_up_block_3` | `0.0600s` | `48.4%` |
| `decoder_postprocess` | `0.0045s` | `3.6%` |

This confirms the remaining VAE time is mostly `decoder_up_block_3`, with
`decoder_up_block_2` second. The earlier rejected stage, `decoder_up_block_3`,
is therefore the big remaining VAE target, but it is also the stage that caused
visible INT8 quality regressions. Any future work on that stage must be
quality-gated with saved comparison crops and lipsync smoke tests.

Also tested the safer FP16 compile knob:

```text
MUSETALK_TRT_STAGEWISE_MIN_BLOCK_SIZE=1
```

It did not materially change timing:

- `decoder_up_block_3`: about `0.0598s`
- total VAE stage timing: about `0.1245s`

Recommendation after this profiling:

- Keep the current five safe INT8 stages.
- Keep `decoder_up_block_3` FP16 in the live path for now.
- Use the new timing hook for A/B tests.
- Next meaningful VAE attempt should be a visual-quality-gated late-block
  experiment or a scheduling/overlap change, not more CPU postprocess tuning.

## 2026-05-29 `decoder_up_block_3` Deep Dive

`decoder_up_block_3` is Diffusers `UpDecoderBlock2D`. In this VAE it contains:

- three `ResnetBlock2D` children
- no upsampler
- each ResNet contains group norm, SiLU, two 3x3 convs, and residual add
- the first ResNet also has a shortcut conv because channels change from `256`
  to `128`

Focused batch-16 profiling showed:

| Child | Avg time |
| --- | ---: |
| `resnets.0` | `0.02798s` |
| `resnets.1` | `0.01879s` |
| `resnets.2` | `0.01880s` |
| full block | `0.06577s` |

Within the block, the six 3x3 convolutions are the main cost. Group norms,
activations, and residual adds are not free, but the bottleneck is mostly
full-resolution convolution at `[batch, 128, 256, 256]`.

For comparison, `decoder_up_block_2` spends about `0.02343s` in its upsampler
plus conv, then enters the same high-resolution tensor shape. That explains why
block 2 and block 3 dominate the remaining VAE decode time.

Quality-gated `decoder_up_block_3` INT8 attempts:

- batch `16`, `onnx_qdq`, `minmax`: TensorRT build failed with GPU OOM/assertion
- batch `8`, `onnx_qdq`, `minmax`: built, but `mae=0.01905`
- batch `8`, `onnx_qdq`, `entropy2`: built, same `mae=0.01905`

Saved comparison sheets:

- `tmp/vae_decoder_int8_experiment/up3_minmax_bs8_20260529/comparison_sheet.png`
- `tmp/vae_decoder_int8_experiment/up3_entropy2_bs8_20260529/comparison_sheet.png`

Conclusion: whole-stage INT8 on `decoder_up_block_3` remains too risky for live
serving. The next serious attempt should be finer-grained than the current
stage switch, such as quantizing only selected high-cost convs inside the block
or using a different fusion/backend strategy for the full-resolution residual
convs.

## 2026-06-11 Lower-Resolution Lipsync ROI Experiment

The current live generation path is still built around a `256 x 256` talking-face
ROI:

- avatar preparation crops the detected face box and resizes it to `256 x 256`
- the VAE encodes that crop into `4 x 32 x 32` latents
- UNet predicts `4 x 32 x 32` latents
- VAE decodes back to a `256 x 256` generated face crop
- composition resizes that generated crop back into the original face box

This matters because the remaining VAE decode bottleneck is dominated by late,
full-resolution decoder work. `decoder_up_block_3` runs at
`[batch, 128, 256, 256]` and accounts for about `48%` of the measured VAE stage
time; `decoder_up_block_2` is second at about `25%`. Lowering the generated ROI
resolution could therefore reduce the same expensive late-block convolutional
work that is currently limiting throughput.

Potential experiment:

1. Add an opt-in generated ROI size, initially `192`, while keeping `256` as the
   default production path.
2. Prepare a separate test avatar cache for the lower ROI size instead of
   reusing existing `256 -> 32` latent caches.
3. Validate the full model contract end to end:
   - avatar prep crop resize
   - cached latent shape
   - UNet PyTorch correctness/smoke behavior
   - VAE TensorRT/stagewise backend compatibility
   - composition resize and mask quality
4. Run visual gates before throughput claims:
   - saved face crops
   - final blended frames
   - short lipsync video smoke
   - close-up mouth/teeth/edge comparison against `256`
5. If `192` passes, test lower degraded-mode sizes such as `160` or `128`.

Expected upside:

- This may be a bigger throughput lever than trying to squeeze a few more
  milliseconds from the existing `256 x 256` late decoder blocks.
- It directly reduces the spatial size of the expensive final VAE decode work.
- It could support a separate lower-quality/high-concurrency serving tier.

Risks and constraints:

- This is not just output downscaling; it changes the latent/model serving
  contract and likely invalidates existing TensorRT artifacts and prepared
  avatar caches.
- The MuseTalk UNet was trained and validated around the current latent/image
  shape, so quality and motion may regress even if the code runs.
- `128 x 128` may be visibly too soft for mouth detail, teeth, and blend
  boundaries; start with `192 x 192`.
- Treat this as a separate experimental serving profile, not a replacement for
  the current quality path until it passes visual and WebRTC load tests.

Current priority read:

1. Keep `decoder_up_block_3` as the primary same-resolution VAE optimization
   target.
2. Put lower-resolution generated ROI close behind it as a potentially larger
   throughput experiment.
3. Avoid spending more time on CPU VAE postprocess or extra warmed buckets until
   these two paths are tested.

### 2026-06-11 First 192px ROI Smoke Result

Implemented an opt-in `MUSETALK_GENERATED_ROI_SIZE` experiment and generated a
direct comparison from the same source avatar/audio:

| ROI size | Cached latent shape | Inference time for 95 frames | Output video |
| ---: | --- | ---: | --- |
| `256` | `[1, 8, 32, 32]` per cached frame | `4.34s` | `results/v15/avatars/roi_compare_256_20260611/vid_output/baseline_256.mp4` |
| `192` | `[1, 8, 24, 24]` per cached frame | `2.84s` | `results/v15/avatars/roi_compare_192_20260611/vid_output/downscaled_192.mp4` |

Side-by-side artifact:

```text
results/roi_resolution_compare_20260611/side_by_side_256_vs_192.mp4
```

Initial read:

- The speed signal is real: `192` was about `35%` faster on this short offline
  PyTorch smoke when including image saving.
- The quality hit is severe: the `192` output visibly smears the mouth region and
  looks unacceptable compared with the `256` baseline.
- Treat `192` as a failed quality candidate for normal serving, not merely a
  slightly degraded high-throughput mode.

Detailed quality analysis:

1. The model contract changed, not just final display scaling.
   - `256` preparation produced cached latents shaped `[1, 8, 32, 32]`.
   - `192` preparation produced cached latents shaped `[1, 8, 24, 24]`.
   - MuseTalk's UNet and VAE path were trained, validated, and previously
     optimized around the `256 -> 32` latent geometry. Feeding `24 x 24` latents
     therefore changes the spatial distribution the UNet sees.

2. The face ROI is too large for a `192` generated crop.
   - The measured face box for this test averaged about `170 x 204` pixels.
   - The `256` path decodes a `256 x 256` face crop, then downsamples it into the
     final face box. That preserves more high-frequency mouth detail before
     blending.
   - The `192` path decodes a `192 x 192` face crop, then resizes it to roughly
     `170 x 204`. Width is slightly downscaled, but height is upscaled by about
     `1.06x`. The mouth patch is therefore generated from fewer latent pixels and
     then stretched vertically into the final frame.

3. The artifact is concentrated where viewers are most sensitive.
   - The cheeks/skin tolerate blur relatively well.
   - Teeth, lip edges, mouth corners, and the inner mouth do not. Those details
     are small, high-contrast structures inside the generated ROI.
   - The result is a soft, smudged mouth patch even though the full-frame
     difference metric is small. A rough frame diff showed low whole-frame
     average difference, but the mouth region changed much more visibly.

4. The blend mask does not solve lost generated detail.
   - Composition still resizes the generated face crop back to the original bbox
     before masked blending.
   - Masking can hide outer face seams, but it cannot recover lip/teeth detail
     that the lower-resolution latent decode never generated.

Conclusion:

- Do not use `192 x 192` generated ROI for the quality path.
- Keep `256 x 256` as the production-quality ROI.
- Lower ROI remains useful as evidence that spatial reduction can improve speed,
  but any acceptable version needs a less aggressive reduction or a different
  model/composition strategy.

Next experiments, if continuing this branch:

1. Test `224 x 224`, because it keeps a `28 x 28` latent grid and reduces the
   final upscale/downscale mismatch.
2. Test adaptive ROI sizing only for small face boxes where the final pasted face
   region is smaller than the generated crop, avoiding upscaling generated lips.
3. Test mouth-preserving composition, such as generating a larger mouth-local ROI
   while reducing only non-mouth areas, if architecture time is available.
4. Keep `decoder_up_block_3` optimization as the cleaner same-quality path,
   because it attacks the bottleneck without changing the learned latent geometry.

### 2026-06-11 224px ROI Follow-Up

Tested the next less aggressive generated ROI size with the same source
avatar/audio and the same opt-in `MUSETALK_GENERATED_ROI_SIZE` path.

| ROI size | Cached latent shape | Inference time for 95 frames | Quality read |
| ---: | --- | ---: | --- |
| `256` | `[1, 8, 32, 32]` per cached frame | `4.34s` | Production-quality baseline |
| `224` | `[1, 8, 28, 28]` per cached frame | `3.47s` | Much closer to `256`; mild softness on inspected frames |
| `192` | `[1, 8, 24, 24]` per cached frame | `2.84s` | Failed quality candidate; visible mouth smear |

Artifacts:

```text
results/v15/avatars/roi_compare_224_20260611/vid_output/downscaled_224.mp4
results/roi_resolution_compare_20260611/side_by_side_256_vs_224.mp4
results/roi_resolution_compare_20260611/side_by_side_256_vs_224_vs_192.mp4
results/roi_resolution_compare_20260611/side_by_side_256_vs_224_contact_sheet.png
```

Why `224` behaved better than `192`:

1. It avoids upscaling the generated face crop in this test.
   - The measured face box averaged about `170 x 204` pixels.
   - `224 -> 170 x 204` still downsamples both axes, with scale factors around
     `0.76x` width and `0.91x` height.
   - `192 -> 170 x 204` downsampled width but upscaled height by about `1.06x`,
     which stretched already lower-resolution generated mouth detail.

2. The latent grid reduction is less disruptive.
   - `224` uses a `28 x 28` latent grid instead of the baseline `32 x 32`.
   - `192` drops to `24 x 24`, which is a larger spatial contract change for the
     UNet/VAE path and leaves fewer pixels for teeth, lip edges, and inner-mouth
     structure.

3. Approximate frame metrics matched the visual read.
   - Whole-frame average absolute diff vs `256`: `224 ~= 0.97`, `192 ~= 1.09`.
   - Mouth-region average absolute diff vs `256`: `224 ~= 2.95`,
     `192 ~= 4.99`.
   - These numbers are not a replacement for watching the video, but they support
     the visual result that `224` is much less damaging than `192`.

Decision:

- Do not promote `224 x 224` to production yet.
- Treat `192 x 192` as rejected for normal visual quality.
- Next validation for `224`: run WebRTC throughput before deciding whether the
  blur is worth keeping as an experimental lower-quality profile.
- If the WebRTC result is only modest, revert the `MUSETALK_GENERATED_ROI_SIZE`
  implementation and keep `256 x 256` as the only supported generated ROI.

### 2026-06-11 224px ROI INT8 WebRTC Load Test

After watching the side-by-side, the quality read changed from mild softness to
noticeably blurry. This load test therefore answers a narrower question: how much
throughput the blurry `224 x 224` generated ROI actually buys.

Reference test shape:

- Matched the recent WebRTC load-test shape from
  `docs/webrtc_load_test_findings_2026-06-07.md`,
  `load_test_webrtc_rtx6000ada_int8_5stage_20fps_20260608.md`, and
  `load_test_webrtc_rtx6000ada_int8_trt_unet_split8_20fps_20260608.md`.
- WebRTC target: `--playback-fps 20 --musetalk-fps 20`.
- Harness: `load_test_webrtc.py`.
- Avatar: `test_avatar_224_load_20260611`, prepared from
  `data/video/chatgpt_moving_vid.mp4`.
- Audio: `data/audio/ai-assistant.mpga`.
- Request batch size: `8`.
- Scheduler buckets / warmup batches: `8,16`.
- H.264 encoder: `libx264`.
- Sync/profile settings: strict FIFO, adaptive fps disabled, video prebuffer
  `2.0s`, audio prebuffer `0.0s`.

Exact load-test commands:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python load_test_webrtc.py \
  --base-url http://127.0.0.1:8000 \
  --avatar-id test_avatar_224_load_20260611 \
  --audio-file data/audio/ai-assistant.mpga \
  --ramp 1,2,3,4,5,6,8 \
  --hold-seconds 10 \
  --segment-duration 1 \
  --playback-fps 20 \
  --musetalk-fps 20 \
  --batch-size 8 \
  --stage-ready-timeout 140 \
  --connection-timeout 60 \
  --completion-timeout 240 \
  --gpu-sample-interval 1 \
  --gpu-log-interval 20 \
  --report-path tmp/load_tests/load_test_webrtc_3090_224roi_int8_20_20_1_2_3_4_5_6_8streams_8_16_20260611.json \
  --detail-report-path tmp/load_tests/load_test_webrtc_3090_224roi_int8_20_20_1_2_3_4_5_6_8streams_8_16_20260611_detailed.json

/workspace/.venvs/musetalk_trt_stagewise/bin/python load_test_webrtc.py \
  --base-url http://127.0.0.1:8000 \
  --avatar-id test_avatar_224_load_20260611 \
  --audio-file data/audio/ai-assistant.mpga \
  --ramp 10,12 \
  --hold-seconds 10 \
  --segment-duration 1 \
  --playback-fps 20 \
  --musetalk-fps 20 \
  --batch-size 8 \
  --stage-ready-timeout 160 \
  --connection-timeout 60 \
  --completion-timeout 300 \
  --gpu-sample-interval 1 \
  --gpu-log-interval 20 \
  --report-path tmp/load_tests/load_test_webrtc_3090_224roi_int8_20_20_10_12streams_8_16_20260611.json \
  --detail-report-path tmp/load_tests/load_test_webrtc_3090_224roi_int8_20_20_10_12streams_8_16_20260611_detailed.json
```

Throughput calculation:

```text
aggregate fps = concurrency / avg_segment_interval_s
```

The smoothness column intentionally follows the recent RTX 5000/6000 WebRTC
notes: an average interval near `0.050s` is not enough by itself. A row is only
treated as smooth when the max interval does not show meaningful throttling.
That is why `C4` is not counted as smooth even though its average interval is
close to target.

Runtime tested:

- GPU: `NVIDIA GeForce RTX 3090`, `24,576 MiB`.
- `MUSETALK_GENERATED_ROI_SIZE=224`.
- `MUSETALK_ALLOW_NON256_TRT_VAE=1`.
- VAE backend: `tensorrt_stagewise_int8_mixed`.
- INT8 frontend: `onnx_qdq`.
- INT8 VAE stages:
  `decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2`.
- UNet backend: PyTorch.
- 224 calibration corpus: `calibration/vae_decoder_224`, with `64` captured
  files and representative `pred_latents` shape `[16, 4, 28, 28]`.
- 224 INT8 engine cache:
  `models/tensorrt/stagewise_int8_224_onnx_qdq_cache`.
- Server log evidence:
  - `Stagewise TRT warmup complete (batches=[8, 16], total=218.22s)`
  - `VAE decode backend active: tensorrt_stagewise_int8_mixed`
  - `UNet backend: PyTorch`
  - `HLS GPU scheduler started (max_combined_batch_size=16, fixed_batch_sizes=[8, 16]...)`

Toolchain note:

- The latest `nvidia-modelopt` dependency chain attempted to pull a much newer
  Torch/CUDA stack, so the working environment was kept on Torch `2.5.1+cu121`
  and ModelOpt was pinned to `nvidia-modelopt==0.23.2` with `onnx==1.17.0`.

Results:

| Concurrency | Completed | Failed | Avg live-ready | Avg interval | Max interval | Aggregate fps | Peak VRAM | Smooth at 20 fps |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 1 | 1 | 0 | 11.061s | 0.051s | 0.070s | 19.6 | 17,756 MB | Yes |
| 2 | 2 | 0 | 1.762s | 0.051s | 0.169s | 39.2 | 17,760 MB | No |
| 3 | 3 | 0 | 2.371s | 0.051s | 0.073s | 58.8 | 17,766 MB | Yes |
| 4 | 4 | 0 | 2.807s | 0.054s | 0.421s | 74.1 | 17,766 MB | No |
| 5 | 5 | 0 | 3.267s | 0.069s | 0.693s | 72.5 | 17,766 MB | No |
| 6 | 6 | 0 | 4.254s | 0.082s | 0.956s | 73.2 | 17,770 MB | No |
| 8 | 8 | 0 | 5.516s | 0.110s | 1.450s | 72.7 | 17,770 MB | No |
| 10 | 10 | 0 | 7.132s | 0.139s | 2.114s | 71.9 | 17,770 MB | No |
| 12 | 12 | 0 | 8.449s | 0.168s | 2.969s | 71.4 | 17,770 MB | No |

Capacity interpretation:

| Read | Value |
| --- | ---: |
| Strict smooth 20 fps capacity | `3` concurrent streams |
| Highest near-target average row | `C4`, `0.054s` avg interval |
| First clearly saturated row | `C5`, `0.069s` avg interval |
| Saturated aggregate plateau | about `72-74 fps` |
| Completion-only stress ceiling tested | `12/12` sessions completed |
| Practical production recommendation | do not use `224`; quality is too blurry |

The `C10` and `C12` rows are useful stress data, but they should not be read as
realtime capacity. They completed without request failures while producing only
about `6-7 fps` per stream on average.

Server-side timing read across the completed 224 ROI WebRTC sessions:

| Metric | Mean | Min | Max |
| --- | ---: | ---: | ---: |
| `avg_gpu_batch` | `0.1863s` | `0.181s` | `0.208s` |
| `avg_unet` | `0.0624s` | `0.058s` | `0.088s` |
| `avg_vae` | `0.1220s` | `0.118s` | `0.125s` |
| `avg_compose` | `0.0692s` | `0.064s` | `0.084s` |

Read:

- The saturated aggregate plateau on this local RTX 3090 is about `72-74 fps`.
- Strict smooth `20 fps` capacity remains `3` concurrent sessions by the same
  max-interval warning rule used in the recent RTX 5000/6000 notes. `C4` is
  close on average, but its `0.421s` max interval is not smooth.
- Higher concurrency completes without request failures, but it only divides the
  same aggregate frame budget across more sessions: `C10` and `C12` are stress
  passes, not good realtime serving capacities.
- Peak resident VRAM stayed near `17.77 GB`; lowering the generated ROI did not
  materially reduce resident memory because the warmed TensorRT engines and
  `8,16` bucket profile dominate what stays loaded.
- The VAE remains the largest model-side slice even at `224`: logged VAE decode
  averaged about `0.122s` versus UNet at about `0.062s`.

Comparison against the closest historical RTX 3090 `256 x 256` runs in
`current_unet_trt_throughput_findings_2026-05-29.md`:

| Concurrency | Historical `256` avg interval | Current `224` avg interval | Historical `256` agg fps | Current `224` agg fps | Aggregate change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | about `0.060s` | `0.054s` | 66.7 fps | 74.1 fps | +11.1% |
| 6 | about `0.091s` | `0.082s` | 65.9 fps | 73.2 fps | +11.1% |
| 8 | about `0.123s` | `0.110s` | 65.0 fps | 72.7 fps | +11.8% |

Read: the `224` ROI experiment gives a real same-GPU throughput signal versus
the historical `256` VAE-only path, but the gain is only about one tenth. That
is not enough to justify the visible mouth/face blur for the normal quality
path.

Against the historical RTX 3090 `256` VAE INT8 + TRT UNet split8 run, the `224`
VAE-only test is roughly in the same aggregate range:

| Concurrency | Historical `256` TRT UNet agg fps | Current `224` PyTorch UNet agg fps | Change |
| ---: | ---: | ---: | ---: |
| 4 | 71.4 fps | 74.1 fps | +3.8% |
| 6 | 72.3 fps | 73.2 fps | +1.2% |
| 8 | 70.8 fps | 72.7 fps | +2.7% |

Those comparisons are useful for direction, but not perfect apples-to-apples:
this workspace did not have a validated local UNet TensorRT split8 artifact, so
the measured 224 run is VAE INT8 plus PyTorch UNet.

Practical comparison read:

- If the production stack has only VAE INT8 plus PyTorch UNet, `224` looks like a
  roughly `10-12%` throughput lever.
- If the production stack also has TRT UNet split8, `224` is only a small
  throughput movement relative to the existing optimized `256` path.
- Since the user-visible quality loss is obvious, the better next target is not
  lower ROI size. It is reducing the same expensive late VAE decoder work while
  preserving the `256 x 256` generated crop contract.

Comparison against the newer Ada docs:

- RTX 5000 Ada VAE INT8 + PyTorch UNet plateaued around `70-72 fps`; this local
  3090 `224` run lands in that same aggregate band.
- RTX 6000 Ada VAE INT8 + PyTorch UNet plateaued around `108-111 fps`.
- RTX 6000 Ada VAE INT8 + TRT UNet split8 plateaued around `112-114 fps`.
- Therefore the `224` ROI change on a 3090 does not approach the current RTX
  6000 Ada optimized path; it is a modest same-GPU throughput tradeoff.

Decision:

- Do not promote `224 x 224` to production quality. The observed blur is too
  visible for a roughly `10-12%` aggregate throughput gain versus the closest
  historical 3090 `256` VAE-only baseline.
- The configurable ROI implementation was reverted after this test. The code
  path is back to the original `256 x 256` generated ROI contract.
- The next same-quality optimization target remains the VAE decoder, especially
  `decoder_up_block_3` and then `decoder_up_block_2`. Reducing ROI confirms that
  spatial VAE decoder work is the right bottleneck, but the quality trade is not
  attractive enough to replace the `256` path.
- The detailed next-bottleneck plan is now tracked separately in
  `docs/next_bottleneck_vae_late_block_plan_2026-06-11.md`.

Artifacts:

```text
tmp/load_tests/load_test_webrtc_3090_224roi_int8_20_20_1_2_3_4_5_6_8streams_8_16_20260611.json
tmp/load_tests/load_test_webrtc_3090_224roi_int8_20_20_1_2_3_4_5_6_8streams_8_16_20260611_detailed.json
tmp/load_tests/load_test_webrtc_3090_224roi_int8_20_20_10_12streams_8_16_20260611.json
tmp/load_tests/load_test_webrtc_3090_224roi_int8_20_20_10_12streams_8_16_20260611_detailed.json
calibration/vae_decoder_224
models/tensorrt/stagewise_int8_224_onnx_qdq_cache
```

## 2026-06-11 Late VAE Up-Block Split Experiment

After rejecting `224 x 224` as too blurry, the next implementation kept the
generated ROI at `256 x 256` and attacked `decoder_up_block_3` directly.

Implemented an opt-in stagewise split:

```text
MUSETALK_TRT_STAGEWISE_SPLIT_UP_BLOCKS=3
```

This allows individual `decoder_up_block_3` ResNet substages to be built as
TensorRT INT8 engines while the rest of the block remains in the stagewise
pipeline. The default runtime remains unchanged unless this env var is set.

Batch-8 same-latent VAE decode results:

| Profile | Avg VAE decode | Decode FPS | Speedup vs safe5 | MAE vs PyTorch | Max abs | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| safe five-stage INT8, full `up_block_3` FP16 | `0.06214s` | `128.75` | baseline | `0.00504` | `0.12866` | current safe reference |
| plus `decoder_up_block_3_resnet_0` INT8 | `0.05829s` | `137.24` | `+6.59%` | `0.00723` | `0.14478` | best candidate |
| plus `decoder_up_block_3_resnet_1` INT8 | `0.05982s` | `133.74` | `+3.88%` | `0.00931` | `0.12109` | lower speedup |
| plus `decoder_up_block_3_resnet_2` INT8 | `0.05994s` | `133.48` | `+3.67%` | `0.01251` | `0.13135` | reject for now |
| plus `resnet_0` and `resnet_1` INT8 | `0.05561s` | `143.86` | `+11.74%` | `0.01125` | `0.13965` | quality risk |

WebRTC load test for the best candidate, VAE INT8 plus PyTorch UNet, batch `8`:

| Concurrency | Completed | Failed | Avg interval | Approx aggregate FPS | Max interval | Read |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 4 | 0 | `0.056s` | `71.4` | `0.184s` | completes, not strict smooth |
| 6 | 6 | 0 | `0.083s` | `72.3` | `0.430s` | plateau |
| 8 | 8 | 0 | `0.113s` | `70.8` | `1.215s` | saturated / tail-heavy |

Quality read:

- `decoder_up_block_3_resnet_0` does not create the obvious softness/blurring
  from the `224` ROI experiment.
- The observed differences are subtler texture and tone changes, especially near
  lips, teeth/inner-mouth, eyes, and skin texture.
- The WebRTC side-by-side was not frame-aligned, so the crop contact sheet and
  same-latent VAE comparison are the more reliable quality artifacts.

Decision:

- Do not promote this candidate yet.
- Keep it as the next candidate to validate at batch `16` and on more avatars.
- The WebRTC plateau did not clearly move despite the isolated `+6.59%` VAE
  speedup, so scheduler/tail behavior and the PyTorch UNet slice still matter.
- More aggressive `resnet_0 + resnet_1` INT8 may become useful later, but only
  after stronger visual validation.

Detailed canonical notes and artifact paths are in:

```text
docs/next_bottleneck_vae_late_block_plan_2026-06-11.md
```
