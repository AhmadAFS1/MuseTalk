# WebRTC Load Test Findings - 2026-06-07

This note captures the RTX 5000 Ada WebRTC load-test thread from 2026-06-07:
the accidental FP16 run, the corrected INT8 `8,16` run, the follow-up INT8
`4,8,16` bucket run, and the optimization read from the existing markdowns.

The workload is MuseTalk lipsync serving for mobile-app AI talking humans. The
user-facing target is smooth concurrent WebRTC playback at `20 fps`.

## Short Answer

- The corrected INT8 run lifted the saturated RTX 5000 Ada WebRTC aggregate
  plateau from about `60-61 fps` to about `70-71 fps`.
- Smooth strict `20 fps` capacity is still `3` concurrent streams. C4 and above
  complete, but they record strict video stalls and miss the required aggregate
  target.
- INT8 materially reduced VRAM: FP16 peaked around `24.4 GB`; INT8 `8,16`
  peaked around `17.7 GB`.
- The saved VRAM gave room to warm an additional batch-4 VAE engine. The
  `4,8,16` run peaked around `19.8 GB`, but it did not improve strict capacity.
- The current live path is not the most optimized path documented in the repo.
  The faster documented path is five-stage VAE INT8 plus TensorRT UNet split8,
  but the required UNet TensorRT artifact is missing on this RTX 5000 Ada node.

## Test Shape

All 2026-06-07 RTX 5000 Ada tests used the same WebRTC request shape:

```text
GPU: NVIDIA RTX 5000 Ada Generation, about 32 GB visible VRAM
base URL: http://127.0.0.1:8000
avatar: test_avatar_2
audio: data/audio/ai-assistant.mpga
ramp: 1,2,3,4,5,6,8
playback_fps: 20
musetalk_fps: 20
request batch_size: 8
hold_seconds: 10
segment_duration: 1.0
same-host WebRTC load-test client
```

The aggregate FPS number is calculated as:

```text
aggregate_fps = concurrency / avg_frame_interval_s
```

For strict smooth `20 fps`, the aggregate targets are:

| Streams | Required aggregate FPS |
| ---: | ---: |
| 4 | `80` |
| 6 | `120` |
| 8 | `160` |
| 10 | `200` |

That is why a `70-72 fps` aggregate plateau can complete many sessions but still
cannot honestly be called smooth `20 fps` at C4/C6/C8.

## Profiles Tested

### Accidental FP16 `8,16`

This was the earlier RTX 5000 Ada run that initially looked like the INT8 result
but was actually running the FP16 stagewise backend.

Report files:

```text
tmp/load_tests/load_test_webrtc_rtx5000ada_20_20_1_2_3_4_5_6_8streams_8_16_20260607.json
tmp/load_tests/load_test_webrtc_rtx5000ada_20_20_1_2_3_4_5_6_8streams_8_16_20260607_detailed.json
```

### Corrected INT8 `8,16`

The server was restarted with the safe five-stage ONNX/QDQ mixed INT8 VAE
decoder:

```text
active VAE backend: tensorrt_stagewise_int8_mixed
UNet backend: PyTorch
HLS_SCHEDULER_MAX_BATCH=16
HLS_SCHEDULER_FIXED_BATCH_SIZES=8,16
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8,16
MUSETALK_TRT_STAGEWISE_PRECISION=int8_mixed
MUSETALK_TRT_STAGEWISE_INT8_FRONTEND=onnx_qdq
MUSETALK_TRT_STAGEWISE_INT8_STAGES=decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2
MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_DIR=./calibration/vae_decoder
MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_ALGO=minmax
MUSETALK_TRT_STAGEWISE_INT8_CACHE_DIR=./models/tensorrt/stagewise_int8_onnx_qdq_cache
```

Startup proof from logs:

```text
batch 8 ready in 157.66s
batch 16 ready in 88.86s
total stagewise warmup 246.53s
```

Report files:

```text
tmp/load_tests/load_test_webrtc_rtx5000ada_int8_20_20_1_2_3_4_5_6_8streams_8_16_20260607.json
tmp/load_tests/load_test_webrtc_rtx5000ada_int8_20_20_1_2_3_4_5_6_8streams_8_16_20260607_detailed.json
```

### Corrected INT8 `4,8,16`

After INT8 reduced VRAM, the server was restarted with an additional warmed
batch-4 VAE engine:

```text
active VAE backend: tensorrt_stagewise_int8_mixed
UNet backend: PyTorch
VAE postprocess: MUSETALK_VAE_FAST_POSTPROCESS defaults on
HLS_SCHEDULER_MAX_BATCH=16
HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=4,8,16
MUSETALK_TRT_STAGEWISE_PRECISION=int8_mixed
MUSETALK_TRT_STAGEWISE_INT8_FRONTEND=onnx_qdq
MUSETALK_TRT_STAGEWISE_INT8_STAGES=decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2
MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_DIR=./calibration/vae_decoder
MUSETALK_TRT_STAGEWISE_INT8_CALIBRATION_ALGO=minmax
MUSETALK_TRT_STAGEWISE_INT8_CACHE_DIR=./models/tensorrt/stagewise_int8_onnx_qdq_cache
```

Startup proof from logs:

```text
batch 4 ready in 114.81s
batch 8 ready in 21.25s
batch 16 ready in 28.17s
total stagewise warmup 164.23s
```

Report files:

```text
tmp/load_tests/load_test_webrtc_rtx5000ada_int8_20_20_1_2_3_4_5_6_8streams_4_8_16_20260607.json
tmp/load_tests/load_test_webrtc_rtx5000ada_int8_20_20_1_2_3_4_5_6_8streams_4_8_16_20260607_detailed.json
```

## Result Tables

### FP16 `8,16`

| Streams | Completed | Avg interval | Aggregate FPS | Max interval | Strict stalls | Strict stall seconds | Peak VRAM | Read |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `1/1` | `0.051s` | `19.6` | `0.107s` | `0` | `0.0s` | `24158 MB` | clean |
| 2 | `2/2` | `0.051s` | `39.2` | `0.112s` | `0` | `0.0s` | `24318 MB` | clean |
| 3 | `3/3` | `0.051s` | `58.8` | `0.089s` | `0` | `0.0s` | `24318 MB` | clean |
| 4 | `4/4` | `0.065s` | `61.5` | `0.562s` | `84` | `19.4s` | `24386 MB` | saturated |
| 5 | `5/5` | `0.082s` | `61.0` | `0.900s` | `128` | `54.8s` | `24386 MB` | saturated |
| 6 | `6/6` | `0.098s` | `61.2` | `1.151s` | `160` | `98.5s` | `24396 MB` | saturated |
| 8 | `8/8` | `0.132s` | `60.6` | `1.862s` | `220` | `229.3s` | `24396 MB` | saturated |

### INT8 `8,16`

| Streams | Completed | Avg interval | Aggregate FPS | Max interval | Strict stalls | Strict stall seconds | Peak VRAM | Read |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `1/1` | `0.051s` | `19.6` | `0.101s` | `0` | `0.0s` | `17636 MB` | clean |
| 2 | `2/2` | `0.051s` | `39.2` | `0.113s` | `0` | `0.0s` | `17636 MB` | clean |
| 3 | `3/3` | `0.051s` | `58.8` | `0.094s` | `0` | `0.0s` | `17646 MB` | clean |
| 4 | `4/4` | `0.056s` | `71.4` | `0.479s` | `44` | `6.2s` | `17646 MB` | better, not strict 20 fps |
| 5 | `5/5` | `0.071s` | `70.4` | `0.735s` | `115` | `34.6s` | `17668 MB` | saturated |
| 6 | `6/6` | `0.084s` | `71.4` | `0.989s` | `144` | `70.4s` | `17668 MB` | saturated |
| 8 | `8/8` | `0.112s` | `71.4` | `1.493s` | `205` | `174.4s` | `17668 MB` | saturated |

### INT8 `4,8,16`

| Streams | Completed | Avg interval | Aggregate FPS | Max interval | Strict stalls | Strict stall seconds | Peak VRAM | Read |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | `1/1` | `0.051s` | `19.6` | `0.094s` | `0` | `0.0s` | `19752 MB` | clean |
| 2 | `2/2` | `0.051s` | `39.2` | `0.118s` | `0` | `0.0s` | `19752 MB` | clean |
| 3 | `3/3` | `0.051s` | `58.8` | `0.089s` | `0` | `0.0s` | `19762 MB` | clean |
| 4 | `4/4` | `0.056s` | `71.4` | `0.400s` | `41` | `6.5s` | `19776 MB` | better tail than `8,16`, still not strict |
| 5 | `5/5` | `0.071s` | `70.4` | `0.757s` | `114` | `34.9s` | `19776 MB` | saturated |
| 6 | `6/6` | `0.083s` | `72.3` | `0.969s` | `134` | `68.6s` | `19776 MB` | saturated |
| 8 | `8/8` | `0.124s` | `64.5` | `30.080s` | `204` | `207.2s` | `19786 MB` | saturated with one large tail stall |

## What The Numbers Mean

### Aggregate FPS

The aggregate FPS number for the corrected INT8 `8,16` run is about:

```text
C4: 4 / 0.056s = 71.4 fps
C5: 5 / 0.071s = 70.4 fps
C6: 6 / 0.084s = 71.4 fps
C8: 8 / 0.112s = 71.4 fps
```

That is the real saturated aggregate throughput. It is enough for roughly three
clean `20 fps` streams, but not enough for four strict streams because C4 needs
`80 aggregate fps`.

### Why The Earlier Run Looked Capped Around 60 FPS

The FP16 stagewise run flattened at about `60-61 aggregate fps` from C4 through
C8. That was not a WebRTC signaling cap or an encoder FPS cap. It was the shared
generation loop hitting its throughput ceiling:

```text
UNet -> VAE decode -> compose -> WebRTC frame delivery
```

Once that loop is saturated, adding streams divides the same generated-frame
budget across more sessions. The symptoms line up with model/scheduler
saturation:

- C1-C3 have zero strict stalls.
- C4 and above record strict stalls and queue underruns.
- FP16 peak VRAM is high, around `24.4 GB`, but the issue is not simply "ran out
  of VRAM"; the stream cadence slows before memory exhaustion.
- INT8 changes the plateau from `60-61 fps` to `70-71 fps`, which means the
  compute path moved, not the WebRTC transport.

### What INT8 Actually Improved

The corrected VAE INT8 `8,16` run improved the saturated plateau by roughly
`15-17%` over the accidental FP16 run:

| Streams | FP16 agg FPS | INT8 `8,16` agg FPS | Delta |
| ---: | ---: | ---: | ---: |
| 4 | `61.5` | `71.4` | `+16.3%` |
| 5 | `61.0` | `70.4` | `+15.4%` |
| 6 | `61.2` | `71.4` | `+17.5%` |
| 8 | `60.6` | `71.4` | `+18.0%` |

It also reduced peak VRAM from about `24.4 GB` to about `17.7 GB`, giving roughly
`6.7 GB` of additional headroom.

The strict capacity still stayed at three streams because C4 still missed the
`80 fps` aggregate target and had strict video stalls.

### What The Extra VRAM Means

The VRAM drop is useful, but it does not automatically create more smooth
streams.

Extra VRAM helps with:

- keeping more TensorRT engines warm;
- adding the missing UNet TensorRT runtime artifact;
- keeping more avatars or larger capture/export jobs resident;
- experimenting with larger or more exact scheduler buckets;
- avoiding OOM during warmup.

Extra VRAM does not help if:

- the active bottleneck is GPU math latency;
- the scheduler is serializing work through the same generation turn;
- the added warmed engine does not match a common actual batch shape;
- the C4/C6/C8 targets require more aggregate FPS than the generation path can
  produce.

The `4,8,16` run proved this directly. It consumed about another `2.1 GB` versus
INT8 `8,16`, but strict capacity stayed at C3.

### Should `4,8,16` Replace `8,16`?

Not by default.

`4,8,16` slightly improved the C4 max interval and reduced C4 strict stalls from
`44` to `41`, but it did not change the C4 aggregate FPS or strict-capacity
answer. It also had a bad C8 tail stall in this run and used more resident VRAM.

Recommended read:

- Use `8,16` as the cleaner INT8 baseline for apples-to-apples throughput work.
- Keep `4,8,16` as an experiment if the live workload frequently forms actual
  batch-4 turns and the goal is tail-latency tuning rather than maximum headroom.
- Do not expect added buckets alone to produce C4/C6/C8 smooth `20 fps`.

## Are We On The Most Optimized Path?

The current RTX 5000 Ada path is the best validated VAE INT8 path available on
this node, but it is not the most optimized WebRTC path documented in the repo.

### Current Available Path

Current available RTX 5000 Ada runtime after this chat:

```text
VAE: five-stage ONNX/QDQ mixed INT8 TensorRT stagewise
VAE fast postprocess: default on in musetalk/models/vae.py
UNet: PyTorch
scheduler buckets: 4,8,16 currently running, 8,16 is cleaner baseline
```

The selected safe INT8 VAE stages are:

```text
decoder_pre
decoder_mid_block
decoder_up_block_0
decoder_up_block_1
decoder_up_block_2
```

Do not blindly add `decoder_up_block_3` to live INT8. Earlier quality-gated
experiments found visible color/texture regressions, and that stage should stay
FP16 unless a narrower, quality-gated quantization strategy is built.

### Faster Documented Path Not Present On This Node

The markdowns document a faster measured path from 2026-05-29:

```text
VAE: five-stage INT8
VAE postprocess: fast tensor-side conversion
UNet: static batch-8 TensorRT split runtime
```

The runtime flags were:

```text
MUSETALK_UNET_BACKEND=trt
MUSETALK_TRT_UNET_ENABLED=1
MUSETALK_TRT_UNET_PATHS=8:models/tensorrt_unet_static_bs8_20260529/unet_trt.ts
MUSETALK_TRT_FALLBACK=0
```

On the RTX 3090 notes, that path improved end-to-end WebRTC aggregate FPS by
about `7-10%` versus VAE INT8 plus PyTorch UNet:

| Streams | VAE INT8 + PyTorch UNet | VAE INT8 + TRT UNet split8 |
| ---: | ---: | ---: |
| 4 | `66.7 fps` | `71.4 fps` |
| 6 | `65.9 fps` | `72.3 fps` |
| 8 | `65.0 fps` | `70.8 fps` |

Fast VAE postprocess later nudged the C8 point from `70.8` to about `72.1 fps`
and reduced the postprocess slice from about `6.9 ms` to about `0.9 ms` per VAE
decode call. It was a worthwhile cleanup but not a breakthrough by itself.

The current RTX 5000 Ada workspace does not have the required UNet TensorRT
artifact. The artifact check found only:

```text
models/musetalkV15/unet.pth
```

The missing expected artifact is:

```text
models/tensorrt_unet_static_bs8_20260529/unet_trt.ts
```

The missing capture directory is:

```text
calibration/unet_static_8_16_20260529_1545
```

So the current RTX 5000 Ada tests are VAE INT8 plus PyTorch UNet. They are not a
test of the documented TRT UNet split8 path.

## Hardware Read

### RTX 5000 Ada Current Result

The RTX 5000 Ada result is now:

```text
strict smooth 20 fps capacity: 3 streams
saturated aggregate throughput: about 70-72 fps with INT8
best current peak VRAM for INT8 8,16: about 17.7 GB
```

This card has enough VRAM headroom for the missing UNet TensorRT artifact and
additional experiments. The throughput gap is now compute/tail-latency, not raw
VRAM capacity.

### RTX 3090 Historical Result

The strongest RTX 3090 markdown result used five-stage VAE INT8 plus static
batch-8 TRT UNet split runtime. It reached about `71-72 aggregate fps`, similar
to the current RTX 5000 Ada VAE-only INT8 plateau.

Important caveat: the RTX 3090 data comes from a different date, host, artifact
state, and power environment. The useful conclusion is not "RTX 3090 equals RTX
5000 Ada"; it is that the current code path is already close to the known
single-GPU plateau unless the missing UNet artifact, VAE tensor decode, or
scheduling overlap moves.

### V100 Historical Result

The V100 32 GB tests showed the same lesson about VRAM:

- Bigger warm profiles could keep more streams alive.
- Batch-24 and batch-28 completed high-concurrency runs.
- They still did not produce smooth `20 fps` at high concurrency.

The best V100 high-concurrency numbers in the older docs were completion-oriented
results, not smooth playback results. Extra memory helped survivability and
experimentation, but not enough aggregate generated FPS.

### RTX 6000 Ada Read

The repo now has two local RTX 6000 Ada WebRTC reports from 2026-06-08:

- `load_test_webrtc_rtx6000ada_int8_5stage_20fps_20260608.md`
- `load_test_webrtc_rtx6000ada_int8_trt_unet_split8_20fps_20260608.md`

The first report is the corrected five-stage VAE INT8 `8,16` bucket baseline,
but it is not the fully optimized `VAE INT8 + TRT UNet split8` path:

```text
active VAE backend: tensorrt_stagewise_int8_mixed
UNet backend: PyTorch
fixed buckets: 8,16
```

Artifact check on the RTX 6000 Ada workspace found the VAE INT8 ONNX/QDQ cache
and plan files under `models/tensorrt`, but no validated `unet_trt.ts` /
`unet_trt_meta.json` artifact. The server log also showed
`UNet backend: PyTorch`, not `UNet backend active: tensorrt_unet_multi`.

The second report is the optimized rerun. It generated and validated a static
batch-8 UNet TensorRT artifact, then loaded it through the split8 runtime:

```text
active VAE backend: tensorrt_stagewise_int8_mixed
UNet backend active: tensorrt_unet_multi
fixed buckets: 8,16
UNet artifact: models/tensorrt_unet_static_bs8_rtx6000ada_20260608/unet_trt.ts
```

Important precision note: INT8 applies to the selected VAE decoder stages and
the scheduler buckets. The UNet artifact in the optimized run is FP16 TensorRT,
not UNet INT8.

RTX 6000 Ada VAE INT8 + PyTorch UNet result:

| Streams | Completed | Avg interval | Aggregate FPS | Max interval | Peak VRAM | Read |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | `4/4` | `0.051s` | `78.4` | `0.082s` | `17920 MB` | strict/near-target |
| 8 | `8/8` | `0.072s` | `111.1` | `0.660s` | `17922 MB` | completes, not strict 20 fps |
| 12 | `12/12` | `0.109s` | `110.1` | `1.499s` | `17922 MB` | saturated |
| 16 | `16/16` | `0.147s` | `108.8` | `2.405s` | `17922 MB` | saturated |
| 20 | `20/20` | `0.185s` | `108.1` | `3.042s` | `18595 MB` | completion/stress only |

RTX 6000 Ada VAE INT8 + TRT UNet split8 result:

| Streams | Completed | Avg interval | Aggregate FPS | Max interval | Peak VRAM | Read |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 4 | `4/4` | `0.051s` | `78.4` | `0.072s` | `19644 MB` | strict/near-target |
| 8 | `8/8` | `0.070s` | `114.3` | `0.661s` | `19644 MB` | completes, not strict 20 fps |
| 10 | `10/10` | `0.089s` | `112.4` | `1.050s` | `19644 MB` | saturated |
| 12 | `12/12` | `0.105s` | `114.3` | `1.297s` | `19644 MB` | saturated |
| 15 | `15/15` | `0.134s` | `111.9` | `2.016s` | `19644 MB` | saturated |
| 16 | `16/16` | `0.143s` | `111.9` | `2.252s` | `19644 MB` | saturated |
| 20 | `20/20` | `0.179s` | `111.7` | `2.843s` | `19644 MB` | completion/stress only |

Operational read:

- Strict smooth `20 fps` capacity is still `4` concurrent streams on both RTX
  6000 Ada runs.
- The saturated aggregate plateau is much better than the older RTX 6000 Ada
  reports: roughly `108-111 fps` for VAE INT8 + PyTorch UNet and roughly
  `112-114 fps` for VAE INT8 + TRT UNet split8, versus about `73-81 fps` before.
- The split8 UNet artifact is a real improvement, but it is incremental on this
  workload: about `2-4%` aggregate FPS versus the PyTorch UNet VAE INT8 baseline.
- VRAM remains far below the older large-bucket RTX 6000 Ada runs. The PyTorch
  UNet baseline peaked around `17.9-18.6 GB`; the split8 UNet runtime peaked
  around `19.6 GB`; older larger-bucket reports used roughly `43-44 GB`.
- Do not mix these two result families. `UNet backend: PyTorch` is the VAE INT8
  baseline; `UNet backend active: tensorrt_unet_multi` is the optimized split8
  UNet result.

## Next Optimization Runbook

The next true "most optimized available path" test on RTX 5000 Ada is to rebuild
the missing TensorRT UNet split8 artifact locally, validate it, then rerun
WebRTC.

### 1. Capture Real UNet Batches

Restart with the current five-stage VAE INT8 profile and enable UNet capture:

```text
MUSETALK_UNET_CALIBRATION_CAPTURE=1
MUSETALK_UNET_CALIBRATION_DIR=./calibration/unet
MUSETALK_UNET_CALIBRATION_MAX_BATCHES=64
```

Run a short representative WebRTC session so the scheduler writes
`unet_io_*.pt` files containing:

```text
latent_batch
audio_feature_batch
timesteps
pred_latents
actual batch size
padded batch size
avatar/request metadata
```

### 2. Validate PyTorch Reference Captures

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/validate_unet_backend.py \
  --capture-dir ./calibration/unet \
  --backend pytorch \
  --limit 8 \
  --report-path tmp/unet_pytorch_reference_validation.json
```

### 3. Export And Validate FP16 UNet TensorRT

Start with FP16 UNet TensorRT. Do not start with UNet INT8.

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

If a separate validation pass is needed:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/validate_unet_backend.py \
  --capture-dir ./calibration/unet \
  --backend trt \
  --trt-path ./models/tensorrt/unet_trt.ts \
  --limit 8 \
  --report-path tmp/unet_trt_fp16_validation.json
```

### 4. Enable TRT UNet Only After Validation

For the split8 style path, use the validated batch-8 artifact and keep fallback
disabled so a failed TensorRT load does not silently become a PyTorch run:

```text
MUSETALK_UNET_BACKEND=trt
MUSETALK_TRT_UNET_ENABLED=1
MUSETALK_TRT_UNET_PATHS=8:path/to/validated/unet_trt.ts
MUSETALK_TRT_FALLBACK=0
```

Then run:

1. one lipsync smoke test;
2. C4/C6/C8 WebRTC load test;
3. detailed strict stall analysis;
4. visual review for mouth quality and temporal jitter.

## Recommended Baseline Going Forward

For future apples-to-apples WebRTC throughput testing on this node, use:

```text
VAE backend: tensorrt_stagewise_int8_mixed
INT8 frontend: onnx_qdq
INT8 stages: decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2
scheduler buckets: 8,16
request batch_size: 8
playback_fps: 20
musetalk_fps: 20
avatar: test_avatar_2
audio: data/audio/ai-assistant.mpga
```

Use `4,8,16` only as a separate tail-latency experiment, not as the default
throughput baseline.

The success bar should remain strict:

- C4 needs about `80 aggregate fps` with zero strict stalls.
- C6 needs about `120 aggregate fps`.
- C8 needs about `160 aggregate fps`.
- "Completed" is not enough; strict stalls and max frame intervals must stay low.
