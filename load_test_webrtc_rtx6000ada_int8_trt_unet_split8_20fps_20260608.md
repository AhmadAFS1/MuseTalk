# RTX 6000 Ada WebRTC Load Test - VAE INT8 + TRT UNet Split8 - 2026-06-08

This is the optimized RTX 6000 Ada rerun for the `20 fps` WebRTC capacity test.
It is the run that should be compared against the earlier
`VAE INT8 + PyTorch UNet` RTX 6000 Ada baseline.

Important precision note: INT8 applies to the five selected VAE decoder stages
and the warmed scheduler buckets. The UNet artifact tested here is FP16
TensorRT, loaded through the split8 runtime. It is not UNet INT8.

## Configuration

- GPU: NVIDIA RTX 6000 Ada Generation, 49140 MiB
- Server profile: `throughput_record`
- WebRTC target: `--playback-fps 20 --musetalk-fps 20`
- VAE backend: `tensorrt_stagewise_int8_mixed`
- UNet backend: `tensorrt_unet_multi`
- UNet runtime shape: static batch-8 TensorRT artifact, split8 for larger turns
- Fixed buckets / warmup batches: `8,16`
- INT8 frontend: `onnx_qdq`
- INT8 VAE stages: `decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2`
- H.264 encoder: `libx264`
- Avatar: `test_avatar`
- Audio: `data/audio/ai-assistant.mpga`
- Harness: `load_test_webrtc.py`

Server log evidence:

```text
Stagewise TRT warmup complete (batches=[8, 16], total=30.40s)
VAE decode backend active: tensorrt_stagewise_int8_mixed
UNet backend active: tensorrt_unet_multi
HLS GPU scheduler started (max_combined_batch_size=16, fixed_batch_sizes=[8, 16]...)
```

UNet TensorRT artifact and validation:

- Capture corpus: `calibration/unet_rtx6000ada_split8_20260608`
- Captures: 64 real scheduler captures, all batch `8`
- Artifact: `models/tensorrt_unet_static_bs8_rtx6000ada_20260608/unet_trt.ts`
- Artifact metadata: `models/tensorrt_unet_static_bs8_rtx6000ada_20260608/unet_trt_meta.json`
- Validation report: `tmp/unet_rtx6000ada_split8_trt_validation_20260608.json`
- Validation result: passed
- Validation summary: `mae_mean=0.001684`, `mae_max=0.001708`, `max_abs_max=0.151123`, `latency_ms_mean=14.987`

## Optimized Results

Smooth 20 fps means an average frame interval near `0.050s` with no meaningful
throttling. The harness warns when max frame interval exceeds `0.100s`.

| Concurrency | Completed | Failed | Avg live-ready | Avg interval | Max interval | Aggregate fps | Peak VRAM | Smooth at 20 fps |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 4 | 4 | 0 | 3.778s | 0.051s | 0.072s | 78.4 | 19,644 MB | Yes |
| 8 | 8 | 0 | 3.296s | 0.070s | 0.661s | 114.3 | 19,644 MB | No |
| 10 | 10 | 0 | 4.019s | 0.089s | 1.050s | 112.4 | 19,644 MB | No |
| 12 | 12 | 0 | 4.891s | 0.105s | 1.297s | 114.3 | 19,644 MB | No |
| 15 | 15 | 0 | 6.629s | 0.134s | 2.016s | 111.9 | 19,644 MB | No |
| 16 | 16 | 0 | 6.933s | 0.143s | 2.252s | 111.9 | 19,644 MB | No |
| 20 | 20 | 0 | 9.041s | 0.179s | 2.843s | 111.7 | 19,644 MB | No |

## Direct Comparison Against VAE INT8 + PyTorch UNet

Both runs used five-stage VAE INT8 and `8,16` fixed buckets. The difference is
the UNet backend:

- baseline: `UNet backend: PyTorch`
- optimized: `UNet backend active: tensorrt_unet_multi`

| Concurrency | PyTorch UNet avg | TRT split8 avg | PyTorch agg fps | TRT split8 agg fps | Aggregate FPS change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.051s | 0.051s | 78.4 | 78.4 | 0.0% |
| 8 | 0.072s | 0.070s | 111.1 | 114.3 | +2.8% |
| 10 | 0.091s | 0.089s | 109.9 | 112.4 | +2.2% |
| 12 | 0.109s | 0.105s | 110.1 | 114.3 | +3.7% |
| 15 | 0.137s | 0.134s | 109.5 | 111.9 | +2.2% |
| 16 | 0.147s | 0.143s | 108.8 | 111.9 | +2.7% |
| 20 | 0.185s | 0.179s | 108.1 | 111.7 | +3.2% |

Tail intervals also improved:

| Concurrency | PyTorch UNet max | TRT split8 max |
| ---: | ---: | ---: |
| 4 | 0.082s | 0.072s |
| 8 | 0.660s | 0.661s |
| 10 | 1.301s | 1.050s |
| 12 | 1.499s | 1.297s |
| 15 | 2.208s | 2.016s |
| 16 | 2.405s | 2.252s |
| 20 | 3.042s | 2.843s |

## `4,8,16` Bucket Follow-Up

After the optimized `8,16` split8 run, the server was restarted with only the
VAE bucket set changed to `4,8,16`. The UNet backend and split8 artifact stayed
the same:

```text
Stagewise TRT warmup complete (batches=[4, 8, 16], total=109.76s)
VAE decode backend active: tensorrt_stagewise_int8_mixed
UNet backend active: tensorrt_unet_multi
HLS GPU scheduler started (max_combined_batch_size=16, fixed_batch_sizes=[4, 8, 16]...)
```

This did not OOM on the RTX 6000 Ada. It did, however, raise resident VRAM from
about `19.6 GB` to about `21.7 GB`.

| Concurrency | `8,16` avg | `4,8,16` avg | `8,16` agg fps | `4,8,16` agg fps | `8,16` max | `4,8,16` max | `4,8,16` peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.051s | 0.051s | 78.4 | 78.4 | 0.072s | 0.081s | 21,728 MB |
| 8 | 0.070s | 0.069s | 114.3 | 115.9 | 0.661s | 0.676s | 21,730 MB |
| 10 | 0.089s | 0.088s | 112.4 | 113.6 | 1.050s | 0.986s | 21,740 MB |
| 12 | 0.105s | 0.105s | 114.3 | 114.3 | 1.297s | 1.236s | 21,736 MB |
| 15 | 0.134s | 0.134s | 111.9 | 111.9 | 2.016s | 2.065s | 21,740 MB |
| 16 | 0.143s | 0.143s | 111.9 | 111.9 | 2.252s | 2.195s | 21,740 MB |
| 20 | 0.179s | 0.180s | 111.7 | 111.1 | 2.843s | 2.840s | 21,740 MB |

Read: `4,8,16` does not materially increase FPS on this optimized RTX 6000 Ada
WebRTC path. It slightly improves a few startup/tail points, slightly regresses
others, and does not change the strict smooth `20 fps` capacity. The cleaner
serving profile remains `8,16`.

The existing markdowns already warned against aggressive larger warmup profiles:
the old RTX 6000 Ada `8,16,32` startup failed near the memory wall when TensorRT
could not create an execution context, and the V100 32 GB docs record explicit
batch-32 CUDA OOM during `4,8,16,32` warmup. This `4,8,16` test is below that
risk boundary on the 48 GB RTX 6000 Ada, but the extra batch-4 engine still did
not buy meaningful throughput.

## Conclusion

The optimized backend is active and measured: five-stage VAE INT8 buckets plus
TRT UNet split8. It improves the saturated aggregate plateau from roughly
`108-111 fps` to roughly `112-114 fps`, with better startup and tail behavior.

Strict smooth concurrent WebRTC capacity at `20 fps` remains `4` sessions. C8
and above complete with zero request failures, but they exceed the per-session
20 fps frame interval target and show throttling spikes.

This means the RTX 6000 Ada improvement from adding TRT UNet split8 is real but
incremental. The large jump versus the older RTX 6000 Ada markdown results came
from the five-stage VAE INT8 path; the split8 UNet artifact adds another
approximately `2-4%` aggregate throughput on this workload.

The follow-up `4,8,16` bucket test does not change the recommendation. It uses
more VRAM and gives at most run-noise-level throughput changes, so `8,16` remains
the preferred optimized serving profile for this RTX 6000 Ada path.

## Artifacts

- `tmp/load_tests/load_test_webrtc_rtx6000ada_int8_5stage_trt_unet_split8_20_20_4_8_12_16streams_8_16_libx264_20260608.json`
- `tmp/load_tests/load_test_webrtc_rtx6000ada_int8_5stage_trt_unet_split8_20_20_4_8_12_16streams_8_16_libx264_20260608_detailed.json`
- `tmp/load_tests/load_test_webrtc_rtx6000ada_int8_5stage_trt_unet_split8_20_20_10_15_20streams_8_16_libx264_20260608.json`
- `tmp/load_tests/load_test_webrtc_rtx6000ada_int8_5stage_trt_unet_split8_20_20_10_15_20streams_8_16_libx264_20260608_detailed.json`
- `tmp/load_tests/load_test_webrtc_rtx6000ada_int8_5stage_trt_unet_split8_20_20_4_8_12_16streams_4_8_16_libx264_20260608.json`
- `tmp/load_tests/load_test_webrtc_rtx6000ada_int8_5stage_trt_unet_split8_20_20_4_8_12_16streams_4_8_16_libx264_20260608_detailed.json`
- `tmp/load_tests/load_test_webrtc_rtx6000ada_int8_5stage_trt_unet_split8_20_20_10_15_20streams_4_8_16_libx264_20260608.json`
- `tmp/load_tests/load_test_webrtc_rtx6000ada_int8_5stage_trt_unet_split8_20_20_10_15_20streams_4_8_16_libx264_20260608_detailed.json`
- `tmp/unet_rtx6000ada_split8_trt_validation_20260608.json`
- `models/tensorrt_unet_static_bs8_rtx6000ada_20260608/unet_trt.ts`
