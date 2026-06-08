# RTX 6000 Ada WebRTC Load Test - VAE INT8 + PyTorch UNet - 2026-06-08

This report is a VAE INT8 bucket test, not the fully optimized
`VAE INT8 + TRT UNet split8` path. The distinction matters for capacity claims:
the server log for this run showed `UNet backend: PyTorch`, and the local
workspace did not contain a validated batch-8 UNet TensorRT artifact.

## Configuration

- GPU: NVIDIA RTX 6000 Ada Generation, 49140 MiB
- Server profile: `throughput_record`
- WebRTC target: `--playback-fps 20 --musetalk-fps 20`
- VAE backend: `tensorrt_stagewise_int8_mixed`
- UNet backend: `PyTorch`
- INT8 frontend: `onnx_qdq`
- INT8 stages: `decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2`
- Fixed buckets / warmup batches: `8,16`
- H.264 encoder: `libx264`
- Avatar: `test_avatar`, prepared from `data/video/chatgpt_moving_vid.mp4`, `bbox_shift=0`
- Audio: `data/audio/ai-assistant.mpga`
- Harness: `load_test_webrtc.py`

Server log evidence:

- `Stagewise TRT warmup complete (batches=[8, 16], total=212.09s)`
- `VAE decode backend active: tensorrt_stagewise_int8_mixed`
- `UNet backend: PyTorch`
- `HLS GPU scheduler started (max_combined_batch_size=16, fixed_batch_sizes=[8, 16]...)`

Local artifact check:

- `models/tensorrt` contained the VAE INT8 ONNX/QDQ cache and plan files.
- No `unet_trt.ts` / `unet_trt_meta.json` artifact was present under the repo.
- Therefore this run cannot be used as the RTX 6000 Ada result for
  `VAE INT8 + TRT UNet split8`.

## Current INT8 Results

Smooth 20 fps means an average frame interval near `0.050s` and no meaningful throttling. The harness warns when max frame interval exceeds `0.100s`.

| Concurrency | Completed | Failed | Avg live-ready | Avg interval | Max interval | Aggregate fps | Peak VRAM | Smooth at 20 fps |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| 4 | 4 | 0 | 9.874s | 0.051s | 0.082s | 78.4 | 17,920 MB | Yes |
| 8 | 8 | 0 | 3.555s | 0.072s | 0.660s | 111.1 | 17,922 MB | No |
| 10 | 10 | 0 | 4.170s | 0.091s | 1.301s | 109.9 | 17,922 MB | No |
| 12 | 12 | 0 | 5.089s | 0.109s | 1.499s | 110.1 | 17,922 MB | No |
| 15 | 15 | 0 | 6.484s | 0.137s | 2.208s | 109.5 | 17,922 MB | No |
| 16 | 16 | 0 | 6.968s | 0.147s | 2.405s | 108.8 | 17,922 MB | No |
| 20 | 20 | 0 | 9.350s | 0.185s | 3.042s | 108.1 | 18,595 MB | No |

## Prior RTX 6000 Ada Baseline Comparison

Against `load_test_webrtc_rtx6000ada_20_20_4_8_12_16streams_8_16_libx264_20260524.json`:

| Concurrency | Prior avg interval | Current avg interval | Prior aggregate fps | Current aggregate fps | Aggregate change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.051s | 0.051s | 78.4 | 78.4 | 0.0% |
| 8 | 0.105s | 0.072s | 76.2 | 111.1 | +45.8% |
| 12 | 0.161s | 0.109s | 74.5 | 110.1 | +47.8% |
| 16 | 0.219s | 0.147s | 73.1 | 108.8 | +48.8% |

High-concurrency comparison against the prior best `8,16,24` bucket report:

| Concurrency | Prior avg interval | Current avg interval | Prior aggregate fps | Current aggregate fps | Aggregate change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.123s | 0.091s | 81.3 | 109.9 | +35.2% |
| 15 | 0.195s | 0.137s | 76.9 | 109.5 | +42.4% |
| 20 | 0.272s | 0.185s | 73.5 | 108.1 | +47.1% |

Compared with the prior `4,8,16,20` bucket report:

| Concurrency | Prior avg interval | Current avg interval | Prior aggregate fps | Current aggregate fps | Aggregate change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 0.129s | 0.091s | 77.5 | 109.9 | +41.8% |
| 15 | 0.198s | 0.137s | 75.8 | 109.5 | +44.5% |
| 20 | 0.268s | 0.185s | 74.6 | 108.1 | +44.9% |

## Conclusion

Strictly smooth concurrent WebRTC capacity at 20 fps is 4 sessions on this
`VAE INT8 + PyTorch UNet` run. INT8 buckets significantly improved aggregate
throughput at higher concurrency, stabilizing around 108-111 aggregate fps
versus roughly 73-81 aggregate fps in the previous RTX 6000 Ada reports, but
8+ concurrent sessions still exceed the 20 fps per-session frame interval target
and show throttling spikes.

The current INT8 path also uses much less VRAM than the prior reports: around 17.9-18.6 GB peak here versus roughly 24.6 GB for the prior `8,16` run and 43-44 GB for prior larger-bucket high-concurrency runs.

This should be treated as the RTX 6000 Ada VAE INT8 baseline, not the optimized
UNet TensorRT result. The optimized RTX 6000 Ada rerun is documented separately
in `load_test_webrtc_rtx6000ada_int8_trt_unet_split8_20fps_20260608.md`.

For an optimized rerun, the server must activate:

```text
MUSETALK_UNET_BACKEND=trt
MUSETALK_TRT_UNET_ENABLED=1
MUSETALK_TRT_UNET_PATHS=8:path/to/validated/unet_trt.ts
MUSETALK_TRT_FALLBACK=0
```

and the server log must show:

```text
UNet backend active: tensorrt_unet_multi
```

## Artifacts

- `tmp/load_tests/load_test_webrtc_rtx6000ada_int8_5stage_20_20_4_8_12_16streams_8_16_libx264_20260607_valid.json`
- `tmp/load_tests/load_test_webrtc_rtx6000ada_int8_5stage_20_20_4_8_12_16streams_8_16_libx264_20260607_valid_detailed.json`
- `tmp/load_tests/load_test_webrtc_rtx6000ada_int8_5stage_20_20_10_15_20streams_8_16_libx264_20260608_valid.json`
- `tmp/load_tests/load_test_webrtc_rtx6000ada_int8_5stage_20_20_10_15_20streams_8_16_libx264_20260608_valid_detailed.json`

Notes:

- The first 4-stream run includes some cold-cache live-ready cost; subsequent stages had the avatar cached.
- This run intentionally used INT8 `8,16` buckets, per the requested INT8 bucket test. Some older high-concurrency baselines used `8,16,24` or `4,8,16,20`, so the throughput comparison is useful but not perfectly bucket-identical for those rows.
- Any future RTX 6000 Ada table should keep separate columns or sections for
  `VAE INT8 + PyTorch UNet` and `VAE INT8 + TRT UNet split8`; otherwise the
  capacity story gets muddied.
