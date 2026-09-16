# Same-avatar three-way talking-head comparison

This directory contains the reproducible comparison recorded on 2026-09-15
on an NVIDIA GeForce RTX 4070 SUPER with 12,282 MiB visible VRAM, compute
capability 8.9, and driver 595.84.

## Video

Watch [three-way-same-avatar.mp4](three-way-same-avatar.mp4). It is a
10-second, 25 FPS side-by-side video with shared AAC audio. All panels use the
same `shared.png` portrait and `audio.wav` input.

The recorded runs were separate and are aligned for visual comparison; the
video does not claim that all three models rendered simultaneously on the GPU.

Detailed measurements, TensorRT repair results, runtime versions, evidence,
and limitations are in [REPORT.md](REPORT.md).

## Ditto interface classification

The Ditto result is labeled **offline** because it was run through the
repository's `inference.py`, which imports `stream_pipeline_offline.StreamSDK`
and uses `checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl`. That path reads the
complete audio, generates all frames, closes the writer, and muxes the audio
into the finished MP4.

The checkout also contains `stream_pipeline_online.py` and
`v0.4_hubert_cfg_trt_online.pkl`, but it does not provide a native WebRTC
server/client adapter. A real Ditto WebRTC result would require wiring that
online pipeline to a WebRTC media track and measuring transport startup,
pacing, stalls, and received FPS. The current Ditto number is therefore an
offline end-to-end throughput measurement, including video writing and audio
muxing, with model loading measured separately.
