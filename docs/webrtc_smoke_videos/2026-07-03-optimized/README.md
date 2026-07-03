# Optimized WebRTC Smoke Video, 2026-07-03

This folder contains a live WebRTC recording after commit
`f139807 Optimize WebRTC handoff and compose path`.

## Artifact

- `recent_avatar_optimized_webrtc_20fps.mp4`

## Run

- Avatar: `avatar_5a584c8b-512f-4f70-848c-a3d1efbb988f_1782690149`
- Audio: `data/audio/ai-assistant.mpga`
- Playback FPS: `20`
- MuseTalk generation FPS: `20`
- Batch size: `8`
- Segment duration: `1`
- Output: live WebRTC recording through `scripts/record_webrtc_session.py`

## Video Probe

- Resolution: `720x1280`
- Codec: `mpeg4`
- Frame rate: `20 fps`
- Frames written: `392`
- Duration: `19.6s`
- Size: `4.3 MB`

## Server-Side Stats

- WebRTC scheduler status: `completed`
- Generated/source frames: `355`
- Frames played: `355`
- Frames dropped: `0`
- Queue underruns: `0`
- Strict video stalls: `0`
- Strict video stall seconds: `0.0`
- GPU batches: `23`
- Batches pushed: `23`
- Avg GPU batch: `0.251s`
- Avg UNet: `0.090s`
- Avg VAE: `0.158s`
- Avg compose: `0.077s`
- Avg callback: `0.006s`

