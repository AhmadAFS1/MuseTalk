# WebRTC 15fps vs 20fps Comparison - Recent Avatar

Date: 2026-07-03

## Test Subject

- Avatar: `avatar_5a584c8b-512f-4f70-848c-a3d1efbb988f_1782690149`
- Audio: `data/audio/ai-assistant.mpga`
- Transport: WebRTC
- Single-session batch size: `8`
- C4 load-test batch size: `8`
- Server: local API at `http://127.0.0.1:8000`

## Review Clips

- `recent_avatar_15_vs_20_webrtc_live_window.mp4`
  - Best clip for visual review.
  - Trims the initial idle/prebuffer region and shows the live section side by side.
- `recent_avatar_15_vs_20_webrtc_side_by_side.mp4`
  - Rawer side-by-side including different idle lead-in lengths.
- `recent_avatar_15fps_webrtc.mp4`
- `recent_avatar_20fps_webrtc.mp4`

## Single-Session Result

| FPS | Live frames | Video stalls | Notes |
| ---: | ---: | ---: | --- |
| 15 | `266` | `0` | Completed cleanly |
| 20 | `355` | `0` | Completed cleanly |

15fps generated about 25% fewer live frames for the same audio, as expected.

## C4 Load Result

| FPS | Completed | Avg live-ready | Avg frame interval | Max frame interval | Wall time |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 15 | `4/4` | `3.964s` | `0.094s` | `0.937s` | `31.3s` |
| 20 | `4/4` | `4.481s` | `0.091s` | `1.082s` | `38.2s` |

Per-session strict video stall seconds were lower at 15fps:

- 15fps: roughly `6.2-7.3s` per session
- 20fps: roughly `13.6-14.9s` per session

## Read

Dropping from 20fps to 15fps materially reduces generated-frame work and reduced
stall time in this C4 run. It did not make this server/profile perfectly smooth
at C4 by itself, because both runs still showed tail frame stalls. Treat 15fps
as a useful capacity lever, not as the complete throughput fix.

## Artifacts

- `load_test_webrtc_recent_avatar_c4_15fps.json`
- `load_test_webrtc_recent_avatar_c4_15fps_detailed.json`
- `load_test_webrtc_recent_avatar_c4_20fps.json`
- `load_test_webrtc_recent_avatar_c4_20fps_detailed.json`
