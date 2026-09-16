# Ditto `fade_type=s` endpoint test

GPU: NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM, compute capability 8.9, driver 595.84. Tested 2026-09-15.

## Configuration

- Same `shared.png` portrait and `audio.wav` used by the comparison.
- Ditto TensorRT offline pipeline, 25 FPS, 250 output frames.
- `fade_type="s"`.
- `fade_out=15`, giving a 0.6 second transition.
- `fade_out_keys=("exp", "pitch", "yaw", "roll", "t")`, so expression, head pose, and translation all return toward the source pose.

## Result

| Measurement | Original Ditto run | `fade_type=s` |
|---|---:|---:|
| Output frames | 250 | 250 |
| End-to-end generation FPS, model load excluded | 32.47 | 32.56 |
| First-to-last exact pixel equality | false | false |
| First-to-last mean absolute difference | 4.31 / 255 | 2.25 / 255 |
| First-to-last maximum channel difference | 176 / 255 | 170 / 255 |
| First frame vs source mean difference | 3.34 / 255 | 3.34 / 255 |
| Last frame vs source mean difference | not measured | 3.02 / 255 |

The fade reduces endpoint drift by about 48%, with essentially no measured throughput cost. The final 15 frames move gradually toward the source pose; adjacent-frame mean changes remain below 0.8 / 255 in this run, so the return is visually smooth.

## Interpretation

`fade_type="s"` returns the generated motion toward Ditto's source pose. It does not replay the first generated output frame, and the renderer, blending, eye handling, and video encoding introduce small pixel differences. Therefore it improves the idle transition but does not provide an exact first-frame/last-frame loop.

For exact endpoint equality, the runtime needs an explicit boundary policy: cache the chosen anchor frame, fade the generated motion toward that anchor, then emit the cached anchor as the terminal frame. The first frame of the turn must use that same cached anchor if exact first/last equality is required. The client should also reuse the cached anchor at the WebRTC boundary if equality is required after lossy encoding.

Evidence: [raw metrics](ditto-fade-s.json), [fade output video](ditto-fade-s.mp4), and [baseline metrics](ditto.json).
