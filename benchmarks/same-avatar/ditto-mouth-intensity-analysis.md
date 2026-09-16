# Ditto mouth-intensity four-way test

GPU: NVIDIA GeForce RTX 4070 SUPER, 12,282 MiB visible VRAM, compute capability 8.9, driver 595.84. Tested 2026-09-16 UTC.

All four runs used the same portrait, audio, seed, TensorRT models, 25 FPS output, and a 15-frame `fade_type="s"` return-to-source transition. The only changed value was Ditto's internal `vad_alpha` expression blend.

## Results

| `vad_alpha` | Interpretation | Mouth-region motion proxy | Relative to 1.0 | Throughput |
|---:|---|---:|---:|---:|
| 0.0 | minimum | 2.144 / 255 | 58.8% | 32.54 FPS |
| 0.3 | low | 2.434 / 255 | 66.8% | 32.64 FPS |
| 0.5 | balanced | 2.730 / 255 | 74.9% | 32.55 FPS |
| 0.7 | medium | 3.068 / 255 | 84.2% | 32.50 FPS |
| 1.0 | original | 3.646 / 255 | 100% | 32.55 FPS |

The motion score is mean absolute change between consecutive decoded frames in a fixed lower-center face region for this identically framed portrait. It confirms the intended monotonic control, but it is not a facial-landmark displacement or lip-sync accuracy score.

Throughput varied by less than 0.15 FPS, so this control had no meaningful performance cost in this run.

## Implementation caveat

The current `ctrl_vad` implementation blends the complete expression vector toward the source expression even though it declares lip indices. This means the control may attenuate eye or other expression motion as well as the mouth. A production `mouth_intensity` control should apply its scalar only to Ditto's six lip expression points and should support a short ramp when its value changes during a call.

Artifacts: [five-way video](ditto-mouth-intensity-5way.mp4), [four-way video](ditto-mouth-intensity-4way.mp4), [0.5 metrics](ditto-mouth-0p5.json), and [original grid metrics](ditto-mouth-intensity.json).
