# Four-pose WebRTC crossfade analysis

Date: 2026-07-29

## Result

Use a **2-frame cosine crossfade at 20 fps (100 ms)** for the current four-pose
library.

- Five frames lasts 250 ms and makes the outgoing pose remain perceptible for
  too long.
- Zero frames exposes a one-frame discontinuity because the decoded pose
  endpoints are not identical.
- Two frames shortens the blend by 60% while reducing the average largest
  boundary step by about 47% versus a direct cut.

The server launcher default is now `WEBRTC_POSE_CROSSFADE_FRAMES=2`.

## Exact endpoint audit

The runtime cache inputs were decoded, and every directed switch among the four
physical videos was checked from the final source frame to the first target
frame.

- Exact decoded pixel matches: **0 of 12**
- Ordinary adjacent-frame head-region color MAE in the clips:
  median `0.8243`, p95 `3.0388`, p99 `3.8146`
- Direct pose boundaries land at the `99.7` to `100.0` percentile of ordinary
  within-clip motion.

| Directed switch | Full-frame MAE | Head-region MAE |
| --- | ---: | ---: |
| listener to smile | 4.7555 | 6.6165 |
| listener to neutral | 4.6148 | 6.2619 |
| speaking to smile | 4.6262 | 6.2520 |
| speaking to neutral | 4.7346 | 6.1530 |
| neutral to listener | 4.4526 | 5.7260 |
| neutral to speaking | 4.3292 | 5.5777 |
| listener to speaking | 4.0396 | 5.1898 |
| speaking to listener | 4.0211 | 5.0604 |
| neutral to smile | 4.0299 | 4.9911 |
| smile to listener | 3.5359 | 4.4715 |
| smile to speaking | 3.4333 | 4.3968 |
| smile to neutral | 3.4893 | 4.2766 |

The final-to-first decoded frame is the strongest first check. Exact equality is
the target for a safe direct cut, but hash equality alone is not a complete
smoothness measure: geometry, motion direction, and velocity at the boundary
also matter.

## Crossfade simulation on the runtime assets

The current implementation uses a frozen final outgoing frame as the anchor and
a cosine alpha ramp over the first incoming frames.

| Setting | Duration | Mean largest head-region step | Worst largest step | Mean total transition pixel travel |
| --- | ---: | ---: | ---: | ---: |
| 0 frames | 0 ms | 4.432 | 5.701 | 4.432 |
| 2 frames | 100 ms | 2.346 | 3.183 | 5.068 |
| 5 frames | 250 ms | 1.446 | 2.011 | 5.561 |

Five frames minimizes the single largest delta, but it spreads more total pixel
travel over a quarter second. Its cosine weights retain approximately 93%,
75%, 50%, 25%, and 7% of the frozen outgoing frame. That is 2.5
frame-equivalents of visible outgoing material. Two frames retain approximately
75% and 25%, or one frame-equivalent, and complete the switch in 100 ms.

## Recorded WebRTC controls

Both controls contain the same 12 silent ordered transitions and two live TTS
paths:

1. empathy reaction to speaking direct to neutral
2. light smile reaction to speaking direct to neutral

- `2_frames/indian_tutor_mvp_four_v2_crossfade_2_labeled_showcase.mp4`
- `0_frames/indian_tutor_mvp_four_v2_crossfade_0_labeled_showcase.mp4`

Both labeled files were fully decoded after encoding. The two-frame session
received 3,583 video frames; the zero-frame session received 3,581 video
frames. Each completed all 12 silent transitions and both TTS streams.

## Release criterion for removing the crossfade

Remove the crossfade only after all four final delivered videos pass:

1. decoded last-frame to first-frame equality for every ordered pair;
2. equality after preprocessing, resize, face blending, WebRTC encoding, and
   decode;
3. boundary motion/velocity checks, not only pixel hashes;
4. a recorded zero-frame full-matrix review with no visible pop.
