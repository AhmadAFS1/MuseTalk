# Japanese idle: two shallow breaths, review candidate v4

**Watch [idle.mp4](idle.mp4) at normal speed before accepting it.** This is an
idle-only LTX 2.3 render for the Japanese FaceTime portrait, not a certified
three-pose MuseTalk bank. The previously accepted
[talking and smiling clips](../japanese_breath_speech_review_v3/README.md)
remain unchanged.

The [v17 prompt pack](../../../../config/prompt_packs/japanese_idle_five_second_breath_seed193_v17.json)
asks for alert listening with one ordinary shallow upper-chest breath in a
five-second native shot. The runner repeats its endpoint-matched 121-frame
cycle twice to make this 241-frame, 10.04-second delivery. It uses the same
Japanese portrait, center-cropped 512×832 guide, seed 193, eight-step LTX 2.3
low-VRAM graph, and 64/16 tiled decode as the preceding idle experiments.
The clip is silent, with exact decoded first/last and internal join equality.
No fade or crossfade was added.

| Check | Measured result |
| --- | ---: |
| Center shirt neckline movement | Two similar 12 px rises, around seconds 2 and 7; earlier rejected v2 had one 28 px rise and rejected v3 had 0 px. |
| Head position | 2.6 px sideways eye-midpoint excursion; 0.8° maximum eye-line tilt. |
| Mouth | 0.4 px maximum central lip gap; visually closed. |
| Eyes | Longest near-closed run 0.17 s; six detected short blinks across the repeated ten seconds. |
| Shoulder midpoint | 7.0 px maximum movement. |

See the [contact sheet](idle-contact.jpg), [face/shoulder audit](idle-motion-audit.json),
and [shirt-neckline audit](idle-neckline-audit.json). The tracker is calibrated
to this portrait's dark shirt and is a proxy for visible shirt motion; the
numbers cannot decide whether the breathing or repeated blinks look natural.
The [source manifest](manifest.json) records the graph and decode settings;
native work files are retained locally but excluded from Git. This review
clip has SHA-256 `6d26ed85b4670acc7a9d5cce6d1b989e6a25c7faf994f45b2cb0c8dada514830`.

**Status: candidate for human review.** The user has not yet accepted this
idle. Shared-anchor certification and MuseTalk integration have not run.
