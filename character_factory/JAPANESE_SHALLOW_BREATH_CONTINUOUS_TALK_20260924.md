# Japanese avatar: shallow-breathing and continuous-talking test

**Subsequent user review supersedes the idle selection below:** v7's repeated
idle looks like rapid head wobble with no actual breathing and is rejected.
The talking and smiling clips were accepted. See the
[idle replacement record](JAPANESE_IDLE_SHALLOW_BREATH_RERUN_20260924.md).

The [selected v2 prompt pack](config/prompt_packs/japanese_selected_native_three_pose_v2.json)
and [v2 review videos](generated/portrait_pose_set_20260923/ltx/japanese_shoulders_head_review_v2/README.md)
are the frozen best-tested baseline. This test changes only the positive
prompts for idle and talking in
[`japanese_idle_talking_shallow_continuous_v3.json`](config/prompt_packs/japanese_idle_talking_shallow_continuous_v3.json).
Both seeds, both negative prompts, the 241-frame length, and the corrected
center-crop guide remain identical. The selected smile is unchanged.

The baseline idle looks like one deep breath, and the talking video stops
articulating halfway through. A full-frame, per-second
[audit](generated/portrait_pose_set_20260923/ltx/japanese_shoulders_head_review_v2/breath_speech_baseline_audit.json)
supports the timing observation:

| Baseline clip | Shoulder midpoint travel | Per-second pattern |
| --- | ---: | --- |
| Idle | 14.1 px | One broad rise centered near second 4, then returns. |
| Talking | 17.2 px | A pronounced rise near second 4; central lip gap exceeds 3 px in 75–100% of frames during seconds 0–3, then in 0% during seconds 4–9. |

The liked [Indian native talking reference](generated/portrait_pose_set_20260923/ltx/indian_talking_timing_reference.json)
has about 3.2 px shoulder-midpoint travel by the same detector. Its measured
lip activity remains substantial through second 7, tapers during second 8,
and closes for its final return to the starting frame. That is a useful motion
reference, though the portrait and body proportions differ.
The [certified Indian idle reference](generated/portrait_pose_set_20260923/ltx/indian_idle_breath_reference.json)
also measures only 3.2 px of shoulder-midpoint travel and has no large single
rise, supporting the user's normal-speed impression that the Japanese idle
breath is too deep.

The shoulder numbers use independent MediaPipe Pose detections on every
decoded frame, smoothed with a 13-frame median. Mouth activity uses FaceMesh
landmarks 13/14 and mouth corners 61/291. These are diagnostics, not a
substitute for normal-speed playback. The proposed v3 prompt asks idle for
two or three small, evenly spaced breaths instead of one deep inhale. It asks
talking for changing syllable shapes throughout the ten-second shot, including
the second half, with only shallow breaths between words. It avoids the v2
phrase “Her speech ends,” which may have encouraged LTX to switch into a
neutral listening pose too early.

Review gates for the new renders:

1. Both clips retain the same portrait, guide, clean shoulder edges, level
   head, 512×832 frame, 24 fps, 241 frames, no audio, and exact decoded
   first/last equality within each clip.
2. Idle shows several small breathing motions without one conspicuous
   shoulder/chest lift, while retaining natural blinks and closed lips.
3. Talking shows varied lip and jaw motion through the middle and late
   seconds, with no several-second neutral stretch; breaths are small and
   do not interrupt the speech rhythm.
4. Compare full-speed video and per-second mouth/shoulder diagnostics to v2.
   Keep v2 selected for any pose whose rerender does not improve the actual
   visual issue.

The experiment outputs are in
`generated/portrait_pose_set_20260923/ltx/japanese_shallow_breath_continuous_talk_v3/`.
Both v3 clips decoded to 241 silent frames with exact matching first and last
frames and the same cropped guide as v2. Full-frame measurements and contact
sheets show:

| V3 clip | Result | Decision |
| --- | --- | --- |
| Idle | Shoulder midpoint travel increased to 17.0 px and still forms one broad rise around seconds 2–5. | Reject; v2 remains better. |
| Talking | Shoulder travel fell to 8.6 px; mouth articulation continued through second 7, tapered in second 8, and rested in second 9. Head vertical eye travel was 15.5 px. | Strong improvement over v2, but still settles before the last second. |

The next controlled pack,
[`japanese_idle_talking_height_anchor_v4.json`](config/prompt_packs/japanese_idle_talking_height_anchor_v4.json),
keeps the same guide, seeds, and negatives. It asks idle to hold the shoulders
at reference height with breathing barely visible in the upper chest. For
talking it removes the verbal return-to-neutral cue and asks for new syllables
through the final second. The v2 selected pack remains unchanged while these
are tested.

The decoded v4 attempt did not solve either main timing problem. Idle still
made one shoulder-rise arc (15.8 px maximum). Talking shoulder travel fell to
7.0 px, but mouth articulation stopped by 6.33 s, leaving a 3.62 s quiet
tail before the forced final frame. V3 talking is therefore the stronger
candidate: its last measured open-mouth frame is at 8.00 s, versus 3.92 s
for v2. The liked Indian reference's last open-mouth frame is at 8.12 s, so
v3 approaches that example's timing but does **not** yet meet the requested
speech-through-the-end behavior.

The next controlled pack,
[`japanese_idle_talking_seed191_v5.json`](config/prompt_packs/japanese_idle_talking_seed191_v5.json),
changes both prompts and the idle/talking seed to 191. It retains the same
portrait, center-crop guide, frame length, negative prompts, and unchanged
smile. The idle prompt emphasizes several tiny regular breaths, and the
talking prompt asks for one uninterrupted sentence through the final seconds.
This seed test is needed because three variants at seed 197 kept the single
idle rise or ended speech early.

V5 talking is the best timing result so far. The mouth remains active across
seconds 0–8 and its last measured lip gap above 3 px is at 8.71 s; the quiet
tail before the final identical frame is 1.25 s. Shoulder travel is 6.3 px,
versus 17.2 px in v2. Its contact sheet shows varied conversational mouth
shapes and no long pause in the middle. It still should be reviewed at normal
speed, and the final quiet tail should be disclosed rather than called
literal speech through frame 240. V5 idle has two rises rather than one, but
they reach 14.2 px and its eye line travels 23.8 px, so it is rejected.

The [v6 short idle prompt](config/prompt_packs/japanese_idle_short_breath_cycle_v6.json)
tests a single 81-frame, 3.375-second shallow breath. If its endpoint and
motion are clean, three copies joined at matching endpoint frames form a
241-frame, 10.04-second idle with three regular breaths. That is an explicit
assembly step and must be recorded as such; it is not a single 10-second
native generation.

The v6 81-frame render succeeded as a motion control: 2.0 px maximum shoulder
travel, 0.7 px eye-line travel, and closed lips. The explicit three-cycle
[assembly](scripts/repeat_short_idle_cycle.py) produced a 241-frame silent
[ten-second review clip](generated/portrait_pose_set_20260923/ltx/japanese_idle_short_breath_cycle_v6/idle-loop-10s.mp4)
with exact decoded joins and endpoints, 2.9 px maximum shoulder travel, and
three shallow cycles. Adjacent frame difference at the joins is within the
ordinary frame-difference range. FaceMesh detected no blink in the source or
assembled clip, so a final [v7 short-cycle prompt](config/prompt_packs/japanese_idle_short_breath_blink_v7.json)
adds one natural blink. Repeating a successful v7 cycle would also repeat the
blink about every 3.3 seconds; review that rhythm for artificial regularity.

V7 appeared to succeed on the *earlier* motion gates, which proved insufficient.
Its native 81-frame source has
5.1 px shoulder travel, 5.2 px eye-line travel, closed lips, and one detected
blink. The [selected 241-frame idle](generated/portrait_pose_set_20260923/ltx/japanese_breath_speech_review_v3/idle.mp4)
has 5.2 px shoulder travel, 5.3 px eye-line travel, and three detected blinks.
All four decoded cycle-boundary frames match exactly, but a later neckline
check found **0 px of shirt motion** and the user rejected the loop's head
wobble. The historical [v3 prompt pack](config/prompt_packs/japanese_selected_native_three_pose_v3.json)
combined this now-rejected idle, v5 talking, and the unchanged v2 smile. The
[review folder](generated/portrait_pose_set_20260923/ltx/japanese_breath_speech_review_v3/README.md)
contains the selected videos, contact sheets, motion audit, source manifests,
and validation.

Post-selection timing check: the exact v5 talking text with seed 193
([v8 pack](config/prompt_packs/japanese_talking_seed193_v8.json)) stopped
articulating at 6.71 s and left a 3.25 s quiet tail. It was rejected. The
next talking-only [v9 pack](config/prompt_packs/japanese_talking_mid_conversation_v9.json)
returns to seed 191, keeps the same guide and negative prompt, and describes
the ten-second shot as the middle of a conversation that continues beyond the
cut. This tests whether removing a sentence-completion cue shortens the
remaining v5 quiet tail.

V9 failed that test: lip opening ended at 2.71 s, leaving a 7.25 s quiet
tail; shoulder travel grew to 14.1 px and eye-line travel to 24.3 px. It is
rejected. The selected v5 talking clip remains the best of the prompt-only
rerolls. This experiment corrected the original halfway stop substantially,
but did not achieve literal articulation through the final second. The exact
closed-mouth endpoint necessarily closes the lips on frame 240, and LTX's
timing before that frame varied markedly across both prompts and seeds.
