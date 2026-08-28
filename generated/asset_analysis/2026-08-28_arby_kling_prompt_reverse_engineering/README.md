# Arby/ARDY → Kling motion reverse engineering for LTX 2.3

Date: 2026-08-28

## Conclusion

The useful behavior in the older MuseTalk clips did **not** come from a detailed text prompt. Kling 2.6 Motion Control received a generic identity/framing constraint while a rendered Arby/ARDY video supplied the choreography. The exact prompt template recorded in the current asset manifest was:

> Transfer the reference video {pose_label} motion to the character in the source image. Preserve the character identity and framing. Keep the natural blinking from the reference and maintain relaxed, direct eye contact with the camera whenever the eyes are open. Keep the camera locked and the mouth closed. No looking sideways.

Therefore, copying that prompt into text-only LTX would not reproduce the motion. The correct translation is to recover the reference timing and express it as simple LTX local prompt segments.

The resulting LTX-ready prompt pack is:

`/workspace/LTX-2.3/musetalk_pose_banks/arby_kling_reference_v5_prompt_pack.json`

## Source evidence

Current Kling derivatives used by MuseTalk:

`/workspace/MuseTalk/generated/downloads/indian_tutor_essential_six_v1/`

Corresponding Arby/ARDY source library and specifications:

`/workspace/MuseTalk/generated/ardy_mvp_four_v4_30frame_boundaries/`

The current runtime assets are:

| Runtime pose | Kling duration | Provenance |
|---|---:|---|
| neutral resting | 10.93 s | New ARDY v4 neutral reference |
| active listening | 10.00 s | New merged listener/empathy reference |
| empathetic head tilt | 10.00 s | Exact alias of active listening |
| speaking direct | 12.00 s | New ARDY v4 speaking reference |
| light smile | 6.00 s | New ARDY v4 smile reference |
| nod / agreement | 2.93 s | Older compatibility asset; not regenerated in the v4 batch |

The Arby/ARDY sources used one full second (30 frames) of canonical rest at both ends. All four source videos had identical decoded opening and closing handles. Kling preserved the general movement but regenerated the pixels, so the Kling outputs do not have exact shared boundaries.

## What the motions actually do

### Neutral resting

- No deliberate gesture.
- The torso and shoulders rise and fall slowly with breathing.
- Only one low-frequency posture arc occurs across the long interior.
- Blinks are quick, sparse, and separated.
- The head remains inside its normal resting area; it does not patrol from side to side.

The ARDY specification used approximately 2.5 breaths and one very slow posture/sway cycle across an eleven-second clip. This explains why the result looks alive without drawing attention to motion.

### Active listening

- Begins with roughly one second of plain resting posture.
- Attention settles slightly forward.
- The head then develops one gradual diagonal listening inclination.
- A tiny acknowledgement nod occurs *inside* that already-established listening posture.
- The diagonal gradually softens and returns toward center.
- There is no repeated nodding and no repeated lateral sway.

The source reaches its main listening angle around 3.45 seconds, places the tiny nod around 4.9–5.35 seconds, and spends approximately 6.75–8.25 seconds returning.

### Empathetic head tilt

There is no independent current source clip: `active_listening.mp4` and `empathetic_head_tilt.mp4` are byte-identical. A genuinely distinct empathy prompt must therefore be derived from the diagonal portion of the merged source while omitting its embedded nod:

- Soft forward attention.
- One gradual diagonal inclination.
- A quiet warm hold.
- A return along the same path.

### Speaking direct

- The body stays front-facing and near its resting axis.
- Motion is organized by conversational phrases, not a continuous sway.
- Small head-led accents occur first; chest or one shoulder follows slightly later.
- Accents differ from each other and are separated by partial releases or low-motion pauses.
- The mouth stays available for MuseTalk rather than supplying visible speech shapes.
- The final portion progressively settles rather than snapping to center.

The twelve-second ARDY reference contains several sub-degree head accents, small diagonal thought shifts, and delayed upper-body follow-through. The important quality is the cadence: head first, torso second, partial release, quiet interval, then a different accent.

### Nod / agreement

The retained Kling clip is a particularly useful reference despite its older provenance:

- Neutral rest through roughly 0.7 seconds.
- One direct chin dip begins around 0.8 seconds.
- Lowest point is around 1.0 seconds.
- Upright again around 1.2 seconds.
- Quiet rest for the remainder of the 2.93-second clip.
- Eyes remain connected to the caller during the gesture.

This is a single 0.4-second yes gesture surrounded by much more stillness. It is not a slow multi-second bow.

### Light smile

- Begins from a neutral face.
- Closed-lip smile starts after the opening rest.
- A tiny chin lift accompanies the onset.
- The mild smile holds briefly.
- Both smile and chin lift release fully.
- The final interval is neutral resting posture.

The source uses one-second boundary handles, smile onset near 1.95 seconds, a mild peak near 2.45 seconds, a hold near 3.35 seconds, and release by about 4.15 seconds.

## Why the previous LTX prompt attempts oscillated between frozen and exaggerated

1. Every v4 pose was compressed into approximately five seconds, while the successful references use pose-specific durations from roughly three to twelve seconds.
2. The positive prompt repeated the action in global and local instructions while also demanding breathing, blinking, posture settling, exact return, stable shoulders, stable gaze, and a still mouth.
3. The negative prompt contained many semantic movement prohibitions. For LTX, that can suppress the intended low-amplitude action along with the unwanted extreme version.
4. Terms such as “passionately,” “head sway,” and “clearly visible” invite semantic amplification when no motion reference controls the skeleton.
5. The previous nod asked the model to solve eye state, head direction, precise amplitude, timing, and loop closure simultaneously. The old Kling nod succeeds because the reference supplies a short down/up trajectory and the prompt merely constrains the character.

## New LTX prompting policy

- Match the reference clip’s spacious timing instead of forcing every action into five seconds.
- Use one behavioral idea per local segment.
- Describe a positive trajectory: rest → action → release → rest.
- Keep the shared prompt short.
- Keep the negative prompt limited to technical continuity and mouth-articulation failures.
- Do not put `large movement`, `head sway`, `overacting`, `frozen`, or similar motion-amplitude language in the negative prompt.
- Preserve a quiet opening and closing interval in the generated motion; then enforce exact decoded first/last handles during certification.
- Evaluate motion at normal playback speed before preparing MuseTalk caches or making another WebRTC session.

## Visual evidence

- [Neutral at 2 fps](contact_sheets/neutral_resting_2fps.jpg)
- [Active listening at 2 fps](contact_sheets/active_listening_2fps.jpg)
- [Speaking direct at 2 fps](contact_sheets/speaking_direct_2fps.jpg)
- [Light smile at 2 fps](contact_sheets/light_smile_2fps.jpg)
- [Nod at 10 fps](contact_sheets/nod_agree_10fps.jpg)

The 2 fps sheets deliberately show whether a pose has gross drift. The 10 fps nod sheet exposes the short down/up timing that a 2 fps sample misses.

## Scope decision

This pass creates and audits the prompt specification only. It does not overwrite v4, generate paid or long-running model outputs, prepare new MuseTalk avatars, or replace the currently configured pose set. The prompt pack is ready to become an `arby-kling-reference-v5` generator profile after review.
