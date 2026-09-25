# Latina idle: native interior-guide comparison

This is a separately versioned experiment, not a replacement for the user-approved
Japanese fixed-distance trio or its prompts. The multipose runtime and final v4
Japanese/Indian receiver evidence are unchanged.

## Why this test exists

Three Latina idle renders using the approved motion text failed the unchanged
3.5-pixel inner-lip-gap gate: seed 193 had 58/241 failing frames, seed 194 had 86,
and seed 195 had 71. Seed 195 visibly parts its lips, including frames 45, 74 and 97;
this is not just a landmark-noise claim. Its longest violation spans frames 37–102
(2.75 seconds), with maximum 8.556 px at frame 74. The reference mouth is closed.

Use the same seed 195 with two additional full-image portrait guides, at native
frame 80 (3.333 s) and 160 (6.667 s), both strength 1.0. Positive and negative prompt
strings, portrait crop, sampler, sigmas, CFG, model, delivery endpoints and decode
settings remain identical. The only graph change is the added guide chain and
rewiring the final guide to receive it. Holding the seed does not guarantee identical noise tensors when guide count
changes. This comparison measures the resulting conditioning configuration; it
does not isolate every internal numerical effect.

## Installed implementation

The installed Comfy `LTXVAddGuide` accepts arbitrary indices for a single PNG.
Its 8-frame rounding applies to multi-frame guide clips. Both guides use the same
LoadImage and VAE as the endpoints. They run before AV concatenation:

`portrait → interior_guide_0 → interior_guide_1 → end_guide`

Concat, positive/negative guider inputs and positive/negative crop metadata still
consume `end_guide`. Crop removes all appended guide latents. Strength 1.0 keeps
these latents unnoised and avoids the additional nonunit-strength attention-mask
path. No new model, mask, LoRA, mouth compositing or silence override is involved.

Installed implementation: `/workspace/experiments/soulx_ltx_motion_pilot_20260922/A1/ComfyUI/comfy_extras/nodes_lt.py`,
`LTXVAddGuide.get_latent_index`, `append_keyframe` and `execute`.
[Official conditioning documentation](https://github.com/Lightricks/LTX-2/blob/main/packages/ltx-pipelines/docs/conditioning.md)
describes guiding latents for interpolation between image keyframes. The newer
2.5 generated-keyframe feature in that document is not used here.

Video tokens increase from 13,728 to 14,560 including guides (6.1%). This is not a
VRAM or elapsed-time guarantee. Baseline seed 195 took 330.54 seconds generation
and 30.04 seconds decode.

## Reusable configuration and integrity

`character_factory/config/prompt_packs/latina_idle_interior_guides_seed195_v1.json`
contains experimental approval status and this extra idle-profile field:

```json
"interior_guides": [
  {"frame_idx": 80, "strength": 1.0},
  {"frame_idx": 160, "strength": 1.0}
]
```

The generator rejects duplicate/noninteger/end-point indices, unknown guide
keys, and nonfinite, nonpositive or greater-than-one strengths before output
creation. Pack hashing already binds the extra configuration into immutable
render identity. Omitted/empty guides preserve the existing graph exactly.
An optional `workflow.interior_guides_by_pose` records nonempty selected profiles
and survives rendering other poses later into the same generation job.

Assembly with `--reference-prompt-pack` requires each selected pose to match that
explicit recipe, excluding only its seed and descriptive citation. It validates
per-pose guide metadata and the complete saved generation graph. Other poses may
come from their original unchanged jobs. Experimental provenance uses neutral
reference fields and retains declared approval status; copying files does not
approve them. The built-in approved-pack provenance remains compatible with
existing seed 194/195 assemblies.

## TASKS and reproduction

- [x] Confirm actual visible baseline mouth parting.
- [x] Audit installed single-image guide indices, guide cropping and memory scope.
- [x] Keep default generation identical to the saved seed 195 graph.
- [x] Add opt-in per-pose guide validation and explicit assembly provenance.
- [x] Run 40 focused CPU tests; preserve focused-regressions.log.
- [x] Generate the single controlled idle candidate.
- [x] Assemble with original talking/smiling, preserving all source manifests.
- [x] Measure every frame with the same reference and gate.
- [x] Compare mouth closure, head geometry, guide neighborhoods, and endpoints.
- [x] Prepare all three API caches and record the three actual WebRTC cases.
- [ ] Obtain normal-speed visual review before promoting any bank.

Experiment root:
`/workspace/experiments/latina_fixed_distance_multipose_20260925/idle-interior-guides-seed195`.
Its `experiment.json` binds the baseline and selected pack. `run.py` records the
exact command, total elapsed time and exit code. It owns its local Comfy worker.
Stop MuseTalk before running it; the two workers cannot share this 12 GB GPU.

```bash
python3 /workspace/experiments/latina_fixed_distance_multipose_20260925/idle-interior-guides-seed195/run.py
```

The generated `sources/manifest.json`, saved graphs, latent, native MP4 and delivery
MP4 retain their own provenance. The delivery replaces only endpoints to obtain
pixel-identical boundaries, as the baseline did. Interior guides are never output
frame substitutions. Source masters and accepted prompt packs remain unchanged.

## Acceptance and limitations

The mouth gate remains 3.5 pixels. Passing it is necessary, not sufficient. Full
portrait guides may freeze breathing or cause periodic pose resets. Inspect
normal-speed motion around 3.3 s and 6.7 s as well as the loop boundary. Compare
mouth width, eye position/scale and per-frame displacement against the same-seed
baseline. Do not substitute smile for idle, weaken the gate or mark a bank
reviewed based only on these measurements.

The new candidate passed the unchanged lip-gap gate in all 241 frames: maximum
0.5328 pixels versus baseline 8.5564, with zero violating frames versus 71. The
byte-bound assembly preserves original talking and smiling. The resulting atlas
has full idle-return coverage for all 241 frames of each physical pose (723/723).
Its package remains `candidate_requires_recorded_review`; passing this gate is
not a visual acceptance decision.

Per-frame eye-opening measurements show seven brief closures, not a
continuous early eye closure. The one-frame-per-second contact sheet happens
to capture two early blinks. With eye opening below half the reference, measured spans are
11–13, 35–36, 63–64, 88–89, 122–125, 171–173 and 205–207. This is a diagnostic
observation, not a new acceptance threshold. Eye landmark displacement and scale
spikes around blinks should not be described as proven whole-head jumps.

Artifacts beneath the experiment root:

- `comparison.json`: hashes, mouth/geometry extrema and guide-neighborhood rows.
- `comparison-frames.csv`: all per-frame values and adjacent changes.
- `review.html`: synchronized baseline/candidate players and contact sheets.
- `selected/assembly-provenance.json`: original per-pose source bindings.
- `selected-measurements.json`: full trio measurements.
- `sources/manifest.json`: render timings and source/native/latent hashes.
- `focused-regressions.log`: 40 passing factory tests.
- `temporary-decode-cleanup.json`: verified retained latent/native/delivery files
  before reclaiming only 723 redundant PNG intermediates from these three idle
  experiments (419,950,596 bytes). Source masters were not deleted.

Fresh package: `/workspace/experiments/realtime_characters/latina_guided_20260925`.
All three actual received WebRTC recordings passed, with four idle returns in
0.3493–0.4000 seconds and zero timestamp anomalies. Entry bridges took
0.2951–0.3004 seconds. Short speech used idle only; the long case used talking and
smiling; interruption returned to idle and accepted a following short reply.
The longest repeated idle frame during prebuffer was one frame in each case.

[Live review gallery](/workspace/experiments/latina_fixed_distance_multipose_20260925/idle-interior-guides-seed195/webrtc/review.html)
and [bound verification](/workspace/experiments/latina_fixed_distance_multipose_20260925/idle-interior-guides-seed195/webrtc/verification.json)
retain the actual receiver artifacts. [Durable summary](ltx_interior_guide_validation_2026-09-25.json)
records source hashes, generation inputs and results. The runtime remains at
6f634ca; only factory configuration/validation and documentation changed here.
The owned validation server was stopped after recording. Normal-speed visual
review remains pending; no candidate was marked reviewed.

Measured generation: 340.81 seconds; decode: 30.04 seconds; full command: 385.507 seconds.

Additional CPU source inspection is in `source-review-notes.md` and
`source-review-evidence/`. The largest measured eye-midpoint jump, +5.603 pixels
at frames 121→122, occurs during a brief blink; forehead median optical flow
moves only +0.756 pixels vertically. This supports caution when interpreting
landmark spikes as head movement. Head/shoulders remain steady and fabric shows
small changes, but stills and optical flow cannot establish convincing shallow
breathing rather than generated texture motion. That also remains part of the
normal-speed visual review.
