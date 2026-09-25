# LumaTalk multipose mission handoff — checkpoint, 2026-09-25

The user authorizes Grok to continue this mission when Codex quota is low. This
checkpoint records completed work and remaining requirements; it does not mark
the overall goal complete. Inspect git status and live processes before acting.
Preserve user assets, approved source masters, and any newer worktree changes.

## Objective and current status

Build a reusable LTX 2.3 portrait → idle/talking/smiling → MuseTalk character tool
and a persistent FaceTime-like WebRTC runtime. Speech under three seconds keeps
the idle body. Longer speech can use talking and smiling sources. Speech endings
and interruptions at any phase must return smoothly to idle, with continuous
transport timestamps and no lost opening phonemes. Support separate characters
with their own sources, prepared materials, and transition banks.

The implementation and automated follow-up validation are complete. **157 CPU
regressions and 12 actual received audio/video recordings passed**, covering
Japanese and Indian characters. Twenty returns took **0.3479–0.4999 seconds**;
receiver audits detected zero RTP timestamp anomalies and missing timestamps.
These are sequential sessions on one GPU, not a concurrent capacity guarantee.
The slowest return has little margin below the half-second bound.

Normal-speed visual acceptance remains open. The original three Latina idle
seeds failed the closed-mouth gate. A separately versioned interior-guide trial
now passes every mouth frame and all-phase return coverage, and its three API
caches are prepared. See the new experiment section below for fresh-character
receiver results. No bank was promoted to reviewed. Passing tests does not prove
that a viewer cannot notice a transition. Do not invent visual acceptance.

## Authoritative files and evidence

- Repo: `/workspace/MuseTalk`. No AGENTS.md was found in it or ancestors.
- Implementation started from commit `6c6001b`; inspect git log for the subsequent
  follow-up commit. The user authorized committing completed implementation work.
- Executable plan/tasks: `docs/MULTIPOSE_LTX_MUSETALK_IMPLEMENTATION.md`.
- Factory integrity/resume/review: `character_factory/REALTIME_PACKAGE_INTEGRITY.md`.
- Exact approved prompts: `character_factory/PERFECT_THREE_POSE_PROMPTS.md` and
  `character_factory/config/prompt_packs/japanese_fixed_distance_shared_anchor_v1.json`.
- Final evidence: `/workspace/experiments/multipose_validation_20260925_v4/README.md`.
- Review galleries: `japanese/review.html`, `indian/review.html`,
  `entry-repeat/review.html`, and `legacy/review.html` beneath that evidence root.
- Durable source-bound summary:
  `docs/multipose_ltx_musetalk_validation_followup_2026-09-25.json`.
- Final regression log: `v4/full-regressions-final.log` under the experiment prefix.
- Earlier v2/v3 native hangs, rejected CPU profile, and RTP-gap capture remain
  archived separately. Do not delete them or represent them as passing evidence.

The final v4 batch exercised short/long speech, talking/smiling, mid-sentence
abort and next turn, pre-speech abort twice, starting from smile, late smile cues,
threshold-length speech, silence, looping, and legacy requests. The final live
batch needed no extra clock-padding slots; a deterministic actual-track regression
covers the previously failing 60-millisecond clock mismatch.

## Implemented contracts

- Motion banks bind sources and routing to hashes, validate exact decoded shared
  endpoints, and require complete admissible returns from every active pose phase.
  Physical sources are neutral_resting, speaking_direct, light_smile; other public
  IDs alias idle. Source frame selection moves forward at the correct FPS.
- Idle continues moving during inference prebuffer. A 0.3-second raw-body bridge
  reaches the exact source underlying generated frame zero before lips/audio start.
  Return uses the last actually displayed composite and raw closed-mouth idle.
- Reservation-owned cancellation exists before asynchronous setup. Obsolete setup,
  callbacks, workers, failures and cleanup cannot own a newer turn.
- Late semantic cues cannot crowd the terminal idle return.
- Entry transport reconciliation emits missing raw-body video slots or waits for
  normal silent audio. It never skips RTP timestamps or generated opening frames.
  Catchup avoids ordinary pacing sleeps, is bounded to 0.5 seconds, and has
  generation-safe cancellation. Entry preparation is bounded to 5 seconds and
  individual flow calls to 1 second. Coroutine limits require a responsive loop.
- Decoder stop/reset explicitly releases its generator, stream and container.
- `scripts/runtime_av_logging.py` restores FFmpeg native ERROR logging after
  dependency imports and model initialization. TorchVision re-enables PyAV's Python
  callback at import. With that callback, partially consumed frame-threaded decoder
  close deadlocked on the first cycle; native logging completed 100 cycles in
  2.53 seconds. Do not remove the startup configuration or re-enable Python FFmpeg
  logging on threaded media paths. Native codec errors remain visible on stderr.
- Registry discovery uses small source/atlas-hash sidecars. Unrelated malformed
  entries are isolated; selected corruption or duplicate matching banks fails.
- Factory generation has immutable image/prompt/graph/options fingerprints and
  verified partial/no-op resume. Packaging validates actual source media and
  measurements, preserves preparation/review state, and atomically publishes.
- `assemble_three_pose_sources.py` accepts verified separate pose manifests.
  Each pose must match the explicitly chosen reference profile except its seed
  and descriptive citation. Optional native interior guides require matching
  per-pose manifest metadata and the exact saved graph. It copies bytes and
  preserves provenance; it does not approve source quality. Experimental reference
  packs use neutral provenance fields and retain their declared approval status.
- `review_motion_evidence.py` verifies real received media and creates galleries.
  `review_realtime_character.py` requires bound evidence, reviewer identity, notes
  and a normal-speed viewing attestation for acceptance. No real receipt was issued.

## Source and package inventory

Japanese approved sources:
`/workspace/experiments/japanese_ltx_fixed_distance_20260925/{idle,talking,smiling}.mp4`,
512×832, 24 fps, 241 frames each. Package:
`/workspace/experiments/realtime_characters/japanese_20260925/`.
Three content-named prepared caches exist.

Indian approved V6/V14/V8 production sources are aliased under
`/workspace/experiments/indian_fixed_distance_multipose_20260925/`.
Original masters remain in
`assets/ltx23_pose_banks/sample_ai_human_facetime_closeup_production_v1/certified/`.
Geometry 480×832, 24 fps, lengths 241/289/145. Package:
`/workspace/experiments/realtime_characters/indian_20260925/`.
All three API caches are prepared and character.json now retains preparation
metadata; wrapper resume log is in v3/indian-package-resume.log.

Fresh Latina experiment root:
`/workspace/experiments/latina_fixed_distance_multipose_20260925/`.
Portrait: `character_factory/generated/portrait_pose_set_20260923/latina_woman.png`.
Original full trio is in sources/; idle-only rerolls are in idle-reroll-seed194/
and idle-reroll-seed195/. Selected assemblies retain separate provenance.

| Idle seed | Frames above 3.5 px lip gap | Maximum gap | Result |
| --- | ---: | ---: | --- |
| 193 | 58 / 241 | 12.354 px | Rejected |
| 194 | 86 / 241 | 15.226 px | Rejected |
| 195 | 71 / 241 | 8.556 px | Rejected |

Rejections and full measurements are hash-bound beside the sources. Seed 195
used identical prompt text/settings and took 5.5 minutes to generate plus
0.5 minutes to decode. No further seed was queued. Do not substitute smiling
for idle, lower the gate, or accept a newer clip merely because it is newer.

The bounded Grok audit concluded the prompts/graphs matched the accepted recipe.
In the installed Comfy sampler, CFG=1 skips negative conditioning; this is ALSO
true of the approved Japanese source. Stored negative text remains provenance.
The evidence does not prove why this portrait opens its mouth. Audit conclusion:
`/workspace/experiments/multipose_validation_20260925_v2/grok-latina-final-answer.log`.
Grok audit session: `01a0d867-09ef-7a90-b1e4-fb74928839cd`. The user's existing Grok
TUI was left untouched; no background Grok task owns the remaining mission.

## Environment and repeat commands

Runtime Python: `/workspace/.venvs/musetalk_trt_stagewise/bin/python`.
LTX/MediaPipe Python:
`/workspace/experiments/soulx_ltx_motion_pilot_20260922/A1/.venv/bin/python`.
One RTX 4070 SUPER, 12 GB. Never run LTX and MuseTalk simultaneously. The owned
validation server is stopped after verification; check processes before GPU work.
Disk was about 0.5 GB free after the new Latina API caches. Do not delete unrelated assets.

Use the original baseline launch, not the rejected diagnostic CPU profile:

```bash
MUSETALK_TRT_PROFILE_ENV_FILE=/workspace/MuseTalk/.runtime/musetalk_trt_local_sm89.env \
WEBRTC_MOTION_ATLAS_DIR=/workspace/experiments/realtime_characters \
WEBRTC_MOTION_ALLOW_UNREVIEWED=1 \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
bash scripts/run_trt_stagewise_server.sh --profile baseline --host 127.0.0.1 --port 8000
```

The diagnostic CPU profile enabled OMP_PROC_BIND and pinned the main server plus
many media tasks to one core. It caused 3.6-second returns and was rejected. It was
not the native-hang fix. Preserve the original baseline plus native logger helper.
The pilot override above is for testing candidates, not implicit visual acceptance.

Reproducible final batch: `v4/run-live-validation.py` beneath the experiment prefix.
It waits for server health, records real receivers, and stops on a failed check.
Use a new output directory for new runtime revisions; preserve prior evidence.
Do not rerun completed GPU work merely to rediscover this state. No Segmind or
paid image/video API was used or is authorized as a replacement for local LTX.

## Next work

1. Inspect current user input for normal-speed review of the final Japanese and
   Indian recordings. Source approval alone does not approve live transitions.
2. Address any observed visual issue with specific evidence, without changing
   accepted masters in place or weakening timestamp/return/source-quality gates.
3. Review the fresh Latina interior-guide candidate and its received recordings.
   Do not regenerate the failed seed-only samples. If a specific defect remains,
   analyze that evidence before another render; keep experimental recipes separate
   from the approved pack.
4. Publish a reviewed bank only with a real viewing decision and bound evidence.
   Do not mark the full goal achieved merely because automated checks passed.

User authorization covers local implementation/tests, prior commits and the
Grok handoff. It does not authorize unrelated deletion, deployment, paid APIs,
or messages to other people. Preserve concrete evidence and report remaining
limitations plainly.


## Controlled native-guide fresh-character trial

Plan and tasks: `docs/LTX_INTERIOR_GUIDE_EXPERIMENT_2026-09-25.md`.
Root: `/workspace/experiments/latina_fixed_distance_multipose_20260925/idle-interior-guides-seed195`.
The seed remains 195 and prompt strings remain exact. Two native same-portrait
image guides at frames 80 and 160, strength 1.0, are added before the final guide.
The experimental pack is `latina_idle_interior_guides_seed195_v1.json`; no approved
pack or source was changed. Default graph equality was checked against the
recorded baseline, and 40 focused factory regressions passed.

The new idle has zero frames above 3.5-pixel lip gap, maximum 0.5328 pixels
(baseline: 71 frames, maximum 8.5564). All 723 source phases have admissible idle
returns. Generation took 340.81 seconds, decode 30.04 seconds, full command
385.507 seconds. Comparison stills and per-frame measures are diagnostics only.
Seven brief eye-closure spans need normal-speed review for excessive blinking.

`selected/` retains original talking/smiling plus new idle and separate assembly
provenance. Package: `/workspace/experiments/realtime_characters/latina_guided_20260925`.
All three content-named API caches are prepared. The bank remains unreviewed.
Actual receiver work is recorded by `run-webrtc.py`, `webrtc.log`,
`webrtc-result.json` and the `webrtc/` directory within the trial root.

To fit the caches, only 723 redundant temporary Comfy decode PNGs from idle
seeds 194, 195 and this controlled trial were reclaimed (419,950,596 bytes),
after verifying retained latent/native/delivery hashes. The exact receipt is
`temporary-decode-cleanup.json`. No source master or unrelated asset was removed.

Fresh Latina receiver outcome: all three cases passed, four idle returns took
0.3493–0.4000 seconds, and there were no timestamp anomalies. `webrtc/review.html`
is the actual received-media gallery, separate from `review.html` for raw source
comparison. Durable results: `docs/ltx_interior_guide_validation_2026-09-25.json`.
Combined with v4, there are now 15 passing recordings across three characters.
The owned test server was stopped. Visual acceptance of these recordings remains
pending, particularly the new idle's blink rate. Do not promote a bank or claim
imperceptible transitions without that review. No more render is queued.
