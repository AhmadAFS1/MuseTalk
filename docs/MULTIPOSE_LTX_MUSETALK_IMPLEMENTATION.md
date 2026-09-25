# Multipose LTX / MuseTalk implementation

Updated 2026-09-25. This is the executable implementation plan and maintenance
runbook for the LumaTalk three-video character runtime. The original reasoning
and alternatives are in [the transition analysis](three_pose_transition_analysis_2026-09-25.md).

## Intended behavior

One persistent WebRTC video track displays a character throughout a call.
TTS audio controls the mouth independently of the body video. Replies shorter
than three seconds keep the idle body source; longer replies can use talking
and smiling sources. A known speech ending returns the body to idle before the
last phoneme. An interruption stops unsent audio immediately and releases the
last displayed face into a matched, closed-mouth idle phase.

The quality target is an unnoticeable change of source. Equal first/last frames,
a correct timestamp sequence, and passing tests do not by themselves prove that
perceptual target. Source videos and received recordings remain reviewable.
The feature is opt-in; a candidate atlas requires an explicit pilot setting.

## Tasks

- [x] Keep the approved Japanese source trio and exact prompt pack unchanged.
- [x] Measure every source frame against a common portrait coordinate system.
- [x] Bind measurements and transition banks to exact source hashes.
- [x] Check shared decoded endpoints, geometry, motion, blink state, and closed-mouth idle destinations.
- [x] Build candidate connections across every phase of all three sources.
- [x] Keep under-three-second replies on idle while MuseTalk supplies lipsync.
- [x] Compile longer replies into matched idle/talking/smiling/idle source choices.
- [x] Use each pose's own prepared materials and forward-only source indices.
- [x] Carry source metadata through the WebRTC queue; reject stale generations.
- [x] Recover from the last emitted composite, including a partially completed transition.
- [x] Separate audio completion from a bounded body/mouth return.
- [x] Cancel audio and generation cooperatively on an ordered abort event.
- [x] Protect newer turns from stale aborts and cleanup callbacks.
- [x] Prepare the three physical avatar caches through the existing API.
- [x] Record real received WebRTC audio/video for short, long, smiling, interrupted, and following-turn cases.
- [x] Add reusable character packaging, content-derived cache IDs, and a multi-character atlas registry.
- [x] Add automated policy, geometry, phase, registry, ownership, and timing regression checks.
- [x] Exercise 1,446 cross-pose candidates and record threshold, silence, prebuffer cancellation, bridge interruption, looping, and legacy-request cases.
- [x] Fix accumulated audio/video pacing drift in the opt-in runtime and retain the failing captures for comparison.
- [x] Save final measurements, visual inspection artifacts, and reproducible evidence.
- [x] Reject changed portraits/prompts/graphs on resume and validate cached artifact integrity.
- [x] Preserve prepared-cache and review state across matching package resumes.
- [x] Isolate unrelated malformed registry entries and atomically publish indexed banks.
- [x] Reserve the complete terminal idle bridge even when a smile cue arrives late.
- [x] Invalidate request setup on abort before scheduler registration, including stale callbacks.
- [x] Keep idle moving during prebuffer and bridge into the exact raw body under generation frame zero.
- [x] Release decoder generators, streams, containers and native workers deterministically.
- [x] Reproduce and fix the TorchVision/PyAV threaded decoder shutdown deadlock.
- [x] Bound entry-worker and recorder-network waits; preserve failed-run evidence.
- [x] Preserve contiguous persistent RTP at speech entry with bounded raw-body/silent-audio alignment.
- [x] Repeat real receiver tests after entry, cancellation, native shutdown, and timestamp fixes.
- [x] Validate the accepted Indian identity with 241/289/145-frame clips alongside Japanese.
- [ ] Exercise a fresh portrait through rendering, packaging, cache preparation, and received playback.
- [ ] Obtain normal-speed visual acceptance of the transition recordings before marking a bank `reviewed`.

## Files and responsibilities

| File | Responsibility |
| --- | --- |
| `scripts/measure_motion_sources.py` | Static per-frame FaceMesh measurements, file hashes, and decoded endpoint hashes. Run in the installed LTX Python environment. |
| `scripts/build_motion_atlas.py` | Candidate lookup table for each source frame and destination pose; reject stale measurements and unloopable sources. |
| `scripts/motion_transitions.py` | Atlas validation/selection, short/long policy, forward source mapping, bounded optical-flow bridge. |
| `scripts/webrtc_pose_router.py` | Convert semantic speech plans into exact pose/source-frame snapshots. |
| `scripts/hls_gpu_scheduler.py` | Use the matching prepared latent/background for each snapshot and bridge the final lip-synced composites. |
| `scripts/webrtc_motion_playback.py` | Recover from the last displayed frame, close into raw idle, track return latency. |
| `scripts/webrtc_tracks.py` | Persistent transport, generation-tagged frame queue, emitted metadata, decoder positioning. |
| `scripts/webrtc_manager.py`, `api_server.py` | Activate matching banks, order events, reject wrong-turn aborts, cancel audio/generation, wait for recovery before the next turn. |
| `character_factory/scripts/build_realtime_character.py` | Render or ingest a trio, measure it, build a bank, write API manifests, optionally prepare caches. |
| `scripts/test_webrtc_motion_transitions.py` | Reproducible real WebRTC recordings, with received audio and video timestamp audits. |
| `scripts/audit_motion_bank.py` | Exercise every candidate using the actual bridge, reporting pixel discontinuities against hard cuts. |

## Source and transition contracts

The accepted source directory is
`/workspace/experiments/japanese_ltx_fixed_distance_20260925`. The three clips are
512×832, 24 fps, 241 frames each. Their source hashes are retained in the package.
The prompt pack is `character_factory/config/prompt_packs/japanese_fixed_distance_shared_anchor_v1.json`.
Use center cropping and `--shared-anchor`; do not add frozen handles or fades to
these source masters.

An atlas contains `version`, `status`, `fps`, `short_reply_seconds`,
`bridge_seconds`, `sources`, `edges`, and `exit_coverage`. For each physical pose
and each original frame, `edges[source][destination][frame]` names the candidate
incoming frame, score, geometry differences, and admissibility.

Matching uses downsampled appearance outside the lower-face region, a four-frame
motion difference, eye height/scale/roll/horizontal position, and eye opening.
Incoming idle must have a measured central lip gap at most 3.5 pixels. All idle
phases must support a consecutive closed-mouth recovery. An expressive source
with any uncovered interrupted-return phase is disabled for live selection.
These numeric gates are pilot diagnostics for the 512×832 close-up recipe,
not a general identity or perceptual-quality classifier.

Atlas registration compares all three prepared MP4 hashes. A registry can hold
many characters; a session selects the bank matching its physical source files.
Duplicate matching banks are rejected. New packages publish a small
`motion-registration.json` with source hashes and the atlas digest, so unrelated
large atlases need not be parsed on every call. Unrelated malformed entries are
logged and skipped; a selected corrupt/stale bank fails explicitly. Files publish
by atomic replacement. Parsed atlases are cached by their exact
contents, because this filesystem can keep the same modification timestamp
across rapid writes. Never trust timestamps alone for review-status changes.

## Runtime algorithm

1. Normalize the public TTS metadata and reserve its existing session stream.
   A v1 request gets a default internal motion plan; a v2 request preserves its
   requested speech/smile cues. The public speech allowlist stays compatible.
   A cancellation token belongs to the reservation before any asynchronous audio
   normalization or decoder preparation. Every setup boundary and delayed callback
   checks ownership/token validity; a cancelled setup cannot resurrect speech.
2. Obtain TTS duration from the actual prepared generation timeline. Below
   three seconds, every body frame is idle. At or above three seconds, begin
   with 0.3 seconds of idle, then select an admissible expressive entry. Idle
   continues moving while inference buffers. Before the first generated lip frame,
   blend the currently moving body into the exact raw source under generation
   frame zero for 0.3 seconds. Keep speech gated and every generated frame queued
   until that entry finishes. This costs about 0.3 seconds of additional initial
   latency and removes the former 0.45–0.90-second visible idle hold.
3. For known-duration audio, start the final idle return 0.5 seconds before
   the end: a 0.3-second bridge plus approximately 0.2 seconds of idle lipsync.
   Refuse late semantic entries whose bridge/cooldown would overlap this return.
4. Advance source phase by `source_fps / generation_fps`. Always use the
   original forward range; APIAvatar's reversed cache half is not the body
   playback policy. Numerical tolerance prevents fractional-phase drift.
5. Compose each frame with that source pose's own face coordinates, mask,
   latent, and background. On a source change, warp outgoing and incoming
   composites toward intermediate geometry and blend with cosine easing.
   The optical-flow displacement is bounded to 32 pixels.
6. Attach pose, render key, source frame, and generation frame to each queued
   video frame. The existing generation token rejects cancelled work.
7. Update recovery state only when a frame is actually emitted. Inference may
   be seconds ahead of playback; its latest frame is not the recovery anchor.
8. At an audio end or accepted abort, discard unsent live frames and preserve
   the last displayed composite. Stop the old audio source on the persistent
   audio transport. Begin a matched return into a raw closed-mouth idle source.
   This does not run a second silence-conditioned MuseTalk pass.
9. Position an idle decoder at the selected target using 16 decode threads.
   Build six transition frames at 20 fps (eight at 24 fps), then continue that
   exact idle phase. Hold the outgoing frame only while this short build runs.
   Record build and total return times independently.
10. Signal motion settled only after all return frames have been emitted.
    A following request can reserve its turn and waits for this signal before
    starting. The initial implementation allows a short wait for recovery;
    it does not begin new phonemes halfway through the optical-flow return.
11. Session close invalidates late recovery work; the worker closes its decoder
    if its result no longer belongs to the session.

Idle and speech use one video sender throughout. Body switching does not reset
RTP timestamps or renegotiate SDP. The opt-in motion path uses steady audio and video deadlines, so small event-loop
scheduling delays do not accumulate into a timestamp jump at a later utterance.
Sub-frame audio leads retain the contiguous video timeline; large mismatches
still use the existing guarded synchronization recovery. Default sessions keep
their original pacing policy. Receiver timestamp checks cover every recorded
packet/frame; see the evidence report for the measured results.

## Reusable character workflow

Use an isolated output directory per character/version. To package an existing
trio, run from `/workspace/MuseTalk`:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python \
  character_factory/scripts/build_realtime_character.py \
  --source-dir /workspace/experiments/japanese_ltx_fixed_distance_20260925 \
  --output-dir /workspace/experiments/realtime_characters/japanese_20260925 \
  --character-id japanese_realtime
```

For a new portrait, replace `--source-dir` with `--image /absolute/portrait.png`.
The wrapper runs the existing native LTX generator with the exact prompt pack,
center crop, and shared anchor. The default keeps the female-specific approved
text. `--subject man` or `--subject person` adapts only identity nouns/pronouns;
the saved derivative is explicitly unreviewed. Motion instructions, negatives,
seeds, graph, and generation settings remain unchanged. A different portrait
still needs visual review; the Japanese approval does not transfer automatically.

Rendering and MuseTalk serving are separate GPU phases on this 12 GB host.
Do not render while the MuseTalk server occupies the GPU. After rendering,
start the server, then run the same packaging command with `--source-dir` and
`--prepare-url http://127.0.0.1:8000` to prepare caches through the API.
Existing matching packages resume; different source hashes require a new output
directory. Cache names include content hashes, preventing a rerender from
silently reusing a previous face cache.

Each output folder contains:

- `character.json`: identity, source hashes, paths, coverage, and preparation status.
- `source-measurements.json`: every measured frame, bound to each input file.
- `motion-atlas.json`: physical transitions and review status.
- `pose-set.json`: test/preparation metadata for six logical IDs and three physical caches.
- `session-pose-set.json`: the actual pose-set payload accepted by session creation.

## Server and mobile integration

For a pilot bank, start the existing local TensorRT server:

```bash
MUSETALK_TRT_PROFILE_ENV_FILE=/workspace/MuseTalk/.runtime/musetalk_trt_local_sm89.env \
WEBRTC_MOTION_ATLAS_DIR=/workspace/experiments/realtime_characters \
WEBRTC_MOTION_ALLOW_UNREVIEWED=1 \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
bash scripts/run_trt_stagewise_server.sh --profile baseline --host 127.0.0.1 --port 8000
```

The registry scans one level of character folders for `motion-atlas.json`.
`WEBRTC_MOTION_ATLAS=/absolute/motion-atlas.json` also supports a single bank.
Use the explicit recorded-review command described in
[package integrity](../character_factory/REALTIME_PACKAGE_INTEGRITY.md) after
normal-speed review. It records the reviewer and evidence hashes and republishes
the atlas with its discovery sidecar. Omit `WEBRTC_MOTION_ALLOW_UNREVIEWED` for
accepted banks. Do not hand-edit status in an indexed atlas or promote a bank
solely because automated diagnostics pass.

The app uses its existing persistent session/offer flow. At session creation,
set `avatar_id` to the package's neutral avatar ID and `pose_set` to the JSON
from `session-pose-set.json`. Send TTS WAV data to
`POST /webrtc/sessions/{session_id}/stream` with a unique `turn_id`, increasing
`seq`, `pose_id=speaking_direct`, `mouth_mode=lip_sync`,
`audio_start=immediate`, and `effective=next_boundary`. Body routing happens
internally; a client does not need to cut videos itself.

For smiling during known speech, use a v2 audio-progress plan with, for example,
`0: speaking_direct`, `350: light_smile`, `650: speaking_direct` in permille.
Set `on_complete=neutral_resting`. To interrupt, send
`POST /webrtc/sessions/{session_id}/events` with
`event=assistant_turn_aborted`, the current turn ID, and a newer sequence number.
A wrong-turn or stale event cannot cancel a newer utterance. Already buffered
receiver audio cannot be retracted; server cancellation stops unsent audio.

The mobile application's repository is not present on this host. These API
contracts are exercised by a real aiortc receiver, rather than a mocked mobile UI.

## Test commands and evidence

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python -m unittest \
  test_motion_transitions test_motion_registry test_webrtc_pose_router \
  test_webrtc_pose_runtime test_webrtc_pose_crossfade test_pose_protocol \
  test_webrtc_av_sync test_webrtc_audio_timeline \
  test_pose_webrtc_recording_timestamps test_pose_webrtc_semantic_timing -q

/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/test_webrtc_motion_transitions.py \
  --asset-dir /workspace/experiments/japanese_ltx_fixed_distance_20260925 \
  --audio-dir /workspace/experiments/japanese_multipose_20260925/audio \
  --pose-set /workspace/experiments/realtime_characters/japanese_20260925/pose-set.json \
  --output /workspace/experiments/japanese_multipose_20260925
```

Run again with `--case edge-cases` for threshold/silence/bridge interruption/
loop playback; add `--legacy --case short-idle` with a separate output directory
to test requests without a v2 pose plan. The harness asserts the observed body
poses and preserves the receiver's audio/video timestamps. Evidence must name
the source bank and retain its hash; no post-render smoothing or dubbed audio
is used to improve the live recording.

Evidence directory: `/workspace/experiments/japanese_multipose_20260925`.
[Final metrics and video links](/workspace/experiments/japanese_multipose_20260925/README.md)
are saved alongside the recordings. The durable
[verification summary](multipose_ltx_musetalk_validation_2026-09-25.json) records
89 passing regression tests, five received recordings, zero detected audio/video
timestamp anomalies, zero observed inference queue underruns, ten completed
returns in 0.349–0.401 seconds, and 1,446 exercised cross-pose candidates.
The initial slower returns, pacing corrections, and prebuffer-cancellation case
are retained separately so the improvements and test corrections remain auditable.
The bank remains a pilot pending normal-speed visual acceptance. The test server
is stopped after verification to release the GPU; the three prepared caches and
all evidence are retained.

## Expanded validation and reusable review (in progress)

The September 25 follow-up uses the accepted Indian closeup V6/V14/V8 sources
without regenerating or changing them. Their common frame size is 480×832 and
source lengths are 241/289/145 frames at 24 fps. The separate package is
`/workspace/experiments/realtime_characters/indian_20260925`.
A fresh Latina image run uses the exact approved Japanese motion prompt pack,
center crop and shared anchor, with separate candidate outputs under
`/workspace/experiments/latina_fixed_distance_multipose_20260925`.
Source approval does not automatically approve either bank's live transitions.

The recording harness now binds each run to the selected routing digest and
all three source hashes, checks actual per-pose frame ranges, requires the first
live generation frame to remain zero, and fails if idle phases freeze during
prebuffer. Optional `--case late-smile` reproduces the terminal-cue regression;
`--case entry-interruption` exercises cancellation before the first phoneme.

After recording, build a source-bound review page with:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/review_motion_evidence.py \
  --directory /absolute/received-recordings \
  --atlas /absolute/character/motion-atlas.json \
  --label "Character live multipose review"
```

This verifies retained audio/video, RTP audits, source routing, entry/return
metrics, and artifact hashes. It creates `verification.json`, `README.md`, and
`review.html`; it does not approve visual quality. Review at normal speed for
visible source changes, frozen expressions, mouth smearing, face displacement,
and hair/shoulder artifacts. Every additional portrait goes through this same
render → integrity → prepare → received-recording → visual-review process.


## Native decoder shutdown and failure containment

The follow-up exposed a reproducible native deadlock before speech entry.
TorchVision enables PyAV's Python FFmpeg logging callback during import. Codec
finalization in installed PyAV 16 can hold the GIL while joining decoder workers;
a worker entering that callback waits for the same GIL. The callback obtains the
GIL before filtering severity, so selecting only ERROR messages does not avoid it.
A CPU-only reproduction on the real idle video hung on its first partial decoder
close. The identical 100-cycle workload completed in 2.53 seconds with FFmpeg's
native logger. Paired logs and both failed live attempts are retained under the
v2/v3 experiment roots.

`scripts/runtime_av_logging.py` restores native FFmpeg ERROR logging after server
imports and again after model initialization. Codec errors remain visible on
stderr. `IdleVideoStreamTrack` separately releases its decode generator, stream
and container during stop/reset, so retained track objects do not retain native
codec workers. Real-codec tests check repeated close/reset with cyclic GC disabled.
Entry setup has a five-second deadline; individual entry flow operations have a
one-second deadline. Their late results cannot enter a newer generation. These
coroutine limits contain responsive-loop failures; they cannot cure a held-GIL
native deadlock on their own.

The temporary diagnostic CPU tuning profile is rejected for these recordings:
its default OpenMP binding pinned the main server and many worker threads to one
core, causing multi-second video returns. Use the original baseline launch shown
above with the native logging fix. The slow experiment is retained separately at
`/workspace/experiments/multipose_validation_20260925_v3/rejected-cpu-profile`.
Do not weaken the half-second return bound to accept that configuration.


## Speech-start transport alignment

A retained receiver failure showed video timestamps jumping from 2.25 to 2.35
seconds when the audio transport was already at 2.36 seconds. This was a runtime
index correction, not network loss. The opt-in motion entry now emits the raw
body at the missing 2.30-second slot and releases generated frame zero at 2.35;
audio remains contiguous at 2.34 silence followed by 2.36 speech. Both transport
counters remain continuous and the initial A/V difference is 10 milliseconds in
the deterministic regression. Catchup slots bypass ordinary pacing sleeps;
regular pacing resumes when the clocks converge. If video leads, silent audio
advances naturally. Reconciliation is limited to half a second and cannot consume
a phoneme or release audio early. Cancellation during this interval belongs to
the original turn and does not disturb the next turn.

The full affected CPU suite now contains 157 passing tests. Current received
verification runs are isolated under
`/workspace/experiments/multipose_validation_20260925_v4`; previous failed native,
CPU-affinity, and RTP-gap runs remain in v2/v3. The transport audit has not been
relaxed to accept missing timestamps or slow returns.


## Final automated follow-up results

The final v4 batch passed **12 real received audio/video recordings** across the
Japanese and Indian characters, including short speech, long talking/smiling,
mid-sentence interruption plus a following turn, pre-speech interruption twice,
starting from smile, late smile cues, silence, threshold-length speech, looping,
and legacy requests. All **20 returns** met the unchanged half-second bound:
**0.3479–0.4999 seconds**. Receiver audits detected no missing timestamps or RTP
discontinuities. This is a single-GPU, sequential-session validation; simultaneous
session capacity was not measured. The slowest return has little margin below
that bound, so these numbers must not be presented as a capacity guarantee.

[Final evidence and review links](/workspace/experiments/multipose_validation_20260925_v4/README.md)
and the [durable validation summary](multipose_ltx_musetalk_validation_followup_2026-09-25.json)
retain the exact artifact hashes and 157-test result. The new clock reconciliation
is covered by a deterministic 60-millisecond-skew regression; the final live batch
did not require extra catchup slots. Earlier failures remain separately archived.

Normal-speed review is still pending; no bank was marked reviewed. The fresh
Latina pilot is also incomplete: idle seeds 193, 194 and 195 all failed the same
3.5-pixel lip-gap gate (58, 86 and 71 frames respectively, out of 241). Seed 195
used identical prompt text and settings and took 5.5 minutes to generate plus
0.5 minutes to decode. These candidates were not prepared as usable avatars.
The reusable tool rejects them; passing software tests does not make a rejected
source visually suitable. No further seed was queued.
