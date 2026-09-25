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
Duplicate matching banks are rejected. Parsed atlases are cached by their exact
contents, because this filesystem can keep the same modification timestamp
across rapid writes. Never trust timestamps alone for review-status changes.

## Runtime algorithm

1. Normalize the public TTS metadata and reserve its existing session stream.
   A v1 request gets a default internal motion plan; a v2 request preserves its
   requested speech/smile cues. The public speech allowlist stays compatible.
2. Obtain TTS duration from the actual prepared generation timeline. Below
   three seconds, every body frame is idle. At or above three seconds, begin
   with 0.3 seconds of idle, then select an admissible expressive entry.
3. For known-duration audio, start the final idle return 0.5 seconds before
   the end: a 0.3-second bridge plus approximately 0.2 seconds of idle lipsync.
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
After recorded visual review, set the specific bank's `status` to `reviewed`
and omit `WEBRTC_MOTION_ALLOW_UNREVIEWED`. Do not promote a bank solely because
its automated diagnostics pass.

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
