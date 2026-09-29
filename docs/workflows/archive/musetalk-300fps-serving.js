export const meta = {
  name: 'musetalk-300fps-serving',
  description: 'Serving-path + memory optimizations (flag-gated), golden replay exactness harness, standing pre-change vs candidate video A/B tool, live multi-stream WebRTC load test, adversarial review',
  phases: [
    { title: 'Build', detail: 'scheduler pipeline / WebRTC+media / avatar memory layout (parallel, separate files)' },
    { title: 'Evidence', detail: 'standing video A/B tool + live load test (pre-change vs candidate)' },
    { title: 'Review', detail: 'adversarial review of exactness, speed, RAM, video evidence' },
  ],
}

const R = '/workspace/MuseTalk'
const PLAN = `${R}/docs/musetalk_4070s_300fps_plan_2026-09-27.md`
const EV = `${R}/docs/fps_comparisons/4070s_300fps_20260927`
const OUT = `${R}/docs/fps_comparisons/4070s_300fps_impl_20260928`
const PY = '/workspace/.venvs/musetalk_trt_stagewise/bin/python'

const CTX = `
## Shared context (read carefully)
Goal: MuseTalk v1.5 real-time server on ONE RTX 4070 SUPER (12 GB) + Ryzen 9 7950X (16C/32T) + 30 GB RAM must deliver
300 fps aggregate (15 WebRTC streams x 20 fps; stretch 20 streams) with NO quality change. Plan: ${PLAN} (read the items
named in your task; §4 Phase 0/1 and §6 validation). Measured evidence + reusable probe code: ${EV}/ (serving/
webrtc_transport_load*.py loopback aiortc harness, bench_serving_cpu.py; cpu_probe/ combined_bench.py, enc_bench.py;
cpu-post-chin/). Repo ${R}, git branch perf/300fps-4070s — do NOT switch branches, stash, reset or commit (the orchestrator
commits). Python ${PY}. Foundation already exists: ${R}/scripts/box_guard.sh (GPU lease), ${R}/scripts/bench_gpu_path.py
(whole GPU path benchmark; sustained baseline today = 260.1 fps, 30.73 ms per bs8 [UNet 24.22, TAESD 6.35, post 0.13]),
multi-avatar UNet corpus ${R}/calibration/unet_multi_avatar_20260928/.

CONCURRENT WORK YOU MUST NOT COLLIDE WITH:
- Another workflow of this session is implementing GPU engines RIGHT NOW and owns these files: scripts/trt_runtime.py,
  musetalk/models/vae.py, scripts/vae_fast_decoder.py, scripts/avatar_manager_parallel.py, scripts/validate_unet_backend.py,
  scripts/unet_stagewise_trt.py, scripts/build_unet_stagewise.py, and it may add flag support to
  experiments/chin_fps_validation_20260927/run.py and character_factory/h3_avatar_workflow/backend.py. Do NOT edit those.
  Integrate only through their public functions (it adds MUSETALK_TAESD_BACKEND=trt and MUSETALK_UNET_BACKEND=trt_stagewise
  with the same interfaces as today's backends). If you need a hook there, implement it in your own file or report it.
- Another Claude session (workspace-a7) is reworking START SCRIPTS (scripts/run_trt_stagewise_server.sh, vast_*.sh,
  run_*.sh, .runtime/*.env). Do NOT edit any existing start script or .runtime/*.env. New launchers/overlays go in NEW
  files only (e.g. experiments/throughput300_candidate/).
- The user's own server may run on :8000. Never bind 8000, never touch the user's launchers
  (experiments/chinese_bob_webrtc_20260927/run_local_api.sh, run_wall_api.sh). Candidate servers use port 8300+.
- Files may be edited by sibling agents in THIS workflow: re-read a file right before each Edit, change only your own
  regions, never reformat unrelated code.

## Quality rule (user decision): "FP16-noise allowed, gated" — but serving/memory levers must be EXACT
- Every serving/scheduling/memory lever here must be E0: pre-encoder frames SHA-identical to today's behavior (use the
  golden replay harness). Encoder settings changes must keep encoded quality: PSNR/SSIM of decoded vs pre-encode frames
  within 0.2 dB / 0.002 of today's configuration at the same bitrate, measured on real avatar frames.
- EVERY change behind a one-line env flag whose DEFAULT reproduces today's behavior exactly. Record every flag.
- Required recipe (TAESD + native encoder + 100% chin + refined seam) must not change.

## Shared-box rules (hard)
- RAM ~12 GB available (do NOT touch /dev/shm/soulx-lfs-state-20260919). Never let MemAvailable fall below 3 GB.
- Every command that initializes CUDA, loads a model, builds an engine, starts a server, or runs multi-stream load MUST run
  through: ${R}/scripts/box_guard.sh run [--min-avail-gb N] [--wait-min M] -- <cmd>. The lease may be busy for a long time
  (engine builds, and possibly the user's :8000 server): do all code writing, CPU-only unit tests and static checks first,
  then queue GPU steps. Keep each leased command < 25 min.
- Outputs (JSON, logs, small images, short videos) under ${OUT}/<your-area>/; < 60 MB per area; no raw frame dumps.
- Tag numbers [M] measured now / [D] doc / [I] inferred.
`

const IMPL_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    files_changed: { type: 'array', items: { type: 'string' } },
    flags: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, default: { type: 'string' }, effect: { type: 'string' } }, required: ['name', 'default', 'effect'] } },
    measurements: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, value: { type: 'string' }, config: { type: 'string' } }, required: ['name', 'value', 'config'] } },
    gates: { type: 'array', items: { type: 'object', properties: { gate: { type: 'string' }, result: { type: 'string', enum: ['pass', 'fail', 'not-run'] }, value: { type: 'string' }, threshold: { type: 'string' } }, required: ['gate', 'result', 'value', 'threshold'] } },
    artifacts: { type: 'array', items: { type: 'string' } },
    issues: { type: 'array', items: { type: 'string' } },
    next_steps: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'files_changed', 'flags', 'measurements', 'gates', 'artifacts', 'issues', 'next_steps'],
}

const S1 = `${CTX}
## Your task: SCHEDULER PIPELINE (plan items 0.2, 0.3, 1.1, 1.2-hookup, 1.4, 1.9, 1.10, part of 1.5). You own
${R}/scripts/hls_gpu_scheduler.py and new files you create (e.g. scripts/replay_scheduler_exactness.py).
1. FIRST build the golden replay exactness harness (item 0.3): scripts/replay_scheduler_exactness.py drives the real
   HLSGPUStreamScheduler (as api_server/webrtc would) with fixed prepared avatars (include a 3-pose WebRTC motion avatar such
   as chinese_bob_pink_bedroom_* and one standard avatar) and fixed WAVs, for N concurrent jobs, and records SHA-256 of every
   decoded face and every composed pre-encoder frame per job in order (hash in memory; no raw dumps), plus optional short
   lossless/crf<=12 mp4s for video review. Capture the BASELINE (no new flags) and prove it reproduces across 2 runs.
2. Implement, each behind its own flag (defaults = today): CUDA-event timing + capacity telemetry (HLS_GPU_EVENT_TIMING=1;
   GPU busy fraction, idle gap, callback-blocked time, batch fill, feeder CPU per batch); double-buffered async GPU loop
   (HLS_GPU_PIPELINE_DEPTH=2: submit batch N+1 before collecting N; per-slot pinned staging + pinned output ring; non_blocking
   D2H + CUDA events; all UNet/TAESD launches on the one scheduler thread; keep HLS_GPU_STAGE_SYNC_TIMING and
   MUSETALK_VAE_DECODE_TIMING_SYNC off in the candidate config); deadline-aware selection with run-ahead cap
   (HLS_SCHEDULER_POLICY=edf, HLS_SCHEDULER_MAX_RUNAHEAD_S, startup slice = prebuffer, packed into full batches); skip GPU for
   frames whose output is discarded (HLS_SKIP_GPU_FOR_RAW=1: exact_silence and WEBRTC_RAW_IDLE_POSE neutral frames); skip the
   crossfade source_frame.copy() when no crossfade is active; optionally produce YUV420 (exact PyAV conversion, not cv2) in the
   compose workers for WebRTC jobs behind WEBRTC_YUV_IN_COMPOSE=1, coordinating the frame format with the WebRTC agent via the
   interface below.
   INTERFACE CONTRACT with the WebRTC agent: when WEBRTC_YUV_IN_COMPOSE=1 the scheduler passes to frame_batch_callback a list
   of objects exposing .bgr (np.ndarray) and .yuv420p (np.ndarray from av.VideoFrame.from_ndarray(bgr,'bgr24').reformat(
   format='yuv420p').to_ndarray() — bit-identical to today's conversion); when the flag is off, today's BGR list is passed.
   Frame order per job must be preserved.
3. Gates: golden replay SHA-identical to baseline for each flag individually and all together (depth 1 vs 2); per-job frame
   order monotonic; then speed: replay harness in 'unpaced null-sink' mode (no WebRTC) at N=8/12/16 jobs measuring generated
   fps and GPU busy fraction for baseline vs candidate flags, >= 60 s each, via box_guard.
Report in the schema (list every flag).`

const S2 = `${CTX}
## Your task: WEBRTC + MEDIA PATH (plan items 0.4, 0.5, 1.5, 1.6-as-flag-only, 1.7, 1.8, 1.11, 1.12). You own
${R}/api_server.py (only the regions named here), ${R}/scripts/webrtc_tracks.py, ${R}/scripts/webrtc_native_vp8.py,
${R}/scripts/webrtc_motion_playback.py, ${R}/load_test_webrtc.py (or a new load_test_webrtc_v2.py), and new files.
1. Non-blocking handoff (WEBRTC_NONBLOCKING_HANDOFF=1): frame_batch_callback must not block the scheduler thread on
   run_coroutine_threadsafe(...).result(); enqueue via loop.call_soon_threadsafe into a bounded per-track deque with a depth
   counter the scheduler can read; never block on a full strict-FIFO queue. push_bgr_frames_batch accepts pre-converted YUV
   (see the INTERFACE CONTRACT: objects with .bgr and .yuv420p when WEBRTC_YUV_IN_COMPOSE=1) so no conversion runs on the
   event loop; timestamps/A-V sync/sequence unchanged.
2. Shared pre-decoded idle/pose frame cache (WEBRTC_IDLE_FRAME_CACHE=1): decode each idle/motion clip ONCE per process into
   yuv420p arrays shared by all sessions (bit-identical to what recv() produces today); recv() and the motion entry/return
   builders index it instead of decoding on the event loop. Report its RAM per avatar and make it bounded
   (WEBRTC_IDLE_FRAME_CACHE_MAX_MB, LRU across avatars).
3. Encoder: (a) WEBRTC_NATIVE_VP8_THREADS (default = today's) for the native VP8 encoder; (b) replace the dead
   enable_h264_nvenc patch with a working H.264 encoder override for aiortc 1.14 (patch H264Encoder._encode_frame or the codec
   registry) behind WEBRTC_H264_IMPL=aiortc|x264tuned|nvenc (default aiortc = today), with preset/threads flags, a
   process-wide NVENC session semaphore (cap 12) with x264 fallback, and a truthful log line. Measure encoded quality
   (PSNR/SSIM decoded vs pre-encode on real avatar frames at 2.5 Mbps) and CPU/frame for today's defaults vs each candidate;
   recommend defaults that keep quality within the gate.
4. Capacity plumbing: lifetime monotonic per-track counters (WEBRTC_LIFETIME_COUNTERS=1) for frames_played /
   frames_duplicated / strict_video_stall_seconds that do not reset per turn, exposed via /webrtc/sessions/stats; make the
   hardcoded group cap (count > 12 at api_server.py ~3539 and ~3982) an env var WEBRTC_GROUP_MAX_COUNT (default 12);
   MUSETALK_DISABLE_LOCAL_TTS=1 makes /webrtc/tts/kokoro return 503; thread caps (idle decoder threads) behind
   MUSETALK_THREAD_CAPS=1.
5. Load harness v2: extend load_test_webrtc.py (or new load_test_webrtc_v2.py) with --musetalk-fps 20 --playback-fps 20
   --batch-size 8 defaults for the new mode, --audio-dir (distinct pre-synthesized WAVs per session; build a corpus of >= 20
   WAVs from existing experiment WAVs — no local TTS), --turns, --chain (pre-queue each stream's next turn), per-second polling
   of lifetime counters (fresh = delta frames_played, held = delta frames_duplicated, stall seconds), server send-timestamp
   cadence, client sharding (<= 5 peers per client process) and CPU pinning to physical cores {12-15,28-31} (verify lscpu),
   and a JSON summary with per-stream fresh fraction, max interval, stall seconds, first-frame latency, and aggregate fresh fps.
6. Gates: I420 frames pushed to tracks are SHA-identical to today's conversion; A/V offset unchanged; CPU-only unit tests for
   the deque/backpressure; then a loopback smoke test through box_guard with a candidate server (port 8300) at N=3.
Report in the schema (list every flag).`

const S3 = `${CTX}
## Your task: AVATAR MEMORY LAYOUT for many distinct avatars (lossless), and RAM attribution (plan item 0.9 + the RAM fixes).
Context: measured from caches, one 3-pose 512x896 avatar (chinese_bob idle/talking/smiling) holds 874 MB of decoded unique
frames + 345 MB of masks (stored 3-channel; all 3 channels are identical) + ~40 MB latents/plans ≈ 1.25 GB; per session the
wall test showed ~0.43 GB (mostly run-ahead queue). Target: 20 sessions on 20 distinct avatars in ~20 GB.
You own ${R}/scripts/api_avatar.py (storage/loading only; compose_frame's outputs must stay bit-identical),
${R}/scripts/avatar_cache.py, and new files.
1. MUSETALK_AVATAR_MASK_CHANNELS=1 (default 3): store masks single-channel and expand exactly where used (verify every use in
   compose paths incl. attenuate_outer_cheeks/current-phoneme/pose code; composite must be bit-identical).
2. MUSETALK_AVATAR_FRAME_STORE=png|decoded (default decoded): keep unique source frames as their on-disk PNG bytes in RAM and
   decode on demand (cv2.imdecode, measured 5.2 ms/frame at 1 thread) with a small bounded per-avatar decoded LRU
   (MUSETALK_AVATAR_DECODED_LRU_FRAMES) sized so compose never stalls; decode must be bit-identical to today's cv2.imread load.
   Measure CPU cost per frame at 300 fps and whether it moves any GIL-bound path.
3. Fix AvatarCache accounting (estimate_memory_usage_bytes) for the new layouts so admission/eviction budgets stay truthful.
4. RAM attribution harness (CPU-mostly; loading avatars uses no GPU unless the code forces it — if it does, run under box_guard):
   load 1/5/10/20 distinct prepared avatars (use every distinct prepared avatar/pose in results/v15/avatars; if fewer than 20
   distinct identities exist, load all poses and extrapolate per-identity) under baseline vs new flags and report measured
   RSS/USS per avatar and per pose, plus load time. Also report the per-session components you can attribute from code (queue
   depth x frame bytes, decoder state) and propose (do not implement if outside your files) the run-ahead-as-faces change.
5. Gates: composite frames bit-identical (baseline vs new flags) across all cycle positions incl. mirror boundaries for >= 3
   avatars (reuse docs/fps_comparisons/4070s_20260922/verify_dedup_correctness.py approach); mutation-safety check.
Report in the schema, including a table: per-avatar RAM (baseline vs new), and the projected capacity for 20 sessions x 20
distinct avatars on a dedicated 30 GB and 36 GB box.`

phase('Build')
const [s1, s2, s3] = await parallel([
  () => agent(S1, { label: 'build:scheduler', phase: 'Build', schema: IMPL_SCHEMA }),
  () => agent(S2, { label: 'build:webrtc-media', phase: 'Build', schema: IMPL_SCHEMA }),
  () => agent(S3, { label: 'build:avatar-memory', phase: 'Build', schema: IMPL_SCHEMA }),
])
const BUILT = JSON.stringify({ scheduler: s1, webrtc_media: s2, avatar_memory: s3 }, null, 1)

phase('Evidence')
const VIDEO = `${CTX}
## Your task: STANDING VIDEO A/B VALIDATION TOOL (the user's explicit requirement: every optimization round must ship video
evidence against the PRE-CHANGE output, so validations can be repeated constantly). Build it in NEW files only:
${R}/scripts/video_ab.py plus ${R}/experiments/video_validation/README.md (index of rounds). Build results so far:
${BUILT}
Requirements:
- 'arms' are named env-flag sets. The 'pre-change' arm = no new flags (today's behavior); cache its renders per clip under
  ${R}/experiments/video_validation/baselines/<clip>/ with SHA manifests, and re-render/verify it if code changes make it stale.
- Clip sources: (1) the offline chin recipe renders (Japanese + Latina: TAESD + 100% chin + refined seam) by invoking the
  existing harness (experiments/chin_fps_validation_20260927/run.py or character_factory/h3_avatar_workflow render path) as
  a SUBPROCESS with env flags — do not edit those files (another workflow may); if they cannot select the new backends yet,
  record that and render what is possible; (2) the live scheduler path via scripts/replay_scheduler_exactness.py
  (pre-encoder frames) for a 3-pose motion avatar and a standard avatar; (3) optionally a WebRTC receiver recording
  (label 'not frame-aligned').
- Output per round: experiments/video_validation/<round>/<clip>_ab.mp4: same playback speed (the clip's native fps; state
  it), column A pre-change vs column B candidate, full frame on top, nearest-neighbour 3x mouth zoom below, an |A-B|x8 diff
  panel, burned-in labels (arm name, flags, measured fps for that arm), crf<=12 or lossless, <= 20 s; plus <clip>_ab.json
  with per-frame SHA equality, PSNR, max/mean abs LSB on full frame and mouth ROI; plus a contact-sheet jpg.
- Run it now (through box_guard) for round 'r1_serving_memory' with the candidate serving+memory flags from the build
  results, and for round 'r1_engines' with MUSETALK_TAESD_BACKEND=trt / MUSETALK_UNET_BACKEND=trt_stagewise if those backends
  exist and load (check; if the engine workflow has not finished, render what exists and say so).
Report in the schema; list every video path.`

const LOAD = `${CTX}
## Your task: INTEGRATION + LIVE MULTI-STREAM LOAD TEST (plan Phase 1 exit + §6). Build results:
${BUILT}
1. Create a candidate launcher experiments/throughput300_candidate/run_candidate_api.sh (NEW file): port 8300, own log,
   oom_score_adj 1000, sources the repo's generated .runtime/musetalk_trt_local_sm89.env then a NEW overlay
   experiments/throughput300_candidate/musetalk_300fps.env where each lever is one line (so rollback = delete the line).
   Use native VP8 (WEBRTC_VP8_ENCODER=native) as the accepted-quality codec, local TTS disabled, and the engines' new backends
   ONLY if they exist, load, and their gates passed (check ${OUT}/ for the engine workflow results; otherwise keep today's).
2. Through box_guard (server + clients together within one lease; MemAvailable guard; stop the server at the end), run the
   load harness v2 with distinct pre-synthesized WAVs, all streams speaking (chained turns), >= 180 s steady state per level:
   BASELINE (overlay empty = today's config) at N = 1, 5, 10 and CANDIDATE at N = 1, 5, 10, 12, 15 (and 20 if RAM allows).
   Measure per stream: fresh fraction, held frames, max interval, stall seconds, first-frame latency; server: generated fps,
   GPU busy fraction, RSS, threads, CPU cores, VRAM, loop lag. Pass criteria (plan §6.4): per-stream fresh >= 99.5% after
   prebuffer, max interval <= 100 ms, 0 stall seconds, aggregate generated >= 20*N.
3. Report where it saturates and why (which ceiling binds), and the RAM curve vs N.
Report in the schema.`

const [video, load] = await parallel([
  () => agent(VIDEO, { label: 'evidence:video-ab', phase: 'Evidence', schema: IMPL_SCHEMA }),
  () => agent(LOAD, { label: 'evidence:load-test', phase: 'Evidence', schema: IMPL_SCHEMA }),
])

phase('Review')
const review = await agent(`${CTX}
## Your task: ADVERSARIAL REVIEW. Assume nothing is true until you reproduce it. Reports:
BUILD: ${BUILT}
VIDEO: ${JSON.stringify(video, null, 1)}
LOAD: ${JSON.stringify(load, null, 1)}
1. Code review 'git -C ${R} diff' + new files for: default-path changes (anything that alters behavior with flags unset is a
   blocker), race conditions in the non-blocking handoff and double buffering (buffer reuse before D2H completes, frame order,
   A/V timestamps), memory leaks, exception paths, thread-affinity of compiled TAESD/CUDA graphs.
2. Re-run the golden replay harness yourself (through box_guard): baseline twice (reproducible) and candidate flags; confirm
   SHA identity claims. Spot-check 2 video A/B outputs: does the JSON match the video; are labels correct; is column A really
   pre-change (flags unset)?
3. Check the load-test numbers are internally consistent (fresh counters vs duration vs N, box_guard log shows no
   contamination, MemAvailable floor respected).
Fix clear bugs minimally (flag-preserving) and list them; list anything that blocks merging. Report in the schema.`,
  { label: 'review:serving', phase: 'Review', schema: IMPL_SCHEMA })

return { s1, s2, s3, video, load, review }
