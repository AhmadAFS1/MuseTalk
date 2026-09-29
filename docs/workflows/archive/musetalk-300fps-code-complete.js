export const meta = {
  name: 'musetalk-300fps-code-complete',
  description: 'CPU-only: finish scheduler pipeline, WebRTC/api_server wiring + load harness v2, avatar memory layout, and the pre-change-vs-candidate video A/B tool in the perf worktree; queue GPU validation steps as scripts; static default-path review',
  phases: [
    { title: 'Code', detail: '4 parallel CPU-only coding agents, separate file ownership' },
    { title: 'Review', detail: 'static default-path + race review, fix, consolidate GPU run order' },
  ],
}

const W = '/workspace/MuseTalk-perf300'
const MAIN = '/workspace/MuseTalk'
const PLAN = `${W}/docs/musetalk_4070s_300fps_plan_2026-09-27.md`
const EV = `${W}/docs/fps_comparisons/4070s_300fps_20260927`
const OUT = `${W}/docs/fps_comparisons/4070s_300fps_impl_20260928`
const PY = '/workspace/.venvs/musetalk_trt_stagewise/bin/python'

const CTX = `
## Shared context (read carefully — the setup changed)
Goal: MuseTalk v1.5 server on ONE RTX 4070 SUPER (12 GB) + Ryzen 9 7950X + 30 GB RAM reaching 300 fps aggregate
(15 WebRTC streams x 20 fps; stretch 20) with NO quality change. Plan: ${PLAN}.
- WORK ONLY IN THE GIT WORKTREE ${W} (branch perf/300fps-4070s, WIP commit 2254bc2 contains earlier partial work).
  NEVER edit anything under ${MAIN} — that is the user's clean main checkout; another Claude session (workspace-a7) is
  reworking start scripts there and the user's live server on :8000 runs from it. Do not commit, branch, stash or reset.
  ${W}/models, results, .runtime, logs, calibration/unet_multi_avatar_20260928 are symlinks to the shared data.
- Always run Python from ${W} (cwd) so the worktree's modules are imported; check that no script you touch imports from
  ${MAIN} (grep for hardcoded '/workspace/MuseTalk' without '-perf300'). Python: ${PY}.
- THIS WORKFLOW IS CPU-ONLY. The GPU is reserved for another session right now. Do NOT run anything that initializes CUDA,
  loads the UNet/TAESD on GPU, builds engines, or starts a server. CPU-only unit tests are fine (numpy/cv2/PyAV/aiortc,
  torch on CPU with small tensors) but keep each process < 2 GB RSS and check 'free -g' first: MemAvailable must stay
  >= 4 GB (the user's server + another session's load test are using RAM).
- Instead of running GPU validation yourself, write it as an executable script ${OUT}/<area>/gpu_sequence.sh with named
  steps; every GPU step wrapped as: ${W}/scripts/box_guard.sh run --min-avail-gb N --wait-min 60 --label <area>_<step> -- <cmd>
  (the lease also honors /workspace/.gpu_lease.pause). Each step must print a clear PASS/FAIL line with measured numbers
  and write JSON to ${OUT}/<area>/. Document expected outputs, thresholds, and runtime per step at the top of the script.
- Existing pieces you can build on (in ${W}): scripts/replay_scheduler_exactness.py (golden replay harness, written but not
  yet run), scripts/bench_gpu_path.py (GPU path bench; baseline 260.1 fps sustained), scripts/webrtc_media_flags.py,
  webrtc_live_handoff.py, webrtc_idle_frame_cache.py, webrtc_h264_override.py (partial WebRTC work), api_avatar.py and
  avatar_cache.py (partial memory-layout work), test_avatar_memory_layout.py, unet_stagewise_trt.py (complete, untested),
  vae_fast_decoder.py TaesdTrtBackend (written, untested), ${OUT}/unet_fp16/run_sequence.sh (queued UNet GPU steps).
  Read 'git -C ${W} show --stat 2254bc2' and the files before changing them. Evidence/probe code: ${EV}/.
- Quality rule (user decision): serving/scheduling/memory levers must be EXACT (pre-encoder frames SHA-identical);
  engine rebuilds allowed within FP16 noise under gates. Every change behind a one-line env flag; DEFAULT = today's
  behavior exactly. Required recipe (TAESD + native encoder + 100% chin + refined seam) unchanged.
- Ownership (edit only your files; re-read before each edit; never reformat unrelated code): see your task.
`

const SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    files_changed: { type: 'array', items: { type: 'string' } },
    flags: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, default: { type: 'string' }, effect: { type: 'string' } }, required: ['name', 'default', 'effect'] } },
    cpu_tests: { type: 'array', items: { type: 'object', properties: { test: { type: 'string' }, result: { type: 'string', enum: ['pass', 'fail'] }, detail: { type: 'string' } }, required: ['test', 'result', 'detail'] } },
    gpu_sequence: { type: 'object', properties: { script: { type: 'string' }, steps: { type: 'array', items: { type: 'string' } }, est_minutes: { type: 'number' }, min_avail_gb: { type: 'number' } }, required: ['script', 'steps', 'est_minutes', 'min_avail_gb'] },
    issues: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'files_changed', 'flags', 'cpu_tests', 'gpu_sequence', 'issues'],
}

const TASKS = [
  { key: 'scheduler', prompt: `## Your task: SCHEDULER PIPELINE (plan items 0.2, 1.1, 1.2-hookup, 1.4, 1.9, 1.10, crossfade copy skip).
You own ${W}/scripts/hls_gpu_scheduler.py and ${W}/scripts/replay_scheduler_exactness.py.
Implement behind flags (defaults = today): HLS_GPU_EVENT_TIMING=1 capacity telemetry (GPU busy fraction, idle gap,
callback-blocked time, batch fill, feeder CPU/batch) exposed in get_stats; HLS_GPU_PIPELINE_DEPTH=2 double-buffered async
loop (submit N+1 before collecting N; per-slot pinned staging + pinned output ring; non_blocking D2H + events; all
UNet/TAESD launches stay on the scheduler thread; results of slot k must not be overwritten before its consumers finish);
HLS_SCHEDULER_POLICY=edf with HLS_SCHEDULER_MAX_RUNAHEAD_S and prebuffer-sized packed startup slices; HLS_SKIP_GPU_FOR_RAW=1
(exact_silence + WEBRTC_RAW_IDLE_POSE neutral frames bypass UNet/TAESD and compose raw); skip crossfade source_frame.copy()
when no crossfade is active; WEBRTC_YUV_IN_COMPOSE=1 producer side per the contract already in webrtc_media_flags.py (read
it; objects with .bgr and .yuv420p made by the EXACT PyAV conversion used today). Make replay_scheduler_exactness.py able to
run against EITHER tree (--repo ${MAIN} for the pre-change arm, --repo ${W} for candidates) by setting sys.path/cwd
accordingly, recording SHA-256 of decoded faces and composed pre-encoder frames per job in order, plus optional short
crf<=12 mp4 dumps for the video tool, and a null-sink unpaced mode for throughput (N jobs). CPU tests: unit-test EDF
selection, depth-2 slot bookkeeping and frame ordering with mocked GPU calls. GPU sequence: golden baseline x2 on MAIN (must
reproduce), then each flag alone and all together on the worktree (SHA identical), then null-sink throughput at N=8/12/16
>= 60 s baseline vs candidate.` },
  { key: 'webrtc', prompt: `## Your task: WEBRTC + API WIRING + LOAD HARNESS v2 (plan items 0.4, 0.5, 1.5, 1.7, 1.8, 1.11).
You own ${W}/api_server.py (only the regions for these items), ${W}/scripts/webrtc_tracks.py, webrtc_native_vp8.py,
webrtc_motion_playback.py, webrtc_live_handoff.py, webrtc_idle_frame_cache.py, webrtc_h264_override.py, webrtc_media_flags.py,
and ${W}/load_test_webrtc_v2.py (new). Finish and wire (defaults = today): WEBRTC_NONBLOCKING_HANDOFF in frame_batch_callback
(no .result() blocking; bounded per-track FIFO; drain task on the loop; conversion off the loop via the exact PyAV call or
pre-converted .yuv420p objects); WEBRTC_IDLE_FRAME_CACHE (bit-identical to today's recv() decode; bounded LRU MB budget);
WEBRTC_NATIVE_VP8_THREADS; WEBRTC_H264_IMPL=aiortc|x264tuned|nvenc working override for aiortc 1.14 with NVENC semaphore
(cap 12) + truthful log; WEBRTC_LIFETIME_COUNTERS monotonic counters in /webrtc/sessions/stats; WEBRTC_GROUP_MAX_COUNT
(replace the hardcoded count>12 checks); MUSETALK_DISABLE_LOCAL_TTS; MUSETALK_THREAD_CAPS. Write load_test_webrtc_v2.py:
distinct pre-synthesized WAVs from an --audio-dir (build a >= 20 WAV corpus manifest from existing experiment WAVs; no TTS),
--turns/--chain, per-second polling of lifetime counters (fresh = delta frames_played; held = delta frames_duplicated; stall
seconds), server-side cadence, client sharding (<= 5 peers/process) pinned to physical cores {12-15,28-31} (verify lscpu),
JSON summary with per-stream fresh fraction, max interval, stalls, first-frame latency, aggregate fresh fps, plus server
RSS/threads/CPU/VRAM sampling. Also write a candidate launcher ${W}/experiments/throughput300_candidate/run_candidate_api.sh
(port 8300, oom_score_adj 1000, sources .runtime/musetalk_trt_local_sm89.env then an overlay musetalk_300fps.env with one
lever per line; WEBRTC_VP8_ENCODER=native; local TTS disabled) — it must NOT edit any existing start script. CPU tests:
FIFO/backpressure/drain ordering with a fake loop; I420 equality of cache vs per-session decode on a real idle clip (PyAV CPU);
encoder quality: PSNR/SSIM decoded-vs-source at 2.5 Mbps for today's default vs each candidate on real avatar frames (CPU
encode is fine). GPU sequence: loopback smoke at N=3 on port 8300, then load levels baseline N=1/5/10 and candidate
N=1/5/10/12/15/20 with >= 180 s steady state each (server+clients in one lease).` },
  { key: 'memory', prompt: `## Your task: AVATAR MEMORY LAYOUT (lossless) + RAM attribution. You own ${W}/scripts/api_avatar.py (storage/loading
only), ${W}/scripts/avatar_cache.py, ${W}/test_avatar_memory_layout.py, and new files under ${OUT}/avatar_memory/.
Finish (defaults = today): MUSETALK_AVATAR_MASK_CHANNELS=1 (single-channel masks, exact expansion at every use);
MUSETALK_AVATAR_FRAME_STORE=png|decoded with bounded decoded LRU (MUSETALK_AVATAR_DECODED_LRU_FRAMES); truthful
estimate_memory_usage_bytes for new layouts. Verify composite frames bit-identical (baseline vs new flags) across all cycle
positions incl. mirror boundaries for >= 3 avatars (CPU-only: compose_frame on CPU with saved/zero/random 256x256 faces is
fine) and mutation safety. RAM attribution (CPU): load 1/5/10/all distinct prepared avatars/poses from results/v15/avatars
under baseline vs new flags in a subprocess with a hard RSS cap (stop if MemAvailable < 4 GB) and report RSS/USS per pose and
per identity + load time; measure PNG-on-demand decode CPU ms/frame and a 300 fps core estimate. Produce the capacity table
(20 sessions x 20 distinct avatars on dedicated 30 GB and 36 GB boxes). GPU sequence: only if any part needs GPU (e.g., latent
cycle on device) — otherwise state none.` },
  { key: 'video', prompt: `## Your task: STANDING VIDEO A/B VALIDATION TOOL (user requirement: every optimization round ships video evidence
against the PRE-CHANGE output, repeatable any time). Own NEW files only: ${W}/scripts/video_ab.py, ${W}/scripts/video_ab_chin_render.py,
${W}/experiments/video_validation/README.md.
Design: an 'arm' = (code tree, env flags). PRE-CHANGE arm = code tree ${MAIN} (clean main HEAD, i.e. what ships today) with no
new flags; CANDIDATE arms = tree ${W} + flags. Clips: (1) chin recipe renders for the Japanese and Latina H3 identities (TAESD +
100% chin + refined seam): make video_ab_chin_render.py a copy/adaptation of /workspace/experiments/chin_fps_validation_20260927/run.py
(read it and character_factory/h3_avatar_workflow/{backend.py,render_stage.py,chin.py}) with a --repo argument so the SAME render
logic can import either tree; do not edit the originals; support env flags so candidate backends (MUSETALK_TAESD_BACKEND=trt,
MUSETALK_UNET_BACKEND=trt_stagewise, MUSETALK_TRT_UNET_CUDAGRAPHS) are picked up; record measured render fps per arm;
(2) live scheduler path via scripts/replay_scheduler_exactness.py dumps (--repo per arm) for a 3-pose motion avatar
(chinese_bob_*) and one standard avatar. Output per round/clip: experiments/video_validation/<round>/<clip>_ab.mp4 — same
playback speed (clip native fps, stated), column A pre-change vs column B candidate, full frame on top, nearest-neighbour 3x
mouth zoom below, |A-B|x8 diff panel, burned-in labels (arm, tree, flags, measured fps), crf<=12, <= 20 s; <clip>_ab.json with
per-frame SHA equality, PSNR, max/mean abs LSB (full frame + mouth ROI); contact sheet jpg; and README.md index of all rounds
with a one-line verdict each. Cache pre-change renders under experiments/video_validation/baselines/<clip>/ keyed by main's git
HEAD so they re-render only when main changes. CPU tests: compose the A/B layout from two synthetic frame sequences (no GPU) and
verify panels/labels/JSON. GPU sequence: render pre-change baselines, then rounds 'r1_engines' (TAESD trt; UNet cudagraph;
UNet stagewise), 'r1_serving' (serving flags; expect SHA-identical), each producing its videos.` },
]

phase('Code')
const results = await parallel(TASKS.map(t => () =>
  agent(`${CTX}\n${t.prompt}`, { label: `code:${t.key}`, phase: 'Code', schema: SCHEMA }).then(r => r ? { key: t.key, ...r } : null)))
const done = results.filter(Boolean)
log(`${done.length}/4 coding agents returned`)

phase('Review')
const review = await agent(`${CTX}
## Your task: STATIC REVIEW + GPU RUN ORDER (CPU-only). Reports from the coding agents:
${JSON.stringify(done, null, 1)}
1. Review 'git -C ${W} diff 2254bc2' and 'git -C ${W} diff main' for: any behavior change with all new flags unset (BLOCKER —
   trace each changed default path), races in the non-blocking handoff / double buffering (buffer reuse before D2H completes,
   per-stream order, A/V timestamps), exception paths that could crash the server, thread-affinity of compiled TAESD / CUDA
   graphs, hardcoded ${MAIN} imports in worktree code. Fix clear bugs minimally in the worktree (flag-preserving) and list them.
2. Re-run all CPU tests the agents wrote (and 'python -m py_compile' on every changed file).
3. Consolidate every area's gpu_sequence.sh (plus ${OUT}/unet_fp16/run_sequence.sh and the TAESD gate scripts under
   ${OUT}/taesd_trt/) into ONE ordered runner ${OUT}/RUN_GPU_ALL.sh: order = (a) pre-change golden baselines on MAIN and default-
   flags equivalence on the worktree, (b) TAESD TRT gates+speed, (c) UNet cudagraph gate, (d) UNet stagewise go/no-go + G-UNET +
   bench, (e) combined GPU path bench >= 180 s, (f) serving golden SHA per flag, (g) video A/B rounds, (h) load tests. Each step
   independent-resumable (skip if its PASS JSON exists), wrapped in box_guard, with total estimated minutes and min RAM.
Report in the schema (gpu_sequence = the consolidated runner).`,
  { label: 'review:static', phase: 'Review', schema: SCHEMA })

return { coders: done, review }
