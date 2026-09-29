export const meta = {
  name: 'musetalk-300fps-chin-multistream',
  description: 'Multi-stream TAESD + 100% chin render pipeline (bit-exact vs accepted renders), aggregate >=300 fps with the new engines, full quality metrics (lip sync, flicker, masking, chin, overall) and pre-change vs candidate videos',
  phases: [
    { title: 'Build', detail: 'multi-stream chin render harness + quality metrics tool (parallel)' },
    { title: 'Prove', detail: 'exactness vs accepted renders, throughput, quality metrics, videos' },
    { title: 'Verify', detail: 'adversarial verification of every claim' },
  ],
}

const W = '/workspace/MuseTalk-perf300'
const MAIN = '/workspace/MuseTalk'
const PY = '/workspace/.venvs/musetalk_trt_stagewise/bin/python'
const FMPY = '/workspace/SoulX-FlashHead/.venv/bin/python'
const OUT = `${W}/docs/fps_comparisons/4070s_300fps_impl_20260928`
const DIV = '/workspace/experiments/avatar_diversity_20260927'

const CTX = `
## Goal and context (read carefully)
Standing user goal: on this RTX 4070 SUPER reach >= 300 fps for the MuseTalk pipeline with quality the SAME OR BETTER in lip sync,
flicker, face masking, chin alignment and overall video quality. The best previously accepted numbers were ~148-171 fps for the
required recipe (TAESD decoder + native avatar encoder + 100% refined chin alignment, one stream, offline) and ~185-193 fps for chin100 v1.
Required recipe code: ${W}/character_factory/h3_avatar_workflow/{chin.py,backend.py,render_stage.py,tracker_worker.py,verify_baseline.py}
(READ render_stage.py fully: it is the accepted render loop). Do NOT edit anything in character_factory/ (user-owned) or under ${MAIN}.
Accepted reference renders (the PRE-CHANGE ground truth; each has render.json with raw_refined_sha256, generated_faces_sha256,
warm_render_fps; plus cache.pt, masks.npz, source.mp4, source_landmarks.npy, speech.wav, refined_raw.mp4, faces.npz,
generated_landmarks.npy, chin_delta.npy, pixel_checks.json): ${DIV}/{black_man_short_beard,black_woman,east_asian_man_goatee,
middle_eastern_man_full_beard,south_asian_woman,white_man_clean_shaven}/. Their renders used compiled TAESD bs8 + the shipping
TensorRT FP16 bs8 .ts UNet at 24 fps, 240 frames each.
Work in the git worktree ${W} (branch perf/300fps-4070s); run Python from ${W}; Python ${PY} (NumPy 1.23.5, OpenCV 4.9, numba 0.67);
MediaPipe FaceMesh only in ${FMPY} (do not modify that venv). No commits/branches/stashes. Other agents are editing api_server.py,
scripts/hls_gpu_scheduler.py, scripts/webrtc_*.py, scripts/api_avatar.py, scripts/avatar_cache.py, scripts/video_ab*.py, the startup
scripts (scripts/run_musetalk_server.sh, musetalk_host_profile.py, unet_engine_store.py, install_musetalk.sh, vast_*.sh, ...), requirements/,
configs/ — do not edit those.
New GPU backends available (flag-gated): MUSETALK_UNET_BACKEND=trt_stagewise + MUSETALK_UNET_STAGEWISE_BATCH=16 (engines built at
${W}/models/tensorrt_unet_stagewise_sm89/bs16; 2.42 ms/frame sustained; G-UNET pass mae_max 0.0025 max_abs 0.431 vs shipping .ts
0.0026/0.402), MUSETALK_TAESD_BACKEND=trt (TaesdTrtBackend in scripts/vae_fast_decoder.py; its gates are being run right now — check
${OUT}/taesd_trt/*.json and window2_sequence.out for PASS/FAIL before relying on it), MUSETALK_TRT_UNET_CUDAGRAPHS=manual (check
${OUT}/unet_fp16/ results). Load backends via scripts/vae_fast_decoder.load_taesd_decoder and scripts/trt_runtime.load_unet_trt_backend
from the worktree (the accepted backend.setup() hardcodes /workspace/MuseTalk — write your own setup that uses ${W}).
Shared box rules: EVERY GPU/model/heavy command through ${W}/scripts/box_guard.sh run --min-avail-gb N --wait-min 90 --label <x> -- <cmd>
(serializes with other GPU jobs; each leased command < 25 min). RAM ~8-12 GB available; never below 3 GB. Outputs under
${OUT}/<your-area>/ (< 80 MB; encode videos crf<=12, no raw dumps). Tag numbers [M]/[D]/[I].
Quality rule (user decision): pipeline/concurrency/scheduling changes must be BIT-EXACT; FP16 engine changes allowed only within
gates + video review; the chin algorithm (chin.py) must run unchanged (or a proven bit-exact port).
`

const SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    files_changed: { type: 'array', items: { type: 'string' } },
    measurements: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, value: { type: 'string' }, config: { type: 'string' } }, required: ['name', 'value', 'config'] } },
    gates: { type: 'array', items: { type: 'object', properties: { gate: { type: 'string' }, result: { type: 'string', enum: ['pass', 'fail', 'not-run'] }, value: { type: 'string' }, threshold: { type: 'string' } }, required: ['gate', 'result', 'value', 'threshold'] } },
    artifacts: { type: 'array', items: { type: 'string' } },
    issues: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'files_changed', 'measurements', 'gates', 'artifacts', 'issues'],
}

phase('Build')
const [harness, metrics] = await parallel([
  () => agent(`${CTX}
## Your task: MULTI-STREAM CHIN RENDER HARNESS. Own new files: ${W}/scripts/chin_multistream_render.py (+ helper modules you create
under ${W}/scripts/chin_multistream/), outputs in ${OUT}/chin_multistream/.
Design (keep the per-stream math identical to render_stage.py):
- N streams, each = one accepted identity dir (streams may reuse identities; --loops K repeats each stream's 240 frames so timed runs
  last >= 60 s). Precompute per stream exactly as render_stage.py does (chin.prepare_refined etc.) BEFORE timing.
- ONE GPU process thread issues all UNet/TAESD work: batches drawn round-robin across streams; --align stream8 keeps each bs8 batch
  inside one stream at offsets base..base+8 exactly like render_stage.py (required for the exactness gate); --pack 16 packs two
  stream-aligned 8-frame chunks into one bs16 call for the stagewise engine. uint8 faces via the exact same postprocess as
  render_stage.generate(). Double-buffer GPU batches (submit next before consuming current) without changing results.
- One ordered WORKER PROCESS per stream (spawned BEFORE any CUDA init, TRT venv so NumPy 1.23.5 math), owning its Tracker
  (character_factory/h3_avatar_workflow/backend.Tracker via import — do not edit it; it spawns the FaceMesh subprocess), its
  3-tap delta filter state and chin.corrected_refined compose, identical to render_stage.emit(); faces delivered through shared memory
  rings with backpressure; each worker hashes its raw refined frames and generated faces (sha256 over tobytes in order) and optionally
  encodes crf<=12 mp4 of the first 240 frames for video review.
- Aggregate fps = total refined frames / wall time from first GPU submit to last frame composed (exclude prep/load), plus per-stage
  telemetry (GPU busy, worker idle/backpressure, tracker ms, compose ms, cores used).
Gates to run (through box_guard): (E) EXACTNESS — baseline backends (compiled TAESD + shipping .ts bs8, --align stream8), N=6 (all six
identities) x loops 1: every stream's raw_refined_sha256 AND generated_faces_sha256 must equal that identity's render.json.
(T) THROUGHPUT, >= 60 s timed each, report median of 2 runs: (a) baseline backends N=6; (b) stagewise bs16 + compiled TAESD N=6 and
N=12; (c) stagewise bs16 + TRT TAESD (only if its gates passed) N=6 and N=12. Also re-measure single-stream serial fps with baseline
backends (should reproduce ~150-170) so the before/after is apples to apples. (C) candidate-backend outputs per stream saved (faces +
refined frames first 240, encoded crf<=12) for the quality comparison.
Report in the schema.`, { label: 'build:chin-harness', phase: 'Build', schema: SCHEMA }),
  () => agent(`${CTX}
## Your task: QUALITY METRICS TOOL (the quality half of the goal). Own new files: ${W}/scripts/quality_ab_metrics.py (+ a FaceMesh helper
run with ${FMPY} if needed), outputs ${OUT}/quality_metrics/.
Input: two renders of the same identity/audio (A = accepted pre-change refined_raw.mp4 or raw frames; B = candidate), plus the identity's
source.mp4, source_landmarks.npy, cache/masks as needed. Output JSON + a short markdown table per identity with, for BOTH arms and their
delta: (1) LIP SYNC — FaceMesh inner-lip aperture time series (normalized by eye spacing) per arm; Pearson correlation A vs B, lag of
max cross-correlation (must be 0), mean |delta aperture| px; optional SyncNet confidence/offset per arm using models/syncnet
(uncalibrated: report relative only; GPU use via box_guard, small); (2) FLICKER — mean |frame(t)-frame(t-1)| in the mouth ROI and in the
chin/jaw band, per arm, ratio B/A (and high-frequency temporal energy); (3) FACE MASKING / SEAM — per-frame max/mean abs difference
A vs B inside the blend-mask band and along the mask boundary (dilated edge ring), plus protected-lip pixels change (must follow
pixel_checks semantics); (4) CHIN ALIGNMENT — chin-target error using character_factory/h3_avatar_workflow/validate_stage.py's
formula (read it) per arm, and FaceMesh jaw+lip landmark deviation A vs B (mean, p99 px); (5) OVERALL — PSNR/SSIM A vs B (full
frame, face bbox, mouth ROI), mouth sharpness ratio (Laplacian variance) B/A, color shift (mean Lab delta in face ROI).
Define PASS thresholds (document them) consistent with the plan's gates: aperture corr >= 0.97 and lag 0; flicker ratio <= 1.05;
chin-target error B <= A + 0.05 px; landmark deviation mean <= 0.05 px / p99 <= 0.15 px for FP16-noise changes; protected lips unchanged
relative to each arm's own standard compose; sharpness ratio >= 0.95; PSNR report-only. CALIBRATE noise floors now on existing data:
A vs itself (must be perfect), and two re-encodes of the same raw (codec noise), and the accepted INT8-vs-TAESD pair in
/workspace/experiments/chin_fps_validation_20260927/ (a known visible-but-accepted difference) to show metric sensitivity. Run on the six
diversity identities (A = accepted refined_raw.mp4 vs B = accepted standard_raw.mp4 is another sensitivity check). CPU-only except the
optional SyncNet (box_guard). Report in the schema.`, { label: 'build:quality-metrics', phase: 'Build', schema: SCHEMA }),
])

phase('Prove')
const prove = await agent(`${CTX}
## Your task: RUN THE QUALITY COMPARISON + VIDEOS for the candidate multi-stream renders. Harness report: ${JSON.stringify(harness, null, 1)}
Metrics tool report: ${JSON.stringify(metrics, null, 1)}
1. For each of the six identities: A = accepted pre-change render (${DIV}/<id>/refined_raw.mp4 / raw frames), B = the best candidate
   configuration that reached >= 300 fps aggregate (and, separately, every other candidate config the harness measured). Run
   scripts/quality_ab_metrics.py; produce a per-identity + overall PASS/FAIL table against the documented thresholds.
2. Videos (user requirement): per identity, a labelled same-speed (24 fps, the render's native fps) side-by-side: column A pre-change
   accepted render, column B candidate; full frame on top, nearest-neighbour 3x mouth zoom below, |A-B|x8 diff panel; burned-in labels
   with backend names and MEASURED fps (A: its render.json warm_render_fps single stream; B: aggregate multi-stream fps and per-stream
   share); crf<=12; plus one 6-identity mosaic summary video. Write to ${W}/experiments/video_validation/r2_chin_multistream/ with a
   README.md (what each column is, the numbers, the verdict) and add a line to ${W}/experiments/video_validation/README.md if it exists
   (append only).
3. If any metric fails for a candidate config, try the next-best config (e.g. compiled TAESD instead of TRT TAESD) and report which
   configuration satisfies BOTH >= 300 fps and all quality gates. Be honest if none does.
Report in the schema.`, { label: 'prove:quality-video', phase: 'Prove', schema: SCHEMA })

phase('Verify')
const verify = await agent(`${CTX}
## Your task: ADVERSARIAL VERIFICATION. Assume nothing until reproduced. Reports:
HARNESS: ${JSON.stringify(harness, null, 1)}
METRICS: ${JSON.stringify(metrics, null, 1)}
PROVE: ${JSON.stringify(prove, null, 1)}
1. Re-run the exactness gate yourself (N=6 baseline backends) and one >= 60 s throughput run of the winning config; confirm the numbers
   (fps definition: refined frames / wall incl. all GPU+CPU work, excluding one-time load/prep) and that no frames were skipped/dropped
   (count per stream, hash chain), no tracking was skipped, chin applied to 100% of frames, and the chin code executed is chin.py unchanged.
2. Audit the quality metrics: are thresholds sensible, computed on the right rows/ROIs, A really the pre-change accepted render, B really
   the candidate output; spot-check 2 videos (labels match the JSON; columns not swapped).
3. State plainly: does the pipeline now sustain >= 300 fps on this GPU with the required recipe and same-or-better quality on all five
   quality dimensions? What caveats remain (e.g., offline multi-stream vs live WebRTC; FP16-noise engine changes)?
Fix clear bugs minimally in the harness/metrics files (not in character_factory). Report in the schema.`,
  { label: 'verify:chin-multistream', phase: 'Verify', schema: SCHEMA })

return { harness, metrics, prove, verify }
