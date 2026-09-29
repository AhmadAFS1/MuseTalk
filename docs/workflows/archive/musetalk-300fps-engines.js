export const meta = {
  name: 'musetalk-300fps-engines',
  description: 'Implement box guard + UNet validation corpus, TensorRT TAESD backend, stagewise FP16 UNet backend (go/no-go bench first), then independently verify gates, speed and quality',
  phases: [
    { title: 'Foundation', detail: 'box_guard/GPU lease, multi-avatar UNet corpus, sustained baseline' },
    { title: 'Engines', detail: 'TRT TAESD backend + stagewise FP16 UNet backend (parallel code, serialized GPU)' },
    { title: 'Verify', detail: 'independent re-run of gates, sustained combined fps, chin/landmark parity, labelled video' },
  ],
}

const R = '/workspace/MuseTalk'
const EV = `${R}/docs/fps_comparisons/4070s_300fps_20260927`
const PLAN = `${R}/docs/musetalk_4070s_300fps_plan_2026-09-27.md`
const PY = '/workspace/.venvs/musetalk_trt_stagewise/bin/python'
const OUT = `${R}/docs/fps_comparisons/4070s_300fps_impl_20260928`

const CTX = `
## Shared context (read carefully)
You are implementing part of a plan to raise a MuseTalk v1.5 real-time talking-head server on ONE RTX 4070 SUPER
(12 GB, 220 W cap, sm_89) + Ryzen 9 7950X + 30 GB RAM to 300 fps aggregate generation (15 streams x 20 fps) WITHOUT
quality loss. The full plan (read the sections relevant to your items): ${PLAN}. Measured evidence, probe scripts and
JSON you should reuse rather than rewrite: ${EV}/ (EVIDENCE_DIGEST.md; unet_probe/ incl. probe_trt_topblocks.py which
already builds the 11 per-block FP16/INT8 TRT engines from in-RAM ONNX with correct Down/Mid/Up/Head/Tail wrappers;
taesd_probe/ incl. p5_trt_taesd.py (TRT TAESD build) and p6_combined.py (whole GPU path timing harness);
unet_probe/tt16_timing_cache.bin). Repo: ${R} on git branch perf/300fps-4070s (do NOT switch branches, do NOT commit —
the orchestrator commits). Python: ${PY} (torch 2.5.1+cu121, torch_tensorrt 2.5.0, TensorRT 10.3, modelopt 0.23.2,
NumPy 1.23.5, OpenCV 4.9). MediaPipe FaceMesh lives only in /workspace/SoulX-FlashHead/.venv/bin/python (do not modify
that venv).

## The user's quality rule (decided): "FP16-noise allowed, gated"
- Bit-exact levers must produce SHA-identical frames.
- FP16 runtime changes (TRT rebuilds) are allowed only if they pass: UNet gate scripts/validate_unet_backend.py
  mae_max <= 0.01 and max_abs <= 0.5 on real captures (multi-avatar corpus); decoded face <= 3 LSB max and <= 0.2 LSB
  mean vs today's compiled TAESD; for anything that feeds the chin tracker: FaceMesh jaw+lip landmark deviation mean
  <= 0.05 px, p99 <= 0.15 px vs the reference, chin-target error unchanged within 0.05 px; and a labelled side-by-side
  video for the user's review.
- EVERY change sits behind a one-line env flag whose DEFAULT reproduces today's behavior exactly (the live server must
  behave identically when no new flag is set). Record every flag you add.
- The required output recipe (TAESD decoder + native avatar encoder + 100% chin + refined seam) must not change.

## Shared-box rules (hard; a previous run OOM-killed the user's server)
- RAM: ~12 GB available (7.9 GB of /dev/shm is another project's state and must NOT be touched). Never let
  MemAvailable fall below 3 GB. Disk: ~25 GB free; keep your total new writes under 4 GB and delete scratch files.
- The GPU may be shared with other agent sessions (Codex, SoulX) that do not know our lease. Before GPU work, check
  'nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv' for foreign processes; if one is active,
  wait (poll every 30 s, up to 30 min) and report it.
- Do NOT start or modify the user's launchers (experiments/chinese_bob_webrtc_20260927/run_local_api.sh,
  run_wall_api.sh) and do not bind port 8000. No server is running now.
- EVERY command that initializes CUDA, loads a model, builds an engine or benchmarks must run through the GPU lease:
  ${R}/scripts/box_guard.sh run [--min-avail-gb N] -- <command>   (created in the Foundation phase; it holds
  flock /workspace/.gpu_lease, so parallel agents serialize). Keep each leased command bounded (< 25 min) so others
  are not starved; split long jobs.
- Put measurement outputs (JSON, logs, small images) under ${OUT}/<your-area>/ and keep them small (< 50 MB per area,
  no raw frame dumps; encode any video losslessly or at crf<=12 and keep it short).
- Timing methodology: CUDA events, warmup, median; sustained runs report duration, clocks and power (nvidia-smi 1 Hz).
  Tag every number [M] (you measured it now) or [D]/[I].
`

const IMPL_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    files_changed: { type: 'array', items: { type: 'string' } },
    flags: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, default: { type: 'string' }, effect: { type: 'string' } }, required: ['name', 'default', 'effect'] } },
    measurements: { type: 'array', items: { type: 'object', properties: { name: { type: 'string' }, value: { type: 'string' }, config: { type: 'string' }, source: { type: 'string' } }, required: ['name', 'value', 'config'] } },
    gates: { type: 'array', items: { type: 'object', properties: { gate: { type: 'string' }, result: { type: 'string', enum: ['pass', 'fail', 'not-run'] }, value: { type: 'string' }, threshold: { type: 'string' } }, required: ['gate', 'result', 'value', 'threshold'] } },
    artifacts: { type: 'array', items: { type: 'string' }, description: 'engines, corpora, videos, JSON written (paths + sizes)' },
    issues: { type: 'array', items: { type: 'string' } },
    next_steps: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'files_changed', 'flags', 'measurements', 'gates', 'artifacts', 'issues', 'next_steps'],
}

phase('Foundation')
const foundation = await agent(`${CTX}

## Your task: FOUNDATION (plan items 0.1, 0.7, part of 0.8)
1. Create ${R}/scripts/box_guard.sh (bash, stdlib tools only; executable) implementing the plan's item 0.1:
   - 'box_guard.sh check' prints GPU compute apps, GPU util/mem, MemAvailable, /dev/shm usage, disk free, load avg,
     cgroup oom_kill count, and co-tenant processes (api_server, run_wall_api, dev_server, drive_wall, trtexec, python
     processes using the GPU), and exits non-zero if the box is not quiet.
   - 'box_guard.sh run [--min-avail-gb N (default 6)] [--need-disk-gb N (default 1)] [--wait-min M (default 30)] -- cmd...'
     acquires 'flock /workspace/.gpu_lease' (waiting up to M minutes), waits for foreign GPU processes to disappear,
     checks MemAvailable and disk thresholds, records memory.events oom_kill before/after, sets oom_score_adj=1000 for
     the child, runs a watchdog that kills the child process group if MemAvailable < 3 GB, and returns the child's exit
     code (non-zero + clear message if the watchdog fired or oom_kill rose). Test it with a trivial command and with a
     second concurrent invocation to prove serialization.
2. Build a MULTI-AVATAR UNet validation corpus (item 0.7) usable by ${R}/scripts/validate_unet_backend.py:
   - Read validate_unet_backend.py and hls_gpu_scheduler.py (_capture_unet_calibration_batch, MUSETALK_UNET_CALIBRATION_*
     flags) to learn the exact capture file format; the existing single-avatar captures are in
     /workspace/benchmarks/same-avatar/unet-captures/.
   - Produce real scheduler-equivalent UNet inputs (masked+reference latents [B,8,32,32], whisper audio prompts
     [B,50,384] with the SAME positional encoding the live path applies) for >= 8 distinct prepared avatars from
     ${R}/results/v15/avatars (include the Japanese and Latina H3 talking avatars used by the chin experiments and
     several diversity-batch identities if prepared; prefer H3/expressive sources) and >= 2 different audio files from
     ${R}/data/audio or experiment WAVs; 16 bs8 batches per avatar is enough. Either drive the real scheduler capture
     path or replicate it exactly (prefer reusing repo code such as docs/fps_comparisons/4070s_20260922/e2e_indian.py or
     the chin harness's input building); save in the validator's format under ${R}/calibration/unet_multi_avatar_20260928/
     (hold out 3 avatars in a separate subfolder 'holdout/'). Keep it < 400 MB.
   - Prove the corpus works: run validate_unet_backend.py against the CURRENT shipping TRT UNet
     (models/tensorrt_unet_sm89_bs8_local/unet_trt.ts) on the corpus and record mae_max/max_abs (expect ~0.002/0.15).
3. Sustained baseline (part of item 0.8, L1-equivalent without a server): adapt ${EV}/taesd_probe/p6_combined.py into
   ${R}/scripts/bench_gpu_path.py, a reusable benchmark of the whole live-equivalent GPU path per batch
   (H2D of real inputs -> UNet backend as selected by the repo loaders/env flags -> TAESD backend as selected -> the
   repo's uint8 BGR postprocess -> pinned D2H), with options --batch, --seconds, --warmup, and 1 Hz nvidia-smi logging
   of clocks/power/temp. It must load backends through the repo's own loaders (scripts/trt_runtime.py
   load_unet_trt_backend / load_vae_trt_decoder or avatar_manager_parallel equivalents) so later backends are picked up
   by env flags. Run it for the current configuration (TRT .ts UNet bs8 + compiled TAESD) for >= 180 s and report
   sustained ms/frame and fps, with clock/power stats. This is the true sustained baseline.
Write results under ${OUT}/foundation/. Report everything in the schema.`,
  { label: 'foundation', phase: 'Foundation', schema: IMPL_SCHEMA })

if (!foundation) return { foundation: null }
log('Foundation done: ' + foundation.summary.slice(0, 200))

phase('Engines')
const FOUND = `\n## Foundation results you can rely on\n${JSON.stringify({ summary: foundation.summary, artifacts: foundation.artifacts, flags: foundation.flags, measurements: foundation.measurements, issues: foundation.issues }, null, 1)}\n`

const TAESD_TASK = `${CTX}${FOUND}
## Your task: TensorRT TAESD backend (plan item 2.1). Files you own: ${R}/scripts/vae_fast_decoder.py (add), and the
minimal dispatch needed in ${R}/scripts/trt_runtime.py::load_vae_trt_decoder and/or ${R}/musetalk/models/vae.py
decode paths. Another agent is concurrently editing trt_runtime.py (UNet loader region only, and a new file
scripts/unet_stagewise_trt.py): re-read files right before editing, touch only the VAE-related functions, and never
reformat unrelated code.
- Implement TaesdTrtBackend with the same public interface as TaesdVaeDecodeBackend (read vae_fast_decoder.py and how
  vae.py / hls_gpu_scheduler.py / avatar_manager_parallel.py / character_factory/h3_avatar_workflow/backend.py and
  experiments/chin_fps_validation_20260927/run.py call the decoder), selected by MUSETALK_TAESD_BACKEND=trt (default
  'compiled' = today's behavior). FULL-HEIGHT decode (the staged row-crop is NOT allowed: the chin tracker reads the
  full generated face).
- FP16 TRT engine built from the vendored TAESD decoder (models/taesd). Provide (a) an fp16 NCHW output path for callers
  that need tensors and (b) a fused uint8 BGR NHWC output path that reproduces the repo's fast postprocess
  (vae.py MUSETALK_VAE_FAST_POSTPROCESS: /2+0.5, clamp, x255, rounding mode, channel flip, layout) EXACTLY given the
  same fp16 decoder output — verify the fused post is bit-identical to the repo post applied to the engine's fp16 output.
- Batch handling: static bs8 profile (measured 0.486 ms/frame at bs8 vs 0.502 at bs16), run larger batches as bs8
  sub-batches; handle partial batches correctly (pad or a second profile).
- Persist the engine (~3 MB) under models/taesd/trt/ keyed by a fingerprint (TRT version, GPU name, onnx hash, batch);
  build on first use if missing; verify at load with a fixed probe batch whose output hash is stored alongside; fall back
  (or refuse, per flag MUSETALK_TAESD_TRT_STRICT) on mismatch. Warmup must not compile anything on the live path.
- Gates (report numbers): G-TAESD vs today's compiled TAESD on >= 256 real post-UNet latents from the multi-avatar corpus
  (decode the corpus inputs through the current UNet, or use stored outputs): max and mean abs LSB on the full 256x256
  face and on rows >= 104; bit-exact check of the fused post vs repo post. Also check torch.compile TAESD vs eager to
  report the noise floor already present in the shipping path.
- Speed: ms/frame at bs8/bs16 standalone and inside ${R}/scripts/bench_gpu_path.py with the shipping .ts UNet for >= 120 s
  sustained (via env flags).
Report in the schema.`

const UNET_TASK = `${CTX}${FOUND}
## Your task: FP16 UNet speedups (plan items 1.2, 2.2, 2.3a, 2.3b). Files you own: new ${R}/scripts/unet_stagewise_trt.py,
the UNet-loader region of ${R}/scripts/trt_runtime.py (TrtUnetBackend / MultiTrtUnetBackend / load_unet_trt_backend),
and ${R}/scripts/tensorrt_export.py if needed. Another agent is concurrently editing the VAE parts of trt_runtime.py and
scripts/vae_fast_decoder.py: re-read before each edit, touch only UNet code, never reformat unrelated code.
Step A — CUDA graph for the shipping .ts UNet (item 1.2): add MUSETALK_TRT_UNET_CUDAGRAPHS=manual|runtime|0 (default 0) to
TrtUnetBackend: capture the call in torch.cuda.CUDAGraph with static input/output buffers; CLONE outputs whenever they
could outlive the next replay (check every caller incl. calibration capture). Gate: max_abs 0 vs non-graph on the
corpus, an overwrite test with two different batches, and speed.
Step B — GO/NO-GO bench (item 2.2) BEFORE writing the backend: build the 11 FP16 per-block engines through the ONNX
parser (reuse ${EV}/unet_probe/probe_trt_topblocks.py wrappers; builder optimization level 5 if build time allows; reuse a
timing cache), build them sequentially in RAM with RSS measured on the first block, at bs16 and bs8, chain them exactly as
a stagewise backend would (skip tensors, shared activation memory, execute_async_v3, one CUDA graph over all enqueues),
and time each chain for >= 60 s sustained at the power cap with clocks/power logged. Record VRAM. Decision (plan §1):
bs16 <= 2.51 ms/frame => GO; 2.51-2.64 marginal (still implement, report); > 2.64 => no-go for bs16 (then implement the
bs8 stagewise only if it is <= 2.60 ms/frame, else stop and report). Also run G-UNET for the chained engines vs PyTorch
FP16 on the multi-avatar corpus (mae_max <= 0.01, max_abs <= 0.5).
Step C — Backend (items 2.3a/2.3b), if Step B allows: implement StagewiseTrtUnetBackend in scripts/unet_stagewise_trt.py
with the same call interface the scheduler uses for TrtUnetBackend (read hls_gpu_scheduler.py and trt_runtime.py to match
inputs, dtypes, timesteps handling, return type), multi-input/multi-output bindings for skip tensors, one shared
activation arena across the 11 contexts, CUDA-graph replay with cloned outputs, batch = MUSETALK_UNET_STAGEWISE_BATCH
(16 or 8) with correct handling of partial/padded batches and of larger batches (split). Selected by
MUSETALK_UNET_BACKEND=trt_stagewise (default unchanged). Engines persisted under
models/tensorrt_unet_stagewise_sm89/bs<N>/ with a manifest (TRT version, GPU, per-block onnx hash, build flags, probe
output hash); build-if-missing tool: scripts/build_unet_stagewise.py (runs under box_guard). Build each engine set twice and
keep the faster if time allows (tactic variance ~5%). Make sure the eager PyTorch UNet does not need to stay resident if
the server frees it (note but do not change avatar_manager_parallel unless trivial and flag-gated as
MUSETALK_FREE_EAGER_UNET=1, default 0).
Gates: validate_unet_backend.py on the corpus (train + holdout) for the new backend; SHA-identical results for the
default path (flags unset). Speed: ${R}/scripts/bench_gpu_path.py with the new UNet backend + compiled TAESD for >= 120 s.
Report in the schema, including the go/no-go numbers.`

const [taesd, unet] = await parallel([
  () => agent(TAESD_TASK, { label: 'impl:taesd-trt', phase: 'Engines', schema: IMPL_SCHEMA }),
  () => agent(UNET_TASK, { label: 'impl:unet-stagewise', phase: 'Engines', schema: IMPL_SCHEMA }),
])

phase('Verify')
const verify = await agent(`${CTX}
## Your task: INDEPENDENT ADVERSARIAL VERIFICATION of the engine work. Assume nothing the implementers reported is true
until you reproduce it. Implementer reports:
FOUNDATION: ${JSON.stringify(foundation, null, 1)}
TAESD-TRT: ${JSON.stringify(taesd, null, 1)}
UNET: ${JSON.stringify(unet, null, 1)}

Do all of the following (through the GPU lease):
1. Code review of every changed file (git -C ${R} diff; new files): correctness bugs, CUDA-graph buffer aliasing, partial
   batch handling, thread-safety assumptions, fallbacks, default-path behavior unchanged. Fix clear bugs yourself
   (minimal, flag-preserving) and list them.
2. Default-path equivalence: with NO new flags set, confirm the UNet and TAESD outputs on corpus batches are identical
   (max_abs 0) to the pre-change code (use 'git -C ${R} stash' is FORBIDDEN — instead compare against outputs computed by
   loading the original functions from 'git show HEAD:<file>' into a temp module, or against stored reference outputs).
3. Re-run the gates yourself: validate_unet_backend.py for each new UNet mode (train + holdout corpus); G-TAESD.
4. G-TRACK + chin parity: extend the offline chin harness (experiments/chin_fps_validation_20260927/run.py and/or
   ${R}/character_factory/h3_avatar_workflow/backend.py; minimal flag-driven changes so it can use the new backends via
   the same env flags) and render the Japanese and Latina identities (the harness's default clips, TAESD + 100% chin +
   refined seam if the harness supports it; otherwise chin100) with (a) the current backends and (b) the best new
   backends. Compare: FaceMesh jaw+lip landmark deviation (mean, p99), chin-target error, protected-lip pixels unchanged,
   and per-frame face PSNR/max LSB between (a) and (b). Thresholds from the quality rule.
5. Sustained combined speed: ${R}/scripts/bench_gpu_path.py with the best new UNet backend + TRT TAESD for >= 180 s, and
   the baseline config for >= 180 s back to back, same conditions; report sustained ms/frame, fps, clocks, power, VRAM.
   State plainly whether the GPU path now sustains >= 300 fps and by what margin.
6. Labelled comparison video (the user's standing requirement): for both identities, a same-speed (20 fps playback is
   fine; the harness renders 24 fps sources — keep the harness's native fps and say so) side-by-side of (a) current vs (b)
   new backends, full frame on top and a nearest-neighbour 3x mouth zoom below, each column labelled with backend names
   and the measured sustained GPU fps; plus an |a-b|x8 diff panel. Write to ${OUT}/verify/ (keep < 60 MB).
Report in the schema; in 'gates' include every gate with pass/fail; in 'issues' list anything that should block merging.`,
  { label: 'verify:engines', phase: 'Verify', schema: IMPL_SCHEMA })

return { foundation, taesd, unet, verify }
