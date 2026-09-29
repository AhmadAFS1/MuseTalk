export const meta = {
  name: 'musetalk-startup-rework',
  description: 'CPU-only implementation of the start/install rework (resolver, engine store incl. stagewise+TAESD-TRT engines, pinned installer cu121/cu128, launch chain) in the perf worktree, extended for the 300 fps levers; queued GPU boot tests; integration review',
  phases: [
    { title: 'Code', detail: '4 components in parallel, separate files' },
    { title: 'Integrate', detail: 'cross-component review, CPU test suite, GPU boot-test sequence' },
  ],
}

const W = '/workspace/MuseTalk-perf300'
const MAIN = '/workspace/MuseTalk'
const C = `${W}/docs/startup_rework_20260928/STARTUP_CONTRACT.md`
const RES = `${W}/docs/startup_rework_20260928/research`
const OUT = `${W}/docs/startup_rework_20260928/impl`
const PY = '/workspace/.venvs/musetalk_trt_stagewise/bin/python'

const CTX = `
## Shared context — read carefully
You are implementing the MuseTalk start/install rework. The binding design is ${C} (written by another session; READ IT FULLY
FIRST). Its file:line research is in ${RES}/r0..r8.txt; the validated venv freeze is
${W}/docs/startup_rework_20260928/venv_freeze_cu121.txt; baseline results in ${W}/docs/fps_comparisons/startup_rework_20260928/.
IMPORTANT CHANGES TO THE CONTRACT (these override it):
1. Work ONLY in the git worktree ${W} (branch perf/300fps-4070s). The contract's rules about "never touch the worktree" and
   "do not modify trt_runtime.py/vae_fast_decoder.py/..." were written for a different session; they no longer apply in that form.
   BUT another workflow of this session is editing, right now, in ${W}: api_server.py, scripts/hls_gpu_scheduler.py,
   scripts/webrtc_*.py, scripts/api_avatar.py, scripts/avatar_cache.py, scripts/replay_scheduler_exactness.py,
   scripts/video_ab*.py, load_test_webrtc_v2.py, test_avatar_memory_layout.py, experiments/throughput300_candidate/,
   experiments/video_validation/. Do NOT edit those. Read them freely. Never edit anything under ${MAIN} (the user's clean main).
   No commits, branches, stashes or resets.
2. The startup must ACCOUNT FOR THE NEW 300 fps IMPLEMENTATIONS in this branch (read 'git -C ${W} log --oneline -3',
   'git -C ${W} show --stat 2254bc2', and the flag lists in scripts/webrtc_media_flags.py, scripts/vae_fast_decoder.py
   (TaesdTrtBackend, MUSETALK_TAESD_BACKEND=trt, MUSETALK_TAESD_TRT_*), scripts/trt_runtime.py (MUSETALK_TRT_UNET_CUDAGRAPHS,
   MUSETALK_UNET_BACKEND=trt_stagewise routing), scripts/unet_stagewise_trt.py + scripts/build_unet_stagewise.py (per-block
   stagewise FP16 engines under models/tensorrt_unet_stagewise_sm89/bs<N>/ with a manifest; bs16 engines were just built on this
   box), scripts/avatar_manager_parallel.py (MUSETALK_FREE_EAGER_UNET), scripts/box_guard.sh):
   - Engine store must manage THREE engine kinds keyed by GPU+TensorRT(+torch_tensorrt) version: (a) the torch_tensorrt bs8 .ts
     UNet (contract), (b) stagewise FP16 UNet engine sets (per batch size; build via build_unet_stagewise.py; adopt the existing
     ${W}/models/tensorrt_unet_stagewise_sm89/ set if its manifest matches), (c) the TAESD TRT engine (models/taesd/trt/; tiny,
     fast build). Each kind has its own validation record; an engine is usable only if validated for this key.
   - Recipes: 'fast' (DEFAULT; the contract's fast recipe: compiled TAESD + validated .ts UNet else eager), 'fast300' (fast +
     the 300 fps levers listed one-per-line in a tracked recipe file configs/recipes/fast300.env; the resolver enables each lever
     only if its prerequisites are met — stagewise engine validated for this key, TAESD TRT engine validated, native VP8 preflight
     ok, etc. — and records every drop with a reason; fast300 must NOT become the default: the user approves it after video
     review), and 'legacy_int8' (old chain, one-line rollback). Leave configs/recipes/fast300.env with the levers COMMENTED OUT
     plus a note; the orchestrator enables them as their gates pass.
   - Serving levers (WEBRTC_NONBLOCKING_HANDOFF, WEBRTC_IDLE_FRAME_CACHE, HLS_GPU_PIPELINE_DEPTH, HLS_SCHEDULER_POLICY,
     WEBRTC_H264_IMPL, WEBRTC_NATIVE_VP8_THREADS, MUSETALK_AVATAR_MASK_CHANNELS, MUSETALK_AVATAR_FRAME_STORE, WEBRTC_LIFETIME_COUNTERS,
     WEBRTC_GROUP_MAX_COUNT, MUSETALK_DISABLE_LOCAL_TTS, MUSETALK_THREAD_CAPS, HLS_GPU_EVENT_TIMING, ...) are pass-through knobs:
     the resolver must never clobber them; document them in configs/musetalk_overrides.env.example and list them in the
     resolver report ('levers' section with effective value + source layer).
   - Encoders: the installer provisions native VP8 (contract) AND the launcher/resolver understand WEBRTC_H264_IMPL
     (aiortc|x264tuned|nvenc). Decision rule for WEBRTC_VP8_ENCODER default: keep 'pyav' in recipe fast; in fast300 use 'native'
     only if (i) native VP8 preflight passes and (ii) H.264-only clients still get a working H.264 path (check the in-progress
     scripts/webrtc_h264_override.py and webrtc_native_vp8.py behavior; if you cannot confirm (ii) from code, keep pyav and say so).
   - Chin alignment stays offline (character_factory — do NOT edit it) but the installer gets an optional '--with-chin-tools' group
     that creates a DEDICATED small venv (mediapipe 0.10.9 + the numpy/opencv versions the chin tracker was validated with — read
     /workspace/SoulX-FlashHead/.venv/lib/python3.10/site-packages/*.dist-info names, do not modify that venv) at
     $WORKSPACE/.venvs/musetalk_chin_tools, and exposes its python as MUSETALK_CHIN_TRACKER_PYTHON in the resolved env, so future
     live chin work does not borrow another project's venv.
   - Blackwell: create the cu128 constraints set and verify it RESOLVES (pip --dry-run) in a scratch venv; mark untested on hardware.
   - S3 engine sharing: implement restore/publish, but publish ONLY when explicitly requested (flag default off). Never upload anything.
3. CPU-ONLY. Do not run anything on the GPU, do not import torch/tensorrt in the main tools (subprocess builds/validations are
   written, not executed), do not start/stop servers, do not pip install into ${PY}'s venv. Scratch venvs only under
   /tmp/claude-0/-workspace/866169db-0af2-4cde-9a98-79a398efeda7/scratchpad/ and delete them; keep >= 12 GB disk free
   (check df -h /workspace); keep each process < 2 GB RSS and MemAvailable >= 4 GB (a GPU benchmark is running).
4. GPU validation you cannot run: write it into ${OUT}/<component>/gpu_sequence.sh (named steps, each wrapped as
   ${W}/scripts/box_guard.sh run --min-avail-gb N --wait-min 60 --label startup_<step> -- <cmd>, clear PASS/FAIL lines, JSON
   outputs under ${OUT}/<component>/), with expected runtime and RAM per step.
5. Style: bash 'set -Eeuo pipefail' with log() helpers like existing scripts; Python stdlib-only where the contract says so.
   Never print secrets (/workspace/.musetalk-runtime.env, .env.webrtc-turn.local, .lingua-control-plane.env).
`

const SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    files_changed: { type: 'array', items: { type: 'string' } },
    interfaces: { type: 'array', items: { type: 'string' }, description: 'CLI commands / functions / file formats other components rely on' },
    cpu_tests: { type: 'array', items: { type: 'object', properties: { test: { type: 'string' }, result: { type: 'string', enum: ['pass', 'fail'] }, detail: { type: 'string' } }, required: ['test', 'result', 'detail'] } },
    gpu_sequence: { type: 'object', properties: { script: { type: 'string' }, steps: { type: 'array', items: { type: 'string' } }, est_minutes: { type: 'number' } }, required: ['script', 'steps', 'est_minutes'] },
    deviations_from_contract: { type: 'array', items: { type: 'string' } },
    issues: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'files_changed', 'interfaces', 'cpu_tests', 'gpu_sequence', 'deviations_from_contract', 'issues'],
}

const SHARED_IF = `
## Cross-component interfaces (fixed; implement/consume exactly)
- scripts/musetalk_engine_keys.py (stdlib-only, owned by the ENGINE STORE agent; the resolver imports it): engine_key(kind, facts),
  slug(), fingerprint read/write + usable(entry, facts) for kinds 'unet_ts', 'unet_stagewise', 'taesd_trt'; default store roots:
  models/tensorrt_unet (unet_ts), models/tensorrt_unet_stagewise (unet_stagewise; must also recognise the legacy
  models/tensorrt_unet_stagewise_sm89/ layout written by build_unet_stagewise.py), models/taesd/trt (taesd_trt).
- scripts/musetalk_host_profile.py (resolver, owned by the RESOLVER agent): 'detect', 'resolve --recipe fast|fast300|legacy_int8',
  'verify-log', 'engine-key --kind K', 'find-engine --kind K'. Resolved env: .runtime/musetalk_resolved.env + .runtime/musetalk_resolved.json.
- scripts/unet_engine_store.py (owned by the ENGINE STORE agent): 'list', 'build --kind K [--batch N]', 'validate', 'adopt', 'restore',
  'publish', 'ensure --kind K --provision auto|adopt|restore|build|off'.
- scripts/install_musetalk.sh + requirements/* + download_weights.sh + scripts/musetalk_selftest.py (owned by the INSTALLER agent).
- scripts/run_musetalk_server.sh, scripts/vast_server_ctl.sh, scripts/vast_onstart.sh, scripts/run_webrtc_relay_api_server.sh,
  scripts/run_turnserver_tcp_relay.sh, scripts/setup_musetalk.sh (shim), configs/musetalk_overrides.env.example,
  configs/recipes/fast300.env, scripts/test_startup_scripts.sh, docs/STARTUP.md (owned by the LAUNCH CHAIN agent). The launch-chain
  agent must NOT edit /workspace/run-musetalk-local-trt.sh (outside the repo; the user's box wrapper) — instead write the proposed
  new version to ${OUT}/launch/run-musetalk-local-trt.sh.proposed and describe the diff.
If you need something from another component, code against these interfaces and note it; do not edit their files.
`

const TASKS = [
  { key: 'resolver', prompt: `## Your component: A — scripts/musetalk_host_profile.py (contract Component A) + the recipe logic for fast / fast300 /
legacy_int8 described above, the 'levers' report, MUSETALK_CHIN_TRACKER_PYTHON emission, and unit tests test_musetalk_host_profile.py
(repo root, unittest, stdlib-only, injected facts via MUSETALK_HOST_FACTS_JSON, fake venv trees). Confirm exact log strings for
verify-log in the code (VAE backend active lines incl. the new TAESD TRT backend name; UNet backend active incl. stagewise).` },
  { key: 'engines', prompt: `## Your component: B — scripts/musetalk_engine_keys.py + scripts/unet_engine_store.py (contract Component B, extended to
the three engine kinds), the portable validation corpus calibration/unet_portable_bs8/ (16 files copied from
calibration/unet_multi_avatar_20260928 incl. holdout; manifest; .gitignore negation so only it is tracked), and
test_unet_engine_store.py (CPU-only, no torch). Adopt paths on this box: models/tensorrt_unet_sm89_bs8_local/unet_trt.ts (.ts, symlink target
tensorrt_unet_static_bs8_20260529) and models/tensorrt_unet_stagewise_sm89/bs16 (stagewise; read its manifest format from
build_unet_stagewise.py). Validation commands: validate_unet_backend.py (read its real CLI incl. the new --backend runtime /
--group-captures options) for UNet kinds; for taesd_trt read how TaesdTrtBackend verifies its probe hash. Do not run builds/validations.` },
  { key: 'installer', prompt: `## Your component: C — scripts/install_musetalk.sh, requirements/{server.in,kokoro.in,legacy-int8.in,chin-tools.in,
constraints-cu121.txt,constraints-cu128.txt,README.md}, download_weights.sh additions (TAESD pinned revision; Kokoro opt-in pre-cache),
scripts/musetalk_selftest.py (py_compile only; extend the contract's selftest to also time the TAESD TRT backend when its engine exists and
to record the engine keys found), --with-chin-tools dedicated venv, and '--check' semantics (this box's existing venv must pass --check
untouched). Verify both constraint sets RESOLVE with pip --dry-run in a scratch venv (python3.10) under the scratchpad and delete it after;
record results in ${OUT}/installer/resolve_*.json. Make sure every package the new 300 fps code imports is covered (grep imports in the
branch's changed/new files, e.g. numba if used by chin kernels, onnx/onnxruntime versions used by build_unet_stagewise.py, pycuda?).` },
  { key: 'launch', prompt: `## Your component: D — the launch chain (contract Component D) + configs/musetalk_overrides.env.example (document every
300 fps lever as commented one-liners with its default and effect, grouped: engines, serving, memory, encoders, telemetry),
configs/recipes/fast300.env (all levers commented out, with the gate each must pass before enabling), scripts/test_startup_scripts.sh
(contract test list + fast300 dispatch + pass-through of unknown knobs + legacy dispatch dry), docs/STARTUP.md (operator guide: install,
boot, recipes, overrides, engine provisioning, rollback, troubleshooting, the shared-box GPU lease), and the proposed
run-musetalk-local-trt.sh. Keep vast_server_ctl.sh's stop/drain/restart/logs/TURN behaviour unchanged; verify-log strict by default.` },
]

phase('Code')
const results = await parallel(TASKS.map(t => () =>
  agent(`${CTX}\n${SHARED_IF}\n${t.prompt}`, { label: `startup:${t.key}`, phase: 'Code', schema: SCHEMA }).then(r => r ? { key: t.key, ...r } : null)))
const done = results.filter(Boolean)

phase('Integrate')
const integ = await agent(`${CTX}\n${SHARED_IF}
## Your task: INTEGRATION REVIEW (CPU-only). Component reports:
${JSON.stringify(done, null, 1)}
1. Check the components actually fit: resolver <-> engine keys/store formats; launcher <-> resolver CLI/exit codes; onstart <-> installer
   --check exit codes and engine ensure; recipe files <-> resolver; overrides example <-> real flag names/defaults in code (grep each).
   Fix mismatches minimally in the owning files (you may edit any file the four components created or modified; still not the files owned
   by the other workflow nor anything in ${MAIN}).
2. Run every CPU test (test_startup_scripts.sh, test_musetalk_host_profile.py, test_unet_engine_store.py, bash -n / py_compile on every
   touched file). Run 'scripts/run_musetalk_server.sh --print-env' with this box's REAL host facts (nvidia-smi query is fine; no CUDA init) for
   recipes fast and fast300 and show the effective recipe it would boot (no server start). Run 'scripts/install_musetalk.sh --check' against
   the real venv (read-only) and confirm it passes untouched.
3. Consolidate the components' GPU steps into ONE runner ${OUT}/RUN_STARTUP_GPU.sh, in order: adopt+validate existing engines (.ts,
   stagewise bs16, TAESD TRT) → selftest → boot recipe fast on port 8310 via the new launcher (verify-log must confirm taesd + trt UNet) →
   short load test with ${W}/docs/fps_comparisons/startup_rework_20260928/run_load_test.sh at 4/8/12 streams (compare with the old-recipe
   baseline 70/57/38 fps) → stop → boot eager fallback (MUSETALK_UNET_MODE=eager) smoke → stop → legacy_int8 rollback boot smoke → stop →
   fast300 boot smoke with whichever levers are enabled. Each step box_guard-wrapped, resumable, with PASS/FAIL JSON. Never bind :8000.
Report in the schema (gpu_sequence = the consolidated runner); list any contract deviations and anything that must be decided by the user.`,
  { label: 'startup:integrate', phase: 'Integrate', schema: SCHEMA })

return { components: done, integration: integ }
