export const meta = {
  name: 'musetalk-startup-understand',
  description: 'Read-only map of MuseTalk docs, start/install scripts, env knobs and perf recipe to plan a startup rework',
  phases: [
    { title: 'Read', detail: '9 parallel read-only readers over docs, scripts, code knobs, in-flight work' },
  ],
}

const COMMON = `
You are helping rework the MuseTalk start-server / install scripts in /workspace/MuseTalk so a fresh machine boots straight into the
high-throughput configuration (TAESD decoder + 100% chin alignment + TRT UNet etc., which reached ~160-240 fps offline on an RTX 4070 SUPER),
on ANY NVIDIA GPU (not just this sm89 box). Right now the start scripts (scripts/vast_onstart.sh -> scripts/vast_server_ctl.sh ->
scripts/run_webrtc_relay_api_server.sh, wrapped by /workspace/run-musetalk-local-trt.sh) still boot the OLD recipe
(MUSETALK_VAE_BACKEND=trt_stagewise int8 SD-VAE, UNet TRT bs8 via .runtime/musetalk_trt_local_sm89.env).

HARD RULES (shared host, live user server):
- STRICTLY READ-ONLY. Do not edit, create, move or delete any file in the repo. Do not git commit/stash/checkout.
- Do NOT run anything on the GPU, do not import torch/tensorrt, do not start/stop servers, do not curl port 8000 except GET /health if truly needed.
- Never print secrets. /workspace/.musetalk-runtime.env, MuseTalk/.env.webrtc-turn.local, .runtime/musetalk_trt_best.env may contain credentials: read key NAMES only (e.g. sed 's/=.*/=<redacted>/').
- Another session has uncommitted edits in scripts/trt_runtime.py, scripts/vae_fast_decoder.py, musetalk/models/vae.py, scripts/avatar_manager_parallel.py,
  scripts/validate_unet_backend.py and untracked scripts (bench_gpu_path.py, box_guard.sh, build_unet_stagewise.py, unet_stagewise_trt.py,
  build_unet_multi_avatar_corpus.py). Read them if relevant, never modify them.
- Use git show / git log / git diff freely for history (read-only).
Be concrete: exact env var names, exact default values, file:line references, exact commands. Prefer facts verified in code over claims in docs;
when a doc and the code disagree, say so. Report what is PROVEN (measured) vs PROJECTED.
`

const FINDINGS = {
  type: 'object',
  properties: {
    summary: { type: 'string', description: 'Dense 10-25 line summary of what matters for the startup/install rework' },
    recommended_env: {
      type: 'array',
      description: 'Env vars / flags that the new default startup should set (or must NOT set), with the value and evidence',
      items: { type: 'object', properties: {
        name: { type: 'string' }, value: { type: 'string' }, scope: { type: 'string', description: 'all-gpus | sm89-only | per-gpu-tier | optional | avoid' },
        evidence: { type: 'string', description: 'file:line or doc section + measured numbers' },
      }, required: ['name', 'value', 'evidence'] },
    },
    procedures: { type: 'array', items: { type: 'string' }, description: 'Install/build/boot procedures and commands found (exact)' },
    machine_specific: { type: 'array', items: { type: 'string' }, description: 'Anything hardcoded to this machine/GPU/sm89/paths/ports that breaks portability' },
    problems: { type: 'array', items: { type: 'string' }, description: 'Bugs, stale config, doc/code discrepancies, failure points in current scripts' },
    open_questions: { type: 'array', items: { type: 'string' } },
    files_read: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'recommended_env', 'procedures', 'machine_specific', 'problems', 'open_questions', 'files_read'],
}

const READERS = [
  { key: 'perf-plan-300fps', prompt: `Read fully: docs/musetalk_4070s_300fps_plan_2026-09-27.md (large, read all of it in chunks) and
docs/fps_comparisons/4070s_300fps_20260927/EVIDENCE_DIGEST.md, plus skim docs/fps_comparisons/4070s_300fps_20260927/ file list.
Extract: the exact serving recipe (TAESD decoder, native encoder, 100% chin, refined seam), which env vars/flags turn each piece on, measured
per-stage timings, live vs offline fps (the '160 fps' offline harness vs live capacity), the serving fixes the plan says are needed for live
throughput (scheduler batch sizes, workers, compose/encode threads, CPU pinning, /dev/shm, disk), Phase 0 items and decisions D1/D2, and
which of those belong in startup scripts. Also list the things marked CLOSED/rejected so startup does not enable them.` },
  { key: 'taesd-chin-history', prompt: `Establish EXACTLY how the TAESD decoder and the '100% chin alignment' + refined seam are enabled today.
Read: git show c284891 (commit 'MASSIVE OPTIMIZATIONS TAESD + chin alignmetn 100%'), git show 9b90b92 --stat and its code diffs,
git log -p for scripts/vae_fast_decoder.py and musetalk/models/vae.py (committed versions), docs/musetalk_4070s_pipeline_component_analysis_2026-09-22.md,
docs/AVATAR_WORK_CONTEXT_2026-09-26.md (all of it), docs/fps_comparisons/4070s_20260922/*.py and *.json (taesd.json, e2e_results.json, capacity.json).
Find the env vars (e.g. MUSETALK_VAE_BACKEND=taesd, MUSETALK_TAESD_*), whether chin alignment is an env var, an avatar-prep parameter
(e.g. parsing_mode='jaw' in api_server.py ~1041, bbox/extra_margin), a per-request API field, or baked into avatar packages; what 'refined seam' is
and how it is enabled; whether avatars must be re-prepared; model files needed (models/taesd/*, HF repo id) and how they get downloaded.
Give file:line for every switch.` },
  { key: 'start-scripts', prompt: `Map the CURRENT start/install scripts end to end. Read fully: scripts/vast_onstart.sh, scripts/vast_server_ctl.sh,
scripts/run_webrtc_relay_api_server.sh, scripts/run_api_server.sh, scripts/run_trt_stagewise_server.sh, scripts/setup_trt_stagewise_server_env.sh,
scripts/setup_trt_experiment_env.sh, download_weights.sh, setup_musetalk.sh, setup_musetalk_cat.sh, entrypoint.sh, inference.sh,
/workspace/run-musetalk-local-trt.sh, /workspace/onstart.sh, /workspace/vast-flashhead-ditto.sh (skim), scripts/select_unet_trt_profile.py,
scripts/trt_artifact_bundle.py, scripts/install_native_vp8.py, scripts/install_webrtc_deps.sh, scripts/bootstrap_aws_secrets.py,
scripts/run_turnserver.sh, scripts/run_turnserver_tcp_relay.sh, requirements.txt, .musetalk_trt_artifact_manifest.json (structure only),
.runtime/musetalk_trt_local_sm89.env. Also tail -300 /workspace/onstart.log and look at /workspace/bootstrap.log head/tail for real boot behaviour.
Produce: call graph of scripts, every PROFILE (e.g. throughput_record) and the env each sets, venv creation + pinned package versions
(torch 2.5.1 cu121, torch_tensorrt, tensorrt, mmcv etc.), weight downloads, TRT artifact restore/build/select flow, health checks, idempotency,
timings from logs, and everything hardcoded to this machine. Note what the scripts would do on a GPU that is not sm89.` },
  { key: 'start-docs', prompt: `Read the startup/ops documentation fully: start_params.md, current_start_param_reference.md, startup_script_improvement.md,
docs/vast_ai_boot.md, docs/vast_startup_optimization_plan.md, current_tensorrt_environment_plan.md, docs/trt_artifacts/README.md,
docs/trt_artifacts/split8_int8_artifact_run_2026-07-10.md, docs/gpu_vram_budgeting.md, vast.ai-new-boot-scripts-12.1.1-context/README.md,
README.md, newcode_readme.md, docs/musetalk_worker_secrets.md, turn-setup.md, TURN_API_README.md, docs/WEBRTC_NATIVE_VP8.md, docs/avatar_s3_persistence.md,
docs/musetalk_avatar_cache_warmup_handoff.md, docs/musetalk_autoscaling_plan.md, webRTC-migration.md (skim).
Extract the intended boot contract (inputs: env vars/secrets from vast.ai template, ports, TURN, S3 artifacts, control-plane registration),
documented profiles and their knobs, improvement ideas already written down but not implemented, and where docs contradict the current scripts.
Cross-check a sample of claims against scripts/vast_onstart.sh and scripts/vast_server_ctl.sh.` },
  { key: 'env-knob-inventory', prompt: `Build the authoritative inventory of runtime env knobs READ BY THE SERVING CODE. grep os.getenv / os.environ.get /
os.environ[...] / _env_flag / _env_int / _env_float helpers in api_server.py, scripts/api_avatar.py, scripts/hls_gpu_scheduler.py, scripts/webrtc_tracks.py,
scripts/webrtc_manager.py, scripts/webrtc_motion_playback.py, scripts/webrtc_native_vp8.py, scripts/webrtc_pose_router.py, scripts/avatar_manager_parallel.py,
scripts/trt_runtime.py, scripts/vae_fast_decoder.py, scripts/inference.py, scripts/runtime_cpu_tuning.py, scripts/concurrent_gpu_manager.py, scripts/avatar_cache.py,
scripts/avatar_s3_store.py, scripts/worker_control_plane.py, musetalk/**/*.py.
Focus on perf-relevant ones: decoder backend (taesd, trt_stagewise, pytorch), TAESD options, UNet backend/TRT paths/batch sizes, scheduler batch/slice,
compose/encode/prep workers, WebRTC encoder (native VP8, libx264), fps, prebuffer, blend (fixed point, shrink bbox, side jaw, source mouth), chin/jaw,
CPU tuning (threads, affinity), whisper batch, warm runtime, avatar cache. For each: name, default in code (file:line), what it does, and the value the
LIVE server uses. The live server's non-secret env is available via: tr '\\0' '\\n' < /proc/3555159/environ | grep -E '^(MUSETALK|WEBRTC|HLS|TRT|AVATAR)'
(skip anything with KEY/SECRET/TOKEN/PASS/CRED in the name; if pid is gone, skip). Flag knobs whose code default differs from the fast recipe.` },
  { key: 'cross-gpu-history', prompt: `Read the throughput history across GPUs to derive how the startup should AUTO-SIZE on any machine.
Read: current_cross_server_throughput_findings.md, current_model_backend_findings.md, current_model_backend_acceleration_plan.md,
current_model_backend_execution_plan.md, current_unet_trt_throughput_findings_2026-05-29.md, docs/webrtc_generation_optimization_results_2026-07-03.md,
docs/webrtc_load_test_findings_2026-06-07.md, load_test_webrtc_rtx6000ada_int8_5stage_20fps_20260608.md, load_test_webrtc_rtx6000ada_int8_trt_unet_split8_20fps_20260608.md,
docs/v100_webrtc_load_test_2026-05-22.md, docs/vast_value_analysis_webrtc_v100_vs_3090_2026-05-23.md, docs/next_bottleneck_vae_late_block_plan_2026-06-11.md,
docs/musetalk_quantization_optimization_plan.md (skim), CPU_OPTIMIZATION_ANALYSIS.md, docs/gpu_vram_budgeting.md, current_webrtc_playback_smoothing_findings.md (skim).
Extract per-GPU (V100 sm70, 3090 sm86, 4090 sm89, RTX 6000 Ada sm89, 4070S sm89, H100/A100 if any) the best batch sizes, worker counts, VRAM use,
stream capacity, which backends work/fail per arch (e.g. INT8 TRT, FP8, torch.compile), CPU-bound limits (compose/encode per core), and rules of thumb
(VRAM -> batch, cores -> workers). Also note what is now superseded by TAESD (the SD-VAE int8 TRT stagewise decoder is no longer the bottleneck).` },
  { key: 'inflight-work', prompt: `Understand the OTHER session's in-flight (uncommitted) 300 fps implementation so the startup rework integrates with it and does not
conflict. Read: git diff (all tracked modified files: scripts/trt_runtime.py, scripts/vae_fast_decoder.py, musetalk/models/vae.py, scripts/avatar_manager_parallel.py,
scripts/validate_unet_backend.py, character_factory/h3_avatar_workflow/BATCH_THREE_POSE.md), and the untracked files scripts/bench_gpu_path.py, scripts/box_guard.sh,
scripts/test_box_guard.sh, scripts/build_unet_stagewise.py, scripts/unet_stagewise_trt.py, scripts/build_unet_multi_avatar_corpus.py, and everything under
docs/fps_comparisons/4070s_300fps_impl_20260928/ (logs, json results, py). Also cat /workspace/.gpu_lease.log and /workspace/.gpu_lease.holder.
Report: new env knobs being added (names, defaults, what they enable: TAESD TRT engine? UNet stagewise FP16 chained? CUDA graphs?), which are validated
(numbers from the logs/json) vs still experimental, what artifacts they build and where (paths, sizes, build time, per-GPU-arch), the bench harness
(bench_gpu_path.py: what it measures, its CLI, its default env - it sets MUSETALK_VAE_BACKEND=taesd around line 50), the box_guard.sh GPU-lease/RAM-guard
contract (CLI, lease file, min-avail), and what a startup script should do NOW vs leave behind flags until that work lands.` },
  { key: 'webrtc-runtime-reqs', prompt: `Find everything the RUNTIME needs provisioned at startup for WebRTC/multipose/character packages to work at high throughput.
Read: docs/WEBRTC_NATIVE_VP8.md, docs/WEBRTC_WALL_AUDIO.md, docs/WEBRTC_EXACT_SILENCE_AND_NOSE_MATCHING.md, docs/WEBRTC_CURRENT_PHONEME_BLEND.md,
docs/WEBRTC_MOTION_EYE_BLEND.md, docs/WEBRTC_MULTIPOSE_AUDIT_2026-09-25.md, docs/WEBRTC_MULTIPOSE_CLIENT_ANCHOR_2026-09-25.md, docs/MULTIPOSE_LTX_MUSETALK_IMPLEMENTATION.md,
docs/GROK_MULTIPOSE_HANDOFF.md, docs/KOKORO_SPANISH_MULTIPOSE_TEST_2026-09-25.md, character_factory/REALTIME_PACKAGE_INTEGRITY.md, character_factory/README.md,
character_factory/CHARACTER_CREATION_WORKFLOW.md (skim), POSE_WEBRTC_LAB.md, WEBRTC_REACT_NATIVE_README.md (skim), api_calls.md.
Plus code: scripts/install_native_vp8.py, scripts/native_vp8_manifest.json, scripts/webrtc_native_vp8.py (how native VP8 is detected/loaded, .runtime/native_vp8),
scripts/kokoro_tts.py (deps), and api_server.py startup hooks (grep for 'startup', 'lifespan', 'warm', 'preload', 'on_event').
Report: packages/binaries/system deps (ffmpeg, coturn/turnserver, libvpx, aiortc 1.6.0 wheel vs 1.11.0 native vp8, kokoro/espeak, mediapipe/facemesh),
env flags each feature needs, avatar/character package locations and warmup, and the env the fast path expects for WebRTC (encoder choice, fps, batch callback).` },
  { key: 'portability', prompt: `Design input for 'works on ANY machine'. Investigate how GPU-architecture-specific artifacts are handled and what breaks off sm89.
Read: scripts/trt_runtime.py (committed version via git show HEAD:scripts/trt_runtime.py and the working copy), scripts/tensorrt_export.py, scripts/select_unet_trt_profile.py,
scripts/trt_artifact_bundle.py, .musetalk_trt_artifact_manifest.json, docs/trt_artifacts/*, scripts/validate_unet_backend.py, scripts/validate_vae_backend.py,
scripts/vae_fast_decoder.py (how TAESD loads: local dir vs HF download, torch.compile mode, TRT option, fallbacks), musetalk/models/vae.py,
scripts/runtime_cpu_tuning.py, scripts/setup_trt_experiment_env.sh (the TRT/torch_tensorrt install matrix), models/ directory layout (ls -la, sizes, symlinks,
models/tensorrt_unet_sm89_bs8_local/unet_trt_meta.json content), and /workspace/.venvs (which venvs exist, python versions,
/workspace/.venvs/musetalk_trt_stagewise/bin/pip list 2>/dev/null | grep -iE 'torch|tensorrt|onnx|diffusers|aiortc|av|opencv|numpy|mmcv|mediapipe|kokoro').
Also check nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv (read-only query is fine), nproc, free -g, df -h /.
Report: exact compatibility matrix (driver/CUDA/torch/torch_tensorrt/TRT versions), what must be rebuilt per GPU arch and how long it takes, what can be
shipped portable (TAESD pytorch/compile needs no TRT), a safe fallback ladder per GPU tier (e.g. TRT UNet if buildable else torch fp16; TAESD always),
VRAM/CPU/RAM/disk preflight requirements, and how UNet TRT engines are currently built/validated (commands, time, disk: the UNet TRT artifact is ~2.2 GB).` },
]

phase('Read')
const results = await parallel(READERS.map(r => () =>
  agent(COMMON + '\nYOUR SLICE (' + r.key + '):\n' + r.prompt, { label: 'read:' + r.key, phase: 'Read', schema: FINDINGS })
    .then(res => res ? { key: r.key, ...res } : null)
))
const ok = results.filter(Boolean)
log(`${ok.length}/${READERS.length} readers returned`)
return ok
