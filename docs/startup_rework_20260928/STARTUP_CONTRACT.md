# MuseTalk startup/install rework — implementation contract (2026-09-28)

Goal: a fresh machine with ANY NVIDIA GPU boots MuseTalk straight into the fast recipe
(TAESD decoder + best available UNet backend), with no hand-kept env files, no silent slow
fallback, and one-line rollback. Everything below is binding for all implementers; if you
must deviate, say so explicitly in your final report.

## Facts established (do not re-derive)
- Fast recipe = `MUSETALK_VAE_BACKEND=taesd` (compiled TAESD, warmup per batch bucket) +
  UNet `trt` (torch_tensorrt static bs8 `.ts` built for THIS GPU arch) or `eager` PyTorch FP16.
  Measured on RTX 4070 SUPER: GPU path TRT-UNet+TAESD 260 fps (200 s); eager UNet ~37 ms/bs8
  vs TRT 24 ms/bs8. Old recipe (int8 SD-VAE `trt_stagewise`) is ~4x slower on the decoder.
- Old chain bug: `run_trt_stagewise_server.sh` `set -a; source`s a generated profile env AFTER the
  caller's exports, so `MUSETALK_VAE_BACKEND=taesd` from the caller is overwritten.
  `select_unet_trt_profile.py` always writes the old int8 recipe.
- `MUSETALK_TRT_FALLBACK` (code default 1) couples UNet-TRT, TRT-VAE and TAESD fallbacks; with 1 a
  TAESD failure silently serves PyTorch SD-VAE (~47 fps). Fast recipe sets 0 and the resolver only
  requests TRT UNet when a validated, arch-matched engine exists.
- `trt_runtime._trt_unet_requested()`: `MUSETALK_UNET_BACKEND in {trt,tensorrt}` or
  `MUSETALK_TRT_UNET_ENABLED=1`. Eager = `MUSETALK_UNET_BACKEND=eager` + `MUSETALK_TRT_UNET_ENABLED=0`.
  `MUSETALK_TRT_ENABLED=1` with an empty VAE backend activates a legacy single-engine VAE; fast
  recipe sets `MUSETALK_TRT_ENABLED=0`. `MUSETALK_UNET_BACKEND=trt_stagewise` belongs to another
  session's in-flight work: pass it through untouched, never set it.
- `MultiTrtUnetBackend` only serves batches that are multiples of its engine batch (8). Compiled
  TAESD is `torch.compile(dynamic=False)` per batch, warmed only for `MUSETALK_TAESD_WARMUP_BATCHES`.
  `api_server.py` snaps WebRTC batch sizes to `MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES` then
  `HLS_SCHEDULER_FIXED_BATCH_SIZES`. Therefore ALL of these must be equal: HLS_SCHEDULER_FIXED_BATCH_SIZES,
  MUSETALK_TAESD_WARMUP_BATCHES, MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES (default "8"), and
  HLS_SCHEDULER_MAX_BATCH = max(buckets), HLS_SCHEDULER_STARTUP_SLICE_SIZE = min(8, max).
- TRT engines are arch-bound (3090 sm86 engine fails on sm89: "No compatible device was found").
  A torch_tensorrt `.ts` embeds a device string matching regex
  `rb"(\d+)%(\d+)%(\d+)%(\d)%(NVIDIA[^%\x00]{1,80})"` = device%major%minor%type%name
  (e.g. `0%8%9%0%NVIDIA GeForce RTX 4070 SUPER`).
- Loading the 2.2 GB UNet `.ts` raises host RSS by ~8.4 GB transiently (VmHWM 9.5 GB).
  A bs8 UNet TRT build took 7m06s and ~9.5-11 GB host RAM on the 4070S; +2.2 GB disk.
- The pinned stack (torch 2.5.1+cu121, torch_tensorrt 2.5.0, TRT 10.3, triton 3.1, CPython 3.10)
  has SASS for sm50..sm90 only: Blackwell (cc >= 10.0) needs the cu128 matrix
  (torch 2.7.1+cu128, torchvision 0.22.1+cu128, torchaudio 2.7.1+cu128, torch_tensorrt 2.7.0 which
  pins tensorrt 10.9.x; all have cp310 manylinux x86_64 wheels — verified on the indexes).
- TAESD weights: HF `madebyollin/taesd` revision `614f76814bbe30edbe2e627ace1c2234c81a2c0e`
  (config.json + diffusion_pytorch_model.safetensors, fp32). Verified: bit-identical to the
  vendored fp16 `models/taesd` after the fp32->fp16 cast the loader does (`torch_dtype=fp16`).
  Loader prefers `<repo>/models/taesd/config.json`. `models/` is gitignored; nothing downloads it today.
- Native VP8 (`WEBRTC_VP8_ENCODER=native`) needs x86_64 + CPython 3.10 + exactly aiortc 1.14.0,
  av 16.1.0, cffi 2.1.1, and `scripts/install_native_vp8.py` (installs into `.runtime/native_vp8`).
  It rejects H264-only offers, so the DEFAULT stays `pyav`; installer provisions native so it is a
  one-line switch. CPU preflight (0.2 s):
  `WEBRTC_VP8_ENCODER=native WEBRTC_NATIVE_VP8_DIR=<dir> CUDA_VISIBLE_DEVICES= <venvpy> -B -c "from scripts import webrtc_native_vp8 as n; n.configure_vp8_encoder('preflight')"`.
- Kokoro TTS (kokoro==0.9.4, misaki[en]==0.9.4, espeakng-loader, spacy 3.8.16, en_core_web_sm 3.8.0,
  model hexgrad/Kokoro-82M) is only hand-installed on this box. `requirements.txt` must NOT be installed
  (pulls tensorflow/gradio).
- The validated venv's full `pip freeze` is in `/tmp/claude-0/-workspace/acfff43b-67b7-4b45-b762-bfdcdc32ca14/scratchpad/venv_freeze_cu121.txt`
  (211 lines; mmcv and en_core_web_sm are URL/file entries).
- "100% chin alignment" is an OFFLINE avatar-creation stage (character_factory/h3_avatar_workflow, mediapipe
  from another project's venv). It is NOT part of this rework; do not touch character_factory/.

## Hard rules for every implementer
- Shared host with a live user server on :8000 and another agent session doing GPU work.
  Do NOT run anything on the GPU, do NOT import torch/tensorrt, do NOT start/stop any server,
  do NOT pip install into `/workspace/.venvs/musetalk_trt_stagewise` (shared, live). Scratch venvs
  only under the scratchpad dir, and delete them when done. Check `df -h /workspace` first; keep >= 12 GB free.
- The shared tree /workspace/MuseTalk is on branch `main` (HEAD 1564568, clean except untracked startup-rework
  files and a user-owned character_factory edit). Another session's 300 fps work lives on branch
  perf/300fps-4070s in the worktree /workspace/MuseTalk-perf300: NEVER touch that worktree. Its in-flight
  knobs (MUSETALK_TAESD_BACKEND=trt, MUSETALK_UNET_BACKEND=trt_stagewise, MUSETALK_TRT_UNET_CUDAGRAPHS,
  MUSETALK_FREE_EAGER_UNET, WEBRTC_NONBLOCKING_HANDOFF, WEBRTC_IDLE_FRAME_CACHE, WEBRTC_H264_IMPL, ...) do not
  exist on main; the launcher must simply pass unknown env through. box_guard now lives at
  /workspace/.tools/bin/box_guard.sh (outside the repo); do not depend on it from repo scripts.
- To avoid merge conflicts with that branch, do NOT modify these files even though they are clean on main:
  scripts/trt_runtime.py, scripts/vae_fast_decoder.py, musetalk/models/vae.py, scripts/avatar_manager_parallel.py,
  scripts/validate_unet_backend.py, scripts/api_avatar.py, scripts/avatar_cache.py, scripts/webrtc_*.py,
  scripts/bench_gpu_path.py, scripts/box_guard.sh, scripts/test_box_guard.sh, scripts/unet_stagewise_trt.py,
  scripts/build_unet_stagewise.py, scripts/build_unet_multi_avatar_corpus.py, scripts/replay_scheduler_exactness.py,
  api_server.py, anything under character_factory/, docs/fps_comparisons/4070s_300fps_impl_20260928/.
  Read them freely. No git commits, no branch changes.
- Never print secrets (/workspace/.musetalk-runtime.env, .env.webrtc-turn.local, .lingua-control-plane.env).
- Match the surrounding style: bash with `set -euo pipefail` (+ `-E` where ERR traps matter), `log()`
  helpers like the existing scripts; Python 3.8+ stdlib-only where specified.

## Layering (the core fix)
Effective value of any knob, highest wins:
1. the caller's environment (explicit export, Vast template env, TURN env file sourced by ctl),
2. operator overrides files: `MUSETALK_ENV_OVERRIDES_FILE` (colon-separated list; default
   `<repo>/.runtime/musetalk_overrides.env`; missing files are skipped) — one lever per line,
3. `<repo>/.runtime/musetalk_resolved.env`, regenerated by the resolver on EVERY launch,
4. code defaults.
Loading rule for 2 and 3: parse `KEY=VALUE` / `export KEY=VALUE` lines, ignore blanks/comments,
export only if the variable is currently UNSET (`[[ -z "${!key+x}" ]]`). Load 2 before 3.
Launchers must not export recipe knobs with `:=` defaults before this loading happens.

## Component A — `scripts/musetalk_host_profile.py` (NEW, stdlib-only, python3 >= 3.8, never imports torch)
CLI:
- `detect [--repo-root R] [--venv V]` → JSON host facts to stdout:
  gpus[] (index,name,compute_capability "8.9",memory_total_mib,memory_used_mib,power_limit_w,
  power_default_limit_w,driver_version) from `nvidia-smi --query-gpu=... --format=csv,noheader,nounits`
  (binary overridable by `MUSETALK_NVIDIA_SMI`; tests can inject facts via `MUSETALK_HOST_FACTS_JSON`=path),
  selected GPU honours CUDA_VISIBLE_DEVICES (first visible index);
  cpu {nproc, affinity, cgroup_quota_cpus, effective}; ram {mem_total_mb, mem_available_mb,
  cgroup_limit_mb, effective_total_mb, effective_available_mb} (MemAvailable = min(/proc/meminfo,
  cgroup memory.max - memory.current + active_file + inactive_file + slab_reclaimable), like box_guard);
  disk_free_gb for repo root; machine (x86_64/aarch64); venv {python_version, torch, torch_cuda_tag
  ("cu121"), tensorrt, torch_tensorrt, triton, aiortc, av, cffi} read from `site-packages/*.dist-info`
  directory names/METADATA (no imports).
- `resolve --repo-root R --venv V --out ENVFILE --report JSONFILE [--recipe fast]` → writes the
  resolved env file (header comment + `KEY='value'` lines, single-quote-escaped) and a JSON report
  {facts, decisions:[{knob,value,reason}], warnings[], errors[], emitted{}}. Exit 0 on success,
  2 on a hard error (no GPU unless MUSETALK_ALLOW_NO_GPU=1; cc>=10.0 with a cu121 venv; MUSETALK_UNET_MODE=trt
  without a usable engine; invalid buckets for a TRT engine). Prints a short human summary to stderr.
  The resolver reads os.environ: for every knob it emits, if the caller already set it, it emits the
  caller's value (so the report shows the truth) and computes dependents from it (e.g. caller buckets
  "16" → TAESD/stagewise warmup "16", max batch 16).
- `verify-log --log L [--offset BYTES] --expect-vae taesd [--expect-unet trt|eager|any] [--timeout S]`
  → scans the server log from byte offset for `VAE decode backend active: <name>` and
  `UNet backend active: <name>` (confirm exact strings in scripts/avatar_manager_parallel.py; TRT UNet logs
  `tensorrt_unet_multi`/`tensorrt_unet`; find the eager wording). Exit 0 match, 1 mismatch (print
  what was found), 3 not found within timeout.
- `engine-key` → prints the engine key for this GPU+venv (see Engine store). `find-engine` → prints
  JSON of the best engine or exits 3.
Resolver decisions (fast recipe):
- MUSETALK_RECIPE=fast, MUSETALK_VAE_BACKEND=taesd, MUSETALK_TAESD_COMPILE=1 unless
  `<repo>/.runtime/gpu_selftest.json` (Component C) matches this GPU name+cc and venv torch version and says
  taesd.compile_ok=false → 0 (warning). MUSETALK_TRT_FALLBACK=0, MUSETALK_TRT_ENABLED=0, MUSETALK_COMPILE=0,
  MUSETALK_WARM_RUNTIME=1, MUSETALK_UNET_CALIBRATION_CAPTURE=0, MUSETALK_VAE_CALIBRATION_CAPTURE=0.
- UNet: `MUSETALK_UNET_MODE` = auto (default) | trt | eager. auto → trt iff an engine is found
  (exact key, else same cc + same TRT + same torch_tensorrt version with a warning), GPU VRAM >=
  MUSETALK_TRT_UNET_MIN_VRAM_GB (default 8), and effective MemAvailable >= MUSETALK_TRT_UNET_MIN_MEM_AVAILABLE_GB
  (default 10); otherwise eager with the reason recorded. trt → same checks but a missing engine is a hard error,
  low RAM only a warning. Emit MUSETALK_UNET_BACKEND=trt, MUSETALK_TRT_UNET_ENABLED=1,
  MUSETALK_TRT_UNET_PATHS=8:<absolute path to unet_trt.ts> for trt; MUSETALK_UNET_BACKEND=eager,
  MUSETALK_TRT_UNET_ENABLED=0 for eager. If the caller set MUSETALK_UNET_BACKEND to something else
  (e.g. trt_stagewise) respect it and do not emit UNet TRT paths.
- Buckets: HLS_SCHEDULER_FIXED_BATCH_SIZES default "8"; with TRT UNet every bucket must be a multiple
  of 8 (else error in trt mode / fall back to eager in auto mode). Emit the coupled set listed above.
- CPU: effective = min(nproc, affinity, ceil(cgroup quota)). HLS_PREP_WORKERS=HLS_COMPOSE_WORKERS=
  HLS_ENCODE_WORKERS = 8 if effective >= 12 else max(2, effective // 2). MUSETALK_AVATAR_LOAD_WORKERS
  = min(8, effective). HLS_MAX_PENDING_JOBS=24. MUSETALK_WHISPER_SEGMENT_BATCH_SIZE=4.
- RAM: AVATAR_CACHE_MAX_MEMORY_MB = clamp(int(0.2 * effective_total_mb), 2048, 16384)
  (host RAM, not VRAM); AVATAR_CACHE_TTL_SECONDS=3600. GPU_TOTAL_MEMORY_GB = VRAM total (avoids the
  silent 24 GB fallback in concurrent_gpu_manager).
- Blend/WebRTC (same values the old launcher applied): MUSETALK_BLEND_FIXED_POINT=1,
  MUSETALK_BLEND_SHRINK_MASK_BBOX=1, WEBRTC_BATCH_FRAME_CALLBACK=1, WEBRTC_SYNC_MODE=strict_fifo,
  WEBRTC_AUDIO_SYNC_STRATEGY=timestamp_locked, WEBRTC_VIDEO_PREBUFFER_SECONDS=2.0, WEBRTC_ADAPTIVE_FPS=0,
  WEBRTC_TRIM_EDGE_SILENCE=0, WEBRTC_POSE_CROSSFADE_FRAMES=2, WEBRTC_POSE_FORCED_CROSSFADE_FRAMES=4,
  WEBRTC_POSE_MAX_SEMANTIC_DRIFT_SECONDS=0.75, WEBRTC_VP8_ENCODER=pyav (default; see native VP8 above),
  WEBRTC_NATIVE_VP8_DIR=<repo>/.runtime/native_vp8. HLS chunk encoder knobs as the old launcher:
  HLS_CHUNK_VIDEO_ENCODER=libx264, HLS_CHUNK_ENCODER_PRESET=ultrafast, HLS_CHUNK_ENCODER_CRF=28,
  HLS_CHUNK_PREPARE_AUDIO_SIDECAR=1. PROFILE=throughput_record, PYTHONFAULTHANDLER=1, PYTHONUNBUFFERED=1.
- Warnings (non-fatal): power.limit < 0.85 * power.default_limit ("GPU power-capped: expect lower fps");
  same-cc-different-GPU engine; VRAM < 8 GB; MemAvailable below the TRT threshold; RAM < 16 GB total.
- Recipe `legacy_int8`: resolver only emits MUSETALK_RECIPE=legacy_int8 (the launcher execs the old chain).
Report must include an estimate line: expected GPU-path fps if gpu_selftest.json has timings.

## Engine store (shared by A and B)
Layout: `<store>/<engine_key>/bs8/{unet_trt.ts, unet_trt_meta.json, fingerprint.json, validation.json}`,
default store `<repo>/models/tensorrt_unet` (override `MUSETALK_UNET_ENGINE_STORE`).
engine_key = `sm<major><minor>-<gpu slug>-trt<tensorrt version>-tt<torch_tensorrt version>`
(slug = lowercase gpu name, non-alnum → '-', squeezed), e.g.
`sm89-nvidia-geforce-rtx-4070-super-trt10.3.0-tt2.5.0`.
fingerprint.json = {"schema":"musetalk_unet_engine_v1","engine_key","gpu_name","compute_capability",
"tensorrt_version","torch_tensorrt_version","torch_version","batch":8,"engine_file":"unet_trt.ts",
"engine_bytes","engine_sha256" (optional, may be null for adopted engines),"embedded_device" (string found in
the .ts or null),"source":"built|adopted|restored","created_utc","validation":{"passed":bool,"mae_max",
"max_abs_max","capture_dir","files"}}.
An engine is USABLE only if fingerprint.validation.passed is true and unet_trt.ts exists (symlinks allowed).
Entries under construction live in `<store>/.<key>.partial-*` and are renamed atomically when complete.

## Component B — `scripts/unet_engine_store.py` (NEW; runs under the venv python; may import torch ONLY in
subprocesses it launches for build/validate; the CLI itself must work without CUDA for list/key/remote ops)
CLI (all take `--repo-root`, `--store`):
- `list` → JSON of entries + usability for this GPU.
- `build [--force] [--timeout-min 30]` → preflight (GPU present; VRAM >= 8 GB; MemAvailable >= 14 GB
  (`MUSETALK_UNET_BUILD_MIN_MEM_AVAILABLE_GB`); disk free >= 5 GB), then run
  `<venvpy> scripts/tensorrt_export.py --components unet --batch-sizes 8 --output-dir <partial>/bs8 --precision fp16
   --save-format exported_program --workspace-gb 2 --min-block-size 1 --unet-capture-dir <corpus>
   --validate-unet-capture-dir <corpus> --validate-unet-limit 16 --validate-unet-padded-batch-size 8
   --validate-unet-report-path <partial>/bs8/validation.json --require-valid-unet` (verify these flags in
  tensorrt_export.py; the file may be saved as torchscript even when exported_program is requested),
  then write fingerprint.json and atomically publish. Build log to `<repo>/logs/unet_engine_build_<key>.log`
  (or `$WORKSPACE/logs/musetalk/`).
- `validate --engine-dir D` → run `<venvpy> scripts/validate_unet_backend.py --backend trt --trt-path <ts>
  --capture-dir <corpus> --padded-batch-size 8 --limit 16 --fail-mae 0.01 --fail-max-abs 0.5 --report-path ...`
  (check its real CLI; it is another session's modified file — call it, never edit it).
- `adopt --ts PATH` → register an existing engine (legacy paths such as
  models/tensorrt_unet_sm89_bs8_local/unet_trt.ts): scan the .ts for the embedded device string (stream in
  64 MB chunks with overlap; cache the result in a sidecar keyed by size+mtime), refuse unless its
  major/minor/name match this GPU, then create the store entry with a SYMLINK to the original file (never copy
  2.2 GB, never modify the original), run `validate`, write fingerprint.json (source "adopted",
  tensorrt/torch_tensorrt versions = the venv's, with a note).
- `restore` / `publish` → remote store `MUSETALK_UNET_ENGINE_REMOTE` (`s3://bucket/prefix` or `file:///path`;
  default `s3://$TRT_ARTIFACT_S3_BUCKET/trt-artifacts/unet-engines` when TRT_ARTIFACT_S3_BUCKET is set, else none).
  Object layout `<remote>/<engine_key>/bs8/{fingerprint.json, unet_trt.tar}`; restore verifies sha256 from
  fingerprint.json, extracts into a partial dir, validates, publishes atomically. boto3 is imported lazily;
  file:// must work without boto3 (tests use it). Publish only when explicitly asked (--publish / MUSETALK_UNET_ENGINE_PUBLISH=1).
- `ensure --provision auto|adopt|restore|build|off` → if a usable engine for this GPU already exists: exit 0.
  auto = adopt known legacy paths whose embedded device matches → restore from remote → build if resources allow
  → give up. Exit 0 when an engine is usable at the end, 3 when none (server will run eager), 1 on unexpected
  error. Log every step with timings; never fatal to the boot unless the caller asks (`--require`).
- Portable validation corpus: create `calibration/unet_portable_bs8/` with 16 capture files copied (not moved)
  from `calibration/unet_multi_avatar_20260928/` (+ holdout/) spanning as many avatars as possible, and a
  manifest.json (source file, avatar, sha256, why). Add a `.gitignore` negation so this dir (and only it) is
  tracked alongside calibration/vae_decoder. Confirm the capture format is what tensorrt_export.py /
  validate_unet_backend.py read.
- Unit tests: `test_unet_engine_store.py` (repo root, unittest, CPU-only): key/slug, fingerprint parsing,
  embedded-device scan on a synthetic file, adopt refusal on mismatch, file:// restore/publish round trip with a
  fake engine, atomic partial handling. Must not import torch.

## Component C — installer
- `scripts/install_musetalk.sh` (NEW canonical installer; `setup_musetalk.sh` becomes a shim that execs it,
  translating the legacy flags --venv-path/--clean/--skip-apt/--skip-weights/--full-stack/--install-avatar-prep-deps/
  --python-bin/--artifact-dir). Options: --venv PATH (default $WORKSPACE/.venvs/musetalk_trt_stagewise),
  --matrix auto|cu121|cu128 (auto: cu128 iff GPU cc >= 10.0, else cu121; no GPU → cu121), --python python3.10,
  --clean, --skip-apt, --skip-weights, --with/--without-kokoro (default with), --with/--without-native-vp8
  (default auto = with on x86_64 + py3.10), --with-avatar-prep (mmcv/mmdet/mmpose via the existing repo wheel logic
  from setup_trt_experiment_env.sh; cu121 only), --with-legacy-int8 (nvidia-modelopt 0.23.2 etc.; cu121 only),
  --no-selftest, --check (read-only; exit 0 ok, 10 = venv missing/incompatible matrix → needs clean install,
  11 = incomplete → repairable in place; a pre-existing venv WITHOUT a stamp but passing all import and file
  checks counts as OK with a warning — this box's venv must pass --check untouched).
  Phases (idempotent, each logged with timing): apt (root only; python3.10 python3.10-venv python3.10-dev coturn
  ffmpeg git curl build-essential ca-certificates libgl1 libglib2.0-0 libsm6 libxext6 libxrender1; deadsnakes only
  if python3.10 is missing) → venv → pip install `-r requirements/server.in -c requirements/constraints-<matrix>.txt`
  with `--extra-index-url https://download.pytorch.org/whl/<matrix> --extra-index-url https://pypi.nvidia.com`
  (+ requirements/kokoro.in, legacy-int8.in when selected) → weights (download_weights.sh, now also TAESD
  pinned revision into models/taesd and, when kokoro is on, the Kokoro-82M model + default voices into the HF cache
  and en_core_web_sm) → native VP8 install + --verify → CPU import smoke with CUDA hidden (torch, diffusers,
  transformers, aiortc, av, fastapi, uvicorn, boto3, librosa, soundfile, and tensorrt/torch_tensorrt) → GPU self-test
  (Component C2) if a GPU is visible → stamp `<repo>/.runtime/install_state.json` {schema, matrix, python, venv,
  constraints_sha256, server_in_sha256, groups, versions, created_utc}.
  Never delete a venv unless --clean or --check said 10. Never require CUDA to install (GPU-less image builds work);
  warn if the GPU check fails. Respect PIP_CACHE_DIR; prefer HF_MAX_WORKERS / HF_XET_HIGH_PERFORMANCE as download_weights.sh does.
- `requirements/server.in` (top-level only), `requirements/kokoro.in`, `requirements/legacy-int8.in`,
  `requirements/constraints-cu121.txt` (from the validated freeze, minus URL/file lines and minus packages only
  the avatar-prep group needs is NOT required — constraints for uninstalled packages are harmless; keep them),
  `requirements/constraints-cu128.txt` (same, minus torch/torchvision/torchaudio/triton/torch_tensorrt/tensorrt*/
  nvidia-*/sympy/networkx lines, plus the cu128 pins listed above). Verify both resolve with
  `pip install --dry-run --ignore-installed --report` in a scratch venv under the scratchpad (not the shared venv),
  and record the result. `requirements/README.md` explains how to regenerate.
- download_weights.sh: add TAESD (pinned revision, `huggingface-cli download madebyollin/taesd config.json
  diffusion_pytorch_model.safetensors --revision <rev> --local-dir models/taesd`, skip if present) and an
  opt-in/auto Kokoro pre-cache; keep all existing behaviour.
- C2 `scripts/musetalk_selftest.py` (NEW; venv python; imports torch; run ONLY by the installer and by humans —
  implementers must NOT execute it, just py_compile it): CUDA init check, TAESD load from models/taesd + compiled
  warmup at the configured buckets + timing (ms per bs8), eager TAESD timing, optional eager UNet bs8 timing
  (`--unet`), torch_tensorrt import check. Writes `<repo>/.runtime/gpu_selftest.json` with schema
  `musetalk_gpu_selftest_v1`: {gpu:{name,compute_capability}, torch_version, cuda_ok, taesd:{compile_ok,
  compile_mode, warmup_s, ms_bs8, eager_ms_bs8, error}, unet_eager:{ok, ms_bs8, error}, trt_import_ok, created_utc}.
  Must catch compile failures and record them instead of crashing.

## Component D — launch chain (bash)
- `scripts/run_musetalk_server.sh` (NEW launcher): args --host --port --venv-path --repo-root --profile (compat,
  ignored except logged) --validate-only --print-env. Steps: resolve paths; if `MUSETALK_RECIPE=legacy_int8` exec
  scripts/run_trt_stagewise_server.sh with the same args (untouched legacy chain); unset PYTORCH_CUDA_ALLOC_CONF and
  MUSETALK_CPU_* like the old launcher; load overrides files (only-if-unset); run the resolver (Component A) →
  on exit 2 die with its message; load the resolved env (only-if-unset); if WEBRTC_VP8_ENCODER=native run the CPU
  preflight (on failure: die unless MUSETALK_VP8_FALLBACK=1, then export pyav with a warning); log a compact recipe
  summary (VAE, UNet + engine path, buckets, workers, cache MB, VP8, overrides files used); --print-env prints the
  effective env of all managed keys and exits 0; --validate-only exits 0 after checks; otherwise
  `cd $REPO_ROOT && exec "$VENV_PY" api_server.py --host "$HOST" --port "$PORT"`.
  Must never import torch itself.
- `scripts/run_webrtc_relay_api_server.sh`: final exec → `${MUSETALK_SERVER_LAUNCHER:-$REPO_ROOT/scripts/run_musetalk_server.sh}`.
- `scripts/vast_server_ctl.sh`: default launcher = run_musetalk_server.sh (relay wrapper when WEBRTC_RELAY_ENABLED=1,
  as today); record the log byte offset before spawning; after /health passes run
  `musetalk_host_profile.py verify-log` (expect taesd for recipe fast, UNet per `.runtime/musetalk_resolved.json`);
  `MUSETALK_VERIFY_RECIPE=strict|warn|off` (default strict: a mismatch stops the server and fails start).
  `status` also prints the resolved recipe summary. Keep stop/drain/restart/logs/TURN behaviour unchanged.
  Do not change the /health polling contract except to allow a longer STARTUP_TIMEOUT_SECONDS default of 900.
- `scripts/vast_onstart.sh`: keep logging, BEGIN/COMPLETE/FAILED markers (add `set -E` so the ERR trap fires in
  functions; make sure every failure path prints `VAST_ONSTART FAILED`), secrets bootstrap and TURN autogen as-is.
  Replace setup with `install_musetalk.sh --check` → skip | repair in place | clean install (SETUP_CLEAN=1 or exit 10);
  AUTO_SETUP=0 → check only, die only if the venv python is missing or exit 10. For recipe fast: skip legacy TRT
  restore and the profile selector entirely; run `unet_engine_store.py ensure --provision ${MUSETALK_UNET_ENGINE_PROVISION:-auto}`
  (non-fatal unless MUSETALK_UNET_MODE=trt) BEFORE the server start (so builds are outside the health timeout).
  For recipe legacy_int8: run the old restore + selector functions unchanged and start with the legacy launcher.
  Stop exporting PROFILE/MUSETALK_TRT_PROFILE_* into the fast chain (legacy only).
- `scripts/run_turnserver_tcp_relay.sh`: widen the default relay range to 49160-49460 (relay-only sessions use ~3
  allocations each; 41 ports capped us at ~13 sessions). Nothing else.
- `/workspace/run-musetalk-local-trt.sh` (outside the repo, this box's wrapper): keep sourcing
  /workspace/.musetalk-runtime.env; drop the sm89 profile-env/restore/selector exports (legacy only); keep
  AUTO_SETUP=0 default; exec vast_onstart.sh. Back it up first to `run-musetalk-local-trt.sh.bak-20260928`.
- `configs/musetalk_overrides.env.example` (tracked) documents the overrides file with commented one-line levers,
  including the other session's flags as commented examples (MUSETALK_TAESD_BACKEND=trt, WEBRTC_VP8_ENCODER=native, ...).
- `scripts/test_startup_scripts.sh` (NEW): bash -n on every touched script; launcher `--print-env` with injected
  host facts (MUSETALK_HOST_FACTS_JSON) and a fake venv tree, asserting layering precedence (caller > overrides >
  resolved), bucket coupling, eager-vs-trt selection, legacy recipe dispatch (dry). No GPU, no torch, no server.
