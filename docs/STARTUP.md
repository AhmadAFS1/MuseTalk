# MuseTalk: install, boot and operate

This is the operator guide for the start/install chain reworked on 2026-09-28. The binding design
is `docs/startup_rework_20260928/STARTUP_CONTRACT.md`.

**Default:** a fresh machine boots recipe **r5** with nothing set. On any Ampere-or-newer GPU (RTX 3090,
RTX 4070 SUPER, RTX 4090, A-series, L40S, A100, H100) the boot restores r5 engines from S3 and serves the
live-tested configuration: an RTX 4070 SUPER gets its own bundle (~400 fps), every other such GPU the portable one
(the same engines built with TensorRT hardware compatibility `AMPERE_PLUS`; 307 fps on the 4070 SUPER). No `.ts`
UNet engine is built. On an older GPU (T4, V100, RTX 20xx) no r5 engine can load; that host serves eager UNet +
compiled TAESD with the same serving levers.
The chain never falls back to a slower backend silently: after `/health` passes, the start script reads the server
log to confirm which backends are actually active, and stops the server if they are not the ones it expected.
`MUSETALK_RECIPE=fast` (the `.ts` UNet recipe) and the old INT8 chain (`MUSETALK_RECIPE=legacy_int8`) remain one
line away.

## 1. The chain at a glance

```
Vast template / box wrapper
  └─ scripts/vast_onstart.sh            install check → secrets → TURN → [r5 bundle] → engines → server
       ├─ scripts/install_musetalk.sh --check     (repairs in place / clean install when needed)
       ├─ scripts/trt_artifact_bundle.py restore  (recipe r5, Ampere+ GPU: pinned S3 engine bundle, verified)
       ├─ scripts/unet_engine_store.py ensure     (adopt → restore → build; BEFORE the health timeout)
       └─ scripts/vast_server_ctl.sh start        (log offset → spawn → /health → verify-log)
            └─ [scripts/run_webrtc_relay_api_server.sh]   (when WEBRTC_RELAY_ENABLED=1)
                 └─ scripts/run_musetalk_server.sh        overrides → resolver → preflights → exec
                      ├─ scripts/musetalk_host_profile.py resolve   (stdlib, never imports torch)
                      └─ exec api_server.py
recipe legacy_int8:  run_musetalk_server.sh → exec scripts/run_trt_stagewise_server.sh (unchanged)
```

| File | Role |
|---|---|
| `scripts/install_musetalk.sh` | Canonical installer. `setup_musetalk.sh` and `scripts/setup_musetalk.sh` are shims that translate the legacy flags. |
| `scripts/vast_onstart.sh` | Boot orchestrator. Every exit path prints exactly one `VAST_ONSTART COMPLETE` or `VAST_ONSTART FAILED` marker. |
| `scripts/vast_server_ctl.sh` | `start` / `stop` / `restart` / `status` / `logs`. Stop drains first; TURN is handled as before. |
| `scripts/run_musetalk_server.sh` | Launcher: layering, resolver, preflights, then `exec api_server.py`. |
| `scripts/musetalk_host_profile.py` | Resolver: `detect`, `resolve`, `verify-log`, `engine-key`, `find-engine`. |
| `scripts/unet_engine_store.py` | Engine store: `list`, `build`, `validate`, `adopt`, `restore`, `publish`, `ensure`. |
| `scripts/lib/musetalk_env_layers.sh` | Shared env-file parser and layering helpers. |
| `configs/musetalk_overrides.env.example` | Every operator lever, commented, with its default and effect. |
| `configs/recipes/fast300.env` | The fast300 levers and their gates. All are commented out today. |
| `configs/recipes/r5.env` | Recipe r5 (the default): the bundle's engines plus the live-tested serving levers. |
| `configs/trt_bundles/<name>.json` | A pinned S3 engine bundle: URI, sha256, the hosts it fits (compute-capability range + TensorRT + VRAM, or one engine key), engine dirs, sidecar dir. |
| `scripts/trt_artifact_bundle.py` | Bundle tool: `create`, `upload`, `restore`, `adopt`, `verify`. |

Files the chain writes under `<repo>/.runtime/` (gitignored):

| File | Contents |
|---|---|
| `musetalk_resolved.env` | The resolver's output. Regenerated on every launch; do not edit. |
| `musetalk_resolved.json` | The resolver's report: host facts, every decision and its reason, warnings, errors, levers, the fast300 groups enabled and dropped, and the expected backends. |
| `musetalk_launch_<port>.json` | The launcher's record of that launch: the effective value and source layer of every managed key, and the expected backends. `vast_server_ctl.sh` verifies against it. |
| `musetalk_overrides.env` | Your overrides. Optional; nothing creates it. |
| `install_state.json` | The installer stamp. |
| `gpu_selftest.json` | GPU self-test timings. The resolver uses them for its fps estimate. |
| `native_vp8/` | The pinned native VP8 encoder. |
| `trt_artifacts/<bundle>/` | A restored pinned bundle's manifest, SHA256SUMS and restore stamp (the resolver's `bundle:` check reads them). |

## 2. Install

```bash
# Fresh machine (as root: apt packages, python3.10 venv, pinned wheels, weights incl. TAESD,
# Kokoro cache, native VP8, CPU import smoke, GPU self-test if a GPU is visible):
bash scripts/install_musetalk.sh

# Read-only check (what vast_onstart runs at every boot). Exit 0 = ok, 10 = needs a clean
# install, 11 = repairable in place. Add --check-imports for the CPU import smoke with CUDA hidden.
bash scripts/install_musetalk.sh --check

# Optional groups
bash scripts/install_musetalk.sh --with-avatar-prep     # mmcv/mmdet/mmpose for /avatars/prepare (cu121)
bash scripts/install_musetalk.sh --with-legacy-int8     # nvidia-modelopt, needed only for the rollback recipe
bash scripts/install_musetalk.sh --with-chin-tools      # dedicated MediaPipe venv at $WORKSPACE/.venvs/musetalk_chin_tools

# Blackwell (compute capability >= 10.0) is picked automatically, or forced (untested on hardware):
bash scripts/install_musetalk.sh --clean --matrix cu128
```

- **SyncNet is training-only.** The API installer omits its checkpoint by default,
  including with `--with-avatar-prep`; DWPose and S3FD remain required. Set
  `DOWNLOAD_SYNCNET_WEIGHTS=1` for an explicit checkpoint download (not a validated
  training environment). Running `download_weights.sh` directly retains its
  historical full-set default; set `DOWNLOAD_SYNCNET_WEIGHTS=0` to omit it there.
- **What gets pinned.** Top-level requirements are in `requirements/server.in` and pins in
  `requirements/constraints-cu121.txt` or `constraints-cu128.txt`. `requirements/README.md`
  explains how to regenerate them. Never `pip install -r requirements.txt`: it pulls in
  tensorflow and gradio.
- **No CUDA needed.** Installing never requires CUDA, so GPU-less image builds work. The GPU
  self-test runs only when a GPU is visible.
- **The venv is never deleted implicitly.** Only `--clean`, or a `--check` that exits 10 in
  `vast_onstart.sh` with `AUTO_SETUP=1`, rebuilds it.
- **Chin tools.** Chin alignment is an offline avatar-creation stage and is not part of serving.
  `--with-chin-tools` creates its own small venv so the chin tracker no longer borrows another
  project's venv. The resolver exports that venv's python as `MUSETALK_CHIN_TRACKER_PYTHON`.

Legacy flags still work through `setup_musetalk.sh`:

| Legacy flag | Translated to |
|---|---|
| `--venv-path` | `--venv` |
| `--python-bin` | `--python` |
| `--full-stack`, `--install-avatar-prep-deps` | `--with-avatar-prep` |
| `--install-modelopt` | `--with-legacy-int8` |
| `--artifact-dir`, `--skip-modelopt` | ignored |

## 3. Boot

**Vast template** (unchanged contract; the recipe defaults to r5, nothing to set):

```bash
PORT=8000 bash scripts/vast_onstart.sh      # any GPU; Ampere+ restores the r5 engines from S3
# the old recipes, by name:
MUSETALK_RECIPE=fast PORT=8000 bash scripts/vast_onstart.sh          # .ts UNet (built on first boot per GPU)
MUSETALK_RECIPE=legacy_int8 PORT=8000 bash scripts/vast_onstart.sh   # the old INT8 chain (RTX 3090 bundle)
```

**The standard Vast onstart template** (unchanged since 2026-09; this box runs it too, as `/root/onstart.sh`) clones
`main` (depth 1) into a stage dir, deletes `/workspace/MuseTalk`, moves the clone into place, exports the runtime
secret's ARN plus the secret-reader key, and runs
`SETUP_CLEAN=1 SETUP_FULL_STACK=1 STARTUP_TIMEOUT_SECONDS=1800 PROFILE=throughput_record PORT=8000 bash scripts/vast_onstart.sh`.
Nothing in it needs to change for r5. What that implies:
- **Every container start is a first boot.** The checkout (weights in `models/`, restored engines, `.runtime/`
  stamps, local avatars in `results/`) is deleted, and `SETUP_CLEAN=1` rebuilds the 9.5 GB venv, so each start
  re-downloads about 5.7 GB of weights and the r5 bundle (1.0 GB portable, 2.6 GB RTX 4070 SUPER). Avatars come
  back from S3 on first use. Uncommitted work in the checkout is lost: push before restarting a box.
- **The install runs before the secret is read,** so nothing in the secret can change install behaviour; runtime
  settings (recipe, `MUSETALK_R5_BUNDLE_RESTORE`, `LINGUA_*`, capacity) can live in the secret.
- **Control plane:** the worker registers only if the secret carries `LINGUA_WORKER_TOKEN` and
  `LINGUA_CONTROL_PLANE_BASE_URL` (`docs/musetalk_worker_secrets.md`).
- **GPUs:** RTX 30xx/40xx, A-series, L40S, A100, H100 get r5 (the 4070 SUPER its own bundle, the rest the portable
  one). Pre-Ampere GPUs boot without r5 (eager UNet). RTX 50xx (Blackwell) cannot boot this template: its cu128
  stack has no avatar-prep build, and `SETUP_FULL_STACK=1` requests avatar prep.

**This RTX 4070 SUPER box:** `/workspace/run-musetalk-local-trt.sh`. The proposed new version is
in `docs/startup_rework_20260928/impl/launch/run-musetalk-local-trt.sh.proposed`, with a `.diff`
next to it. It sources `/workspace/.musetalk-runtime.env`, sets `AUTO_SETUP=0`, and provisions
engines with `adopt`, so the local sm89 engines are adopted and never rebuilt at boot on the shared
box. It sets the old profile env, restore and selector knobs only when the recipe is
`legacy_int8`, then execs `vast_onstart.sh`.

What `vast_onstart.sh` does, in order:

1. **Install check.** Runs `install_musetalk.sh --check`, with `--with-legacy-int8` for the legacy
   recipe and the `SETUP_*` group flags.
   - `AUTO_SETUP=1` (default): exit 0 skips setup, 11 repairs in place, 10 or `SETUP_CLEAN=1` does
     a clean install.
   - `AUTO_SETUP=0`: check only. The boot stops only if the venv python is missing or the check
     exits 10.
2. **Post-setup validation.** Log-only. It lists model files, TAESD included. Its torch import can
   be skipped with `ONSTART_POST_VALIDATE_IMPORTS=0`.
3. **Secrets and TURN.** The AWS Secrets Manager bootstrap and the TURN env autogen run exactly as
   before.
4. **r5 bundle** (recipe r5, the default). Takes the candidate bundles that `configs/recipes/r5.env` names in its
   `r5_engines` group (`bundle:rtx4070super-r5-srcg50-int8|ampere-plus-r5-srcg50-int8`, each described by
   `configs/trt_bundles/<name>.json`, so the boot restores exactly what the resolver checks) and asks the resolver
   which ones this host fits (`musetalk_host_profile.py bundle-check --host-only`: the exact GPU model for a
   GPU-specific bundle; compute capability 8.0-9.0, TensorRT 10.3.0 and >= 8 GB VRAM for the portable one; the same
   rule the resolver's `bundle:` prerequisite applies at every launch).
   - None fits (e.g. a T4): logged and skipped; the host serves eager UNet + compiled TAESD.
   - Otherwise, in order, until one succeeds: `trt_artifact_bundle.py restore --sidecar-dir .runtime/trt_artifacts/<name> --skip-if-verified`
     from `s3://$TRT_ARTIFACT_S3_BUCKET/<s3_key>` (or `MUSETALK_R5_BUNDLE_URI`). It stages the archive in
     `tmp/trt_artifact_stage` (`MUSETALK_TRT_ARTIFACT_STAGE_DIR`), checks its sha256, extracts it, verifies every
     file and writes the stamp. Peak disk is about twice the archive size (`docs/trt_artifacts/README.md`). A
     reboot whose files still verify skips the download.
   - `MUSETALK_R5_BUNDLE_RESTORE=required` (default) fails the boot when no fitting candidate can be restored;
     `auto` boots without it; `off` skips the step.
5. **Engines.** For fast and fast300: `unet_engine_store.py ensure --kind unet_ts
   --provision ${MUSETALK_UNET_ENGINE_PROVISION:-auto}`. fast300 also ensures `taesd_trt` and
   `unet_stagewise`.
   - This runs before the server starts, so a build never counts against the health timeout.
   - It is non-fatal: with no engine, the resolver picks eager. The exception is
     `MUSETALK_UNET_MODE=trt`, which adds `--require`.
   - For r5 (the default) the `.ts` provisioning defaults to off, with or without the bundle: no multi-minute
     `.ts` build at boot. `MUSETALK_UNET_ENGINE_PROVISION=auto` builds or restores it anyway.
   - For legacy_int8, the old TRT artifact restore and profile selector run unchanged instead.
6. **Server.** `vast_server_ctl.sh start`: spawn, wait for `/health` (`STARTUP_TIMEOUT_SECONDS`,
   default 900), then verify the recipe (section 7).

**Manual, in the foreground** (debugging):

```bash
bash scripts/vast_server_ctl.sh stop
bash scripts/run_musetalk_server.sh --print-env        # what would run, with sources; changes nothing
bash scripts/run_musetalk_server.sh --validate-only    # resolve + preflights, writes *.validate.* files
PORT=8000 bash scripts/run_musetalk_server.sh          # foreground server
```

`bash scripts/vast_server_ctl.sh status` prints the process, `/health`, the resolved recipe
(VAE, UNet and engine, buckets, VP8, H.264, cache) and the last verification result.

## 4. Recipes

| Recipe | What boots | How to select |
|---|---|---|
| `fast` | `MUSETALK_VAE_BACKEND=taesd`, compiled TAESD warmed for every scheduler bucket. The UNet is TRT `.ts` bs8 when a validated engine exists for this GPU key, VRAM >= 8 GB and MemAvailable >= 10 GB; otherwise eager (the reason is in the report). `MUSETALK_TRT_FALLBACK=0`. Buckets `8/8/8`, `MUSETALK_TRT_ENABLED=0`, workers and cache sized from CPU and host RAM. | `MUSETALK_RECIPE=fast` |
| `fast300` | fast, plus each lever group of `configs/recipes/fast300.env` whose prerequisites hold on this host (engine validated for this key, code present, preflight ok). Every dropped group is recorded with its reason. **All groups are commented out today**, so fast300 currently equals fast. | `MUSETALK_RECIPE=fast300` |
| `r5` (default) | fast, plus the groups of `configs/recipes/r5.env`. Group `r5_engines` requires `bundle:rtx4070super-r5-srcg50-int8|ampere-plus-r5-srcg50-int8` (first that fits and is restored): the UNet is the stagewise INT8 srcg50 set bs16 and the VAE is TensorRT TAESD (`BUILD=0`, `STRICT=1`; `HW_COMPAT=ampere_plus` for the portable bundle), buckets 16. The serving and memory groups (deadline pacing, non-blocking handoff with packed I420, idle frame cache + warm, GC freeze and thresholds, off-loop diagnostics, thread caps, scheduler syncs off, lean avatar layout) apply on any GPU. Local Kokoro and the load-test telemetry ship off. | nothing |
| `legacy_int8` | The old chain, exactly: `run_trt_stagewise_server.sh` with its profile env (INT8 SD-VAE, about 4x slower on the decoder). Overrides files and the resolver are not applied. | `MUSETALK_RECIPE=legacy_int8` |

**fast300 levers.** Each group in `configs/recipes/fast300.env` names its human gate, which must
PASS on this box before the orchestrator uncomments it. It also names its machine-checked
prerequisites: `engine:unet_stagewise`, `engine:taesd_trt`, `engine:unet_ts`, `unet:trt_any`,
`vp8:native_preflight`, `h264:native_fallback`, `gpu:nvenc` and `code:<path>`.

- **Engines:** stagewise FP16 UNet bs16 with buckets 16, TAESD TRT with `BUILD=0` and `STRICT=1`,
  and CUDA graphs on the `.ts` UNet.
- **Memory:** free the eager UNet; the avatar mask and frame layouts.
- **Serving:** non-blocking handoff, idle frame cache, GPU pipeline depth 2, EDF scheduler,
  skip-GPU-for-raw, thread caps, `MALLOC_ARENA_MAX`.
- **Encoders:** `WEBRTC_H264_IMPL=x264tuned`, native VP8.

fast300 must not become the default. The user approves it after reviewing the labelled video.

**r5.** The configuration of the 2026-09-29 live WebRTC test on one RTX 4070 SUPER
(`docs/fps_comparisons/live15_r5_20260929/README.md`: 10 calls per worker pass with margin, 15 is at the knee),
minus the test-rig lines, and the default recipe.

- **Every Ampere+ GPU, the fastest bundle that fits.** TensorRT plans are compiled for the GPU architecture they
  were built on (the RTX 4070 SUPER set does not load on an RTX 3090). The portable bundle is the same r5 built with
  TensorRT hardware compatibility `AMPERE_PLUS` (`build_unet_stagewise.py --hardware-compat ampere_plus`,
  `MUSETALK_TAESD_TRT_HW_COMPAT=ampere_plus`): it loads on every GPU of compute capability 8.0 or newer with the
  same TensorRT version, with the same accuracy but 23% fewer fps on the 4070 SUPER (307 vs 401), where the
  GPU-specific bundle is listed first (`docs/fps_comparisons/ampere_plus_r5_20260930/README.md`). A GPU-specific
  bundle for another model is one build on that GPU plus a descriptor (`docs/trt_artifacts/README.md`).
- **Pinned, not store-validated.** The engines are not engine-store entries (the source-cache INT8 blocks are
  outside the store's layout, and the numeric gates are mixed: UNet max_abs above the store's 0.5 bar, G-TAESD 5 LSB
  against 3). r5 was accepted on the labelled video review and is pinned by sha256 instead: the resolver enables
  `r5_engines` only when the stamp in `.runtime/trt_artifacts/<bundle>/` binds the pinned archive, every bundle file
  is present with its recorded size, and the host is in the bundle's range.
- **Load checks.** The server checks each plan's sha256 and runs a probe batch: bit-exact on the build GPU; on any
  other GPU model within the recorded relative-L2 bound (the plans may round differently there).

By hand on a box that already has the files:

```bash
set -a; . /workspace/.musetalk-runtime.env; set +a      # runtime credentials read trt-artifacts/*
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/trt_artifact_bundle.py --repo-root . --strict \
  --sidecar-dir .runtime/trt_artifacts/<name> adopt \
  --uri s3://$TRT_ARTIFACT_S3_BUCKET/<s3_key from the descriptor> --expected-sha256 <sha256 from the descriptor>
bash scripts/run_musetalk_server.sh --print-env    # r5_engines enabled, from which bundle?
```

`adopt` reads the manifest from the archive's first few hundred KB and hashes the local files against it; nothing
is downloaded or overwritten. Replace `adopt` with `restore ... --stage-dir tmp/trt_artifact_stage
--skip-if-verified` to download instead.

**Encoders.** `WEBRTC_VP8_ENCODER` stays `pyav` in every recipe. The media path was meant to
negotiate H.264 (`api_server.prefer_h264`), but that call runs after `setRemoteDescription`, so in practice
the answer takes the client's first codec, VP8 for Chrome and aiortc (`docs/MUSETALK_PIPELINE.md` §2.5). When
H.264 is negotiated, the lever that matters is `WEBRTC_H264_IMPL`:

- `aiortc` (default, libx264 medium)
- `x264tuned`
- `nvenc`, not recommended because of VRAM and GPU contention

Native VP8 needs two things: the native preflight must pass, and H.264-only clients must still get
a working path. The second is false in this checkout. `webrtc_offer` rejects H.264-only offers with
HTTP 400 whenever `WEBRTC_VP8_ENCODER=native`. The installer still provisions native VP8, so it is
a one-line switch once that changes.

## 5. Overrides and layering

The effective value of any knob is decided by these layers, highest first:

1. **Caller environment.** Explicit exports, the Vast template env, and the TURN env file sourced
   by ctl or the relay wrapper.
2. **Overrides files.** Set by `MUSETALK_ENV_OVERRIDES_FILE` (colon-separated, earlier files win).
   The default is `<repo>/.runtime/musetalk_overrides.env`. Missing files are skipped.
3. **`<repo>/.runtime/musetalk_resolved.env`.** Rewritten by the resolver on every launch; this is
   where enabled fast300 levers land.
4. **Code defaults.**

Rules:

- **Files are parsed, never sourced.** A file may contain `KEY=VALUE`, `export KEY=VALUE`, blank
  lines and `#` comments. Values may be bare, `'single-quoted'` (with the `'\''` idiom) or
  `"double-quoted"` (no expansion). An inline ` # comment` after whitespace is allowed.
- **Rejected lines are skipped with a warning** that gives the line number: `$`, backticks, and
  `; | & < > ( )` outside single quotes. A file can never run code.
- **Only-if-unset.** A key is exported only while it is still unset. A caller value that is set but
  empty still counts as set.
- **The resolver sees layers 1 and 2** and emits the caller's value for anything it manages, so
  dependents follow. For example, caller buckets `16` give TAESD and stagewise warmup `16` and max
  batch `16`.
- **Stale caller values are dropped.** Like the old launcher, the launcher first unsets the caller's
  `PYTORCH_CUDA_ALLOC_CONF`, `MUSETALK_CPU_*` and `HLS_CHUNK_ENCODER_{TUNE,QP}`. An overrides file
  may set them again deliberately.
- **Unknown knobs pass through untouched.** The launcher never changes the 300 fps and serving
  levers. It rejects only values the server itself would refuse at startup, and does so early:
  - `WEBRTC_H264_IMPL`, `MUSETALK_TRT_UNET_CUDAGRAPHS`, `HLS_SCHEDULER_POLICY`,
    `WEBRTC_NATIVE_VP8_THREADS`
  - `MUSETALK_UNET_BACKEND=trt_stagewise` without a complete manifest while fallback is off
  - `MUSETALK_TAESD_BACKEND=trt` with `BUILD=0` and `STRICT=1` but no engine
- **Launch-control knobs are honoured from the overrides file too.** `vast_onstart.sh` and
  `vast_server_ctl.sh` run before the launcher, but they still read `MUSETALK_RECIPE`,
  `MUSETALK_UNET_MODE`, `MUSETALK_VERIFY_RECIPE`, `MUSETALK_*_PROVISION` and
  `MUSETALK_VP8_FALLBACK` from the overrides files.
- **Relay mode.** The ICE and TURN knobs come from the TURN env file or the caller, as before.
- **`legacy_int8` ignores the overrides files.** Only `MUSETALK_RECIPE` itself is read from them.

See where every value came from:

```bash
bash scripts/run_musetalk_server.sh --print-env | grep -v '^#'
#   HLS_SCHEDULER_FIXED_BATCH_SIZES='16'  # overrides:/workspace/MuseTalk/.runtime/musetalk_overrides.env
#   MUSETALK_UNET_BACKEND='trt'           # resolved
jq '.env.MUSETALK_UNET_BACKEND' .runtime/musetalk_launch_8000.json
jq '.decisions, .warnings, .levers' .runtime/musetalk_resolved.json
```

## 6. Engine provisioning

The store manages three engine kinds, each keyed by GPU and TensorRT version (plus the
torch_tensorrt version for the `.ts`):

| Kind | Store root | Engine | Build |
|---|---|---|---|
| `unet_ts` | `models/tensorrt_unet` | torch_tensorrt static bs8 `unet_trt.ts` | about 7 min, 9.5-11 GB host RAM, +2.2 GB disk |
| `unet_stagewise` | `models/tensorrt_unet_stagewise` (also adopts the legacy `models/tensorrt_unet_stagewise_sm89/bs<N>/`) | 11 per-block FP16 plans + manifest | bs16 about 9 min at opt level 5, 1.6 GB disk |
| `taesd_trt` | `models/taesd/trt` | TAESD decoder + fused post plans | about 15 s, about 3 MB |

- **Store layout:** `<store>/<engine_key>/bs<N>/{engine files, fingerprint.json, validation.json}`.
  An example key is `sm89-nvidia-geforce-rtx-4070-super-trt10.3.0-tt2.5.0`.
- **An engine is usable only once validated for this key.** For UNets that means
  `validate_unet_backend.py` against the portable corpus `calibration/unet_portable_bs8`
  (mae_max <= 0.01, max_abs_max <= 0.5); for TAESD, `vae_fast_decoder.py verify`.

```bash
python scripts/unet_engine_store.py list                          # entries + usability on this GPU
python scripts/unet_engine_store.py ensure --kind unet_ts --provision auto
python scripts/unet_engine_store.py adopt --ts models/tensorrt_unet_sm89_bs8_local/unet_trt.ts
python scripts/unet_engine_store.py build --kind unet_stagewise --batch 16
python scripts/musetalk_host_profile.py find-engine --kind unet_ts   # what the resolver would pick
```

`ensure --provision` modes:

- `auto`: adopt known legacy engines whose embedded device matches, then restore from the remote,
  then build if resources allow.
- `adopt`, `restore`, `build`: that step only.
- `off`: nothing.

Exit codes: 0 usable, 3 none usable, 2 with `--require` and none usable, 1 failed, 4 refused.

**Remote sharing.** `MUSETALK_UNET_ENGINE_REMOTE` accepts `s3://bucket/prefix` or `file:///path`.
When `TRT_ARTIFACT_S3_BUCKET` is set, the default is
`s3://$TRT_ARTIFACT_S3_BUCKET/trt-artifacts/unet-engines`.

- `restore` verifies the sha256, validates, and publishes atomically.
- Nothing is ever uploaded unless you ask for it: `publish`, `--publish`, or
  `MUSETALK_UNET_ENGINE_PUBLISH=1`.

**Engines are arch-bound.** An sm86 engine fails on sm89 with "No compatible device was found". A
new GPU type therefore needs an engine built or restored for its own key, or it runs eager until
it has one.

## 7. Recipe verification after /health

`/health` does not report backends. So `vast_server_ctl.sh start` works from the log instead:

1. Before spawning, it records the log's byte offset.
2. After `/health` passes, it runs `musetalk_host_profile.py verify-log --log … --offset …
   --expect-vae <X> --expect-unet <Y>` with the expectations stored in the launch state:
   - VAE: `taesd`, or `taesd_trt` when TAESD TRT is requested.
   - UNet: `trt`, `eager` or `trt_stagewise`.

`MUSETALK_VERIFY_RECIPE` controls what happens on a mismatch:

| Value | On mismatch | Last result recorded in `$LOG_DIR/api_server_<port>.verify` |
|---|---|---|
| `strict` (default) | stops the server and fails `start` | FAIL |
| `warn` | logs only | WARN |
| `off` | skips the check | OFF |

For `legacy_int8` the check is skipped and recorded as SKIP. A PASS is recorded as PASS.

Strict mode catches these cases:

- A TAESD failure that silently served SD-VAE because `MUSETALK_TRT_FALLBACK=1`.
- A TAESD TRT load or verify failure that silently served compiled TAESD (`STRICT=0`).
- A TRT UNet or stagewise UNet that fell back to eager.
- A launcher that never ran the resolver: the launch state file is stale or missing.

## 8. Rollback

One line, in the environment or in `.runtime/musetalk_overrides.env`:

```bash
MUSETALK_RECIPE=legacy_int8
```

This runs the old chain unchanged: restore, then the profile selector, then
`run_trt_stagewise_server.sh`. On this box the proposed wrapper also keeps
`MUSETALK_TRT_ARTIFACT_RESTORE=0` and the sm89 profile env, so the rtx3090 bundle is never
extracted over the local sm89 engines.

Finer rollbacks:

- **One fast300 lever:** comment its line out in `configs/recipes/fast300.env`, or set the key to
  its default in an overrides file.
- **The TRT UNet only:** `MUSETALK_UNET_MODE=eager`.

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `Resolver refused this host/config (exit 2)` | No GPU; a Blackwell GPU with the cu121 venv; `MUSETALK_UNET_MODE=trt` without a usable engine; TRT buckets that are not multiples of 8 | Read `errors[]` in the report or the launcher output. Reinstall with `--matrix cu128`, provision an engine, or change the buckets. `MUSETALK_ALLOW_NO_GPU=1` is for tests only. |
| `start` fails with `recipe verification failed` | The server activated a different backend than resolved | Run `grep -E 'backend active\|backend: PyTorch\|TAESD TRT' $LOG_DIR/api_server_<port>.log`. Fix the engine or flag; `MUSETALK_VERIFY_RECIPE=warn` keeps it up meanwhile. |
| UNet is eager on a TRT-capable GPU | No validated engine for this key, VRAM < 8 GB, MemAvailable < 10 GB, or buckets not multiples of 8 | Check `jq .decisions .runtime/musetalk_resolved.json`, then `unet_engine_store.py list` and `ensure`. |
| `native VP8 preflight failed` | `.runtime/native_vp8` missing, or aiortc/av/cffi pins drifted | `python scripts/install_native_vp8.py && python scripts/install_native_vp8.py --verify`, or `WEBRTC_VP8_ENCODER=pyav`, or `MUSETALK_VP8_FALLBACK=1`. |
| `trt_stagewise … manifest.json is missing or incomplete` | Stagewise engines not built or adopted for `MUSETALK_UNET_STAGEWISE_BATCH` | `unet_engine_store.py ensure --kind unet_stagewise --batch 16` |
| An overrides line has no effect | The caller already exports that key (higher layer), or the line was rejected | Check `--print-env` sources and the `WARNING: <file>:<line> ignored` lines. |
| `VAST_ONSTART FAILED` right after the install check | `AUTO_SETUP=0` and `--check` exit 10 | `AUTO_SETUP=1` (optionally `SETUP_CLEAN=1`), or `bash scripts/install_musetalk.sh --clean`. |
| Health timeout on the first boot of a new GPU | Cold `torch.compile` max-autotune warmup | Raise `STARTUP_TIMEOUT_SECONDS`; set `TORCHINDUCTOR_CACHE_DIR` to persistent storage. Engine builds already run before the timeout starts. |
| `GPU power-capped` warning | `power.limit` < 0.85 × default | Expect lower fps; check `nvidia-smi -q -d POWER`. |

## 10. The shared-box GPU lease (`scripts/box_guard.sh`)

This host is shared by several agent sessions, the user's live server on :8000, SoulX and Codex.
Work that initialises CUDA, loads models, builds engines or benchmarks runs under the lease:

```bash
scripts/box_guard.sh check                     # is the box quiet? (GPU apps, util, MemAvailable, disk, load)
scripts/box_guard.sh lease                     # who holds /workspace/.gpu_lease (exit 1 if held)
scripts/box_guard.sh run --min-avail-gb 12 --wait-min 60 --label startup_engines -- \
  /workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/unet_engine_store.py ensure --kind unet_ts
```

`run` does the following:

- Waits for the flock on `/workspace/.gpu_lease`.
- Waits until no foreign GPU compute app is visible, then for a settle period.
- Waits for MemAvailable. This is min(`/proc/meminfo`, the cgroup estimate).
- Starts the command with `oom_score_adj=1000` and kills its process group if MemAvailable drops
  below 3 GB.
- Returns the command's exit code. Its own codes: 75 lease or GPU wait timeout, 76 RAM, 77 disk,
  86 watchdog kill, 87 cgroup oom_kill.
- Stands aside while `/workspace/.gpu_lease.pause` exists.

**Never run the long-lived server under box_guard.** It would hold the lease for the server's whole
lifetime and make the server the first OOM victim. Boot validation runs are the exception: they
start the server, verify it and stop it inside one guarded command. The steps are in
`docs/startup_rework_20260928/impl/launch/gpu_sequence.sh`, each followed by a safety stop.

Budgets on this 30 GB box:

| Operation | MemAvailable needed |
|---|---|
| Server start with the `.ts` UNet | about 14 GB (VmHWM about 9.5-10.5 GB while the 2.2 GB engine loads) |
| `.ts` build | at least 14-16 GB |
| TAESD TRT build | about 6 GB |

Keep at least 12 GB of disk free.

## 11. Knob reference (launch chain only)

The server levers are documented in `configs/musetalk_overrides.env.example`. The knobs below
belong to the launch scripts themselves.

| Knob | Default | Read by |
|---|---|---|
| `MUSETALK_RECIPE` | `r5` | all |
| `MUSETALK_ENV_OVERRIDES_FILE` | `<repo>/.runtime/musetalk_overrides.env` | all |
| `MUSETALK_RUNTIME_DIR` | `<repo>/.runtime` | launcher, ctl |
| `MUSETALK_RESOLVED_ENV_FILE` / `MUSETALK_RESOLVED_REPORT_FILE` | `$RUNTIME_DIR/musetalk_resolved.{env,json}` | launcher |
| `MUSETALK_LAUNCH_STATE_FILE` | `$RUNTIME_DIR/musetalk_launch_<port>.json` | launcher, ctl |
| `MUSETALK_RESOLVER` / `MUSETALK_RESOLVER_PYTHON` | `scripts/musetalk_host_profile.py` / venv python | launcher |
| `MUSETALK_VP8_FALLBACK` / `MUSETALK_VP8_PREFLIGHT_TIMEOUT_SECONDS` | `0` / `60` | launcher |
| `MUSETALK_LAUNCHER_DRY_RUN` | `0` | launcher: print the final exec, exit 0 |
| `MUSETALK_SERVER_LAUNCHER` | `scripts/run_musetalk_server.sh` | ctl, relay wrapper |
| `MUSETALK_VERIFY_RECIPE` / `MUSETALK_VERIFY_TIMEOUT_SECONDS` | `strict` / `60` | ctl |
| `STARTUP_TIMEOUT_SECONDS` | `900` | ctl |
| `AUTO_SETUP` / `SETUP_CLEAN` / `SETUP_SKIP_APT` / `SETUP_SKIP_WEIGHTS` | `1` / `0` / `auto` / `0` | onstart |
| `SETUP_FULL_STACK` / `SETUP_INSTALL_AVATAR_PREP_DEPS` | `0` | onstart → `--with-avatar-prep` |
| `SETUP_KOKORO` / `SETUP_NATIVE_VP8` / `SETUP_CHIN_TOOLS` / `SETUP_MATRIX` / `SETUP_SELFTEST` / `SETUP_CHECK_IMPORTS` | installer defaults | onstart |
| `MUSETALK_UNET_ENGINE_PROVISION` / `MUSETALK_TAESD_TRT_PROVISION` / `MUSETALK_UNET_STAGEWISE_PROVISION` | `auto` | onstart |
| `ONSTART_POST_VALIDATE_IMPORTS` | `1` | onstart |
| legacy only: `MUSETALK_TRT_ARTIFACT_RESTORE`, `MUSETALK_SELECT_BEST_TRT_PROFILE`, `MUSETALK_TRT_PROFILE_ENV_FILE`, `MUSETALK_TRT_PROFILE_ENV_LOAD`, `PROFILE` | unchanged | onstart, legacy launcher |

## 12. Tests

```bash
bash scripts/test_startup_scripts.sh             # CPU only: no GPU, no torch, no server, no network
bash scripts/test_startup_scripts.sh -k launcher # one group: syntax parser launcher ctl onstart relay shim integration
bash docs/startup_rework_20260928/impl/launch/gpu_sequence.sh --list   # GPU validation steps (box_guard-wrapped)
PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest test_musetalk_host_profile test_trt_artifact_bundle   # resolver, bundle tool
```

The CPU suite runs each scenario in a throw-away repo skeleton. It uses stub resolver, server,
installer and store scripts, a fake venv whose python is torch-free, and file:// health URLs.

The `integration` group runs the real resolver with injected host facts
(`MUSETALK_HOST_FACTS_JSON`). It covers:

- eager versus TRT selection
- bucket coupling
- `MUSETALK_UNET_MODE=trt` hard errors
- fast300 with every lever off
- r5 (the default) without a bundle (engine group dropped, serving levers on), with the portable bundle on a
  4070 SUPER and a 3090, with both bundles (the GPU-specific one wins on its GPU only), on a T4 (dropped) and after
  a bundle file changed
- the `verify-log` exit codes

The `onstart` group covers the r5 bundle step: the default boot's pinned restore, a GPU only the portable bundle
fits, falling back to the next candidate when a restore fails, a GPU no bundle fits, a failed restore under
`required` / `auto` / `off`, and `MUSETALK_RECIPE` set by the runtime secret. `test_trt_artifact_bundle.py`
restores real (tiny) bundles: legacy root sidecars unchanged, `--sidecar-dir` with relative symlinks, the stamp,
`--skip-if-verified`, a wrong sha256 and `adopt`.
