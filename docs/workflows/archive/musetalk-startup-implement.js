export const meta = {
  name: 'musetalk-startup-implement',
  description: 'Implement the MuseTalk fast-recipe start/install rework (4 parallel owners) and integrate with CPU-only tests',
  phases: [
    { title: 'Implement', detail: 'resolver, engine store, installer, launch chain in parallel (disjoint files)' },
    { title: 'Integrate', detail: 'run all CPU-only tests end to end, fix cross-component mismatches' },
  ],
}

const SP = '/tmp/claude-0/-workspace/acfff43b-67b7-4b45-b762-bfdcdc32ca14/scratchpad'
const COMMON = `
You are one of four implementers of the MuseTalk start/install rework in /workspace/MuseTalk (git branch main).
READ FIRST, fully: ${SP}/STARTUP_CONTRACT.md — it is binding (interfaces, file ownership, env names, hard rules).
Background research (read the parts relevant to you; they have file:line evidence):
  ${SP}/r7.txt start/install scripts map, ${SP}/r8.txt portability, ${SP}/r5.txt env-knob inventory,
  ${SP}/r6.txt WebRTC runtime needs, ${SP}/r0.txt in-flight work, ${SP}/r1.txt 300fps plan, ${SP}/r2.txt TAESD/chin.
HARD RULES (repeat of the contract): no GPU use, no torch/tensorrt imports in your own test commands, no server start/stop,
no curl to :8000 other than GET /health, no pip install into /workspace/.venvs/musetalk_trt_stagewise, no edits to files you do
not own (listed below), no edits to the do-not-modify list, no git commit/stash/checkout/branch, never print secrets, never touch
/workspace/MuseTalk-perf300. Scratch work goes under ${SP}/<your-key>/. Keep >= 12 GB free on /workspace (check df first).
Write production-quality code that matches the surrounding repo style (look at scripts/vast_onstart.sh and
scripts/select_unet_trt_profile.py + test_select_unet_trt_profile.py for idioms). Add concise comments only where they help.
When done, return the structured report. Be honest about anything untested.
`

const REPORT = {
  type: 'object',
  properties: {
    files_created: { type: 'array', items: { type: 'string' } },
    files_modified: { type: 'array', items: { type: 'string' } },
    cli_summary: { type: 'string', description: 'Exact CLI/usage of what you built, as other components must call it' },
    tests_run: { type: 'array', items: { type: 'object', properties: {
      command: { type: 'string' }, result: { type: 'string' } }, required: ['command', 'result'] } },
    deviations_from_contract: { type: 'array', items: { type: 'string' } },
    open_issues: { type: 'array', items: { type: 'string' } },
    notes_for_integrator: { type: 'string' },
  },
  required: ['files_created', 'files_modified', 'cli_summary', 'tests_run', 'deviations_from_contract', 'open_issues', 'notes_for_integrator'],
}

const OWNERS = [
  { key: 'A-resolver', prompt: `YOU OWN Component A: scripts/musetalk_host_profile.py and test_musetalk_host_profile.py (repo root, unittest).
Implement detect / resolve / verify-log / engine-key / find-engine exactly per the contract (stdlib only, python3>=3.8, no torch).
Engine lookup reads the Engine store layout from the contract (fingerprint.json files); do not implement building (Component B does).
Confirm the exact log strings for verify-log by reading scripts/avatar_manager_parallel.py and scripts/trt_runtime.py and
scripts/vae_fast_decoder.py (grep 'backend active'); handle both TRT and eager UNet wording.
Test with MUSETALK_HOST_FACTS_JSON fixtures covering: 4070S sm89 12 GB with a matching engine -> trt; same with low MemAvailable -> eager;
RTX 3090 sm86 24 GB without engine -> eager; RTX 5090 sm120 with a cu121 venv -> hard error (exit 2); H100 sm90 80 GB; T4 sm75 16 GB;
power-capped GPU warning; no GPU -> exit 2; caller overrides (HLS_SCHEDULER_FIXED_BATCH_SIZES=16 couples warmups and max batch;
MUSETALK_UNET_BACKEND=trt_stagewise passes through without TRT paths; MUSETALK_UNET_MODE=trt without engine -> exit 2; buckets 4 with a TRT
engine -> eager in auto); cgroup CPU quota and memory limit handling (point the module at fake /proc and /sys paths via env or function args);
gpu_selftest.json compile_ok=false -> MUSETALK_TAESD_COMPILE=0; env-file quoting round-trips values with spaces/quotes/'$'.
Also run 'python3 scripts/musetalk_host_profile.py detect --repo-root /workspace/MuseTalk --venv /workspace/.venvs/musetalk_trt_stagewise'
on the REAL host (read-only: nvidia-smi query + /proc) and a real 'resolve' into your scratch dir (NOT .runtime/), and include the resolved
env and the report's decisions in your notes. On this box a usable engine will not exist in the store yet (Component B adopts it), so expect eager
with a reason — that is correct.` },
  { key: 'B-engine-store', prompt: `YOU OWN Component B: scripts/unet_engine_store.py, test_unet_engine_store.py (repo root), the new
calibration/unet_portable_bs8/ corpus (+ manifest.json) and the .gitignore negation for it. Implement list/build/validate/adopt/restore/publish/ensure
per the contract. Read scripts/tensorrt_export.py (argparse, how it loads --unet-capture-dir and writes unet_trt_meta.json / the .ts file name and
save-format fallback) and scripts/validate_unet_backend.py (argparse; it is main-branch code now) so your subprocess command lines are exactly right.
Read calibration/unet_multi_avatar_20260928/manifest.json to choose 16 captures across avatars (main + holdout) for the portable corpus; copy them
(cp, not mv; ~0.5 MB each) and write a manifest with provenance + sha256. Check git check-ignore shows the new dir is NOT ignored after your
.gitignore change while calibration/unet_multi_avatar_20260928 still IS ignored (do not git add anything).
For the embedded-device scan, test on a synthetic file AND measure (read-only) how long a real scan of models/tensorrt_unet_sm89_bs8_local/unet_trt.ts
takes and at which byte offset the device string sits (report both; the file is 2.2 GB, reading it is fine, do not modify it).
Do NOT run build/validate/adopt for real (they need the GPU) — implement them, unit-test the orchestration with a fake venv python / fake
tensorrt_export script via an env override (e.g. MUSETALK_UNET_ENGINE_PYTHON) so tests stay CPU-only. file:// remote round trip must be tested.
'ensure --provision auto' on this box must, later on the GPU, adopt models/tensorrt_unet_sm89_bs8_local/unet_trt.ts (a symlink to
../tensorrt_unet_static_bs8_20260529/unet_trt.ts) — make sure symlinked legacy paths resolve correctly and the original is never modified.
Share the engine_key/slug/fingerprint helpers with Component A by importing from scripts/musetalk_host_profile.py IF that file exists when you
finish; otherwise implement them locally with identical semantics and note it for the integrator (the integrator will dedupe).` },
  { key: 'C-installer', prompt: `YOU OWN Component C: scripts/install_musetalk.sh (new), setup_musetalk.sh (becomes a shim), download_weights.sh
(add TAESD + Kokoro pre-cache, keep all existing behaviour), requirements/ (server.in, kokoro.in, legacy-int8.in, constraints-cu121.txt,
constraints-cu128.txt, README.md) and scripts/musetalk_selftest.py (new; py_compile only — never execute it, it uses the GPU).
Derive constraints-cu121.txt from ${SP}/venv_freeze_cu121.txt (the validated live venv). Derive server.in from what the old installer
(scripts/setup_trt_experiment_env.sh phases 3-6) installs plus cffi==2.1.1 and onnx (keep onnx; drop nvidia-modelopt into legacy-int8.in).
Kokoro group: kokoro, misaki[en], espeakng-loader, phonemizer-fork, num2words, loguru, spacy + spacy-curated-transformers, and the en_core_web_sm 3.8.0
wheel URL (with its sha256 from the freeze) — confirm what kokoro/misaki import at runtime by reading their site-packages in the live venv (read-only).
Avatar-prep group: port the mmcv/mmdet/mmpose logic from scripts/setup_trt_experiment_env.sh (repo wheel third_party_wheels/mmcv first) — cu121 only.
Verify both constraint sets resolve: create a scratch venv under ${SP}/C-installer/ with /usr/bin/python3.10 -m venv, then
pip install --dry-run --ignore-installed --report <json> -r requirements/server.in -r requirements/kokoro.in -c requirements/constraints-<m>.txt
--extra-index-url https://download.pytorch.org/whl/<m> --extra-index-url https://pypi.nvidia.com with PIP_CACHE_DIR set inside ${SP}/C-installer
(check df -h /workspace before; abort the cu128 check if free space would drop below 12 GB; delete the scratch venv and pip cache afterwards).
Report the resolved torch/tensorrt/torch_tensorrt/triton/numpy versions for each matrix. Run 'bash scripts/install_musetalk.sh --check --venv
/workspace/.venvs/musetalk_trt_stagewise' against the REAL live venv: it must be read-only (prove it: compare 'pip freeze' before/after and
ls -la --time-style=full-iso of site-packages top level) and should pass (exit 0) with a 'no stamp' warning on this box. Also test --check against
a nonexistent venv (exit 10) and a scratch venv with a missing package (exit 11). bash -n everything. Test download_weights.sh's new TAESD step in a
temp copy of the repo tree layout (download into ${SP}/C-installer/models/taesd, verify the sha256 of the safetensors equals
db169d69145ec4ff064e49d99c95fa05d3eb04ee453de35824a6d0f325513549 and config.json exists) — do NOT write into /workspace/MuseTalk/models.` },
  { key: 'D-launch-chain', prompt: `YOU OWN Component D: scripts/run_musetalk_server.sh (new), scripts/vast_server_ctl.sh, scripts/run_webrtc_relay_api_server.sh,
scripts/vast_onstart.sh, scripts/run_turnserver_tcp_relay.sh (relay range only), /workspace/run-musetalk-local-trt.sh (back it up to
/workspace/run-musetalk-local-trt.sh.bak-20260928 first), configs/musetalk_overrides.env.example, scripts/test_startup_scripts.sh.
Implement exactly the layering and phases of the contract. The resolver (scripts/musetalk_host_profile.py), engine store
(scripts/unet_engine_store.py) and installer (scripts/install_musetalk.sh) are being written IN PARALLEL by other agents: code against the CLIs in
the contract. For your tests, if those files do not exist yet when you test, create minimal FAKE stand-ins inside your scratch dir and point the
scripts at them via env overrides (add MUSETALK_HOST_PROFILE_PY / MUSETALK_ENGINE_STORE_PY / MUSETALK_INSTALLER_SH override variables, defaulting
to the repo paths) — never write fakes into the repo. Preserve every existing behaviour of vast_onstart.sh / vast_server_ctl.sh that the contract
does not change (secrets bootstrap, TURN env autogen, markers, drain-on-stop, logs, pid files, public IP/port discovery). Keep the legacy chain
(run_trt_stagewise_server.sh, select_unet_trt_profile.py, trt_artifact_bundle.py restore) reachable only via MUSETALK_RECIPE=legacy_int8 and do not
edit run_trt_stagewise_server.sh itself. Make sure 'set -E' + ERR trap produce 'VAST_ONSTART FAILED' for failures inside functions (write a test that
forces a failure with fake components in a temp WORKSPACE/ONSTART_LOG and asserts the marker). vast_onstart.sh writes ONSTART_LOG defaulting to
/workspace/onstart.log: in tests ALWAYS set ONSTART_LOG, WORKSPACE, LOG_DIR and PORT (use a port like 18xxx) to scratch values and make sure nothing
touches /workspace/logs/musetalk, the live pid files, :8000, or the real TURN server. Do not actually start api_server.py: for ctl tests use a fake
launcher (MUSETALK_SERVER_LAUNCHER) that serves /health with python3 -m http.server-like stub and prints the expected 'backend active' log lines, so
you can test the verify-log step end to end (strict mismatch must stop the fake server and fail). bash -n all touched scripts.` },
]

phase('Implement')
const reports = await parallel(OWNERS.map(o => () =>
  agent(COMMON + `\nYOUR KEY: ${o.key}\n` + o.prompt, { label: 'impl:' + o.key, phase: 'Implement', schema: REPORT })
    .then(r => r ? { key: o.key, ...r } : null)
))
const ok = reports.filter(Boolean)
log(`${ok.length}/4 implementers returned`)

phase('Integrate')
const integ = await agent(COMMON + `
YOUR KEY: E-integrator. The four components have been implemented in parallel. Their reports:
${JSON.stringify(ok, null, 1)}

Your job: make them work TOGETHER, CPU-only.
1. Read every created/modified file end to end. Fix interface mismatches (CLI flags, exit codes, file paths, env names, JSON schema fields,
   engine key/fingerprint helpers duplicated between musetalk_host_profile.py and unet_engine_store.py -> make the store import the shared helpers).
2. Remove any fake stand-ins' influence: the real scripts must default to the real repo components.
3. Run ALL tests: python3 -m unittest test_musetalk_host_profile test_unet_engine_store test_select_unet_trt_profile (cd /workspace/MuseTalk,
   PYTHONDONTWRITEBYTECODE=1), bash scripts/test_startup_scripts.sh, bash -n on every touched shell script, python3 -m py_compile on new python files.
4. Real read-only dry run on this host: 'bash scripts/run_musetalk_server.sh --print-env --venv-path /workspace/.venvs/musetalk_trt_stagewise
   --repo-root /workspace/MuseTalk' with MUSETALK_RESOLVED_ENV_FILE and the report path pointed into ${SP}/E-integrator/ (add such overrides if missing)
   so .runtime/ is not written; confirm it prints TAESD + eager (no engine adopted yet) with correct buckets/workers/cache, and that an explicit
   caller export (e.g. HLS_ENCODE_WORKERS=3) and an overrides file line both win as specified. Also 'bash scripts/install_musetalk.sh --check --venv
   /workspace/.venvs/musetalk_trt_stagewise' must exit 0 and be read-only.
5. Walk through, by reading, what vast_onstart.sh will do on (a) this box via /workspace/run-musetalk-local-trt.sh, (b) a fresh Vast 3090 with
   nothing installed, (c) an RTX 5090, (d) MUSETALK_RECIPE=legacy_int8 — and fix anything that would break.
Report every fix you made. Do not start servers or use the GPU.`, { label: 'integrate', phase: 'Integrate', schema: REPORT })

return { implementers: ok, integrator: integ }
