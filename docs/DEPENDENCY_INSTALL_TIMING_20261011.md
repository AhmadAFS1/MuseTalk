# Separate server and Kokoro installation timing

Prepared 2026-10-11 against `main` revision `5cc706e90e50e93da1310628c025a84199cd8042`.
Status: **implemented locally; syntax and mock tests passed; no real dependency installation tested**.

Update 2026-10-12: the timing patch was subsequently exercised on isolated Linux
RTX 3090 rentals. Latest A3 timers: main 33m19s, incremental Kokoro 2m27s, avatar
preparation 1m10s; total 36m56s. Earlier A2 total was 16m34s. Detailed scope,
failures, S3 comparison and receipts: [experiment progress](Python-s3-venv-progress.md).
The original verification statements below describe this document's preparation
point; the Mac still received no pip/CUDA installation tests. Changes remain unpushed.

## Why

The latest stock-template RTX 3090 baseline recorded **994s = 16m34s** for Python packages:
**931s = 15m31s** for one combined server/Kokoro pip call, then **63s** for avatar preparation.
That combined interval includes resolving, downloading, wheel building and installation. It cannot
tell us how much time was specifically attributable to Kokoro or to transfer alone.

These are historical A2 measurements from this chat, instance `55426495`, observed 2026-10-11,
source `5cc706e`, advertised download 915.7 Mbps. They are not measurements from this Mac.
The evidence remains in the pre-existing `musetalk-r5-execution` checkout under
`docs/fps_comparisons/rtx3090_source_startup_20261011/a2/`; it has not been copied or published
by this instrumentation change.

## Change

[install_musetalk.sh](../scripts/install_musetalk.sh) now runs these sequential steps inside
Phase 3, using the existing strict timing/error recorder:

1. `Install server packages (server.in)`; adds `legacy-int8.in` to this label/command only if selected.
2. `Install Kokoro packages (kokoro.in)`, only when Kokoro is enabled.
3. `Install avatar-prep stack (mmcv/mmdet/mmpose)`, only when avatar preparation is selected, as before.

Each step logs `START`, `DONE`/`FAILED`, elapsed wall-clock seconds and its own final summary row.
Failures stop subsequent package steps. The total Phase 3 duration remains available; do not add
that subtotal to its component timers.

Both pip calls retain the same constraints file, PyTorch/NVIDIA indexes and optional wheelhouse.
No pins, package groups, CUDA matrix choices, Kokoro defaults, model-download behavior, engine
provisioning, rendering settings or WebRTC settings were changed. Dependency resolution remains
enabled, not bypassed with `--no-deps`.

`--plan` now prints:

- `PIP_COMMAND`: the server command, preserving the existing output key.
- `PIP_KOKORO_COMMAND`: the separate Kokoro command, or `disabled`.

Kokoro's timer measures **incremental installation into the shared venv after the server step**.
Shared packages such as PyTorch/Transformers are charged to the server step. This is not an
isolated Kokoro-only clean-install benchmark. Kokoro model/voice downloads remain in Phase 4,
not in its Python-package timer. Downloads versus resolver/unpacking time are still not separately
measured. Two constrained resolves may introduce overhead; no speedup is claimed by splitting them.

## Verification and workspace safety

The original `/Users/ahmadsmacair/MuseTalk` checkout contains uncommitted WebRTC edits overlapping
incoming main changes. Its files/index were left untouched. Main was fetched there, then pulled
with `--ff-only` in a separate sparse worktree at
`/Users/ahmadsmacair/code/musetalk-startup-timing`, branch `codex/dependency-install-timing`.
The separate checkout is based on `5cc706e`; it does not merge or stash the original WebRTC work.

Checks performed on macOS:

```bash
bash -n scripts/install_musetalk.sh scripts/test_install_musetalk.sh
python3 -B -m unittest test_install_package_timing -v
git diff --check
```

**12 mock tests passed.** [test_install_package_timing.py](../test_install_package_timing.py)
loads only selected shell function definitions, substitutes a shell function for the venv
interpreter, and advances a fake clock. It tests cu121/cu128 command construction, common pins
and indexes, optional-group skipping, wheelhouse/path quoting, independent elapsed records,
and fail-fast behavior. It never runs the installer entry point, pip, apt, venv creation or CUDA.
The existing [Linux installer test suite](../scripts/test_install_musetalk.sh) plan expectations
were updated but that full suite was not executed on this Mac.

No pip install/dry-run, dependency resolution experiment, Linux runtime install, GPU rental,
S3 upload, template change or production change was performed. Changes are uncommitted/unpushed.
A future matching-Linux test must establish actual durations and dependency/runtime compatibility.

## Next analysis

[S3 environment archive feasibility, costs and integration pitfalls](PYTHON_ENV_S3_ANALYSIS_20261011.md).
