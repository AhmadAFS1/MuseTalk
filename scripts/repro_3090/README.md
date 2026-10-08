# RTX 3090 r5 measurement wrappers

Status: **CPU contract tests only; real RTX 3090 integration remains unverified.**
These entry points wrap the existing GPU path, six-avatar chin renderer, numerical
gates, pixel/landmark tool and local WebRTC rig. They do not redefine the composition
or call an HTTP health response deployment readiness.

## Contract and exits

Every command requires `--profile`, `--engine-root`, `--taesd-key`, `--taesd-dir`,
`--input-manifest`, `--out`, and a new `--label`. The engine paths/key must be actual
verified artifacts, not guessed 3090 names. Profiles contain literal non-secret
settings; they are parsed, not sourced. Explicit profile/CLI values override ambient
MuseTalk settings. The runner writes the sanitized effective profile and its hash.
It checks the loaded UNet directory, TAESD key/plan hash, and rejects fallback.
Inherited `REPRO_`, `LIVE15_`, BLAS and PyTorch/Inductor tuning settings are removed;
discarded key names (not their values) are recorded. UNet hash/probe checks are
mandatory: exact probe hashes must match the frozen manifest, or portable
cross-GPU numerical evidence must meet that manifest's original bound.

Each command creates `<out>/<label>_<suite>/report.json`. Existing directories are
refused so a stale successful report cannot hide a failed rerun. Child logs, commands,
return codes, durations, GPU-process observations and raw reports remain beside it.

| Exit | Status | Meaning |
|---|---|---|
| 0 | `PASS` | Only the named measurement/preflight met its checks |
| 1 | `FAIL` | A valid measurement missed its stated numerical/performance gate |
| 2 | `INVALID` | Missing/bad evidence, crash, wrong GPU/backend, short run, foreign load, etc. |
| 3 | `HISTORICAL_QUALITY_EXCEPTION` | Reserved for a separately documented reference decision; never rewritten to strict PASS |

GPU identity defaults to exactly RTX 3090/sm86, **not 3090 Ti**. `--general-gpu` is an
explicit diagnostic escape hatch that labels the actual GPU; such results are not
3090 acceptance evidence. Native/portable labels are checked against the manifest.
Preflight also checks the pinned torch 2.5.1+cu121 / torch_tensorrt 2.5 / TRT 10.3
matrix, complete plans and probe invariants, SHA-256 hashes and TAESD fingerprint.
It records physical/visible VRAM, CPU affinity and cgroup allocation, RAM, shared
memory, free disk, versions, GPU clocks/power/temperature and Git state.

All CUDA work, including the preflight CUDA query, uses `box_guard.sh`. Each heavy
child additionally samples GPU PIDs and aborts the **owned process group** if foreign
GPU work appears or the observer fails. It never stops another server. Drain/stop
only the owned test server before running. Runtime scripts need Linux `/proc`,
`nvidia-smi`, `flock`, `setsid`, and the normal pinned r5 dependencies.

## Freeze inputs before a comparison

Restore the canonical inputs using `repro_400fps/05_fetch_inputs.sh`. The suite
requires all 352 main and 96 holdout captures, all files of the six original fixture
identities, UNet model/config, and quality-critical workflow/blending source hashes.
Freeze every other runtime weight, the real-speech/audio corpus, and recipe too:

```bash
python3 scripts/repro_3090/input_manifest.py \
  --corpus calibration/unet_multi_avatar_20260928 \
  --accepted-root /workspace/experiments/avatar_diversity_20260927 \
  --model-root models/musetalkV15 --model-root models/sd-vae \
  --model-root /actual/local/taesd/model/snapshot \
  --extra experiments/throughput300_candidate/audio_corpus \
  --extra docs/fps_comparisons/4070s_400fps_20260928/int8_study/recipe_gmac_0.50.json \
  --out /actual/run/harnesses/inputs.json
```

Paths above containing `actual` must be resolved from this host. The manifest tool
does not download assets, infer missing model locations, or overwrite a frozen
manifest. File paths are relative to the manifest. Optional `s3_objects` entries
are `{ "bucket": "...", "key": "...", "bytes": 123 }`; preflight HEADs every entry
with existing credentials, rejects missing/size-mismatched objects, and never treats
multipart ETags as content SHA-256. This is not the full production-avatar audit.

The six identity names, 240 frames, 512×896 composition, 256×256 face, original
chin refinement and blending remain canonical. `--accepted-root` relocates fixtures.
`--workspace` relocates the parent of `SoulX-FlashHead/.venv/bin/python`; this layout
is retained because the frozen canonical Tracker uses it. `--python` selects the
main pinned environment. Nothing silently installs either environment.

## Commands

Use a shell array so paths containing spaces remain intact:

```bash
COMMON=(--profile scripts/repro_3090/profiles/native.env
        --engine-root /actual/models/tensorrt_unet_stagewise_sm86_r5_v1
        --taesd-dir /actual/taesd/plan-directory --taesd-key ACTUAL_FINGERPRINT_KEY
        --input-manifest /actual/run/harnesses/inputs.json
        --out /actual/run/native --label native_v1 --target native)
bash scripts/repro_3090/00_check.sh "${COMMON[@]}"
bash scripts/repro_3090/10_gpu.sh "${COMMON[@]}" \
  --comparison-root /actual/models/tensorrt_unet_stagewise_ampere_plus_r5
bash scripts/repro_3090/20_aggregate.sh "${COMMON[@]}" --stages T SUST N15
bash scripts/repro_3090/30_quality.sh "${COMMON[@]}"
bash scripts/repro_3090/40_live.sh "${COMMON[@]}" \
  --live-avatar-file /actual/restored-test-avatar-ids.txt --live-cpus 0-7 \
  --live-levels '5 10 15' --live-soak-n 15
```

For the portable baseline select `profiles/portable.env`, its actual TAESD key/dir,
the portable engine root, `--target portable` and a separate output/label. The GPU
suite's first portable run must use `--baseline-only` when native engines do not
yet exist. It measures one engine's blocks and the same two 180-second GPU runs,
explicitly makes no comparative claim, and cannot include comparison roots. Later
comparisons require distinct roots (repeating the candidate is rejected). The GPU
block comparison must contain only engine sets that actually load on the current
device; **never give sm89 plans to sm86**. Native roots preserve exact build GPU
model matching as well as architecture. Published old comparisons retain their
defaults unless the new explicit parameters are selected.

- `10_gpu.sh`: interleaved per-block diagnostics against explicit comparison roots,
  then at least two 180-second full-GPU-path runs with real captured latents/audio,
  20-second warmup, 10-second windows, CUDA-event timing and thermal/power telemetry.
  Block inputs are accurately labelled **synthetic random buffers**; their summed
  time is not a pipeline or quality claim. Finite checks cover all distinct inputs
  in the untimed golden pass; device timing has no extra stage synchronizations.
- `20_aggregate.sh`: T=6 streams×2 windows, SUST=6×5 consecutive windows,
  N15=15×10 windows. Every window must be at least 60 seconds. Frames are validated
  against completed worker counts, divided by **one shared wall interval**, never
  independently timed per-stream FPS. Portable target is 300; native target is 400,
  without rounding for T/SUST. N15 reports every FPS but gates integrity/stability,
  not an additional invented 400 FPS requirement. All requested native T/SUST
  windows must pass. Throughput runs
  exclude capture/encoding overhead. Default 24 clip loops gives headroom against
  accidentally short windows; tune `--loops` from observations, never lower the
  60-second bar. A 120-second untimed thermal warmup occurs after CPU preparation;
  its final ~30 seconds require at least 40 telemetry samples and ≤3°C range.
  If still warming, rerun with greater `--thermal-warmup-s`. These explicit
  preconditioning bounds are recorded, not inferred from a fast first window.
- `30_quality.sh`: original main/holdout UNet limits, source-prefix cached/permuted
  equality, original TAESD limits/exactness, then separate six-avatar full-height
  video/array captures and per-avatar lip/aperture/pixel/landmark/flicker/chin metrics.
  Known strict failures remain FAIL. It does **not** accept inherited exceptions,
  compare a frozen reference envelope, inspect visual artifacts, audit all 16
  production avatars, or make the release quality decision. Those remain separately
  required evidence under the execution plan. Exit 0 is not visual approval.
- `40_live.sh`: isolated localhost diagnostic, explicit avatars and allocated CPU
  IDs. It runs 1/3 smoke, ascending ramp, and a ≥3600-second scored soak at the
  explicitly selected ramp-passing level. Default 15 is a candidate, **not** claimed
  capacity; if it fails, rerun a fresh label at the largest prior passing level with
  measured margin. Observer recording is disabled for scored runs. Missing freshness,
  content-frame or client/server trace-join evidence is INVALID, even if the legacy
  tool's permissive summary said PASS. P1–P3 are recomputed; unrounded original
  verdicts are retained, and send-cadence-caused late gaps fail. Co-resident CPU
  contention is disclosed. A separate actual-browser EC2/TURN call is mandatory for
  deployment acceptance; never join monotonic clocks across different machines.
- `50_startup.py`: separately maintained EC2 request-to-real-call observer. Use its
  own `--help` and contract; its API discovery is not replaced by a guessed endpoint.

For a short real-GPU sanity run, invoke the underlying canonical tool under
`box_guard` with these same explicit profiles/paths and a fresh **diagnostic** output
directory. It is expected to fail this suite's sustained-duration acceptance until
the full run is performed. Do not promote the diagnostic result.

## CPU validation and summary

```bash
python3 -m unittest discover -s scripts/repro_3090 -p 'test_repro_3090*.py' -v
python3 scripts/repro_3090/report.py /actual/run/*/report.json --out /actual/run/summary.json
```

Tests cover common-wall denominators, 399.96 FPS rejection, missing windows/frames,
thermal settling, timestamp order, wrong GPU/Ti, backend fallback and identity,
nonfinite values, failed children, stale/missing evidence, input hashes, missing S3,
literal profiles and CLI failure reports. These are **CPU tests**, not substitute
RTX 3090 smoke, speed, image quality, live capacity or startup measurements.

The old `repro_400fps/20_gate.sh` and `30_benchmark.sh` still have historical reporting
semantics. Use the new wrappers to obtain fail-closed acceptance. Their `lib.sh`
also accepts `MUSETALK_REPRO_PYTHON`, `MUSETALK_REPRO_CORPUS`,
`MUSETALK_REPRO_COMPARISON_ROOT`, `MUSETALK_REPRO_RUNTIME_ENV`, and
`MUSETALK_REPRO_EXPLICIT_PROFILE=1` (preserves candidate TAESD knobs). The native
builder now refuses undetectable GPU architecture instead of guessing sm89.
Native non-sm89 builds request strict timing-cache provenance. The underlying
builder automatically enforces it outside the historical RTX 4070 SUPER/sm89
target: no 4070 seed, no unlabelled or mismatched existing timing cache, and no
relabeling of foreign engine manifests. Same-target cache reuse records target
metadata and SHA-256 in `timing_cache.bin.json`, and TensorRT mismatch bypass is
disabled. Use a fresh native root; old unlabelled caches fail closed.
