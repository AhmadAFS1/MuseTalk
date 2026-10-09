# RTX 3090 r5 measurement wrappers

Status: actual RTX 3090 measurements exist, but 400 FPS/quality release acceptance
is unmet. The isolated final-Conv FP32 candidate routing is CPU-tested only.
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

### Explicit final-Conv FP32 diagnostic candidate

`taesd_fp32_candidate_child.py` is an opt-in, single-child launcher for the
unchanged full TAESD gate or canonical six-avatar capture. It binds an explicit
absolute manifest and SHA, requires the full 352-main/96-holdout corpus, prevents
gate metadata writes and rejects reduced/tuned captures. A scoped import hook
selects the isolated loader only when the original target imports the decoder;
CPU spawn workers remain free of GPU imports. Loader, argv, environment, path
and working directory are restored even on failure. Numerical failure exits are
not changed into passes. Successful candidate invocations and engine identities
must appear in the separate child receipt.

`taesd_fp32_candidate_quality.py` wraps the original quality runner without
editing it. Only the gate and capture children are routed; UNet/source-prefix
and per-avatar metric children, GPU guard/watch, full input preflight and
original report schema remain unchanged. Exact runner flags are required (no
abbreviations or general-GPU escape). Each launcher executes its SHA-verified
bytes at execution time, not a stale earlier check or bytecode-cache import.
The inert true `-c` main lets spawn import only the CPU worker module.

After an actual explicit candidate build, use the real manifest/key/digests:

```bash
python scripts/repro_3090/taesd_fp32_candidate_quality.py --enable \
  --manifest /absolute/candidate/taesd_trt_ACTUAL_KEY.json \
  --manifest-sha256 ACTUAL_MANIFEST_SHA256 \
  --child-sha256 ACTUAL_LAUNCHER_SHA256 --proof /absolute/new-routing-proof.json \
  -- quality --profile scripts/repro_3090/profiles/native.env \
  --engine-root /absolute/native-unet --taesd-dir /absolute/candidate \
  --taesd-key ACTUAL_KEY --input-manifest /absolute/frozen-inputs.json \
  --out /absolute/new-output --label actual_candidate_quality
```

This is not an input-lineage waiver. The original 878-file preflight must still
pass for its original lineage. A reviewed successor comparison needs explicit
provenance; never substitute a successor SHA for an original-input PASS or
widen the frozen 698 bounds. Current protected-host download metadata cannot
restore the exact original metadata bytes. All candidate acceptance fields
remain false: CPU routing tests are not a GPU build, actual precision evidence,
quality-parity decision, FPS measurement or release authorization.

The A2 scheduler pair has an explicit successor input lineage, not a rewritten
reference manifest. `freeze_tracking_lineage.py` requires the original 878-file
manifest digest and retains every path. Only the reviewed default-off worker
revision and 12 allowlisted Hugging Face download-metadata files may differ;
each metadata ETag must identify its unchanged original model payload. Missing
files, changed weights/fixtures/canonical math, unknown worker bytes and any
other change fail closed. `tracking-a2-lineage-v1.json` records every difference.
This scope is scheduler equality only: the old input-check failure and 698
frozen quality bounds remain intact, and native release quality stays rejected.

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
  A separate opt-in `scripts/chin_multistream_cpu_telemetry.py` entry point records
  current-cgroup-v2 CPU counters before and after each window, without editing
  the canonical renderer. Missing, unsupported, reset or migrated counters remain
  unavailable/invalid, never zero-throttling claims. This interval includes final
  worker-report collection and does not replace the composed-frame FPS clock.
  Kernel throttled time is not lost wall time or causal proof. Historical pinned
  renderers/launchers remain unchanged; use the wrapper only through a separately
  versioned/preregistered harness binding, never a hidden historical pin update.
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

### Default-off ordered tracking overlap experiment

`20_aggregate.sh ... --tracking-overlap --tracking-parity-report /actual/pair/report.json`
explicitly enables one outstanding
canonical `Tracker.track()` call per stream on a helper thread. The next call
starts before current-frame filter/composition; frame order, shared-memory writer
count, original Tracker/FaceMesh source, copied landmarks, three-tap chin math,
clip reset boundaries, GPU batch/decode shapes and shared FPS denominator are
unchanged. Without the flag, the serial worker path remains selected. Serial-mode
rendering and non-aggregate suites reject the flag instead of ignoring it.

Reports bind the mode in CLI arguments and every worker; mismatches are INVALID.
In overlap mode `tracking_ipc_ms` measures the full canonical call's service time,
which overlaps composition/filter and is **not additive critical-path time**.
`tracking_overlap_wait_ms` separately records main-thread blocking wait and
`tracking_overlap_submit_ms` records submissions. FaceMesh time remains a subset
of tracking service. Cleanup aborts only the exact retained owned Tracker process
when a call remains outstanding; normal completion never signals it.

CPU tests use synthetic inputs to exercise the actual worker loop, two clip
resets, ordering, three-tap output/clip hashes, explicit activation, timeout/error
cleanup, and deterministic event-based proof of concurrency. They establish no
real FaceMesh, pixel, GPU throughput, quality or production acceptance. Before
long GPU measurements, paired same-engine serial/overlap captures must match
generated faces, generated landmarks, chin deltas and raw refined-frame hashes
for all six canonical identities. Only then run the unchanged T/SUST gates. This
scheduling experiment cannot repair or excuse an independently rejected engine.

`25_tracking_parity.sh` performs the required pair under the same frozen-input
preflight and GPU-process watchdog as the other suites. Both children use six
streams, one240-frame clip per identity, identical code/config/backends and raw
array/video capture; only the explicit overlap flag and output label differ.
The wrapper checks GPU UUID before and after each child and rechecks frozen
files. The CPU comparator validates actual uint8 faces and finite exact-shape
FP32 landmarks/FP64 chin arrays without NumPy/pickle, plus completed refined-frame
hashes. Byte differences are FAIL; absent, corrupt, stale, changed-source or
wrong-mode evidence is INVALID. No encoded video bytes or capture FPS are used
to assert parity or400FPS performance.

The successful pair's `report.json` is mandatory for overlap aggregate tests.
Before a long run the runner re-reads both captures and arrays, recomputes the
comparison, and requires the same GPU UUID, complete engine/decoder/input/profile
identity and current harness hashes. A receipt is not transferable to a different
host or candidate. Quality-reference/visual acceptance remains separate.

```bash
# COMMON uses the actual verified roots/key/input manifest from above.
bash scripts/repro_3090/25_tracking_parity.sh "${COMMON[@]}"
# Use a distinct new label/output directory; do not reuse a prior suite path.
bash scripts/repro_3090/20_aggregate.sh "${COMMON[@]}" \
  --tracking-overlap --tracking-parity-report /actual/pair/report.json --stages T SUST
```

```bash
PYTHONPATH=scripts/repro_3090 python3 -B -m unittest discover \
  -s scripts/repro_3090 -p test_tracking_overlap.py -v
PYTHONPATH=scripts/repro_3090 python3 -B -m unittest discover \
  -s scripts/repro_3090 -p test_tracking_parity.py -v
```

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

## Separate production 16×3 pose render audit

`production_pose_audit.py` adapts **restored production caches**, not the six
canonical fixture identities. Default is a CPU-only plan; actual GPU integration
remains unverified until run after the candidate engine/quality freeze. It does
not download, prepare, repair, re-encode or write into the avatar caches.

```bash
POSE=(--publication character_factory/generated/lumatalk_four_language_wardrobe_v2/s3_publication.json
      --avatars-root /actual/restored/avatars
      --audio /actual/known-real-speech.wav --audio-sha256 ACTUAL_64_HEX_SHA256
      --speech-source 'Actual recording provenance / transcript reference')
python3 scripts/repro_3090/production_pose_audit.py "${POSE[@]}" \
  --out /actual/run/production_pose_plan

# Separate, NEW output; run only after benchmark/quality candidate freeze:
/actual/pinned/venv/bin/python scripts/repro_3090/production_pose_audit.py "${POSE[@]}" \
  --out /actual/run/production_pose_render --execute-render \
  --profile scripts/repro_3090/profiles/native.env \
  --engine-root /actual/native/engine-root --taesd-dir /actual/taesd/dir \
  --taesd-key ACTUAL_20_HEX_KEY --whisper-root /actual/local/whisper \
  --tracker-python /actual/SoulX-FlashHead/.venv/bin/python \
  --frozen-inputs /actual/run/production_pose_inputs.json
```

Execution self-wraps the existing GPU lease and foreign-GPU watchdog; do not nest
another `box_guard run`. Use the pinned main Python environment. No model download
or TensorRT build is allowed. All selected caches must already exist. The existing
tracker/recipe requires native 512×896 source frames, 256×256 generated faces,
24 fps and at least 240 source/cycle frames; other resolutions are rejected, never
resized to fit. The first 240 saved cycle frames share the same first ten seconds
of supplied speech across poses. `--avatar-id ID` (repeatable) permits a smoke
subset, explicitly labelled incomplete—not a full-48 result.

The frozen-input manifest uses the existing `files: [{path, sha256}]` schema;
relative paths resolve against its directory. It must cover the exact real-speech
file, **every file under the supplied local Whisper directory**, and every source
listed in the adapter's `CRITICAL` constant (chin/tracker/blending, audio/positional
encoding, GPU issuer, render helpers and runtime backends). The earlier six-fixture
manifest does not necessarily contain these extra sources/audio. Create a new
manifest with those `--extra` inputs; never overwrite the earlier measurement
freeze. Engine plans, TAESD fingerprints and loaded backend/probe identities are
verified separately. Selected cache files are SHA-bound before and after rendering;
their encoder/preprocessing provenance is not inferred from tensor shape.

Each pose writes `pose.json`, native-resolution source | standard | 100%-refined-chin
contact samples, lossless `frames.mkv`, an audio-muxed `review_with_audio.mkv`, and
source/generated landmarks plus chin deltas. Existing `GpuIssuer`, frozen chin
functions, recipe checks and lossless clip verification are reused without changing
their math. This offline audit deliberately makes no scheduling/throughput claim.

`PASS` means only that the selected offline renders and recipe checks passed;
`FAIL` preserves a recipe-check failure, and `INVALID` covers missing/incompatible
inputs, backend/probe mismatch or execution failure. Every report retains
`visual_review_status=NOT_PERFORMED` and `release_ready=false`. Inspect every pose's
speech-linked clip/contact samples and record a separate Codex visual review before
claiming visual approval. A single shared speech clip is not multilingual coverage.

**Live API parity is a separate gap:** the current `APIAvatar.compose_frame` uses
the standard saved-mask blend, not this refined chin-tracking path. Successful
offline production samples cannot establish 100% chin behavior in live API calls,
pose transitions, original encoder compatibility, or release readiness.

## Freeze quality noise before native candidate evaluation

`quality_envelope.py` is a CPU-only, two-phase evidence comparator. First complete
two separate canonical `30_quality` runs with the portable AMPERE_PLUS profile and
the same frozen quality-input manifest; then freeze their bounds **before starting
the native quality run**. Run this helper on the machine where the manifest's
relative paths, restored fixture bundle, and calibration bundle still resolve.
It freshly hashes those inputs and raw/video artifacts; copied JSON alone is not
sufficient for a freeze.

```bash
python3 scripts/repro_3090/quality_envelope.py freeze \
  --reference /actual/portable_ref1_quality/report.json \
  --reference /actual/portable_ref2_quality/report.json \
  --inputs /actual/frozen/quality-inputs-v1.json \
  --fixture-root /workspace/experiments \
  --fixture-sidecar /workspace/MuseTalk/.runtime/trt_artifacts/repro-avatar-diversity-20260927 \
  --calibration-root /workspace/MuseTalk \
  --calibration-sidecar /workspace/MuseTalk/.runtime/trt_artifacts/repro-calibration-unet-multi-avatar-20260928 \
  --out /actual/quality/reference-envelope.json

# Record the emitted envelope SHA before launching native candidate evaluation.
python3 scripts/repro_3090/quality_envelope.py compare \
  --candidate /actual/native_quality/report.json \
  --inputs /actual/frozen/quality-inputs-v1.json \
  --envelope /actual/quality/reference-envelope.json \
  --envelope-sha256 ACTUAL_64_HEX_SHA256 \
  --out /actual/quality/native-comparison.json
```

The required manifest covers canonical fixture files and audio, all 448 calibration
captures, models, accepted chin/blending, and metric/tracker implementations. The
helper requires the pinned historical commit `5cc706e90e50e93da1310628c025a84199cd8042`
to be available locally for `git show`. Historical 4070 reports are eligible only
with matching pinned report hashes, fixtures/conditioning/audio lineage, A-frame
hashes, metric AST (only two filesystem-routing assignments may differ), library
versions, and unchanged numerical-gate code. Missing comparability is `INVALID`,
not a silently dropped reference. Existing bundle receipts and sidecars are checked;
the helper does not re-download their original archive bytes.

Each of 99 named metrics per canonical avatar and 104 UNet/TAESD metrics has an
explicit error/similarity direction. TAESD includes separate errors for each of
14 calibration avatars. Error limits are the worst applicable historical/portable
reference plus that metric's observed portable-repeat spread; similarity limits
are the lowest reference minus the spread. There is no rounding epsilon, invented
4070 value, candidate-derived noise, or post-candidate widening. Unknown/missing
metrics, changed input/engine/recipe hashes, incomplete captures, broken hard
exactness, or an earlier candidate timestamp are `INVALID`.

Outputs are exclusive-create and bind the helper, policy, reports, engines, raw
capture pixels, arrays and review videos by SHA-256. Numerical parity can be `PASS`
while original strict gates remain `FAIL`; both are preserved. Optional repeatable
`--visual-evidence FILE` only hashes externally recorded review evidence. The helper
never performs or claims visual inspection, production-pose acceptance, or release
approval (`release_ready=false`). Exit codes are 0 for a valid freeze/numerical
parity, 1 for numerical non-parity, and 2 for invalid evidence.
