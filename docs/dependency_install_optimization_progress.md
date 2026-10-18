# Dependency installation optimization — execution results

Started 2026-10-15 (America/Chicago; execution receipts use UTC).
Source contract: `5cc706e90e50e93da1310628c025a84199cd8042`.

This is the execution log for
[the approved plan](DEPENDENCY_INSTALL_OPTIMIZATION_PLAN_20261015.md), not a
claim that its experiments are complete. Keep failed tests and rejected branches
alongside successful ones. No custom Docker work is included.

## Current conclusion

Three **local-wheel** repetitions favor compiled uv: median **49.785s versus
132.036s** for pip through imports/CUDA validation, with the same 194 package
versions. The clean three-sample continuous four-way uv median is **132.176s**
versus source-pip **294.191s** (**55.07% less time**); all four retained candidate
samples give **133.064s / 54.77%**. Full S3 restore took **150.322s** through
matched validation and is not promoted. All **28 controlled videos** match;
equal cold-JIT policy closes the first-use comparison on this host. Canonical
reuse and real warm onstart passed; the original rental is destroyed/verified.
The fresh different-host canary also passes canonical/native/API and decoded
output checks, but took **511s for packages** because TensorRT CDN delivery alone
took **431s**. The fast first-host result is not universal. Second-host S3 lost
at package readiness (685.786s), then failed API restart; original venv restoration
passed but app recovery did not. Both completed rentals are destroyed/verified.
Full-payload overlap, corrected link-mode screens and consolidation are now
closed: overlap 10.98% and actual hardlinks 8.108% screening gains did not earn
promotion. All three task-owned rentals are destroyed/absence verified and
their expiry timers inactive. The completed gated experiment/report recommends
exact-wheel four-way acquisition, compiled uv and copy mode as the opt-in
candidate, with explicit host/CDN and measurement limits.
**Production defaults remain unchanged.**

## Baseline evidence (previous experiment, not a new matched control)

Source: [detailed profile](RTX3090_DEPENDENCY_STARTUP_PROFILE_20261015.md),
[timing receipts](dependency_profile_20261015/summary.json).

| Boundary | Seconds | Qualification |
|---|---:|---|
| Venv and packaging tools | 7 | Previous fresh US RTX 3090 |
| Main/server packages | 266 | Includes acquisition and installation |
| Incremental Kokoro packages | 38 | Separate optional group |
| Avatar-prep group | 26 | Includes its import check |
| All package groups | 330 | Same-capability comparison target |
| Main local requirement installation | 82.214 | Nested in 266s; not additive |
| Previous S3 full-venv restore to packages | 1,361.720 | Different historical run; not a matched comparison |

Targets are hypotheses: main acquisition-to-validation <=180s initially,
<=120s stretch; adoption requires >=20% lower median than a matched control,
equivalent capabilities and no meaningful first-use regression.

## Test ledger

| ID | Variant | State / evidence |
|---|---|---|
| WP0 | Exact artifacts, group closures, trusted hashes, common harness | 194-wheel Linux contract/closure and private small-wheel publication verified |
| B0 | Current source pip, fresh venv and package cache | Three continuous controls passed; validated 293.050 / 294.191 / 297.623s |
| B1 | Exact source acquisition + pip | Serial full-delivery sample passed: 249.601s packages / 265.828s validated |
| B2 | Parallel prefetch: 2, 4, conditional 8 | Two- and four-way passed; insufficient evidence/headroom benefit for eight |
| B3 | Complete local wheelhouse + pip | Three compiled repetitions passed; 132.036s validated median |
| B4 | Same wheels + pinned uv, compiled bytecode | Three repetitions passed; 49.785s validated median |
| B5 | Alternate bytecode policy + first imports/use | uv no-compile lost through imports; equal cold-JIT application comparison passed |
| B6 | Best complete delivery + installation, repeated trials | Clean candidate median 132.176s, all-four median 133.064s; source median 294.191s |
| B7 | Validated warm restart and targeted repair | Reuse ~0.4s; repair 4.31s/0 remote bytes; actual warm in-container onstart 24.400s; no provider restart claim |
| B8 | S3 venv, including bounded parallel range retrieval | First-host native/API/rollback passed, 150.322s matched validation loses; second host 685.786s packages and failed API restart/recovery, retained as failure |
| B9 | Capability audit / optional serving-only contract | Read-only audit complete; no default removal; separate reduced-capability benchmark not adopted |
| B10 | Fresh second-host confirmation | Verified US canary 56211950 passes fresh canonical/native/API and two decoded outputs; 511s package phase reveals CDN variance |
| Reliability | Interruptions, integrity/closure, ABI, auth, disk, concurrency | 135 local source/mock tests; Linux faults/locks/stamps/repair and live AWS expired GET passed; final installer hardening passed fresh Linux |
| Acceptance | Imports, CUDA/TRT/probe, codecs, generated audio/video, throughput | Controlled 28 outputs identical, SSIM 1.0; equal cold-JIT first use ~25.2s for all three methods |
| Rollout | Opt-in integration, one-worker canary, reviewed defaults | Fresh canonical boot/output passed; source-pip defaults remain unchanged pending final review |

## 2026-10-16 UTC — authoritative preflight

- Re-fetched `origin/main`; it still matches the source contract above. Existing
  local edits were preserved; no reset, stash, commit or push was performed.
- Prior profiling services are terminal, not live tests. The historical a1
  service is failed. No service was restarted on that basis.
- EC2 has only **1.8 GB free** on its 20 GB root disk. Do not download/build the
  multi-GB CUDA wheelhouse there. Store only small control-plane evidence there;
  use an owned matching Linux test worker for package tests.
- Six complete pip installation reports and their group-labelled traces remain
  in protected EC2 evidence. They describe 119 main additions, 51 Kokoro additions,
  and 20 avatar-prep additions, plus four bootstrap packages. Bootstrap has no
  pip report because its original pip did not support `--report`.
- Four source archives need a controlled one-time Linux wheel build: TensorRT's
  small Python shim, antlr4 runtime, docopt and chumpy. Their input hashes must be
  retained separately from the hashes of the resulting wheels.
- An unprivileged inventory probe could not read the existing protected
  credential-bootstrap module. It did not issue a create request. Use the
  existing root-owned EC2 control-plane path; do not weaken file permissions.

### WP0 artifact capture and acquisition safety checks

- Added `scripts/dependency_artifacts.py`: captures exact additions per group,
  preserves wheel/source hashes and metadata, records requirements fingerprints,
  and refuses to generate an offline lock while any selected input is still an
  sdist or has an unmeasured size. An unrelated source commit is provenance, not
  automatic dependency-cache invalidation.
- Initial capture **failed safely** because the historical Jinja2 and MarkupSafe
  reports had empty `archive_info`. Re-downloaded only these small files from
  their recorded PyTorch sources on EC2 and matched them to official PyPI hashes.
  This establishes current source identity, not the absent historical payload
  hashes; that limitation is retained in the manifest. No package was installed.
- Capture now succeeds: **123 server distributions including bootstrap, 174 with
  Kokoro, 194 full-stack**. Candidate artifact identity:
  `7f4c881eb68718f01fc4e5ed24d3a8a0e16ba52caadf4a320a87fde964f4e471`.
  See [acquisition evidence](dependency_optimization_20261015/README.md).
- Added `scripts/fetch_dependency_wheels.py`: independently pinned manifest
  digest, hash/size verification before publishing, bounded concurrency/retries,
  separate partial files, disk preflight, per-cache/per-file writer locks, and
  redacted failures. Downloaded bytes and verified cache hits are distinguished.
- **24 stdlib/mock tests passed** on macOS: source/manifest checks, wrong GPU
  version, missing hashes, corrupt/truncated/oversized data, cache corruption,
  authentication failure, transient retries, concurrent writer rejection,
  low-disk preflight, symlinks and stale cache identity. This is tooling evidence,
  not Linux performance or CUDA acceptance. Linux package tests remain pending.
- Fresh provider inventory also contains another user's/task's active RTX A5000
  (`56194577`); it is pre-existing and protected, along with `51074906`.
- Current US discovery found no verified offer within *all* quote caps: the
  Virginia offer's adjusted rate is $0.32144/hour, above $0.30 despite its lower
  displayed base rate. The previous Pennsylvania 8-core/unverified machine is
  available at approximately $0.18788/hour adjusted; no rental has been created
  by this optimization campaign yet. Revalidate availability before any create.

### Linux experiment tooling prepared

- Added `build_dependency_wheelhouse.py`: Linux/Python 3.10 guard, source checks
  before executing build code, pinned bootstrap/NumPy, controlled no-isolation
  builds, wheel metadata/path checks, input/output hashes and build provenance.
  It emits locks only after all four source archives have verified wheel outputs.
- Added `benchmark_dependency_install.py`: fresh claimed trial/cache, identical
  offline locks, explicit pip/uv bytecode policy, child CPU/IO and wall timers,
  exact installed inventory, package-consistency diagnostics, first/second import
  timings. Receipts explicitly exclude remote acquisition and do not claim full
  application acceptance. It refuses all package tests on macOS.
- Added `dependency_optimization_ec2.py`: unique create intent, exact reviewed
  host and quote checks, protected pre-existing IDs, two-hour independent expiry,
  temporary read-only worker credentials, source-timing patch and fresh baseline
  pip cache. Create/observation/expiry are separate; an observation failure cannot
  trigger another create. The stock template and production defaults stay intact.
- **80 tests passed** across the new safety suite and the existing profiling,
  timing and S3 safety suites; `git diff --check` passed. These are stdlib/mock
  tests, not measured installer speedups.
- Official PyPI metadata identifies the proposed uv tool as **0.13.1**. Its Linux
  wheel is 17,279,385 bytes, SHA256
  `696dcf6d807d85da978f4db8472da81dccc1254b18b52cbd1367c4d923c421e0`.
  The actual binary will be verified/pinned before execution; tool retrieval must
  be charged to full-delivery measurements. CLI behavior will be tested on Linux.
- Proceeding with one two-hour Pennsylvania test lease, **$2 campaign tranche**,
  retaining the existing <=$0.30/hour and <=$0.0015/GB caps. The matching source
  baseline uses the normal full-stack install and unchanged models/engine recipe;
  only package-group timing and an empty test-owned pip-cache path differ.

### Live owned experiment — 04:21 UTC

- Created **56199065** exactly once at 04:21:19 UTC on Pennsylvania machine
  92165 / host 443275. Adjusted quote: **$0.187875434/hour**; 100 GB disk;
  advertised 527.7 Mbps download / 39.5 Mbps upload; transfer in each direction
  $0.00000390625/GB. Independent EC2 expiry is active for **06:21:18 UTC**.
  Protected IDs: `51074906`, `56194577`. Neither was modified.
- Source checkout: 04:21:38.876 -> 04:22:45.937 UTC, **67.062s**. Apt: **19s**;
  venv/bootstrap: **7s**. Main dependency installation is still running as of
  04:25:25 UTC. These are partial measurements, not a completed new baseline.
- Initial SSH probes failed during key propagation; the same instance succeeded
  on the next bounded observation. No new VM or installation was submitted.
- Reviewed tool files and the manifest were staged under the owned experiment
  directory with per-file SHA256 checks. A durable screening job was launched:
  `040f6cef78274a39b93294a08527a5f7`, worker PID **6371**, process start ticks
  **152390791**. `/proc` confirmed the process live at 04:25 UTC.
- The job waits for both successful `/health` and `VAST_ONSTART COMPLETE` before
  fetching packages, so it cannot contend with baseline dependency downloads.
  It then performs verified acquisition, small-wheel builds and initial offline
  pip/uv compiled-bytecode comparisons. These are screening microbenchmarks;
  full repeated delivery and native/application acceptance remain required.
- Metadata-only closure preflight passes with **zero missing/version-conflicting
  active requirements**: server 123 distributions / 185 edges; server+Kokoro
  174 / 290; full-stack 194 / 343. Optional extras captured include imageio's
  ffmpeg, misaki's en and requests' socks. This is not import or GPU evidence.

### 04:27 UTC — first new baseline timing

- Main/server pip finished at 04:27:05 UTC: **233s (3m53s)**, compared with the
  earlier profiled 266s. **No optimization has been applied to this step.** The
  difference is host/network/cache/instrumentation variability, not a claimed
  speedup from the new installer. Later trials must use a matched control.
- This source baseline has a genuinely new test-owned pip cache. Remote CDN and
  stock-image host caches are unknown. The container became available much
  sooner than in the previous rental, but no container-start change was made;
  that difference cannot be credited to dependency optimization.

### 04:30 UTC — source installation complete

| Boundary, same fresh source worker | Seconds |
|---|---:|
| Main/server packages | **233** |
| Incremental Kokoro | **30** |
| Avatar-prep + its import check | **24** |
| Full package phase | **287** |
| Model weights, unchanged/outside optimization scope | 89 |
| Native VP8 setup | 1 |
| CPU imports | 5 |
| Installer GPU self-test | 42 |
| Canonical setup, excluding checkout | 450 |

The package phase is now a 287s reference, not a 330s performance guarantee.
Source imports/GPU self-test passed, but server readiness and screening results
are still pending at 04:30:49 UTC. The 42s installer self-test includes TAESD
compile/timing and is not evidence that r5 engines are rebuilt at API startup.

A separate read-only resource sampler started after the main/Kokoro installation
(job `8ea859c6e28a4709873c2cb76a411907`, PID 8194). It samples CPU, memory,
network, disk and process counters without argv/environment data. It can cover
the upcoming comparisons, not retrospectively profile the completed baseline.
The first four-way acquisition is wheelhouse preparation for installer screening;
formal source/prefetch comparisons still require fresh caches and matched trials.

### 04:31 UTC — baseline API healthy; screening acquisition started

- Source startup completed at **04:31:09 UTC**; local `/health` was confirmed by
  the screening job at **04:31:10.658 UTC**. Canonical startup wrapper: **503s
  (8m23s)**, excluding the 67.062s checkout and provider/container startup.
- Prebuilt r5 bundle restore: **29s**; engine provisioning/build: **0s**;
  API start-to-health-and-verify: **19s**. This does not yet establish first usable
  REST video, direct connectivity, WebRTC/TURN or production registration.
- The four-way, hash-verified original-source acquisition began only after that
  readiness gate. Screening job PID/start identity remains live and unchanged.
- Collected complete source bootstrap/onstart/API logs, install stamp, GPU
  self-test and screening events into protected EC2 evidence (seven files,
  356,092 bytes at the first collection). No credential-bearing files or launch
  environments were copied. Transfer trials exclude bulk resource-log collection.

### 04:32 UTC — acquisition succeeds; controlled build catches a hidden installer

- Four-way original-source acquisition: **77.618s** inner / **77.627s** including
  manifest lookup. Verified **5,400,032,352 bytes**, **194 artifacts**, **zero cache
  hits**, no partial files left. Average verified payload rate is approximately
  **69.57 MB/s (556.6 Mbps)**, including the acquisition phase's local writes and
  integrity checks. This is one preparation sample, not three matched B2 trials.
- The first builder stopped after **16.554s**. Three small sdists built, but the
  TensorRT Python shim attempted its upstream nested `pip install` of the exact
  10.3.0 libraries/bindings. With indexes disabled and no network fallback, the
  build failed rather than downloading uncontrolled dependencies.
- Inspected the exact downloaded shim's `setup.py`. It supports
  **`NVIDIA_TENSORRT_DISABLE_INTERNAL_PIP=1`**, which declares those same pinned
  library/binding requirements normally instead of running nested pip. Scoped
  this switch to the builder; did not modify upstream code, versions or binaries.
- The first screening job has a terminal **failed result** (its generic transport
  wrapper completed normally; that is not a test pass). Its PID is absent and all
  logs are preserved. No live job was restarted merely because polling timed out.
- Staged corrected build tools in **`tools-v2`**, leaving v1 unchanged. Started an
  explicit recovery under **`screening-a2` / `wheelhouse-build-a2`**, reusing and
  re-verifying the downloaded inputs without repeating the 5.4 GB transfer.
  Recovery job: `b90d1029c928410a8997d4b821874d8b`, PID **9466**, same original
  06:21:18 UTC rental expiry. Linux pip/uv comparisons are still pending.

### 04:34–04:39 UTC — complete wheels; first offline install and harness correction

- Corrected build succeeded in **15.400s** overall, re-verifying the preserved
  sources and producing all four small wheels. Wheelhouse artifact identity:
  `725c6fe7564f9641fce16845a1346da99a75bf3cb54d993a273f34ceb6de7194`.
  All four generated locks contain only package pins/hashes/local lookup policy,
  not remote URL requirements. (Supported group locks are bootstrap, server,
  server+Kokoro and full; the directory also retains build input/output evidence.)
- Pinned uv acquisition/verification succeeded: **0.872s** inside the helper
  (0.967s including process startup); executable SHA256
  `db919fb9428cc8ad04377ba3c898e0ec5fbee83700da5b982b1c9539f1e1b34f`.
- First offline pip trial installed **exactly 194 expected distributions**, with
  **no package-consistency conflicts**. Separate local timings:

  | Boundary | Seconds |
  |---|---:|
  | Hash preflight | 3.829 |
  | Venv | 2.771 |
  | Bootstrap tooling | 2.219 |
  | Main offline pip | **93.168** |
  | Incremental Kokoro offline pip | **6.782** |
  | Incremental avatar-prep offline pip | **8.185** |
  | pip check | 0.817 |
  | Package-inventory/consistency readiness | **118.070** |

  Acquisition is excluded: this is **not a 118s fresh-instance result**.

- This trial's final smoke was **failed**, not accepted: the initial harness
  mistakenly imported `torch_tensorrt` while `CUDA_VISIBLE_DEVICES` was empty.
  Torch-TensorRT 2.5.0 initializes a device at import. The canonical installer
  deliberately defers that module from its CUDA-hidden smoke.
- Controlled diagnostic: the original working baseline fails the same hidden
  import (**4.307s**, `No CUDA GPUs are available`). The newly installed candidate
  passes with GPU visibility (**3.754s**), reports Torch **2.5.1+cu121 / CUDA 12.1 /
  Torch-TensorRT 2.5.0 / TensorRT 10.3.0**, and computes the expected CUDA result
  **140.0**. This isolates a harness mistake, not a demonstrated binary failure.
- Corrected the harness to keep CPU imports hidden and require a separate
  GPU-visible import/execution probe. No package was changed or check omitted.
  **81 safety tests pass**. Original failed receipts/logs remain unchanged.
- Started a fresh v3 comparison, **uv first then pip**, with new venv/cache paths;
  no re-download, rebuild or VM restart. Job
  `98e0877c913341c88261cb6577f2b591`, PID **9962**, writes `screening-a3` results.
  It retains the original expiry. Completed, corrected smoke and matched
  repetition evidence are still pending.

### 04:42 UTC — first corrected matched local installer pair passes

Both trials use the same 194-wheel manifest, Linux/Python/CUDA matrix, fresh venv
and installer cache, explicit compiled bytecode and **copy** link mode for uv.
The order was uv then pip; page-cache state is unknown. These are one sample per
variant, not repeated medians or full remote-delivery results.

| Local boundary | pip (s) | uv 0.13.1 (s) |
|---|---:|---:|
| Common venv creation | 2.770 | 2.771 |
| Common pinned bootstrap | 2.269 | 2.270 |
| Main/server installation | **91.526** | **19.701** |
| Incremental Kokoro installation | 6.980 | 1.418 |
| Incremental avatar-prep installation | 8.134 | 1.668 |
| Package inventory + consistency readiness, including hash preflight | **116.633** | **32.818** |
| First CPU-import process | 6.007 | 6.482 |
| Second CPU-import process | 5.549 | 5.896 |
| GPU-visible imports + CUDA execution | 3.846 | 3.796 |
| Complete screening validation | **132.036** | **48.993** |

Each has **zero installed-version differences**, a clean `pip check`, successful
CPU imports and the explicit CUDA/Torch-TensorRT probe. uv is **62.9% faster** at
this local complete-validation boundary (71.9% at the package-ready boundary).
This does not yet prove the full-delivery gain, application throughput or the
second-host requirement. The few-tenths import differences need repeated trials.

The acquisition receipt includes **38,646,285 repo-local bytes** (mmcv), not just
network payload: **5,361,386,067 bytes** came from remote artifacts. Its earlier
69.57 MB/s figure is **effective artifact acquisition throughput, not pure wire
bandwidth**. TLS/metadata traffic is not counted in that payload sum.

Sanitized interim [results](dependency_optimization_20261015/results.json) and
the [built-wheel manifest](dependency_optimization_20261015/wheelhouse-manifest-a2.json)
are now retained locally as well as in protected EC2 evidence.

### Application acceptance started (same existing rental)

An independent EC2 service is comparing the original source environment with
the compiled uv environment using the same prepared avatar, audio, recipe and
prebuilt plans. It uses internal SSH forwarding, a separate candidate port and
separate runtime files; it does not replace the source API/environment.

The source-environment case has passed: decoded H.264 **249 frames, 512x896,
25 FPS, 9.96s**, plus AAC **10s**. Output SHA256:
`646614c320c1af514d9b61998bb854f3658471871a910208a78504de9ba589e3`.
Candidate API/output validation is still running. Avatar download/warm state is
not identical between cases, so those warm times are not a speedup comparison.

### 04:49 UTC — candidate application output passes; quality is not bitwise-certified

- The compiled uv environment reached observed `/health` **20.458s** after EC2
  launch dispatch, including SSH transport/polling. This is not directly matched
  to the source wrapper's 19s start timer.
- Both cases generated/downloaded/decoded H.264 **249 frames at 512x896, 25 FPS,
  9.96s**, plus AAC **10s**, from the exact same audio SHA256
  `8628701b894babe55b8e04769363925528e50c1352d3edcebf1d31a44a65247e`.
  Candidate video: 1,324,893 bytes, SHA256
  `a18262a5eb4d0b7ade0d9ae0fafd2c222241de229f58ca84926a5feb2b09a50e`.
- Effective managed launch-environment values, recipe and expected backends are
  identical. Resolver reports have no errors; both static native VP8 checks pass.
  H.264-only WebRTC/native-VP8 fallback is reported unsupported by **both**
  environments, a pre-existing limitation, not fixed/validated by this REST test.
- Candidate loads the same prebuilt r5 UNet directory. TAESD log reports
  `built_now=False`, `probe=exact`, `hw_compat=ampere_plus`. The manifest's
  historical quality `gate=FAIL` label is unchanged; this experiment did not
  rebuild or re-approve those engines.
- MP4 hashes differ. Supplemental FFmpeg decoded comparisons: **SSIM 0.978965**,
  **PSNR 42.381554 dB**. These measurements do not establish bitwise/frame-quality
  equivalence, nor identify whether variation is inference, composition or
  encoding. Keep this visible for the matched-generation follow-up; do not call
  the output identical merely because dependency versions match.
- Source smoke total 40.249s and candidate 30.598s include different avatar-cache
  states, HTTP polling and EC2 decoding. **Do not claim these as a latency win.**
  Short matched throughput and first-usable-output measurements remain pending.
- Candidate API was stopped by its recorded PID/process-group/start identity;
  `/proc/11347` absence was independently verified. Original source API/rental
  stays running for further tests. Acceptance service is terminal with exit 0.

### Repetitions and bytecode matrix launched

- Started durable job `2e169bc6eeee44428c7583bcb358b913`, PID **12298**.
  It will bring each compiled installer to three samples with alternating order,
  then test uv and pip without compile-bytecode on fresh environments. It retains
  the compiled uv and no-bytecode uv environments for application follow-up.
- Disposable terminal pip trials can be retired to preserve disk headroom; only
  generated `venv` / `package-cache` children are removed. Their logs, receipts,
  exact wheels, locks and retirement receipts remain. The canonical source venv
  and selected compiled uv venv are explicitly excluded. uv's cache maintenance
  uses its supported `cache clean` command rather than editing cache internals.
- **83 stdlib/mock safety tests pass**, including explicit matrix dispatch so an
  unhandled command cannot fall through to rental cleanup. No production default
  or autoscaler/template configuration has changed.

At 04:55 UTC the matrix retired exactly the two prior terminal pip trial venvs
and their private package caches, reclaiming approximately **20.1 GB**. It also
cleaned the selected uv candidate's **installer cache only** through uv's own
command. Original/candidate installed environments, exact wheels and evidence
remain. The first repeated pip trial is running; PID/start identity is **12298 /
152574123**. Hard rental expiry remains **06:21:18 UTC**, not extended by recovery.

### WP6 read-only capability audit (no removals)

- `api_server.py` imports the Kokoro service wrapper, but that wrapper imports
  `kokoro.KPipeline` lazily. `/capabilities` advertises availability, and the
  Kokoro POST route explicitly returns 503 when `MUSETALK_DISABLE_LOCAL_TTS=1`.
  This supports testing an explicit external-TTS contract; it does not authorize
  silently removing a currently supported endpoint from the default worker.
- `scripts/api_avatar.py` intentionally imports mmpose preprocessing lazily for
  new-avatar preparation. Existing-avatar serving can be a separate capability
  trial; keep `/avatars/prepare` support in the equivalent full-stack baseline.
- `scripts/trt_runtime.py` still registers torch-tensorrt for serialized engine
  and fallback paths. ONNX-export paths also remain in the decoder/UNet modules.
  No GPU/export package has been removed merely because r5 plans are prebuilt.

## Recording rules / completion gates

For every trial retain variant, group, package/manifest hashes, source revision,
installer version, host, separate cache states, all timing boundaries, bytes,
cost estimate, failures, correctness and cleanup. Report individual trials,
median and range, never a fleet p95 inferred from one machine.

Local macOS work is limited to source/manifest processing and stdlib/mock tests.
It provides no CUDA installation, Linux performance or application acceptance
evidence. API/REST success over a tunnel is not WebRTC/TURN/production readiness.

New rentals must be uniquely labelled, <=$0.30/hour including quoted disk costs,
and <=$0.0015/GB Vast transfer in each direction. Use bounded leases and independent
EC2 expiry; preserve all pre-existing instances and verify owned-instance removal.
S3 charges must be recorded separately. Do not change IAM or provision a new AWS
builder to work around limited EC2 disk without an explicit cost/scope decision.

## Historical next execution gates (before the later completion receipts)

Retained as the contemporaneous queue; later 06:00–06:38 entries supersede its
live-service/pending descriptions. Do not restart those now-terminal jobs.

1. Reconcile the live **application-a3** JIT-isolated comparison on its existing
   EC2 service; preserve first-use fairness and investigate any remaining delay.
2. Pilot S3 transport is now fast enough that the historical slow full-GET result
   alone cannot reject today's route. Decide/measure the conditional full archive
   branch, including checksum/audit/extraction and canonical-path rollback; do not
   overlap it with live API timing. Budget/canonical-state checks still apply.
3. Exercise the canonical installer opt-in, actual warm startup preservation,
   remaining live auth/rollback cases and a clean delivery confirmation sample.
4. Recheck different-host availability under unchanged caps. Complete fresh-host
   confirmation and one-worker canary before an adoption recommendation; do not
   relabel same-host integration as either gate.
5. Preserve resources/final receipts, verify owned-rental removal, update the
   interim report/checklist and final worked/did-not-work conclusions. No protected
   pre-existing instance, production default, commit or push is in scope here.

## 05:01 UTC — small wheel publication is durable and private

- Preserved the four locally built wheels, their four source archives and four
  generated locks on the existing EC2: **495,010 bytes**. The immutable manifest
  adds **253,253 bytes**, SHA256
  `ef70f31aa4681df7658d9c8014a11db96c199502de6efe0708af2f32e7750d5c`.
- Published and independently downloaded/hash-checked all **13 objects** under
  `s3://lingua-musetalk-s3-storage/python-venv/wheelhouse/725c6fe7564f9641fce16845a1346da99a75bf3cb54d993a273f34ceb6de7194/`.
  Verified expected account/EC2 role, us-east-1, all four public-access blocks
  enabled and AES256 object encryption. Used conditional create, not overwrites.
- No IAM change, public access, new AWS compute or large CUDA-wheel upload was
  performed. Source archives retain build/licensing provenance alongside outputs.
- Added in-memory expiring read-ticket support to the downloader for these four
  private wheels only. It verifies bucket, owner, immutable object path and final
  trusted SHA256; tickets are neither argv nor persisted logs/manifests. Network
  payload and repo-local copy bytes now have separate receipt fields.
- **87 safety tests pass**. New downloader changes are reserved for a new staged
  tool version; the currently running local matrix keeps its immutable v3 tools.

## 05:04 UTC — local repetitions and bytecode screens complete

Three compiled-bytecode samples per installer, alternating order, same 194
wheels, fresh venv/cache, uv copy mode, Linux/Python/CUDA and validation checks:

| Boundary | pip median [range] (s) | uv median [range] (s) |
|---|---|---|
| Package readiness | 116.633 [116.348, 116.833] | 33.945 [32.818, 34.418] |
| Imports + CUDA complete | **132.036 [131.156, 132.185]** | **49.785 [48.993, 50.257]** |

The local validated median reduction is **62.3%**. All six corrected compiled
trials pass inventory, package consistency, CPU imports and CUDA execution.
One-host repetitions do not establish fleet p95 or fresh-download timing.

Bytecode screens (one sample each):

| Policy | Package ready (s) | First CPU imports (s) | Complete validation (s) |
|---|---:|---:|---:|
| uv without compile | 29.582 | 12.852 | **52.257** |
| pip without compile | 88.572 | 12.167 | **110.817** |

For uv the small install saving is outweighed by deferred imports; keep compiled
bytecode for full-delivery tests. pip without compile is faster than compiled
pip locally but remains substantially slower than compiled uv. Application
first-use for the retained no-compile uv env is still a separate open check.

All repeated disposable env/cache children were retired only after terminal
receipts; exact inputs/logs/receipts remain. The source, compiled uv and no-compile
uv environments remain available. The matrix job is terminal with a complete
test result; it was not restarted.

## 05:10–05:14 UTC — continuous delivery harness and signed-read repair

- Added a continuous timer covering manifest retrieval, package acquisition,
  method-specific tool delivery, installation and validation. Source controls
  keep normal three-index pip commands and avatar-prep order. They use fresh
  package caches and validate pip report hashes; the two historically hashless
  packages additionally compare installed files with trusted wheel RECORD hashes.
- The first delivery attempt failed after **0.184s**, before package acquisition:
  HTTP **400 / InvalidBucketOwnerAWSAccountID**. Its receipt is preserved. This
  was not missing upload/pull permission or slow bulk transfer.
- Read-only diagnosis showed botocore generated **SigV2 presigned URLs** in
  us-east-1 despite displaying s3v4 in default config. Owner binding became a
  query field while the client also supplied the owner header. Explicitly pinning
  `Config(signature_version="s3v4")` fixes the signing format.
- Confirmed the corrected private manifest GET on EC2: **HTTP 200**, 253,253
  bytes, trusted SHA256 matched. A regression guard rejects legacy presigning
  before dispatch. **92 tests pass**; no IAM or bucket-policy changes were made.
- Started a new delivery attempt in **tools-v5 / delivery-matrix-a2**, preserving
  the failed v4 attempt. Job **`af9ffea2117b46e7be8b253c70c610aa`**, PID **15101**,
  process start ticks **152686457**. It runs matched source controls, serial locked
  pip, two-/four-way uv acquisition and repeated four-way full-delivery trials.
  Original rental expiry remains **06:21:18 UTC**. Signed reads stay in memory;
  only sanitized event/receipt data is retained.

## 05:26–05:30 UTC — first continuous comparisons pass

- The matched original-source control passed exact package inventory, source
  artifact verification, package consistency, CPU imports and CUDA execution:
  **277.776s package-ready / 293.050s validated-ready**. Bootstrap acquisition is
  included; it does not redundantly prefetch the full wheelhouse for this control.
- The two-way-prefetch + compiled-uv case passed the same package/native checks:
  **94.090s acquisition / 130.635s package-ready / 147.582s validated-ready**.
  This is one observed continuous full-stack dependency result, **49.6% faster**
  than the first source control, not a median or full-instance startup claim.
- Four-way acquisition and the remaining matched source/uv repetitions are live
  in the original delivery job. No create/restart was issued on a timeout.
- Added a bounded, read-only S3 transport pilot for the existing immutable venv
  archive: three 128-MiB prefix samples each with one and four connections,
  alternating order. It checks VersionId, ETag, Content-Range and byte counts;
  equal prefix hashes are not a substitute for the full archive's trusted SHA.
  The pilot waits for the delivery matrix to finish, so it cannot compete with
  those transfer measurements. No full archive download or restore is implied.
- **98 stdlib/mock tests pass**, including range truncation, changed object
  identity, wrong ranges/status, unapproved endpoints, macOS refusal and explicit
  pilot dispatch that cannot fall through to rental cleanup. macOS still runs
  no pip/CUDA installation tests. Rental hard expiry is unchanged.

## 05:32–05:35 UTC — opt-in integration prepared; four-way is not automatically better

- First four-way compiled-uv delivery passed: **100.839s acquisition / 136.806s
  package-ready / 153.327s validated-ready**. It is slower than the two-way
  147.582s sample. Keep the remaining matched repeats; do not increase to eight
  merely because more connections are possible.
- Independently read source and candidate `pip-check.log` receipts: all queried
  checks exit **0**, `No broken requirements found`. The comparison harness's
  diagnostic-only check did not conceal a conflict in these observed cases.
- Added `install_locked_dependencies.py` and an **experimental opt-in** in the
  canonical installer. Default remains `source-pip`. Locked modes preserve exact
  cu121 inputs, acquire the pinned manifest/wheels/tool, compile bytecode, verify
  full inventory and require package consistency. They own venv/bootstrap work
  so it is not fetched twice. Unsupported group/matrix/clean combinations fail.
- Reuse checks independently verify the cached manifest digest, dependency-input
  hashes, complete installed inventory and existing repo state-checker verdict.
  Unrelated application code is not part of dependency identity. New installs
  cannot adopt/overwrite foreign venvs; repairs require an explicit flag, no
  active process, and missing packages only. Version/ABI changes require a new
  path, preserving the old environment for rollback. No implicit source fallback.
- Prepared a new immutable Linux followup: real loopback HTTP auth/truncation/
  disconnect/retry faults, corrupt-cache refusal, fresh opt-in installation,
  three no-network reuse checks, lock contention, stale stamps and a recoverable
  missing-package repair. It waits for both timed delivery and the S3 pilot;
  it is **same-host integration**, not a fresh-host canary.
- Initial shell-fixture tests correctly failed when their isolated function
  harness lacked the new mode variable. Updated that fixture to declare the
  unchanged source default, and added a mocked locked-mode dispatch test.
  **107 stdlib/mock tests now pass**, plus Bash syntax validation. No Mac package
  installation, production template/default switch, commit or push occurred.
- Read-only US/Canada offer search returned no acceptable different host under
  the unchanged hourly/transfer/hardware caps. B10 remains open, not waived.

## 05:36–05:40 UTC — repetitions continue; application comparison queued

- Second source control: **278.844s package-ready / 294.191s validated-ready**.
  Second four-way uv: **80.117s acquisition / 116.056s package-ready / 132.176s
  validated-ready**. The network variation is material; single samples cannot
  establish two versus four connections as the winner. Third matched pair is live.
- The isolated followup is dispatched once: job
  `48c3837136324976a808dcfa8c831103`, PID **17942**, immutable **tools-v6**.
  Its dependency cache and venv are outside the source checkout. It waits behind
  the delivery matrix and S3 transport pilot; no competing package traffic starts
  until they are terminal. The original source/candidate environments are retained.
- Added a separate EC2 application matrix, using a fresh isolated API process
  for source pip, compiled uv, no-compile uv, and the opt-in installed environment.
  Each case warms the same disk-resident avatar, then generates four identical
  audio requests (first request plus three warm repetitions), downloads/decodes
  both tracks, and records API readiness and request-to-output timing. It captures
  the server's rendering FPS separately from REST end-to-end frames/second.
  Neither is a new certification of historical aggregate 400+ FPS performance.
- Within-source/within-uv decoded SSIM will be compared with cross-environment
  SSIM to investigate the earlier output variation. Same versions are not treated
  as proof of identical frames. Candidate APIs are sequential, loopback-only and
  stopped by recorded process identity; the original source API remains untouched.
- Application service is bounded to 30 minutes and the existing rental's hard
  expiry remains **06:21:18 UTC**. **108 safety/mock tests pass**. This is still
  same-host validation, not a second-host or production-routing acceptance claim.

## 05:43 UTC — continuous dependency matrix complete

Authoritative job is terminal: `af9ffea2117b46e7be8b253c70c610aa`, nested outcome
`complete`, original PID absent. All eight delivery cases passed. No restart.

| Method | Individual validated-ready samples (s) | Median [range] (s) |
|---|---|---|
| Source pip, fresh caches | 293.050, 294.191, 297.623 | **294.191 [293.050, 297.623]** |
| Locked pip, serial acquisition | 265.828 | 265.828 (screen only) |
| Compiled uv, two-way acquisition | 147.582 | 147.582 (screen only) |
| Compiled uv, four-way acquisition | 153.327, 132.176, 133.953 | **133.953 [132.176, 153.327]** |

Four-way package-ready median is **117.768s**; complete import/CUDA validation is
**133.953s**. The matched source-pip validated median reduction is **54.47%**.
This is acquisition-through-dependency-validation of the same full-stack 194
packages, **not total Vast create-to-ready time**. Every case has fresh venv,
installer cache and acquisition directory; OS/CDN cache state remains unknown.

An evidence collection of **12,567,400 bytes** finished at **05:42:40.816 UTC**,
during the third uv trial's post-acquisition installation window (its acquisition
ended about 05:42:25.53). Resource/evidence sampling was not entirely absent from
that sample. Preserve this observer-interference caveat; repeat a clean candidate
sample if needed before adoption rather than presenting pristine fleet evidence.
No other package installer or API workload competed with the timed matrix.

The S3 pilot and Linux integration jobs now proceed on their original handles.
Source default, template, inference settings and rental expiry are unchanged.

## 05:46–05:52 UTC — Linux integration and controlled application matrix complete

- Linux followup is terminal, nested outcome **passed**, PID **17942 absent**.
  Fresh opt-in dependency install **130.166s** (inventory/pip consistency,
  not a full import/GPU/API readiness boundary). Reuse checks **0.440 / 0.399 /
  0.435s**; missing-docopt repair **4.310s**, **0 remote package bytes**.
  Corruption/auth/truncation/disconnect/transient-retry loopback fixtures,
  concurrent writer, stale stamp and missing-package checks all passed. Moved
  docopt files were retained in the experiment's backup, not irrecoverably deleted.
- S3 pilot completed all six version-bound 128-MiB samples: one-connection median
  **2.479s**, four-connection **2.528s**, total **805,306,368 bytes**. Prefix hashes
  match. Parallelism did not help this pilot. No full-archive restore is proven.
- Application service is terminal, exit **0**, MainPID **0**; all four sequential
  APIs stopped and their loopback port closure verified. Original source API
  was not replaced. Every one of **16 generated MP4s** has the same SHA256:
  `646614c320c1af514d9b61998bb854f3658471871a910208a78504de9ba589e3`.
  Both tracks decode (H.264 249 frames, AAC), and all five within-/cross-env SSIM
  comparisons are **1.0**. Earlier nonidentical output remains recorded; it was
  not reproduced in this controlled matrix and has no established root cause.

| Environment | API healthy (s) | First request complete (s) | Three warm requests (s) |
|---|---:|---:|---|
| Source pip, new API process | 19.292 | 13.323 | 12.017, 11.927, 12.008 |
| Compiled uv | 18.695 | 13.316 | 12.130, 12.146, 12.140 |
| No-compile uv | 19.374 | **26.296** | 12.225, 11.829, 12.210 |
| New opt-in uv install | 19.527 | **25.827** | 11.879, 12.104, 12.251 |

Warm request medians are within 2% of source, but **first-use acceptance is still
open**: newly audio-unused envs take about 13s longer on their first request.
Earlier source/compiled-uv envs have prior audio/JIT/cache history. Do not call
these four cases equally cold merely because their API processes were fresh.
Investigate with independently isolated first-use caches before attributing the
delay to an installer or declaring no readiness regression. Bytecode deferral is
not the only possible cause: the newly compiled opt-in env also shows the delay.

Server render logs report 250 frames at about 40–41 FPS including blending/PNG IO;
the MP4 decodes to 249 frames. These matched paths differ from the historical
multi-stream aggregate 400+ FPS harness. Same prebuilt TAESD reports
`built_now=False`, `probe=exact`, `hw_compat=ampere_plus`; its historical quality
`gate=FAIL` label was not changed or reapproved.

## 05:55 UTC — isolated first-use comparison dispatched

The a2 application service is terminal; it was not restarted or overwritten.
A new **application-a3** service runs source, compiled uv and the opt-in env with
a distinct initially empty `NUMBA_CACHE_DIR` for each case. Equal cache-debug
logging records whether compiled cache files are actually created. It retains
the same audio/avatar/recipe/engines and does not delete package-local caches.
This tests the JIT-cache-history hypothesis, not a proposed performance-default
change. [Numba's environment reference](https://numba.readthedocs.io/en/stable/reference/envvars.html)
and [cache notes](https://numba.readthedocs.io/en/stable/developer/caching.html)
were checked before selecting this control.

The service is bounded to **900s** and the rental's **06:21:18 UTC** hard expiry
is unchanged. **110 stdlib/mock tests pass**, including a syntax/empty-cache
dispatch fixture; no CUDA/pip tests were performed on macOS. Final a2 and Linux
followup receipts are preserved/exported alongside all eight delivery cases.

## 06:00 UTC — cold-JIT control closes the first-use comparison on this host

The **application-a3** service completed successfully. With a newly empty Numba
cache for every case, first requests were **25.258s source / 25.191s compiled uv /
25.171s opt-in uv**. Each case created **46 cache files** and logged **36 compiled
cache saves**; API health was **18.157 / 18.304 / 18.650s**. Three warm requests
per case stayed around 11.6–11.9s. All 12 videos remain byte-identical and all four
decoded SSIM comparisons are **1.0**.

This controlled intervention supports JIT-cache history as the explanation for
the a2 first-request gap: no material source-versus-uv first-use regression remains
under this equalized cache policy. It does **not** make cold first calls fast;
the roughly 13s cold-vs-warm audio tail still exists for all three methods. No
production cache setting or inference parameter was changed. Prior a2/initial
quality anomalies and their limitations remain in the record.

Prepared the conditional full S3 branch after this live API matrix completed:
one full GET versus four-way 32-MiB ranges of the same immutable 4.51-GB archive,
then trusted full SHA256, archive path/link audit, extraction, canonical-path
publish with original venv retained, matched inventory/pip/import/CUDA checks,
extra canonical validation and API health. The original venv/API is restored
afterward; no TensorRT engine build is allowed. Cache maintenance uses uv's own
command and removes only the owned **installer cache**, not installed envs.

The new full-S3 job uses immutable **tools-v7**, the existing 100-GB rental and
its unchanged expiry. Code/mock checks: **112 pass**. Two full archive downloads
are within the existing test tranche; final billing/free AWS allowance is not
assumed. Both private signed requests remain memory-only.

## 06:05–06:07 UTC — full S3 result and recoverable rollback verified

Full-S3 job `7ab52b79ebd149138eebbc34484b9cef` passed the complete archive path,
then restored the original canonical venv and healthy source API. Both archive
downloads match trusted SHA256
`d8a1fada0b0f59dc7c7df1c289e9101f2b215313f08895fd9ee92acb669195d6`.

| Boundary | Seconds / result |
|---|---|
| One full GET, 4,509,210,997 bytes | **77.365s, 58.285 MB/s** |
| Single-GET checksum | 3.139s |
| Four-way range retrieval, 135 requests, same full bytes | **64.831s, 69.553 MB/s** |
| Ranged checksum | 3.075s |
| Archive member/link audit | 27.358s |
| Extraction | 30.676s |
| Continuous ranged restore → package inventory + pip check | **135.239s** |
| Continuous ranged restore → matched imports/CUDA validation | **150.322s** |
| Plus additional canonical native validation | 163.197s |
| Continuous ranged restore → API health | **182.682s** |
| Original venv restored / original API healthy | **true / true** |

Four-way retrieval improved this **full object** transfer by about 16%; the prefix
pilot did not predict that improvement. But **150.322s matched validation is
slower than the wheel/uv median 133.953s**. The archive audit and extraction erase
the transfer advantage. One screen does not establish a population distribution;
the S3 route is not promoted for repetition/defaults because it lost this gate
and adds transfer cost. Do not compare S3's 182.682s API boundary with uv's 134s
dependency-only boundary as though they were the same metric.

Only the owned opt-in **installer cache** was cleaned with uv, reclaiming about
**9.78 GB**. Installed environments remain, the original venv was restored, and
the restored test venv/archives are retained with receipts until owned-rental
cleanup. The remote immutable archive remains available for recovery.

The canonical-wrapper/checkout-identity/expired-ticket check is next. Its first
control-plane dispatch was refused **before any job claim** because the normal
read-ticket helper intentionally requires >=300s expiry. Added a separate
one-second fixture restricted to the already reviewed manifest, preserving that
normal guard; no IAM or permission scope was changed. All source/mock checks pass.

## 06:07–06:14 UTC — canonical wrapper, expired auth and clean B6 confirmation

- Canonical installer opt-in passed, including normal native/GPU self-test:
  **20.552s** with dependencies reused. Its read-only restart check took
  **0.316s**. This is not a fresh installation: apt/weights were explicitly
  skipped and existing package/model/engine files were retained.
- On Linux, a separate source-only checkout fixture reused the accepted venv in
  **0.445s** with no private ticket read/network request. Changing a dependency
  input fingerprint was rejected before network access. Real source files were
  not edited to construct that fixture.
- Actual one-second S3 GET expiry returned **HTTP 403 / AccessDenied**, with
  AWS confirming the request expired. The package fetcher failed without fallback,
  publishing an artifact or leaving a partial file. Normal ticket lifetime guards
  and IAM permissions remain unchanged.
- Clean fourth full-delivery confirmation passed: **79.389s acquisition,
  115.336s package-ready, 131.177s validated-ready**, **0 cache hits**, 5.361 GB
  remote payload. This did not overlap bulk evidence collection or another
  package/API workload. It replaces the observer-affected third sample for the
  clean sensitivity comparison; all four raw results remain retained.
- To provide bounded scratch space, retired exactly two **test-downloaded archive
  files** after verifying their trusted hashes, plus the completed disposable
  no-compile trial venv. Approximately **18.90 GB** reclaimed. The immutable S3
  archive, exact wheels/locks and all receipts remain for recovery. The original,
  compiled uv and opt-in installed environments were preserved.
- Dispatched the real existing `vast_onstart.sh` using its supported alternate
  installer hook, accepted opt-in venv, port **8002** and isolated runtime/logs.
  It runs after the clean transfer test, retains normal import checks, verifies
  existing r5 artifacts and health, and stops only its candidate API afterward.
  This tests **in-container warm onstart**, not Vast provisioning, Git checkout,
  fresh model delivery, TURN or production registration. No template/default edit.
- Seven-day quota check after the user's new guard: **12% used** (seven-day
  window); below the **75% used** stop threshold. Continue checking before further
  goal work. If reached, preserve results, terminate owned test resources and stop.

## Final consolidated comparison

An [interim consolidated report](DEPENDENCY_INSTALL_OPTIMIZATION_REPORT_20261016.md)
now combines completed timing screens, repeated full-delivery results, negative
findings, full S3 restore, equalized first-use results, canonical/warm integration
and cleanup. Different-host/fresh-canary confirmation remains required.

## 06:16–06:29 UTC — teardown, evidence preservation and fresh-boot hardening

- Real warm `vast_onstart.sh` completed in **24.400s**, with the accepted env,
  existing models/engines, 1s bundle verification and 18s server health/backend
  verification. Candidate stop took 7.282s; original API remained healthy.
- Destroyed only owned rental **56199065** at **06:16:29.675 UTC**. Provider
  absence was verified; expiry timer stopped. Rechecked provider absence at
  06:29 UTC. No old worker job is live or eligible to be restarted.
- Preserved 339 small evidence files plus **3,138 two-second resource samples**.
  Compressed capture: 7,434,789 bytes, SHA256
  `53aed6a0f085d7663c09f6eaaa119b560030a8c5dee274a582acdb5204b073fa`.
  Raw capture: 50,204,169 bytes, SHA256
  `41dfe4e94b34a1799b465788d8a5f796f50ad83dd0c389528b70cd833cf1238e`.
  Selected-process aggregate sampled RSS reached 13.18 GB; the 61.22-GB cgroup
  counter is not isolated worker RSS. Sampling missed early baseline work and
  instantaneous peaks; no continuous free-disk peak was measured.
- Corrected the fresh-boot path: Vast may supply `--clean` when no venv exists.
  Locked mode now allows this absent-path case, but refuses existing dirs or
  dangling symlinks. It creates the venv parent before handing off bootstrap.
  Read-only locked checks now verify trusted manifest/input hashes, accepted
  owner/group and the entire inventory, not just a small pin subset.
- Added a missing-config refusal test. It caught Bash's empty-array/nounset
  behavior in the source-mode branch; removed the unnecessary array. **118
  source/mock tests passed**. These post-rental changes need fresh Linux proof.
- Reporting skill used to reconcile the requested repository Markdown against
  the final receipts; no report app/site or external publication was substituted.
  Seven-day quota **14% used**, below the **75% used** stop guard.

## 06:29 UTC — different host now available; bounded fresh canary prepared

A new read-only offer search returned a **verified Oregon** worker, offer
**52272582**, machine **4749**, host **24953**: 24 effective cores, 32,215 MB RAM,
700.3 Mbps advertised download, 100 GB requested disk, adjusted **$0.274444/hour**,
both transfer directions **$0.001333333/GB**. This is a genuinely different host
and satisfies the existing caps; it must be rechecked immediately before create.

Prepared `dependency_fresh_canary_ec2.py` with a **new ownership namespace**, one
create intent, **one-hour independent hard expiry**, $2 tranche and the same stock
template/source SHA. It privately publishes a small hash-bound final-source
overlay and supplies read-only dependency tickets through an inherited descriptor.
No manual SSH staging delay is hidden in the boot timer; the launch performs its
own source delivery, fresh dependency install, models, prebuilt r5 restore and
health checks. Existing production/template/autoscaler defaults remain untouched.

The allowlisted source overlay is **189,864 bytes**, SHA256
`4163f1ec30760c24d2a94857d20c35c99f84e9782eb26e179de482129df56bda`.
Three new corruption/path/source-symlink fixtures pass; **121 source/mock tests
pass** overall. The original checkout files are backed up on the owned canary.
No canary instance has been submitted at this preparation checkpoint.

## 06:36–06:38 UTC — preflight refusal and a single fresh canary create

- The first source-staging command could not create a directory under EC2's
  root-owned state parent. Created only the new tools directory with explicit
  ownership; did not change the parent or production permissions. All staged
  source copies completed before publication.
- Published the 189,864-byte source overlay privately, conditional PUT plus
  independent GET/SHA verification and AES256; no IAM change or large upload.
- Oregon disappeared before preflight. The command returned **before creating a
  ledger or issuing a provider create**. This was not an ambiguous create retry.
- The next read-only search found verified offer **48396264**, machine **136634**,
  host **399360**, location reported only as **United States, US**. 10 effective
  cores, 32,224 MB RAM, 938.7 Mbps advertised download, 100 GB requested disk,
  adjusted **$0.248107639/hour**, each transfer direction **$0.001302083/GB**.
  Rebound the explicit reviewed canary target and rechecked every cap.
- Created **56211950 exactly once at 06:38:00.519 UTC**, new label
  `musetalk-dependency-canary-20261016-a1`, hard expiry **07:38:00 UTC**. At creation,
  protected pre-existing IDs were **51074906 and 56208380**; neither is this test.
  No production template/default or AWS infrastructure changes.
- A resource-file audit initially asserted total JSONL lines equal sample count.
  Both full hashes matched; the extra line is the observer header, not another
  sample. Verified **1 header + 3,138 resource samples**, retaining the original
  hashes and sample definition. No capture corruption was found.

## 06:40–06:43 UTC — fresh canary live; overlap screen prepared

- Provider transitioned the canary to running. Initial SSH probe failed during
  key propagation; a bounded retry of the same instance succeeded through both
  the provider proxy and the mapped direct SSH port. No new instance was created.
- Stock bootstrap began **06:40:14.577 UTC**, about **134s after create**. Full
  Git checkout is still live: Git/index processes and increasing transfer/write
  counters were observed. No dependency timing or success is claimed yet.
- Prepared the remaining limited-overlap screen: exact wheel acquisition plus
  the unchanged 1.0-GB r5 bundle download, serial versus concurrent into fresh
  isolated destinations. It must wait for canonical startup and application
  acceptance, so it cannot contaminate those timers. Both paths verify the same
  trusted hashes. This screens acquisition overlap, not a full boot optimization.
- Two new wrong-target/truncated-engine fixtures pass. **123 source/mock tests
  pass**; no macOS installation/CUDA test. The canary's installer overlay itself
  remains exactly the published source hash, not this separate diagnostic tool.

## 06:48 UTC — fresh checkout complete; bounded acceptance service started

The full checkout finished **06:47:56.378 UTC**, **461.801s (7m42s)** after clone
start. This is slower than the first host's 67.062s checkout despite a higher
advertised link; do not attribute it to pip/uv or equate advertised Mbps with
GitHub throughput. Stock create-to-bootstrap was separately about **134s**.

The hash-bound final installer overlay loaded successfully and phase-0 preflight
passed. Normal apt/setup is now live. No early package/API result is claimed.
Started exactly one EC2 acceptance service, unit
`musetalk-dependency-canary-20261016-a1-accept`, **900s RuntimeMaxSec**. It waits for
healthy API, requires the actual onstart COMPLETE marker, checks the reviewed
overlay/full dependency contract, and generates/decodes two same-input videos.
This remains internal REST, not production registration or direct TURN proof.

## 06:57 UTC — second-host dependencies accepted; slow CDN dominates

Canonical full-stack locked installation finished **06:57:00 UTC**, shell phase
**512s**, helper receipt **511.097s**. All 194 versions and pip consistency passed.
Acquisition was **431.275s** with no retry on the slowest artifact:

| Artifact / step | Seconds |
|---|---:|
| TensorRT cu12 libraries, 2,037,510,258 bytes, pypi.nvidia.com | **431.270** |
| cuDNN, download.pytorch.org | 156.117 |
| PyTorch, download-r2.pytorch.org | 143.378 |
| cuBLAS, pypi.nvidia.com | 87.302 |
| Fresh venv / bootstrap | 4.781 / 3.324 |
| Offline compiled-uv server / incremental Kokoro / avatar-prep | **55.213 / 2.521 / 3.124** |
| pip consistency | 1.268 |

Download spans overlap and are **not additive**. The TensorRT library averaged
about **4.72 MB/s**, making it the acquisition critical path. Advertised host
download was 938.7 Mbps; that is not a measured rate from NVIDIA. This demonstrates
network/path variance and contradicts any universal 132s readiness promise. No
matched source-pip control has been run on this second host, so do not calculate
a cross-host installer speedup from the earlier 294s median.

Normal weights/import/native/API startup is still proceeding. Based on this new
evidence, reopened the conditional S3 branch for **second-host** confirmation:
the first host's S3 loss does not prove it loses on a CDN-limited host. Prepared
the existing full-restore tool to accept only this additional owned root/cache
and to omit repeating the already-screened single GET. It must wait for app
acceptance, preserve the canonical env and validate/rollback exactly as before.

## 07:02 UTC — fresh canonical boot complete; actual cold start remains long

Fresh canary reached `VAST_ONSTART COMPLETE` at **07:02:17 UTC** and `/health`
was independently observed HTTP 200 / `ok=true`. Create-to-complete is about
**24m17s**, not five minutes. The stock bootstrap delay (~134s), full clone
(461.801s) and onstart (858s) are separate, sequential boundaries; minor overlay
and shell timestamp rounding account for the remaining few seconds.

| Completed onstart stage | Seconds |
|---|---:|
| apt | 29 |
| Full dependency acquisition/install/inventory | 512 shell / 511.097 helper |
| All model/voice weights | 119 |
| Native VP8 | 4 |
| CPU imports | 14 |
| GPU self-test | 67 |
| Canonical setup total | 746 |
| Post-setup validation | ~8 |
| TURN bootstrap (not external call proof) | 2 |
| Same prebuilt r5 engine bundle restore | 68 |
| Engine provisioning | **0** |
| Server start → health/backend verification | 34 |
| Total onstart | **858** |

The final hardened canonical checks passed on this genuinely fresh different
host, without changing package versions or building r5 engines at server start.
Fresh output acceptance is not yet established by health alone.

The first acceptance service is authoritatively **failed / MainPID=0 / status=2**,
receipt detail `Canary API health deadline`: its ten-minute wait expired while
the original startup continued. Preserved it; did **not** restart installation or
create another instance. After health/COMPLETE, started one separately claimed
`accept-a2` output-check service, RuntimeMaxSec **600s**, with a guard allowing
only this known terminal deadline retry. The original canary expiry stays 07:38.
Quota checked **18% used**, below 75%; no reset credit consumed.

## 07:05–07:08 UTC — fresh output acceptance passed; second-host S3 live

- `accept-a2` is authoritatively **inactive / MainPID=0 / exit=0**, receipt
  **passed**. The final reviewed source overlay/full-inventory check passes.
  Two outputs both decode **249 H.264 frames, 512×896, 25 FPS, 9.96s** and AAC
  audio **10.00s**. Both MP4 hashes are
  `2715c44caab05a6aee931e52d6a0a00b02fb9a2bd821f5c61dbb2f56203ad642`.
  Their complete avatar-warm + generation-poll + download/decode paths were
  **72.271s / 20.387s**; these are not request-only or first-frame timings.
- This different-host hash is **not** the prior host's `646614…` output. Preserve
  the difference; no cross-host identity or installer-caused regression is proven.
  The S3-env test adds same-input output **on this same host** before rollback,
  allowing a properly matched comparison with the new locked environment.
- Started exactly one second-host S3 confirmation: job
  **ce2fe97d762747deb2f662149a5277ac**, worker **PID 11206**, start ticks
  **105466569** confirmed live. Same immutable archive, version/ETag-bound four-way
  full GET, trusted SHA, archive audit/extraction, canonical execution/native/API/
  decoded REST video, then original-env/API restoration. The already-screened
  single GET is not repeated here. One full 4.51-GB transfer is budgeted; no new
  environment archive or large wheel mirror is uploaded.
- Supported uv cache clean reclaimed **9.78 GB**, leaving installed copy-mode
  environments intact. Full range retrieval started **07:07:24.315 UTC**.
- The tool accepts only the additional owned canary root and exact reviewed cache.
  Archive native validation uses baseline **read-only** pins/import checks plus
  independently verified full inventory/archive SHA; it does not manufacture a
  locked-installer ownership marker or run pip to make the old archive pass.
- **124 source/mock tests pass** after the new root/path guard. S3 job is not yet
  complete. Original one-hour lease and **07:38 UTC** hard expiry remain.
- Saved [sanitized canary receipts](dependency_optimization_20261015/canary-results.json)
  separately from the first-host data; its interim export is `goal_complete=false`.

## 07:19–07:38 UTC — second-host S3 loss, failed restart and verified cleanup

The full second-host S3 job is terminal, **failed**, not cancelled:

| Boundary | Seconds |
|---|---:|
| Four-way full 4,509,210,997-byte GET, 135 requests | 501.489 |
| Trusted full SHA256 verification | 28.597 |
| Archive audit / extraction | 70.486 / 74.232 |
| Restore through inventory and pip consistency | **685.786** |
| Matched imports/basic CUDA | 713.425 |
| Additional read-only native validation | 732.274 |

Payload rate was **8.992 decimal MB/s**. Package readiness was **34.2% slower**
than this host's 511.097s locked-wheel path. S3 is not promoted. The same-host
API/output gate did **not** pass: local TURN startup failed before the restored
environment produced video. The original venv was restored, but original API
health was false. A performance-abort attempt was refused by the existing
backup guard **before any signal was sent**; the original job finished normally.

Two bounded recovery receipts remain failures. The first failed to launch;
the second preserved the three public Vast IP/port mapping fields and stopped
with `API PID file remains`. A read-only reproduction found the existing shell
PID1-environment reader returns **141/SIGPIPE** for `VAST_TCP_PORT_1455` with this
18,336-byte launch environment. The test harness now preserves only the reviewed
public mapping fields instead of forcing this fallback. This does not establish
a production TURN fix, and it did not recover the canary API before teardown.
Final independent `/health` observation returned no healthy response.

The copy/hardlink screen also failed **before installing any case**:
`Explicit experiment root required`, `cases=[]`. Corrected its benchmark-root
selection without weakening the benchmark's dedicated-root guard. The corrected
screen and full-payload overlap remain unrun; insufficient remaining lease was
not a reason to relax their reserves or extend this rental.

Preserved 17 additional receipt/log files on EC2 before expiry. Only owned
**56211950** was destroyed at **07:37:29 UTC**; provider absence was verified and
its expiry timer stopped. The temporary instance disk is gone; immutable
artifacts and protected evidence remain. This teardown did not alter protected
instances. Sanitized canary export now includes failed S3/recovery/link receipts
and cleanup: 110,874 bytes, SHA256
`c04ceefc05f55411a2b6e1bf9fddf7f0eb720e5c8c271caafecc6fb92fff0682`.

Source/mock suite is **126 passing tests**. Seven-day quota is **19% used**,
below the user's 75% stop threshold; no reset credit consumed. Goal remains
incomplete pending bounded overlap/link tests and final coverage audit. A
subsequent lease, if used, needs its own ledger and independent expiry, not an
extension or replay of this canary.

## 07:42–07:45 UTC — final-screen harness and separate bounded lease

Prepared a separate acquisition/link-only worker controller and source/mock
safety tests (**130 passing** overall). Unlike the fresh app canary, this worker
does not clone the full repo, download models, run the app or register production
callbacks. It keeps the stock PyTorch template, same canonical apt prerequisites,
exact 194-wheel contract and original r5 archive. Common fixture/tool setup is
reported separately, never counted as a download-overlap saving.

The same verified US offer is again available at **$0.248108/hour adjusted** with
100 GB disk and **$0.001302083333/GB** both directions. A new one-hour/$2-tranche
create is submitted under `musetalk-dependency-final-screen-20261016-a1`, with
its own ownership ledger, one-create intent, protected inventory and independent
expiry. No prior rental is extended/replayed. Planned traffic is two complete
wheel deliveries plus two unchanged 1-GB engine downloads; no new large S3 upload.

Screens use fresh wheel destinations for serial versus concurrent acquisition.
Then copy/hardlink use those verified local wheels and independent venv/caches,
checking full inventory/imports/CUDA, actual link counts, supported cache clean
and CUDA/hash survival after cleaning. Only a >=20% promising initial link gain
earns additional matched repetitions. This is not a full startup or application
benchmark and will not change the copy/default install policy.

The original host failed the create-time offer preflight; **no ownership ledger
or create intent was written and no VM was created**. A broader read-only search
found a verified Oregon alternative (machine 4749 / host 24953), 24 effective
cores, 32,215 MB RAM, 700.3 Mbps advertised down and 6,475 MB/s advertised disk.
Its 100-GB adjusted quote is **$0.274444/hour**, both transfer directions
**$0.001333333333/GB**. Updated only this final-screen controller's reviewed
selection. A fresh preflight, single-create ledger and independent one-hour
expiry still apply; failed preflights do not authorize ambiguous-create retries.

The Oregon quote likewise disappeared at final preflight; no create intent was
written. The remaining in-cap verified Virginia offer is **$0.321441/hour
adjusted**, so it was rejected despite its sub-$0.30 raw GPU quote. The original
Pennsylvania screening host (machine 92165 / host 443275) remains available at
**$0.187875/hour adjusted**, 8 effective cores, 32,222 MB RAM, 527.7 Mbps down,
**$0.00000390625/GB** both directions. It is **unverified**, as disclosed for
the original campaign, and is the sole reviewed fallback for these non-production
screens. Updated selection to that same previously tested host; no cap increase,
existing instance reuse, callback exposure or arbitrary-host fallback.

Provider accepted exactly one create for final-screen **56218707** at
**07:47:08 UTC**, Pennsylvania source host, adjusted **$0.187875/hour**.
Independent hard expiry is **08:47:07 UTC**. Protected create-time inventory is
`51074906`; all previous task-owned workers are already gone. No app install,
full checkout or production registration is requested on this new worker.

Preparation passed on stock Linux x86_64, CPython **3.10.12**, glibc **2.35**,
107.31 GB free at preflight: canonical apt update/prerequisites **2.921/14.400s**;
the exact 38.65-MB repo-local mmcv wheel **1.251s**, SHA matched; pinned uv delivery
and verification **0.877s**. Common fixture total **19.455s** is excluded from
both acquisition cases and disclosed, not credited as a startup optimization.
No full repo clone or model/application setup occurred.

Acquisition job **a52d996950874aa69ae91e9ebe89c107**, PID **6083** / start ticks
**153620417**, is confirmed live. The same handle is observed; a waiting folder
alone is not treated as a running process. Initial serial then concurrent full
deliveries use separate empty destinations, four wheel workers plus at most one
engine transfer. The fixed 08:47:07 expiry is unchanged.

Refreshed the earlier canary export's ledger from its authoritative cleaned
state. Dataset remains 110,874 bytes; updated SHA256 is
`ac4881c58f45d208456caeacc63538a88b6e4ba238d95fb702cf84d70b06a2af`.
The previous `c04cee…` hash refers to the prior snapshot with its pre-cleanup
ledger; its separately included cleanup receipt already showed verified absence.

## 07:53 UTC — full-payload overlap passed, limited 11% screening gain

Both cases fetched the same 194-wheel closure (**5,361,408,398 remote bytes**
plus the same local mmcv copy) and same **1,000,524,577-byte r5 archive**, into
empty destinations, with all trusted hashes verified and **zero wheel cache hits**.

| Acquisition case | Overall (s) | Wheels (s) | Engine archive (s) |
|---|---:|---:|---:|
| Serial | **104.506** | 78.642 | 25.862 |
| Four wheel fetches + one engine fetch concurrently | **93.034** | 93.031 | 43.563 |

Concurrent acquisition saved **11.471s / 10.98%** in this one ordered pair.
Wheels slowed 14.389s and the engine slowed 17.701s under contention; timings
overlap and are not additive. This is **not** a fresh app boot or dependency
installation result. The gain is below the plan's 20% promotion threshold, so
no repeated/adopted overlap implementation or startup saving is claimed.
No full repo/models/app workload was present; OS/CDN cache state is unknown.
The corrected native copy/hardlink screen is dispatched next on these already
verified local wheels, with separate empty venv/tool-cache paths.

## 07:56 UTC — actual hardlinks work; 8.1% gain does not earn promotion

Corrected native link screen is terminal **passed**, not the earlier pre-install
guard failure. Same full 194-wheel closure, bytecode enabled, fresh independent
venv/cache per case, same Linux filesystem:

| Link mode | Package ready (s) | Imports/CUDA validated (s) | Sample link count before/after supported cache clean |
|---|---:|---:|---|
| Copy | 32.781 | **48.502** | 1 → 1 |
| Hardlink | 29.365 | **44.570** | 2 → 1 |

Hardlinks were actually realized, not inferred from a requested flag. Full
inventory/pip consistency/imports/CUDA passed; the installed sample file hash
and GPU execution survived uv's supported cache cleaning in both cases. Only
owned inactive trial venv/cache children were then retired; no working app env
or user cache existed on this isolated screen worker.

The observed validated gain is **3.933s / 8.108%**, one ordered pair. It does not
meet the 20% promotion threshold, so the adaptive harness did not run extra
repetitions or claim a median, application-output acceptance or a default change.
Copy remains the reviewed candidate policy. Together with the full-payload
overlap result, this closes both previously missing screens with modest positive
but non-promoted results, not fabricated failed speedups.

Preserved **33 receipt/log files, 197,154 bytes** on EC2, with checksums. Initial
sanitized final-screen export is 371,996 bytes; cleanup is in progress. Quota is
**20% used** (no reset consumed). The source/mock suite now has **132 passing
tests**, including retaining a failed trial beside successful cleanup in export
and rejecting signed-credential patterns before public output.

## 07:57–08:05 UTC — final teardown, consolidation and coverage audit

Final-screen **56218707** destroyed/absence verified **07:57:26.950 UTC**,
well before its fixed 08:47:07 expiry. Its 618.656s accepted-to-cleanup span at
the quoted rate is approximately **$0.0323 rent**, not a final invoice or the
whole campaign's cost. Temporary disk contents are gone; 33 protected receipt/log
files and immutable artifacts remain. Independent provider inventory at **07:59:20**
shows **no task-owned rentals**, protected 4070 SUPER still exited, and all three
task expiry timers **inactive**. No delete was issued for a protected instance.

Final sanitized C dataset: **372,535 bytes**, SHA256
`00ee6f912cfc30bb4b6cff7b7efc40c1eee67398c2d62e861693b06b0e79d00e`.
It includes the fresh owned-resource/timer inventory and cleaned ledger.
[Final-screen receipts](dependency_optimization_20261015/final-screen-results.json)
remain separate from first-host medians and the different-host app canary.

The [requirement coverage audit](DEPENDENCY_INSTALL_OPTIMIZATION_COVERAGE_20261016.md)
now reconciles all work packages, B0–B10, native/output/first-use gates, conditional
non-promoted branches, retained failures and cleanup. The consolidated
[report](DEPENDENCY_INSTALL_OPTIMIZATION_REPORT_20261016.md) includes what helped,
what lost and what remains untested; it does not equate dependency time with
total startup, infer fleet p95, promise five minutes, remove capabilities, or
claim direct WebRTC/TURN/production readiness.

`verify_dependency_optimization_evidence.py` passed its **offline captured-evidence
audit** against the independently pinned wheel manifest, six clean full-delivery
trials' native/inventory exits, 28 identical controlled videos, fresh canary,
S3 success/failure retention, actual full overlap/link payloads and final absence.
An initial verifier schema assumption expected generic S3 `passed`; corrected
it to the receipt's exact `restored-native-and-health-passed` state, without
altering evidence. Added negative fixtures: relabeling failed second-host S3 or
removing verified cleanup makes the audit fail. **135 stdlib/mock tests pass**;
shell syntax and `git diff --check` pass. No macOS package/CUDA tests were run.

The first-host median generator now labels its scope explicitly and calls its
old missing gates `not_proven_by_this_input`, not current whole-goal gaps.
Historical capture `goal_complete=false` metadata remains untouched; the final
coverage/report carry the current decision. Gated experiment/report complete;
production source-pip/defaults, templates and inference config unchanged. No
Git commit/push, Docker revival, new AWS compute or IAM change. Last quota is
**20% used**, below 75%; no reset credit used. No GPU test remains running.
