# MuseTalk dependency installation optimization plan

Prepared 2026-10-15, America/Chicago. **Plan only: no implementation, installs,
GPU rentals, uploads, permission changes or production changes are authorized by
this document.** Baseline source: `5cc706e90e50e93da1310628c025a84199cd8042`.

## 1. Objective and recommended direction

Reduce the time needed to obtain a **correct, usable Python environment** on a
fresh RTX 3090, while making restarts reuse an already validated environment.
Keep the current model output, GPU binaries, recipe and supported capabilities.

The recommended first implementation is:

1. Capture the exact validated package/artifact contract.
2. Generate a complete, hashed wheel manifest and an offline wheelhouse.
3. Compare normal pip, locked pip and offline-wheelhouse pip.
4. Compare pip with a pinned `uv` against **identical wheels**.
5. Select the fastest reliable delivery/installation combination, integrate it
   as an opt-in mode, and test a fresh host before changing defaults.

Do not start by moving everything to S3 or rewriting the whole installer. Keep
the existing pip path as a documented rollback while validating each change.
Dependency locking and artifact selection come before a tool swap.

### Scope

In scope: venv creation/bootstrap tools, dependency resolution, index lookups,
package transfers, wheel builds, installation, bytecode, package caches, optional
dependency groups, environment reuse and dependency-related first-use delays.

Out of scope: stock-container provisioning, SSH/Jupyter launch mode, Git checkout
size, model/voice weights, avatar-cache transfers, TensorRT engine restoration or
building, autoscaler capacity policy, and changing inference quality/FPS settings.
Those remain fixed when testing dependencies. API startup and first usable output
are **validation boundaries**, not opportunities to move installation work out of
the timer. This plan does not revive the custom Docker/GHCR project.

## 2. Measured starting point

Sources: [fresh-host report](RTX3090_DEPENDENCY_STARTUP_PROFILE_20261015.md),
[structured timing receipts](dependency_profile_20261015/summary.json), and
[per-package events](dependency_profile_20261015/pip-events.jsonl).

| Measurement | Baseline | Implication |
|---|---:|---|
| Create venv + packaging tools | 7s | Small cost; do not begin by optimizing this |
| Main/server shell step | **266s** | Primary fresh-install target |
| Main pip execution inside that step | 264.964s | Similar boundary, excludes some wrapper overhead |
| Incremental Kokoro packages | 38s | Separate capability-dependent target |
| Avatar-prep packages + import check | 26s | Separate capability-dependent target |
| All three package groups | **330s** | Full-stack comparison boundary |
| Main resolver/preparation | 159.544s | Includes metadata and initial downloads, not pure solver CPU |
| Remaining preparation/downloads | 21.628s | Additional outer phase |
| Main wheel builds | 0.457s | Source compilation was not the main problem |
| Main requirement installation, 119 packages | **82.214s** | Unpacking/files/bytecode still cost time after downloads |
| Main downloader calls | 113.320s | Nested in preparation; do not add twice |
| Main unnecessary extra-index 403/404 requests | 15.492s | A measurable, bounded source-selection opportunity |
| Main DNS spans | 0.279s | No evidence for changing DNS as the first optimization |

Heavy main packages, download / wheel-install seconds:

| Package | Download | Install |
|---|---:|---:|
| TensorRT 10.3.0 libraries, ~2.04 GB displayed size | 33.225 | 19.854 |
| PyTorch 2.5.1+cu121, ~780 MB | 13.960 | 13.890 |
| cuDNN 9.1.0.70, ~665 MB | 9.895 | 6.370 |
| cuBLAS 12.1.3.1 | 6.722 | 3.895 |

These are good prefetch/cache targets. They do not establish that one package
always causes the historical 16–37 minute installations. Earlier runs lacked
comparable package-level traces. The latest sample is one unverified US host,
not a fleet distribution or an installation-time guarantee.

### Existing S3 venv evidence is a warning, not a default solution

The [earlier S3 experiment](Python-s3-venv-progress.md) published a private
4,509,210,997-byte archive. One fresh GET took **1,263.912s at 3.568 MB/s**;
checksum/extraction/validation brought package readiness to **1,361.720s**.
That is much slower than the latest 330s full-stack package phase. The archive
was functionally valid, but its delivery path has not earned a fast-start default.

## 3. Success criteria

Targets below are **experiment objectives**, not promised results.

| Boundary | Objective | Conditions |
|---|---|---|
| Fresh main dependency step, artifact acquisition included | First target **≤180s**; stretch **≤120s** | Same Linux/Python/CUDA package contract and validation |
| Warm restart dependency check | **≤10s**, no package download/install if valid | Existing compatible venv and unchanged dependency contract |
| Full-stack dependency phase | Report separately against 330s | Identical enabled groups; no credit for disabling capabilities |
| Candidate adoption | At least **20% lower median** fresh dependency time than its matched control | Repeated trials and no new failures/correctness regression |
| First API readiness and first usable output | No meaningful regression; investigate >5% or >1s, whichever is larger | Compare the same input, recipe and cache state |

The main timer starts **before** fetching the lock, wheel bundle, installer binary
or venv archive needed by that method. An offline install into a warm cache is a
useful microbenchmark, but is not a fresh-instance result.

Keep common venv/bootstrap time as its own field for every variant. Charge any
additional method-specific tool bootstrap to that method, including uv delivery.
Record `dependency-ready = common bootstrap + method setup + acquisition +
verification + installation + package validation` for sequential tests. Imports,
API readiness and first usable output remain separately measured acceptance
boundaries. For overlapped boot experiments, use the event timeline/critical path
instead of adding overlapping phase durations.

Do not equate these targets with five-minute total scale-out. Dependency work
does not remove the separate provider/container and model-download stages.

## 4. Non-negotiable compatibility contract

Before experimenting, record and enforce:

- Linux x86_64, CPython 3.10, interpreter/ABI and wheel platform tags, compatible
  glibc floor and the actual base environment used for validation.
- cu121 matrix with `torch==2.5.1+cu121`, `torchvision==0.20.1+cu121`,
  `torchaudio==2.5.1+cu121`, and TensorRT libraries/bindings/shim **10.3.0**.
- **`torch_tensorrt===2.5.0` must select the validated artifact**, not the different
  `2.5.0+cu121` build. Record wheel source and SHA256, not just a similar-looking
  version string. See [requirements notes](../requirements/README.md).
- Existing pins including NumPy 1.23.5, ONNX 1.17.0, setuptools 60.2.0, wheel
  0.48.0, packaging 24.2 and pip 26.2.1. A tested uv version becomes an additional
  pinned tool, not a reason to refresh package versions.
- Native VP8 compatibility: aiortc 1.14.0, PyAV 16.1.0 and cffi 2.1.1; preserve
  the separate encoder manifest/provenance checks.
- Existing avatar-prep installation constraints/order: local mmcv 2.1.0 wheel,
  chumpy's controlled build environment, and mmengine/mmdet/mmpose. Do not
  reintroduce openmim's conflicting dependency chain.
- Group flags and recipe selection; no engine rebuilds or replacement engines
  introduced to hide a dependency incompatibility.

Do not use the upstream root `requirements.txt`, switch to CPU PyTorch, inherit
arbitrary base-image site-packages, or loosen pins to make a benchmark pass.

## 5. Prioritized work packages

| Priority | Work | Fresh instance? | Why / measured opportunity |
|---|---|---|---|
| P0 | Exact contract, lock and reproducible benchmark harness | Foundation | Prevents wrong-binary or unfair comparisons |
| P1 | Source-specific artifact selection and bounded parallel prefetch | Yes | Reduce repeated discovery and serial transfer |
| P1 | Complete verified wheelhouse, install with indexes disabled | Yes | Deterministic acquisition/install; avoids runtime source builds |
| P1 | Controlled pip-versus-uv comparison | Yes | Tests the 82s local installation cost without changing wheels |
| P1 | Validated venv/cache reuse | **Restart only** unless cache delivered | Avoid work already completed on the same worker |
| P2 | Bytecode-policy comparison | Yes, possible first-use trade-off | Identify a real readiness saving, not a shifted cost |
| P2 | Optional-group and serving-only dependency audit | Yes, capability-dependent | Smaller contract, but not the same-capability baseline |
| P3 | Optimized S3 wheel/env delivery | Conditional | Existing single-GET env result was too slow |
| P3 | Reuse selected packages from the stock environment | Conditional/high risk | Only if exact ABI/artifact compatibility is proven |

### WP0 — Freeze artifacts and improve the test harness

Deliverables:

1. A machine-readable package inventory and complete dependency closure for
   `server`, `server+kokoro`, and `server+kokoro+avatar-prep`, plus separate
   bootstrap tooling. Legacy/chin/cu128 stay separate and retain current behavior.
2. A manifest per selected wheel: normalized name, exact version, filename,
   byte size, SHA256, origin, compatibility tags, capability group and build
   provenance if produced locally. Hash all inputs affecting the environment.
   Store stable canonical source URLs or object IDs only; obtain temporary auth
   at retrieval time, never embed expiring signed URLs or credentials in locks.
3. Distinct **online acquisition manifest** and **offline install lock**. The
   former can record original URLs; the latter must not retain remote direct
   URLs that cause downloads even with indexes disabled.
4. Tool-independent phase timers: venv/bootstrap, lock lookup, acquisition,
   verification, build, install, bytecode, package checks, first imports and
   first usable output. Record cache state, disk peak and retries/fallbacks.

Use the accepted pip reports and installed metadata as evidence. Do not treat
every entry in `constraints-cu121.txt` as a package that needs installing:
constraints also describe optional, inactive groups. Validate the resolved
closure, including bootstrap tools and environment markers.

The existing pip hooks do not instrument uv internals. Reuse the external
resource sampler and wall-clock runner for both tools; record hook coverage
and tool-specific diagnostics instead of inventing comparable per-package spans.

**Gate:** exact expected GPU artifacts and group inventories can be verified
before any candidate is allowed to become the production install mode.

### WP1 — Avoid unnecessary discovery and prefetch exact packages

Implementation outline:

1. Derive an acquisition list from the validated manifest, not fresh unbounded
   resolution on every worker. Keep package filename/source/hash choices fixed.
2. Test pinned direct artifact acquisition against the current three-index pip
   path. Main has 188 negative extra-index requests (~15.5s); do not advertise
   removal of the entire 159.5s resolver span as a saving.
3. Fetch the largest wheels first with bounded concurrency **2, then 4**. Try 8
   only if the prior runs show available network/CPU/IO headroom. Measure actual
   bytes/rates, not advertised link capacity.
4. Write uniquely named partial files into an experiment-owned cache. Verify
   size/hash, then publish each completed wheel. Resume/retry only known failed
   transfers with bounded backoff; never accept a partially downloaded wheel.
5. Once complete, install from those files. Confirm the installer does not
   download the same bytes again. Do not have multiple package installers write
   to the same venv concurrently.
6. After isolated improvements are measured, test limited prefetch overlap with
   other boot work. Report the dependency path's own start and the overall
   readiness delta; shared network contention can negate apparent overlap gains.

Keep the current NVIDIA/PyTorch indexes for any normal-source fallback. Merely
adding `MUSETALK_WHEELHOUSE` is **not offline mode**: the current installer still
adds both extra indexes. Explicit index disabling and complete locks are needed.

For optional Kokoro, separately test PyPI/GitHub-only acquisition: all 100 GPU
index requests were misses in the latest run (~10s request spans). This does not
count as an improvement to the main/server group.

**Gate:** identical installed versions/artifact hashes, no unexpected network
fetches after acquisition, and lower total acquisition-plus-install time.

### WP2 — Build a complete offline wheelhouse

A wheelhouse is a directory of prebuilt package files, **not an installed venv**.
It still requires local installation, but avoids per-worker source builds and
uncontrolled package discovery. This is an established
[pip deployment workflow](https://pip.pypa.io/en/stable/topics/repeatable-installs/).

Builder procedure:

1. Use an isolated matching Linux/Python environment with sufficient scratch
   disk. Do not use macOS or fill the production EC2 disk to build CUDA wheels.
2. Collect the exact upstream wheels selected in WP0. Reuse the validated local
   mmcv wheel. Retain required notices and provenance.
3. Prebuild wheels for the small sdist-only items, such as the TensorRT shim,
   antlr4 runtime and chumpy when that group is enabled. Use the existing pinned
   build requirements and chumpy build-isolation behavior; no runtime nvcc build.
4. Record input-sdist hash, build toolchain, output-wheel hash and compatibility.
   Building once removes a failure surface; the measured subsecond main wheel
   builds mean this alone is not expected to save minutes.
5. Generate complete platform-specific hashed locks for each supported group.
   Convert the optional spaCy model URL and local mmcv reference into verified
   local artifacts for the offline variant. Preserve extras through the resolved
   closure. A missing group wheel must fail preflight, not silently use an index.
6. In a fresh Linux scratch venv, install with indexes disabled and prove there
   were no package-network requests after acquisition. Run the checks in §8.

Illustrative command, **only after generating the complete offline lock**:

```bash
"$VENV_PY" -m pip install \
  --no-index --find-links "$VERIFIED_WHEELHOUSE" \
  --require-hashes --only-binary=:all: \
  -r "$OFFLINE_GROUP_LOCK"
```

`server.in` plus the existing constraints is not itself this hashed closure.
Every dependency must be pinned and hashed for hash-checking mode; see
[pip secure installs](https://pip.pypa.io/en/stable/topics/secure-installs/).
Do not add blanket `--no-deps` until closure completeness is independently
verified, and always verify the resulting environment.

Delivery alternatives to compare, rather than assume:

| Delivery | Benefits | Risks / test requirement |
|---|---|---|
| Exact original CDN URLs + parallel fetch | No new artifact storage service | Several hosts/requests; availability and rate variance |
| Individual private S3 wheel objects | Content-addressed reuse, per-object concurrency | S3 transfer cost, many requests, path may be slower |
| One private wheelhouse archive | One retrieval, simple manifest | Slow single connection, extra temporary space/extraction |
| Existing local verified wheelhouse | Very fast repeat installs | Not a fresh-host acquisition result |

Already-compressed `.whl` files may gain little from an additional archive
compression layer. Measure plain tar versus low-cost compression; count both
delivery and extraction. Do not compress/upload several GB merely on assumption.

**Gate:** complete offline installation succeeds with unchanged capabilities,
and acquisition overhead does not erase the savings.

### WP3 — Test uv against pip, without a hidden dependency change

Start **offline**, giving both installers the identical wheelhouse and locks.
This isolates local installation from CDN/resolver variability. Pin and verify
the uv binary; include obtaining it in a fresh-host deployment's measured cost.

Run at least these variants:

- Current pip bytecode behavior, as the control.
- uv with bytecode compilation enabled for comparable eagerness.
- uv with its default bytecode behavior, measuring imports and first use too.

uv's documented differences include index selection, pip-specific configuration
handling, build isolation and default bytecode behavior. Therefore do not simply
replace the command and assume the same GPU artifacts or timing boundary.
[Compatibility reference](https://docs.astral.sh/uv/pip/compatibility/).

Project-specific safeguards:

- Explicitly supply interpreter, constraints/lock, cache and source configuration;
  do not depend on inherited pip configuration being honored.
- Prefer offline wheels or explicit per-package sources; do not broadly enable
  an unsafe multi-index strategy as a shortcut for the TensorRT/PyTorch pins.
- Assert the arbitrary-equality torch-tensorrt artifact and cu121 local versions.
  If a syntax/behavior differs, express the same artifact with a verified direct
  wheel/local reference, not a different binary or looser version.
- Do not run a synchronizing command that removes bootstrap tools or optional
  groups omitted from an incomplete lock. Preserve explicit group contracts.
- Exercise mmcv, chumpy and optional groups separately. Mixed pip/uv operation
  must be an explicitly tested mode, not an unnoticed fallback.
- Inspect cache/link behavior and filesystem compatibility. Never modify tool
  cache internals to accelerate the test; use supported cache operations.
  [uv cache guidance](https://docs.astral.sh/uv/concepts/cache/).

**Gate:** a repeated installation/readiness gain with exact GPU artifacts and
no extra incompatibilities. If uv does not win sufficiently, retain pip and use
the wheelhouse/source-selection improvements independently.

### WP4 — Bytecode, local IO and warm-cache reuse

Bytecode is a separate test, not an automatic optimization. Compare pip's normal
installation with `--no-compile`, and uv's policies above. If deferring work
increases first imports or first-call latency enough to cancel the install gain,
reject it. Do not defer unbounded compilation into a user's first call.

Measure wheel-install CPU, bytes written, peak space and first-import cost. Test
native supported cache linking/copy behavior on the actual worker filesystem;
do not move cache paths to another filesystem and still claim zero-copy results.
Do not disable integrity checks or change global filesystem/cache settings.

For an existing worker:

1. Keep the venv and package cache outside a replaceable source checkout.
2. Use dependency-contract identity plus the existing install-state checker to
   distinguish unchanged, repairable and incompatible environments.
3. Skip package installation when valid; repair only missing/wrong packages when
   safely possible. Reserve `--clean` for an explicitly required recreation.
4. A source-only code change must not trigger full dependency redelivery. Recipe,
   packaging inputs, groups or ABI changes can invalidate the dependency contract.
5. Test interrupted installs and stale/missing stamps; never trust a stamp alone
   over conflicting package facts. Keep cache size bounded with tool-supported
   maintenance, outside active installation windows.

These are **restart gains**. A new Vast instance cannot benefit from another
instance's local cache unless that cache is actually delivered and timed.

### WP5 — Conditional S3 venv restoration, after fixing transfer economics

Keep this as a gated alternative to wheelhouse installation, not the first default.
The historical archive passed same-host checks, not portable second-host proof.

1. Establish a fresh end-to-end target from the best pip/uv wheelhouse candidate.
2. On an approved isolated Linux test host, compare one GET with bounded parallel
   range retrieval of the **same immutable archive**. Record request setup,
   retries, aggregate transfer, checksum and extraction separately. This trial
   has not already been run or authorized by the old S3 experiment.
3. Bind every range to the expected immutable object identity/size; validate
   response ranges and final trusted SHA256. Do not splice parts from versions
   that changed during the transfer or accept multipart ETags as content hashes.
4. Use a pilot transfer to reject a demonstrably unsuitable path, but do not call
   a pilot a full-archive benchmark. Any full redownload counts in cost/time.
5. Preserve exact base-interpreter/OS/ABI and canonical path requirements:
   `/workspace/.venvs/musetalk_trt_stagewise`. Audit archived paths/symlinks and
   check required system libraries. A venv is not universally relocatable.
6. Validate staged files, publish under an install lock, then execute the **new
   canonical interpreter and entrypoints**, not the old venv via stale shebangs.
   Preserve a recoverable old environment until native/runtime checks pass.
7. Require successful fresh restoration on a second matching worker before
   production integration. No fallback that quietly rebuilds TensorRT engines.

Adopt only if **download + verification + extraction + correctness checks** beat
the current best equivalent dependency route with acceptable transfer cost and
failure behavior. Fast extraction from a previously local archive does not qualify.

### WP6 — Capability-based reduction, after same-contract optimization

This is a separate comparison because removing functionality changes the contract.

- Local Kokoro is a candidate to disable on workers using external OmniVoice TTS.
  Confirm the route contract first; measured incremental package opportunity is
  38s. Its weights are outside this plan's package timer.
- Keep avatar-prep unless `/avatars/prepare` is explicitly moved off these nodes.
  Its package opportunity is 26s, not the full model-download stage.
- Audit serving-time imports and fallback routes for ONNX, torch-tensorrt,
  torchvision/torchaudio and other export/training-oriented dependencies. Their
  presence in the current server list is **not proof they can safely be removed**.
  Document why each stays or goes, and test public routes and error/fallback paths.
- Never remove TensorRT's runtime libraries just because engines are prebuilt.
  Do not claim the size of a removed transitive package as savings if another
  required package still pulls it into the resolved closure.
- Preserve the full-stack installer and explicit capability reporting. If a
  serving-only group is introduced, make it opt-in until verified and accepted.

Reusing base-image PyTorch/system packages is a last-resort separate experiment:
the base may contain a different Torch/CUDA/TRT build. Do not enable unrestricted
`--system-site-packages` for the production server. An incompatible preinstalled
stack is not a free substitute for the validated binaries.

## 6. Artifact and cache design

Proposed layout/controls below do **not** exist yet:

```text
requirements/locks/linux-x86_64-cp310-cu121/<group>.txt
configs/dependencies/<artifact-id>.json
scripts/build_dependency_wheelhouse.py
scripts/fetch_dependency_wheels.py
scripts/benchmark_dependency_install.py
docs/dependency_install_optimization_progress.md
```

Prefer a dependency artifact ID derived from selected group inputs, complete
resolved artifact hashes, Python/platform/ABI contract and build toolchain.
Store the source commit for provenance, but **do not invalidate a reusable
dependency bundle for every unrelated application-code commit**.

If S3 delivery wins, a candidate private prefix is
`python-venv/wheelhouse/<artifact-id>/` in the existing bucket, clearly separate
from installed-env archives. Confirm actual narrowly scoped read/publish access
before any upload. New IAM changes require separate explicit approval. No public
package repository, registry publication, new always-on cache server or new AWS
compute is part of this plan's execution by default.

The manifest must be trusted independently of the downloaded payload: pin its
digest in reviewed configuration, or use a verified signing policy. Hashing an
untrusted payload against a hash supplied only alongside that same payload does
not establish authenticity.

Never include AWS/GitHub credentials, runtime secrets, `.env` files or model/avatar
data in dependency bundles. Keep workers read-only; use the approved EC2-mediated
short-lived credential flow for private artifact retrieval. Redact URL query
tokens and headers from logs. Retain license notices, manifests and build inputs.

Plan disk headroom for downloaded wheels + installed env + tool cache + staging
and any recoverable old venv. Check before download/extraction; do not solve a
full disk by deleting user files or unrelated environments.

## 7. Benchmark sequence and cache fairness

Do not run the entire matrix immediately. Promote only candidates that pass
earlier stages; re-use one authorized isolated test worker where appropriate.

| ID | Experiment | What it isolates |
|---|---|---|
| B0 | Current source pip, same group contract, fresh venv and test-owned empty package caches | Matched baseline |
| B1 | Exact artifact manifest + source-specific acquisition + pip | Discovery/source-selection change |
| B2 | B1 with bounded parallel prefetch | Acquisition concurrency only |
| B3 | Local complete wheelhouse + pip | Offline installation microbenchmark; excludes remote acquisition explicitly |
| B4 | Same local wheelhouse + uv, compiled bytecode | Installer comparison |
| B5 | Best installer, alternate bytecode policy | Install versus first-use trade-off |
| B6 | Best full wheelhouse delivery + installer on a fresh environment | Real acquisition-to-readiness candidate |
| B7 | Validated restart on the same worker | Reuse; must not be labelled fresh install |
| B8 | Optional repaired S3 env route | Compare complete restore against B6, only if justified |
| B9 | Optional serving-only group | New capability contract; report separately |
| B10 | Fresh second-host confirmation of winner | Portability and environmental variance |

Screen B3/B4 before spending repeatedly on multi-GB transfers. Once a candidate
looks useful, run at least **three matched trials** and report individual samples,
median and range. Alternate order where feasible to reduce first-run bias.
One host/three repetitions cannot establish fleet p95; do not manufacture one.

Record for each trial:

- source/lock/artifact hashes; interpreter, installer version, groups and recipe;
- host allocation, filesystem, free disk, CPU/RAM, network observations and load;
- venv, package-cache and downloaded-wheel cache states independently;
- whether OS page cache and remote CDN caches are unknown/warm; do not drop host
  system caches or change shared-machine settings to manufacture a cold test;
- all acquisition/verification/install/import/readiness/first-output timers;
- downloaded bytes, retries, network endpoints, peak RSS/disk and package diffs;
- failures, fallback reason, fallback duration and **total time including fallback**.

Use fresh test-owned venv/cache paths, not destructive cleanup of a working
environment. Stop test API processes before environment replacement. Never
install into the production worker's active venv for an A/B comparison.

## 8. Correctness and performance acceptance

Every candidate must pass:

1. Exact expected package inventory and source/wheel hash checks; no surprise
   CPU Torch, alternate CUDA build or torch-tensorrt local-version substitution.
2. Existing `musetalk_install_state.py` checks and import smoke. Run package
   consistency checks; investigate any baseline diagnostics and prohibit new
   unapproved dependency conflicts. Never loosen pins merely to silence them.
3. CUDA execution, Torch/TensorRT imports, and loading the **same prebuilt engine
   bundle**, including its checksum/compatibility/output probe. No engine rebuild.
4. Codec/native VP8 verification, audio processing, and an existing-avatar named
   REST generation that downloads and decodes both video and audio tracks.
5. Equivalent first-import, API-ready and first usable output timings. If package
   work moved into those phases, it remains charged to readiness.
6. A short same-input/same-recipe throughput sanity check, with investigation of
   >5% regression under comparable conditions. This is not re-certification of
   the historical 400+ FPS config or permission to change that config.
7. Same enabled optional groups/routes, or an explicitly separate serving-only
   trial. No false pass because an untested capability was simply omitted.

A tunnel-only test can validate the dependency stack, but cannot establish direct
external connectivity, production registration or WebRTC/TURN call readiness.
Keep those distinctions in the report.

## 9. Failure handling and rollout

- Default initially remains the current source-pip path. Proposed modes should
  be explicit opt-ins, for example `source-pip`, `locked-wheelhouse-pip`,
  `locked-wheelhouse-uv`, and later `verified-venv-restore`; these are design names,
  **not current supported flags**.
- Installer outcome must specify requested/actual mode, artifact ID, cache hit,
  native validation outcome and any fallback. No silent path changes.
- Missing/incompatible artifact: use a bounded normal-source fallback only if
  policy allows it. A hash/authenticity mismatch is a failure to investigate,
  not a reason to install the suspect package or silently bypass validation.
- Do not run a second installer over a half-finished env and call that equivalent
  to a clean comparison. Recover into an isolated known state, account for time,
  and preserve the prior accepted env/receipts where applicable.
- Test network interruption, bad/truncated wheel, missing closure entry, wrong
  Python/glibc/CUDA matrix, expired credentials, low disk and changed dependency
  inputs. Ensure concurrent startup attempts cannot corrupt a shared env.
- After local/mock checks, Linux isolated validation, repeated measurements and
  second-host confirmation, canary the winner on **one** new worker. Observe its
  failures/readiness before broader defaults or autoscaler rollout.
- Roll back by selecting the previous install mode/artifact contract. Keep
  known-good immutable artifacts; avoid deleting the only recoverable env.

## 10. Cost, permissions and execution boundaries

This request is to **write the plan**, not start another paid experiment.
No rental, upload, IAM change or automation is performed while preparing it.

For a later implementation/testing request:

- Reconfirm total experiment budget and lease; use the existing $0.30/hour and
  $0.0015/GB Vast transfer ceilings unless explicitly changed. Include disk and
  the applicable quoted rates; do not assume old offers still exist.
- Budget each full artifact download, retry, GET, S3 transfer and storage using
  then-current verified prices/allowances. Do not assume the account's free
  egress allowance is available or that S3 delivery costs only GPU rent.
- Start with metadata/code/mock work and local-wheel installer comparisons.
  Batch useful tests into a bounded rental rather than create a new GPU for
  every variant. Abort poor candidates rather than complete every matrix cell.
- Bind one uniquely labelled test worker to an ownership ledger and independent
  hard expiry. Preserve every pre-existing instance, collect receipts and verify
  destruction. Do not reuse old campaign authority or retry an ambiguous create.
- The current EC2's limited free disk makes it unsuitable for an unchecked full
  CUDA environment/wheel build. Any extra builder/storage infrastructure needs
  an explicit choice, cost cap and cleanup plan before provisioning.
- macOS work is limited to source analysis, manifest generation from existing
  evidence, syntax and stdlib/mock tests. No pip resolution/install/CUDA tests on
  the Mac. Package-building/install tests must use the matching Linux environment.

## 11. Implementation checklist and stop points

Execution status, updated 2026-10-16; this overlay does not convert unrun
alternatives into completed tests. See the [report](DEPENDENCY_INSTALL_OPTIMIZATION_REPORT_20261016.md)
and [chronological receipts](dependency_install_optimization_progress.md).

- [x] Confirm current main/requirements revision without overwriting user edits.
- [x] WP0 manifest, complete group locks, artifact checks and benchmark boundaries.
- [x] WP1 source-specific acquisition and bounded hash-verified 2/4-way prefetch.
- [x] WP1 limited overlap with other boot downloads: full wheels plus unchanged
      r5 archive 104.506s serial versus 93.034s concurrent (10.98%, one pair);
      below promotion threshold, not integrated or called a boot-time saving.
- [x] WP2 complete Linux wheelhouse and offline pip control; no runtime downloads.
- [x] WP3 isolated pip-versus-pinned-uv comparison with the same package artifacts.
- [x] WP4 bytecode/first-use comparison and unchanged-env restart reuse.
- [x] WP4 native copy/hardlink screen: full inventory/imports/CUDA and supported
      cache clean passed; hardlink44.570s versus copy48.502s validated (8.108%,
      one pair), below promotion threshold; copy retained.
- [x] Select a promising combination; bounded four-way compiled uv leads.
- [x] B6 repeated full-delivery measurements and same-host §8 acceptance checks.
- [x] WP5 full immutable S3 restore, native/API checks and recoverable rollback;
      slower matched readiness, so not promoted for repetitions/default use.
- [x] WP6 read-only capability/import audit; full-stack path retained. Reduced
      capabilities/base-image substitution are conditional, not adopted/tested defaults.
- [x] Same-host opt-in canonical integration, Linux failure/rollback and warm onstart.
- [x] Fresh second-host confirmation of the final hardened installer. Canary
      56211950 passed canonical/native/output checks; 511.097s dependency time
      and ~24m17s create-to-complete expose substantial host/network variance.
- [x] One-worker fresh canary acceptance and cleanup; defaults stay source-pip
      until an accepted results review. No production rollout is inferred.

The second-host conditional S3 confirmation lost at package readiness
(685.786s versus 511.097s) and failed API restart; original venv restoration
passed, original API recovery did not. Preserve that failure separately from
the earlier successful same-host rollback. Canary destroyed/absence verified
07:37:29 UTC. The initial copy/hardlink screen failed before installation due to
a test-root guard; a corrected dedicated-worker pair later passed. Full-payload
overlap also passed; neither modest screening gain earned promotion/repetitions.

Eight-way prefetch was not promoted because four-way effective payload rates
were already close to the first host's advertised link. Large S3 wheel mirrors
or wheelhouse archives were not published: original-CDN delivery won the tested
path and the full S3 env screen lost. These remain explicitly untested delivery
alternatives, not established failures. Exact-artifact reuse and copy-mode cache
behavior were tested; filesystem-specific zero-copy promotion was not required.

Execution closed 2026-10-16 after full-payload overlap and actual hardlinks
passed their bounded screens but stayed below the 20% promotion threshold.
All three owned rentals are destroyed/absence verified, with expiry timers
inactive. Final requirement coverage, conditional-branch decisions and exact
receipts are in the [coverage audit](DEPENDENCY_INSTALL_OPTIMIZATION_COVERAGE_20261016.md).
This closes the experiment/report, not a production-default rollout.

After each implementation or test, update the proposed progress Markdown with:
change/source revision, hypothesis, exact variant/group/cache state, timings and
costs, correctness result, evidence paths, failure/recovery details, cleanup and
the decision to keep/reject/continue. Keep raw credential-bearing URLs private.

## 12. Source references

Project evidence and current implementation:

- [Fresh dependency-level startup profile](RTX3090_DEPENDENCY_STARTUP_PROFILE_20261015.md).
- [S3 venv results](Python-s3-venv-progress.md).
- [Installer](../scripts/install_musetalk.sh), [state checker](../scripts/musetalk_install_state.py),
  [requirement contract](../requirements/README.md).
- [Server group](../requirements/server.in), [Kokoro](../requirements/kokoro.in),
  [avatar prep](../requirements/avatar-prep.in), [cu121 pins](../requirements/constraints-cu121.txt).

Official technical documentation checked during planning:

- [pip repeatable installs / wheelhouses](https://pip.pypa.io/en/stable/topics/repeatable-installs/).
- [pip hash-checked secure installs](https://pip.pypa.io/en/stable/topics/secure-installs/).
- [pip install options](https://pip.pypa.io/en/stable/cli/pip_install/).
- [uv compatibility with pip](https://docs.astral.sh/uv/pip/compatibility/).
- [uv cache semantics and supported maintenance](https://docs.astral.sh/uv/concepts/cache/).

**Planning outcome:** prefer a reproducible locked-wheel path and a measured
installer comparison first. Keep repaired venv delivery and dependency removal
conditional. Success is a faster correct worker, not a shorter pip progress bar.
