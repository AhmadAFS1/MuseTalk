# Locked wheels plus uv halve dependency setup on one RTX 3090; CDN speed limits another

**Completed gated experiment and results report, 2026-10-16 UTC.** Conditional
alternatives that did not earn promotion remain explicitly untested, not failed.
This is not a production rollout or a five-minute startup certification. Production source-pip,
templates, inference configuration and autoscaling defaults remain unchanged.
Results and failures are maintained in the [execution log](dependency_install_optimization_progress.md).

## Executive summary

- Full-stack dependency delivery through import/CUDA validation fell from a
  **294.2s source-pip median to 132.2s** with exact wheels, four-way acquisition
  and compiled-bytecode uv: **55.1% less time** on this host. The clean comparison
  uses three controls and three candidate samples; all four candidate samples
  remain reported below, including the observer-affected one.
- The local installer comparison explains a substantial opportunity: with the
  same wheels already present, compiled uv took **49.8s** versus pip **132.0s**.
  This local result does not include remote acquisition and is not a boot time.
- Disabling bytecode did **not** improve uv readiness through first imports.
  Serial locked pip improved only about **9.6%** in its one delivery screen.
  Merely changing download sources is not the whole observed improvement.
- An opt-in Linux installer now installs the full contract and reuses it in
  about **0.4s**, without fetching packages. A missing-package repair took
  **4.31s**, with no remote package bytes. The real warm in-container onstart
  reached verified API health in **24.4s**. Full S3 restoration lost the matched
  dependency comparison (**150.3s**); equalized cold-JIT tests found no material
  installer first-use regression. The different-host canary now passes canonical
  install/native checks, API health and two decoded outputs. Second-host S3
  lost at package readiness (685.8s versus 511.1s) and failed app restart.
  Full-payload overlap and actual hardlinks gave only **11.0% / 8.1%** gains in
  single screening pairs, below the promotion threshold. Defaults stay unchanged.

A **different verified US host** now supplies an important limit: its full-stack
locked package phase took **511.1s**, of which acquisition took **431.3s**. The
2.04-GB TensorRT library from NVIDIA alone took **431.3s** (one attempt); local
server/Kokoro/avatar-prep installation took **55.2 / 2.5 / 3.1s**. This is package
readiness, before later import/GPU/API checks, not directly the first host's
validated-ready boundary. No matched source-pip control exists on the new host.
Its advertised 938.7 Mbps does not establish fast downloads from every CDN.
The full-S3 alternative was therefore tested again on this host and also lost;
the first-host loss alone was not a universal S3 rejection. Fresh boot completed in **24m17s from
create**, including **7m42s cloning** and **14m18s onstart**. Both generated MP4s
match each other byte-for-byte but differ from the first host's MP4 hash; do not
claim cross-host identity or infer an installer regression from this unmatched
comparison. The later S3 test failed before its same-host output check.

## What the timings measure

The full-delivery timer starts before the worker fetches the manifest, wheels
and method-specific installer tool. It includes verification, fresh venv/common
packaging bootstrap, installation, package inventory/consistency, CPU imports
and a CUDA/Torch-TensorRT execution check. EC2 ticket issuance is outside this
worker timer for both compared routes and is disclosed in receipts.

**It excludes Vast provisioning, stock image startup, Git checkout, models,
prebuilt engine restoration, API startup and avatar readiness.** Those were not
made faster by relabelling them as dependency work. All 194 full-stack packages,
including Kokoro and avatar preparation, remain enabled.

## Complete delivery: the combined approach wins on this host

| Method | N | Validated-ready samples (s) | Median [range] (s) |
|---|---:|---|---|
| Original three-index source pip | 3 | 293.050, 294.191, 297.623 | **294.191 [293.050, 297.623]** |
| Exact wheels, serial download, pip | 1 | 265.828 | 265.828; screen only |
| Exact wheels, two-way download, compiled uv | 1 | 147.582 | 147.582; screen only |
| Exact wheels, four-way download, compiled uv, clean samples | 3 | 153.327, 132.176, 131.177 | **132.176 [131.177, 153.327]** |
| Same candidate, all retained samples | 4 | 153.327, 132.176, 133.953, 131.177 | **133.064 [131.177, 153.327]** |

The clean candidate's **package-ready** median, before import/CUDA checks, is
**116.056s**. Its complete validated median is **162.015s below** the source
control (**55.07% reduction**). Keeping all four samples gives **54.77%**.
Reduction is calculated as `100 * (1 - candidate_median / control_median)`.
These ranges are observed sample spread, not confidence intervals or fleet p95.

All source controls use fresh caches and the same ordinary source commands/order.
They also verify selected source hashes, with trusted installed-file verification
for the two historical hashless reports. Offline candidates use the identical
reviewed wheel contract. No TensorRT engine is built by these package tests.

One measurement caveat: a **12.6 MB evidence collection** overlapped the third
four-way trial's installation, after its acquisition ended. It is retained in
the all-four result. A fourth fresh-env/fresh-cache confirmation, with no bulk
collection or competing API/package workload, took **131.177s**, with zero wheel
cache hits and 5.361 GB remote wheel payload. The clean sensitivity result uses
samples 1, 2 and 4, not the three fastest samples. OS page/CDN caches are
unknown and were not forcibly cleared. One rented host does not prove fleet
portability or total scale-out latency.

## Which parts helped, and which did not

| Change | Observed evidence | Decision at this point |
|---|---|---|
| Complete exact-wheel acquisition and offline installation | Both pip and uv pass the same inventory/native checks; no source builds during these installs | Keep as reproducibility foundation |
| Compiled uv rather than compiled pip, identical local wheels | Validated medians **49.785s vs 132.036s**, three each; **62.3%** reduction locally | Keep testing as the leading installer |
| Serial exact-wheel delivery + pip | **265.828s**, vs source median 294.191s; one sample | Alone does not meet the 20% adoption threshold |
| Two-/four-way prefetch | Acquisition: serial **130.082s**, two-way **94.090s**, four-way clean median **80.117s** (all-four **80.638s**) | Bounded parallelism helps; two-vs-four ranking needs care because transfer variance is material |
| uv without bytecode compilation | Package-ready **29.582s**, first imports **12.852s**, complete **52.257s** | Reject as current readiness winner; install saving shifts work into imports |
| pip without bytecode compilation | Complete local **110.817s**, one sample | Better than compiled pip locally, but slower than compiled uv |
| Warm accepted-environment reuse | Three checks **0.440 / 0.399 / 0.435s**, no package requests | Keep; this is a restart gain, not a fresh-host result |
| Missing-package repair | **4.310s**, no remote package bytes; moved files retained for recovery | Keep explicit/owned/inactive-only repair; do not silently repair arbitrary ABI/version changes |
| Full private S3 venv restore | **150.322s** through matched inventory/import/CUDA checks; one screen | Do not promote: slower than clean wheel/uv median **132.176s**, with additional transfer cost |
| Full wheel/engine download overlap | **104.506s serial vs 93.034s concurrent**, one complete-payload pair; hashes/zero cache hits verified | 10.98% acquisition-only gain, below promotion threshold; contention slows both transfers; not an established boot saving |
| Hardlink rather than copy installation | **44.570s vs 48.502s** local validated readiness, one native pair; actual links/hash/CUDA survive supported cache clean | 8.108% local gain below promotion threshold; keep copy; no full-boot/application acceptance claimed for hardlinks |
| Eight-way acquisition | Not run; no demonstrated need beyond four-way effective payload rates near this host's advertised link | Conditional gate not promoted |
| Remove Kokoro/avatar-prep, reuse arbitrary base packages | No default removal or system-site-package substitution | Keep the equivalent full-stack baseline; capability/ABI changes are separate decisions |

Acquisition timings above cover exact-wheel acquisition/verification only. The
source control's receipt calls its small bootstrap fetch `acquisition`; its main
pip downloads happen later inside pip. **Do not compare that approximately 1s
bootstrap field against the offline route's full 80s acquisition field.**

## S3: faster transport still loses after safe restoration

The existing 4.51 GB private venv archive was probed read-only using the same
version-bound 128-MiB prefix. Three samples per concurrency, alternating order:

| Connections | Samples (s) | Median (s) |
|---|---|---:|
| One | 2.875, 2.479, 2.336 | **2.479** |
| Four | 2.528, 2.624, 2.477 | **2.528** |

All prefix bytes matched; ETag, VersionId and response ranges were checked.
Total pilot payload was **805,306,368 bytes**. Four connections did not improve
the median prefix result here. The previous **1,263.912s full GET** is historical
and not representative of this pilot's effective rate; it must not be used to
claim today's S3 path still takes 21 minutes.

The subsequent **full-object experiment** used the same immutable archive,
4,509,210,997 bytes. Both complete downloads passed the independently trusted
SHA256. The ranged request used four workers, 32-MiB parts and 135 requests.

| Full-restore boundary | Seconds |
|---|---:|
| Single GET / checksum | 77.365 / 3.139 |
| Four-way range retrieval / checksum | 64.831 / 3.075 |
| Archive member/link audit | 27.358 |
| Extraction | 30.676 |
| Continuous ranged restore → inventory + pip consistency | 135.239 |
| Continuous ranged restore → matched imports/CUDA validation | **150.322** |
| Including additional canonical native checks | 163.197 |
| Continuous ranged restore → API health | 182.682 |

Single/ranged payload rates were **58.285 / 69.553 MB/s** (decimal). Parallelism
improved the full-object transfer by **16.2%**, despite not helping the small
prefix pilot. Path/link auditing and extraction erased that transfer advantage:
the matched **150.322s** result is slower than the wheel/uv **132.176s** median.
This branch is therefore not promoted for repetitions or default use. It is one
same-host screen, not a portable distribution or a claim that S3 is always slow.

The new canonical interpreter/entrypoints and unchanged prebuilt plans were
tested, then the retained original venv and healthy original API were restored.
Models, system packages and code remained on disk. Do not compare the S3
**182.682s API boundary** against uv's dependency-only boundary.

## Reliability and application acceptance

Local source/mock checks cover manifest provenance, artifact corruption, wrong
platform/contract, auth/retries, disk preflight, concurrency and dispatch safety.
The isolated Linux followup additionally passed real loopback HTTP auth,
truncation, disconnect and transient-retry fixtures; corrupt-cache refusal;
per-venv lock contention; stale failed/accepted stamps; and missing-package repair.
The later live AWS fixture also passed: a deliberately one-second manifest GET
returned **403 / AccessDenied**, with expiry confirmed in the AWS response.
The fetcher refused it without fallback, a published artifact or a leftover
partial file. Normal read tickets retain the original 300–3600s lifetime guard.

The initial source/uv application test produced decoded 249-frame H.264 video
and AAC audio in both cases, but cross-video SSIM was **0.978965**, not bitwise
equivalence. The controlled fresh-process matrix is now complete: **all 16 MP4s
match byte-for-byte**, and all five within-/cross-environment decoded SSIM
comparisons are **1.0**. Its same prepared avatar/audio and unchanged r5 plans
therefore pass the output-equivalence check. The earlier anomaly is retained;
it was not reproduced here and has no established root cause.

| Environment | API healthy (s) | First request (s) | Warm-request median (s) |
|---|---:|---:|---:|
| Source pip | 19.292 | 13.323 | 12.008 |
| Compiled uv | 18.695 | 13.316 | 12.140 |
| No-compile uv | 19.374 | **26.296** | 12.210 |
| New opt-in uv | 19.527 | **25.827** | 12.104 |

Warm medians differ by less than 2%; API health times show no material regression.
The approximately 13s first-request gap above initially blocked first-use
acceptance. A second controlled matrix gave each environment an independent,
initially empty `NUMBA_CACHE_DIR`, with identical cache-debug settings:

| Environment, equal cold-JIT policy | API healthy (s) | First request (s) | Warm-request median (s) |
|---|---:|---:|---:|
| Source pip | 18.157 | **25.258** | 11.719 |
| Compiled uv | 18.304 | **25.191** | 11.819 |
| Opt-in compiled uv | 18.650 | **25.171** | 11.811 |

Every case created 46 cache files and logged 36 compiled cache saves. All 12 new
videos are byte-identical to the prior 16; all four decoded SSIM comparisons are
1.0. This intervention supports unequal audio/JIT-cache history as the source of
the earlier gap, with **no material installer first-use regression on this host**
under equal cache policy. It does not remove the shared approximately 25s cold
first request or establish that all first-use cache state is controlled. No
production Numba policy was changed.

The no-compile environment's CPU-import test has already generated some bytecode.
Its later API case is not a pristine no-bytecode first-use experiment. REST output
25 FPS is a media rate; request throughput and rendering-log FPS are separate,
and neither recertifies historical aggregate 400+ FPS operation.

Native VP8 verification passed, but a separate H.264-only native-fallback check
was unsupported in both baseline and candidate; this pre-existing behavior is
not fixed here. The unchanged TAESD log reports `built_now=False`, `probe=exact`
and `hw_compat=ampere_plus`. Its historical `gate=FAIL` label remains recorded;
matching baseline output does not reapprove that old quality gate. Tests used
internal REST/tunnels, not direct WebRTC/TURN or production registration.

## Implementation and deployment boundaries

The [opt-in guide](DEPENDENCY_INSTALL_OPT_IN.md) documents
`locked-wheelhouse-pip` / `locked-wheelhouse-uv`; `source-pip` remains the default.
The wrapper preserves venv/cache outside the checkout, enforces the reviewed
contract, pins uv, verifies full inventory, and refuses implicit fallbacks or
adoption of foreign environments. Unsupported CUDA/group combinations fail.

Canonical Linux integration reused accepted dependencies and ran the normal
native/GPU self-test in **20.552s**; its read-only check took **0.316s**. An isolated
source-only checkout reused the same env in **0.445s**, without reading a private
ticket or making package requests. Changed requirement fingerprints were refused.
These are reuse/integration results, not fresh installs; apt/weights were skipped.

The existing `vast_onstart.sh`, using its supported alternate-installer hook and
isolated runtime/port, reached verified health in **24.400s** with a warm accepted
env. Setup check took 0s, existing r5 bundle verification 1s (no download), engine
provisioning 0s, and server-to-health/strict-backend checks 18s. Candidate shutdown
took **7.282s**, and the original API remained healthy. Stock provisioning, Git,
fresh models and external TURN connectivity were not part of that timer.

**Post-first-rental hardening has now passed a fresh Linux canonical boot.** The
canonical read-only check now verifies cached manifest digest, dependency-input
fingerprints, accepted ownership and the entire group inventory. It fails if
locked-mode configuration is missing. Fresh `--clean` is permitted only when
the venv path is absent (including refusal of dangling symlinks); an existing
venv is never removed, and a missing parent directory is created safely. This
fixes the fresh-boot path where Vast supplies `--clean` for a missing environment.
All **135 stdlib/mock tests** pass. Canary 56211950 executes the independently
hash-bound final installer overlay and passed fresh `--clean`, post-install
checks, CPU imports, GPU self-test and API/backend health. Its output acceptance
has now passed; earlier first-host receipts alone did not validate these later edits.

This campaign's new S3 wheel publication contains only the manifest, four tiny built wheels, their inputs and
locks—13 verified objects, about 0.75 MB total—not a new multi-GB wheel mirror.
Large wheels retain their exact original CDN sources. No new AWS compute, IAM
policy, public access, Docker registry or production template was introduced.
The fresh canary adds one **190-KB immutable private source overlay**, with an
independently pinned digest, so it tests final uncommitted installer code without
editing GitHub or hiding manual SSH staging time in its startup measurement.

The rented 100-GB Pennsylvania RTX 3090 quote was **$0.187875434/hour adjusted**;
both Vast transfer directions were **$0.00000390625/GB**. That first owned rental,
**56199065**, was destroyed at **2026-10-16 06:16:29 UTC**, before its two-hour
expiry; provider absence was verified and the expiry timer stopped. Its quoted
115-minute runtime is approximately **$0.36 in rent**, not a final invoice.
Two full S3 downloads transferred about 9.02 GB, plus the 0.81 GB prefix pilot;
the account's remaining AWS egress allowance and final bill are unverified.

Cleanup issued a delete only for the owned instance. Pre-existing 4070 SUPER
`51074906` was already exited and remained so. The pre-existing A5000 was absent
from the final listing; its disappearance is unexplained, and this task issued
no write against it. Do not infer from the cleanup's `protected_production_running`
field that the task stopped the already-exited 4070 SUPER.

Receipts and **3,138 two-second resource samples** are preserved, including the
[compressed raw capture](dependency_optimization_20261015/resources.jsonl.gz).
Selected processes' maximum sampled aggregate RSS was **13.18 GB**. The reported
61.22-GB cgroup counter exceeds the nominal 32-GB allocation and cannot be treated
as isolated worker RSS or required RAM. Sampling began after baseline installation
started; host-wide kernel counters, non-instantaneous maxima and the absence of
continuous filesystem-free-space sampling limit resource/peak-disk conclusions.

## Second-host S3 was slower and did not pass application restart

Four-way retrieval of the same immutable 4.51-GB archive took **501.489s** at
**8.992 MB/s**, followed by **28.597s** checksum, **70.486s** audit and **74.232s**
extraction. Inventory/consistency readiness was **685.786s**, **34.2% slower**
than this host's 511.097s locked-wheel path. Matched imports/basic CUDA reached
713.425s; additional read-only native checks reached 732.274s. These are dependency
boundaries, not successful API readiness.

The job then failed during local TURN startup. It restored the original venv,
but neither its recovery nor two subsequent bounded attempts established a
healthy API. The last attempt stopped at `API PID file remains`; final health
observation was unsuccessful. A read-only reproduction found a SIGPIPE in the
existing shell PID1-environment fallback for a public port mapping. Preserving
those public mappings in the test harness does not prove a production TURN fix.
No restored-env video was produced on this host. Keep this failed application
gate and rollback limitation even though the branch already loses on time.

Canary **56211950** was destroyed at **07:37:29 UTC**, before its unchanged
one-hour expiry, with provider absence verified. Seventeen final receipts/logs
were preserved privately; the sanitized canary dataset includes test failures
and cleanup. No protected instance was mutated.

## Final recommendation and experiment boundaries

Use the reviewed **exact-wheel, four-way acquisition, compiled-uv, copy-mode**
path as the candidate for a separately approved deployment. It meets the >=20%
same-host median gate, preserves all capabilities, passes controlled output and
first-use checks, and passed a different fresh host's canonical startup. Keep
source-pip as the current production default/rollback until deployment review.

Preserve accepted venv/cache outside replaceable code and use **SETUP_CLEAN=0**
for reusable locked workers; the canonical check handles an absent venv. Do not
ship the old template's unconditional clean flag as a restart optimization.

Do not adopt S3 venv restoration or deferred uv bytecode here. Do not add overlap
or hardlinks to startup on the strength of one sub-threshold screen. The actual
link test passed inventory/imports/CUDA, demonstrated sample nlink2→1 after normal
cache cleaning, and preserved installed-file hash/GPU behavior; it did not earn
application/whole-boot promotion. S3 wheel mirrors/archives, eight-way acquisition,
reduced capabilities and arbitrary base-package reuse were conditional branches,
not run variants or established failures. The [coverage audit](DEPENDENCY_INSTALL_OPTIMIZATION_COVERAGE_20261016.md)
records these decisions against every work package and acceptance gate.

Faster dependencies do **not** guarantee a five-minute worker: the different-host
canary required 24m17s create-to-complete, including a 7m42s full checkout and a
431s TensorRT CDN transfer. Those are evidenced separate bottlenecks, not hidden
inside the local install result. Direct WebRTC/TURN, production registration and
historical 400+FPS recertification are not claimed by these dependency tests.

The final-screen worker **56218707** ran 07:47:08–07:57:27 UTC, approximately
**$0.0323 quoted rent** before bandwidth/other charges, not a final bill. Its
33 receipt/log files were preserved and provider absence verified. The 07:59 UTC
inventory independently shows **zero task-owned rentals** and all three expiry
timers inactive. Protected 4070 SUPER remains exited, as it was before this work.
Temporary rented disks are gone; protected evidence and immutable artifacts remain.

All required gated screens, candidate repetitions, same-host application checks,
fresh-host canary, failure retention, source/mock checks and cleanup are closed.
The [offline evidence audit](dependency_optimization_20261015/evidence-audit.json)
cross-checks full payloads, inventories, native step exits, 28 controlled videos,
retained S3 failure and final resource absence. It is a consistency/coverage
check of actual receipts, not a substitute for their Linux executions. Earlier
snapshot `goal_complete=false` fields are retained as capture metadata, not the
current experiment completion decision. No Git commit/push, production switch,
Docker work, new AWS compute or new IAM change occurred. Last quota check:
**20% used**, below the user's **75% used** stop guard; no reset credit consumed.

## Evidence and reproduction

- [Sanitized receipts](dependency_optimization_20261015/results.json) and
  [derived timing summary](dependency_optimization_20261015/timing-summary.json).
- [Different-host canary receipts](dependency_optimization_20261015/canary-results.json),
  kept separate so the first host's medians are not silently mixed with another host.
- [Final overlap/link receipts and cleanup](dependency_optimization_20261015/final-screen-results.json)
  and [requirement coverage](DEPENDENCY_INSTALL_OPTIMIZATION_COVERAGE_20261016.md).
- [Exact wheel manifest](dependency_optimization_20261015/wheelhouse-manifest-a2.json),
  [metadata closure](dependency_optimization_20261015/metadata-closure.json) and
  [pinned delivery contract](../configs/dependencies/rtx3090-cp310-cu121-screening.json).
- [Chronological progress/failures](dependency_install_optimization_progress.md)
  and [original plan](DEPENDENCY_INSTALL_OPTIMIZATION_PLAN_20261015.md).

Recompute the medians without network, pip or CUDA:

```bash
python3 -B scripts/summarize_dependency_optimization.py \
  --input docs/dependency_optimization_20261015/results.json \
  --output docs/dependency_optimization_20261015/timing-summary.json

python3 -B scripts/verify_dependency_optimization_evidence.py \
  --evidence-dir docs/dependency_optimization_20261015 \
  --output docs/dependency_optimization_20261015/evidence-audit.json
```

Source authority is the owned-worker receipts, not this prose or historical logs
from another host. The reporting workflow preserves timing boundaries, sample
sizes and negative results in the requested repository Markdown format; no
additional report website or external publication is created.
