# Dependency optimization: requirement-by-requirement evidence audit

Final requirement audit, 2026-10-16 UTC. The gated experiment/report is complete;
conditional alternatives below are not represented as executed tests. Governing
scope is the [approved plan](DEPENDENCY_INSTALL_OPTIMIZATION_PLAN_20261015.md),
with tests/results in the [progress log](dependency_install_optimization_progress.md)
and conclusions in the [report](DEPENDENCY_INSTALL_OPTIMIZATION_REPORT_20261016.md).

## Evidence sets and comparison boundaries

- **A:** [first-host receipts](dependency_optimization_20261015/results.json),
  [recomputed medians](dependency_optimization_20261015/timing-summary.json),
  [wheel manifest](dependency_optimization_20261015/wheelhouse-manifest-a2.json),
  [closure](dependency_optimization_20261015/metadata-closure.json).
- **B:** [fresh different-host canary](dependency_optimization_20261015/canary-results.json),
  including accepted canonical install, two decoded outputs, later failed S3
  experiment/recoveries and verified teardown. Do not merge its times with A.
- **C:** [final-screen evidence](dependency_optimization_20261015/final-screen-results.json),
  isolated rental56218707. Full-payload overlap and copy/hardlink screens passed;
  33 files preserved, destruction verified07:57:27, final inventory07:59 UTC shows
  all owned workers absent and all expiry timers inactive.
- **S:** source/mock tests, not Linux performance or native acceptance.
  Run `python3 -B -m unittest test_dependency_artifacts test_dependency_startup_profile
  test_install_package_timing test_python_s3_venv_safety test_dependency_fresh_canary
  test_dependency_evidence_audit` (**135 passing tests**).

Dependency-ready, import/CUDA-validated, API-ready, first-output and full
create-to-ready are different boundaries. A's local-wheel tests exclude remote
acquisition; complete-delivery tests include it. B's package timer precedes its
later canonical imports/GPU checks. No fleet p95 or five-minute total promise.

## Work packages and numbered experiment matrix

| Requirement | Evidence inspected / result | Audit state |
|---|---|---|
| WP0 exact platform, ABI, pins, group closure and independent manifest trust | A manifest/closure; pinned contract; exact cu121/Torch-TensorRT artifact checks; full 194 packages | Proved for the tested contract, not cu128/other ABIs |
| WP0 distinct acquisition manifest and offline hashed locks | `dependency_artifacts.py`, wheelhouse build receipt and full offline installs; build inputs/toolchain retained | Proved |
| WP0 phase/cache/resource observability | A per-trial phases and 3,138 resource samples; actual bytes/retries; B per-artifact timing | Proved with disclosed missing early samples and no continuous peak-disk proof |
| B0 matched original source-pip control | A `delivery-a2-source-control-{1,2,3}`, fresh env/caches; 293.050/294.191/297.623s validated | Proved, three trials |
| B1 exact source selection + pip | A `delivery-a2-locked-pip-serial-1`, 265.828s validated | Screen passed; <20% gain, not promoted |
| B2 bounded 2/4-way acquisition | A delivery receipts, zero destination cache hits; 147.582s two-way screen, clean four-way median132.176s | Proved; two-way not separately promoted to repeats |
| WP1 limited overlap with other boot downloads | C serial104.506s/concurrent93.034s, full wheels plus same r5 archive, zero cache hits/trusted hashes | Tested; 10.98% one-pair gain below promotion gate, no full-boot saving claimed |
| B3 same local full wheelhouse + pip | A three compiled cases, validated median132.036s | Proved, remote acquisition excluded |
| B4 identical wheels + pinned compiled uv | A three compiled cases, median49.785s; inventory/imports/CUDA | Proved, remote acquisition excluded |
| B5 bytecode trade-off and first use | A no-compile pip/uv screens; application-a2 plus equal empty Numba caches in application-a3 | Proved; no-compile uv not the winner; shared cold first request remains ~25s |
| WP4 actual filesystem/link semantics | C copy48.502s/hardlink44.570s validated; actual sample nlink2→1 after supported clean; hashes/CUDA survive, full inventory/imports passed | Tested; 8.108% one-pair gain below promotion gate; copy retained |
| B6 full-delivery winner with repeats | A clean trials1/2/4 median132.176s, range131.177–153.327s; all4 retained median133.064s | Proved on A; no universal network guarantee |
| B7 restart reuse and repair | A followup, canonical integration and warm-onstart; ~0.4s reuse, 4.310s repair, 24.400s warm onstart | Proved within stated boundaries, not a fresh boot |
| B8 S3 single versus ranges and safe complete restoration | A repeated prefix pilot plus both full GETs/hash, audit/extract/new canonical interpreter/native/API/rollback; matched150.322s | Tested and rejected on time |
| B8 portability confirmation if considering S3 adoption | B full ranged GET/hash and native checks, 685.786s packages; failed TURN/API; original venv restored but API not recovered | **Failed**, no S3 adoption claim |
| B9 optional capability reduction/import audit | WP6 source audit in progress log; Kokoro/API and avatar-prep lazy routes; serialized TRT and ONNX paths retained | Conditional branch not activated; no removal or savings claimed |
| B10 final hardened installer on different fresh host | B hash-bound source overlay, canonical checks, CPU imports/GPU self-test, health and two decoded outputs | Proved functional portability, slower CDN path |
| One-worker canary observation/teardown | B create once; complete boot24m17s; failed first health wait retained; accepted second output check; absence verified07:37:29 | Proved; no production rollout |

## Acceptance gates, failure handling and invariants

| Gate | Evidence / boundary | Audit state |
|---|---|---|
| Exact inventory and trusted artifacts, including bootstrap and native GPU builds | A source artifact reports, installed-file check for two historical hashless entries, wheel hashes; B accepted full inventory | Proved for tested routes |
| Canonical state checker, imports and no new package conflicts | A/B native checks and pip consistency; S covers altered/missing contracts/marker inputs | Proved |
| CUDA execution, TensorRT and same prebuilt plan checksum/probe, no rebuild | A application logs built_now=False/probe=exact; B normal fresh r5 restore and strict backend check; historical TAESD quality gate remains unchanged | Proved for wheel winner; not B's failed S3 app |
| Codec/native VP8, audio and existing-avatar REST video | A VP8/native and decoded video/audio; 28 controlled outputs byte-identical and decoded SSIM1.0; B two decoded equal outputs | Proved within internal REST/tunnel scope |
| Comparable first imports, API and first usable output | A controlled application and equal empty-JIT-cache intervention | Proved on A; unexplained initial SSIM anomaly retained |
| Same-input throughput sanity, investigate >5% regression | A repeated warm-request medians differ<2%; no 400+FPS recertification | Proved for bounded sanity, not load certification |
| Same optional capability groups | All comparisons preserve 194-package full-stack inventory | Proved; no capability removal as a speedup |
| Interrupted/truncated/corrupt transport, bounded retries and expired credentials | A real Linux loopback faults and real one-second AWS GET403; S covers path/hash/closure/ABI/disk checks | Proved at stated fixture scope; not fault injection into provider infrastructure |
| Concurrent install, stale stamps, missing-package repair and changed requirements | A followup/canonical fixtures; S ownership/locks, changed fingerprints, wrong modes | Proved; no silent fallback |
| Canonical S3 rollback/failure evidence | A original env/API restored; B env restored but app recovery failed and was torn down | Mixed result retained; S3 not recommended |
| macOS restriction | Mac commands limited to source, stdlib/mock processing; all real package installs on owned Linux workers | Preserved |
| Secrets and production boundaries | Public exports credential-pattern checked; private GET tickets only stdin/memory; no default/template/autoscaler/IAM changes or Git push | Preserved; secret-bearing raw logs remain private |
| Ownership, quota, budgets and independent expiry | A/B/C provider-absence receipts; C final inventory owned_remaining=[] and all timers inactive; last quota20%<75% | Proved, no reset consumed |

## Explicit conditional alternatives, not invented test results

- Eight-way prefetch: not promoted; first-host four-way effective rate was already
  near its advertised link. No eight-way speedup/failure is asserted.
- Large individual S3 wheel mirror / wheelhouse archive / compression variants:
  not published or performance-tested. The winning original-CDN wheel route
  avoids new large storage; complete S3 venv restoration lost on both tested
  paths. Those observations do **not** prove a wheel mirror would lose.
- Removing Kokoro/avatar prep or inheriting base-image system-site-packages:
  would change capability/ABI contract. Audit retained them; no same-contract
  improvement is credited to their removal.
- Direct WebRTC/TURN, production registration, full concurrency and historical
  400+FPS recertification are not provided by internal REST/native smoke checks.
  They are not expanded rollout work under this dependency experiment.

## Completion decision

**Complete for the approved gated experiment/report scope.** All required
screens, promoted candidate repeats, application/first-use and fresh-host gates,
failure records, final source/mock verification and owned cleanup have current
evidence. The [offline audit output](dependency_optimization_20261015/evidence-audit.json)
binds the source datasets by SHA256 and checks their key coverage/calculations.
It intentionally fails if the second-host S3 failure is relabeled or cleanup is
missing. No production deployment is included. Conditional non-promoted branches
above remain untested, not retroactively marked passed or failed.
