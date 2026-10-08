# RTX 3090 r5 execution — in progress

This is a live evidence index, not a completed release claim. The protected RTX
4070 SUPER worker and production EC2 service have not been replaced or restarted.

## Current measured outcomes

| Measurement | Current result |
|---|---|
| GPU-path FPS | Portable r5: 278.288586320095 / 277.4908139113453 over separate ≥180 s windows; diagnostic, not full-recipe FPS |
| Full-recipe aggregate FPS | Portable six-avatar T is running; native release requires every T/SUST window ≥400 unrounded |
| Live delivery / capacity | Local smoke decoded 1,371 frames, but a 278 ms gap and missing telemetry prevent a strict PASS; no proven capacity yet |
| Request-to-usable-call startup | Still unmeasured through EC2/TURN. Request-to-verified-health was 1,046.45 s (17m26s), not usable-call readiness |

Instance A is Vast **54798270**, label `musetalk-r5-3090-dev-20261008`, created once
through the existing EC2 API. Source revision is
`dae1e88ad3587fddbebe6d41f85faa569001f6c1`, with explicit control-plane registration
disabled for isolation. The production worker **51074906** remains protected.

Observed timeline (UTC, October 8):

- 07:15:25.547: EC2 create request started.
- 07:15:25.986: provider acceptance returned (0.439 s client-observed interval).
- 07:18:52: provider log endpoint still reported no container.
- 07:19:32: provider reported running; this is not application readiness.
- 07:19:36: source bootstrap began.
- 07:20:34: first successful operator SSH; hardware verified.
- 07:24:17: checkout revision verified; dependency installation still downloading TensorRT 10.3 libraries.
- 07:29:04: model download phase started.
- 07:31:21–07:32:33: portable engine bundle restore (72 s).
- 07:32:52: canonical startup completed after 631 s; final server health verification took 19 s.
- 07:36:29: one production talking-pose warm request began; response after 23.82 s, including 21.77 s S3 restore and 1.43 s cache load.
- 07:43: owned API/TURN drained and stopped before isolated GPU work.
- 07:56:50: two three-minute portable GPU-path runs completed; no foreign GPU workload observed.

This separates several minutes of provider/image startup from source cloning,
installation, and later model/avatar warmup. Exact image-pull boundaries and
provider cache state are not known. The original health milestone was observed
without manual repair. Cross-host UTC clock skew has not been independently
calibrated; stage durations from individual processes are reported separately.

## Hardware and spending controls

- Actual GPU: NVIDIA GeForce RTX 3090, sm86, 24,576 MiB visible VRAM, driver
  595.91.07, 350 W configured power limit.
- Vast machine 153039; 150 GB instance disk, about 149 GB initially free;
  `/dev/shm` about 31 GB. Cgroup CPU quota is **18.432 CPUs**, not the 96 visible
  host CPU IDs. See the exact memory/CPU evidence below.
- User-approved total: **$30**, including experiment costs. Vast download and
  upload rates must each be no more than $1.50/TB.
- Selected provider rate: about $0.241111/hour including selected storage;
  conservative adjusted reservation uses $0.249249/hour. Provider reports
  $1.333333/TB transfer each way. First resource reservation is $5.584740.
- An EC2-owned experimental-only expiry timer is active for **19:00 UTC**,
  with bounded retries. It is best-effort software cleanup, not a provider-side
  hard cap. No production instance is an eligible cleanup target.
- AWS S3 request/storage/egress costs are separate from Vast traffic rates and
  must remain within the same $30 total. Do not assume AWS's account-wide free
  transfer allowance is available.

## Verified preparation

- CPU tests cover isolation, startup, harness reports, timing-cache provenance,
  Docker lifecycle and private-model delivery. Individual reports distinguish
  actual passes from platform/dependency skips; these do not establish GPU/image acceptance.
- 64 expected S3 objects passed fresh HEAD checks: 16 portraits and 48 pose
  caches. This proves availability/size/metadata, not archive or latent integrity.
- The exact pinned portable r5 and load-test audio archives were absent at their
  configured S3 keys. Their original known-good archives were recovered read-only
  from the protected worker, checksum-verified, and uploaded to those missing
  content-addressed keys. Fresh S3 downloads matched both hashes. The portable
  archive also passed a clean CPU restore with 17 verified files.
- Quality policy is frozen separately. Historical strict-gate failures remain
  failures. Portable-3090 repeated references/noise bounds, candidate quality,
  and actual visual inspection are still incomplete.
- Docker source has 50 passing CPU tests and one Linux-only skip on the operator;
  independent review is underway. No Docker image has been built
  or published; no Vast template has been edited or promoted.

## Evidence and unresolved work

- [Run state](run_state.json)
- [EC2 provisioning contract](provisioning/EC2_CONTRACT.md)
- [Instance A initial observations](provisioning/instance-a-initial-observations.json)
- [Original artifact recovery](provisioning/baseline_artifact_recovery.json)
- [Live S3 availability audit](avatars/s3_head_audit.json)
- [Quality acceptance policy](quality/quality_acceptance.json)
- [Startup optimization targets](startup/acceptance.json)
- [Source-install startup breakdown](startup/source_install_baseline.json)
- [Local smoke assessment](portable/source_install_smoke_assessment.json)
- [Sustained portable GPU-path evidence](portable/portable_v1_gpu/report.json)
- [Budget checkpoint](provisioning/budget-checkpoint-0748.json)

Registry publishing access remains unresolved. A read-only native x64 GitHub
runner probe succeeded with about 86 GiB free disk, 4 CPUs, 16 GB RAM, and Docker/
Buildx available; see [builder inventory](release/builder_inventory_summary.json).
The full image build and audited artifact delivery remain unvalidated.
Native engines, ≥400 FPS evidence, quality parity, production-pose render audit,
image publication, browser template edit, fresh Instance B, real EC2/TURN media,
repeated startup trials, and live soak/scale-in tests remain required. No overall
completion or release approval is implied by the CPU preparation.
