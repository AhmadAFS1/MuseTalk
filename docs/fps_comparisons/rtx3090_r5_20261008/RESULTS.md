# RTX 3090 r5 execution — in progress

This is a live evidence index, not a completed release claim. The protected RTX
4070 SUPER worker and production EC2 service have not been replaced or restarted.

## Current measured outcomes

| Measurement | Current result |
|---|---|
| GPU-path FPS | Native v1: 385.75668080919866 / 388.64419629013895; earlier portable: 278.288586320095 / 277.4908139113453. Separate ≥180 s windows; diagnostic, not full-recipe FPS |
| Full-recipe aggregate FPS | Native T: 252.59611538153948 / 230.39816210925395; SUST: 231.089–246.552. Recovered portable control T: 253.59753865499397 / 240.87736112161846. Valid failures against their unchanged 400 / 300 targets |
| Native quality | Rejected: 236 / 698 frozen bounds fail; full-frame TAESD max 7 LSB versus reference 5. No bounds widened |
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
- 08:05:08: portable full-recipe T completed; 34,560 frames per shared window of 134.64657760900445 / 133.51156995497877 seconds.
- 08:20:46 / 08:33:24: two portable quality runs completed. Exact source-prefix, batching, fused-post and repeat checks pass; original UNet max-error, TAESD max-LSB and landmark failures remain FAIL.
- 08:35:40.744: measured reference envelope frozen before native candidate evaluation.
- 08:38:54: native-build preflight failed before any engine build. NVML queries were observed temporarily blocked for about a minute; bounded runtime diagnosis is in progress, not a successful native build.
- 08:41:54: corrected bounded diagnostic reproduced CUDA driver initialization failure. No OOM; physical cause not established.
- 08:48: a 14GB repository backup completed on the rented host. Subsequent SSH routes became unreachable; planned recovery guard was not applied and no reboot was sent.
- 08:52:18: Vast reported machine153039/instance54798270 **offline**. Replacement offers were checked but no additional rental was purchased.
- 11:17:36: Vast again reported **running**; SSH and a CUDA tensor allocation succeeded on the original hostname and GPU UUID. Automatic startup had replaced the experiment checkout with the initial source revision and restarted API/TURN. The preserved experiment directory survived.
- 11:18–11:21: owned API/TURN drained/stopped; preserved checkout restored; builder dependencies restored on the same pinned matrix. A hostname-specific guard now suspends this development worker's automatic reinstall/autostart. No operator reboot was sent.
- 11:23: native preflight passed against the exact frozen 878-file manifest. Native sm86 v1 engine build started on revision `d4e78b79105e0c1f5b732fb651e5dcd7fcb5dca3`.
- 11:29: native FP16 `down0rest,up3,tail` build completed successfully after 324.1 seconds under its GPU lease. INT8 recipe blocks are building. Complete-chain, native TAESD, quality, and throughput remain unverified.
- 11:45: native INT8 `down1,down2,down3,mid` plan files exist; the builder remains active on subsequent blocks. This is partial build progress, not a complete or accepted engine set.
- 11:52:57: complete native UNet manifest finalized with all 11 blocks, graph/direct equality and repeat determinism. Probe output SHA is `23950ef8e4117aef4488a7e14e032450875c808837ef844151af30958b79908c`. Most exported ONNX hashes differ from the portable reference (tail matches); numerical parity is not inferred from successful compilation.
- 11:53:52: native TAESD completed, actual key `1e967e6e715c9f1a8375`, opt3, full-height bs8, hardware compatibility `none`; fused/repository post probe has zero mismatched bytes. Total observed native build interval was 1,823 seconds; this is development compilation, which must not recur at normal image boot.
- Native suite preflight then passed against all 878 frozen files, followed by sequential quality/envelope, paired GPU diagnostics and full-recipe T/SUST evaluation.
- Native v1 quality completed: original strict gates remain **FAIL**, and frozen-reference parity is **FAIL** with 236 of 698 bounds missed. TAESD full-frame max rose to **7 LSB** from portable **5 LSB**, so the candidate is rejected; no bounds were widened. Source-prefix and fused/repeat byte invariants pass. Actual native visual inspection and production-pose validation remain incomplete.
- Two native GPU-path runs completed with 69,440 / 69,968 valid frames over 180.00984416999927 / 180.0309915029993 seconds: **385.75668080919866 / 388.64419629013895 FPS**. Diagnostic validity passes, but this is below 400 and excludes the full composition/live workload.
- Native full-recipe T completed as valid **FAIL**: **252.59611538153948 / 230.39816210925395 FPS**. SUST5 completed as valid **FAIL**: **246.552153419512 / 231.08910512636766 / 233.40398117752895 / 231.26400993583042 / 231.96003877267862 FPS**. All windows contain 34,560 completed valid frames over their own shared wall times; no padding is counted.
- T had 1,888 / 2,470 partial jobs of 3,104 / 3,395 total; GPU credit waits consumed 11.8% / 14.2% of wall time. A software-thermal slowdown was separately observed during SUST. These are diagnostic signals, not proof of a single cause. Thread profiles differ only in hardware compatibility.
- Recovered portable control completed: 34,560 valid frames over 136.27892519499983 / 143.47550072400009 seconds, **253.59753865499397 / 240.87736112161846 FPS**, valid **FAIL** against 300. Native and portable full-pipeline ranges are close despite faster native GPU diagnostics. This is the same recovered host and frozen inputs, but not a fully interleaved full-pipeline experiment; elapsed thermal/host conditions remain confounders.
- Native serving-only diagnostic archive copied off-host, SHA256 `1f766487cf9272929988d17f9a4d6f76ee9c8c8fdde9b149174e1e02dbcafc5e`, 983,926,034 bytes. Fresh operator CPU restore verified all 16 payload files. It is explicitly rejected diagnostic evidence, not the active r5 bundle or a releasable artifact.
- The same archive was conditionally uploaded to a new private checksum-keyed S3 object, followed by an independent fresh GET and fresh CPU restore verifying its archive SHA and all 16 files. No existing object or active artifact mapping was overwritten.
- All 32 native quality capture/report files were separately archived, conditionally persisted in private S3, freshly downloaded and CPU-restore verified: SHA256 `079f79938d45d62dfe5316e84be984cd86d7c648ea324023bd34b55d87090bb9`, 274,657,275 bytes. Numerical rejection and partial visual-review limitations remain unchanged.
- Operator fallback conditionally uploaded all five private model objects after checking their exact frozen hashes. Worker-side fresh content verification then timed out at 300 seconds on its first file; delivery remains **incomplete**, not a boot-speed pass. The ongoing 48-pose audit had completed 21 objects without reported failures at the last checkpoint. Concurrent CPU/network work is disclosed; no single cause or steady-state transfer speed is established.

## Current bottleneck interpretation

The source bootstrap itself took 631 seconds and portable engine restore 72 seconds;
one avatar warm took 23.82 seconds, including 21.77 seconds in S3 restore. Building
native engines took 1,823 seconds in development and must never occur at normal
boot. These observations support an immutable prebuilt runtime plus checksum-pinned
engine/private-model delivery, not a blind snapshot containing credentials and caches.
The actual dependency-only Docker build now passes, but its 19.66 GB uncompressed
size excludes model weights and plans. Compressed pull size, registry digest,
three fresh/restart distributions, and request-to-usable-call remain unmeasured.

For throughput, native T filled about 69.6% / 63.6% of padded batch slots. Its
per-job GPU service time was 37.87 / 36.56 ms; CPU FaceMesh and composition work,
batch formation and GPU credit waits are measured leads. Reported nested IPC and
FaceMesh wall times must not be added as independent costs. The thermal flag is
an observation, not a proven explanation; GPU memory temperature is unavailable.
No power/clock/fan changes were made. A rejected numerical candidate is not promoted
because its GPU kernels are faster. Existing releases and templates remain unchanged.

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
- A separate $2 AWS audit/transfer contingency was added under the same $30
  authorization before the 48-pose content download, retaining the original A1
  reservation. Combined reservation is $7.584740, not an invoice. The shared EC2
  ledger was amended under its lock with an exact previous-SHA guard and a backup.
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
- Quality policy and measured reference envelope are frozen separately. The two
  portable captures have identical canonical output pixels and all 594 per-avatar
  metrics; 104 explicitly registered UNet/TAESD metrics are also bounded, with
  per-metric observed repeat spread only. Historical references were checked for
  matching inputs, metric implementations and supporting artifact hashes. All
  original strict failures remain failures. Native parity and visual acceptance
  are still incomplete.
- Docker lifecycle/private-delivery changes passed independent review and CPU
  tests. The first dependency CI failed a root-only startup-test assumption;
  that test was corrected for non-root runners. Its successor passed Linux CPU
  and startup tests, then exposed a real MMEngine 0.10.4/PyTorch 2.5.1 Adafactor
  registration collision in `mmcv.ops`. That failure was reproduced on the GPU
  worker. A narrow upstream registry-name backport preserves the package pins;
  actual avatar-prep imports are now included in the installer smoke. The next
  [dependency CI run](https://github.com/AhmadAFS1/MuseTalk/actions/runs/37769890795)
  is in progress. No serving image has been built or published; no Vast template
  has been edited or promoted.
- The isolated live adapter requires full lifetime telemetry, matched first
  frames, complete send rings, and observed H264 payloads. Each load level has a
  separate invocation so a failed level stops escalation. These CPU contracts
  passed; native GPU/live validation remains outstanding.
- Provider finalized-charge access returned HTTP 401 at 08:21:58 UTC. Actual
  invoiced spending remains unavailable, not zero; reservation and expiry controls
  remain in force. No permissions were widened or billing request repeatedly retried.

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
- [Portable full-recipe T](portable/portable_v1_aggregate/report.json)
- [First portable quality reference](quality/portable_ref1_quality/report.json)
- [Second portable quality reference](quality/portable_ref2_quality/report.json)
- [Frozen measured quality envelope](quality/reference-envelope-v2.json)
- [Budget checkpoint](provisioning/budget-checkpoint-0748.json)
- [Finalized-charge availability check](provisioning/budget-checkpoint-0820.json)
- [Instance A CUDA failure and host outage](provisioning/instance-a-host-outage.json)
- [Actual portable still-frame review, with limitations](quality/portable_reference_visual_inspection.json)
- [Recovered pinned-input native preflight](native/native_v1_preflight_recovered_v2_check/report.json)
- [Complete native UNet manifest](native/native_v1_build/engine_manifest.json)
- [Native decoder metadata](native/native_v1_build/taesd_trt_1e967e6e715c9f1a8375.json)
- [Native candidate preflight](native/native_v1_check/report.json)
- [Native v1 rejection decision](quality/native_v1_quality_decision.json)
- [Native full-recipe T/SUST evidence](native/native_v1_aggregate/report.json)
- [Recovered portable control T](portable/portable_recovered_v1_aggregate/report.json)
- [Native partial direct still-frame inspection](quality/native_v1_partial_visual_inspection.json)
- [Private native diagnostic persistence and fresh restore](release/native_v1_diagnostic_persistence.json)
- [Private native capture persistence and fresh restore](release/native_v1_media_persistence.json)
- [Private-model fallback upload/read assessment](release/private_model_operator_fallback_assessment.json)

Registry publishing access remains unresolved. A read-only native x64 GitHub
runner probe succeeded with about 86 GiB free disk, 4 CPUs, 16 GB RAM, and Docker/
Buildx available; see [builder inventory](release/builder_inventory_summary.json).
The full image build and audited artifact delivery remain unvalidated.
The renewed [dependency CI](https://github.com/AhmadAFS1/MuseTalk/actions/runs/37769890795)
uses revision `b00cc20fa5b77c051e468d993c6522a4ae921f9e`, with the narrowly scoped
MMEngine optimizer-registration backport and an actual `mmcv.ops` import check.
It passed; the [retrieved evidence](release/dependency_ci_b00cc20/provenance.json)
has a matching 25,022-byte artifact archive digest. The dependency image is
19,659,561,663 bytes uncompressed before weights/native plans; compressed registry
transfer size and boot latency are unmeasured. This dependency-only image contains
no models, excludes Kokoro, was not GPU-tested or published, and is not promotable.
The earlier failed run remains evidence.
Accepted native engines, ≥400 FPS evidence, quality parity, production-pose render audit,
image publication, browser template edit, fresh Instance B, real EC2/TURN media,
repeated startup trials, and live soak/scale-in tests remain required. No overall
completion or release approval is implied by the CPU preparation.
