# RTX 3090 r5 execution — in progress

This is a live evidence index, not a completed release claim. The protected RTX
4070 SUPER worker and production EC2 service have not been replaced or restarted.

## Current measured outcomes

| Measurement | Current result |
|---|---|
| GPU-path FPS | Portable r5: 278.288586320095 / 277.4908139113453 over separate ≥180 s windows; diagnostic, not full-recipe FPS |
| Full-recipe aggregate FPS | Portable six-avatar T: 256.67195270538247 / 258.85397057089455; valid FAIL against 300 target. Native release still requires every T/SUST window ≥400 unrounded |
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
- Native suite preflight then passed against all 878 frozen files. Sequential native quality/envelope, paired GPU diagnostics and full-recipe T/SUST are running; none is yet an acceptance result.

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

Registry publishing access remains unresolved. A read-only native x64 GitHub
runner probe succeeded with about 86 GiB free disk, 4 CPUs, 16 GB RAM, and Docker/
Buildx available; see [builder inventory](release/builder_inventory_summary.json).
The full image build and audited artifact delivery remain unvalidated.
The renewed [dependency CI](https://github.com/AhmadAFS1/MuseTalk/actions/runs/37769890795)
uses revision `b00cc20fa5b77c051e468d993c6522a4ae921f9e`, with the narrowly scoped
MMEngine optimizer-registration backport and an actual `mmcv.ops` import check.
Its result is still pending; the earlier failed run remains evidence.
Native engines, ≥400 FPS evidence, quality parity, production-pose render audit,
image publication, browser template edit, fresh Instance B, real EC2/TURN media,
repeated startup trials, and live soak/scale-in tests remain required. No overall
completion or release approval is implied by the CPU preparation.
