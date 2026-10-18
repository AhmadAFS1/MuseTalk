# Python S3 venv progress

## Final A3 outcome — publication and same-host restore succeeded; rental closed

Updated 2026-10-12 05:33 UTC / Oct 12 00:33 CDT. Evidence:
[A3 receipts](python_s3_venv_20261011/a3-receipts.json).

- **Usable private S3 venv archive exists:** 4,509,210,997 bytes, all 194 packages,
  trusted SHA-256 and adjacent manifest. Full path/hash are in the publication
  entry below. Final HEAD reconfirmed size/hash and AES256 SSE-S3.
- **Fresh S3 restore passed**, without pip: packages in **22m41.720s**, CUDA/TRT
  checks passed, healthy API in **23m35.966s**. GET alone was **21m03.912s at
  3.568 MB/s**. A3 source packages took **36m56s**; this sample saved 14m14s
  (38.55%) to package readiness. This is not five-minute startup.
- **Important variance:** A2 packages took only **16m34s**, faster than this S3
  restore. Consequently this does **not** establish that a single-GET archive is
  consistently faster than pip on this host. Matching A2/A3 package contracts
  and widely different install timings highlight network/resolver variability.
- Avatar warm passed in **62.172s**; generation status reported `completed` and
  `success: true`. Download returned **HTTP 404**, so no MP4/audio decode pass
  or usable video evidence was obtained. Source review shows a naming mismatch:
  [manager](../scripts/avatar_manager_parallel.py) supplies a random fallback to
  inference when `output_name` is omitted but reports `None.mp4` to the download
  route. The helper now supplies a unique output name; **mock-tested only**, not
  GPU-retested after this rental. Production API code was not changed.
- Test **55492021** destroyed; absence verified **05:27:22.651614 UTC**, after
  **1h53m07.531s**, before the original two-hour cutoff. Final recheck at
  05:30:31.894924 UTC confirms absence, protected RTX 4070S still running, zero
  pending multipart uploads under A3, archive retained, expiry inactive. Removed
  only the two A3-only EC2 SSH key files; original EC2 key was untouched.
- At the recorded quote, rental-only estimate is **$0.436 total / $0.514 adjusted**,
  excluding bandwidth/S3; not an invoice. One archive GET occurred, with transfer
  roughly $0.40 after shared free allowance (actual account allowance/bill unknown).
  No new AWS compute or custom Docker resources remain from this retry.
- **37 stdlib/mock tests pass**, shell syntax and whitespace checks pass. No Mac
  pip/CUDA install tests, Git commit/push, production template integration,
  second-host boot, WebRTC/TURN acceptance, or new FPS benchmark occurred.
- Optional parallel-range comparison was asked about but **not authorized or
  run**. Do not rent another GPU under the A3 authorization. Next decisions are a
  separately approved parallel-download comparison and slim-checkout/integrated
  installer work; keep ABI/path validation and record any pip fallback explicitly.


## 2026-10-12 05:24 UTC / Oct 12 00:24 CDT — final smoke uses shorter bounds

- First finish-service attempt started 05:22:04 UTC and safely refused its
  original 13-minute-reserve guard. No avatar warm or generation request ran.
- Live restored health and native checks already pass. Prepared a shorter optional
  REST smoke: **120 seconds maximum per avatar/generation phase**, with a seven-
  minute minimum remaining lease before starting. Existing request/video decode
  timeouts and cleanup `finally` remain. Independent expiry remains 05:34:15 UTC.
- No additional archive GET, rental, production callback or timer extension.
  Success is not claimed until a video/audio decode receipt exists; if the short
  smoke fails, preserve that failure and still destroy only this owned worker.

## 2026-10-12 05:19 UTC / Oct 12 00:19 CDT — restore accepted, live health confirmed

**The full private venv is published and its same-host fresh S3 restore passed.**
Fine-grained worker evidence was copied to EC2 before cleanup.

| Stage | Measured A3 time |
|---|---:|
| Fresh S3 GET, 4,509,210,997 bytes | **1,263.912s (21m03.912s), 3.568 decimal MB/s** |
| Fresh-download SHA-256 check | 20.807s |
| Extract and publish canonical venv path | 76.652s |
| Match all 194 packages/base interpreter | 0.347s |
| Removed venv → packages restored | **1,361.720s (22m41.720s)** |
| Native import/CUDA/TensorRT checks | 20.352s |
| Restored server start → healthy API | 33.893s |
| Removed venv → healthy API, including checks | **1,415.966s (23m35.966s)** |

- Exactly one fresh archive GET; the locally built tar was not reused. Trusted
  checksum matched. Previously audited identical bytes allowed audit reuse after
  checksum verification. No pip installation or TensorRT-plan build in restore.
- Compared with A3's 2,216s package phase, packages restored **854.280s (14m14.280s)
  sooner**, **38.55% less time / 1.627x speedup**. API readiness is a separate
  boundary; do not compare the table's subtotals by adding them twice.
- This is **not fast enough for the five-minute cold-start goal**. The single GET
  dominates; source clone alone took 6m45.620s. No fresh second-host boot or stock
  production-template integration has been validated. System/model/engine files
  and external caches stayed on the same host.
- The EC2 observer exceeded its 25-minute receipt-observation window while SSH
  status reads were intermittent. This was a bookkeeping/transport failure: the
  private detached receipt is `complete`, worker result is accepted, and a new
  live API probe returned HTTP 200. Reconciled the result without another GET.
- Running final optional REST video smoke under the remaining original lease;
  cleanup is guaranteed in its `finally` path and by the independent expiry.
  No optional parallel GET has been authorized or started.

## 2026-10-12 05:00 UTC / Oct 12 00:00 CDT — fresh GET is slow; validation pending

- HEAD independently confirms the full object and SHA metadata, with **AES256
  SSE-S3**. The original working venv path is absent; its recoverable backup is
  outside the canonical import path. This is a fresh S3 GET, not local tar reuse.
- At worker monotonic 701391.489271, downloaded bytes were **2,315,255,808**.
  Since venv removal at 700752.571724, that is **638.918 seconds**, about
  **3.624 decimal MB/s so far**. This is an in-progress interval including GET
  startup, not the final measured transfer rate. Download has not finished.
- This host's advertised 927.2 Mbps download does not describe the observed
  single-connection S3 rate. No S3 throttling/rate-limit cause is established.
- Asked whether, only if time remains, an additional four-range parallel GET
  comparison is desired on this same GPU (roughly another $0.40 transfer).
  **No second comparison is authorized or started without that reply.** The
  original 05:34:15 UTC cutoff remains unchanged; continue the approved first test.

## 2026-10-12 04:49 UTC / Oct 11 23:49 CDT — full archive published; fresh restore started

- **Publication succeeded** at 04:48:49.046530 UTC. All 135 S3 part sizes verified;
  CompleteMultipartUpload succeeded, and HEAD matched **4,509,210,997 bytes** and
  trusted SHA-256 metadata. A manifest is saved next to the archive.
- Private artifact: `s3://lingua-musetalk-s3-storage/python-venv/20261011/musetalk-python-s3-venv-20261011-a3/4cf134d4749f3fa5719a8df075c1f7b02b9f5a379feb7b4bea5eddcff2bd8599/venv.tar.zst`.
- Upload elapsed **773.883 seconds (12m53.883s)**, effective **5.827 decimal MB/s**
  including missing-part retries, transport observation and completion. This is
  upload performance, **not S3 download or cold-start performance**. The second
  batch repaired the 22 missing parts without retransmitting confirmed parts.
- Detached fresh restore launched at 04:48:51.923650 UTC. It moves the working
  venv to a recoverable backup, performs a new S3 GET rather than reusing the local
  built archive, verifies checksum, extracts to the canonical path, compares
  packages/interpreter, runs native checks and starts the test API.
- Restore timing and runtime acceptance are still pending. No stock-template
  production integration, fresh-host five-minute result, WebRTC or FPS claim yet.

## 2026-10-12 04:46 UTC / Oct 11 23:46 CDT — partial upload retained, missing parts retried

- First detached upload batch completed in **611.079 seconds**. Worker recorded
  113 successful parts and 22 failures classified `URLError`; no raw error URLs
  or signed tickets were logged. S3 receipts, not worker return status, determine
  which parts need retransmission. Existing confirmed bytes remain intact.
- EC2 signer uses IAM-role credentials expiring **10:38:40 UTC**, so the observed
  pause is not an expired publisher credential. Specific transport cause is not
  established by `URLError` alone. There was no public-key/SSH dependency in PUTs.
- Controller reconciles ListParts and retries only missing parts in the same
  multipart upload. Added safe reason-type/errno and HTTP-status fields for any
  subsequent part failures (no raw exception text). Original expiry is unchanged.
- Publication and fresh restore are still pending; do not treat partial parts as
  an installable S3 artifact.

## 2026-10-12 04:37 UTC / Oct 11 23:37 CDT — audited archive uploading

- Detached build completed and its manifest reached EC2 at 04:35:54.910903 UTC.
  Archive size: **4,509,210,997 bytes** (4.509 decimal GB); compression 31.474s.
  SHA-256: `d8a1fada0b0f59dc7c7df1c289e9101f2b215313f08895fd9ee92acb669195d6`.
- Contract fingerprint: `4cf134d4749f3fa5719a8df075c1f7b02b9f5a379feb7b4bea5eddcff2bd8599`.
  The 194-package/interpreter contract matches A2; archive bytes differ because
  this is a new snapshot. Source native validation passed in 22.768 seconds.
- Multipart identity was saved before launching one detached four-thread batch
  at 04:35:59.612261 UTC. EC2 uses S3 ListParts for authoritative receipts; signed
  tickets are sent over SSH/child stdin only, not saved as config files.
- Publication and restoration are still pending. No claim of a usable S3 archive
  is made until CompleteMultipartUpload and HEAD checks succeed.

## 2026-10-12 04:34 UTC / Oct 11 23:34 CDT — API health reached; archive build started

- Recovered API health observed at **04:33:14.315805 UTC**. UTC wall-clock
  difference from create request is **3,539.195 seconds (58m59.195s)**, including
  initial loading, clone, failed apt/DNS, manual recovery delay and successful
  canonical setup. This is not a clean/uninterrupted fast-start benchmark and
  not WebRTC/production-call acceptance. New observer has no original monotonic
  `START`, so this outer interval is explicitly a wall-clock calculation.
- One detached archive build launched at 04:33:17.236782 UTC. It verifies the
  installed environment and GPU, stops only the test API, archives the canonical
  venv path, audits members/links and hashes bytes before publication.
- No archive has been confirmed in S3 yet. The independent original expiry is
  still 05:34:15 UTC; approximately one hour remains for upload and restore.

## 2026-10-12 04:31 UTC / Oct 11 23:31 CDT — package groups complete; runtime verification

- Avatar-preparation dependencies: **70 seconds (1m10s)**, 04:26:42 → 04:27:52 UTC.
- Full Python-package phase: **2,216 seconds (36m56s)**. This equals core 1,999 +
  incremental Kokoro 147 + avatar preparation 70; do not double-count the subtotal.
- Model downloads finished; the avatar-preparation group took 97 seconds. CPU
  import smoke completed in 13 seconds; canonical GPU self-test is underway.
- Its logs include torch/Triton autotuning, separate from loading the prebuilt
  r5 TensorRT engine bundle. A Python dependency snapshot does not archive system
  libraries, model weights, r5 engines or external compiler caches.
- No venv archive or restore timing has been published yet; API health is pending.

## 2026-10-12 04:27 UTC / Oct 11 23:27 CDT — core and Kokoro timers completed

- Main server dependencies: **1,999 seconds (33m19s)**, 03:50:56 → 04:24:15 UTC.
- Incremental Kokoro dependencies: **147 seconds (2m27s)**, 04:24:15 → 04:26:42 UTC.
- Avatar preparation began 04:26:42 UTC; the bundled mmcv 2.1.0 wheel installed.
  That group and the overall package phase are not yet complete.
- These are successful A3 phase durations including DNS retries, resolver,
  downloads, builds and installation. They are not throughput-only measurements
  and do not overwrite the previous A1/A2 observations. Canonical setup continues
  after monitor handoff; there is still no published full venv or restore timing.

## 2026-10-12 04:24 UTC / Oct 11 23:24 CDT — setup monitoring handoff

- Main package log reached `Installing collected packages` after downloads/wheel
  building. Main timer has exceeded 32 minutes on this run; exact final duration
  remains pending. Do not substitute this for the previous attempt's timings.
- The initial detached setup wrapper had an incidental 2,200-second timeout,
  shorter than the approved lease. With the independent EC2 observer and expiry
  verified active, transferred supervision so remaining canonical setup can finish.
- Verified the exact private job's recorded supervisor PID, session identity,
  stdout file, and sole child argv (`bash scripts/vast_onstart.sh`) before signaling
  **only that wrapper supervisor**. The installer child was not signaled and
  remains running, now adopted by PID 1. A private handoff receipt records this.
- The original 05:34:15 UTC GPU expiry, read-only session lifetime, ownership and
  cost caps are unchanged. No package reinstall, checkout or instance replacement.
  Setup completion will be confirmed from canonical logs and API health, not the
  stopped wrapper's missing final receipt.

## 2026-10-12 04:15 UTC / Oct 11 23:15 CDT — original health window expired; worker retained

- Original source observer reached its 40-minute health limit. It recorded
  failure and preserved this same worker under its unchanged 05:34:15 UTC expiry.
  This is **not a successful fresh-start measurement**.
- The detached setup retry remains active. Latest package log is downloading
  `tensorrt_cu12_libs-10.3.0` (2,037.5 MB); core package installation is incomplete.
- Starting the prepared same-instance observer after verifying original service
  inactive, no archive build started, created ledger ownership, and expiry active.
  Observation is bounded to 40 minutes and reserves at least 30 minutes before
  original expiry for publication/restore. No extra rental or lifetime extension.

## 2026-10-12 04:08 UTC / Oct 11 23:08 CDT — same-instance observation recovery prepared

- Main pip work is still progressing through CUDA wheels; it has not finished.
  cuDNN's completed download log reports 664.8 MB in 3m23s (pip's rate display
  showed 4.7 MB/s). Small-wheel rate displays are not overall network benchmarks.
- Added a recovery mode for use **only after** the original baseline observer is
  inactive, before any archive build, and while the existing created ledger and
  original expiry still verify. It observes health on the same instance, then
  performs the archive experiment. It cannot call instance creation or extend
  the lease. This avoids losing the experiment if manual apt recovery makes
  startup exceed the baseline helper's original 40-minute observation window.
- Independently verified expiry runtime imports/API reads without inherited
  PYTHONPATH, active expiry timer, and protected production still running.

## 2026-10-12 03:50 UTC / Oct 11 22:50 CDT — canonical setup retried on same worker

- Container DNS configuration was already `1.1.1.1` / `8.8.8.8`; current getent
  resolves Ubuntu, PyPI and the S3 bucket, and direct DNS probes return answers.
  **No DNS file was modified.** The original apt failure was transient.
- Launched canonical `scripts/vast_onstart.sh` as a detached private job on the
  same owned worker, after confirming the venv path does not exist. Full stack,
  clean initial setup, throughput profile and separate core/Kokoro timing remain.
  Runtime read-only credentials are recovered in memory from PID 1; no EC2 role
  keys or persistent credentials are copied. Production callbacks remain excluded.
- No repeat checkout, instance replacement or lease extension. The failed apt
  interval and manual-recovery delay remain part of observed end-to-end A3 time,
  not hidden as a successful untouched fresh boot.

## 2026-10-12 03:56 UTC / Oct 11 22:56 CDT — dependency installation underway

- Setup retry launched at 03:50:08.966714 UTC. System packages completed in
  **29 seconds**, venv creation in **18 seconds**; main package timer began
  03:50:56 UTC. Kokoro and avatar-preparation steps have not yet completed.
- Main pip logs also show intermittent DNS retries for `download.pytorch.org`,
  followed by the Torch 2.5.1+cu121 wheel download starting. This is observed
  network unreliability, not a measured registry rate limit. No DNS file changed.
- Short reads confirm the detached setup job still running. Occasional SSH
  transport timeouts do not restart the job or create another instance.

## 2026-10-12 03:47 UTC / Oct 11 22:47 CDT — apt stopped before Python installation

- Checkout completed at 03:44:52.345329 UTC, **405.620 seconds** after its start.
  The first canonical install failed in system-package apt work after 81 seconds,
  exit 100, at 03:46:14 UTC. No Python environment was installed by this attempt.
- Logs show name-resolution failures for Ubuntu archive/security, NVIDIA and
  Launchpad repositories, not a package constraint or CUDA error. Preserve this
  failed baseline interval; any same-instance repair will be reported separately.
- Checking the test container's DNS and recovery options before retrying canonical
  setup. No new rental, production DNS change or deadline extension is authorized.

## 2026-10-12 03:44 UTC / Oct 11 22:44 CDT — SSH recovered, detached GPU probe passed

- Provider reported the container running at request +205.076 seconds; stock
  bootstrap began 03:38:06 UTC. Initial direct SSH rejected the existing test key.
  Reattachment reported the key already associated; both direct and proxy probes
  initially failed. No instance restart or replacement was performed.
- Added a new temporary A3-only SSH key after ownership/running checks; its private
  key is confined to the root-private EC2 evidence directory. The original EC2
  key was neither overwritten nor removed, and no account-wide key was registered.
- By 03:43:58 UTC both keys authenticated successfully. Causal attribution to
  delayed key propagation versus refresh is unproven. The active controller's
  original-key detached GPU probe completed; SSH readiness observed at +568.336s.
- Clone/install continues on the same instance. The original 05:34:15 UTC expiry
  remains active. No venv publication or restore timing is claimed yet.

## 2026-10-12 03:34:15 UTC / Oct 11 22:34:15 CDT — A3 rental accepted

- Created exactly one test instance **55492021** using stock source template
  `4dc0868ee56886c667c217d6c3634a58`, frozen main `5cc706e`, 128 GB disk.
  Create acceptance: 0.324 seconds. Current offer quote passed original caps.
- Durable ledger and active independent expiry timer were established before the
  create request. Maximum cutoff: **Oct 12 00:34:15 CDT / 05:34:15 UTC**, no extension.
  Read-only runtime session expires at the same time. Production `51074906` is
  protected and its configuration is unchanged.
- Attached existing EC2 SSH public key to this owned test instance only (API
  success verified). No account-wide SSH key or persistent worker credentials.
- First provider observation is `loading`. Installation, publication and S3
  restore results are still pending; the preflight artifact is not the full venv.

## 2026-10-12 03:34 UTC / Oct 11 22:34 CDT — pre-rental checks passed

- 36 stdlib/mock tests, shell syntax checks and `git diff --check` pass. No Mac
  packages installed. Remote `main` is still the pinned `5cc706e` revision.
- Real EC2 detached job completed **after** its launching SSH connection ended,
  with a deliberate 45-second delay before transfer. It uploaded two parts
  (5,242,880 and 31 bytes); S3 ListParts confirmed sizes, completion succeeded,
  and a fresh GET matched SHA-256. SSE-S3 reports AES256.
- Protected production `51074906` is the only instance in inventory and is running;
  no A2 controller service remains active. No third GPU was rented during preflight.
- Fresh eligible US quote: machine `136634`, host `399360`, 128 GB disk,
  10 effective cores, 32,224 MB RAM; $0.231111/hour total, $0.272778/hour adjusted.
  Both transfer directions are $0.0013020833/GB. Advertised download/upload:
  927.2/874.6 Mbps, not measured end-to-end S3 throughput.
- Proceed with one A3 rental, resolving the current equivalent offer slot before
  exactly one create request. Independent two-hour expiry is installed first.

## 2026-10-12 03:23 UTC / Oct 11 22:23 CDT — two-hour retry approved

- User approved one new retry capped at two hours, preserving $0.30/hour and
  $0.0015/GB in both Vast transfer directions. No third rental exists yet.
- Long worker jobs now detach from SSH. Each launch has a durable identity before
  SSH; uncertain launches are reconciled by that identity, not blindly resubmitted.
  Signed tickets are not saved in files or passed as process arguments.
- Multipart identity is saved before transfers. One detached four-thread uploader
  uses 32 MiB parts; S3 ListParts is authoritative. Transport failure no longer
  causes immediate abort. Expiry destroys the owned worker before multipart cleanup.
- New namespace: `musetalk-python-s3-venv-20261011-a3`. A1/A2 ledgers remain
  unchanged. Read-only worker sessions and expiry share the two-hour maximum.
- Before spending: stdlib/mock checks, real detached EC2 job across an SSH
  disconnect, and tiny signed multipart transfer with S3 receipts. No Mac pip
  installation, new archive or restore timing is claimed yet.

## 2026-10-11 — plan prepared, before rental

- User approved a one-hour RTX 3090 experiment and private S3 `python-venv/` upload.
- [Experiment plan](PYTHON_S3_VENV_EXPERIMENT_PLAN_20261011.md) written before any
  rental or upload. Docker remains abandoned.
- Working checkout: separate `codex/dependency-install-timing` worktree, main base
  `5cc706e90e50e93da1310628c025a84199cd8042`. Original dirty Mac checkout preserved.
- Core/Kokoro timing changes are local, tested with mocks/syntax only, not pushed.
- Source review: `download_weights.sh` still uses Hugging Face/Google Drive; S3
  engine/avatar storage does not mean all model weights use S3 in this boot path.
- No new instance or S3 venv artifact exists yet. EC2/preflight checks are next.

## Results — first attempt ended, S3 comparison incomplete

**Retry approved by user on 2026-10-11 local date; preparation began
2026-10-12 01:56 UTC.** First-attempt numbers below remain unchanged. Retry
results will be added separately, never substituted for the original receipts.

**Retry ended without publication:** test `55481655`, accepted `2026-10-12
02:05:38.907596 UTC`; original expiry `03:05:38 UTC`. Provider absence verified
at `03:05:39.767210 UTC`. The archive was built and audited on the GPU worker,
but S3 multipart completion failed before expiry. **No usable venv archive exists
in S3 and no restore benchmark was completed.** See the retry outcome below.

The source-install baseline reached `/health`, but the subsequent archive worker
command returned unsuccessful. The controller had not saved its return code or
stderr classification; the precise failure is **not established**. Automatic
cleanup destroyed the one test worker at `21:39:18.612893 UTC`.

| Stage | Measured duration |
|---|---:|
| Request → provider accepted | 0.274 s |
| Request → container reported running | 51.571 s |
| Full checkout + publication/timing patch | 405.784 s (6m45.784s) |
| Main Python dependencies | 989 s (16m29s) |
| Kokoro Python dependencies | 137 s (2m17s) |
| Avatar-preparation Python dependencies | 65 s (1m05s) |
| Full Python-package phase | **1,191 s (19m51s)** |
| Model weights, including Kokoro cache | 133 s (2m13s) |
| Canonical setup total | 1,442 s (24m02s) |
| Prebuilt r5 engine restore/check/extract | 96 s (1m36s) |
| Server start → health and verification | 33 s |
| Inner on-start total (excludes checkout/provider startup) | 1,580 s (26m20s) |
| EC2 request → observed API health | **2,070.312 s (34m30.312s)** |
| S3 venv upload / restore / restored API health | **Not completed or measured** |

- This machine advertised 923 Mbps down / 876 Mbps up. Individual pip downloads
  reported roughly 7–11 MB/s. Advertised bandwidth is not a measured S3 rate.
- No venv artifact or manifest was published. A post-cleanup S3 prefix listing
  contains only `preflight.json` (39 bytes). The prior 4.53 GB/$0.40 estimate is
  still an estimate, not an artifact measurement or cost incurred for a restore.
- No restored-runtime video, WebRTC test, 400 FPS benchmark or five-minute
  fresh-host result was obtained. Baseline health is not full call acceptance.
- Provider absence was independently rechecked after cleanup. The protected
  RTX 4070S `51074906` is still running. No second rental was submitted.
- At quoted $0.243333/hr total / $0.292161/hr adjusted, the ~34m53s rental is
  approximately $0.14–$0.17 before bandwidth, **not an invoice**. The existing
  r5 engine restore and source/model downloads have separate transfer costs.
- Evidence: [sanitized receipts](python_s3_venv_20261011/receipts.json). Raw
  redacted boot logs remain on EC2 under
  `/home/ec2-user/.local/state/musetalk-python-s3-venv-20261011/`.

## 2026-10-11 21:39–21:43 UTC — failure, cleanup and diagnostics repair

- API health observed at `21:38:55.682458 UTC`. Archive helper invocation failed
  at `21:39:18.040768 UTC`, before the controller recorded any worker JSON event
  or captured manifest. This does not prove whether the cause was SSH, Python
  launch, validation or another worker failure. Its raw output was not retained.
- Cleanup confirmed the test absent at `21:39:18.612893 UTC`, well before the
  `22:04:25 UTC` expiry. The timer was stopped after verified cleanup.
- Local helper repair now emits an acknowledgement before executing worker source,
  catches source-execution errors, persists return code/output byte counts and
  static error categories without storing raw stderr or signed URLs, and records
  command start/completion/timeout markers. This is an observability repair,
  **not a demonstrated fix for the unknown runtime failure**.
- 25 stdlib/mock tests pass, including wrapper-source syntax failure, transport
  failure evidence and credential/URL-safe classification. No macOS installs.
  Changes remain uncommitted/unpushed and were not deployed onto production.
- Requested permission for one additional bounded attempt. Until approved, no
  new instance is created and the previous ledger/evidence must not be reused.

## Follow-up — user requested command retry and failure explanation

- Rechecked provider inventory: first test `55446768` remains absent; protected
  production `51074906` is running. The actual archive operation cannot be
  retried on that terminated worker, and its lost local failure logs cannot be
  recovered through the retained controller evidence.
- Replayed the original deployed command wrapper on **EC2**, with a deliberately
  expired lease. It correctly returned one JSON `ValueError` (`Valid bounded
  lease required`), exit code 2, 109 stdout bytes and zero stderr bytes. This
  exits before GPU commands, writes or installs. It verifies wrapper parsing and
  error reporting on EC2, **not** archive validation on the deleted RTX 3090.
- Deployed worker source SHA-256 in that replay:
  `ca07badc1f57e9e8a963bd7d76045205486ca0c4586c0f81d95f14185b85e1df`.
- Original failure remains undiagnosed: no worker JSON was retained, and old
  controller logic discarded transport return-code/stderr details before always
  cleaning up. This was a diagnostics/experiment-lifecycle gap, not evidence
  that S3 refused the upload or that venv packaging is impossible.
- All 25 local stdlib/mock tests still pass. No new rental or macOS installation
  was performed for this follow-up.

## 2026-10-12 01:56–02:04 UTC — second attempt prepared, before rental

- User explicitly authorized a new retry and asked to upload the venv this time.
  New namespace: `musetalk-python-s3-venv-20261011-a2`; the first ledger, tools,
  receipts and permission-check object are preserved. No second create yet.
- Initial Virginia quote was for less disk than needed. At 150 GB its adjusted
  estimate exceeded the cap; it was not rented and later disappeared. A fresh
  128 GB search found verified machine `136634`, host `399360`, offer `48396264`:
  total $0.231111/hr, adjusted $0.272778/hr, both transfer directions
  $0.0013020833333333333/GB, 10 effective cores, 32,224 MB RAM, 927.2/874.6 Mbps
  advertised down/up. The generic provider location is `United States, US`.
- Retry uses 128 GB instead of 150 GB because host free disk is 137.1375 GB.
  No production template or autoscaler default is changed. Actual disk free
  space will still be checked before environment replacement.
- Added a read-only helper probe on first successful SSH, before the long
  install; attach only the existing EC2 public key to the ownership-verified test
  instance. Command timeouts and return-code diagnostics are saved without raw
  stderr or signed URLs.
- On a helper/baseline failure, collect sanitized check logs and preserve this
  **same** instance until the original one-hour timer. Successful runs clean up
  immediately. Failure does not extend the lease or trigger another rental.
- All 28 local stdlib/mock checks pass; main still points to pinned revision
  `5cc706e90e50e93da1310628c025a84199cd8042`. No pip/CUDA installs on macOS.

## 2026-10-12 02:05 UTC — retry created and one-hour timer active

- Exactly one second-attempt create accepted: instance `55481655`, label
  `musetalk-python-s3-venv-20261011-a2`, offer `48396264`, machine `136634`.
  Request-to-acceptance: 0.350 seconds. Both main service and independent expiry
  timer are active. Hard deadline: `03:05:38 UTC`, set before submission.
- 128 GB allocated disk; accepted total/adjusted quotes match the preflight
  $0.231111/$0.272778 per hour. Original price and bandwidth caps are unchanged.
- Test-only SSH public-key attachment returned HTTP 200 at `02:05:39.073807 UTC`,
  after verifying ownership. No private key/account-wide SSH setting was changed.
- 30 local mocked checks pass, including multipart upload range handling and
  restore disk budget. The updated wrapper was exercised on EC2 with an expired
  lease before rental, confirming source parsing and error reporting without GPU
  commands/installations. Early actual-GPU helper probe remains pending.

## 2026-10-12 02:07 UTC — early SSH/helper probe passes on the actual worker

- Provider container-running observation: +51.853 seconds. Actual helper probe
  completed over SSH at +90.389 seconds with exit code 0, parsed JSON and exact
  RTX 3090 verification. No archive directory writes or installations in probe.
- This confirms the updated command can launch on the new GPU worker; it does
  not identify the previous failure or prove archive/validation success yet.
- Full source clone is in progress. First-attempt logs/ledger remain untouched.

## 2026-10-12 02:11–02:12 UTC — checkout complete; main packages start

- Checkout markers: `02:06:56.237130441` → `02:11:00.440346946 UTC`,
  **264.203 seconds (4m24.203s)** including publication/timing patch. This is
  faster than the first attempt's 405.784 seconds, but not a pure network timer.
- Source revision matches the same pinned main SHA. Apt: 28 seconds; venv
  creation: 16 seconds. Main/server packages began at `02:11:45 UTC`.
- Actual helper probe interpreter: Python 3.10.12, Linux x86_64, glibc 2.35.
  API health, built archive, publication and S3 restore remain pending.

## 2026-10-12 02:19 UTC — TensorRT library download; cost rates checked

- Main install is downloading the 2,037.5 MB TensorRT 10.3.0 library wheel.
  This is package installation, not a TensorRT engine build. No archive yet.
- Read the official AWS regional price-list feeds (read-only):
  [S3 us-east-1 feed](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonS3/current/us-east-1/index.json)
  published `2026-09-28`, and
  [internet-transfer feed](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AWSDataTransfer/current/us-east-1/index.json)
  published `2026-09-16`. First storage tier: $0.023/GB-month; internet outbound
  first paid tier: $0.09/GB after the shared global 100 GB allowance. The S3 feed's
  zero-priced `S3-DT-AWS Outbound` product is not the external internet rate.
- Final archive cost estimates will use its actual byte size. Account-wide free
  allowance consumption and the actual invoice have not been inspected.

## 2026-10-12 02:25 UTC — main packages finish; signed multipart transport passes

- Main/server packages: `02:11:45` → `02:25:21 UTC`, **816 seconds (13m36s)**,
  versus 989 seconds on attempt one. Groups/pins are unchanged; transfer and
  resolution variability means this is not evidence of a code optimization.
- Kokoro installation began as a separate step. Its duration and full-package
  total are pending.
- EC2 completed a real presigned HTTPS multipart PUT, completion and HEAD
  verification of a **68-byte** private test object under the retry prefix:
  `presigned-multipart-probe.bin`. Encryption: SSE-S3 `AES256`. No URL was printed
  or persisted. This tests the signing/PUT path, not large-file Vast throughput.
- The full venv archive is still pending; this small object is not the deliverable.

## 2026-10-12 02:27 UTC — Kokoro packages finish

- Kokoro pip step: `02:25:21` → `02:27:13 UTC`, **112 seconds (1m52s)**.
  Main plus Kokoro: **928 seconds (15m28s)**. Kokoro model-cache downloads remain
  separate in the weights phase. Avatar-preparation packages are installing next.

## 2026-10-12 02:28 UTC — full Python-package baseline complete

| Retry step | Measured duration |
|---|---:|
| Main/server packages | 816 s (13m36s) |
| Kokoro packages | 112 s (1m52s) |
| Avatar-preparation packages | 66 s (1m06s) |
| Full Python-package phase | **994 s (16m34s)** |

- Full phase finished at `02:28:19 UTC`. Model downloads follow. Archive and
  restore results are still pending; no S3 speedup is claimed from this baseline.

## 2026-10-12 02:29–02:32 UTC — weights and runtime checks complete

- Weights: `02:28:19` → `02:29:37 UTC`, **78 seconds (1m18s)**, including
  optional Kokoro cache. Canonical verify phase: **76 seconds (1m16s)**.
  Final installer check completed at `02:30:57 UTC`.
- Prebuilt r5 bundle restoration followed; API process has spawned. Awaiting
  `/health` before archive build. No TensorRT engine rebuild was performed.

## 2026-10-12 02:33–02:41 UTC — archive operation recovered after SSH disconnect

- Archive validation passed: canonical check/imports 10.775 seconds, GPU self-test
  12.641 seconds, CUDA and TensorRT checks true. API stop returned 0 in 8.335 seconds.
- Archive helper's SSH invocation returned **255**, with JSON progress through
  the start of `tar`; the controller therefore preserved the worker instead of
  destroying its evidence. S3 publication had not begun.
- A read-only reconnect at `02:35:30 UTC` found a 4,509,211,440-byte archive and a
  durable successful tar completion: **34.291 seconds, exit 0**. The original
  helper lost its output channel before recording the final manifest. Cgroup
  memory events showed zero recorded OOM/OOM-kill counters; this is not a full
  host-kernel OOM audit. No tar/zstd/old helper process remained.
- Added SSH keepalives (5-second interval), durable event printing that survives
  a broken output pipe, and a finalize path requiring the existing successful
  tar receipt. A reconnect/drop during a silent operation is observed; an idle
  transport timeout is a hypothesis, not proven as the original attempt's cause.
- Recovery service started on **the same instance**, with the same `03:05:38 UTC`
  expiry. It revalidates the environment, audits/decompresses the existing archive,
  hashes it, creates its manifest, reverifies bytes/packages, then uploads/restores.
  No compression rebuild, lease extension or new rental is requested by recovery.
- 31 mocked safety checks pass. S3 upload and restore are still pending.

## 2026-10-12 02:41–02:55 UTC — durable manifest recovered; publication resumed

- Another SSH disconnect occurred during finalization, despite keepalives. The
  helper continued successfully after its output pipe closed because durable
  logging was repaired. A short reconnect retrieved the completed manifest.
- Archive safety audit: **70,873 members, 70.693 seconds**. SHA-256:
  `cb3ff3402393debda89ff9e1db060c917ebf9649d7c1bea9be078ba62e3a538f`.
  Compressed bytes: **4,509,211,440**; payload file bytes: **9,883,578,602**;
  installed distributions: **194**. All three dependency groups are included.
- Added bounded retries for early read-only/idempotent transport failures and an
  operation lock preventing simultaneous build/finalize/restore. Keepalives alone
  did not eliminate the disconnects; exact transport root cause remains unknown.
- Recovery now starts publication from the completed manifest, rather than
  budgeting another full build. The same original expiry remains authoritative.
  Restore is attempted only with ≥6 minutes remaining and its timeout is reduced
  to remaining lease minus 40 seconds. No extra rental or deadline extension.
- A fresh S3 GET will be hashed. If its SHA-256 matches this exact same worker's
  audited/built archive, that safety audit is reused instead of decompressing it
  a second time just for audit; extraction/package/GPU checks still run. This is
  an experiment-local audited-byte reuse, not trusting arbitrary S3 metadata.
- 32 mocked tests pass. Publication service started at ~`02:55:29 UTC`; actual
  S3 completion and restore measurements are not yet confirmed.

## Retry outcome — 2026-10-12 03:05–03:15 UTC

| Retry measurement | Result |
|---|---:|
| Request → observed API health | 1,625.311 s (27m05.311s) |
| Inner source on-start total, excludes checkout/provider startup | 1,300 s (21m40s) |
| Canonical setup | 1,196 s (19m56s) |
| Main / Kokoro / avatar-prep packages | 816 / 112 / 66 s |
| Full Python-package phase | 994 s (16m34s) |
| Prebuilt r5 engine restore/check/extract | 57 s |
| Server start → health/verification | 38 s |
| Built archive | 4,509,211,440 bytes, 194 distributions |
| Tar/zstd compression | 34.291 s |
| Archive safety audit | 70.693 s, 70,873 members |
| Full S3 archive publication | **FAILED / not present** |
| Fresh S3 restore and restored API health | **NOT MEASURED** |

- Multipart upload initiated at `02:56:03 UTC`. At `02:59:57 UTC`, S3 confirmed
  parts 1–19 of 34: 2,550,136,832 bytes. These are partial-upload bytes, not a
  usable object or full-upload speed measurement. SSH-driven part calls repeatedly
  returned 255 while remote HTTP uploads could continue, producing unnecessary
  retries and an unreliable completion path.
- Suspended, then terminated only this experiment's publisher service to recover
  server-confirmed receipts without its abort path racing manual completion.
  A manual same-upload resume at `03:04:34.890970 UTC` found 19 confirmed parts
  and attempted the remaining 15. It could not obtain all 34 before the original
  lease's final safety margin at `03:05:27.133389 UTC`; it aborted the upload.
- Independent expiry destroyed only RTX 3090 `55481655`. Provider absence and
  production RTX 4070S `51074906` running status were rechecked. The GPU's local
  temporary archive is lost with that destroyed container; its manifest and
  checksums are preserved on EC2/in the local receipts.
- One residual incomplete multipart upload remained visible after the initial
  abort, so it was aborted again after the GPU was gone. Rechecked **zero remaining
  uploads** for this exact owned key. No pre-existing user objects were deleted.
- S3 retains only the small probes plus
  `python-venv/20261011/musetalk-python-s3-venv-20261011-a2/diagnostics/environment-manifest-only.json`.
  That diagnostic explicitly says `archive_published: false`; it is **not** an
  installable venv or a claim of completed publication.
- Evidence: [retry receipts](python_s3_venv_20261011/retry-receipts.json), including
  the complete package/version manifest. All 32 local mocked tests pass; no Mac
  package installs, production changes, Git commit/push or third rental occurred.

### What must change before another paid attempt

The transfer architecture was inadequate: long work and return receipts depended
on SSH sessions that repeatedly closed. Keepalives/retries did not solve it and
consumed the one-hour window. A further attempt must use a **detached Linux transfer
job**, persist the multipart upload ID/progress on EC2 before sending data, poll
durable results over short connections, and adopt S3-confirmed parts instead of
resending them. Test detached execution/receipt retrieval with a tiny object before
renting. This architecture has **not yet been implemented or validated**.

No startup-template integration or five-minute fresh-host claim is justified by
these attempts. A new or longer paid test requires fresh user authorization; do
not extend leases, reuse cleaned ledgers or automatically create another worker.

## 2026-10-11 21:35 UTC — weights complete; runtime validation next

- Weight phase: `21:33:06` → `21:35:19 UTC`, **133 seconds (2m13s)**.
  Parallel weight-group downloads consumed 130 seconds after helper preparation.
- Native VP8 provisioning/checking and canonical validation follow. Baseline
  API health, archive publication and S3 restore have not yet been confirmed.

## 2026-10-11 21:33 UTC — full Python-package baseline recorded

| Step | Measured duration |
|---|---:|
| Main/server packages | 989 s (16m29s) |
| Kokoro packages | 137 s (2m17s) |
| Avatar-preparation stack | 65 s (1m05s) |
| Total Python-package phase | **1,191 s (19m51s)** |

- The full package phase finished at `21:33:06 UTC`. Model downloads have begun
  through the unchanged source installer, not through this dependency archive.

## 2026-10-11 21:32 UTC — Kokoro packages timed

- Kokoro pip step: `21:29:44` → `21:32:01 UTC`, **2m17s (137 seconds)**.
  Main plus Kokoro package installation totals **18m46s (1,126 seconds)**.
- Avatar-preparation packages are installing next; they are not included in
  the preceding two timers. Kokoro model-cache/weight downloads are also outside
  its pip timer and will remain in the weights phase.

## 2026-10-11 21:30 UTC — main dependencies timed; Kokoro begins

- Main/server pip step completed at `21:29:44 UTC`, started `21:13:15 UTC`:
  **16m29s (989 seconds)**. This includes resolution, downloads, wheel building
  and installation; it is not a pure network transfer timer.
- Kokoro now runs as its own step against the same constraints. Its final time,
  avatar-prep time and total Python-package time remain pending.
- Advertised download was 923 Mbps; observed individual large-wheel transfers
  were roughly 7–11 decimal MB/s. Neither is the forthcoming measured S3 rate.

## 2026-10-11 21:28 UTC — main downloads complete; package installation running

- The ~2,037.5 MB TensorRT library wheel completed; pip has reached `Installing
  collected packages` for the main group. At the latest snapshot the main step
  has run ~15 minutes but has not yet printed its final duration. Kokoro remains
  separately timed and has not begun.
- Re-ran 22 local stdlib/mock tests successfully. No dependency installation,
  GPU testing or environment construction was performed on macOS.
- Upload/restore remain pending. Independent expiry is still `22:04:25 UTC`;
  the controller will not begin venv replacement without its reserved time margin.

## 2026-10-11 21:19–21:20 UTC — main-package download observations

- Main installation remains in progress, without a logged failure. The 664.8 MB
  cuDNN wheel completed at a pip-reported 7.9 MB/s in 80 seconds; the 196.0 MB
  cuSPARSE wheel reported 8.8 MB/s in 22 seconds. These are individual downloads,
  not whole-install averages and not measured S3 speed.
- The one-hour expiry remains `22:04:25 UTC`. No second rental/create request,
  archive publication, destructive venv replacement or restore has occurred.

## 2026-10-11 21:17 UTC — checkout and security observations

- Exact checkout markers: `21:05:45.223993433` → `21:12:31.008131685 UTC`:
  **405.784 seconds (6m45.784s)**. This includes clone plus checkout publication
  and the timing-patch application before the ready marker, not just wire transfer.
- Apt phase: 28 seconds; venv creation: 15 seconds. Main package installation
  started at `21:13:15 UTC` and is still running. Kokoro has not started yet.
  One completed 14.1 MB CUDA-library wheel reported **10.9 MB/s** in pip's log;
  this is a single wheel's observed transfer, not a general link/S3 benchmark.
- Read-only bucket checks show all four S3 public-access blocks enabled and
  default SSE-S3 (`AES256`) encryption. No ACL, bucket policy or encryption
  configuration was changed. The future archive stays in this existing bucket.

## 2026-10-11 21:12 UTC — pinned checkout ready; normal installer starts

- Worker snapshot now verifies main revision `5cc706e90e50e93da1310628c025a84199cd8042`.
  The full checkout took more than six minutes from the logged clone start; final
  exact duration will be computed from the checkout markers in saved evidence.
- Canonical installer has begun system-package setup. This baseline keeps full
  stack and Kokoro enabled; no environment-restore shortcut is used yet.
- A fast dependency restore alone cannot establish a five-minute fresh boot with
  this full clone. Slim/partial checkout remains a separate next optimization.

## 2026-10-11 21:09–21:10 UTC — SSH succeeds; multipart permission verified

- SSH became observable at +254.249 seconds, after manual public-key attachment;
  this timing includes the access repair and must not be treated as intrinsic
  container startup. A read-only worker snapshot succeeded.
- Bootstrap began at `21:05:45 UTC` and clone began at `21:05:45.223993433 UTC`.
  At the latest ~21:10 snapshot the clone was still in progress; its temporary
  Git pack occupied 2,068,787,199 bytes. Pack size is not an internet speed test.
- Independently verified S3 multipart-create **and abort** permission on an empty
  test upload under the experiment prefix. It was aborted successfully; no
  completed object or uploaded parts were left by this check.
- Dependency timing has not started yet. Archive build/restore remain pending.

## 2026-10-11 21:05–21:08 UTC — container running; SSH key attached to test only

- Provider reports the container running at +51.571 seconds. SSH initially returned
  `Permission denied (publickey)`, so this was an observation-access delay, not
  evidence of installation being stalled or complete.
- Attached the existing EC2 public key **only to test instance `55446768`**, using
  the official per-instance [SSH attachment API](https://docs.vast.ai/api-reference/instances/attach-ssh-key).
  The first request returned a same-origin trailing-slash redirect (no attachment);
  the validated redirected request succeeded with HTTP 200 at `21:08:21.622049 UTC`.
  No private key, production-worker key or account-wide SSH setting was changed.
- One provider inventory read hit HTTP 429; the existing controller backed off and
  did not retry creation. Continue collecting bootstrap logs once SSH propagation
  completes. The independent expiry remains active.

## 2026-10-11 21:04 UTC — test instance created; expiry verified active

- First controller invocation failed during offer selection, **before any create
  intent/ledger existed**. A fresh search returned changing slot IDs on the same
  machine (`48396261`, `48396258`, then `48396264`). Updated only this experiment's
  library snapshot to resolve the currently returned equivalent slot ID before
  exactly one create request; machine/host, verified status and original caps
  still must match. The older baseline helpers/ledgers were not modified.
- Create request accepted at `21:04:25.644958 UTC`: test instance `55446768`,
  offer `48396264`, machine `136634`, host `399360`. Provider acceptance took
  0.274 seconds. This is **not** SSH/API readiness.
- Independent EC2 expiry timer was created and verified active **before** submitting
  the create request. Durable cutoff: `22:04:25 UTC` (5:04 PM America/Chicago).
  The only protected pre-existing instance is running RTX 4070S `51074906`.
- Source/template frozen at main `5cc706e90e50e93da1310628c025a84199cd8042` and stock
  template `4dc0868ee56886c667c217d6c3634a58`, with core/Kokoro timing-only patch.
  22 local stdlib-only mocked checks passed before deployment; no Mac installs.
- Baseline install is running. No S3 dependency archive or restore timing yet.

## 2026-10-11 21:02 UTC — replacement within original caps found

- Fresh read-only search found that approved Kentucky offer `35170923` and its
  machine were no longer returned. No create request was submitted for it.
- A new offer on the earlier eligible US machine is now available: `48396261`,
  machine `136634`, host `399360`, verified, generic `United States, US` location.
  Quotes with 150 GB: $0.243333/hr total, $0.292161/hr adjusted, both bandwidth
  directions $0.0013020833333333333/GB. 10 effective cores, 32,224 MB RAM,
  923 Mbps advertised down / 876 Mbps up. It meets the original stricter caps;
  the Kentucky bandwidth exception need not be used.
- Updated the fresh-run selector to bind to this exact replacement's ID/machine/
  host and original caps. It is not an authorization for arbitrary other offers.

## 2026-10-11 21:00 UTC — Kentucky quote exception approved

- User approved proceeding with the quoted Kentucky offer despite its higher
  bandwidth rates. The helper now binds approval to offer `35170923`, machine
  `32797`, host `198030`, verified Kentucky US RTX 3090, rather than broadly
  relaxing constraints for arbitrary offers.
- Fresh total quote must remain ≤ $0.264/hr and adjusted quote ≤ $0.410/hr;
  download ≤ $0.00390625/GB and upload ≤ $0.005208333333333333/GB. One-hour
  cutoff, 150 GB allocated disk, protected production IDs and exactly-one-create
  rule remain unchanged. Revalidate availability before submitting a request.

## 2026-10-11 20:28–20:29 UTC — prefix upload verified; offer availability gate

- After the user's policy-creation confirmation, EC2's `linguaEc2role` successfully
  uploaded and HEAD-verified a 112-byte preflight object at
  `python-venv/20261011/musetalk-python-s3-venv-20261011/preflight.json`.
  Upload permission is now operational. Reading the inline policy document through
  this EC2 identity still returns AccessDenied; this check does not claim to audit
  all IAM permissions or prove that writes outside the prefix are denied.
- Inventory still contains only protected production RTX 4070S `51074906`, running.
  No new RTX 3090 was created. Three read-only searches were used: the original
  capped query, a broader 65-offer query to diagnose exclusions, and a North American
  comparison. No US or North American offer met all current requirements.
- The previously eligible US offer `48396264` was absent from current results.
  Current US verified candidates exceed the $0.0015/GB bandwidth limit. The viable
  Kentucky candidate `35170923` (machine `32797`, host `198030`) quotes $0.263333/hr
  total with 150 GB allocated storage, and $0.409818/hr adjusted. It has 24 effective
  CPU cores, 48,068 MB RAM, 788.8 Mbps advertised download and 785.2 Mbps upload.
  Transfer quotes: $0.00390625/GB download and $0.005208333/GB upload (approximately
  $3.91/$5.21 per decimal TB). These exceed prior limits; **not rented**.
- Request a user decision before relaxing bandwidth/adjusted-price caps. The one-hour
  limit, stock image, isolated ownership and production protection remain unchanged.
  No archive, restore timings, GPU test or WebRTC results have been produced yet.

## 2026-10-11 20:13 UTC — signed in, browser controls unavailable; test tooling draft

- The AWS console is signed in again and shows the correct `linguaEc2role` with
  nine existing permissions policies. Native clicks still fail with
  `noWindowsAvailable`; the new publisher policy is **not attached**. Asked the
  user to open Add permissions → Create inline policy → JSON manually.
- Drafted isolated EC2/worker experiment helpers and the dependency-timing patch.
  The controller reuses the reviewed stock-source baseline ownership checks,
  protects all pre-existing instance IDs and installs an independent expiry timer.
  Upload and restore use object-specific signed S3 URLs; EC2 role keys stay on EC2.
- Added archive-member/link checks, external `.pth` dependency rejection, lease
  checks, recovery by venv rename, and lease-time gating for optional REST video.
  These helpers still need mocked checks and Linux validation before any claim of
  successful operation. They are not production installer integration.
- **No GPU rented, no archive uploaded, no restoration timed, no Mac pip test.**

## 2026-10-11 20:15 UTC — local mocked safeguards pass

- Ran 21 stdlib-only tests: 12 existing core/Kokoro timing tests plus 9 new
  archive-link/path, signed-destination, SSH endpoint, lease-expiry and worker
  receipt tests. All passed. No pip installation, AWS call or rental occurs in
  these tests; Linux execution and S3 restoration remain unverified.
- Browser input remains blocked, so the narrow IAM JSON is ready for manual
  creation by the user. Spending remains gated on a successful EC2 upload check.

## 2026-10-11 19:59 UTC — publisher grant approved; CLI authority check

- User explicitly approved prefix-only upload access on the existing EC2 role.
- Exact role remains `linguaEc2role`; policy name proposed:
  `MuseTalkPythonVenvPrefixPublish-20261011`.
- EC2's current role cannot call `iam:PutRolePolicy` (AccessDenied). The approved
  policy has **not** been attached. Checking the existing local AWS session/UI is
  the next safe alternative; no self-escalation or broad policy is requested.
- Saved the exact narrow [policy JSON](python-venv-s3-publisher-policy.json).
- Still no GPU rented, upload completed, or restore timing measured.

## 2026-10-11 20:01 UTC — console authentication required

- EC2 and the existing local `lingua-backend-user` CLI identity both returned
  AccessDenied for `iam:PutRolePolicy`. Neither attempt attached the policy.
- Located the already-open browser IAM tab for the correct role. Before policy
  creation, AWS displayed **"You have been signed out"** and reported that sessions
  in all AWS tabs had expired. Browser interaction was also interrupted by active
  user navigation; no changes were made to unrelated pages.
- User must reauthenticate the AWS console before the approved narrow grant can be
  applied. Do not issue new persistent credentials or broaden CLI permissions to
  bypass this. No IAM change, S3 artifact upload or GPU rental has occurred.

## 2026-10-11 19:40–19:44 UTC — EC2 and S3 preflight

- SSH to the existing EC2 controller succeeded. Its role is `linguaEc2role` (exact
  capitalization). No applicable remote AGENTS instruction file was found in the
  inspected ancestor paths. EC2 production services were not changed.
- Vast inventory contained only the protected running RTX 4070S (`51074906`);
  **no new RTX 3090 has been rented**.
- One eligible verified US offer was returned: `48396264`, machine `136634`, host
  `399360`, 10 effective CPU cores, 32,224 MB RAM, 150 GB allocated disk. Advertised
  network: 923 Mbps down / 876 Mbps up; this is **not measured S3 throughput**.
  Quotes: $0.243333/hour total, $0.292161/hour adjusted; each transfer direction
  $0.0013020833/GB. All fall within the existing caps. Availability may change.
- Bounded object listings found no keys under `python-venv/`, `docker-assets/`,
  `model-weights/` or `models/`. These checks do not establish that no model archive
  exists anywhere else in the bucket. The source installer's HF/Drive path is confirmed.
- Attempted a small `preflight.json` upload to the specifically requested prefix
  with each existing identity. **Both returned AccessDenied; neither created an
  object.** Worker identity: `musetalk-s3-runtime`; publisher: `linguaEc2role`.
- Read-only IAM inspection succeeded. EC2 has `AmazonS3ReadOnlyAccess` but no
  successful prefix upload permission. No IAM policy was added or broadened.
- The plan now contains the narrow publisher policy request: PutObject and failed
  multipart-upload cleanup on `lingua-musetalk-s3-storage/python-venv/*` only.
- **Paused before spending:** permission approval is required before the rental,
  environment build/upload and restore timing can proceed. No timings are invented;
  no Mac package installation or production-worker mutation occurred.
