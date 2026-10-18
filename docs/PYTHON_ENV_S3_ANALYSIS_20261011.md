# Can MuseTalk restore Python dependencies from S3 instead of installing them?

Prepared 2026-10-11, source analysis against `main` `5cc706e`.
Original status at preparation: **feasible experiment, not implemented or measured**.
Update 2026-10-12: a private 4.51 GB archive was published and same-host fresh restore
passed, with packages restored in 22m42s and API healthy in 23m36s. Single GET took
21m04s; it is not consistently faster than prior pip runs and does not meet the
five-minute startup goal. See [final progress and evidence](Python-s3-venv-progress.md).
Production installer integration and fresh-host portability remain unvalidated.
The analysis below is the original pre-experiment framing, not a current cloud-state audit.

## Conclusion

Yes: keep the stock Vast base and deliver a **versioned, compressed Linux dependency environment**
from private S3. A validated hit could skip repeated pip resolution, wheel downloads, source builds
and package installation. This is not a whole-machine snapshot and does not revive the custom
Docker/GHCR project. The same artifact could later be served from another object store.

It is worth a bounded experiment because Python packages dominated the latest boot. But an ordinary
venv is not generally relocatable, S3-to-Vast egress is billed, and replacing this step alone does
not establish five-minute end-to-end readiness.

## 1. Evidence versus estimates

Historical A2 stock-template run from this chat (RTX 3090, US, 2026-10-11, instance `55426495`,
source `5cc706e`; evidence retained in the pre-existing execution checkout):

| Item | Measured result | Scope |
| --- | --- | --- |
| Server + Kokoro packages | 931s / 15m31s | One pip call; resolver/download/build/install combined |
| Avatar-preparation packages | 63s | Separate pip phase |
| All Python packages | 994s / 16m34s | Sum of the two preceding intervals |
| Full clone/check/publication | 210.801s / 3m30.801s | Not transfer alone |
| Model downloads + validation | 78s | Parallel groups |
| r5 S3 engine restore | 49s | Download/hash/extraction/file verification combined, not pure transfer |
| EC2 create request to healthy API | 1,621.488s / 27m01.488s | Not successful video/WebRTC readiness |

The supplied `rtx3090_startup_timing_and_s3_costs_2026-10-11.md` note reports an existing Linux
venv at **9.5 GB installed**, compressed to **4,527,001,419 bytes** with zstd level 3.
That is 4.527 decimal GB / 4.216 GiB. It is a prior sizing measurement, **not** a newly measured
clean artifact or a venv from this Mac. The clean artifact may differ.

For that archive size, transfer-only scenarios are:

| Sustained artifact download | Download alone |
| --- | --- |
| 11.7 MB/s | 387s / 6m27s |
| 20 MB/s | 226s / 3m46s |
| 50 MB/s | 91s / 1m31s |
| 100 MB/s | 45s |

These are **calculations**, not promised S3 performance. MB/s uses decimal bytes.
11.7 MB/s was pip's logged rate for one TensorRT wheel in A2, not a speed test or S3 measurement.
The 49s engine-restore interval cannot be treated as an isolated S3 throughput measurement.
Download verification, unpacking and validation add time; none has been timed for this environment.

Replacing all 994s of package work with a 120s restore would project roughly **12m27s**, if every
other stage remained unchanged. Even an impossible zero-second replacement leaves **10m27s**.
These are illustrative substitutions into one historical sample, not a new boot benchmark.
Further improvements to checkout and the rest of startup are needed for a five-minute target.

## 2. What compatibility requires

[Python's venv documentation](https://docs.python.org/3.10/library/venv.html) explains why ordinary
venvs are generally non-portable: executable scripts contain absolute interpreter paths, and the
environment depends on its base Python installation. Therefore **do not package the Mac venv**,
or assume any existing Linux venv can simply be copied to any host.

Build from a clean, matching Linux x86_64 stock-base environment, with:

- The canonical final path `/workspace/.venvs/musetalk_trt_stagewise`.
- CPython 3.10, known base interpreter path/version and standard library; record `pyvenv.cfg`,
  interpreter symlinks, shebangs, `.pth` entries and any editable/local-path dependencies.
- A recorded base image digest, distro, glibc/libstdc++ and other native-library requirements.
  The rolling `cuda-12.1-auto` tag by itself is not a sufficient compatibility guarantee.
- The selected cu121 constraints: Torch 2.5.1+cu121, Torch-TensorRT 2.5.0, TensorRT 10.3.0 and
  the same PyAV/aiortc/native package contract. Other matrices require separate validated artifacts.
- Explicit optional-group membership (Kokoro, avatar preparation, legacy INT8). Full-stack
  preparation includes the matching mmcv binary. Chin tools, if needed, have a separate venv.
- An installed package manifest plus dependency/constraint hashes and a trusted archive SHA256.

The snapshot does **not** include the host NVIDIA driver, coturn, ffmpeg or all apt/system libraries;
those must still be present. Do not copy GPU-specific JIT caches or TensorRT plans into this Python
artifact. Engines retain their own versioned/verified delivery contract. GPU and WebRTC/native
imports must be tested on the target, not merely inferred from matching dist-info metadata.

## 3. Important integration trap in today's startup

[vast_onstart.sh](../scripts/vast_onstart.sh) checks the **whole installation**, not only packages.
With `AUTO_SETUP=1`:

- `SETUP_CLEAN=1` requests `--clean` even when a restored installation looks valid; the installer
  then deletes the restored venv.
- Missing model files or required native assets can produce exit 11, causing a repair. The current
  installer then calls pip again even if all restored packages already exist.

Thus “download a venv at the top of the old boot script” is not enough to guarantee a pip-free boot.
Future implementation should integrate archive restoration/package validation into the canonical
environment/package phases, and explicitly skip those pip steps **only on a verified compatible
hit**. Continue system provisioning, model downloads, native VP8 handling, CPU/GPU smoke checks,
engine integrity/probes and final install-state checks. Keep the normal install as an explicitly
recorded fallback, or fail closed in strict archive mode; never claim a fast restore if pip ran.

The existing [install-state helper](../scripts/musetalk_install_state.py) checks Python/matrix,
package pins, top-level module availability, optional groups and files. It is useful but does not
prove all native ABI compatibility. Regenerate target-local state after successful validation;
do not copy a source machine's success stamp and GPU self-test as target evidence.

## 4. Proposed artifact and boot sequence (not implemented)

1. Build/install once in a clean matching Linux environment; run import/GPU/native validation.
   Package only the allowlisted dependency venv, excluding credentials, AWS/GitHub tokens,
   runtime secret files, pip/Hugging Face/JIT caches, models and avatar/customer data.
2. Publish an immutable private object and manifest keyed by dependency pins, Python ABI/base
   compatibility and optional groups. Code-only main changes need not rebuild it; dependency/ABI
   changes do. Example **proposed**, not existing, key:
   `python-env/linux-x86_64-cp310-cu121/<dependency-contract-hash>/venv.tar.zst`.
3. EC2 signs a short-lived object-specific download URL. The bootstrap can use curl before boto3
   exists, without receiving EC2's AWS credentials for this artifact. Other runtime permissions
   remain separate. [AWS presigned URLs](https://docs.aws.amazon.com/AmazonS3/latest/userguide/using-presigned-url.html)
   are bearer credentials: redact them from logs and ensure expiry allows provider startup,
   downloads and bounded retries; signing-session expiry can shorten their lifetime.
4. Download to staging; verify the trusted SHA256/manifest and reject unsafe archive paths/links.
   Extract safely and publish to the exact final venv path without overwriting a live process's
   environment. Budget disk for archive + expanded environment + rollback copy when applicable.
5. Validate the restored dependency contract, record a hit/miss and skip package installation only
   on a validated hit. Finish the normal startup and target-local state checks. Do not disable
   Kokoro/preparation groups just to make an archive test faster.

Fresh restore staging alone could occupy roughly 4.5 GB archive + 9.5 GB payload, before code,
weights, engines and runtime data; retaining a previous venv for rollback adds more space.
Measure extraction: zstd compression settings do not by themselves guarantee parallel extraction.

Start with a simple measured downloader. If transfer dominates, test a small bounded set of
parallel byte-range requests. [AWS's performance guidance](https://docs.aws.amazon.com/AmazonS3/latest/userguide/optimizing-performance-guidelines.html)
supports concurrent range downloads, but it does not guarantee a Vast host's advertised bandwidth.
Do not enable paid Transfer Acceleration or add a CDN merely for this feasibility test.

## 5. S3 costs and ongoing operation

Direct S3-to-Vast traffic leaves AWS. Same-region S3-to-EC2 transfer exemptions do not cover it.
[AWS's network FAQ](https://aws.amazon.com/about-aws/global-infrastructure/global-network/faqs/)
confirms 100 GB/month internet egress is shared across services/regions, not reserved for this worker.
Remaining allowance/account invoice was not queried here.

For planning, use approximately **$0.09 per billed GB** for the first paid internet-egress tier in
us-east-1, consistent with the [official AWS S3 cost-attribution example](https://aws.amazon.com/blogs/storage/amazon-s3-audit-logging-part-1-analyzing-server-access-logs-with-amazon-athena-for-performance-insights/)
and [S3 pricing](https://aws.amazon.com/s3/pricing/). At 4.527 decimal GB / 4.216 GiB, that is roughly
**$0.38-0.41 per fresh restore**, before free allowance, requests, Vast ingress, retries and taxes.
The distinction reflects unit conventions, not a newly measured invoice. Use actual billed units
and current regional rates when estimating production spend.

If free allowance is already consumed, 100 restores add roughly $38-41 and 1,000 roughly $380-410
at the same tier, excluding the separate engines/models/avatar traffic. Storage for a few GB is
small compared with repeated egress; verify retention/versioning costs before retaining many builds.
The host's $1.50/TB transfer ceiling is **separate** from AWS egress; it does not cap S3 charges.

For scale, compare serving the same archive from [R2 Standard](https://developers.cloudflare.com/r2/pricing/),
which has no internet-egress charge but storage/request costs. Its download speed to the workers
still needs measurement. No R2 account/bucket or other infrastructure was provisioned.

This trades money for scale-out latency, not necessarily cheaper GPU rental: 16m34s of the
baseline host's $0.243333/hour quote was only about **$0.067**, and a real restore still takes time.
Keep weights on the currently fast Hugging Face path unless another measured bottleneck justifies
mirroring them. Snapshot dependencies separately from weights, engines, code and avatars.

## 6. Next acceptance experiment

After explicit implementation/test authorization, restore the artifact onto a fresh matching US
RTX 3090 using the stock template. Record provider/base startup, checkout, environment download
bytes/seconds/retries, SHA verification, extraction, package validation, any fallback, weights,
engine restore, API health and a known-fixture WebRTC call. Verify runtime backends and actual
FPS/quality separately; this archive should not change inference settings.

Pass only if restoration is compatible, package pip steps are demonstrably skipped, all required
groups/native features work, and the end-to-end gain is measured. Keep the previous installer
path available. No instance was rented for this analysis.

## 7. Progress in this local change

- Fetched current main and pulled it in an isolated worktree, preserving existing WebRTC edits.
- Implemented [separate server/Kokoro timers](DEPENDENCY_INSTALL_TIMING_20261011.md); 12 mocked
  command/timing/failure tests, Bash syntax and whitespace checks passed. No pip installation
  or resolution tests ran on macOS; actual split durations are not yet available.
- Completed this feasibility/cost/source analysis. **Archive build/upload/restore integration,
  Linux validation, autoscaler/template rollout and real-world performance testing remain undone.**
