# Fresh RTX 3090 startup: dependency-level profile

Experiment requested October 15, 2026, America/Chicago. Measurements below are
October 16 **UTC**. Source `main`: `5cc706e90e50e93da1310628c025a84199cd8042`.

## Result

**Main dependencies took 4m26s, all Python package groups 5m30s, and the canonical
installer 8m15s. Request → independently observed API health took 15m41s**
(the worker's local success log corresponds to approximately 15m35s).

The Python bottleneck is real but not a single dependency taking 10–16 minutes
on this host. The largest item was TensorRT's 2 GB library wheel: **33.2 seconds
download + 19.9 seconds local installation**. PyTorch was **14.0 + 13.9 seconds**.
Meanwhile, stock-image/container provisioning took almost **five minutes before
the bootstrap began**, and the full checkout took another **80 seconds**.

The prebuilt r5 engine path worked. Engine restore/verification took **33s**;
server start → health and recipe verification took **19s**. There was **no r5
TensorRT engine build**. The separate installer GPU self-test still spent **42s**
on its compile/timing checks.

One existing-avatar REST video was generated, downloaded and decoded with both
video and audio. The test instance was **destroyed and its absence verified**.
This is not a production/WebRTC/TURN/registration or 400-FPS acceptance test.

## Evidence and reproduction

- [Experiment plan and chronological progress](DEPENDENCY_PROFILE_PLAN_20261015.md).
- [Structured summary, every package's spans, download spans, phase markers and receipts](dependency_profile_20261015/summary.json).
- [5,063 timestamped pip events](dependency_profile_20261015/pip-events.jsonl).
- Instrumentation: [dependency_startup_profile.py](../scripts/dependency_startup_profile.py).
- Owned-instance controller: [dependency_profile_ec2.py](../scripts/dependency_profile_ec2.py).
- Offline analyzer: [dependency_profile_analyze.py](../scripts/dependency_profile_analyze.py).

Full private raw logs remain on EC2 under
`/home/ec2-user/.local/state/musetalk-dependency-profile-20261015-a2/`:
`complete-worker-evidence.tar.gz`, `complete-worker-files/`, ownership ledger,
provider progress and final audit. **26 files**, a **13,403,322-byte compressed
archive**, and per-file SHA256 receipts were collected before teardown. This
includes native pip DEBUG logs/reports, resource samples, bootstrap/onstart/API
logs, GPU self-test and install stamp. Raw debug URLs are not published here.

Analyze the copied logs without installing any dependencies:

```bash
python3 -B scripts/dependency_profile_analyze.py /path/to/complete-worker-files
python3 -B -m unittest test_dependency_startup_profile test_install_package_timing test_python_s3_venv_safety -q
```

No pip installation or CUDA test ran on the Mac. No git commit/push or production
deployment was performed for this experiment.

## Host and test contract

| Field | Measured/recorded value |
|---|---|
| Instance | `56191133`, uniquely labelled test only |
| Machine / host | `92165` / `443275`, Pennsylvania, US |
| GPU | RTX 3090, 24,576 MiB, 350 W power limit |
| Driver / Linux | 595.58.03 / kernel 6.8.0-111, glibc 2.35 |
| CPU | AMD EPYC 7502; **8 effective cores** allocated |
| Allocated memory / disk | 32,222 MB / 100 GB |
| Advertised network | **527.7 Mbps down**, 39.5 Mbps up |
| Advertised disk benchmark | 2,924 MB/s, not measured application IO throughput |
| Host status | **Unverified** budget-host fallback; reliability quote 99.72% |
| Hourly quote | $0.187778 total; $0.187875 adjusted, including requested disk |
| Vast transfer quote | $0.00000390625/GB in each direction |
| Stock template | `4dc0868ee56886c667c217d6c3634a58` |
| Base image | `vastai/pytorch:cuda-12.1-auto`; cache state unknown |
| Stack | CPython 3.10.12, pip 26.2.1, cu121 pinned requirements |
| Installer groups | Main, Kokoro, avatar-prep enabled, clean venv |
| Environment restore | **None**; normal pip download/install, no S3 venv |
| Safety | Expiring S3-read session, isolated from production callbacks |

The custom Docker/GHCR approach was not used. Vast still uses its normal stock
container underneath the source template. No AWS resources or IAM policies were
created/changed for this run.

### Access caveat

The advertised direct SSH port and direct API port timed out from EC2. Vast's
Jupyter SSH proxy worked using **`ssh_port + 1`**, as implemented in the
[official Vast CLI](https://github.com/vast-ai/vast-cli/blob/master/vast.py).
The observer was handed to that route without restarting the GPU or installation,
and without extending its deadline. All early on-worker traces were recovered.

Health was verified on worker loopback through SSH; video used a temporary
**EC2-loopback-only SSH tunnel**. This host was **not validated as publicly
reachable or ready for production autoscaling traffic**. Port reachability must
be part of host acceptance, independently of install speed.

## Full timeline

These shell phase durations are rounded to whole seconds. UTC profile prefixes
also provide subsecond event receipt times. Child phases are not additional time
to add to their parent. Parallel model-download times must not be summed.
The **15m41.396s observer duration uses one EC2 monotonic clock**. The ~15m35s
local-log comparison correlates worker UTC with EC2 UTC; no separate clock-offset
calibration was performed. Its displayed subsecond arithmetic is not a claim of
subsecond cross-machine timing accuracy.

| Stage | UTC interval / observation | Duration |
|---|---|---:|
| Request accepted | 03:03:12.910 → 03:03:13.222 | 0.312s |
| Request → provider running observed | → 03:08:09.810 | **4m56.900s** |
| Request → bootstrap/checkout start | → 03:08:15.162 | **5m02.252s** |
| Git clone + publish checkout | 03:08:15.162 → 03:09:34.718 | **1m19.556s** |
| Installer preflight | 03:09:35 | <1s |
| System packages | 03:09:35 → 03:09:54 | **19s** |
| Create venv + packaging tools | 03:09:54 → 03:10:01 | **7s** |
| Main/server packages | 03:10:01 → 03:14:27 | **4m26s** |
| Kokoro incremental packages | 03:14:27 → 03:15:05 | **38s** |
| Avatar-prep stack + import check | 03:15:05 → 03:15:31 | **26s** |
| **All Python package groups** | 03:10:01 → 03:15:31 | **5m30s** |
| Model weights, including setup/validation | 03:15:31 → 03:17:00 | **1m29s** |
| Native VP8 install/verification | 03:17:00 → 03:17:02 | **2s** |
| CPU import smoke | 03:17:02 → 03:17:08 | **6s** |
| GPU self-test | 03:17:08 → 03:17:50 | **42s** |
| Stamp + post-install check | 03:17:50 | <1s each |
| **Canonical installer total** | 03:09:35 → 03:17:50 | **8m15s** |
| Runtime/secret/TURN wrappers | after installer, before engine restore | ~5s wall overhead |
| r5 engine restore + verification | 03:17:55 → 03:18:28 | **33s** |
| Engine build/provisioning | skipped | **0s** |
| Server → local health + recipe verification | 03:18:28 → 03:18:47 | **19s** |
| **Canonical `vast_onstart.sh` total** | 03:09:35 → 03:18:47 | **9m12s** |
| Outer bootstrap start → onstart complete | 03:08:15 → 03:18:47 | **~10m33s** |
| **Request → local health success in log** | → 03:18:47.796 | **15m34.887s** |
| Independent EC2 observer sees health | 03:18:54.306 | **15m41.396s** from request |
| Existing avatar ready | 03:19:09.158 | **12.850s** warm/restore step |
| Video downloaded + decoded | 03:19:36.791 | **16m23.881s** from request |
| Complete evidence archived | 03:19:42.558 | before deletion |
| Destroyed, absence verified | 03:19:43.019 | **16m30.109s** rental wall time |

Stock provisioning includes provider image/wrapper/container work, not a measured
pure download. Host logs say image loaded at 03:08:03 and container running at
03:08:06. No host-side per-layer throughput or cache-hit assertion is available.

### Model downloads

All model groups ran in parallel; the stage's critical group was avatar prep:

| Group | Logged duration |
|---|---:|
| MuseTalk model | 84s |
| Avatar-preparation weights | 87s |
| Base runtime weights | 35s |
| Face parsing | 27s |
| Optional Kokoro cache | 56s |

These are the normal source downloader's paths, not an all-model S3 snapshot.
The separately restored S3 r5 engine archive was 1,000,524,577 bytes. From the
restore-start log to archive SHA verification was ~16s; extraction and payload
verification brought the complete engine phase to 33s. No engine compilation
was substituted for a download failure.

## Inside main pip: what actually consumed the 4m26s?

The instrumented pip execution was **264.964s**; the shell step was 266s with
wrapper/rounding overhead. The following major sequential outer spans account
for almost all of pip's wall time:

| Pip phase | Time | Meaning |
|---|---:|---|
| Resolver + initial requirement preparation | 159.544s | Index discovery, metadata, initial downloads and resolving — **not pure solver CPU** |
| Deferred preparation/download completion | 21.628s | Remaining wheel preparation/downloads |
| Two source wheel builds | 0.457s | TensorRT shim and antlr4 runtime; not CUDA/TensorRT engine builds |
| Install 119 requirements | 82.214s | Unpack, file writes, bytecode and pip bookkeeping |
| Other pip overhead | ~1.121s | Difference from the above outer spans |

Nested measurements, **not additional exclusive time**:

- 218 downloader calls (wheels, metadata, sdists): **113.320s** total.
- 575 HTTP request/header spans: **56.318s** total, median **0.095s**, max **0.525s**.
- Union of downloader/request/body spans: **148.225s**. Downloader spans include
  local writes; this is not pure time waiting on the network or a wire-byte total.
- Five recorded main-process DNS spans: **0.279s** total; no minutes of DNS delay.
- Source-distribution metadata preparation: **8.778s**, nested in resolving.
- Main pip process sampled CPU: **139.68 CPU-seconds**, excluding child build
  processes, over a ~263s sampled interval. Peak observed main-process RSS ~198 MiB.
- Container received **5.299 GB** during the sampled main-pip interval, averaging
  **161.3 Mbps over the whole phase**. This includes headers/other container traffic;
  it is not an exclusive wheel-byte or billable-wire total. Download bursts were
  much faster than that whole-phase average because installation does not download.

No retry, timeout, resolver-backtracking or resolution-conflict signal was found
by the documented pattern scan of the seven full pip DEBUG logs. This is evidence
for this successful run, not proof that the older slow runs had no such problems.

### Individual heavy dependencies

Downloads below are direct completed downloader spans; installs are wheel-install
spans. Sizes are pip's displayed decimal sizes, not newly measured archive sizes.
The complete JSON contains timings for every observed package and download.

| Package | Download | Local wheel install | Observation |
|---|---:|---:|---|
| `tensorrt-cu12-libs==10.3.0` (~2,037.5 MB) | **33.225s** | **19.854s** | Largest individual package, NVIDIA index |
| `torch==2.5.1+cu121` (~780.4 MB) | **13.960s** | **13.890s** | PyTorch CDN |
| `nvidia-cudnn-cu12==9.1.0.70` (~664.8 MB) | **9.895s** | **6.370s** | ~67 MB/s displayed-size/span estimate |
| `nvidia-cublas-cu12==12.1.3.1` | **6.722s** | **3.895s** | NVIDIA index |
| `nvidia-cusparse-cu12==12.1.0.106` | 3.166s | 1.772s | Download dominates |
| `nvidia-nccl-cu12==2.21.5` | 3.086s | 1.735s | Download dominates |
| `triton==3.1.0` | 3.078s | 2.322s | Prebuilt wheel, not a Triton source build |
| `nvidia-cusolver-cu12==11.4.5.107` | 2.023s | 1.242s | Prebuilt wheel |
| `llvmlite==0.49.0` | 2.014s | 0.871s | Prebuilt wheel, no LLVM compilation |
| `sympy==1.13.1` | See complete JSON | 4.109s | Many Python files/bytecode, comparatively local-install-heavy |
| `transformers==4.39.2` | See complete JSON | 3.908s | Local install matters as well as transfer |

The TensorRT and PyTorch download/install pairs together are ~81 seconds. They
are sensible cache targets, but **neither took multiple minutes by itself here**.
The historical multi-minute mmcv source build also did not occur: the repo-local
mmcv wheel pip invocation took **0.957s**.

### Index fan-out: measurable unnecessary work

Main pip queried every package against all three configured indexes:

| Index | Requests | Outcomes |
|---|---:|---|
| PyPI | 119 | 119 × 200 |
| PyTorch index/CDN entry host | 124 | 40 × 200, **84 × 403** |
| NVIDIA | 133 | 29 × 200, **104 × 404** |
| PyTorch R2 CDN | 4 | 4 × 200 |
| Python-hosted files | 195 | 195 × 200 |

The 188 negative extra-index lookups consumed **15.492s** of HTTP request spans
in main pip (10.928s NVIDIA, 4.564s PyTorch). These did **not** mean the install
failed: pip found the required pinned packages on the appropriate other index.

For Kokoro, **all 50 PyTorch index lookups returned 403, and all 50 NVIDIA lookups
returned 404**. Their request spans totalled **10.026s** out of the 37.014s pip
execution. Selected new Kokoro dependencies came from PyPI/GitHub, so retaining
GPU indexes for that group is a clear candidate for a controlled follow-up test.
Do not remove the NVIDIA/PyTorch sources from the main GPU group indiscriminately.

## What to improve next — ranked, without custom Docker

These are recommendations from the traces, **not implemented speedups or measured
promises**. Keep the validated pins and compare generated output after any change.

1. **Include base provisioning and direct-port acceptance in autoscaler readiness.**
   The first five minutes precede MuseTalk. Prefer hosts with observed fast stock
   base startup; keep a small warm-capacity pool if a firm low-latency target is
   required. Measure cold versus cached starts, not just advertised bandwidth.
   Reject unreachable direct ports early. A proxy-only diagnostic pass is not a
   production-ready worker.
2. **Replace the full media-heavy clone with a validated slim runtime checkout.**
   There are ~80 seconds to target here. Use partial clone + sparse checkout or a
   versioned slim source bundle, retaining runtime scripts/config/data and the
   mmcv wheel if avatar preparation remains enabled. Verify imports, avatar-cache
   restore and video output before adopting it; no sparse-checkout saving was
   measured in this run.
3. **Target the expensive GPU wheels rather than guessing all packages are slow.**
   Prioritize TensorRT libs, PyTorch, cuDNN and cuBLAS. Use a versioned wheel/env
   cache with platform/Python/CUDA/TRT/hash checks; avoid downloading an already
   validated environment on restarts. A small parallel prefetch can overlap
   transfers with apt/other work, but pip must consume those exact downloaded
   wheels rather than download them twice. Source pins must not silently change.
   The prior single-GET S3 venv restore took ~22m42s to packages-ready, so it is
   **not** a demonstrated improvement over this 5m30s package phase. Profile the
   S3 transfer path separately before making it the default.
4. **Reduce index fan-out with group-specific sources or an exact wheel lock.**
   Start with Kokoro's GPU-index misses (~10s request overhead), then map main
   wheels to their correct source/hash to avoid unnecessary lookups. The measured
   main negative-index overhead is ~15.5s, not a hidden ten-minute solver stall.
   Validate resolved versions/artifact hashes on Linux.
5. **Split serving from optional capabilities only when the worker contract allows.**
   Omitting externalized local TTS would remove this run's **38s** Kokoro package
   step and its optional model downloads. Avatar-prep packages cost **26s**, but
   removing them is only valid if `/avatars/prepare` moves elsewhere. Their model
   group was the 87s critical downloader, but MuseTalk itself took 84s in parallel:
   **do not claim that dropping avatar prep saves another full 87s**. Less competing
   traffic may help; quantify that in a separate trial.
6. **Review the 42s GPU self-test, preserving essential correctness checks.**
   It runs a TAESD compile/timing exercise even though serving restores prebuilt
   engines. Consider a serving-mode minimal probe plus a separate exhaustive
   validation job. Do not skip engine SHA checks, compatibility checks or runtime
   health/probe validation just to make a timer look better.
7. **Test parallel local installation or pre-extracted env delivery last.**
   The local 119-package install was **82s**, so transfer optimization alone cannot
   erase it. A different installer or pre-extracted env could reduce this, but
   compatibility, relocation, bytecode, native ABI and exact version verification
   are required. No alternate installer was tested here.

### Can dependency work alone produce five-minute scale-out?

**Not on the measured critical path.** Removing the entire 5m30s Python package
phase from the 15m35s request-to-health interval would still leave about **10m05s**.
Removing only main dependencies leaves about **11m09s**. Using the independently
observed 15m41s instead adds ~6 seconds to those remainders. These are simple
counterfactual subtractions, not benchmark predictions.

A five-minute end-to-end target requires attention to base/container startup,
checkout, model availability, overlap and validation as well as Python packages.
The good news is the old 9–10 minute server/engine warmup is gone on this recipe:
the current engine restore + server readiness tail is about **52 seconds**.

## Acceptance, cleanup and limitations

- Internal API health returned `ok=true`, and r5 backend verification passed.
- Existing avatar ready in 12.850s. Named-output REST generation, download and
  ffmpeg decode passed: H.264 **249 frames**, 512×896, 25 FPS, 9.96s video;
  AAC audio 10.0s. MP4 size 1,323,949 bytes; hash and stream receipts are in JSON.
  Output frame rate is **not aggregate inference throughput**.
- Direct external API check timed out. No production registration, live WebRTC,
  TURN call, complete avatar roster, quality comparison or 400-FPS load test ran.
- One unverified host/one successful boot; no p50/p95 or causal claim explaining
  the earlier 16–37 minute dependency phases. Those runs need comparable package
  traces to attribute their slowdowns. Advertised Mbps alone did not predict them.
- Internal pip hooks report their availability. pip 26 removed two optional older
  aliases; active resolver/download/install/build hooks and native DEBUG logs
  supplied the evidence. Requests body counters miss bulk wheel streams; do not
  treat them as the complete download volume.
- Logging overhead was not calibrated with an uninstrumented twin. 341 resource
  samples were taken every ~2s. `/proc` system memory/CPU may describe the physical
  host; allocation comes from the offer. Per-process CPU uses measured 100 Hz
  clock ticks, RSS uses 4,096-byte pages, excludes short/child processes as noted.
- Initial overlong launch was rejected with no instance created; shortening the
  payload was followed by acceptance. Its exact rejected error was not retained,
  so argument length remains an inference, not a proven original error code.
- Test instance absence was verified 03:19:43 UTC and independently audited again
  **03:21:18 UTC** ([final audit receipt](dependency_profile_20261015/final-audit.json)).
  Expiry timer stopped. Pre-existing RTX 4070S instance 51074906
  was `exited` before and after; this work did not start, stop or modify it.
- Quote-based rent estimate **~$0.052** for the test's 16m30s wall time. This is not
  an invoice and excludes metered transfers, S3 egress/requests and the existing
  EC2's normal running costs. No new builder, ECR/GHCR image or IAM mutation.
