# Fresh RTX 3090 dependency profile — plan and progress

Requested 2026-10-15 (America/Chicago). This experiment uses the stock source
template, **not the abandoned custom Docker image or an S3 environment restore**.

**Complete.** [Final measured findings and ranked next steps](RTX3090_DEPENDENCY_STARTUP_PROFILE_20261015.md).
The test GPU was destroyed and independently verified absent; production unchanged.

## Scope and guardrails

- One new RTX 3090 created through the existing EC2 control machine. No automatic
  replacement or ambiguous-create retry. Protect every pre-existing instance.
- Preserve the $0.30/hour ceiling, including the allocated disk/adjusted quote,
  and $0.0015/GB in **both** Vast transfer directions. Prefer a verified US host
  with fast network, enough CPU/RAM, and 100 GB disk. Do not silently relax costs.
- Independent EC2 expiry destroys only the uniquely labelled experiment after
  at most two hours; normal completion collects evidence then destroys it early.
- Use expiring S3-read credentials, no production registration/callbacks, no IAM
  changes, no credential literals in the template/reports. No Mac pip/CUDA tests.
- Preserve dependency versions, indexes, groups and normal pip cache behaviour.
  Retain the previously tested separation of main, Kokoro and avatar-prep steps.
  Instrumentation is a test-only patch, not a production deployment.

## Measurement

1. Record EC2 request, provider acceptance/running, first SSH, clone start/end,
   apt, venv/bootstrap tools, main dependencies, Kokoro, avatar-prep, weights,
   native encoder, smoke/self-tests, engine restoration and API health.
2. Wrap normal pip invocations with observational hooks and native pip debug logs.
   Record monotonic/UTC spans for resolver, preparation, wheel/source builds,
   individual wheel installs, network requests, DNS and response-body reads.
   Save pip reports where supported. Hooks report their availability explicitly.
3. Sample container CPU, memory, network and disk counters every two seconds.
   Correlate low CPU/network activity with index/DNS/retry waits and high CPU/IO
   activity with extraction, compilation or installation. Do not log process env.
4. Copy durable incremental receipts to EC2 while startup runs, so SSH transport
   loss does not cancel startup or discard the measurements. Bound all work by
   the independent lease and collect before teardown.
5. Check `/health` and, if time permits, generate/decode one existing-avatar REST
   video. This is not a 400-FPS, production registration, WebRTC or TURN test.

## Interpretation rules

- Advertised link capacity is not observed package-server throughput.
- Resolver/preparation/network spans overlap: do not sum them as exclusive time.
- HTTP request duration includes connection/DNS/retries/headers; body consumption
  is measured separately. A cached response is not fresh internet transfer.
- Installation timing includes unpacking/files/bytecode as performed by pip.
- Internal pip hooks are version-sensitive; record coverage/gaps and keep native
  debug logs as independent evidence. Profiling adds some overhead.
- Report failed/recovered stages separately from a clean uninterrupted baseline.
- Rank measured bottlenecks before proposing optimizations. Do not alter the
  dependency stack mid-run or claim an untested optimization meets five minutes.

## Progress

- Existing local user changes preserved. Remote `main` rechecked; still
  `5cc706e90e50e93da1310628c025a84199cd8042`.
- EC2 reachable. Pre-existing instance 51074906 currently reports `exited`;
  this experiment will not start, stop or modify it.
- Initial strict offer search found no eligible host. Investigating the quoted
  fields/availability without relaxing the cost or transfer caps.
- Per-package instrumentation and stdlib-only tests being prepared.

### Preflight and launch

- 44 stdlib/mock tests passed locally; no pip/CUDA installations on macOS.
- EC2 template generation, shell/Python syntax and SSH keypair checks passed.
- Selected reviewed Pennsylvania machine 92165, host 443275. Equivalent offer
  slot IDs change; resolve the current ID on this exact machine before the one
  create request. Quote $0.187778/hour, adjusted $0.187875/hour, including 100 GB
  requested disk. Both Vast transfer rates $0.00000390625/GB.
- Advertised 527.7 Mbps download / 39.5 Mbps upload, 8 effective cores, 32,222 MB
  RAM, 2,924 MB/s disk benchmark. **Unverified budget-host fallback**, not a
  premium-host prediction. No price or transfer-cap relaxation.
- Launching one uniquely labelled test, `musetalk-dependency-profile-20261015-a1`.
  The independent two-hour expiry is armed before the create request. Initial
  strict search found no verified US offer satisfying all original caps.

### Live run

- First launch request was rejected before creation. Inventory/cleanup verified
  no instance under the original label. Its 35,957-byte inline launch was then
  reduced to roughly 11 KB using private inline compression. The exact original
  provider error text was suppressed, so the argument-size cause is an inference,
  consistent with the provider's documented argument-length cap, not a captured
  error-code confirmation. No duplicate or ambiguous-create retry occurred.
- New namespace `musetalk-dependency-profile-20261015-a2`; instance **56191133**,
  same Pennsylvania machine/host and original cost caps. Request started
  **2026-10-16 03:03:12.910 UTC** (Oct 15 Chicago); accepted **0.312 seconds** later.
- Independent expiry active, deadline approximately **05:03:12 UTC**.
- Initial provider status pending; per-package profiling begins in the on-start
  script and survives collector SSH disconnects. Stock base-image cache state
  is unknown; do not call its pull latency a universally cold-image measurement.

### First measured stage

- Provider running first observed **03:08:09.810 UTC**, **296.900 seconds** after
  the request (~4m57s), before any MuseTalk install time. Host-side logs record
  stock image loaded 03:08:03 and container running 03:08:06 UTC.
- EC2's SSH public key was attached only to this verified-owned test instance,
  after it became running. Waiting for the first successful SSH snapshot.
- Local test suite now passes **47** stdlib/mock cases, including overlap-safe
  analysis and credential-free URL summaries.

### Source setup and first package results (still in progress)

- Bootstrap began 03:08:15 UTC; code checkout ready ~03:09:35 (~80 seconds).
  System packages 19 seconds; venv/bootstrap tools 7 seconds.
- Main/server pip step **4m26s** (03:10:01 → 03:14:27), exit 0. Its detailed pip
  execution span is 264.964 seconds; shell timer includes wrapper overhead and
  whole-second rounding. This is substantially below earlier 13–33 minute main
  steps, despite this host advertising only 528 Mbps.
- Largest completed wheel-download spans so far: TensorRT libs **33.225s**,
  PyTorch **13.960s**, cuDNN **9.895s**, cuBLAS **6.722s**. Corresponding local
  wheel installs: TensorRT libs **19.854s**, PyTorch **13.890s**, cuDNN **6.370s**,
  cuBLAS **3.895s**. Exact trace retained; these are not engine builds.
- Main resolver span 159.544s **includes downloads and metadata preparation**;
  do not present it as 159 seconds of pure solver computation.
- Direct SSH port times out. The officially documented Jupyter proxy uses
  `ssh_port + 1`; that route succeeded. Collector was handed over at 03:13:49
  without restarting the instance/install or extending its original deadline.
  Earlier on-worker trace files were recovered intact. External direct API
  reachability will be distinguished from internal API health.
- pip 26.2.1 has no `BatchDownloader` or `req_install.install_wheel` alias; those
  optional hooks report unavailable. Resolver/preparation/download/requirement
  install/wheel install/source-build hooks work. Bulk downloads bypass Requests'
  `iter_content`, so body counters alone would miss large wheels. Use downloader
  spans, native pip logs and container network samples together.

### Completion

- Main/server shell step **266s**, Kokoro **38s**, avatar prep **26s**; all
  package groups **330s**. Model weights **89s** (parallel groups), CPU imports
  **6s**, GPU self-test **42s**. Canonical installer **495s**.
- Engine restore/verification **33s**, engine provisioning/build **0s**, API
  start-to-health/recipe-verification **19s**. Local health success recorded
  **03:18:47.796 UTC**, **934.887s** after create request. EC2 observed it at
  03:18:54.306 UTC over the working SSH route.
- Direct external API port timed out; this is an **internal health/REST test**,
  not production traffic acceptance. No WebRTC/TURN registration test.
- Existing avatar warm **12.850s**; video generation/download/decode passed
  **03:19:36.791 UTC**. H.264 249 frames, 512×896, 25 FPS; AAC audio present.
- Complete private archive **26 files / 13,403,322 compressed bytes**, plus
  **5,063 pip events / 341 resource samples**, collected before destruction.
  The detailed timing JSON and credential-free pip events were copied locally.
- Test instance destroyed and absent **03:19:43.019 UTC**; fresh independent
  audit **03:21:18 UTC** confirmed absence and inactive expiry. Original RTX
  4070S 51074906 remained `exited`, as it was before the experiment.
- Quote-based test rent estimate **$0.052**, not an all-inclusive invoice.
- The current path did not reproduce the older multi-minute-per-dependency
  pathology. Detailed observations support targeted GPU-wheel caching, reducing
  index fan-out, slim checkout, reviewing optional groups and self-test cost;
  end-to-end latency also requires addressing the five-minute base startup.
  No follow-up optimization was deployed or measured in this experiment.
