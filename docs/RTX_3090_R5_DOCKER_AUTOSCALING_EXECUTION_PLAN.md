# RTX 3090 r5 performance, Docker delivery, and EC2 autoscaling execution plan

**Current priority — October 9, 2026:** follow the
[Docker fast-startup plan](DOCKER_FAST_STARTUP_PLAN.md) for the private GHCR image,
ephemeral builder, measured boot-to-usable-capacity work, GitHub/AWS access handoff,
and rollout. It supersedes the original Docker execution order and public/searchable
registry choice below, as well as the later ECR proposal. The original October 8 plan
is preserved as historical context; its 400 FPS and quality verdicts must not be
silently changed.

Prepared October 8, 2026. Repository inspected at `5cc706e90e50e93da1310628c025a84199cd8042`.

**Objective:** provision a new RTX 3090 through the existing EC2/Vast integration; validate portable r5; build reusable RTX 3090 benchmark harnesses; achieve sustained **400+ aggregate FPS** with quality at least matching the established RTX 4070 SUPER r5 reference; publish the validated runtime as a usable, searchable Docker image; update **Templates → My Templates → `(NEEDS UPDATES) Musetalk`** in the user's browser; launch another new RTX 3090 through EC2 with that template; minimize and measure the time until it can serve a real call.

This document is an execution plan. It does not claim that a 3090 was rented, benchmarked, or that a Docker image/template has already been published. Run the work packages in order and update their evidence as work finishes.

## 0. Autonomous overnight execution contract

The user will launch this as one overnight Codex goal and will be unavailable for manual quality validation. **Codex owns implementation, numerical quality testing, visual inspection, candidate selection, Docker publication, browser template editing, fresh-instance verification, and rollback. Human quality sign-off is not a dependency.** Tomorrow's user review is optional follow-up, not a release gate.

- Execute all work packages continuously from the same goal. Checkpoints support recovery; they do not mean stop and ask the user to start the next task.
- Freeze the autonomous acceptance criteria in section 9 before tuning. Evaluate the actual frames/contact sheets with available image tools as well as numerical reports. A passing script or a fluent summary is not visual inspection.
- Make routine implementation and candidate-selection decisions independently. Resolve nonessential names from existing conventions: `musetalk-r5` in the configured registry namespace, distinct per-instance SSH aliases, and timestamped report directories.
- Resolve SSH/API/registry credentials, authenticated browser access, and a spending ceiling from existing configuration/session context during preflight. If a rental ceiling has not been established, or an external login requires the absent user's MFA, mark that dependent action blocked and finish every independent authorized task. Never fabricate access or treat silence as a credential/budget grant.
- Do not wait overnight for a video-review answer. Reject a candidate that fails quality, select the best candidate that passes, or continue improving the candidate within the goal's resource limits.
- Preserve reproducible artifacts after each phase. Use bounded retries, reconcile ambiguous cloud actions, and roll back a failed template/image deployment automatically.
- If the goal has a time/spend limit, reserve roughly the final third for image publication, template/fresh-instance validation, evidence upload, and cleanup. Prefer a small number of measured improvements over unbounded search that leaves nothing deployable.
- Failure to reach 400+, quality parity, or true fresh-instance readiness remains an unmet objective. Continue useful packaging/testing of the best safe candidate, label it accurately, and leave the established production release intact when promotion conditions fail.

**Preflight environment requirement:** start the goal from a Codex environment that can access the user's SSH config and authenticated browser, the EC2/Vast APIs, S3, Git, and the Docker registry. The current remote Linux worker alone lacks the user's SSH config and computer-use capability. A goal cannot complete those external steps from an environment that does not expose them.

## 1. Outcomes and measurement contract

| Outcome | Required evidence |
|---|---|
| Existing EC2 control plane is accessible | Correct SSH alias, repository/service locations, running revision, and actual provisioning API contract recorded without secrets |
| Instance A: RTX 3090 development worker | Creation response, Vast instance ID, actual host GPU/CPU/RAM/disk, SSH alias, and verified boot |
| Portable r5 starting point | Measured throughput on the 3090 before native-engine optimization; 300+ FPS is the target, not an assumed result |
| Reusable RTX 3090 harnesses | Runnable GPU-stage, aggregate-throughput, live-WebRTC, and startup-timing entry points, with machine-readable reports and meaningful failure exit codes |
| Native RTX 3090 r5 | Native sm86 UNet and TAESD plans, exact manifests/hashes, quality comparisons, and sustained 400+ aggregate FPS |
| Avatar caches/latents | Existing cache compatibility checked; required new preparations versioned, verified, and persisted without replacing known-good avatars |
| Docker image | Reproducible Dockerfile, immutable tag/digest, registry page/search result, clean pull, GPU run, and measured performance |
| Updated Vast template | Existing template edited in the user's browser, saved settings read back, previous settings retained for rollback |
| Instance B: fresh template validation | A different instance created via the EC2 provisioning path using the new template, reaching ready state without manual repair |
| Faster autoscaling | End-to-end timing from EC2 scale-out request to first usable stream, stage breakdown, cold/warm classification, and repeated-run evidence |

**FPS definitions must stay separate:**

1. **GPU-path FPS:** UNet + decoder/postprocessing/transfers. Useful for diagnosis; insufficient for a 400 FPS release claim.
2. **Aggregate full-recipe FPS:** total completed output frames across concurrent streams divided by their shared measured wall time. This is the comparison with the historical approximately 400 FPS result.
3. **Live delivery FPS:** decoded frames arriving at each WebRTC client, with freshness, buffering, latency, and codec recorded. Aggregate GPU capacity does not establish live-call capacity.
4. **Startup latency:** EC2 request to usable call, including provider provisioning, image download, startup, registration, and required avatar warming. HTTP `/health` alone is insufficient.

The final summary must present all four. Do not add per-stream FPS computed over different denominators, count padding or repeated held frames as newly generated output, or use a short burst as sustained throughput.

## 2. What is established, and what remains unmeasured

### 2.1 Historical baseline

Historical performance below was measured on an **NVIDIA GeForce RTX 4070 SUPER**, a 12 GB-class card, with the recorded 220 W power cap. Driver 595.84, torch 2.5.1+cu121, and TensorRT 10.3.0 are recorded in the reproduction/bundle documents. Read original run metadata for exact physical/visible VRAM and co-resident load; do not substitute the new machine's facts.

| Historical result | Evidence and limitation |
|---|---|
| Native r5: 401.8 / 400.1 aggregate FPS | Six streams, two measured windows of at least 60 seconds; [reproduction runbook](../scripts/repro_400fps/README.md) |
| Native r5 sustained: 404.0 down to 399.96 FPS | Very little margin around 400; do not round a sub-400 3090 result up to a pass |
| Portable `AMPERE_PLUS` r5: 307.7 / 307.1 FPS | Measured on the **4070 SUPER**, not the 3090; [portable-bundle report](fps_comparisons/ampere_plus_r5_20260930/README.md) |
| Live WebRTC: 10 calls passed short runs; 15-call one-hour soak did not pass | Event-loop stalls and RSS growth remained; [live report](fps_comparisons/live15_r5_20260929/README.md) |
| Existing r5 quality gates are mixed | UNet mean-error gates pass, strict max-error gates fail; TAESD full-frame max was 5 LSB against a 3 LSB bar; some landmark comparisons fail |

**No inspected record proves 400 FPS on an RTX 3090 with the new r5 pipeline.** The older RTX 3090 `split8-int8` bundle belongs to `legacy_int8`; it is not the desired native r5 bundle.

### 2.2 What the 4070 optimization actually contains

- Stagewise UNet, batch 16, `srccache` architecture.
- Selective INT8 on 117 layers using `recipe_gmac_0.50.json`, with sensitive paths retained in FP16.
- TensorRT TAESD **decoder**, normally batch 8, full-height face output and fused postprocessing.
- CUDA graphs, pinned transfers, batching, and the full chin/seam composition used by the benchmark.
- Live serving improvements: deadline pacing, nonblocking handoff, packed I420, idle-frame warming, thread caps, GC changes, and lean avatar storage.
- A separately installed native VP8 media encoder, whose actual activation must be verified from the resolved runtime and negotiated stream.

Distinguish three concepts when implementing the user's request for new encoders and latents:

| Component | RTX 3090 work |
|---|---|
| TensorRT execution plans | Build on the 3090; these are GPU/runtime-specific |
| Avatar VAE encoding and saved `latents.pt` | Validate existing caches first. They are model/preprocessing artifacts, not automatically GPU-model-specific. Reprepare where necessary and compare results |
| TAESD decoding | Build the matching native decoder plans; do not silently substitute a different avatar encoder |
| VP8/H.264 video encoding | Preserve and measure the chosen WebRTC encoder; it is distinct from latent creation and image decoding |

Native plans alone cannot guarantee 400 FPS. Keep quality constant and investigate actual bottlenecks before changing precision or workload.

## 3. Execution inputs, boundaries, and host roles

### 3.1 Resolve these inputs before their dependent action

| Input | Current state / resolution |
|---|---|
| `EC2_SSH_ALIAS` | Discover in the user's actual SSH config. This Linux worker has no `/root/.ssh/config`; do not invent an alias or assume the browser machine's keys are here |
| EC2 backend repository/service/API | Discover after SSH; existing worker code reveals registration endpoints, not the instance-creation API |
| Vast credentials and account | Use the existing integration/account and its approved credential source |
| Hourly limit, total experiment budget, lifetime | Resolve from account/session policy or the goal launch inputs before renting; missing spending authorization blocks purchases, not independent engineering |
| Docker namespace and repository | Unspecified. Discover the user's registry account; proposed repository name is `musetalk-r5`, subject to their namespace |
| Image visibility | User requested searchability. Plan a public, distributable image; keep private assets outside it. If required payload cannot be public, resolve the image split before publishing |
| Template ID/hash | Locate the exact named template in My Templates and record its actual identifier |
| Browser/computer-use session | Must be the user's authenticated Vast browser. This worker session does not currently expose computer-use tools |
| S3 access | Previous live check failed because `lumatalk-root` login expired. Renew operator access when needed; autoscaling must use the established noninteractive worker credential path |
| Production TTS and selected codec | Inspect EC2 routing and effective worker settings; do not change these merely to make a benchmark faster |
| Numeric startup SLA | Not supplied. Measure the original path, set an explicit target before startup optimization, and report whether it is reached |

The user has requested a plan now. When invoked as the execution goal, proceed with the requested purchases, publication, and template changes within the resolved budget and destination. There is no manual quality-review checkpoint. Record environmental blockers and continue independent work instead of repeatedly asking an unavailable user for confirmation.

### 3.2 Host responsibilities

- **Operator machine:** user's SSH config, existing identities/agent, authenticated browser, and orchestration console.
- **Existing EC2:** provisioner/control plane, worker registry, routing, and authoritative request-to-ready timing.
- **Instance A:** RTX 3090 development, native engine builds, quality evaluation, throughput optimization.
- **Docker builder:** EC2, CI, or another existing capable builder. A rented Vast container may not have Docker privileges; never assume Docker-in-Docker or a host socket is available.
- **Instance B:** a separate new RTX 3090 created from the final image/template. No developer-installed dependencies, copied local caches, or SSH repairs may be required for success.

Keep the present 4070 worker and existing EC2 services intact. Run only one GPU-heavy experiment at a time on each rented GPU. Reconcile instance IDs after ambiguous API timeouts instead of creating duplicates. Archive artifacts before releasing a paid development instance; cleanup decisions must account for storage charges as well as compute.

## 4. Evidence and handoff convention

Create one run root per execution, for example `docs/fps_comparisons/rtx3090_r5_<UTC-date>/`, with small reports in Git and large media/engines in checksum-addressed S3 objects.

Required outputs:

```text
run_state.json
environment.json
provisioning/                  # sanitized requests/responses, instance IDs, template identity
harnesses/                    # harness revision, input manifests, self-check results
portable/                     # portable r5 GPU/aggregate/live results
native/                       # build manifests, GPU/aggregate/live results
quality/                      # numerical verdicts, Codex visual inspection, acceptance decisions
avatars/                      # per-avatar/pose cache and latent verification
release/                      # bundle descriptor, image digest, template before/after
startup/                      # EC2 observer events, worker events, timing reports
RESULTS.md
```

Every benchmark report must record actual GPU model, compute capability, physical and visible VRAM, driver/runtime versions, power cap, clocks/temperature, CPU allocation, available RAM, shared memory, disk, code revision, dirty status, engine hashes, input hashes, run date, and co-resident load. Label GPU measurements, CPU-only tests, historical results, estimates, and unverified fields distinctly.

`run_state.json` should preserve the current phase, completed checks, unresolved blockers, instance/worker IDs, SSH aliases, artifact locations, image digest, template ID/hash, and the exact next command. Store credential **references**, never credential values. A later Codex session must be able to resume without rerenting or guessing which engine set was tested.

## 5. Work package A — SSH into EC2 and discover the real provisioning path

1. On the operator machine, inspect the existing SSH host blocks and `Include` files. Resolve the EC2 alias with `ssh -G <alias>`; retain its user, identity, forwarding, and jump-host behavior.
2. SSH with the existing alias. Do not disable host-key verification or copy private keys to rented hosts. Resolve a genuinely changed key against the actual new instance identity.
3. Record EC2 hostname/instance identity, region, repository paths, deployed revision, process manager, health endpoint, worker registry, and autoscaler settings. Inspect configuration names and API schemas without printing secret values.
4. Locate the existing Vast offer-search/create/status/terminate implementation. Discover whether it consumes a template ID, hash, inline image/on-start fields, or a combination. Find the exact request body and response mapping.
5. Record which overrides EC2 sends. A hardcoded old `image`, `onstart`, or environment override can defeat a template update; this must be tested later.
6. Verify access to the necessary S3 model/engine/avatar objects and runtime secret. Confirm the worker secret includes valid control-plane registration settings; old documented exports lacked these.
7. Identify how to create an experimental worker without automatically routing production users to it. Use an existing pool/tag/drain feature if present; implement the smallest missing distinction if necessary.
8. Check Docker builder capability and registry access now, so container publication does not get blocked after GPU work is complete.

Known worker endpoints from [worker_control_plane.py](../scripts/worker_control_plane.py) are `/api/runtime/workers/register` and `/api/runtime/workers/heartbeat`. They are not provisioning endpoints. Discover the latter on EC2 rather than inventing them.

**Exit:** SSH and provisioning contract known, budget/destination inputs resolved for the next phase, and sanitized inventory saved.

## 6. Work package B — Rent Instance A, collect Vast details, and verify first boot

### 6.1 Select and create

1. Search via the existing API for **one RTX 3090**, verified host, adequate CPU/network/disk, and a compatible driver. Confirm the actual GPU is not a 3090 Ti or a misleading multi-GPU allocation.
2. Planning envelope: at least 8 effective CPU threads, preferably 32 GB available host allocation with at least 16 GB free during builds; at least 8 GiB `/dev/shm`; at least 60 GB free working disk for builds, caches, captures, and image preparation. Adjust from measured needs and the resolved budget; do not treat these as observed host facts.
3. Compare total rental/storage/network cost and download speed. Save selected offer facts before creation; recheck availability immediately before accepting it.
4. Start EC2 timing **before** the creation request. Use a unique run label and the current deployment baseline without importing files from the 4070 worker.
5. Save the returned instance ID and poll its state through the API. On a timeout, query by instance ID/label before retrying a purchase.

Vast's documented direct creation endpoint is `PUT /api/v0/asks/{offer_id}`; a successful response returns `new_contract`. A request's image and other fields can override template values. Prefer the existing EC2 wrapper and record its actual request. [Vast create-instance reference](https://docs.vast.ai/api-reference/instances/create-instance).

### 6.2 Fetch connection information and use SSH config

1. Open Vast in the user's browser, locate the exact new instance by ID/label, and read its connection details. Cross-check them against API status.
2. Capture current SSH host, mapped SSH port, user, public IP, mapped API/media ports, disk allocation, image identity, and host machine ID. Do not reuse the current 4070's ports or addresses.
3. Back up the user's SSH config and add a distinct host block, for example `musetalk-3090-build-<instance-id>`, using the same identity conventions as the existing worker. Use `ProxyJump` only if the existing topology requires it.
4. Validate with `ssh -G`, connect by alias, and compare `hostname`/GPU identity with the provisioned instance.

### 6.3 Verify the boot before optimizing it

Use the current [startup guide](STARTUP.md), not the older legacy launch examples in `vast_ai_boot.md`.

```bash
cd /workspace/MuseTalk
nvidia-smi --query-gpu=name,uuid,compute_cap,memory.total,driver_version,power.limit --format=csv
df -h /workspace /dev/shm
bash scripts/install_musetalk.sh --check
bash scripts/vast_server_ctl.sh status
curl --fail --silent --show-error http://127.0.0.1:8000/health
```

Inspect boot and server logs for:

- `VAST_ONSTART COMPLETE` and successful recipe verification, with no concealed fallback.
- r5 selected; portable bundle `ampere-plus-r5-srcg50-int8` selected on the initial 3090 boot.
- Stagewise UNet batch 16 and TensorRT TAESD batch 8; verified plans and successful probes.
- Correct native VP8 installation/activation if it is the selected codec path.
- Correct worker ID, mapped URL, control-plane registration/heartbeat, and S3 access.
- A real smoke call with audio, decoded video, idle/talking/smiling transitions, stop/interruption, and teardown.

Record the original source-install startup time. If the current template performs destructive recloning or a forced clean install, document it; do not reuse that behavior in the final Docker template.

**Exit:** Instance A reachable by its own alias, correct hardware, verified portable-r5 boot, smoke call works, and original startup timing retained.

## 7. Work package C — Create the RTX 3090 harness suite

Create explicit, documented 3090 entry points by reusing existing measurement implementations. Avoid a copied benchmark stack that can drift from production. Proposed new interface below is a deliverable to implement; these files do not exist yet.

```text
scripts/repro_3090/
  README.md
  00_check.sh
  10_gpu.sh
  20_aggregate.sh
  30_quality.sh
  40_live.sh
  50_startup.py
  report.py
  profiles/portable.env
  profiles/native.env
  profiles/release.env
```

### 7.1 Common harness contract

- Explicit engine root, TAESD identity, runtime profile, input manifest, output directory, run label, and comparison roots.
- Detect RTX 3090/sm86 and record the actual device. Reject accidental 4070 execution for a report labelled RTX 3090; allow another GPU only through an explicit general-purpose mode that labels it correctly.
- Preserve exact model/input/chin/blending hashes. Changes to data or quality-critical code start a new comparison lineage.
- Assert expected loaded backend/engine hashes, no fallback, required reports present, minimum measurement duration met, and finite outputs.
- Exit nonzero for a crash, OOM, foreign GPU workload, wrong backend, absent report, invalid duration, or failed required criterion. Distinguish `PASS`, `FAIL`, `INVALID`, and a documented historical quality exception.
- Capture sanitized effective settings. Do not copy production tokens into harness env files or report dumps.
- Use `scripts/box_guard.sh` for GPU work. Stop/drain only the owned test server before running an isolated GPU benchmark; never have its model allocations silently compete with the harness.

### 7.2 Specific portability fixes required by the current code

| Current assumption | Required adaptation |
|---|---|
| `repro_400fps/lib.sh` defaults to `sm89` engine roots and published sm89 comparisons | Parameterize the comparison and candidate roots; pass sm86 explicitly |
| `30_benchmark.sh BENCH` always loads the published 4070 engine set | Compare portable-vs-native engines that both load on the 3090; never load sm89 plans there |
| `20_gate.sh` can finish after individual gate failures | New wrapper parses all verdicts and propagates failure/incomplete status |
| `30_benchmark.sh` can continue after a failed stage | New wrapper requires every requested stage's report and verdict |
| `bench_gpu_path.py` defaults to the old live sm89 env | Use `--no-live-env --overlay <explicit-3090-profile>` and record precedence |
| `chin_multistream/gpu.py` reads `.runtime/musetalk_trt_local_sm89.env` | Add the narrowest explicit runtime-env selection, retaining existing defaults for old reproductions |
| `chin_multistream/paths.py` hardcodes six fixture identities, FaceMesh path, and old report directory | Keep the six canonical identities for historical parity; expose required path/output inputs without changing their math |
| Build script falls back to `sm89` if compute-capability detection fails | 3090 preflight must fail when device detection is unavailable; never label that fallback a native 3090 build |
| `repro_400fps/lib.sh` unsets TAESD/timing knobs | Make candidate-specific settings explicit and verify the resulting engine fingerprint; an environment variable alone is not proof it was used |
| Live rig hardcodes 4070 engine overrides, 15 avatar IDs, and CPU masks | Provide 3090 overrides and available CPU sets; restore compatible test avatars or document a separate new workload |

### 7.3 GPU-specific throughput harness

Base it on [bench_stagewise_blocks.py](../scripts/bench_stagewise_blocks.py) and [bench_gpu_path.py](../scripts/bench_gpu_path.py).

- Interleave portable and native block timing on the **same 3090**. Report per-block median/distribution, missing blocks, batch size, and complete-chain sum.
- Run the full GPU path with real captured latents/audio: pinned staging, H2D, UNet, TAESD, postprocessing, D2H.
- Use CUDA events with correct synchronization for device timing and monotonic wall time for completed throughput. Keep diagnostic synchronizations out of the final throughput run.
- Warm up explicitly, then measure at least 180 seconds and repeat. Save 10-second windows and thermal/power telemetry.
- The existing block microbenchmark initializes some buffers synthetically; label this accurately. Its isolated block numbers are diagnostic, not full-pipeline quality or throughput evidence.
- Derive the 400 FPS budget from the measured pipeline: batch 16 requires roughly 40 ms per completed batch, but the decoder/transfer budget must be measured on this GPU.

### 7.4 Aggregate throughput harness

Wrap/adapt [chin_multistream_render.py](../scripts/chin_multistream_render.py) and its `scripts/chin_multistream/` helpers.

- Preserve the canonical six avatars, 240-frame fixture clips, 512×896 composition, 256×256 face output, original chin refinement, and blending.
- Run `T`: six simultaneous streams, two measured windows each at least 60 seconds.
- Run `SUST`: at least five consecutive measured windows each at least 60 seconds, with the GPU thermally settled.
- Run `N15`: 15 streams over the six fixtures for ten approximately one-minute windows, reporting every stream and window.
- Report completed valid frames/shared elapsed time, per-stream frame counts and timing, queue wait, memory, determinism, errors, and hashes.
- Save quality/video captures separately from throughput runs so recording overhead is explicit. Keep native media encoding costs in the live suite; do not imply the offline harness includes live RTP delivery.
- Initial portable target: both valid `T` windows at least 300 FPS. Native release target: both `T` and **every** `SUST` window at least 400 FPS. Aim for margin; 399.96 is below the requested gate.
- A miss must produce a valid failure report, not relabel the workload or lower image quality. Portable missing 300 does not prevent investigating native engines; it remains an unmet baseline target.

### 7.5 Live aggregate and per-client harness

Reuse [load_test_webrtc_v2.py](../load_test_webrtc_v2.py), [live rig](../experiments/live15_r5/README.md), and `live_trace_report.py`.

- Smoke at 1/3 clients; ramp 5/10/15; then one-hour soak at the largest level that passes with margin. Test higher levels only after evidence supports them.
- Record selected encoder, negotiated codec, bitrate, frame size, actual client arrivals, send cadence, fresh/held/idle frames, first-frame latency, and RSS/VRAM growth.
- Preserve existing P1–P3 criteria: every anchored one-second window at least 20 decoded frames; speaking fresh fraction at least 0.995; held runs at most 2; at least 18 content frames per second; max gap 120 ms; over-100-ms gaps at most one per stream per ten minutes and none caused by server send cadence.
- Run an isolated local test for diagnosis and a separate real-browser call through the EC2 routing/TURN path for deployment acceptance. Name their network conditions.
- Run load clients off the server host for the deployment test when possible; otherwise disclose CPU contention. Avoid observer-video startup glitches in scored runs.
- Advertise only the tested live capacity to EC2. `400 / 20 = 20` is not a valid capacity calculation by itself.

### 7.6 Harness validation

Add focused CPU tests for report aggregation, common elapsed-time denominators, engine selection, timestamp ordering, and failure propagation. Include wrong GPU, missing S3 object, missing report, failed child process, too-short measurement, and hidden backend fallback cases. Run a short real-GPU sanity case before a long paid experiment.

**Exit:** documented runnable 3090 entry points exist, negative cases fail correctly, portable baseline measured with them, and reports can be compared without manual JSON editing.

## 8. Work package D — Build native sm86 engines and validate latents/caches

### 8.1 Restore exact inputs and builder dependencies

Use the pinned cu121 matrix: Python 3.10, torch 2.5.1+cu121, torch_tensorrt 2.5.0, TensorRT 10.3.0, and the remaining repository constraints. Keep the native-build comparison on this stack before experimenting with upgrades.

```bash
cd /workspace/MuseTalk
bash scripts/install_musetalk.sh --matrix cu121 --with-legacy-int8 --with-avatar-prep
bash scripts/install_musetalk.sh --with-chin-tools --chin-venv /workspace/SoulX-FlashHead/.venv
bash scripts/repro_400fps/05_fetch_inputs.sh
bash scripts/repro_400fps/00_check.sh --deep
```

`--with-legacy-int8` supplies build-time ModelOpt dependencies; it does not select the legacy serving recipe. Use the approved credential environment without echoing it. Restore canonical fixtures/corpus from their pinned S3 bundles, not by generating replacement portraits or speech.

The corpus has 352 main and 96 holdout bs8 captures. Maintain the holdout split. Main/holdout quality references must remain independent of engine tuning decisions.

### 8.2 Native build

After preflight confirms sm86, use a fresh engine directory:

```bash
cd /workspace/MuseTalk
export MUSETALK_REPRO_OUT="docs/fps_comparisons/rtx3090_r5_${REPRO_RUN_ID:?set REPRO_RUN_ID}/native"
export MUSETALK_REPRO_ROOT=tensorrt_unet_stagewise_sm86_r5_v1
bash scripts/repro_400fps/10_build_engines.sh --set r5 --hardware-compat none
bash scripts/repro_400fps/20_gate.sh models/tensorrt_unet_stagewise_sm86_r5_v1
bash scripts/repro_400fps/30_benchmark.sh models/tensorrt_unet_stagewise_sm86_r5_v1 T SUST Q V N15
```

These are existing build/measurement commands. Run them under the new suite's result validation; a zero shell exit from the old scripts alone does not satisfy the release gate. Set `REPRO_RUN_ID` to this run's unique label before execution. Do not invoke its unadapted `BENCH` or `PAIR` stages on the 3090.

Build order is FP16 `down0rest,up3,tail`; INT8 recipe blocks `down1,down2,down3,mid,up0,up1,up2`; then FP16 `prefix` and finalization. Build native TAESD with hardware compatibility `none`. Capture actual GPU-derived keys; never reuse the 4070's TAESD key or rename a portable plan to look native.

Compare ONNX hashes with the published recipe, where the graph is unchanged. Engine bytes/tactics can differ. Verify complete manifest, every plan hash, native load, direct-vs-CUDA-graph equality, repeat determinism, and source-prefix cached-vs-full equality including shuffled rows.

### 8.3 Avatar latents and production preparations

1. Start from [the 16-character publication manifest](../character_factory/generated/lumatalk_four_language_wardrobe_v2/s3_publication.json): 16 portraits, 48 production pose caches, about 15.5 GB of recorded cache objects.
2. Renew live access and `HEAD` every expected cache and portrait. Compare size, avatar metadata, and recorded ETag; multipart ETags are not SHA-256 content hashes. Download/check object content when validating archive integrity or latent equivalence.
3. Restore original caches on the 3090 and confirm tensor dtype, shape, device mapping, encoder/preprocessing versions, masks, frames, metadata, and source-video hash.
4. Create a separate 3090-prepared candidate set where fresh preparation is required. Use the canonical `/avatars/prepare` pipeline, preserving source clips, crop geometry, mask/chin settings, native avatar encoder, dtype, and reproducible random seed policy.
5. Compare old/new latents and decoded results. If VAE sampling is stochastic, document and control it rather than attributing random differences to the GPU. Reuse the compatible original set when it is already correct; regenerate all required poses if a validated preprocessing/encoder change requires it.
6. Exercise every production avatar's idle/talking/smiling path. Keep the canonical six-avatar benchmark distinct from this 16-avatar deployment audit.
7. Save any new cache under a distinct ID/version or prefix, upload through the existing store, verify remotely, and record the mapping. Do not overwrite existing `avatars/v15/` objects solely to label them RTX 3090.

## 9. Work package E — Reach 400+ FPS while preserving quality

### 9.1 Quality checks before promotion

- Run UNet main and holdout gates with their existing bars: maximum per-file MAE 0.01 and strict max-absolute error 0.5. Preserve failing verdicts.
- Run TAESD load/probe/full-frame gate, existing max 3 LSB and mean 0.2 LSB, plus fused/nonfused consistency.
- Produce identical-input comparisons against canonical reference outputs and the portable-r5 outputs captured on this 3090; compare against saved 4070-r5 raw results when available. Do not attempt to execute sm89 plans on sm86.
- Run lip correlation, mouth aperture, landmarks, mouth/full-face differences, temporal flicker, and chin/seam checks through the existing quality tools. Include teeth, beard, skin/identity consistency, closed-mouth silence, and transitions in labelled videos.
- Include real human speech in evaluation. Keep audio cross-attention K/V projections FP16: prior experiments showed TTS-only calibration clipping real-speech features.
- Preserve full-height decoding and 100% chin refinement. Do not reach 400 by reducing resolution, omitting composition, dropping valid frames, removing avatars, or weakening the holdout.

### 9.2 Autonomous quality acceptance and selection

Codex must make the quality decision itself. Use two independent result fields: `strict_original_gates` and `quality_parity_with_reference`. The original strict numerical failures remain visible even when a candidate matches the established reference. Historical prose saying a human decision was pending does not create an overnight approval step under the user's current instruction.

Before evaluating optimized candidates, create `quality/quality_acceptance.json` with exact reference file hashes, input sets, metric definitions, bounds, measurement-noise allowance, and the following policy:

| Check | Autonomous acceptance rule |
|---|---|
| Source assets, framing, masks, chin strength, audio, and resolution | Match the comparison contract; no workload or visual-quality shortcut |
| Finite output, determinism, CUDA graph/direct equivalence, source-prefix equivalence | Every applicable hard invariant passes; any failure rejects the candidate |
| UNet mean error | Original MAE limit of 0.01 still applies to main and holdout; also compare against the reference envelope |
| UNet strict maximum error / TAESD strict gate | Report original verdict unchanged. An inherited reference exception is eligible only if the candidate is no worse than the frozen, measured reference envelope; no new or larger exception |
| Pixel, lip, aperture, landmark, and temporal metrics | Per-avatar non-regression against stored native-4070 r5 metrics and the repeated same-input portable-3090 reference; preserve direction of improvement for each metric |
| Visual inspection | Codex inspects every canonical avatar and each production avatar/pose sample, including highest-error and transition frames; unexplained new artifacts reject the candidate |
| Live output | Decoder-visible image quality and codec settings remain consistent; cadence/freshness failures do not pass just because GPU FPS is high |

Derive the reference envelope from exact saved JSON values, not rounded README values. For an error metric use the worst observed reference value over the applicable reference runs plus only a noise allowance measured from repeated reference evaluations. For a similarity metric use the lowest reference value minus that same independently measured allowance. Record the source of every bound before candidate results are evaluated. Do not widen tolerances after seeing a candidate fail. If a reference metric or raw capture is absent, recreate the reference evaluation using the pinned portable recipe/canonical outputs, document that limitation, and require the remaining evidence; do not invent missing 4070 values.

The portable and native r5 share the intended graph/precision recipe, but their exact outputs differ. Report direct candidate-vs-reference differences as well as both candidates' quality relative to the canonical accepted outputs. Existing strict failures such as approximately 5 LSB TAESD max error remain labelled inherited failures; they are never rewritten as a strict pass.

Codex visual procedure:

1. Decode comparison videos to aligned frames at the start, middle, end, peak mouth opening, silence, largest numerical differences, and pose transitions. Include motion/contact sheets to detect flicker and geometry drift.
2. Inspect matched A/B crops at native resolution: lips, teeth, chin edge, beard, cheeks, eyes, and a full-frame view. Inspect all six canonical identities and samples of all 16 production avatars across idle/talking/smiling; inspect additional frames wherever metrics flag an outlier.
3. Write `quality/visual_inspection.json` with actual inspected asset/frame references, observed differences, and pass/reject reasons. Record any inability to render/inspect evidence as incomplete validation, not a pass.
4. Write `quality/quality_decision.json` with `accepted`, `rejected`, or `incomplete`, the immutable candidate identity, all gate results, any inherited exceptions, and the selected alternative if rejected. Attach labelled comparison videos for tomorrow's optional review.

Select the highest-quality passing candidate that satisfies 400+ FPS. If candidates are visually indistinguishable and metric differences are within measured noise, prefer the stable faster candidate with simpler runtime behavior. Codex performs this decision without waiting for user input.

### 9.3 Optimization sequence

| Order | Experiment | Required check |
|---|---|---|
| 1 | Native plans with the unchanged r5 precision/graph | Establish isolated benefit over portable plans |
| 2 | Block/tactic timing, thermal stability, power limits, host CPU and memory stalls | Use measured 3090 bottlenecks; do not copy the 4070's 220 W assumption or change power limits without host support |
| 3 | Same-precision TensorRT build options and fresh timing caches | Record effective options; repeated paired timings and quality comparison |
| 4 | TAESD optimization-level experiment and exact postprocessing/transfer improvements | Explicit candidate fingerprint; no silent ignored settings |
| 5 | Batch scheduling, CUDA graphs, memory reuse, transfers, thread allocation | Preserve workload; measure latency and fairness as well as aggregate FPS |
| 6 | Source-prefix caching in the serving scheduler if profiling justifies it | Existing cached-vs-full equality, correct invalidation on avatar/pose/source changes, and multi-avatar ordering tests |
| 7 | Media handoff/encoder/event-loop work | Optimize live delivery based on CPU traces; GPU capacity may already be sufficient |
| 8 | A new selective-INT8 recipe only if still necessary | Separate artifact lineage, full holdout/pixel/video validation; no quality trade hidden behind FPS |

Run one change at a time, paired with the baseline on the same host. Recheck final results without intrusive profiling. When 400+ is achieved, test whether some precision can be restored while retaining the throughput target; select the highest-quality candidate that meets the measured requirement.

If the 3090 cannot meet the target within the hardware and quality constraints, record the best valid candidate, bottleneck evidence, and remaining gap. Continue the Docker and fresh-boot work with a clearly labelled candidate image where this can be tested safely. Do not promote a quality regression, lower the 400+ acceptance threshold, or mark the full goal achieved. If the existing named template is temporarily used for an isolated candidate trial, ensure production provisioning stays on the saved release and restore the template automatically when promotion criteria fail. This contingency requires no overnight user decision.

**Exit:** both `T` windows and every `SUST` window are at least 400 FPS; stability/live limits are documented; quality decision and artifact hashes identify one selected candidate.

## 10. Work package F — Publish the native bundle and make r5 select it

1. Package the selected sm86 UNet plans and all required TAESD decoder/post plans, metadata, probe references, and manifests. Resolve symlinks or include all targets; no artifact may depend on Instance A's filesystem.
2. Use [trt_artifact_bundle.py](../scripts/trt_artifact_bundle.py) and [bundle instructions](trt_artifacts/README.md). Pass explicit required files/directories and clear legacy optional defaults when building a serving-only bundle.
3. Upload to a new checksum-addressed key, for example `trt-artifacts/rtx3090/r5-srcg50-int8/sha256-<actual-sha>/<archive>.tar.gz`. Record real size, SHA-256, and tool manifest; verify a fresh restore before selection.
4. Create `configs/trt_bundles/rtx3090-r5-srcg50-int8.json` with the actual native engine key, TensorRT version, directories, TAESD key, sizes, checksum, and sidecar location. Obtain the key with the repository's engine-key helper on the real 3090.
5. Insert the 3090 candidate before the portable fallback in `configs/recipes/r5.env`, preserving the 4070 candidate: `rtx4070super-r5-srcg50-int8 | rtx3090-r5-srcg50-int8 | ampere-plus-r5-srcg50-int8`.
6. Add meaningful resolver tests: 3090 picks its restored native bundle; 4070 retains its bundle; other supported GPUs retain portable selection; missing/corrupt/incompatible bundles cannot masquerade as a verified native release.
7. Boot through `vast_onstart.sh` and `vast_server_ctl.sh`, confirm resolved native selection, and rerun the final performance/quality smoke from the actual launch configuration.
8. Persist changes and reports in Git, push the execution branch or autonomously validated release revision, and record the commit used for the image. Store large media and plans in S3 with links from Git. Follow the repository's actual branch/protection rules; do not introduce a human-review dependency that the repository does not require.

Maintain the original portable bundle and template as rollback artifacts. A portable fallback can restore service but must be reported as degraded relative to the native 400+ release.

## 11. Work package G — Build the Docker release after performance validation

### 11.1 Implement a reproducible image

Proposed deliverables: `docker/musetalk/Dockerfile`, a narrow `.dockerignore`, `docker/musetalk/entrypoint.sh`, registry README, and an image-validation script. Build from the exact selected code revision and a compatible CUDA base pinned by digest.

Image contents:

- Pinned server runtime and required OS libraries, ffmpeg, native VP8 payload and license files.
- Model weights required for the selected serving path, where redistribution permits them; otherwise a documented authenticated, checksum-pinned retrieval path.
- Validated RTX 3090 native bundle and enough verification metadata to establish its provenance without a build at boot.
- Startup code, bundle resolver, secret/bootstrap integration, and the supported avatar-prep dependencies if the final worker is expected to offer `/avatars/prepare`.
- A small permitted readiness fixture if needed; no private customer media, original SSH identities, credential caches, `.runtime` secrets, logs, or entire development workspace.

Start with a complete functional image, then reduce it by measured removal of unused build tools, package caches, calibration corpora, development-only quality tools, and redundant layers. Keep any required prep capability explicit; a serving-only image must not quietly retain a promise of avatar creation it cannot fulfill.

Use explicit `COPY` sources or an allowlisted build context. Credentials needed for private build inputs must use BuildKit secrets/SSH mounts, not Dockerfile `ARG`/`ENV` or copied files. [Docker build secrets](https://docs.docker.com/build/building/secrets/).

Do not assume the GPU-less image builder can create CUDA graphs or valid GPU probes. Copy validated native plans and perform device/runtime verification and per-process warmup on the actual worker.

### 11.2 Startup contract for the image

- Choose a fixed code/venv/model layout and prove it survives Vast's mounts. If `/workspace` is overmounted, baked contents placed there may be hidden; use a durable image path or explicit minimal initialization rather than a full checkout copy.
- No `git clone`, `git pull`, `rm -rf /workspace/MuseTalk`, `SETUP_CLEAN=1`, apt/pip installation, ModelOpt build, or TensorRT engine build during a successful normal release boot.
- Replace the old destructive on-start wrapper. Reuse canonical startup verification instead of a second divergent server launcher.
- For the baked image, `AUTO_SETUP=0`/`SETUP_CLEAN=0` is the intended runtime policy after the image passes install checks. Missing dependencies should fail with a concrete cause, not trigger a slow install that gets scored as success.
- Ensure bundled engine adoption/restore stamps are generated correctly from the verified image artifacts. Do not ship copied host-specific readiness or self-test stamps as if the new GPU had passed them.
- Secrets, per-instance public URLs, IDs, registration, TURN/media mapping, and selected avatar warming still happen at runtime.
- Handle termination/drain cleanly; make startup idempotent. Verify restart does not duplicate API/TURN processes.
- Measure SHA verification, import, model loading, CUDA-graph warmup, and avatar warm costs before optimizing them. Keep correctness checks in the ready path.

### 11.3 Build, pull, and run validation

Build on the available builder for the actual architecture, normally `linux/amd64`. Use an immutable tag such as `<namespace>/musetalk-r5:rtx3090-<date>-<git-sha>` and record its registry digest.

Validate:

1. CPU-only build/import/install checks succeed without a visible GPU.
2. Clean GPU container with sufficient shared memory starts from the image and runtime secret injection alone.
3. Correct sm86 backend, probes, model versions, and encoder are loaded; no runtime package/engine construction occurs.
4. Production avatar restoration and actual EC2 registration work from fresh state.
5. Native aggregate `T`/`SUST` still meet 400+ FPS inside the image. The live codec and quality are unchanged.
6. Restart/termination and negative cases (missing credentials, missing/corrupt bundle, wrong GPU) behave correctly.
7. The **pushed** digest can be pulled and inspected from another environment, not merely run from the builder's local cache.

Use Docker's GPU flags only where a Docker daemon is actually available. Final Vast instance validation is still required even if a local GPU container passes.

## 12. Work package H — Publish a usable, searchable image

1. Create/use the resolved registry namespace and repository, publish the tested immutable tag, and retain its digest. A moving tag can be a convenience but is not the recorded deployment identity.
2. Provide a concise registry description and README with GPU/runtime requirements, startup command, ports, shared-memory requirement, external secrets, model/license notes, verified FPS scope, and rollback version.
3. Docker Hub public repositories appear in search and permit public pulls; private ones do not appear in public search. Use the requested searchable configuration for the distributable image. [Docker Hub access/visibility](https://docs.docker.com/docker-hub/repos/manage/access/).
4. Verify the exact repository page, `docker search <namespace>/musetalk-r5`, tag, manifest/digest, and clean pull. If indexing is delayed, report searchability as pending rather than asserting it passed.
5. Inspect image history/layers and the final filesystem for unintended credential/host state. Confirm large private avatar caches are still fetched using the approved runtime path.

**Exit:** registry URL, searchable result, immutable digest, clean-pull evidence, and image-based 3090 test reports saved.

## 13. Work package I — Update the existing Vast template through the user's browser

Use computer interaction with the user's authenticated browser, as requested. Backend APIs may read back the configuration for verification; they do not substitute for the requested browser-edit step.

1. Navigate **Vast.ai → Templates → My Templates**.
2. Open the exact **`(NEEDS UPDATES) Musetalk`** template. Match owner and template ID/hash, not just similar text.
3. Save a sanitized before snapshot: template identity, image/tag/digest, launch mode, on-start script, port mappings, disk/shared-memory requirements, environment variable names, filters, and privacy. Preserve secret-bearing rollback data in protected storage only.
4. Set the image field to the published release tag/digest supported by the UI. Vast pulls that image; do not add a nested `docker pull` inside the rented container to try to replace its running filesystem.
5. Use the tested launch mode. In Vast SSH/Jupyter mode, Vast replaces the image entrypoint, so the template's on-start must explicitly invoke the packaged bootstrap. ENTRYPOINT mode requires the image itself to provide the desired access/lifecycle behavior. [Vast template launch settings](https://docs.vast.ai/guides/templates/template-settings).
6. Replace the old clone/delete/install commands with the exact packaged startup entry point. Retain the required secret-bootstrap reference, not plaintext credentials.
7. Set the validated r5/native selection, appropriate CPU/RAM/disk and shared-memory requirements, API/media mappings, and dynamic worker identity. Confirm TURN/relay routing still matches EC2 expectations.
8. Save the **existing** template. Reopen it and verify persisted values. Record any changed template hash/version and update EC2's reference if required.
9. Inspect EC2's final outgoing create request. Remove obsolete hardcoded image/on-start overrides that would select the old behavior despite the template edit.

If the UI cannot express a required setting, document the actual limitation and implement it in the appropriate supported request/image mechanism. Do not claim an unsaved form is a deployed template.

**Exit:** browser screenshots/readback and sanitized API comparison prove the correct existing template now resolves to the tested image and startup contract.

## 14. Work package J — Launch Instance B and measure true autoscaling readiness

### 14.1 Add startup instrumentation before the request

Implement `scripts/repro_3090/50_startup.py` with the discovered EC2 API contract. It should start the timer, submit one idempotently tracked creation request, poll provider/worker states at a recorded interval, collect startup events, initiate a readiness call, and write JSON plus a timeline report. Use monotonic time for durations within each process and UTC for cross-system correlation; never subtract unrelated hosts' monotonic clocks.

| Event | Meaning / evidence source |
|---|---|
| `request_started` | EC2 controller begins scale-out request |
| `provider_accepted` | Vast instance ID received |
| `image_pull_started/finished` | Provider events/logs if exposed; otherwise mark unavailable or bounded by observations |
| `container_started` | First reliable container/provider event |
| `ssh_ready` | First successful noninteractive SSH probe from the operator |
| `bootstrap_started` | First timestamp from packaged startup |
| `secrets_ready` | Required credentials/config loaded, values never logged |
| `artifacts_verified` | Correct weights and native engine manifests validated |
| `models_loaded` | Runtime model/engine load complete |
| `gpu_warm` | Required graph/probe/warmup work complete |
| `health_ready` | Health endpoint succeeds and backend verification passes |
| `registered` | EC2 accepts correct worker identity/endpoint |
| `avatar_ready` | Selected readiness avatar and required pose/idle caches warm |
| `routable` | EC2 scheduler will assign the test call to this worker |
| `first_usable_frame` | EC2-routed WebRTC client decodes the first valid response frame to test audio |
| `steady_call_ready` | Short stream meets freshness/cadence checks without a late deferred build/download |

Record parallel operations as intervals; do not add overlapping durations as if they were sequential. Mark polling uncertainty and unavailable provider metrics explicitly. Separately report the time to first frame and the time until its validity is established.

### 14.2 Launch and observe a genuinely new instance

1. Provision a **different** RTX 3090 via the existing EC2 path using the updated template ID/hash and image digest. Record offer, physical host ID, instance ID, and all request overrides.
2. Obtain its current connection information from Vast; add a separate SSH alias such as `musetalk-3090-release-<instance-id>` and connect for observation.
3. Do not fix dependencies, copy the builder's caches, or manually run a missing startup command. If any repair is needed, this fresh-boot trial fails; fix the image/template and rerun on another fresh instance.
4. Confirm the image revision/digest and native bundle actually used, normal startup markers, unique worker registration, S3 restoration, and valid external media routing.
5. Complete a real call through EC2 and a short cadence/freshness validation. A container marked running or SSH-ready is not a ready autoscaling worker.
6. After startup timing is captured, isolate/drain the owned test worker and run the release aggregate/live checks. Confirm 400+ FPS survives deployment to a second physical 3090 host; report host variation separately.

## 15. Work package K — Optimize startup latency and repeat

### 15.1 Measure three distinct cache states

| Trial | What is cold | What it proves |
|---|---|---|
| Fresh instance on a different/provider-uncached host | Image layers where observable, instance disk, runtime state, selected avatar cache | Autoscaling cold-start behavior |
| Fresh instance where provider image layers are reused | Container state remains new; image-cache status documented | Benefit of layer reuse, not equivalent to a cold pull |
| Restart of the same instance | Process/GPU state; persistent disk may remain warm | Recovery/restart behavior, not new-capacity provisioning |

Aim for at least three independent fresh-instance trials and three restart trials within budget, using sequential rentals where possible. A new instance does not prove a cold image pull; if the provider does not reveal cache state, label it unknown. With a small sample report every observation, median, and worst observed; do not invent a statistically reliable p95.

### 15.2 Optimize the observed critical path

1. **Image download/extraction:** measure compressed bytes and throughput; remove redundant weights/engines/build tooling; choose layer boundaries that preserve reuse across code revisions.
2. **Baked vs downloaded assets:** compare a complete native image with a smaller runtime plus checksum-pinned downloads if image transfer dominates. Include both transfer and extraction time; moving bytes from Docker to S3 is not automatically faster.
3. **Install/reclone work:** eliminate it from normal startup; ensure an image revision change does not trigger a clean installer through stale fingerprints.
4. **Artifact verification:** avoid duplicate downloads and unnecessary repeated full reads while preserving integrity guarantees. Use content-addressed manifests and the existing verified restore/adopt mechanism.
5. **Python/model warmup:** remove measured unused imports/backend initialization; warm only the batch shapes actually served. GPU graphs and contexts must be initialized on the target host.
6. **Avatar restore:** restore the requested/hot avatar and required poses first; do not fetch all roughly 15.5 GB of 48 caches before any worker can become useful. Measure one-avatar readiness and full-roster readiness separately. Background warming must not degrade active calls.
7. **Parallel work:** overlap independent secret/config retrieval, artifact availability checks, and other proven-independent I/O, with bounded resource use and a clear readiness barrier.
8. **Provider selection:** use actual disk/download performance and compatible CPUs/drivers along with price; persistent slow hosts can dominate deployment latency.
9. **EC2 orchestration:** remove unnecessary sleeps, reconcile provider states correctly, and route immediately after the explicit readiness condition passes. Avoid polling aggressively enough to hit API throttling.
10. **Warm capacity:** only after cold-start improvement is measured, evaluate whether a small warm pool or earlier scale-out is warranted for the observed arrival patterns and approved cost. Label it separately from faster boot.

Set a concrete latency objective from the original measured baseline and the user's desired autoscaling behavior before choosing a final candidate. Prefer the shortest reproducible request-to-usable-call time that preserves 400+ aggregate FPS, validated live capacity, and quality. Record failures/timeouts as failures, not omitted observations.

If startup optimization changes code, dependencies, engine selection, image layers, or warmup, publish a new immutable digest, update the template, and repeat the fresh-instance test. The final report must describe one matching tuple of code commit + engine bundle + image digest + template version + EC2 provisioning revision.

### 15.3 EC2 autoscaling readiness and capacity

- Distinguish starting, healthy, avatar-ready, routable, busy, and draining workers using the existing registry model or the smallest necessary extension.
- Never count a starting worker's full capacity as available or route users before the required readiness checks complete.
- Configure `LINGUA_WORKER_DEFAULT_CAPACITY` from the 3090 live soak result with headroom; do not copy the historical 4070 value uncritically.
- Preserve `/avatars/{id}/cache/warm` before routing when that avatar is not ready.
- Size launch deadlines and scale-out lead time from observed end-to-end latency plus an explicit margin. Separate capacity exhaustion from a broken boot.
- Run one controlled scale-out and drain/scale-in cycle with test traffic, confirming no broken in-flight calls, orphaned instances, duplicate workers, or stale capacity accounting.

## 16. Single-goal execution, checkpoints, and resume prompt

Execute these checkpoints sequentially inside the single overnight goal. Do not yield at a checkpoint expecting the user to launch a new session. CPU-only documentation/container preparation can overlap GPU work where it cannot contaminate measurements; only one owner modifies each artifact or template. Separate sessions are a recovery option if execution is interrupted, not a requirement.

| Checkpoint | Scope | Must leave behind |
|---|---|---|
| 1 | Packages A–B: EC2 discovery, Instance A, SSH and portable boot | API contract, budget, instance/alias, environment, original cold-start trace |
| 2 | Package C: 3090 GPU/aggregate/live/startup harnesses | Runnable scripts, failure tests, documented profiles, portable baseline |
| 3 | Packages D–E: native engines, avatar/latent audit, quality and 400+ FPS | Native candidate, all reports, selected quality decision, S3/Git evidence |
| 4 | Package F: native bundle and default r5 selection | Uploaded/restored bundle, descriptor, resolver tests, release commit |
| 5 | Packages G–H: Docker build and registry | Dockerfile/entrypoint, immutable digest, registry/searchability and GPU validation |
| 6 | Packages I–J: browser template update and fresh Instance B | Before/after template, EC2 request, new SSH alias, complete startup timeline and call |
| 7 | Package K: startup optimization and autoscaling validation | Repeated trials, final release tuple, measured latency/capacity and rollback instructions |

Suggested text to use when launching the overnight goal:

> Execute `docs/RTX_3090_R5_DOCKER_AUTOSCALING_EXECUTION_PLAN.md` end to end as one autonomous goal. Read and maintain `run_state.json`. Connect to EC2, provision the RTX 3090, create the GPU and aggregate harnesses, measure portable r5, build and optimize native engines/caches for sustained 400+ FPS, perform numerical and visual quality validation yourself, publish the Docker image, update the existing Vast template through my browser, launch a fresh 3090 through EC2, and optimize request-to-usable-call latency. Do not wait for manual quality review or routine confirmations. Respect established spending limits, preserve the existing 4070/production services, automatically roll back failed candidates, and keep all artifacts in Git/S3/registry. Continue across checkpoints and context resets. Mark the goal complete only when the plan's required outcomes are evidenced; report hard access/resource blockers and finish all independent work if an external prerequisite is unavailable.

## 17. Completion checklist and rollback

- [ ] EC2 SSH and the actual provisioning API are documented and usable.
- [ ] Instance A was created by that API, checked in Vast, and accessed with the user's SSH-config conventions.
- [ ] Portable r5 was genuinely measured on the RTX 3090; 300+ target status is explicit.
- [ ] RTX 3090 GPU, aggregate, live, and startup harnesses exist and reject invalid runs.
- [ ] Native sm86 plans and matching TAESD artifacts were built and verified on the 3090.
- [ ] Required avatar latents/caches were audited or reprepared, and every expected S3 object was checked live.
- [ ] Native full-recipe `T`/`SUST` satisfy 400+ FPS; Codex's numerical and visual quality decision and live-call limits are recorded without a human-review dependency.
- [ ] Native bundle is persisted and default r5 chooses it on a 3090 without breaking other GPUs.
- [ ] Docker image is reproducible, pullable, searchable, secret-free, and preserves the measured performance.
- [ ] The exact existing `(NEEDS UPDATES) Musetalk` template was updated through the browser and read back.
- [ ] EC2 creates Instance B from that template/image without overriding it with obsolete settings.
- [ ] Instance B reaches a usable call with no manual repairs; all startup stages and cache states are reported.
- [ ] Startup optimization is validated by repeated fresh-instance trials, not just restarting the development worker.
- [ ] EC2 capacity/readiness, controlled scale-out, drain, and scale-in behavior are tested.
- [ ] Git commits, bundle hashes, image digest, template version, deployment config, reports, and next operator instructions agree.
- [ ] Paid resources are either intentionally retained with owner/lifetime or released after artifact persistence is verified.

Rollback must identify the previous template snapshot and image/code revision, the portable r5 bundle, and the control-plane settings to restore. If the native image fails, mark the worker unroutable, preserve logs, and redeploy the known-good release. Do not destroy the 4070 worker, overwrite old S3 caches, or publish experimental precision changes as the established r5 release.

## 18. Source map and precedence

Use current code and runtime inspection ahead of older prose. In particular, the legacy 3090 boot guide and some reproduction prose predate the current ordered bundle selection.

| Source | What to use it for |
|---|---|
| [STARTUP.md](STARTUP.md) | Current installer, boot chain, recipes, bundle restore, destructive old template behavior |
| [r5.env](../configs/recipes/r5.env) | Actual active r5 levers, commented-out TTS/telemetry settings, ordered bundle selection |
| [Native 4070 descriptor](../configs/trt_bundles/rtx4070super-r5-srcg50-int8.json) | Descriptor structure; do not copy its GPU key into the 3090 descriptor |
| [Portable descriptor](../configs/trt_bundles/ampere-plus-r5-srcg50-int8.json) | Exact portable-bundle checksum and compatibility range |
| [TRT artifact guide](trt_artifacts/README.md) | Bundle creation, immutable upload, restore/adopt, old 3090 legacy distinction |
| [Reproduction package](../scripts/repro_400fps/README.md) | Pinned inputs, dependency paths, benchmark definitions and historical caveats |
| [400 FPS engineering record](fps_comparisons/4070s_400fps_20260928/README.md) | Selective INT8 decisions, audio sensitivity, optimization experiments |
| [Portable measurements](fps_comparisons/ampere_plus_r5_20260930/README.md) | 307 FPS provenance and missing 3090 measurements |
| [Live measurements](fps_comparisons/live15_r5_20260929/README.md) | P1–P3, event-loop limitations, one-hour soak caveats |
| [Live test driver](../experiments/live15_r5/README.md) | Reusable live harness and isolation rules |
| [Worker secrets](musetalk_worker_secrets.md) | Runtime credential bootstrap, registration requirements, per-instance identity |
| [Avatar publication evidence](../character_factory/generated/lumatalk_four_language_wardrobe_v2/s3_publication.json) | 16 avatars/48 cache mapping and explicitly historical verification status |

External API/template/registry references linked above were inspected while preparing this plan. Recheck the installed CLI/API schema and actual UI at execution time; never assume a historical example exactly matches the current account or template.
