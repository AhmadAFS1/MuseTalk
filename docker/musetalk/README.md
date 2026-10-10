# MuseTalk r5 native RTX 3090 image — release groundwork

For the current private-GHCR delivery sequence, GitHub/AWS access handoff, and
fresh-instance boot-latency measurement/rollout, use the
[Docker fast-startup plan](../../docs/DOCKER_FAST_STARTUP_PLAN.md).

This is an **unbuilt candidate implementation**, not a published image or a claim
of 400 FPS/startup latency. The preserved native-v1 archive now has an explicitly
nonpromotable descriptor and `r5_3090_candidate.env` recipe. Neither is selected
by the default r5 configuration. Its exact archive/hash and engine paths are
preserved; this does not invent a validated release manifest. Assembly fails
until all required reviewed inputs exist. Do not point production or
the named Vast template at an unvalidated candidate.

### Private GHCR build/publish implementation

`musetalk-ghcr.yml` establishes the package with a non-sensitive placeholder,
verifies actual private visibility/anonymous denial, and independently pulls it.
Its dependency image remains a non-serving diagnostic.

`musetalk-ghcr-candidate.yml` now implements the **full candidate** build path:
exact clean source commit, separately hashed metadata, direct private S3 archive
downloads, existing manifest/source/model/license/evidence checks, full Docker
build/CPU check, every-layer credential scan, private candidate publication,
then an independent digest pull and offline CPU check on a fresh runner. It has
not yet built or published a full candidate. It cannot publish a validated tag
or make a candidate eligible for production.

The one-time Actions secret `MUSETALK_PRIVATE_BUILD_INPUTS` is a JSON object with
exactly the keys `musetalk-docker-metadata.tar.gz`, `weights.tar.gz`, and
`native.tar.gz`. Each value must be a presigned HTTPS URL for the existing
`lingua-musetalk-s3-storage.s3.us-east-1.amazonaws.com` endpoint, an approved
`docker-build-inputs/` or `trt-artifacts/` object prefix, and a signature lifetime
of at most one hour. Prepare fresh URLs just before dispatch, pass them through
the secret channel, and delete/rotate the secret after the run. Never put URLs,
AWS keys, registry tokens or private inputs in Git/GitHub public release assets.
The exact weight archive is now staged privately in the existing S3 bucket, with
version-pinned size/SHA/encryption readback and anonymous denial. The native archive
was preserved previously. Metadata upload, Actions secret provisioning and full
candidate dispatch have **not** been performed. No new AWS role,
builder, bucket, ECR resource or permanent compute is required by this path.

To dispatch on the task branch before the workflow is registered on the default
branch, commit `.github/musetalk-candidate-request.json` with exactly
`source_revision` (reviewed earlier clean 40-character commit) and
`metadata_sha256` (actual 64-character metadata tarball hash). The request commit
is not the image source commit: generate and verify the source inventory for
the pinned earlier commit. There is intentionally no live request file yet.
Once registered, manual workflow inputs offer the same pinned contract.
Do not dispatch until the reviewed actual manifest and secure inputs exist.
No build/publish request overrides existing redistribution or evidence gates.

Lingua's companion `codex/musetalk-ghcr-rollout` branch supplies default-off,
digest-pinned headless launches with a separate pull-only token, per-launch
secret refresh, response redaction, and both-direction $1.50/TB transfer caps.
See its `backend/docs/MUSETALK_GHCR_ROLLOUT.md`; production remains unchanged.

## Runtime contract

The image keeps code at `/opt/musetalk/app`, Python 3.10/cu121 at
`/opt/musetalk/venv`, and native VP8 at `/opt/musetalk/native_vp8`. A Vast
`/workspace` mount cannot hide them. Mutable logs/uploads/avatar results live in
`/workspace/musetalk-runtime` (`MUSETALK_STATE_DIR` can choose another absolute
persistent location). Never mount over `/opt/musetalk`.

The dedicated `scripts/vast_docker_onstart.sh` is the Dockerfile's entrypoint
(under `tini`). Use the digest-pinned image with Vast **headless `runtype:"args"`**
and its default `serve` command. Do not layer the source-install template over it,
run nested Docker, or pass a registry token into the worker environment. The host
pulls through `image_login`; the startup script runs **inside the pulled image**.
It rejects missing baked files, emits a timestamped `VAST_DOCKER ENTRYPOINT`
marker, and execs the existing lifecycle wrapper. This timestamp is after pull,
not scale-up-to-ready time. `check` runs the existing offline CPU contract only.

`entrypoint.sh` invokes the existing `vast_onstart.sh` → `vast_server_ctl.sh` →
`run_musetalk_server.sh` chain. Only the opt-in `MUSETALK_IMMUTABLE_RUNTIME=1`
changes source-install behavior: every nonzero install check is fatal, coturn
cannot be installed at runtime, and baked native files/host compatibility must
verify before launch. The normal source installer remains unchanged. The image
forces r5, strict backend verification, exact native probes (zero tolerance),
native sm86 plans, no engine provisioning, no TAESD builds, no TensorRT/encoder
fallback, and offline Hugging Face model access.
The release policy also pins the recipe file, overriding caller/secret values.
Native-v1 diagnostics are allowed only in the standalone candidate channel;
they cannot be relabeled as a validated release.
Release policy is reapplied after runtime secret bootstrap. It never clones,
pulls, invokes apt/pip, or constructs engines during successful normal boot.

All baked files are hashed on startup. The canonical bundle verifier then writes
a fresh artifact-integrity stamp; this is **not** a copied GPU-selftest or
readiness stamp. CUDA contexts, graphs, load probes, registration and avatar
warming still run on the target worker. Repeated full hashing is deliberately
not optimized until its measured cost is available.

### Default-off model-load experiment

`MUSETALK_SKIP_EAGER_UNET=1` skips constructing/loading the eager UNet that the
API otherwise loads before replacing it with TensorRT. It retains the original
SD-VAE encoder, positional encoding, Whisper, face parsing and runtime warming.
The default remains `0`; no production configuration or image policy enables it.
The canonical checkpoint remains required by the image inventory/install checks.
This change does not reduce image bytes. A strict-owned3090 local0/1/1/0 test
now measures original model initialization at9.12/9.10s versus3.19/3.18s with
the flag: about5.92s saved, with identical tested UNet, decoder and seeded
diagnostic VAE-encoder output hashes. See the
[paired diagnostic report](../../docs/fps_comparisons/rtx3090_r5_20261008/startup/a5_model_startup_pair_summary_0824.json).
These fresh local processes used warm filesystem cache and quality-rejected
native-v1 artifacts; this is not Docker boot or EC2 usable-call acceptance.

Opt-in requires CUDA, `MUSETALK_UNET_BACKEND=trt_stagewise`,
`MUSETALK_TRT_FALLBACK=0`, stagewise SHA verification and exact probes enabled,
and probe tolerance `0`. Missing/corrupt/unexpected backends are fatal: no eager
fallback or engine repair. The loaded backend must be FP16-interface srccache,
batch16. `MODEL_STARTUP` records base-model loading and full initialization/warm
durations without credentials. Nine operator CPU contracts pass using actual
manager method bodies with mocked models. The narrow real-GPU diagnostic
equality test now passes; full controlled-seed avatar preparation, live behavior
and paired image-based boot timing remain required before enabling this in an
image or claiming end-to-end startup improvement.

The supervisor serializes lifecycle ownership with `flock`, watches API/TURN
processes, and calls canonical drain/stop on SIGTERM. Python handles TERM during
bootstrap immediately, interrupts/reaps that process group before drain, and
enforces a total `MUSETALK_SHUTDOWN_TIMEOUT_SECONDS` deadline (default 360 s).
Canonical stop may otherwise spend up to two drain-timeout intervals; the image
deadline bounds it. Provider stop grace must exceed the complete image deadline
(for Docker, start with `--stop-timeout 390`, then measure). The image uses fresh
per-supervisor `/run/musetalk/processes-*` PID files, unique owner markers and
Linux start ticks. Its opt-in ctl path rechecks identity and signals by pidfd,
so recycled PID numbers cannot target unrelated processes. The host kernel and
container syscall policy must support `pidfd_open`/`pidfd_send_signal`; this is
probed before launch. GPU/container restart and real call drain still require a
real GPU test. `/health` is **not** proof of a usable call.

This first full-API image requires avatar-prep dependencies and all required
weights before readiness; the tightly scoped private-runtime option below may
keep unresolved-redistribution weights out of public layers. Local
Kokoro capability is explicitly chosen in the manifest. If false, its endpoint
is explicitly disabled; this must agree with production TTS routing. If true,
the pinned materialized HF cache must be included, so the first TTS request
cannot cause an untracked model download. No private avatars, portraits, voice
recordings, or credentials are allowed in the image.

## Inputs and release manifest

Provide an external audited directory, used as BuildKit named context `release`:

```text
release.json
native.tar.gz                  # Serving-only trt_artifact_bundle.py output
weights.tar.gz                 # Regular model files only; no links or ./ prefix
licenses/...                   # Actual redistributable-model/dependency notices
evidence/...                   # Small hashed quality and measurement reports
```

Every archive member must be listed in `model_files`, except the native bundle's
two canonical root sidecars. Links, traversal, devices, duplicate names, unknown
members and overwrites are rejected. Materialize model-cache symlinks before
packaging. The archive files are mounted during build, **not copied into a
layer**, avoiding duplicate compressed and extracted copies. Retain the original
S3 checksum-addressed native bundle separately for rollback.

`release.json` is JSON with this exact contract (no deployable example values):

| Field | Required value/meaning |
|---|---|
| `schema` / `status` | `musetalk_docker_release_v1` / `validated` |
| `source_revision` | Actual clean 40-character Git commit used for the image |
| `cuda_base` / `platform` / `matrix` | Tested Ubuntu 22.04 CUDA 12.1 devel image at `@sha256:<actual digest>` / `linux/amd64` / `cu121` |
| `cuda_runtime_base` | Optional exact matching CUDA/cuDNN runtime digest supported by the helper; omission retains `cuda_base`. The final stage must use this manifest-selected base |
| `bundle_name` | `rtx3090-r5-srcg50-int8`; matching real descriptor must exist and appear in r5 recipe |
| `bundle_manifest_sha256` | SHA-256 of `.musetalk_trt_artifact_manifest.json` inside native bundle |
| `redistribution_reviewed` | Public/validated images require `true`. An explicitly private candidate may retain `false` with hashed packaging findings; no blanket use-rights clearance is inferred. |
| `image_visibility` | `private` for the selected candidate; public release transport rejects this metadata. Missing means legacy public policy. |
| `packaging_review_file` | Required private-candidate evidence, bound to exact source/native hashes, preserving notices and remaining findings. |
| `avatar_prep` / `kokoro` / `vp8_encoder` | `true` / explicit boolean matching production TTS / actual tested `native` or `pyav` |
| `apt_packages` | Array of exact `name=version` pins resolved on selected base; helper lists required package names |
| `source_files` | Map of every exported code-relative path → `{sha256, size_bytes}`; obtained after commit from context inventory |
| `model_files` | Map of `models/...` path → `{sha256, size_bytes, license_id, public_redistribution:true}` |
| `external_model_files` | Optional private-runtime map for only the five paths below; never included in public archives/layers |
| `archives` | Exactly `weights.tar.gz` and `native.tar.gz` → `{sha256, size_bytes}` |
| `notices` | Map of `licenses/...` path in release input → `{sha256, size_bytes}` |
| `evidence` | Map of evidence-relative path in release input → `{sha256, size_bytes}` |
| `quality_decision_file` | Key in `evidence`; see quality contract below |
| `aggregate_acceptance_file` | Key in `evidence`; see aggregate contract below |

The quality decision must contain `decision:"accepted"`, exact
`bundle_sha256`, `quality_parity_with_reference:"PASS"`, an unchanged
`strict_original_gates` verdict (may preserve a justified inherited failure),
and `visual_inspection_file` naming a separately hashed evidence record.
The build validator checks these links, not the pictures itself: the actual
numerical analysis/visual inspection must already have happened.

The aggregate acceptance adapter must contain `gpu:"NVIDIA GeForce RTX 3090"`,
the same `bundle_sha256`, `T` (at least two windows), and `SUST` (at least five
consecutive windows). Each window has `elapsed_seconds` (finite, >=60),
`completed_valid_frames` (positive integer), `status:"PASS"`, and `raw_report`
naming a hashed source report in `evidence`. Every ratio must be >=400 without
rounding. These are links to real harness reports, never substituted synthetic
results. Live capacity and final image performance are additional release gates;
they are not inferred from these offline windows.

`apt_packages` pins fail closed if unavailable. Freeze the actual base-package
inventory and repository/snapshot state alongside build evidence. A base digest
and application constraints improve repeatability but do not promise bit-for-bit
image reproduction from mutable external package repositories. This initial
candidate retains all required apt packages and the complete `/opt/musetalk`
payload. Its fresh final stage avoids shipping build/install layers. The optional
runtime base is restricted to the exact CUDA12.1/cuDNN8 pair exercised by the
dependency-only experiment; this is not full-image/GPU acceptance. No TensorRT
resource pruning is applied to the full image. Record actual full-image layer
sizes, compressed transfer and GPU/preparation tests before claiming savings or
promoting the resulting digest.

## Optional private runtime models

The engineering model-license review is **not** blanket publication or legal
approval. Private retrieval does not cure unresolved model usage rights. A
manifest using private delivery must set `redistribution_scope` to
`baked_model_files_only` and retain `private_model_usage_rights` as `unresolved`
or `reviewed`; `redistribution_reviewed:true` then covers only public image assets.
Do not assert `reviewed` without the actual decision. Public baked models still
need their own exact-byte provenance and complete notices.

`external_model_files` accepts only:

- `models/syncnet/latentsync_syncnet.pt`
- `models/face-parse-bisent/79999_iter.pth`
- `models/face-parse-bisent/resnet18-5c106cde.pth`
- `models/auxiliary/s3fd-619a316812.pth`
- `models/face_detection/s3fd.pth`

SyncNet remains allowlisted for optional training/historical payloads, but is not
required by the canonical API or `/avatars/prepare` model contract. Normal API
images should omit its unused checkpoint; this does not change any usage-rights
or redistribution gate. DWPose and S3FD remain required for full avatar preparation.

Each entry requires actual `sha256`, `size_bytes`, `public_redistribution:false`,
and `private_delivery_authorized:true`. Its `source` requires `type:"s3"`, exact
`bucket`, AWS `region`, 12-digit `expected_owner`, and a `key` ending in
`/sha256/<actual SHA-256>/<exact model filename>`; optional `version_id` further
pins a versioned object. There is no arbitrary URL, endpoint override, anonymous
download, or path outside this allowlist. Bucket/key/owner are references, never
credentials. The operator must verify private object access and delivery scope
before setting authorization; no private object has been uploaded by this code.

For immutable images only, canonical secret bootstrap now runs once before the
install check. The image reapplies its policy, restores these private models,
then runs the **full** canonical model/dependency check. Source-install ordering
is unchanged. Retrieval uses only explicitly injected `AWS_ACCESS_KEY_ID`,
`AWS_SECRET_ACCESS_KEY` and optional `AWS_SESSION_TOKEN`; it does not fall back to
an unrelated instance profile. S3 calls are signed, expected-owner checked and
ignore configured endpoint overrides. Only the existing canonical helper fetches
the runtime secret; this is not a second secret-credential implementation.

Files are SHA/size verified in the private cache under
`$MUSETALK_STATE_DIR/private-model-cache` (default state directory above), then
atomically materialized at their required `/opt/musetalk/app/models/...` paths.
Every existing target/cache hit is rehashed on every startup. Corrupt or symlinked
inputs fail closed rather than overwriting unknown files. The entire
`Immutable private model restore` phase and its downloaded bytes/cache hits are
logged in the cold-start trace. Those bytes and time cannot be excluded from a
fresh-instance readiness claim. No runtime pip/apt/engine build is permitted.

Build/isolated CPU checks reject private payloads in the image, verify all baked
assets, and use the canonical installer's `--skip-weights` only to defer the
declared private files. The validator independently requires **every** canonical
full-API model to be declared publicly or privately, so other missing models do
not silently pass. Full target validation does not skip weights. Never publish a
`docker commit`/snapshot of a running worker: its writable layer may now contain
these private files, runtime secrets, uploads and generated media.

## Build on a real linux/amd64-capable builder

First run CPU-only contract tests (no Docker, GPU, network or heavy packages):

```bash
python3 -m unittest discover -s docker/musetalk/tests -v
bash -n docker/musetalk/entrypoint.sh scripts/vast_onstart.sh
python3 docker/musetalk/context.py --root "$PWD"
```

The inventory prints tracked allowed source hashes and dirty state. It does not
add files or mark a release validated. Commit reviewed changes before the final
inventory; untracked Docker files cannot be exported. Export verifies the exact
manifest and refuses dirty source/output overwrite:

```bash
python3 docker/musetalk/context.py --root "$PWD" \
  --manifest "$RELEASE_DIR/release.json" --output "$SOURCE_CONTEXT"
docker buildx build --platform linux/amd64 \
  --build-context release="$RELEASE_DIR" \
  --build-arg CUDA_BASE="$PINNED_CUDA_BASE" \
  --build-arg CUDA_RUNTIME_BASE="$MANIFEST_SELECTED_CUDA_RUNTIME_BASE" \
  --build-arg BOOTSTRAP_PYTHON_VERSION="$PINNED_PYTHON3_APT_VERSION" \
  --build-arg SOURCE_REVISION="$RELEASE_COMMIT" \
  --file "$SOURCE_CONTEXT/docker/musetalk/Dockerfile" \
  --tag "$REGISTRY_NAMESPACE/musetalk-r5:$IMMUTABLE_TAG" \
  --load "$SOURCE_CONTEXT"
```

All uppercase values above are actual resolved inputs, not defaults. The CUDA
build and final bases must contain `apt-get`; the pinned bootstrap `python3` package must also
appear identically in `apt_packages`. The canonical installer uses pinned
requirements and verifies CPU imports with CUDA hidden; driver-dependent
TensorRT imports remain deferred to real GPU verification. Avatar-prep may
need the compatible nvcc toolchain when no compatible wheel is available.
GPU-less source compilation sets `FORCE_CUDA=1`, `MMCV_WITH_OPS=1` and explicit
`TORCH_CUDA_ARCH_LIST=8.6`; the CPU image check rejects mmcv whose compiled CUDA
version is not 12.1. Import success alone is not enough. Run a real GPU avatar
preparation in clean-container validation before claiming `/avatars/prepare`
support—the static check does not prove its kernels execute correctly.

Do not send the live worker's home directory or workspace as build context.
The context exporter permits only tracked source and rejects recognized private
keys/access-token patterns without displaying their values. This is defense in
depth, not a complete DLP/security audit. Inspect full source, final filesystem,
history and every exported layer before publication. Build inputs are locally
pre-staged via the authorized credential path; if retrieval moves into the
Dockerfile, use a BuildKit secret mount, never credential `ARG`, `ENV`, or `COPY`.
See [Docker named contexts](https://docs.docker.com/build/concepts/context/#named-contexts)
and [build secrets](https://docs.docker.com/build/building/secrets/).

## Runtime/Vast configuration

Inject only approved per-instance secret references/credentials, worker ID and
mapped endpoint, control-plane URL/token, S3 avatar configuration, TURN settings,
and measured capacity. Do not bake them into source/release JSON. Existing
`MUSETALK_AWS_SECRET_ID` bootstrap is retained. Registration is required; token
or control-plane configuration absence fails canonical startup. Instance identity
comes from Vast's runtime fields; the image supplies only the code revision.
The image ignores persistent control-plane env files and regenerates its TURN env
file from current runtime inputs, so stale files cannot override release policy.
Supply control-plane credentials through runtime environment/secret bootstrap,
not a mounted `control-plane.env` file. Validated images force registration on;
standalone candidate images explicitly force it off.

GPU run prerequisites: exact RTX 3090 (not Ti), compatible host driver for the
validated cu121/TRT 10.3 stack, suitable CPU/RAM/disk, and measured shared memory
(initial planning requirement 8 GiB). Expose API TCP 8000 and the actual configured
TURN/relay ports; `EXPOSE 8000` alone does not establish external WebRTC routing.

Vast SSH/Jupyter mode replaces image ENTRYPOINT, so its on-start command must
explicitly run:

```bash
bash /opt/musetalk/app/docker/musetalk/entrypoint.sh onstart
```

ENTRYPOINT mode uses the image supervisor/Tini but does not provide an SSH/Jupyter
service itself. Select/test the launch mode before changing the existing template.
Vast pulls the configured image; **never add a nested `docker pull` inside the
rented container**. See [Vast launch settings](https://docs.vast.ai/guides/templates/template-settings).

## Publication and outstanding gates

`validate_image.sh IMAGE@sha256:DIGEST NEW_OUTPUT_DIR` pulls an immutable digest,
saves inspection/history, and runs the isolated GPU-less check with networking
disabled. This does not publish and does not certify all-layer secret scanning.
The manifest inside the image is `/opt/musetalk/release.json`; retain it with the
matching Git revision, native bundle SHA, image digest and template version.

The current delivery target is private GHCR, not a public/searchable registry.
Follow the fast-startup plan for private package permissions, separate publisher/
pull credentials and digest-pinned deployment; no ECR resources or IAM grants are
required. The existing build-only CI workflow does not yet publish to GHCR.

`.github/workflows/musetalk-ghcr.yml` adds private registry bootstrap, a native
amd64 dependency build, all-layer credential/path scanning, exact-digest publication,
and a second clean runner's authenticated pull/CPU import check. It uses job-scoped
`GITHUB_TOKEN` permissions, never copied interactive credentials or an AWS builder.
Its reviewed task-branch request is `.github/musetalk-ghcr-request.json` with only
`{"mode":"bootstrap"}` or `{"mode":"dependencies"}`. A `dependency-<commit>` image
contains no model/native release or Kokoro and exits instead of serving. Publication
of that foundation is not full-serving-image, GPU or startup acceptance. The legacy
dependency-only workflow now requires an explicit request to avoid duplicate builds.

Before promoting the image, record all of:

- Successful real build, CPU import checks and complete layer/license/secret audit.
- Clean RTX 3090 run, native backend/probes and production avatar restore.
- T/SUST still >=400 FPS inside the image; quality unchanged; measured live capacity.
- Missing credentials, corrupt/missing bundle, wrong GPU, restart and drain tests.
- Pushed digest pulled elsewhere with the intended pull-only identity, private
  package visibility/readback, and denied anonymous/publish/delete access for pullers.
- Existing exact Vast template edited in the browser with rollback/readback.
- Different fresh instance through EC2, with no manual repair, usable external call,
  complete request-to-frame/steady-readiness trace, repeated fresh/restart trials.

Registry description must state only measured results and limitations. Public
searchability is not a requirement for private delivery. Keep the previous
template/image/code tuple,
portable bundle, and control-plane settings for rollback. This directory contains
none of that external completion evidence yet.

## Nonpromotable contingency image

If the real native candidate misses 400 FPS or quality is incomplete/rejected,
packaging diagnostics can continue without weakening the release gate. Set
`status:"candidate"`, `promotion_eligible:false`, and a concrete
`candidate_reason` in the manifest, and explicitly pass
`--build-arg RELEASE_CHANNEL=candidate`. The default validated build rejects this
manifest. The image label `io.musetalk.release-channel` records the channel.

Candidate quality evidence still identifies the exact real bundle and preserves
`accepted`/`rejected`/`incomplete` plus strict-original verdicts. Aggregate evidence
must preserve `status` (`PASS`, `FAIL`, `INVALID`, or `NOT_RUN`) and a nonempty
`limitation`; missing measurements must not be represented as measured zeros or
passing windows. File/hash/native-hardware/license checks remain mandatory.

Candidate startup requires explicit `MUSETALK_CANDIDATE_STANDALONE=1`. Its policy
disables required callback and clears all registration credentials/URLs **after**
secret bootstrap; it also prevents ctl from sourcing a control-plane env file.
It is only for isolated diagnostics, cannot register into production autoscaling,
and is not a fresh EC2-routed-call acceptance result. Use a clearly named
`candidate-...` tag and never update the production Vast template to it. Promotion
requires a new validated manifest with full original release evidence and a new
immutable digest, not toggling a runtime flag.
