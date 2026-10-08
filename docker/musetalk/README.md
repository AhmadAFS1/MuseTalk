# MuseTalk r5 native RTX 3090 image — release groundwork

This is an **unbuilt candidate implementation**, not a published image or a claim
of 400 FPS/startup latency. No real native-3090 descriptor, bundle, registry
namespace, base digest, or acceptance manifest is invented here. Assembly fails
until those independently validated inputs exist. Do not point production or
the named Vast template at an unvalidated candidate.

## Runtime contract

The image keeps code at `/opt/musetalk/app`, Python 3.10/cu121 at
`/opt/musetalk/venv`, and native VP8 at `/opt/musetalk/native_vp8`. A Vast
`/workspace` mount cannot hide them. Mutable logs/uploads/avatar results live in
`/workspace/musetalk-runtime` (`MUSETALK_STATE_DIR` can choose another absolute
persistent location). Never mount over `/opt/musetalk`.

`entrypoint.sh` invokes the existing `vast_onstart.sh` → `vast_server_ctl.sh` →
`run_musetalk_server.sh` chain. Only the opt-in `MUSETALK_IMMUTABLE_RUNTIME=1`
changes source-install behavior: every nonzero install check is fatal, coturn
cannot be installed at runtime, and baked native files/host compatibility must
verify before launch. The normal source installer remains unchanged. The image
forces r5, strict backend verification, exact native probes (zero tolerance),
native sm86 plans, no engine provisioning, no TAESD builds, no TensorRT/encoder
fallback, and offline Hugging Face model access.
Release policy is reapplied after runtime secret bootstrap. It never clones,
pulls, invokes apt/pip, or constructs engines during successful normal boot.

All baked files are hashed on startup. The canonical bundle verifier then writes
a fresh artifact-integrity stamp; this is **not** a copied GPU-selftest or
readiness stamp. CUDA contexts, graphs, load probes, registration and avatar
warming still run on the target worker. Repeated full hashing is deliberately
not optimized until its measured cost is available.

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
| `bundle_name` | `rtx3090-r5-srcg50-int8`; matching real descriptor must exist and appear in r5 recipe |
| `bundle_manifest_sha256` | SHA-256 of `.musetalk_trt_artifact_manifest.json` inside native bundle |
| `redistribution_reviewed` | `true` only after all included assets/package licenses were reviewed for public distribution |
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
candidate retains development tools for full functionality; reduce layers only
after a successful measured build.

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
  --build-arg BOOTSTRAP_PYTHON_VERSION="$PINNED_PYTHON3_APT_VERSION" \
  --build-arg SOURCE_REVISION="$RELEASE_COMMIT" \
  --file "$SOURCE_CONTEXT/docker/musetalk/Dockerfile" \
  --tag "$REGISTRY_NAMESPACE/musetalk-r5:$IMMUTABLE_TAG" \
  --load "$SOURCE_CONTEXT"
```

All uppercase values above are actual resolved inputs, not defaults. The CUDA
base must contain `apt-get`; the pinned bootstrap `python3` package must also
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

Before promoting/publicizing the image, record all of:

- Successful real build, CPU import checks and complete layer/license/secret audit.
- Clean RTX 3090 run, native backend/probes and production avatar restore.
- T/SUST still >=400 FPS inside the image; quality unchanged; measured live capacity.
- Missing credentials, corrupt/missing bundle, wrong GPU, restart and drain tests.
- Pushed digest pulled elsewhere, registry page and actual public search result.
- Existing exact Vast template edited in the browser with rollback/readback.
- Different fresh instance through EC2, with no manual repair, usable external call,
  complete request-to-frame/steady-readiness trace, repeated fresh/restart trials.

Registry description must state only measured results and limitations. A pending
search index is pending, not passed. Keep the previous template/image/code tuple,
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
