# Docker fast-startup plan for Vast.ai autoscaling

Updated October 9, 2026 (America/Chicago).

## Objective and current priority

Reduce the time from an EC2 autoscaler scale-up request to a usable MuseTalk
worker on Vast.ai. Here, **fast means boot-to-usable-capacity**, not rendering FPS.
The user reports roughly 15-minute total startup and 9–10-minute readiness/warmup;
these are reported baselines, not a measured Docker comparison.

Prioritize building, privately publishing, and testing the serving image before
further throughput or precision tuning. Preserve the reviewed rendering behavior.
The user selected **private GitHub Container Registry (GHCR)** on October 9,
replacing the earlier AWS Elastic Container Registry (ECR) proposal. The approved
temporary Linux builder remains an option, with narrowly scoped credentials and
bounded lifetime. No ECR repository or ECR IAM permissions are needed. This document
does not authorize unrelated infrastructure changes or claim that an image has
already been published.

This is the current Docker/startup delivery plan. The
[original performance-and-autoscaling execution plan](RTX_3090_R5_DOCKER_AUTOSCALING_EXECUTION_PLAN.md)
remains historical context; its original public/searchable-image objective is not
the current private-registry delivery choice.

## Established facts and remaining work

| Item | Status and evidence |
|---|---|
| Docker build/runtime implementation | Exists in [docker/musetalk](../docker/musetalk/README.md), including a Dockerfile, manifest validation, immutable startup, and supervised shutdown. Full serving image remains unbuilt/unpublished. |
| Dependency-only build | Built and CPU-checked in CI: 12,334,990,710 uncompressed bytes, 37.257% smaller than the original dependency build. No models/native release/Kokoro, published digest, compressed pull size, or GPU/boot acceptance. [Evidence](fps_comparisons/rtx3090_r5_20261008/release/dependency_size_delta_ead7e01.json). |
| Latest 3090 latent A/B | Base/reference: 315.352860 FPS; fresh 3090 latents: 308.571727 FPS. Six concurrent avatars, full tracking and 100% chin composition, five sustained windows per arm, no encoding/network overhead. Fresh latents were 2.15% slower in sequential tests; causation is not established. [Results](fps_comparisons/rtx3090_r5_20261008/native/a6_latent_ab_20261009/a6-latents-ab-summary.json). |
| Video evidence | Six labeled base-versus-native comparisons plus original clips, with hashes and capture reports. [Receipt](fps_comparisons/rtx3090_r5_20261008/native/a6_latent_ab_20261009/videos/video-evidence.json). |
| Small model-load experiment | Default-off eager-UNet skipping saved approximately 5.924 seconds in same-host fresh processes with warm filesystem cache and equal tested output hashes. Not Docker cold boot, full avatar preparation, or live-call acceptance. Keep disabled pending broader validation. [Evidence](fps_comparisons/rtx3090_r5_20261008/startup/a5_model_startup_pair_summary_0824.json). |
| Registry access | GitHub access as `AhmadAFS1` and repository admin/push permissions verified via the existing authenticated CLI without copying its token. Private GHCR bootstrap/publication workflow implemented; actual publication, independent pull and production pull credential still need verification. |
| Remaining AWS access | Local `lingua-backend-user` was denied EC2 inspection. If an AWS builder is used, scoped provisioning/input access is still needed. Earlier ECR denials no longer block registry setup; do not add ECR permissions to work around them. Existing S3/runtime-secret dependencies remain. |
| Production EC2 builder suitability | Existing control plane had approximately 2.9 GB free on a 20 GB root disk and no Docker runtime. Do not build the image there. Recheck facts when execution resumes. |
| Fresh image startup | Not measured. No production Docker launch-template change has been made. |

The user accepted the latest lower FPS for now. That is not a 400+ FPS pass or
blanket numerical quality acceptance. Existing release gates must retain their
actual verdicts; see the candidate/production distinction below.

## 1. Build once, rather than install on every instance

Use a separate, temporary native Linux `amd64` builder. Build a reproducible image
from a reviewed Git commit and allowlisted inputs, not `docker commit` of the live
machine or a copy of its entire workspace.

Prefer the existing GitHub-hosted Linux CI route if measured memory/disk capacity
and secure private-input delivery suffice; otherwise use the approved temporary
EC2 builder. The current `.github/workflows/musetalk-build.yml` has read-only
repository permissions, downloads public release inputs, and does not publish.
It needs an explicitly reviewed private-input/full-release publishing extension, not merely a
registry-name substitution. Never upload private inputs as public GitHub release
assets to make that workflow work. Check actual Actions billing/runner capacity
before choosing CI; free GHCR does not imply all build compute is free.

Implementation checkpoint: `.github/workflows/musetalk-ghcr.yml` and
`docker/musetalk/ghcr.py` now implement an empty private-registry bootstrap followed
by optional dependency-only publication, all-layer credential/path checks and an
independent clean-runner pull. A reviewed non-secret request selects `bootstrap`
or `dependencies`; ordinary helper/docs edits no longer launch duplicate legacy
dependency builds. Full serving publication/private build-input transport are
separate remaining steps; the foundation image must not be launched as a worker.

Package:

- The exact MuseTalk code revision and RTX 3090 runtime configuration.
- The pinned CUDA-compatible Python environment, TensorRT, avatar-preparation
  libraries, video encoders, TURN server, and required system libraries.
- Required serving and avatar-preparation model weights where packaging is
  permitted, with exact hashes and notices.
- The selected prebuilt native RTX 3090 UNet and TAESD decoder plans, including
  manifests, fingerprints, exact probes, and provenance.
- The existing startup chain and process supervisor.

Keep credentials, private avatar media, voice recordings, generated outputs, and
copied readiness/self-test stamps out of all image layers. The host NVIDIA driver
is supplied by Vast's machine, not installed inside the image; verify compatibility.

Code and dependencies live under `/opt/musetalk`, so Vast's `/workspace` mount
cannot hide them. Logs/uploads/results and authorized private model/avatar caches
live in mutable runtime storage. Explicitly match local TTS capability to actual
production routing; do not quietly disable an endpoint to make the build pass.

### Work moved out of startup

| Work | Image-build/prepublication time | Every instance's startup |
|---|---|---|
| System/Python installation and extension compilation | Complete and check it once | Verify; no apt/pip or compilation |
| TensorRT engine construction | Build and validate native plans before publication | Verify and load; no engine rebuilding |
| Packaged model weights | Obtain and verify them once | Load into CPU/GPU memory |
| Excluded private models | Persist exact authorized private objects | Fetch missing pinned objects and verify every hit |
| Secrets, identity, endpoints | Never bake them | Inject/fetch current per-instance values |
| GPU context, graphs, warmup | Cannot snapshot live GPU state into the image | Initialize and validate on the actual host |
| Avatar readiness | Persist compatible prepared caches outside the image | Restore and warm the avatars needed for declared capacity |

The successful immutable path must not clone/pull application source, invoke
apt/pip, compile extensions, rebuild engines, or perform untracked model downloads.
Missing/corrupt/incompatible inputs fail clearly instead of triggering a long
automatic installation repair. Preserve the canonical source-install path for rollback.

## 2. Publish privately and make downloads efficient

Use private GHCR. Based on the existing Git remote, the proposed image name is
`ghcr.io/ahmadafs1/musetalk-rtx3090`; confirm namespace ownership, existing package
state, and publish access before use. No package has been created by this plan.
Use versioned candidate tags such as `candidate-rtx3090-<date>-<git-sha>`, but deploy
and retain rollback references as `ghcr.io/ahmadafs1/musetalk-rtx3090@sha256:<digest>`.
A tag is mutable and must not be treated as the deployment identity.

GHCR's first publication defaults to private. Inspect any existing package first;
for a new package, establish it with a non-sensitive placeholder and verify private
visibility before pushing the model-bearing image. The source repository is public;
keep package access explicitly controlled, remove inherited permissions if needed,
and grant only the intended publishing workflow and pull identity access. Verify
anonymous pulls fail. Do not make the image public for authentication convenience.
[GHCR visibility](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry),
[package access controls](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility).

For CI publication, use the job's `GITHUB_TOKEN` with `contents: read` and
`packages: write`, only in a trusted, reviewed publishing job. Keep ordinary checks
read-only and pin third-party actions by commit. If publishing from the temporary
EC2 builder instead, use a separate expiring classic PAT with `write:packages` and
an authorized publishing identity; avoid unnecessary `repo`/`delete:packages`
scopes. Never copy the existing interactive GitHub CLI token to a builder or Vast
worker. Supply login secrets through stdin or an approved secret mechanism, not
build arguments, image layers, command-line arguments, or logs.
[GitHub CI publication](https://docs.github.com/en/actions/tutorials/publish-packages/publish-docker-images),
[GHCR authentication](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry).

1. Assemble the exact source, model, native-engine, license, and evidence manifests.
   Existing helpers require a real serving-only native bundle and matching descriptor;
   raw diagnostic archives must not be mislabeled as accepted release bundles.
2. Build the full image and run its isolated CPU/dependency checks.
3. Audit the final filesystem, image history, and exported layers for credentials,
   unnecessary payloads, private media, and license/provenance problems.
4. Push a clearly labeled candidate tag to the verified private GHCR package; record
   the actual immutable manifest digest and read back visibility/access settings.
5. Independently pull that digest with the intended pull-only identity and recheck
   its embedded release manifest. Verify no write/delete access through scopes and
   grants; any negative push test must target a disposable, non-sensitive tag.
   Never attempt deletion of a real serving or rollback version to test permissions.

Measure compressed registry transfer bytes, extracted size, peak builder disk use,
and download/extraction time. The dependency-only image size is not the full image's
size. Remove development caches, build debris, benchmark videos, and duplicate
archives from the shipped payload, without untested runtime-library pruning.
Where practical, separate stable dependencies/models from frequently changing code
layers. Verify the resulting layer layout rather than assuming it is cache-efficient.

Same-image host caching may improve subsequent launches, but a newly selected Vast
host may have no cached layers. An uncached launch is a required test, not an edge
case to omit.

Retain the user's Vast upload/download price requirement of no more than $1.50/TB;
verify provider units and the effective offer filters. GitHub currently charges
**$0 for GHCR container-image storage and bandwidth**, including private images;
its general GitHub Packages quotas are not the GHCR container billing rule. GitHub
says it will give at least one month's notice before changing this policy. Recheck
at rollout; this is a current policy, not a permanent price guarantee.
[GitHub Packages billing — container registry exception](https://docs.github.com/en/billing/concepts/product-billing/github-packages).

Vast must download directly from GHCR, not through EC2, S3, or an ECR mirror.
This avoids introducing AWS image-storage/image-egress charges. GHCR being free
does not remove Vast-side network charges, existing private-model S3 egress,
builder compute/egress, Actions artifact costs, or the time to transfer/extract the
image. Record those separately and still measure compressed bytes and cold pulls.

### Minimal ongoing AWS cost guardrails

The user confirmed execution on October 9 and requested minimal ongoing AWS cost.
At the latest access check, only the local `default` profile was configured,
identifying `lingua-backend-user` in account `211125449207`. No registry, builder,
IAM policy, or production launch configuration has been changed as part of this
handoff. The GHCR decision removes the proposed recurring ECR charges and the
registry-related AWS access requirement, not existing application AWS dependencies.

- Use one private GHCR package. Do not create ECR resources, cross-region registry
  replication, a permanent registry server, EKS, an ALB, a NAT gateway, paid
  interface endpoints, or a new KMS key for this task. Preserve existing account-wide
  security settings rather than downgrading them for cost.
- Prefer a suitable existing ephemeral CI runner. If AWS compute is needed, run
  one CPU-only, native `amd64` builder, sized from actual memory/disk needs.
  Record its current compute, root-volume, public IPv4, and transfer rates before
  launch. Start with a four-hour maximum lifetime; any extension must be deliberate
  and costed. Establish an independent cleanup mechanism before starting paid work.
- Terminate the builder rather than leaving it stopped. Set its task-owned root
  volume to delete on termination; do not create snapshots or persistent additional
  volumes by default. Remove task-owned temporary networking resources after use.
  Reconcile instance, volume, and address state after cleanup; termination alone
  is not proof that all associated charges ended.
- Retain the deployed image, its usable rollback image, and the active candidate.
  Measure unique retained layer bytes, not simply tag count. GHCR cleanup is not an
  ECR lifecycle policy: inventory the exact package versions/digests and confirm
  they are unused before any deletion. Do not delete deployed/rollback versions or
  shared multi-architecture manifests through a blanket age rule. No automated
  package deletion is required for initial delivery.
- Keep build inputs in their existing authorized storage; do not make extra permanent
  copies of model/video archives just for Docker. Bound any new diagnostic log
  retention and avoid recurring paid polling/monitoring infrastructure.
- Report one-time build/validation cost separately from recurring registry storage
  and every scale-out's remaining transfer cost. Forecast current GHCR registry
  storage/transfer charges as $0, not ECR's earlier estimates. Do not assume AWS
  free allowances are unused, or count an undocumented host cache as guaranteed.

Before production rollout, record expected monthly cold pulls and actual per-pull
bytes, including private-model downloads. Confirm the remaining AWS/S3 and
Vast-side costs, and retain the $1.50/TB Vast requirement. Review GHCR pricing and
service limits if GitHub announces a policy change; changing registry or visibility
again requires an explicit decision, not a silent fallback.

## 3. Launch the image through the existing EC2 autoscaler

Keep the current division of responsibility:

- EC2 detects load, selects an eligible Vast offer, creates/reconciles instances,
  tracks readiness/capacity, and routes calls.
- Vast pulls the configured image and starts its container on a compatible 3090.
- The container loads the prepared runtime, verifies native engines, restores
  required caches, warms the GPU, and starts API/TURN services.

After candidate validation, prepare an experimental launch configuration that
references the exact image digest. Inspect the deployed `create_request` and all
template overrides: an old `image`, `onstart`, or environment field can defeat an
otherwise correct template update. Never add a nested `docker pull` or Docker
installation inside the rented container.

Test direct ENTRYPOINT mode (`runtype: args`) for the production path. Vast's
SSH/Jupyter modes replace the image entrypoint; an SSH diagnostic launch must
explicitly invoke `bash /opt/musetalk/app/docker/musetalk/entrypoint.sh onstart`.
Keep SSH access for diagnostics where needed, and verify API/TURN/relay port mappings
in either mode. [Vast API launch documentation](https://docs.vast.ai/api-reference/creating-instances-with-api).

Use a separate classic PAT with only `read:packages` and a GitHub identity granted
Read access to the private image. A classic PAT is not itself package-scoped: limit
that identity's accessible packages where practical. Never send a publisher token
or CI `GITHUB_TOKEN` to a Vast host.
[GitHub package permissions](https://docs.github.com/en/packages/learn-github-packages/about-permissions-for-github-packages).

Store the pull credential using the existing approved runtime-secret system. EC2
reads the current credential for each launch and supplies it only in Vast's private
registry-authentication field; verify the installed API/CLI contract and a fresh
private GHCR pull. Fail closed on authentication errors; do not switch to public.
Keep credentials out of templates, image layers, logs, error messages and persisted
request dumps. Use an explicit expiry and rotation procedure tested before expiry;
GHCR does not use ECR's 12-hour token exchange. Reconcile retries after rotation,
and test that missing/revoked credentials fail without repeated rental creation.
Do not grant ECR IAM access to the EC2 role for this route.

Preserve the existing runtime secret bootstrap, S3 access, unique worker identity,
control-plane callbacks, current endpoint mappings, and measured capacity settings.
Production changes require readback and a retained rollback configuration.

## 4. Diagnose and shorten the remaining warmup

Docker removes installation/build work, not provider delays, uncached image transfer,
model loading, live CUDA initialization, or required avatar warming. First measure
which phases actually explain the reported 9–10-minute readiness delay.

Investigate one change at a time:

- Repeated model/engine loading and duplicate initialization.
- Required startup GPU warmup versus unnecessary repeated work.
- Restore of compatible prepared avatar caches instead of full preparation at boot.
- Bounded warming of the initial working set, with other avatars warmed on demand
  only where routing/readiness semantics safely support it.
- Private-model download, extraction, and full-hash verification costs.
- Unnecessary orchestration waits after readiness is genuinely established.

Do not skip exact engine probes, change quality/precision, remove required avatar
preparation, or count unwarmed capacity merely to obtain a smaller startup number.
Any warmup change needs output, preparation, and live-behavior regression checks.
A warm spare or earlier predictive scale-out is a later, separately costed option
if the product needs near-immediate capacity; it is not evidence of faster cold boot.

## 5. Measure request-to-usable-output, not just process health

Record these milestones and their elapsed times:

1. EC2 begins the scale-up/create request.
2. Vast accepts it and assigns a reconciled instance ID.
3. Image download/extraction completes and the container starts.
4. Secrets/private inputs are ready; model/engine loading and GPU warmup complete.
5. API/TURN and the required avatar working set are ready.
6. An external test client receives the first usable generated video frame with audio.
7. For a production-eligible release, the worker becomes routable at its tested capacity.

Measure the headline request-to-first-usable-output interval on one observer's
monotonic clock. Retain worker/provider UTC events for phase attribution, recording
clock offset/uncertainty rather than blindly subtracting different hosts' clocks.
Report overlapping phases and unobservable provider detail honestly; do not double-count
or invent a pull-completion timestamp.

Compare the existing source-install path and image path with matching hardware,
configuration, avatar working set, and readiness criteria. Run repeated fresh-host
trials and separate same-container/restart trials. Record cache state, host CPU/disk/
network, image bytes/digest, model bytes fetched, timeouts, and failures. Report each
trial plus median/range; only claim tail-percentile performance with enough samples.
Set a numeric startup target from the measured baseline before optimization trials;
no agreed SLA or proven Docker startup saving exists yet.

HTTP `/health`, a successful container start, a registry heartbeat, and an offline
FPS benchmark are not substitutes for a usable external call.

## 6. Candidate validation and production rollout

First test a standalone candidate on a different fresh RTX 3090, created through
EC2. It must boot without copied developer environments, hidden warm caches, manual
SSH repairs, or dependency/engine rebuilding.

Verify:

- Native backend/host compatibility, exact probes, and no fallback.
- Model access and actual `/avatars/prepare` execution.
- Reviewed visual behavior and sustained aggregate throughput inside the container.
- External API/WebRTC audio/video, negotiated encoder, freshness, and live capacity.
- Correct behavior with missing secrets, corrupt inputs, wrong GPU, and startup failure.
- Restart, bounded shutdown/drain, and no orphaned processes or duplicate workers.

The current manifest supports nonpromotable `candidate` images with
`MUSETALK_CANDIDATE_STANDALONE=1`. They explicitly disable production registration.
The existing validated-release gate still requires sustained 400+ FPS and linked
quality evidence; the latest 315/309 FPS results do not pass it. Do not forge those
verdicts, silently lower the gate, or toggle a candidate into production with a
runtime flag. Document any separately approved lower-throughput release criteria
and implementation explicitly before production rollout.

After the applicable release checks pass, update only the intended Vast template/
EC2 launch configuration. Retain the previous configuration and image/code tuple,
read the saved settings back, and launch another fresh instance through the actual
autoscaling path. Test a controlled scale-out and drain/scale-in cycle with test
traffic before wider rollout. Starting workers must not count as usable capacity;
set routable capacity from live tests with headroom, not aggregate FPS alone.

On failure, leave the candidate unroutable, preserve evidence, and restore the known-good
launch configuration. Keep the existing 4070 and production EC2 services intact.

## GitHub/AWS access handoff and execution checklist

The registry prerequisites are GHCR publish access, verified private package
permissions, and a safely provisioned pull-only credential. Do not paste tokens
in chat. Existing GitHub CLI authentication does not establish package scope.
Verify the intended identities and use the publishing mechanism in section 2.

AWS access is now needed only where the chosen builder or existing private-input/
runtime-secret path requires it, not to create or authenticate the registry.
For an EC2 builder, prefer temporary/SSO credentials configured locally and provide
the profile name rather than secret keys. No full administrator access is required.

Access should be scoped to:

- GitHub publisher: publish only the intended GHCR package through a reviewed CI
  job or separate builder publishing identity; verify package visibility afterward.
- Pull identity: Read access only to the intended image, with `read:packages` and
  no write/delete scopes. Verify the grants rather than claiming PAT scope isolates it.
- AWS operator, if needed: provision/manage only the tagged temporary builder and
  its associated resources; no ECR setup or ECR publish/pull permissions.
- Builder: read only exact approved private build inputs; no production application
  secrets or copied interactive credentials. Keep build-input credentials distinct
  from GHCR login and the production pull secret.
- EC2 launch path: access only to its established runtime secrets and the dedicated
  GHCR pull-secret location if newly required. Present any secret/IAM change first.
- IAM setup, if needed: create/configure/pass only the named task roles. Present actual
  policies and existing-role changes before applying them; do not attach broad admin policies.

Before renting, record builder/validation-instance cost estimates, ownership, exact
expiry, and a reliable cleanup mechanism. Preserve verified artifacts before expiry.
Do not assume an old development rental remains available: the recorded A6 rental
`55103198` expires October 9, 2026 at 9:12:28 p.m. CDT. Reconcile its actual state and
[owned-target record](fps_comparisons/rtx3090_r5_20261008/provisioning/a6_owned_target.json)
before any reuse; the Docker plan must not depend on keeping it indefinitely.

- [x] Save this plan and link it from the Docker and original execution documentation.
- [x] Preserve the latest latent A/B results and video evidence separately from startup claims.
- [x] User selects private GHCR instead of the earlier ECR proposal; temporary builder approval remains.
- [x] Verify current GHCR free storage/bandwidth policy and document remaining costs.
- [x] Remove ECR resource/IAM/token requirements; preserve minimal AWS cost guardrails.
- [ ] Verify GHCR namespace, private visibility, publishing access and pull-only credentials.
- [ ] Select a capable ephemeral builder and establish secure private-input delivery.
- [ ] If using EC2, obtain/verify scoped AWS provisioning and input permissions.
- [ ] Confirm remaining costs, resources, lifetimes and cleanup; establish private package/builder.
- [ ] Assemble clean, exact source/model/native inputs and the candidately labeled manifest.
- [ ] Build, CPU-check, audit, privately publish, and independently pull the full serving image.
- [ ] Validate fresh-container GPU behavior, preparation, quality, throughput, and external calls.
- [ ] Measure repeated cold-host request-to-usable-output traces; optimize the measured bottleneck.
- [ ] Resolve applicable production release criteria without misrepresenting 400 FPS/quality gates.
- [ ] Switch the intended launch configuration, retain rollback, and test real autoscaler scale-out/in.
- [ ] Record the final commit/bundle/image/template/provisioner tuple and clean up temporary resources.

Resume with GHCR credentials/package visibility, builder selection and resource
reconciliation first. Then build and test the image; ECR access is no longer a
prerequisite. Do not restart unrelated throughput tuning or public publication work.
