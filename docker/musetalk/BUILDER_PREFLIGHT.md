# Builder/access preflight — October 8, 2026

Initial inspection was read-only; no runtime was installed, VM created,
repository permission changed, secret added, or registry publication performed.
Root subsequently ran the narrowly scoped public-runner inventory described below.

## Existing operator Mac

- macOS 27.0.1, arm64, 18 logical CPUs, 24 GiB RAM, 698 GiB free workspace disk.
- Homebrew and GitHub CLI exist. Local `gh` is not authenticated.
- No Docker/Podman/Colima/Lima/QEMU/buildctl executable or Docker/OrbStack/Podman app found.
- Homebrew has none of the Docker/Colima/Lima/QEMU formulas installed.
- Rosetta installation receipt exists. Linux-amd64 PyTorch/TRT import behavior
  under emulation has **not** been tested.

A task-specific Colima VM with Docker/Buildx, native arm64 virtualization and
Rosetta amd64 emulation is a concrete local option after installation approval.
Start around 8 CPUs/16 GiB RAM/150 GiB disk, preserving headroom for the host;
confirm the installed CLI flags and resources before creating it. Always build
`linux/amd64`; the Mac cannot validate NVIDIA GPU runtime or engine performance.
An emulated build may be substantially slower than native x64, especially mmcv
compilation. This option has useful disk headroom and keeps private input delivery
local; it does not solve registry authentication.

Current [Colima configuration](https://github.com/abiosoft/colima/blob/main/embedded/defaults/colima.yaml)
supports `vmType: vz`, `rosetta: true`, and configurable CPU/memory/disk. The
available Homebrew metadata reported Colima 0.10.3, Docker 29.8.2, Buildx 0.37.2
and Lima 2.2.1 at inspection time; these were not installed.

## Existing GitHub account/repository

Read through the existing authenticated `/usr/bin/gh` on `3-way-head-talk`;
credentials were neither extracted nor copied. Repository `AhmadAFS1/MuseTalk`:

- Public, unarchived; default branch `main`.
- Current account has admin/maintain/push/pull permissions.
- Actions enabled; all actions allowed; no enforced SHA-pinning policy.
- Zero workflows, repository Actions secrets, or self-hosted runners.

An x64 `ubuntu-22.04` hosted workflow is available in principle, but there is no
existing build pipeline to run. Official [runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners)
list free standard public-repository jobs with 4 CPU/16 GB RAM/**14 GB SSD**.
That advertised disk allocation is a serious risk for this complete CUDA/PyTorch/
weights image, whose development plan budgets 60 GB of working disk. A tiny
read-only inventory workflow would establish actual free space before committing
to a heavy build; do not assume advertised limits or cleanup yield enough room.

Private S3 delivery to a public workflow is also unresolved: there is no existing
AWS Actions secret/OIDC binding. Do not paste long-lived AWS credentials or signed
download URLs into public YAML, logs, or workflow-dispatch inputs. No new CI secret
or cloud role is authorized by this investigation.

[GHCR](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry)
can publish a repository-associated image using a scoped workflow `GITHUB_TOKEN`,
without copying the worker's GitHub token. A first package is private by default,
and public visibility/search needs explicit verification. GHCR publication is not
Docker Hub `docker search` proof. No workflow permissions or packages were created.

## Measured native runner and recommended next decision

Root's [capacity-only run](https://github.com/AhmadAFS1/MuseTalk/actions/runs/37743477971)
succeeded at 2026-10-08 07:26:57 UTC, image `20260927.309.1`: native x86_64,
4 CPUs, 16,765,411,328 bytes RAM, filesystem 155,897,610,240 bytes total and
92,680,658,944 bytes free (about 86.3 GiB), Docker 28.0.4/Buildx 0.37.1/overlay2,
zero existing Docker images/cache. The 60 GiB planning gate passed. This is enough
to attempt the first build, not proof of actual peak usage or build success.
No Mac virtualization install is needed while this path is viable. Public input
redistribution review and authenticated/checksummed artifact delivery remain
gates. Production EC2 remains unsuitable and must not become the builder.

## Native-runner capacity probe

`.github/workflows/musetalk-builder-preflight.yml` is a five-minute job on standard
`ubuntu-22.04`, with read-only repository permission. Root enabled a push trigger
restricted to this task branch and this one workflow's path, then ran the probe.
It performs no checkout, cleanup, image pull, secret access or publication. It
records actual CPU/RAM/filesystem and Docker/Buildx facts in logs, job summary,
and a three-day small artifact; the official artifact action is pinned by commit.
The 60 GiB free-disk check is a planning requirement, not observed usage.
GitHub requires a `workflow_dispatch` definition on the default branch before a
manual run can target another ref; the narrow push trigger addressed this for the
preflight without altering the default branch.

The [official Ubuntu runner image inventory](https://github.com/actions/runner-images/blob/main/images/ubuntu/Ubuntu2204-Readme.md)
lists Docker client/server and Buildx preinstalled. GHCR publication can use a
subsequent dedicated workflow with `packages: write` and repository-associated
`GITHUB_TOKEN`, avoiding copied user tokens. Public-repository association does
**not** imply public package visibility; GitHub documents an explicit
[package-settings visibility change](https://docs.github.com/en/packages/learn-github-packages/configuring-a-packages-access-control-and-visibility).
Do not claim public pulls until an anonymous digest pull succeeds, or Docker Hub
searchability from a GHCR package page. Registry-specific search is separate.

## Nested Vast builder feasibility

[Rootless BuildKit](https://github.com/moby/buildkit/blob/master/docs/rootless.md)
still needs user namespaces, RootlessKit/OCI runtime, and permitted mount/proc
operations. Its documented container setup relaxes seccomp/AppArmor/proc masks;
being uid 0 inside a Vast container does not establish those host capabilities.
The `native` snapshotter can avoid overlay/FUSE requirements but does not remove
the namespace/exec requirements. Do not change host sysctls, request privileged
access or disable container isolation to force a build. The new development 3090
(`musetalk-3090-build-54798270`) reports kernel 6.8.0-146, 126 GiB available of
150 GiB, cgroup memory limit 144,869,163,008 bytes and CPU quota 18.43199 cores.
It has `unshare` but no Docker/BuildKit/RootlessKit/PRoot; seccomp filtering is
enabled. User namespaces are configured on the host, but allowed mount/userns
operations are still unverified. No namespace probe or install was attempted
while its canonical startup was active. GitHub's measured native runner is now
the preferred initial builder, so bypassing nested-container restrictions is
unnecessary.

## Build-only CI prepared for review

`.github/workflows/musetalk-build.yml` and `docker/musetalk/ci.py` implement the
initial assembly attempt, with no registry write permission or publication. It
runs CPU contracts and the Linux startup suite before building. No request file
is supplied: code edits alone do not trigger a build. Root may activate it by
adding `.github/musetalk-build-request.json` on this task branch, containing only
`source_revision`, `release_tag`, `metadata_sha256`, and `channel`. The full commit
must already exist; the new request commit points at it, not at itself. Manual
dispatch is also supported once the workflow exists on the default branch.

Inputs come only from an explicitly reviewed public Release in
`AhmadAFS1/MuseTalk`. `musetalk-docker-metadata.tar.gz` is independently SHA-pinned
by the request and contains exactly `release.json`, its listed `licenses/...`,
and its listed `evidence/...` files (32 MiB expansion limit). Every model must
already pass the public-redistribution contract; do not upload uncertain assets
just to unblock CI. No AWS secrets, signed URLs, or production worker identity
belong in a workflow/request or public Release.

[GitHub Release files must be under 2 GiB](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases).
For larger native/weights archives, the manifest's optional
`github_release_assets` map binds each canonical archive to an ordered list of
`{name,sha256,size_bytes}` pieces, named `native.tar.gz.part000`, `...part001`
(or corresponding weights names). Each part must be below 2 GiB; the assembled
archive must still match the canonical archive SHA and size. Small single assets
retain canonical names. The driver downloads only these names, validates before
assembly, and removes only verified task-local downloaded chunks after copying.

The build exports a clean allowlisted source context, rechecks 60 GiB free disk,
passes exact manifest-derived build args, uses `--load` (never `--push`), and runs
an isolated network-disabled CPU import check. Only small inventory/history/build
reports survive as three-day artifacts; the image is ephemeral. This is not a
complete layer/secret audit, GPU test, public image, or promotion decision. Later
GHCR delivery must explicitly audit the exact resulting image, and remains
separate from the Docker Hub searchability requirement.

[PRoot](https://github.com/proot-me/proot/blob/master/doc/proot/manual.rst) is a
ptrace-based rootfs execution tool, not a Dockerfile/BuildKit/OCI-image builder.
It may also be restricted by the container's syscall policy. Replacing Docker
RUN semantics with a hand-built PRoot export would add a new unvalidated build
pipeline and does not automatically support the Dockerfile's named contexts and
BuildKit mounts. It is not the preferred shortcut for this release.
