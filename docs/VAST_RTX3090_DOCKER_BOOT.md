# RTX 3090 Docker startup script

The updated copy/paste script is
[`scripts/vast_3090_docker_boot.sh`](../scripts/vast_3090_docker_boot.sh).
It starts the baked application, not a source checkout. It does not clone Git,
install packages, build engines, or log in to a registry.

## Image selection happens before bash

Select the **full serving image by immutable GHCR digest** in the Vast template
or EC2 instance-create request. Supply the separately stored pull-only GHCR
credential through Vast's `image_login`, not through the worker environment or
this script. Vast pulls the image before running its entrypoint/on-start script.

The published dependency digest `7cd567bd77d421565c6813e5372561be31a8b3cab73f307430abd6b95d0067f6`
does **not** include models/native release metadata and cannot boot MuseTalk.
The full serving digest and fresh startup measurement are pending; do not use
the dependency image as the new production template.

For headless/args mode preserve the baked ENTRYPOINT and its default `serve`
argument. Vast SSH/Jupyter mode can override ENTRYPOINT: paste the new script
into On-start, or invoke `/bin/bash /opt/musetalk/app/scripts/vast_3090_docker_boot.sh onstart`
if that path is available in the selected full image. Do not run Docker inside
the container to pull another image.

## Runtime configuration

Default `MUSETALK_RUNTIME_CONFIG_SOURCE=injected`: EC2 reads
`lingua/musetalk-worker-runtime` using `linguaEc2role`, and passes only the
required worker runtime values in the authenticated instance-create request.
The worker needs its scoped S3 runtime identity for four external models. It
does not inherit EC2's IAM role; the script removes the worker secret ID to
avoid an unnecessary Secrets Manager fetch. Never pass EC2 role credentials.

An optional `secretsmanager` mode is available only when a separately authorized
worker secret-reader identity is injected outside the script. This is not
needed for the default path, and this change grants no such identity.

Keep AWS/GHCR secrets out of the script, repository, image and logs. Rotate the
AWS key pair exposed in the old pasted script; rotation is not verified here.
The script defaults to port 8000, strict S3 verification, a 1800-second startup
deadline, `AUTO_SETUP=0` and `SETUP_CLEAN=0`. The baked release manifest selects
and verifies the native 3090 r5 configuration; no floating main-branch checkout
or arbitrary PROFILE override is used. Kokoro is intentionally omitted.

The first private candidate test injects `MUSETALK_CANDIDATE_STANDALONE=1` at
creation. This prevents accidental production registration while the image is
being validated; it is not another privacy/owner permission requirement. The
original strict quality/FPS failures remain recorded, not rewritten as passes.

## Startup validation

Record create-request time, provider running/pull completion when observable,
entrypoint time, `/health`, completed avatar preparation and first usable video
output separately. Container entrypoint time excludes image download. Provider
host image-cache state may be unknown: label that uncertainty rather than claim
a proven cache-empty cold start. `/health` alone is not app/call acceptance.

Logs: `/workspace/bootstrap.log` and the canonical supervisor's startup logs.
No new startup timing has been measured yet. The existing 9–10 minute warmup
and approximately 15 minute startup are user-reported reference values, not a
paired benchmark for this image.
