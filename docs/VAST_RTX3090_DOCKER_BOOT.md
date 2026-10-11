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
The full private serving candidate is now published and independently pulled:

```text
ghcr.io/ahmadafs1/musetalk-rtx3090@sha256:388ebf73de4de996ed2fff9c1d88d669a288c32cf603ce2e41c69eef26d5e6fc
```

All-layer audit, private visibility/anonymous denial, clean-runner pull/offline
CPU check and the actual Secrets Manager pull-only deployment credential pass.
Compressed registry layers total 12,792,965,703 bytes (about 12.8 GB).
This is ready for an **isolated RTX 3090 startup/app test**, not a production
rollout: no fresh GPU boot or startup speedup has been measured for this image.
Use the standalone candidate flag described below. Do not use the dependency
image as a serving template.

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

## October 10 execution

The bootstrap is pushed at `c06624da9d7cebd6aa8f3dd6ad4a0dc8306ec6d6`.
The full-image build is
[run 38085895073](https://github.com/AhmadAFS1/MuseTalk/actions/runs/38085895073).
Its existing dependency predecessor took approximately 56 minutes to build,
audit and publish; this is context, not an ETA guarantee for the full image.

`scripts/repro_3090/run_docker_boot_sequence.py` is a bounded one-shot execution
driver for this exact build, not a recurring automation. It verifies publication,
the independent digest pull and offline CPU-check job before one EC2-created
RTX 3090 rental. The owned test has a verified four-hour expiry before creation,
the $0.30/hour ceiling and $1.50/decimal-TB upload/download caps. Its $12
conservative reservation includes compute, allocated disk, transfer and margin.
It never retries an ambiguous create or enables production rollout.

Generated progress is saved to
`docs/fps_comparisons/rtx3090_r5_20261008/startup/docker_boot_test_20261010/progress.md`.
The test targets `/health`, one existing avatar's S3 restore/cache warm, REST
generation and a locally decoded MP4 frame. This is not a live WebRTC/RTP test,
48-avatar warm readiness, matched cache-empty-host comparison or a 400 FPS pass.
Failures stop the sequence with safe diagnostics; any created worker keeps its
bound expiry. The worker may remain available until that four-hour deadline.

### Publication retry

The original run built/CPU-checked the image but failed at the layer export
before publication. Its boot sequence stopped before renting a GPU.
The corrected publisher scans `docker save` stdout without creating another
19 GB image archive and has a real tiny-image export smoke gate.
[Retry run 38093257223](https://github.com/AhmadAFS1/MuseTalk/actions/runs/38093257223)
uses request commit `874203e1157f74a6cf581d3c175155c4c54a0362`, with the
same `c06624da` serving source and input hashes. Publication/independent pull
must pass before using its immutable digest. The failed boot driver is not
automatically restarted by this publication retry.

### Verified publication — October 11 UTC

Both jobs in retry run 38093257223 passed. Verified artifact ZIP hashes and
the exact image/config identity are preserved in
[`publication_retry_2257/evidence/verified-full-image.json`](fps_comparisons/rtx3090_r5_20261008/release/publication_retry_2257/evidence/verified-full-image.json).
The deployment pull credential separately passed an authenticated, hash-checked
manifest fetch for the same full digest; no token was written to disk or logs.

CI timings: build/offline CPU check 54m28s; full streaming audit/private push
25m16s; separate pull/offline CPU check 7m02s. These are **CI execution times,
not Vast startup benchmarks**. The serving code remains pinned to `c06624da`,
with the same reviewed models/native configuration and original quality/FPS
limitations. No new GPU, timer, production template or autoscaler setting was
created/enabled. The next step is the bounded fresh-3090 startup/video test.
