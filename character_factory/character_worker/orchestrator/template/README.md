# Vast template for the character-creation worker

The orchestrator rents instances by **template hash**, so the template carries everything a fresh box needs; the
orchestrator only passes the offer id, disk size and a label. Create it once with `create_template.sh` (prints the hash)
or in the console under Templates -> My Templates with these fields:

| Field | Value |
| --- | --- |
| Image | `vastai/pytorch:cuda-12.1.1-auto` (any Ubuntu 22.04 image with Python 3.10 works; CUDA wheels are pinned by the installers) |
| Launch mode | SSH (direct) - Vast then runs the on-start script after injecting SSH |
| On-start script | contents of [`onstart_cmd.sh`](onstart_cmd.sh) |
| Docker options | `-p 8010:8010 -p 22:22 -e CHARACTER_WORKER_PORT=8010 -e CHARACTER_WORKER_TOKEN=... -e MUSETALK_AWS_SECRET_ID=... -e MUSETALK_AWS_SECRET_REGION=us-east-1 -e MUSETALK_SECRETS_STRICT=true -e MUSETALK_SECRETS_VERIFY_S3=1 -e AWS_DEFAULT_REGION=us-east-1 -e AWS_ACCESS_KEY_ID=<secret-reader> -e AWS_SECRET_ACCESS_KEY=<secret-reader> -e H3_WORKFLOWS_GIT_TOKEN=<read-only token>` |
| Disk | 200 GB |
| Search filters | `num_gpus=1 gpu_ram>=12 cpu_ram>=30 disk_space>=200 driver_version>=580.65.6 cuda_vers>=13.0 inet_down>=500 reliability>0.98 compute_cap>=890 compute_cap<1000 rented=False` |

Why these filters: the pinned H3 stack is torch 2.14+cu130 (host driver >= 580.65); the reference run peaked at
11.5 GB VRAM and 27.6 GiB host RAM; the H3 weights are 61 GB; the accepted recipe was produced on an Ada GPU (sm89) and
Blackwell has no MuseTalk avatar-prep build. Widen `compute_cap>=890` to `>=860` only after an RTX 3090 has been
tested end to end.

Secrets: `CHARACTER_WORKER_TOKEN` and `H3_WORKFLOWS_GIT_TOKEN` may live in the AWS runtime secret
(`lingua/musetalk-worker-runtime`) instead of the template; the installer exports every key of that secret. The
secret must also provide `AVATAR_S3_BUCKET` and the S3 runtime key (docs/musetalk_worker_secrets.md).
