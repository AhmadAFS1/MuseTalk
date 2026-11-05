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
| Search filters | `num_gpus=1 gpu_name=RTX_3090 gpu_ram>=24 cpu_ram>=128 disk_space>=200 driver_version>=580.65.6 cuda_vers>=13.0 inet_down>=500 reliability>0.98 compute_cap=860 rented=False` |

Why these filters: the 2026-11-05 Reina qualification uses exact RTX 3090 compute8.6 with24 GB VRAM and an advertised 128 GB host. Provider RAM fields use the query parser's GB×1000 convention. After rental, onstart independently requires exact `NVIDIA GeForce RTX 3090`, total VRAM>=24,500 MiB and min(host RAM,cgroup limit)>=128,000 MiB. It preserves the old 8.9–9.x Ada alternative at 12 GB / 29 GB, rejects other Ampere by default, and keeps driver/disk/installer/engine checks. Do not infer lower-RAM qualification from reclaimable page-cache telemetry. H3 and MuseTalk asset preparation run sequentially, one active job.

The actual creation profile was H3 v2 plus eager teacher MuseTalk/compiled TAESD asset QC; no AMPERE_PLUS TensorRT execution was qualified. Phone playback uses the shared approved Core ML student, source 24 fps / render 25 fps, independently of the creation GPU. Set `MUSETALK_REF` explicitly to the final immutable reviewed commit; `onstart_cmd.sh` accepts raw SHA or named references. The existing private factory c4de... template's bootstrap was independently verified SHA-compatible, so its code pin is reachable.

Secrets: `CHARACTER_WORKER_TOKEN` and `H3_WORKFLOWS_GIT_TOKEN` may live in the AWS runtime secret
(`lingua/musetalk-worker-runtime`) instead of the template; the installer exports every key of that secret. The
secret must also provide `AVATAR_S3_BUCKET` and the S3 runtime key (docs/musetalk_worker_secrets.md).
