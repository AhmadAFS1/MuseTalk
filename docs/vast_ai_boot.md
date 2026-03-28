# Vast.ai Boot Flow

This repo already has a current canonical setup path and launch path:

- setup: `setup_musetalk.sh`
- launch: `scripts/run_trt_stagewise_server.sh`

The Vast.ai helpers added here wrap those entrypoints so autoscaled instances
can bring themselves up without manual SSH steps.

## Files

- `scripts/vast_onstart.sh`
  - idempotent Vast instance boot wrapper
  - optionally runs setup if the venv / model files are missing
  - starts the MuseTalk server and waits for `/health`
- `scripts/vast_server_ctl.sh`
  - `start|stop|restart|status|logs` helper for the TRT-stagewise server
  - writes logs and a PID file under `/workspace/logs/musetalk` by default

## Recommended Vast.ai Layout

Use a persistent workspace and keep the repo there.

Example repo path:

```bash
/workspace/MuseTalk
```

Default Vast helper paths:

- repo: `/workspace/MuseTalk`
- venv: `/workspace/.venvs/musetalk_trt_stagewise`
- logs: `/workspace/logs/musetalk`

This avoids hard-coding the Colab-style `/content/...` paths on Vast.

## Vast.ai On-Start Script

Paste this into the Vast.ai on-start command field:

```bash
cd /workspace/MuseTalk
PROFILE=baseline \
PORT=8000 \
bash scripts/vast_onstart.sh
```

For the widened throughput profile:

```bash
cd /workspace/MuseTalk
PROFILE=throughput_record \
PORT=8000 \
bash scripts/vast_onstart.sh
```

## Behavior

On boot, `scripts/vast_onstart.sh` will:

1. check whether the TRT-stagewise venv and key model files already exist
2. if missing, run `setup_musetalk.sh` with the configured `VENV_PATH`
3. call `scripts/vast_server_ctl.sh start`
4. wait until `http://127.0.0.1:$PORT/health` responds

If the server is already running, it does not start a duplicate process.

## Useful Environment Overrides

All of these can be set in Vast.ai environment variables or inline in the
on-start command:

- `PROFILE=baseline|throughput_record`
- `PORT=8000`
- `HOST=0.0.0.0`
- `REPO_ROOT=/workspace/MuseTalk`
- `VENV_PATH=/workspace/.venvs/musetalk_trt_stagewise`
- `AUTO_SETUP=1`
- `SETUP_CLEAN=0`
- `SETUP_SKIP_APT=auto`
- `SETUP_SKIP_WEIGHTS=0`
- `STARTUP_TIMEOUT_SECONDS=600`

`SETUP_SKIP_APT=auto` means:

- if the script is running as root, it allows apt installation on first boot
- if it is not running as root, it automatically falls back to `--skip-apt`

## Manual Control Over SSH

Once the instance is up, you can manage the server with:

```bash
cd /workspace/MuseTalk
bash scripts/vast_server_ctl.sh status
bash scripts/vast_server_ctl.sh logs
bash scripts/vast_server_ctl.sh restart
```

## Suggested MVP Rollout

For the first autoscaling pass:

1. keep the repo on the persistent Vast workspace
2. use `scripts/vast_onstart.sh` as the instance boot command
3. start with `PROFILE=baseline`
4. verify with `bash scripts/vast_server_ctl.sh status`
5. hit `http://<instance-ip>:8000/health`

Once that is stable, switch selected instances to:

```bash
PROFILE=throughput_record bash scripts/vast_onstart.sh
```

## Notes

- these helpers do not replace the canonical setup / launch scripts
- they exist only to make boot-time automation safe and idempotent on Vast.ai
- if you later bake the venv and model weights into a custom image, keep
  `AUTO_SETUP=0` and use `scripts/vast_onstart.sh` only as the startup wrapper
