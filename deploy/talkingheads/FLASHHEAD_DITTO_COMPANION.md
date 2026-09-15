# Add SoulX-FlashHead and Ditto beside an existing MuseTalk installation

This companion installs **your SoulX-FlashHead fork** and
**[antgroup/ditto-talkinghead](https://github.com/antgroup/ditto-talkinghead)**,
including their separate dependencies and checkpoint downloads. It does not
invoke the MuseTalk installer or modify its checkout, Python environment,
configuration, secrets, or serving process.

## Run now on the new Vast host

Leave your current MuseTalk startup running. In a separate SSH terminal:

```bash
curl -fL --retry 5 \
  https://raw.githubusercontent.com/AhmadAFS1/MuseTalk/deploy/vast-talkingheads-20260915/deploy/talkingheads/vast-flashhead-ditto.sh \
  -o /workspace/vast-flashhead-ditto.sh
nohup bash /workspace/vast-flashhead-ditto.sh \
  > /workspace/flashhead-ditto-launch.log 2>&1 < /dev/null &
```

Save the script outside `/workspace/MuseTalk`: the supplied MuseTalk bootstrap
replaces that checkout. Do not restart that bootstrap to start this companion.
Public GitHub repositories need no token; private source access requires your
Git credentials or an SSH `SOULX_URL` configured separately.

Watch progress:

```bash
tail -f /workspace/logs/flashhead-ditto/install.log
```

Success ends with `FLASHHEAD + DITTO INSTALL COMPLETE`. A failure names the line
and exits. The `nohup` command keeps the installer alive when SSH disconnects.

For future starts, run the saved script in its own job, or invoke it after the
existing MuseTalk script finishes. It has an independent single-installer lock
and completion markers. Do not run the older three-repository installer on top
of an in-progress MuseTalk setup.

## Coordination with the supplied MuseTalk bootstrap

Default behavior: wait up to four hours for the latest `VAST_ONSTART COMPLETE:`
marker in `/workspace/onstart.log`, checking every 15 seconds. A later
`VAST_ONSTART BEGIN:` supersedes any earlier completion. A MuseTalk failure marker
stops the companion. If MuseTalk fails before creating this log, the companion
waits until timeout; inspect `/workspace/bootstrap.log` too.

This queues package installation to avoid two installers competing for apt/dpkg
locks or exhausting shared disk during first boot. It does not wait on the
MuseTalk `flock`: that descriptor can remain inherited by the background server.
The additional environments still share host CPU, disk and network with MuseTalk.

If MuseTalk was installed through a different path, **after verifying its setup
has finished**, bypass the log wait:

```bash
WAIT_FOR_MUSETALK=0 bash /workspace/vast-flashhead-ditto.sh
```

Overrides: `MUSETALK_LOG`, `MUSETALK_WAIT_SECONDS`, `WORKSPACE`, `SOULX_MODELS=lite`,
`MIN_FREE_GIB`, `SOULX_URL`/`SOULX_REF`, `DITTO_URL`/`DITTO_REF`.
A previous log's completion is not sufficient when another bootstrap is just
starting; for simultaneous future reboot jobs, launch the companion only after
the MuseTalk command returns, or ensure its new BEGIN marker exists first.

## Installed layout and scope

| Component | Location / runtime |
|---|---|
| SoulX code and environment | `/workspace/SoulX-FlashHead`, `.venv`; torch 2.7.1 cu128, FlashAttention 2.8.0.post2, WebRTC dependencies |
| SoulX models | `SoulX-FlashHead/models`; Lite + Pro, corresponding VAEs, wav2vec2 |
| Ditto code | `/workspace/ditto-talkinghead`; upstream commit `c3e47ee` |
| Ditto environment | `/workspace/.venvs/ditto`; torch 2.5.1 cu121, TensorRT 8.6.1 and upstream TRT dependencies |
| Ditto checkpoints | `ditto-talkinghead/checkpoints`; TRT, ONNX, PyTorch files and configs |
| Companion state / temporary files | `/workspace/.flashhead-ditto` |
| Companion HF cache | `/workspace/.cache/flashhead-ditto/huggingface` |
| Installation log | `/workspace/logs/flashhead-ditto/install.log` |

Ditto's configured execution path is **TensorRT**. Its alternative PyTorch/ONNX
execution path additionally needs ONNX Runtime and MediaPipe; downloaded
PyTorch checkpoint files do not imply that alternative environment was tested.
SoulX's optional Kokoro TTS and experimental TensorRT backends are not installed.
The fork's basic WebRTC service is included.

Use Ubuntu 22.04 x86_64 / Python 3.10 and host driver >=570.26. The companion uses
prebuilt CUDA wheels, so it does not replace the container's existing CUDA
toolkit. The pinned Ditto torch/cu121 stack is for Ampere/Ada/Hopper GPUs;
Blackwell/RTX 50-series needs a separate dependency port and is rejected before
downloading large packages.

Budget **65 GiB free after MuseTalk finishes** for installation. Expected added
steady usage is roughly **40–50 GiB**, with temporary download/unpack space above
that; these are estimates, not a measured fresh install. 200 GB total remains a
reasonable allocation. `SOULX_MODELS=lite` saves about 6.1 GiB. Interrupted installs
with a completed dependency stage use a 20-GiB resume floor; that floor is a guard,
not a guarantee that every remaining package/download will fit.

## Launch after installation

No servers are started automatically, and no GPU tensor arithmetic is run by
default. Import checks do not establish inference quality or performance.
Optional `RUN_GPU_SMOKE=1` performs a small CUDA tensor check; use it only when
that small allocation is acceptable alongside the serving workload.

SoulX WebRTC:

```bash
cd /workspace/SoulX-FlashHead
bash start_webrtc.sh --port 8001
```

Ditto's upstream sample inference:

```bash
cd /workspace/ditto-talkinghead
/workspace/.venvs/ditto/bin/python inference.py \
  --data_root ./checkpoints/ditto_trt_Ampere_Plus \
  --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl \
  --audio_path ./example/audio.wav \
  --source_path ./example/image.png \
  --output_path ./tmp/result.mp4
```

Only launch when there is enough free VRAM for the additional model. Existing
MuseTalk use reduces that headroom. Ditto's supplied TensorRT engines may need
rebuilding for your GPU; see the upstream conversion instructions and the
[original installation guide](README.md#launch-and-verify-on-the-new-gpu).

## Validation and GPU provenance

Development/audit host: **NVIDIA GeForce RTX 4070, 12 GB (12,282 MiB visible),
driver 570.181**, observed during this change. Executed checks are Bash syntax,
ShellCheck, plan mode, mocked completion/failure/timeout coordination, pinned
checkout/retry behavior and stage-resume controls. These are **CPU/static tests**;
no new GPU inference workload or fresh companion installation was run here.
The new Vast host's GPU is **unverified** until its own log is available. The
script records its GPU model, visible VRAM and driver in `.flashhead-ditto/gpu-host.csv`.

The supplied MuseTalk script contained AWS credentials. They were not copied
into this installer or guide. Rotate the exposed credential and provision its
replacement separately; this companion requires no AWS credentials.
