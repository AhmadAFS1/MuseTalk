# Vast.ai: MuseTalk + SoulX-FlashHead + Ditto

> **GPU provenance — installer validation:** The audit host has an NVIDIA GeForce RTX 4070, 12 GB (12,282 MiB visible), driver 570.181. The installer checks performed on this host were shell/static, dependency-resolution and control-flow checks; no fresh three-repository GPU install or inference validation was performed. RTX 3090/4090/A100 above are deployment suggestions, not GPUs tested by this installer run. Existing MuseTalk RTX 3090 results are historical and documented separately.

## Instance settings

- Disk: **200 GB recommended** (about 186 GiB if Vast bills decimal GB).
- Image: Ubuntu 22.04 x86_64 with **CUDA 12.1 development toolkit**, e.g. the existing project template `vastai/pytorch:cuda-12.1.1-auto`; verify `nvcc --version` reports 12.1. A runtime-only CUDA image is insufficient for MuseTalk preprocessing.
- GPU: use RTX 3090 / 4090 or A100 for this pinned stack. At least 24 GB VRAM is a sensible starting point for running one model at a time. Do not use RTX 50-series with this MuseTalk/Ditto CUDA 12.1 stack without a separate port.
- Host NVIDIA driver: **570.26 or newer**, chosen to support SoulX's CUDA 12.8 PyTorch wheels without relying on limited minor-version compatibility. The container toolkit remains 12.1 for MuseTalk. See [NVIDIA's CUDA 12.8 release notes](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-toolkit-release-notes/).
- Private forks: configure Git authentication on the new host before installing, or supply `MUSETALK_URL` / `SOULX_URL` as authenticated SSH URLs. Do not paste credentials into this file.

## Run

Copy `vast-startup.sh` to `/workspace/vast-startup.sh` on the new instance. Paste this into Vast's on-start field:

```bash
bash /workspace/vast-startup.sh
```

For an entirely fresh instance, the adjacent `vast-onstart.sh` is the pasteable downloader. It fetches the installer from the deployment branch on GitHub using Git and then executes it. Public repositories need no token; private access uses `GITHUB_TOKEN` supplied separately in the instance environment. Git authentication for the private source clones must also be configured (the downloader sets up a temporary credential helper when `GITHUB_TOKEN` is supplied).

Preview without installation:

```bash
bash /workspace/vast-startup.sh --plan
```

Log: `/workspace/logs/talkingheads/install.log`. A failed command stops installation. Rerunning resumes successful installation stages, retries model downloads, and reruns import/GPU checks. Existing checkouts at another commit or with tracked edits are rejected. The script does not delete or upgrade existing work. Use an empty workspace on the new instance.

Overrides: `WORKSPACE`, the three `*_URL` / `*_REF` variables, `SOULX_MODELS=lite` (saves about 6.1 GiB), `MIN_FREE_GIB` (default 100). Source commits and SoulX/Ditto model revisions are pinned. MuseTalk's existing downloader uses its upstream default model revisions. Transitive Python dependencies are not fully locked; installed package lists are saved under `.talkingheads/` for diagnosis.

## What is installed

| Project | Runtime | Models |
|---|---|---|
| Your MuseTalk fork, `e8e5de5` | Python 3.10; torch 2.5.1 cu121; TRT 10.3; full avatar preparation stack | V1.5, VAE, Whisper, parsing, DWPose, SyncNet, S3FD |
| Your SoulX fork, `5033786` | Python 3.10; torch 2.7.1 cu128; FlashAttention 2.8.0.post2; WebRTC | Lite + Pro, both VAEs, wav2vec2 |
| [Upstream Ditto](https://github.com/antgroup/ditto-talkinghead), `c3e47ee` | Python 3.10; torch 2.5.1 cu121; TRT 8.6.1 | ONNX, Ampere+ TRT engines, PyTorch checkpoint files and configs |

Ditto's installed execution path is TensorRT. The PyTorch checkpoints are downloaded for later experimentation; its alternative PyTorch/ONNX execution path needs additional upstream dependencies (including ONNX Runtime and MediaPipe). No TTS weights, SoulX experimental TRT environments/engines, historical output media, local secrets, or prepared runtime caches are migrated.

Installation does not start three GPU processes. Launch the desired model after installation. Installing all three on disk does not guarantee enough VRAM or throughput to run all three concurrently.

## Launch and verify on the new GPU

### Ditto

```bash
cd /workspace/ditto-talkinghead
/workspace/.venvs/ditto/bin/python inference.py \
  --data_root ./checkpoints/ditto_trt_Ampere_Plus \
  --cfg_pkl ./checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl \
  --audio_path ./example/audio.wav \
  --source_path ./example/image.png \
  --output_path ./tmp/result.mp4
```

Check that the result plays with audio. If engine deserialization fails on the chosen GPU, rebuild locally with the supplied conversion script:

```bash
/workspace/.venvs/ditto/bin/python scripts/cvt_onnx_to_trt.py \
  --onnx_dir ./checkpoints/ditto_onnx --trt_dir ./checkpoints/ditto_trt_custom
```

Then use `--data_root ./checkpoints/ditto_trt_custom`. Reserve extra disk and time for conversion. These are the [upstream engine compatibility instructions](https://github.com/antgroup/ditto-talkinghead#-inference).

### SoulX

```bash
cd /workspace/SoulX-FlashHead
PATH="$PWD/.venv/bin:$PATH" bash inference_script_single_gpu_lite.sh
# Or launch the fork's WebRTC service:
bash start_webrtc.sh --port 8001
```

Use the repository's `WEBRTC.md` for client, authentication, and network configuration. Review the generated video before treating the new GPU as validated.

### MuseTalk

The dependencies and base weights are installed. The existing production launch additionally requires your S3 TensorRT artifact bundle and runtime configuration. Supply AWS access separately, then on a compatible RTX 3090 host:

```bash
cd /workspace/MuseTalk
AUTO_SETUP=0 SETUP_FULL_STACK=1 \
TRT_ARTIFACT_S3_BUCKET=lingua-musetalk-s3-storage \
PORT=8000 bash scripts/vast_onstart.sh
```

The existing wrapper restores and verifies the artifact and waits for health. See [artifact details](../../docs/trt_artifacts/README.md) and [Vast boot documentation](../../docs/vast_ai_boot.md). The published RTX 3090 engine must be rebuilt/validated for other GPU architectures; a driver upgrade alone does not make a serialized engine portable.

## Disk estimate (measured 2026-09-15)

Current machine: `df -h /workspace` reports 100 GiB total, 97 GiB used, 3.5 GiB available. `/workspace` occupies about 87 GiB. Measurements use `du`, so symlinks/shared files must not be counted twice.

| Current allocation | GiB, rounded |
|---|---:|
| MuseTalk checkout including models and history | 13 |
| MuseTalk environment | 9.3 |
| SoulX including models, environment, TRT experiment and compile caches | 33 |
| Shared wheelhouse | 7.8 |
| Workspace download/package caches | 11 |
| OmniVoice | 7.2 |
| Experiments outside the main checkouts | 6.3 |
| Root pip cache (outside workspace) | 5.8 |

Fresh-install estimate for this script:

| Component | Estimated GiB |
|---|---:|
| MuseTalk full runtime, base models, shallow source | 17–22 |
| SoulX runtime, Lite + Pro models, source | 25–30 |
| Ditto TRT runtime, all checkpoint formats, source | 15–20 |
| Container/system/tooling allowance | 8–13 |
| **Steady installation** | **65–85** |
| **Peak during installation / engine builds** | **90–120** |

The ranges for environments and installation peaks are estimates, not a measured clean boot. Checkpoint sizes were obtained from Hugging Face's file-size metadata:

- [Ditto](https://huggingface.co/digital-avatar/ditto-talkinghead/tree/main): **6.452 GiB** total; ONNX 2.359, PyTorch 2.156, TRT 1.938.
- [SoulX](https://huggingface.co/Soul-AILab/SoulX-FlashHead-1_3B/tree/main): **13.34 GiB** selected Lite/Pro/VAEs plus **0.352 GiB** for one wav2vec2 weight format. Lite-only is about 7.60 GiB including audio.
- Existing MuseTalk models total 8.0 GiB, including about 2.3 GiB of local TRT/cache artifacts; fresh base downloads are about 5.7 GiB.

**Choose 200 GB.** 100 GB can fit a lean install under favorable conditions, but cannot accommodate the upper installation estimate or meaningful experimental headroom. Migrating today's whole machine and adding Ditto/SoulX Pro would start around 120–130 GiB before new outputs; use 250–300 GB if keeping all experiment environments, duplicate caches, and growing video archives.

## Validation limits

Validated locally: Bash syntax, ShellCheck, plan mode, dependency resolution for SoulX (163 packages), actual local Git clone/repeat/dirty-checkout refusal, failed-stage marker handling and successful-stage skipping, source refs and Hugging Face model metadata. The existing MuseTalk/SoulX installations provided size and version evidence. **The three-repository installer has not completed a fresh GPU install here:** only 3.5 GiB remained at audit time. On the new machine, require `INSTALL COMPLETE`, then run the actual inference checks above. Imports and CUDA tensor arithmetic alone do not prove model inference or serving performance.
