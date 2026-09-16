# Same-avatar comparison and TensorRT repair

GPU: **NVIDIA GeForce RTX 4070 SUPER**, 12,282 MiB physical/visible VRAM reported by `nvidia-smi`, compute capability 8.9. Driver 595.84. Tested 2026-09-15 UTC. Each model ran separately, without other GPU model processes during measured runs.

## Shared inputs and video

[Watch the three-way video](three-way-same-avatar.mp4) — 10 seconds, 1152 × 800, H.264 with shared AAC audio; panels play at 25 FPS.

The original Indian tutor JPG referenced by the old manifest was absent. We extracted frame zero from `MuseTalk/generated/downloads/indian_tutor_essential_six_v1/neutral_resting.mp4` and resized it once to 384 × 672. All three models used this same [portrait](shared.png) and [10-second audio](audio.wav). MuseTalk requires a video avatar, so it received a one-second static loop of that portrait, without pre-generated head motion. All model outputs are 384 × 672. SoulX and Ditto used seed 50.

The comparison aligns speaking media, removing MuseTalk's startup idle frames using recorded RTP timestamps. It shows appearance and lip sync at normal playback speed. Initial waiting times and generation throughput are measured separately below; this is a composite of separate runs, not simultaneous GPU execution.

## Measured results

| Model | Interface | Warm latency | Playback/result | Throughput scope |
|---|---|---|---|---|
| MuseTalk, rebuilt TRT UNet + mixed INT8 VAE | WebRTC | 1.005 s to live readiness | 250 source frames; 24.4 received FPS during playback | Paced at 25 FPS; this is not maximum inference throughput |
| SoulX-FlashHead Lite, compiled, 4 steps, batch 1 | WebRTC | 0.806 s to first received video | 250 frames; 0.000 s stalls | 28.23 FPS through server generation completion; 21.17 FPS including connection/startup/drain |
| Ditto, TensorRT Ampere_Plus | Offline | 1.134 s to first frame handed to writer | 250 frames / 7.700 s | 32.47 FPS including setup, generation, video writing and muxing; model load excluded |

Latency definitions differ: MuseTalk includes its two-second prebuffer policy; Ditto has no tested WebRTC path. These figures do not establish a uniform end-to-end WebRTC ranking. One warm measurement per model is a baseline, not a latency distribution or capacity test. SoulX's recorded clip was captured separately from its measurement run to avoid recording overhead.

Ditto's first run was 8.651 s (28.90 FPS), with 2.183 s model loading measured separately.

## TensorRT cause and repair

The downloaded VAE engine failed with `expecting compute 8.9 got compute 8.6`. The downloaded UNet's serialized device metadata contains `0%8%6%0%NVIDIA GeForce RTX 3090`; loading it separately failed with `No compatible device was found`. This was a precompiled artifact compatibility failure, not evidence that the RTX 4070 SUPER cannot run TensorRT.

The cache loader reused existing engine files without checking this GPU, and the profile selector accepted a historical validation flag without a device check. Previous successful tests on other machines are not reproduced here; their cached engines and profiles are unknown. NVIDIA documents that serialized engines have hardware compatibility constraints: [TensorRT compatibility documentation](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/version-compatibility.html).

Rebuilt all five INT8 VAE plans from the supplied QDQ ONNX graphs on this GPU. Decoder comparison against PyTorch on eight real predicted latent frames: mean absolute pixel error **0.005869**, maximum **0.115479**, on a 0–1 scale. Rebuilt the batch-8 FP16 UNet from model weights and real input captures; numerical validation on 16 batches passed the existing MAE ≤ 0.01 and maximum absolute error ≤ 0.5 limits. See [UNet validation](unet-validation.json) and [VAE validation](vae-validation/report.json).

Installed validated engines at the original default paths, retaining the downloaded originals under `/workspace/MuseTalk/models/trt_downloaded_backup_20260915`. MuseTalk was tested with TensorRT fallback disabled and restored on port 8000. Temporary SoulX benchmarking service was stopped. No Ditto server was launched.

Runtime versions: MuseTalk PyTorch 2.5.1+cu121 / CUDA 12.1 / TensorRT 10.3.0; SoulX PyTorch 2.7.1+cu128 / CUDA 12.8; Ditto PyTorch 2.5.1+cu121 / CUDA 12.1 / TensorRT 8.6.1. The CUDA 13.2 displayed by `nvidia-smi` is driver capability, not these environments' runtime version.

## Still unverified

- WAN/browser/mobile WebRTC latency, jitter and packet loss; these tests used a client on the same machine.
- Ditto over WebRTC, simultaneous three-model GPU capacity, long calls, and concurrent-user capacity.
- Reboot behavior or future artifact downloads overwriting these GPU-local engines. The provided `run-musetalk-local-trt.sh` launcher disables artifact restoration and selects the local profile.
- Other GPU models and machine environments. Local rebuilt engines should not be assumed portable.
- Lip-sync/identity quality beyond visual inspection of the comparison; no objective quality score or human study was run.

## Evidence

- [MuseTalk measured session](musetalk-load-detailed.json), [server log](musetalk-server.log), [final recording metadata](musetalk-trt.mp4.json)
- [SoulX measured session](soulx-metrics.json), [recording metadata](soulx-record.json)
- [Ditto timings](ditto.json)
- [VAE build](vae-build.json), [UNet build log](unet-build.log), [installed engine paths](installed-engines.json)
- [Video alignment](comparison-alignment.json)

Input SHA-256:

```text
shared.png cb7f390757c2d1a5e782a5fe0a0660d5d0fc848d54fc1f6eed72e4a3757feac9
audio.wav  999011601ba01d0c4cba544222997ce421f219419e4e3e9471a4ca056742ba22
```
