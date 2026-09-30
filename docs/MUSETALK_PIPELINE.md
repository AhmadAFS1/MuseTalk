# The MuseTalk pipeline: components, and what a machine needs beyond git

This covers what every part of the MuseTalk server does, why it is there, and how much it matters. It then lists
every file a serving machine needs that is not in git, and says where each should be stored and restored from.

Snapshot of `main` at `f09b461` on the RTX 4070 SUPER box, 2026-09-30. "Live" means the WebRTC server that the
companion platform calls. "Offline" means work done once per avatar or once per GPU type.

## 1. The pipeline at a glance

```mermaid
flowchart LR
  subgraph once_per_avatar[Once per avatar]
    SRC[source video<br/>character_factory / Kling] --> PREP[avatar prep<br/>DWPose + S3FD + BiSeNet + SD-VAE encode]
    PREP --> AV[(results/v15/avatars/id<br/>+ S3 avatars/v15)]
  end
  subgraph per_turn[Per call turn, live]
    AUD[turn audio] --> TL[ffmpeg trim]
    TL --> WH[Whisper-tiny<br/>audio features]
    AV --> CACHE[avatar cache in RAM]
    WH --> SCH[GPU scheduler<br/>batches of 16 frames across calls]
    CACHE --> SCH
    SCH --> UNET[UNet<br/>TensorRT r5 engines]
    UNET --> DEC[TAESD decoder<br/>TensorRT]
    DEC --> COMP[compose / blend<br/>CPU]
  end
  subgraph delivery[Delivery, live]
    COMP --> TRK[WebRTC video track<br/>idle / live switch, pacing]
    TL --> ATRK[audio track, A/V sync]
    TRK --> ENC[VP8 encoder]
    ENC --> TURN[coturn relay] --> BROWSER[browser]
    ATRK --> TURN
  end
```

In one sentence: each avatar is prepared once into per-frame face latents and blend masks. At call time, every
audio turn becomes Whisper features. The UNet turns features plus face latents into new mouth-region latents 16
frames at a time, the decoder turns those into pixels, and the CPU pastes them back into the avatar's frames. Each
call's WebRTC track then paces the frames out at 20 fps, together with the audio.

## 2. Components

For each component: what it does, why it is there, what depends on it, and where it lives. GPU and CPU mark where
the work runs.

### 2.1 Creating the avatar (upstream of MuseTalk)

| Component | What and why | Where |
|---|---|---|
| **Portrait and source video** | MuseTalk only moves the mouth region, so it needs a short video of the character to animate. The video comes from `character_factory/` (H3 three-pose, LTX, Comfy) or from `POST /avatars/generate`, which uses an OpenAI image model and then Segmind Kling motion. External APIs, not this GPU. | `character_factory/`, `scripts/avatar_generation.py` |

### 2.2 Avatar preparation (once per avatar; GPU and CPU)

Run by `POST /avatars/prepare` through `APIAvatar(preparation=True)` in `scripts/api_avatar.py`. The output is
everything the live path reuses, so no face analysis happens during a call.

| Step | What and why | Model and artifact |
|---|---|---|
| Frame extraction (CPU) | Splits the source video into frames. Frames are stored forward and then reversed (a "ping-pong" cycle), so an idle loop never jumps. | `full_imgs/*.png` |
| Face detection and landmarks (GPU) | Finds the face in every frame and builds the crop box that the UNet will repaint. Landmarks come from DWPose (68 face points); S3FD confirms that a face exists. | `models/dwpose/dw-ll_ucoco_384.pth`, `models/face_detection/s3fd.pth` → `coords.pkl` |
| Face latents (GPU) | Encodes each 256×256 face crop with the Stable Diffusion VAE. The result is 8 channels: the crop with its lower half masked out, plus a reference crop. This is the UNet's picture input, computed once per avatar instead of once per frame. | `models/sd-vae` → `latents.pt` |
| Blend masks (GPU and CPU) | BiSeNet face parsing marks the jaw and cheek region, and a blurred mask decides where generated pixels replace original ones. This gives a seamless mouth edit. | `models/face-parse-bisent/*` → `mask/*.png`, `mask_coords.pkl` |
| **Result on disk** | `results/v15/avatars/<id>/`: `avator_info.json` (the typo is in the code), `input_video.mp4`, frames, masks, `coords.pkl`, `mask_coords.pkl`, `latents.pt`. Typically 160–350 MB per avatar. | 33 avatars, 8.4 GB on this box |
| **S3 copy** | With `AVATAR_S3_ENABLED=1`, prepare uploads `s3://lingua-musetalk-s3-storage/avatars/v15/<id>.tar.gz`. A worker that lacks the avatar restores it on first use (`avatar_manager_parallel._restore_avatar_from_s3_if_needed`). | `scripts/avatar_s3_store.py` |

**Risk:** `POST /avatars/prepare` runs on the server's event loop, not in a worker thread. Preparing an avatar on a
worker that is serving calls would freeze every live stream until it finishes. `/avatars/generate` does use a thread.

### 2.3 Loading an avatar for calls (CPU and RAM)

| Component | What and why |
|---|---|
| **Avatar cache** (`scripts/avatar_cache.py`) | Keeps loaded avatars in RAM (TTL + LRU + a memory cap), so a turn never reads the disk. It is per process: the platform must send a call's requests to the worker that warmed its avatar. |
| **Warm endpoint** (`POST /avatars/{id}/cache/warm`) | Restores the avatar from S3 if needed, loads it, and with `WEBRTC_IDLE_FRAME_CACHE_WARM=1` also decodes its idle video. The control plane calls this before routing a call. |
| **Compose plans** | Per-frame paste regions and alpha masks. They are rebuilt on every load and never stored. |
| **Lean layout flags** (`MUSETALK_AVATAR_FRAME_STORE/MASK_STORE=png`, `MASK_CHANNELS=1`, `PLAN_FLOAT_ALPHA=0`) | Keep frames and masks PNG-compressed in RAM and decode them on demand. Output is bit-exact with less memory, which is what fits 15 avatars on this box. |

### 2.4 Per-turn generation (live)

| Step | What and why | Where and model |
|---|---|---|
| Upload and audio timeline (CPU, ffmpeg) | Takes the turn's audio (`POST /webrtc/sessions/{id}/stream`) and trims leading and trailing silence. The same trimmed file drives both lip-sync and playback, so they cannot drift. | `webrtc_audio_timeline.py` |
| **Whisper features** (CPU mel + GPU encoder) | Converts speech into per-frame feature windows (50×384) that the UNet reads through cross-attention. Only the Whisper *encoder* is used; nothing is transcribed. | `models/whisper` (openai/whisper-tiny), `musetalk/utils/audio_processor.py` |
| **GPU scheduler** (`scripts/hls_gpu_scheduler.py`) | One thread batches frames from every active call into GPU batches. With the r5 engines the batch is exactly 16, padded when short. This is how one GPU serves many calls: a batch of 16 is far more efficient per frame than 15 single-frame calls. Prep workers (Whisper) and compose workers (blending) run around it. | `HLS_SCHEDULER_*` |
| **UNet** (GPU) | The core model: MuseTalk v1.5, one denoising step, ~80% of GPU time. From face latents and Whisper features it produces the new mouth-region latent. Backends:<br/>• PyTorch (default, slowest)<br/>• TensorRT `.ts` bs8 (the 252 fps "BEFORE" profile)<br/>• **TensorRT stagewise bs16**: 11 per-block engines in one CUDA graph. The r2 set (350 fps) is FP16 plus 2 INT8 blocks; the **r5 set (~400 fps)** has 7 INT8 blocks chosen by error per operation. | `models/musetalkV15/unet.pth` (source weights), `models/tensorrt_unet_stagewise_sm89_srcg50` (r5), `scripts/unet_stagewise_trt.py` |
| **Decoder** (GPU) | Turns latents back into a 256×256 face. The original SD-VAE decoder is accurate but slow. **TAESD** is a tiny distilled decoder at a fraction of the cost, and its TensorRT build also produces uint8 BGR on the GPU. | `models/taesd` (weights), `models/taesd/trt/taesd_trt_6111…` (bs8 engine), `scripts/vae_fast_decoder.py` |
| **Compose / blend** (CPU, 6–10 threads) | Resizes the generated face into the crop box and alpha-blends it into the avatar frame with the mask. Fixed-point math keeps it exact and fast. | `musetalk/utils/blending.py` |
| *Chin extension / FaceMesh chin tracker* | **Offline only.** The "100% chin" recipe in the fps records was measured by the offline harness (`scripts/chin_multistream_render.py`). The live server uses the standard blend. | `character_factory/h3_avatar_workflow/chin.py`; needs mediapipe |

### 2.5 Delivery (live; CPU and network)

| Component | What and why | Where |
|---|---|---|
| **WebRTC session** | One aiortc `RTCPeerConnection` per viewer: a video track, a persistent audio track, a sync clock, and a pose router. | `scripts/webrtc_manager.py`, `api_server.py` (`/webrtc/sessions/*`) |
| **Video track** (`SwitchableVideoStreamTrack`) | Plays the avatar's idle video between turns and switches to generated frames once enough are buffered (strict FIFO, 2 s prebuffer). It paces output at 20 fps on a fixed deadline. This is the part that decides whether a viewer sees buffering. | `scripts/webrtc_tracks.py` |
| **Audio track and A/V sync** | A silent audio transport runs for the whole call. Each turn's speech is armed on it, and the first speech audio and first live video frame are released together. | `SyncedAudioStreamTrack`, `VideoSyncClock` |
| **Idle frame cache** | Decodes each avatar's idle video once per process instead of on the event loop in every session. | `scripts/webrtc_idle_frame_cache.py` |
| **Pose sets and motion bank** | Optional multi-pose avatars (resting, speaking, smiling) with optical-flow transitions. | `pose_protocol.py`, `webrtc_pose_router.py`, `webrtc_motion_playback.py` |
| **Video encoder** | **VP8 in practice.** `prefer_h264()` runs after `setRemoteDescription`, too late to apply, so the answer takes the client's first codec, VP8 for Chrome and aiortc. H.264 options (libx264, NVENC) exist but are not negotiated. A codec decision is pending. | aiortc, `webrtc_h264_override.py` |
| **TURN (coturn)** | Relays media for browsers that cannot reach the machine directly (Vast exposes only a few ports). Its relay-port range caps concurrent calls: the running coturn has 41 ports, about 11 calls when the server also relays and about 20 when only browsers do (`experiments/live15_r5/run_demo.sh`). | `scripts/run_turnserver_tcp_relay.sh`, `.env.webrtc-turn.local` |
| **The event loop** | One Python asyncio loop paces every call. Anything that holds it (GC, blocking calls, GIL-heavy threads) freezes all calls at once, and it is today's limit on concurrent calls, not the GPU. The flag-gated fixes are in `experiments/live15_r5/loopfix.env`; see `docs/fps_comparisons/live15_r5_20260929/README.md`. | `api_server.py`, `scripts/gc_tuning.py` |
| Groups and wall pages | Test pages that open N calls in one browser. | `/webrtc/groups/*`, `templates/webrtc_wall.py` |
| HLS mode | An alternative delivery mode: the same generation, with ffmpeg chunks (NVENC by default) written to a playlist. | `/hls/*` |
| Kokoro TTS | Local text-to-speech on CPU, a test aid for the wall. Real calls send their own audio. `MUSETALK_DISABLE_LOCAL_TTS=1` turns it off. | `scripts/kokoro_tts.py`, HF cache `hexgrad/Kokoro-82M` |
| Worker control plane | Registers the worker with the Lingua platform, sends heartbeats with load metrics, and supports draining. | `scripts/worker_control_plane.py` |

### 2.6 Startup and configuration

| Component | What and why | Where |
|---|---|---|
| Installer | apt packages (ffmpeg, coturn), the venv from pinned requirements, weights, the native VP8 build, and a GPU self-test. | `scripts/install_musetalk.sh`, `requirements/*.in` + `constraints-cu121.txt`, `download_weights.sh` |
| Secrets bootstrap | Reads the worker secret (`lingua/musetalk-worker-runtime` in AWS Secrets Manager) into the environment: S3 bucket names, keys, and Lingua control-plane settings. | `scripts/bootstrap_aws_secrets.py` |
| Boot script | install check → secrets → TURN → engines → server start → registration. | `scripts/vast_onstart.sh`, `scripts/vast_server_ctl.sh` |
| Resolver and recipes | Pick backends and batch sizes for the GPU. `fast` (default): `.ts` UNet if present plus TAESD. `fast300`: adds TAESD TRT and stagewise, but its r5 levers are still commented out. `legacy_int8`: the old RTX 3090 profile. Override files (`MUSETALK_ENV_OVERRIDES_FILE`) sit on top, and that is how r5 was run (`experiments/live15_r5/common.env`). | `scripts/musetalk_host_profile.py`, `configs/recipes/*.env`, `scripts/run_musetalk_server.sh` |
| Engine store | Builds, adopts, validates, publishes and restores TensorRT engines by fingerprint (GPU, TensorRT version). It **cannot handle the r5 set**: the source-cache layout, INT8 blocks, and r5 fails its default validation bar. | `scripts/unet_engine_store.py` |

## 3. What a machine needs that is not in git

Sizes are from this box. **Portable** works on any GPU. **GPU-bound** only works on the same GPU architecture
(sm_89 here), TensorRT version (10.3.0) and, in practice, the same driver generation.

### 3.1 Needed to serve r5

| Item | Size | Portable? | Produced by | Stored today |
|---|---|---|---|---|
| MuseTalk v1.5 UNet weights `models/musetalkV15` | 3.2 GB | yes | `download_weights.sh` (HF `TMElyralab/MuseTalk`) | re-downloaded per machine |
| SD-VAE `models/sd-vae` | 320 MB | yes | HF `stabilityai/sd-vae-ft-mse` | re-downloaded |
| Whisper-tiny `models/whisper` | 145 MB | yes | HF `openai/whisper-tiny` | re-downloaded |
| TAESD weights `models/taesd` | 5 MB | yes | HF `madebyollin/taesd` (pinned revision) | re-downloaded |
| **r5 + r2 UNet engines** (`…_srcg50`, `…_srcmix`, and the 4 block folders they link into) | 2.5 GB | **GPU-bound** | `scripts/repro_400fps/10_build_engines.sh` (20–25 min, 14 GB RAM, needs the calibration data below) | **this machine only.** A bundle is built (`tmp/trt_bundle/`), and its upload waits on a publisher login |
| **TAESD TRT engine** `models/taesd/trt/taesd_trt_6111…` | 10 MB | **GPU-bound** | `vae_fast_decoder.py build` (~15 s) | this machine only (in the same bundle) |
| Prepared avatars `results/v15/avatars/*` | 160–350 MB each | yes | `POST /avatars/prepare` | S3 `avatars/v15/<id>.tar.gz` for avatars prepared with S3 on. **Which of the 33 local avatars are in S3 has not been checked** |
| Python venv `/workspace/.venvs/musetalk_trt_stagewise` | 9.5 GB | per CUDA and OS | installer, from pinned requirements | rebuilt per machine (~minutes) |
| Secrets (S3 keys, Lingua token) | – | – | operator | Secrets Manager. This box also has a static copy in `/workspace/.musetalk-runtime.env` |
| TURN password `.env.webrtc-turn.local` | – | per machine | generated at boot | local only (fine) |

### 3.2 Needed only to prepare avatars or rebuild engines

| Item | Size | Portable? | Produced by | Stored today |
|---|---|---|---|---|
| DWPose, S3FD, BiSeNet face parsing | ~570 MB | yes | `download_weights.sh`. The face-parsing weights come from a **Google Drive link** (fragile) | re-downloaded |
| **INT8 calibration data** `calibration/unet_multi_avatar_20260928` | 218 MB | yes | `scripts/build_unet_multi_avatar_corpus.py` from specific avatars and audio | this machine only (in the r5 bundle). Needed for an exact r5 rebuild |
| Quality-harness avatars `/workspace/experiments/avatar_diversity_20260927` | 763 MB | yes | external portrait and video tools | this machine only; **cannot be regenerated** |
| FaceMesh environment (chin tracker, quality tools) | uses mediapipe from `/workspace/SoulX-FlashHead/.venv` | yes | `install_musetalk.sh --with-chin-tools` defines a proper one | borrowed from another project's venv |
| SyncNet `models/syncnet` | 1.4 GB | yes | HF `ByteDance/LatentSync` | re-downloaded; quality checks only |

### 3.3 Evidence and history (not needed to run)

| Item | Size | Stored today |
|---|---|---|
| Sign-off and comparison videos `experiments/video_validation/` | 1.4 GB | this machine only |
| Raw quality and render captures `docs/fps_comparisons/*/chin_multistream/{V_*,Q_*}` | 2.5 GB | this machine only |
| Live-test runs `tmp/live15_r5/` | 533 MB | this machine only; the summaries are in git |

### 3.4 On disk but not needed (cleanup candidates, after a backup)

| Item | Size |
|---|---|
| Old RTX 3090 engines `models/trt_downloaded_backup_20260915` (sm86; unusable here, and in S3 already) | 2.2 GB |
| Experiment engine folders (r3/r4 sets, other recipe candidates, partial builds) | ~5.4 GB |
| Unused bs32 stagewise engines | 1.6 GB |
| Duplicate S3FD copy (`models/auxiliary` = `models/face_detection`) | 86 MB |
| `.ts` bs8 UNet `models/tensorrt_unet_static_bs8_20260529` | 2.1 GB, still used by the current default launchers here |

## 4. Where each kind of artifact should live

The rule: **git for code and small inputs; S3 for anything expensive or impossible to recreate; rebuild or
re-download everything else.** Every S3 object is checksum-addressed, so a restore can verify it.

| Kind | Store in | How | Status |
|---|---|---|---|
| Code, configs, recipes, small corpora, docs | **git** | – | done |
| **TensorRT engines** (GPU-bound) | **S3 `trt-artifacts/<gpu>/<profile>/sha256-<hash>/<bundle>.tar.gz`**, the existing convention | `trt_artifact_bundle.py` format: manifest + SHA256SUMS inside, restore verifies every file. One bundle per GPU type and TensorRT version. Record GPU, TensorRT and driver in the bundle's notes | r5/r2 bundle built and verified locally (2.58 GB, sha256 `8e3f4b56…3ebff`, 535 files); **upload pending `aws login` for the publisher profile** |
| INT8 calibration data | S3, inside the engine bundle (already) | – | done in the bundle |
| **Prepared avatars** | S3 `avatars/v15/<id>.tar.gz` (existing flow, restored lazily) | keep `AVATAR_S3_ENABLED=1` wherever avatars are prepared | exists; check the 33 local avatars against S3 |
| Base model weights (5.7 GB) | public sources + **an S3 mirror** `models/<name>/sha256-…` | the installer downloads from HF; the mirror is insurance against link rot (the Google Drive face-parsing file especially) and HF rate limits | not done (recommended) |
| Quality-harness avatars, evidence videos, raw captures | S3 archive prefix (Infrequent Access / Glacier) | one bundle per record, e.g. `evidence/4070s_400fps_20260928/…` | not done (recommended for the harness avatars; optional for the rest) |
| Python environments | rebuild from pinned requirements; optionally a Docker image | `install_musetalk.sh --matrix cu121` | done; an image would cut boot time |
| Secrets | AWS Secrets Manager only | `bootstrap_aws_secrets.py` | done; the static keys on this box are a local exception |
| Runtime-generated state (`.runtime/*.env`, idle frame cache, compose plans, TensorRT timing caches, `/dev/shm` arenas) | nowhere; regenerated | – | – |

**IAM:** serving machines read with `musetalk-s3-runtime`, which can read `trt-artifacts/*` but not write it (by
design). Publishing needs a separate identity with `s3:PutObject` on `trt-artifacts/*`. Today that is the
`lumatalk-root` login (`AWS_PROFILE=lumatalk-avatar-batch`).

## 5. Gaps found while writing this

1. **A fresh machine cannot serve r5 yet.** No boot step restores the r5 engines. The engine store cannot adopt or
   build them, and the launcher's preflight stops the server when the pinned engine folder is missing. Fix: publish
   the r5 bundle, then add a restore step for `fast300`, the same way `legacy_int8` restores the RTX 3090 bundle,
   before engine provisioning. Alternatively, teach the engine store the source-cache and INT8 layout.
2. **No recipe selects r5.** Its levers in `configs/recipes/fast300.env` are commented out, and r5 has run only
   through `experiments/live15_r5/common.env`. The launchers installed on this box
   (`experiments/chinese_bob_webrtc_20260927/run_local_api.sh`, `/workspace/run-musetalk-local-trt.sh`) serve the
   `.ts` bs8 UNet, the 252 fps class.
3. **The TAESD TRT engine** is built at boot only under `fast300`, and its quality gate reads FAIL (5 LSB against a
   3 LSB bar). With strict mode and no engine, the launcher stops.
4. **`fast300` always ensures the `.ts` UNet as well** (a 7-minute build) even though r5 does not use it. Set
   `MUSETALK_UNET_ENGINE_PROVISION=off` for r5 machines.
5. **Stale docs.** `docs/musetalk_worker_secrets.md` and `docs/trt_artifacts/README.md` still describe the RTX 3090
   restore as the default boot step. It runs only for `legacy_int8`. The tracked root files
   `.musetalk_trt_artifact_manifest.json` / `SHA256SUMS` describe that old bundle.
6. **`POST /avatars/prepare` blocks the event loop** (§2.2). Run it in a thread before preparing avatars on a
   serving worker.
7. **Video codec is VP8 by accident** (§2.5). This needs a decision.
8. **coturn has only 41 relay ports** (§2.5). Widen the range (`TURN_INTERNAL_RELAY_MAX_PORT`, which defaults to
   49460 in the current script) before running more than about 11 relayed calls per machine.
9. **Engines are tied to this machine's GPU, TensorRT version and driver.** A driver or TensorRT upgrade needs a
   rebuild and a re-gate. Keep the bundle's notes current.

## 6. Suggested order

1. Upload the r5/r2 bundle (needs the publisher login), then record its URI and sha256 in
   `docs/trt_artifacts/README.md`.
2. Add the `fast300` restore step and an r5 recipe, then verify on a fresh machine that boot → restore →
   `verify-log` passes.
3. Back up the quality-harness avatars (irreplaceable) and check the 33 avatars against S3.
4. Mirror the base weights to S3.
5. After the backups, delete the cleanup candidates in §3.4 (about 9 GB).
