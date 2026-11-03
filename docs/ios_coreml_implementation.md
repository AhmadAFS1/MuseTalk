# MuseTalk on Apple devices with Core ML: how it works and what has been done

*Living document. Started 2026-10-31; covers the work of 2026-10-29 to 2026-10-31 on branch `claude/ios-unet-bench`
(worktree `~/code/musetalk-ios-bench`, PR AhmadAFS1/MuseTalk#5, open, additive only: nothing under the server path
is touched). The chronological measurement log with every table and JSON pointer is
`docs/ios_unet_bench_2026-10-30.md` (referred to below as "the bench doc"); this document is the system
description and the tracker. When something here and in the bench doc disagree, the bench doc's later section wins
and this file needs an edit.*

---

## 0. In one paragraph

MuseTalk's live path (one UNet step plus a tiny decoder per video frame, driven by Whisper-tiny audio features,
pasted onto a pre-rendered avatar frame) has been ported to Core ML and runs end to end on the Apple Neural Engine
(ANE), on this Mac (M5 Pro) and on a real iPhone 12 Pro. The avatar stays a **premade cache** (frames, masks, face
boxes, UNet input latents) exported from the same server avatar directory the GPU workers use; only the per-frame
lipsync is computed on the device. The UNet ships as **two Core ML chunks with fp16 boundaries** (a single
850M-parameter program is killed by iOS), with int8 weights, in Apple's ANE-friendly layout. Measured: **59.6 ms per
UNet frame on the iPhone 12 Pro** (16–17 fps sustained, not the server's 20), 18.9 ms on the M5 Pro ANE. Quality:
the ANE's fp16 arithmetic puts the UNet at latent mae 0.013 against the server's own gate (limit 0.01, shipping
TensorRT INT8 is 0.003); in pixels that is 0.18/255 on the composed face, and an r5-style labelled review shows five
of six production characters inside the band the servers already accepted and one (Japanese) failing the lip
aperture thresholds. A human blind pass is still pending. A streaming lipsync view (bundled avatar + bundled clip,
audio-clock pacing, frame skipping, a two-stage pipeline) plays on the iPhone 12 Pro at **16–17 fps shown, no
stalls**, the audio never waiting for the video. The A14's Neural Engine is **compute-bound** (batch 4 moved the
frame by 3.8 %), so this phone stays below the 20 fps line and newer phones are the way to real time.

---

## 1. Status tracker

| # | Component | State | Evidence |
|---|---|---|---|
| 1 | UNet exported to Core ML (diffusers layout), fp16 / int8w / int8wa, batch 1/4/8 | done | bench doc §2, §4; `scripts/ios/export_coreml.py` |
| 2 | UNet in Apple's ANE layout (ml-stable-diffusion), fp16 / int8w, batch 1/4 | done, −17–20 % | §11.3; `export_coreml_ane.py` |
| 3 | UNet split in two chunks with fp16 boundaries (iOS requirement) | done, 100 % ANE on A14 | §12.3; `export_coreml_ane_chunks.py` |
| 4 | TAESD decoder in Core ML | done | §2; `export_coreml.py` |
| 5 | Whisper-tiny front end (log-mel + encoder) in Core ML | done, validated through the UNet | §6.2; `export_whisper_coreml.py` |
| 6 | Per-frame audio window gather ported to Swift | done, matches the repo to the ANE encoder's error | `HarnessCore/Lipsync.swift` |
| 7 | Fixed-point masked blend ported to Swift | done, 0.17/255 vs Python on the shipped JPEG | `Lipsync.swift`, `bench-mac --lipsync-check` |
| 8 | Avatar cache → phone bundle exporter | done (one avatar, 46 MB) | `scripts/ios/prepare_avatar_bundle.py` |
| 9 | On-device timing harness (Swift package + CLI + iOS app) with the repo gate | done, reproduces the Python bench on the Mac | §12.1–12.2; `ios/MuseTalkBench/` |
| 10 | iPhone 12 Pro measured (UNet, chain, sustained, Whisper, gate) | done | §12.3; `harness_iphone12pro_*.json` |
| 11 | Streaming lipsync view (avatar + clip, audio-clock pacing, frame skipping, two-stage pipeline) | **played end to end on the iPhone 12 Pro 2026-10-31: 16.4–17.1 fps generated and shown, 0 stalls, 0 late drops, audio never waited** (ANE stage 65.7 ms, CPU stage 10.4 ms hidden by the pipeline); two phone-only bugs fixed on the way (a frame-buffer self-deadlock, and a skip-ahead that aimed at the clock itself) | README "Lipsync tab"; `lipsync_iphone12pro_pipelined_run{1,2}.json`; `HarnessCore/Lipsync.swift` (`LipsyncPipeline`), `App/Sources/LipsyncPlayer.swift` |
| 12 | Repo quality gate on the Core ML UNet and attribution of the ANE error | done: fails at mae 0.013, cause is ANE conv/linear fp16 accumulate | §11.1–11.2 |
| 13 | r5-style labelled A/B review package (12 clips, 6 characters) | rendered, metrics computed; **human blind pass pending** | §12.4; `docs/ios_unet_bench/review_ane_vs_fp16/` |
| 14 | Speed levers tried and rejected (int8 activations, 6/4-bit palettization, fp32 islands, GroupNorm rewrite) | measured, rejected | §4, §11.2, §11.3 |
| 15 | Batch-4 two-chunk UNet: the experiment that tells bandwidth-bound from compute-bound on the A14 (§5.4) | **measured on the phone 2026-10-31: 54.2 vs 56.4 ms/frame (−3.8 %), gate identical; verdict: the A14 ANE is compute-bound** | bench doc §12.5; `harness_iphone12pro_b4_vs_b1.json`, `chunk_unet_anec0_int8w_b4_m5pro.json`, `harness_mac_b4_chunked_m5pro.json` |
| 16 | GPU accuracy anchor on the phone | not measured (killed with the single program); low value | §12.3 |
| 17 | Audio from a TTS WAV (any sample rate, any length) instead of the bundled clip | **done 2026-10-31**: `loadWaveform16k` + segmented windows; an OmniVoice 24 kHz WAV played on the phone (§5.2). A microphone / streaming source is still open | §2.4; `AudioInput.swift` |
| 18 | Avatar download from S3 at runtime | **done 2026-10-31** for one avatar: phone-only bundle format, `pack_phone_bundle.py`, `RemoteBundle` + `TarArchive`; Latina downloaded (27 MB in 5.4 s) and played on the phone. Multiple avatars and the pose router are open | §2.3; `avatars/ios/v1/` |
| 19 | Integration into the Lingua app (native module) | **Phase 0 done 2026-10-31**: `LipsyncSession` (the player core the module wraps) in HarnessCore, pre-roll parameter, model pack on S3 with a manifest; the Lingua backend half runs in its own session; plan in the Lingua repo | §6; Lingua `docs/musetalk-on-device-renderer-plan.md` |
| 20 | 4 GB phones: the memory allowance emulated on the 12 Pro (`--emulate-limit-mb`, `--ballast-mb`, `--mem-trace`) | **measured 2026-11-01**: the chunked cold compile peaks at 54 MB of process footprint (148 MB over the whole run, playback included); under a 2,098 MB cap compile + playback keep ≥ 1.9 GB (≥ 1.6 GB with a 300 MB host app), no kill; a real 4 GB device remains the check for system-wide pressure | §5.5; `memory_emulation_iphone12pro_2026-11-01/` |
| 21 | The motion-bank body policy on the phone's three caches (`docs/MULTIPOSE_LTX_MUSETALK_IMPLEMENTATION.md` "Runtime algorithm"): idle body under the lipsync for replies under `short_reply_seconds` (10 here), else 0.3 s of idle → bridge into talking at frame 0 → smile cues after a cooldown → return into idle 0.5 s before the end on the continued idle phase; forward-only clips; the bank's optical-flow bridge is the cosine crossfade over the same 0.3 s | **built 2026-11-02** (`Poses.swift` `FramePlan.compile`, `LipsyncSession` idle track + seams, `LipsyncPipeline` blends; `currentFrame()` returns a `DisplayFrame`). A first version replicated the six-pose cue router instead and was replaced the same day at the user's correction. Simulator-validated through the Lingua bench; phone run pending | Lingua build plan §3.12 "design as built" |
| 22 | Toward 25 fps on the A14 (§5.6): SPLIT_EINSUM_V2 attention, the constant time embedding folded out of the UNet, TAESD decoded on the GPU in the compose stage, two Neural Engine workers | **measured on the Mac 2026-11-02: full pipeline 21.0 → 15.5 ms/frame (1.34×), gate unchanged (mae 0.01309 vs 0.01311), composed frames unchanged**; phone run pending (the phone was not connected) | §5.6; `fps25_mac_2026-11-02/`; branch `claude/ios-25fps` |

---

## 2. The pipeline, end to end

### 2.1 What the server does per frame (what is being reproduced)

```
offline (avatar prep, once per character)            live (per 30 s turn)              live (per frame i)
──────────────────────────────────────────           ─────────────────────             ─────────────────────────────────
video frames ─► face box, soft mask, crop box        TTS audio 16 kHz                  cycle index j = i mod 2N  (ping-pong)
             ─► VAE encode masked face + reference    ─► log-mel [80,3000]              latent[j] [8,32,32]  ─┐
             ─► latents[j] [8,32,32]  (2N entries)    ─► Whisper-tiny encoder           window[i] [50,384]   ─┴► UNet ─► [4,32,32]
                                                      ─► 5 hidden states [1500,5,384]                              ─► TAESD ─► face 256×256
                                                      ─► per-frame window [50,384]                                  ─► resize to face box
                                                                                                                    ─► masked blend onto frame[j]
```

The UNet is 849.9M parameters and about 178 GFLOP per frame; everything else is small. The avatar directory a server
worker loads (`full_imgs/`, `mask/`, `coords.pkl`, `mask_coords.pkl`, `latents.pt`) is exactly the input of the
phone bundle exporter, so a character that exists for the servers exists for the phone without re-running face
analysis or the VAE.

### 2.2 What runs where on the device

| Stage | Runs on | Core ML model (bundle file) | Input → output | Size |
|---|---|---|---|---|
| Log-mel spectrogram | CPU (fp32) | `whisper_logmel_fp32.mlmodelc` | waveform [1, 480000] f32 → logmel [1, 80, 3000] f32 | 0.7 MB |
| Whisper-tiny encoder | ANE (fp16) | `whisper_encoder_fp16.mlmodelc` | logmel [1, 80, 3000] f16 → features [1, 1500, 5, 384] f16 | 16 MB |
| Window gather | CPU, Swift | — | features → window [50, 384] per frame | — |
| UNet chunk 1 | ANE | `unet_anec0_int8w_b1_chunk1.mlmodelc` | sample [1, 8, 32, 32] + audio [1, 50, 384] → h [1, 1280, 4, 4] + res_0…res_11 (all f16) | 330 MB |
| UNet chunk 2 | ANE | `unet_anec0_int8w_b1_chunk2.mlmodelc` | h + res_0…res_11 + audio → pred_latents [1, 4, 32, 32] f16 | 486 MB |
| TAESD decoder | ANE | `taesd_fp16_b1.mlmodelc` | latents [1, 4, 32, 32] f16 → faces [1, 3, 256, 256] f16 in [0, 1] | 2.4 MB |
| Resize + blend | CPU, Swift | — | face 256² → face box; fixed-point alpha blend onto the cached frame | — |
| Display | main thread | — | CGImage per `CADisplayLink` tick | — |

Boundary tensors between the chunks: 13 fp16 tensors (`h` [1,1280,4,4]; residuals at 320×32×32 ×3, 320×16×16,
640×16×16 ×2, 640×8×8, 1280×8×8 ×2, 1280×4×4 ×3), 3.4 MB per frame. The sinusoidal positional encoding on the audio
window and the fixed timestep 0 are inside the UNet graph, so the Core ML models take the raw Whisper window.

Compute units are pinned to `cpuAndNeuralEngine` for the UNet chunks and TAESD, `cpuOnly` for the log-mel model.
`.all` was 2× slower on the iPhone right after a compile and lets Core ML move the decoder anywhere (§6.2).

### 2.3 The avatar bundle (premade cache on the phone)

`scripts/ios/prepare_avatar_bundle.py --avatar <server avatar dir> --audio <clip>` writes `HarnessBundle/avatar/`:

| File | Content | Notes |
|---|---|---|
| `avatar.json` | fps, frame size, unique/cycle counts, file lists, face boxes (`coords`), crop boxes (`mask_coords`), audio description, window and blend parameters | decoded by `AvatarMeta` |
| `frames/NNNN.jpg` | the N unique full frames (JPEG q92) | the server cache stores 2N entries: N frames then the same reversed; the exporter checks that symmetry and keeps one copy |
| `masks/NNNN.png` | the N unique soft masks, 8-bit single channel | server mask is the crop-box-sized blend alpha |
| `latents_f16.bin` | [N, 8, 32, 32] fp16 UNet inputs (masked face latent + reference latent) | the VAE never runs on the phone |
| `audio_f32.bin` | the clip at 16 kHz mono float32 (≤ 30 s, one Whisper segment) | input of the log-mel model |
| `audio.m4a` | the same seconds, AAC 128k | the `AVAudioPlayer` clock |
| `windows_ref_f16.bin` | [frames, 50, 384] the repo's own per-frame windows (`AudioProcessor`, torch fp32) | check for the Swift gather |
| `check/NNNN_{composed,face}.png` | a few frames composed in Python with the same Core ML models onto the shipped JPEG | check for the Swift frame loop |

Bundled today: `latina_guided_20260925_talking_84c5bc80b8` (512×832 portrait, 24 fps, 241 unique of 482 cycle
entries, masks 480×480) with 17.76 s of `data/audio/ai-assistant.mpga` (426 frames at 24 fps). 46 MB.

**Phone-only bundles (2026-10-31, Lingua Phase 0).** `prepare_avatar_bundle.py --phone-only` writes just the first
four rows (`avatar.json` without the `audio`/`check` keys; `AvatarMeta.audio` is optional); it needs no audio,
Whisper or Core ML models, so it can run wherever the v15 cache is. `pack_phone_bundle.py <avatar dir> --out
<dest>` packs those four items as a **plain ustar tar** (frames and masks are already compressed; iOS has no zstd)
plus a sidecar `<avatar_id>.json` (`avatar_id, bytes, sha256, frames, cycle_length, fps, width, height, format,
exporter_commit, created_at`), with fixed metadata so the same input gives the same sha256. The phone downloads
the tar (`RemoteBundle.fetchAvatarBundle`: SHA-256 check, `TarArchive.extract` streaming in 1 MB chunks) into
`Application Support/avatars/<avatar_id>/`, which `AvatarBundle(dir:)` then loads unchanged. Published:
`s3://lingua-musetalk-s3-storage/avatars/ios/v1/latina_guided_20260925_talking_84c5bc80b8.{tar,json}` (27.3 MB,
241 frames). The Lingua backend's export job (its Phase 0 session) runs the same two scripts from the v15 tarball.

Cycle mapping (`AvatarBundle.uniqueIndex`): `j = i mod cycle_length`; if the cache is ping-pong and `j ≥ N`, the
unique index is `cycle_length − 1 − j`. The server's `realtime_inference.py` indexes the 2N-entry cache the same
way, so the pose motion is identical.

### 2.4 Per-turn audio: log-mel → encoder → windows

`computeAudioWindows` in `HarnessCore/Lipsync.swift` mirrors `musetalk/utils/audio_processor.py`
(`encode_whisper_feature` + `build_audio_prompts`):

0. Any audio file AVFoundation reads (the OmniVoice and Kokoro 24 kHz WAVs, m4a, mp3) becomes mono float32 at
   16 kHz through `loadWaveform16k` (`HarnessCore/AudioInput.swift`, `AVAudioConverter`), the resample librosa does
   on the server. The bundled harness clip ships pre-resampled (`audio_f32.bin`).
1. Cut the waveform into 30 s segments (480,000 samples, the last one zero-padded); for each, run the log-mel model
   (CPU, fp32), convert to fp16, run the encoder (ANE). The encoder output is the five hidden states (embeddings +
   four layers, the last after the final layer norm) stacked as [1500, 5, 384] at 50 Hz; the segments' first
   `floor(segment samples / 16000 × 50)` steps are concatenated into one feature timeline, as the repo's
   `AudioProcessor` does, so a window at a segment boundary sees both sides. A TTS reply is one segment.
2. `actual = floor(samples / 16000 × 50)` feature steps carry audio; `frames = floor(samples / 16000 × fps)`.
3. The feature timeline is padded with 4 zero steps on the left and 12 on the right. For frame `i`,
   `a = (i × 50) / fps` (integer division) and the window takes padded steps `a … a+9` (10 steps = 200 ms, 4 steps
   before the frame's own two steps and 4 after); steps outside `[0, actual)` stay zero.
4. Layout of the 50 rows: row `s × 5 + l` is step `s`, layer `l`; each row is 384 wide.

Mac check: the Swift windows differ from the repo's torch fp32 windows by max 9.19 / mean 0.004, which is the ANE
encoder's fp16 error on Whisper's outlier channels (|values| up to ~250), not a gather error; through the UNet this
moves the composed face by 0.015/255 (bench doc §6.2).

### 2.5 Per-frame: UNet → decoder → blend

`LipsyncEngine.renderFrame(i)`:

1. `u = uniqueIndex(i)`. The sample `MLMultiArray` for every unique frame is prebuilt at load (241 arrays of
   8×32×32 fp16); the audio window is wrapped as [1, 50, 384] fp16.
2. `StagedUNet.predict`: chunk 1 takes `sample` and `audio`; chunk 2 takes `h`, `res_0…res_11` and `audio`, matched
   **by name** from the inputs and chunk 1's outputs (the same code runs a single-program UNet when the bundle has one).
3. TAESD decodes `pred_latents` → `faces` [1, 3, 256, 256] in [0, 1]; `faceRGB` converts to 8-bit RGB with
   round-half-even (`vae.py`'s convention without the BGR flip).
4. `compose(frame:face256:plan:)`: the face is resized to the face box with a bilinear resize written to match
   `cv2.INTER_LINEAR`, then blended into the crop-box ROI with the soft mask in **fixed point**:
   `out = (face × a + base × (255 − a)) / 255` per channel, integer arithmetic. The `BlendPlan` (computed once per
   unique frame and cached) ports `get_image_blending`'s geometry: the ROI is the crop box clipped to the frame, the
   mask is shrunk to its non-zero bounding box, and pixels outside the pasted face rectangle are the identity.
5. Returned as an RGBA `RGBAImage` plus `FrameTiming` (unet / decode / compose ms). Decoded JPEG frames live in a
   24-entry LRU; blend plans are cached per unique frame.

Mac check (`bench-mac --lipsync-check HarnessBundle/avatar`): frames 0, 24, 48, 96 composed in Swift vs the Python
check frames differ by 0.17/255 mean over the whole frame; the loop costs 27 ms/frame on the M5 Pro (UNet ~19,
decoder ~2, compose ~6 in unoptimised Swift).

### 2.6 Pacing: audio clock, render-ahead, frame skipping (`App/Sources/LipsyncPlayer.swift`)

- **Load** on a detached task: `LipsyncEngine.load()` (both chunks + TAESD, first launch compiles) then
  `prepareAudio()` (log-mel, encoder, windows for the whole clip). Progress lines print as `[lipsync] …`.
- **Two stages on two threads** (`LipsyncPipeline` in `HarnessCore/Lipsync.swift`): the Neural Engine stage
  (`inferFace`: UNet chunks + TAESD → 256 px face) and the CPU stage (`composeFrame`: JPEG decode, resize, blend;
  then the CGImage in the app's emit closure), joined by a queue three frames deep. A frame costs the slower stage,
  not the sum: on the Mac 23.6 ms sequential became 20.8 ms wall, equal to the ANE stage alone.
- **Pre-roll**: the pipeline renders frames 0, 1, 2 … into a `FrameBuffer` (lock-protected dictionary); when
  `min(total, fps)` frames (1 s) are buffered, the `AVAudioPlayer` starts and the start time becomes the clock.
- **Render-ahead and skipping**: the pipeline keeps at most `fps` frames (≈1 s) finished or in flight. When the
  clock has passed frame `i` before the UNet ran it, the Neural Engine stage jumps to `clock + lead` and counts the
  gap as **skipped** (the server does the same when it drops), where `lead` is the pipeline's own latency in
  frames plus one (a moving average of UNet + TAESD + compose over the frame duration, `Stats.leadFrames`; 1 until
  the first frame is timed). Aiming at `clock + 1` was the first phone run's mistake: a 64 ms inference lands
  1.5 frames after the clock position it was aimed at, so 130 of 298 generated faces arrived late and were
  dropped. A face that is still late by the time it reaches the CPU stage is **dropped late** instead of blended.
- **Display**: a `CADisplayLink` (30–60 Hz) takes the newest buffered frame with index ≤ `floor(t × fps)`, drops
  older ones, and counts a **stall** when nothing new is available for a frame the clock has reached.
- **Finish**: when the clip is done (or 2 s past its end) the player prints one line
  `LIPSYNC_RESULT {"avatar", "fps", "frames", "seconds", "generated", "generated_fps", "gen_ms_per_frame",
  "stage_infer_ms", "stage_compose_ms", "skipped", "dropped_late", "lead_frames", "pipelined", "shown", "stalls",
  "load_s", "audio_prep_s", "audio_played", "audio_route", "system_volume", "unet"}`; with `--exit` the app quits
  (exit 0, or 1 on a load error). The audio session is `.playback` (it ignores the silent switch); the player's
  own volume is 1, so what you hear is the phone's volume on the route the console line `[lipsync] audio: …`
  names (Speaker, or whatever Bluetooth device is connected), also shown in the status line under the picture.
- Launch flags: `--lipsync` opens the Lipsync tab and autoplays; `--exit` as above. Without flags: Play / Stop
  buttons in the tab.

Measured on the A14 (2026-10-31, `lipsync_iphone12pro_pipelined_run2.json`): the Neural Engine stage is 65.7 ms
(UNet ~60 + TAESD ~5, phone warm after the compile), the CPU stage 10.4 ms in parallel, so the frame costs the ANE
stage alone: **291 faces generated and 290 shown over the 17.7 s clip = 16.4 fps, 0 stalls, 0 late drops, 135
frames skipped**, lead settled at 3 frames (125 ms behind the clock position it aims at, inside the 1 s
pre-roll). With the compile cache warm and no compile before playback (run 3) the same clip gives 305 generated,
303 shown = 17.1 fps, ANE stage 62.5 ms, CPU stage 9.1 ms. The audio never waits for the video.

---

## 3. Model export: decisions and why

All export scripts live in `scripts/ios/` and run in `.venv-ct` (Python 3.12, torch 2.7.0, torchvision 0.22.0,
coremltools 9.0, diffusers 0.42.1). Converted models go to `models/ios_bench/coreml/<name>.mlpackage` and are listed
in `models/ios_bench/coreml/manifest.json` (not in git) with their kind, variant, batch, bytes, and for chunks
`chunk_of`, `chunk_index`, `cut`, `inputs`, `outputs`.

### 3.1 The UNet (`export_coreml.py`, then `export_coreml_ane.py`)

- **Tracing**: `torch.jit.trace` fails in coremltools 9 on 592 `aten::Int` shape casts (the `height×width` reshapes
  and `batch // heads` in attention). `torch.export` + `run_decompositions({})` specialises them; this is the
  exporter for every UNet variant.
- **Graph boundary**: the Core ML model is the whole per-frame step: `sample` [B, 8, 32, 32] (masked face latent +
  reference latent) and `audio` [B, 50, 384] (raw Whisper window) → `pred_latents` [B, 4, 32, 32]; the positional
  encoding and timestep 0 are inside. One model per batch size (the ANE wants static shapes).
- **Variants**: `fp16` (1,622 MB); `int8w` = int8 per-channel weights on linear/conv, fp16 activations (812 MB,
  `coremltools.optimize.coreml.linear_quantize_weights`); `int8wa` = int8 weights and per-tensor activations
  calibrated on 16 real frames (813 MB). int8wa is faster (19.0 ms on the Mac ANE) but fails quality (latent max_abs
  1.68, worst pixel 138/255) and is not used.
- **Apple layout** (`export_coreml_ane.py`, needs a clone of `apple/ml-stable-diffusion`, imported not installed):
  the compute plan showed 16 % of the diffusers-layout model's cost in reshape/transpose/split, the signature of a
  [B, S, C] transformer on hardware that wants [B, C, 1, S]. Apple's `python_coreml_stable_diffusion.unet` is the
  same UNet rewritten with Conv2d everywhere, `LayerNormANE` and split-einsum attention; it loads MuseTalk's weights
  through its state-dict hooks (0 missing / 0 unexpected keys) and reproduces the diffusers UNet to 1.6e-5 in fp32.
  Result: 22.6 ms fp16 / **18.9 ms int8w** at batch 1 on the M5 Pro ANE (−17–20 %), identical numerics, and batch 4
  now compiles for the ANE (the diffusers layout fell to the CPU at batch > 1). All phone models are this layout
  (`unet_ane_*`, `unet_anec0_*`).

### 3.2 Chunking for iOS (`chunk_coreml.py` rejected, `export_coreml_ane_chunks.py` used)

Why: on the iPhone the ANE program is compiled inside the app's process. The int8 weights expand to fp16 during that
compile (the A14 ANE has no int8 weight path): the footprint reached **2,464 MB against a 3,070 MB allowance**, and
the compiled program (~1.6 GB) must be written to the app's cache (the phone had 1,040 MB free). Every launch after
the first was SIGKILLed with no crash report. Apple's ml-stable-diffusion README states the rule: chunking "is
required for Neural Engine deployment on iOS and iPadOS if weights are not quantized to 6-bits or less".

- `coremltools.models.utils.bisect_model` (Apple's own chunk script; `chunk_coreml.py`) cuts the ML Program where
  half the weight bytes are consumed (420 + 393 MB) and is bit-identical to the whole model on the Mac ANE. But it
  declares the 13 boundary tensors **fp32**, and the A14 compiler then schedules both chunks on the CPU (870 ms/frame
  under CPU_AND_NE) or GPU (388 ms under ALL). Rejected.
- `export_coreml_ane_chunks.py --cut 0` cuts the network in **PyTorch at the mid/up boundary** and exports each half
  with fp16 inputs and outputs, so each chunk looks to the compiler exactly like the whole model did. Chunk 1 =
  time embedding, `conv_in`, down blocks, mid block (344M params, 330 MB int8). Chunk 2 = up blocks,
  `conv_norm_out`/act/`conv_out` (506M, 486 MB); it recomputes the constant time embedding and the audio hidden
  states itself (2M params) so it needs only `h`, the residuals and `audio`. Exact against the diffusers UNet in fp32
  (1.25e-5), bit-identical to the single program on the Mac ANE, and on the A14 **all 3,056 ops stay on the Neural
  Engine** with the process footprint at 53–64 MB after both compiles. `--cut k` moves the first `k` up blocks into
  chunk 1 if a different balance is ever wanted. `--batch 4` exports the same cut at four frames per call
  (`unet_anec0_int8w_b4_chunk1/2`, same bytes, bit-identical output on the Mac ANE): the bandwidth experiment of
  §5.4. At batch > 1 the Apple layout's permutes leave non-contiguous boundary tensors that coremltools rejects as
  chunk-2 inputs, so chunk 1 returns them contiguous.

### 3.3 TAESD decoder

`madebyollin/taesd` decoder, the server's production decoder, exported fp16 with the server's `(x/2 + 0.5).clamp`
baked in: [B, 4, 32, 32] → [B, 3, 256, 256] in [0, 1], 2.4 MB, 1.5 ms on the Mac ANE, 5.1 ms on the A14 ANE
(26.4 ms if it lands on the A14 CPU, hence the pinning).

### 3.4 Whisper-tiny front end (`export_whisper_coreml.py`)

- `whisper_logmel_fp32`: `WhisperFeatureExtractor`'s torch path rewritten as a conv1d STFT so it converts without the
  complex-number dialect; fp32, meant for the CPU; matches the HF extractor to 2e-5; 1.2 ms per 30 s on the Mac,
  3.7 ms on the A14.
- `whisper_encoder_fp16`: the 4-layer encoder with its five hidden states stacked as the server's
  `encode_whisper_feature` does (equal to `transformers`' `output_hidden_states` to 0.0); `torch.jit.trace` works
  here; 100 % ANE; 12.2 ms per 30 s on the Mac, 28.1 ms on the A14. The ANE's fp16 accumulate makes the feature
  error larger than the GPU's (max 9.5 vs 2.2 on outlier channels) but through the UNet it is 0.015/255.
- The per-frame 50×384 gather is indexing, not a model (§2.4).

### 3.5 Levers measured and rejected

| Lever | Result | Where |
|---|---|---|
| int8 activations (per-tensor, calibrated) | gate max_abs 1.68, worst pixel 138/255 | bench doc §4 |
| 6-bit palettization (Apple layout fp16) | 18.6 ms (vs 18.9 int8w), gate mae 0.0186, visible 0.7/255 | §11.3 |
| 4-bit palettization | 17.9 ms, gate mae 0.0565, 2.3/255 | §11.3 |
| fp32 islands for GroupNorm / softmax / attention matmuls | no change in gate mae (0.0126–0.0128) | §11.2 |
| fp32 for the 1,280-channel middle or long accumulations | Core ML put the whole model on the CPU; not a hybrid | §11.2 |
| GroupNorm as chained short reductions on the ANE | no change | §11.2–11.3 |
| batch 4 on the ANE | compiles with the Apple layout; slower per frame on the Mac (compute-bound); on the A14 54.2 vs 56.4 ms/frame (−3.8 %), so compute-bound there too, at four frames of latency | §11.3, §12.5 |
| `MLComputeUnits.all` on the phone | 130 ms transient after a compile; moves TAESD to the CPU | §12.3 |
| `bisect_model` chunks | fp32 boundaries → CPU/GPU on the A14 | §12.3 |
| Deep-feature reuse across adjacent frames (DeepCache-style: on alternate frames run only the 32×32 level, or the 32×32 and 16×16 levels, on the previous frame's deeper features) | latent mae 0.045 / 0.027 and max_abs 4.1 / 4.3 against the server reference (the full UNet: 0.0004 in fp32, 0.0131 on the ANE): the avatar's head moves between frames, so the cached features are misaligned. Fails the self-test limits (0.02 / 1.2) | §5.6; `scripts/ios/check_deep_reuse.py` |
| SPLIT_EINSUM_V2 query chunk 128 or 256 instead of 512 | 16.69 / 16.4 vs 16.66 ms on the Mac: within noise | §5.6 |
| `MLOptimizationHints.specializationStrategy = .fastPrediction` | 16.67 / 16.49 vs 16.63 / 16.56 ms on the Mac: no effect there; kept as an option (`ModelTuning.fastPrediction`) for the phone | §5.6 |

---

## 4. Quality: what the Neural Engine does to the UNet, and how it was judged

### 4.1 The repo gate (`scripts/ios/validate_unet_coreml_gate.py`)

The server's TensorRT engines are gated by `scripts/validate_unet_backend.py` on production captures
(`calibration/unet_portable_bs8/`: 16 files, 128 frames, inputs as the server's UNet saw them, reference = the fp16
PyTorch outputs), per file mae ≤ 0.01 and max_abs ≤ 0.5 on the predicted latents. The same statistics on the same
captures:

| UNet | Runs on | mae mean / max | max_abs max | Gate |
|---|---|---|---|---|
| PyTorch fp32 (anchor) | Mac CPU | 0.00038 / 0.00048 | 0.047 | pass |
| server TensorRT fp16 | 4070S | 0.0018 / 0.0025 | 0.414 | pass |
| server TensorRT INT8, shipping recipe (r5) | 4070S | 0.0032 / 0.0044 | 0.777 | mae pass |
| Core ML fp16 | Mac GPU | 0.00041 / 0.00053 | 0.055 | pass (= fp32) |
| Core ML int8w | Mac GPU | 0.0032 | 0.23 | = shipping INT8 |
| Core ML fp16 | Mac CPU | 0.0064 / 0.0080 | 0.470 | pass |
| Core ML fp16 / int8w, any layout | **Neural Engine (Mac)** | 0.0128 / 0.0184, 0.0131 / 0.0187 | 1.00–1.03 | **fail, every file** |
| Core ML int8w chunked | **Neural Engine (iPhone 12 Pro)** | 0.01317 / 0.01876 | 1.03 | fail, identical to the Mac |

Reading: the int8 weights cost exactly what the shipping INT8 engine costs (0.0032, measured on the GPU); the whole
remaining gap is the ANE's conv/linear fp16 multiply-accumulate (the GPU accumulates in fp32). It is not in
normalisation or attention (fp32 islands change nothing), it travels with the Neural Engine and not the device
generation, and no cheap fix exists (§3.5). In pixels the gap is 0.16–0.30/255 mean on the composed face crop
(46–55 dB) and ~0.57/255 on the raw 256 px decoder output; the production decoder's own TAESD-vs-SD-VAE difference
is 0.74–1.03/255.

### 4.2 The r5-style labelled review (`scripts/ios/review_ane_vs_fp16.py`, `docs/ios_unet_bench/review_ane_vs_fp16/`)

Whether 0.013 is visible is a human verdict, so the package the server's INT8 recipe (r5) was accepted on was rebuilt
for the phone chain (bench doc §12.4, package `README.md`):

- **Arms**: A = PyTorch fp32 UNet + torch TAESD (reference); N = PyTorch fp16 on the Mac GPU (the server's own
  precision: the noise the product already accepts); B = the phone chain (two-chunk int8 ANE UNet + Core ML TAESD,
  bit-identical to the iPhone on the gate frames); Btd = B's latents through the torch decoder (metrics only).
- **Set**: six production characters (`en_01_maya`, `en_03_marcus`, `es_01_valeria`, `es_03_mateo`,
  `japanese_realtime`, `latina_guided`) × two audios (`ai-assistant.mpga` TTS, `eng.wav` recording), 10 s at 25 fps:
  12 clips, 3,000 frames per arm. Per clip: `A.mp4`, `B.mp4`, `N.mp4`, a labelled `_ab.mp4` (A | B, 3× mouth zoom,
  |A−B|×8 face panel, metrics line) and a blind `_blind.mp4` (sides randomised, key in `key/blind_key.json`), raw faces,
  latents and FaceMesh landmarks (`experiments/video_validation/ane_vs_fp16/`, gitignored, 1.3 GB).
- **Metrics** on raw composed frames: latent mae; raw-face mae and PSNR; face-box and mouth-ROI PSNR/SSIM; flicker
  and sharpness ratios; FaceMesh inner-lip aperture (mediapipe 0.10.14 in `.venv-mp`): Pearson corr, lag, mean |Δ|.
  r5 thresholds: corr ≥ 0.97, lag 0, |Δ| ≤ 0.5 px, flicker ≤ 1.05, sharpness ≥ 0.95.
- **Result before the human pass**: five characters pass all six thresholds on all ten clips (lip corr 0.9952–0.9992,
  |Δ| 0.19–0.28 px, mouth PSNR 45.5–48.8 dB, mouth SSIM 0.996–0.997): r4's accepted band, short of r5's. **The
  Japanese character fails the two aperture thresholds on both audios** (corr 0.956–0.963, |Δ| 0.78–0.90 px, mouth
  PSNR 39.5–40.5 dB); its gate mae was also the highest (0.016–0.017), and Btd shows it is the UNet, not the decoder.
  The noise arm N is at 0.0003 latent mae, 0.010–0.015/255, corr ≥ 0.9999 on every clip: whatever a reviewer sees in
  B is the ANE.
- **Acceptance rule** (r5's, made countable): after the blind pass, B is accepted when no clip has severity ≥ 2
  attributed to B, every identity has a clip where B was not called worse, and B is called worse at most three
  times more often than A. One failing identity → accepted with a reservation for that character.
- **Open**: the blind pass itself (`review_sheet.csv`, then unblind with the key). Nothing in the package is a verdict.

---

## 5. Speed: measured

### 5.1 Mac, M5 Pro Neural Engine (same ANE generation as the iPhone 17 Pro)

| Measurement | Value |
|---|---|
| UNet diffusers layout fp16 / int8w, batch 1 | 27.1 / 23.5 ms |
| UNet Apple layout fp16 / int8w, batch 1 | 22.6 / **18.9** ms (Swift harness: 18.7, p99 19.0) |
| UNet Apple layout fp16 / int8w, batch 4 (per frame) | 25.1 / 20.1 ms (compute-bound here, and on the A14 too: §5.2) |
| TAESD | 1.5 ms |
| UNet + TAESD chain | 23.7–25.3 ms; 750 frames p99 26.1, max 28.5, none over 40 ms |
| Whisper encoder / log-mel per 30 s | 12.2 ms ANE (3.1 GPU) / 1.2 ms CPU |
| First load of the 812 MB program / again in-process | 6.7 s / 0.05 s |
| 12 back-to-back 10 s turns, int8w | 1.57× real time every turn, no drift |
| Batch-4 two-chunk int8w, per frame | Swift harness 20.6 ms (batch-1 chunks: 18.8), 100 % ANE; Python incl. I/O 22.8 (unchunked batch 1: 20.3); output bit-identical to batch 1 |
| Lipsync view frame loop | 23.6 ms sequential (UNet 18.9 + TAESD 1.7 + compose 3.0); **20.8 ms wall pipelined** (ANE stage 20.7, CPU stage 1.4 + CGImage) |

### 5.2 iPhone 12 Pro (A14, 6 GB, iOS 26.2; `harness_iphone12pro_*.json`)

| Measurement | Value |
|---|---|
| Single 850M program (cold launch only; later launches killed) | 51.8–54.3 ms/frame, 100 % ANE |
| **Two-chunk int8w UNet, CPU_AND_NE, 200 frames** | **59.6 ms** (p50 59.6, p99 61.8, max 62.1), 100 % ANE, footprint 64 MB |
| Same under ALL right after the compile | 130.5 ms (transient; next ALL load 54) |
| Sustained 30 s (509 frames) | 58.9 mean; 5 s buckets 54.5 → 57.2 → 59.0 → 60.2 → 61.2 → 62.4; thermal "serious" |
| Chain UNet → TAESD | 59.7 = 54.6 + 5.1 ms (TAESD on the CPU instead: 81.5 = 55.1 + 26.4) |
| Gate, 128 production frames | mae 0.01317 / max 0.01876, max_abs 1.03 (= Mac ANE) |
| Whisper encoder / log-mel per 30 s | 28.1 ms ANE / 3.7 ms CPU |
| First prediction after a compile | 65–78 ms |
| Compile (first launch) | 12–32 s per chunk, per compute-unit setting, ~1.7 GB cache writes |
| Warm relaunch (compiled chunks cached) | load 0.59 s (0.17 + 0.42), first prediction 61.5, then 54.7 ms (p99 56.3) |
| **Batch-4 two-chunk UNet, 50 calls of 4, phone at "fair"** (2026-10-31; `harness_iphone12pro_b4_vs_b1.json`) | **54.2 ms/frame** (p50 54.4, p99 55.7, max 55.8) = 217 ms per call; batch 1 in the same session 56.4 (p50 56.3, p99 58.1); gate identical (mae 0.01369); 100 % ANE |
| Batch-4 compile / first call | 46.4 s (19.7 + 26.7) / 222 ms |
| Sustained 30 s from a warm start ("fair" after 70 s of compiles) | 67.1 mean (p99 83.6, 448 frames); buckets 54.1 → 55.6 → 69.7 → 78.4 → 78.5 → 76.4 |
| **Lipsync view, Latina clip, 426 frames at 24 fps** (2026-10-31; `lipsync_iphone12pro_pipelined_run2.json`) | **291 faces generated and 290 shown in 17.7 s = 16.4 fps, 0 stalls, 0 late drops**, 135 frames skipped by the clock; ANE stage 65.7 ms (UNet + TAESD, phone warm), CPU stage 10.4 ms in parallel; 25 frames pre-rolled; load 27.7 s (compile after a reinstall), audio prep 4.4 s (Whisper compile included) |
| Same, warm compile cache (run 3: load 0.57 s, audio prep 0.17 s, no compile before playback) | **305 generated, 303 shown = 17.1 fps**, 0 stalls, 0 late drops, 121 skipped; ANE stage 62.5 ms, CPU stage 9.1 ms |
| Same, audio-diagnostics build after a reinstall (run 4) | 296 generated, 295 shown = 16.6 fps, 0 stalls; audio: route Speaker, `play()` true, system volume 0.25 (quiet, the phone's own setting) |
| Same, with the skip aimed at the clock itself (run 1) | 298 generated but only 167 shown (9.4 fps): 130 faces arrived 1–2 frames late and were dropped |
| **Phone bundle from S3 + OmniVoice WAV, warm** (2026-10-31; `lipsync_iphone12pro_s3_bundle_run2_warm.json`) | bundle already on the device, models loaded 0.96 s, 0.4 MB WAV downloaded 0.6 s, windows 0.16 s; **clock start 0.79 s after `speak()`** (= windows + 10-frame pre-roll at 62 ms); 157 generated, 156 shown over the 9.1 s clip = **17.2 fps**, 0 stalls, 0 late; ANE 61.6 ms, CPU 9.2 ms; audio via Speaker |
| Same, cold (first launch after a reinstall; `…_run1_cold.json`) | bundle 27.3 MB downloaded in 5.4 s, unpacked 1.1 s; UNet compile 25.7 s; Whisper compile inside the 3.4 s window build → clock start 4.0 s after `speak()`; 155 generated, 154 shown = 16.9 fps, 0 stalls |

Reading: 52–56 ms cool, 59–62 ms hot and up to 78 ms when the phone starts warm, plus 5 ms decoder, so 16–17 fps
sustained from cool and 13–15 fps warm, against the server's 20 fps (50 ms budget). **Batch 4 moved the frame by
2 ms (−3.8 %), so the A14 Neural Engine is compute-bound, not starved for weight bandwidth** (§5.4). Projections by
the same arithmetic, unmeasured: iPhone 14 Pro ~35 ms, 15/16 Pro 30–40 ms (real time at 20 fps), 17 Pro 25–30 ms.
Phones from the A17 Pro on also keep int8 weights as int8 on the ANE, which removes the compile peak that made
chunking mandatory here.

### 5.3 Operating rules learned on the phone

1. Ship the UNet as two chunks with fp16 boundaries; never a single program.
2. Pin `cpuAndNeuralEngine` for the UNet and TAESD; `cpuOnly` for the log-mel.
3. Load once per session; never hold two UNet instances; let the compile finish before anything else.
4. Expect 30–60 s of compile on first launch per compute-unit setting (the cache is keyed per setting) and ~1.7 GB
   of cache writes; check free disk first (the kills were a full disk plus the in-process fp16 expansion).
5. `devicectl` can only launch while the phone is unlocked; the harness retries on "Locked".
6. A reinstall empties the Core ML compile cache: the first launch after every install compiles again (27 s for
   the batch-1 chunks, 46 s for the batch-4 pair, ~4 s for the Whisper models). In product terms every app update
   costs one compile; do not measure playback in the same minute (the chip is at "fair" afterwards).
7. Never call a lock-taking method from inside a closure that already holds the same `NSLock`: the first phone run
   of the Lipsync view hung on exactly that (the frame buffer's `count` read inside its own `with`), with the
   title on screen and no frames; the Mac path never exercised it. `FrameBuffer` now takes its lock once per method.

### 5.4 Where the bottleneck is (2026-10-31; bench doc §12.5)

Asked as "is it RAM, CPU, or the accelerator", the phone's situation is three separate answers.

- **RAM: was the problem, solved.** The kills were the in-process fp16 expansion of the int8 weights during the
  single program's compile (2,464 MB of a 3,070 MB allowance) plus a full disk. Chunked, the process sits at
  53–64 MB after compile; the lipsync view adds ~80 MB (frame cache 41, audio windows 33, latents 4). Not a limit.
- **CPU: was 15 % of the frame, now off the critical path.** The CPU stage (JPEG decode, resize, blend, CGImage) ran
  in series after the two Neural Engine calls on one thread while the CPU idled during them. The two-stage pipeline
  (§2.6) overlaps it: Mac 23.6 → 20.8 ms wall, equal to the ANE stage. On the A14 the CPU stage is ~8–10 ms, hidden
  the same way as long as it stays under the ~60 ms ANE stage.
- **Neural Engine: the bottleneck, and it is compute-bound.** 52–56 ms cool / 59–62 ms hot per UNet frame. Two
  hypotheses fit the batch-1 numbers. Bandwidth: the compiled program on the phone is ~1.6 GB for the 813 MB int8
  model, so the A14 streams fp16 weights every frame; 1.6 GB at ~34 GB/s is a 48 ms floor, within 10 % of the cool
  measurement. Compute: the M5 Pro ANE is compute-bound at 18.9 ms and the A14 ANE is roughly a third per op,
  ~57 ms, also matching. **Batch 4 separated them** (weights read once per four frames, 12 ms floor, compute
  unchanged): measured 2026-10-31 in one session, **54.2 ms/frame at batch 4 against 56.4 at batch 1, −3.8 %**, with
  the phone a thermal step warmer for the batch-4 run and the gate identical. A bandwidth-bound frame would have
  lost most of its 48 ms; it lost 2. Weight streaming is a few milliseconds of the frame and the rest is
  arithmetic. Batch 4 stays available as a 2 ms lever that costs four frames of latency (167 ms at 24 fps) and a
  222 ms first call; not worth taking alone.
- **Compute-bound means no cheap ANE lever remains** (int8 activations fail quality, palettization saves bytes not
  operations, the FLOPs are fixed): the A14 is below the 20 fps line and the 15 Pro and later are the answer.
  Levers worth a few ms each, not yet taken: output backings for the 13 boundary tensors (3.4 MB allocated per
  frame), TAESD fused into chunk 2, the blend in Metal.
- **Thermal state is the larger variable.** From a cool start the sustained run lost 13 % over 30 s (54.5 → 62.4 ms);
  from a warm start (after 70 s of compiles) it lost 45 % within 15 s (54.1 → 78.4 ms, 67.1 mean). The product
  figure for this phone is the warm one, 13–15 fps, and keeping the chip cool (no compile in the same minute as
  playback, shorter turns) is worth more than the remaining levers.

### 5.5 Memory: a 4 GB phone emulated on the 12 Pro (2026-11-01)

The question: do the 4 GB iPhones after the 12 Pro (12, 12 mini, 13, 13 mini, SE 2022; about 2,098 MB of
foreground allowance, the "ActiveHard 2098 MB" of their jetsam reports) survive the cold compile that killed the
single program here (2,464 MB of this phone's 3,071 MB)? The harness app can now emulate that allowance:
`--emulate-limit-mb N` holds (this process's limit − N) MB of dirty, incompressible memory before the models load
(`App/Sources/MemoryBallast.swift`; 974 MB here, limit measured as footprint 11 + headroom 3,061 MB at launch),
`--ballast-mb N` adds a host app's own footprint, and a 250 ms footprint trace prints on the console so a kill
leaves its last footprint behind. Lipsync tab, bundled Latina avatar, the 17.8 s clip, `--silent`; a reinstall
before each cold run empties the Core ML compile cache. Logs, `summary.json` and the parser:
`docs/ios_unet_bench/memory_emulation_iphone12pro_2026-11-01/`.

| Run | Allowance | Compile | Ballast | Peak footprint | Engine's share | Least headroom | Pressure | ANE stage | Shown |
|---|---|---|---|---|---|---|---|---|---|
| A | none (3,072 MB) | cold, 26.8 s | 0 | 148 MB | 148 MB | 2,924 MB | none | 65.1 ms | 280/426, 15.7 fps, 0 stalls |
| B | 2,098 MB | warm, 0.7 s | 974 MB | 1,145 MB | 171 MB | 1,927 MB | none | 64.8 ms | 281, 15.9 fps, 0 stalls |
| C | 2,098 MB | cold, 26.4 s | 974 MB | 1,161 MB | 187 MB | 1,911 MB | none | 94.5 ms | 199, 11.2 fps, 0 stalls |
| D | 2,098 MB + 300 MB host app | cold, 26.9 s | 1,274 MB | 1,460 MB | 186 MB | 1,612 MB | 1 warning at 31.6 s (playback) | 99.6 ms | 183, 10.3 fps, 0 stalls |
| C2 (C repeated after a 4 min cool-down, thermal nominal before and after) | 2,098 MB | cold, 25.7 s | 974 MB | 1,172 MB | 198 MB | 1,900 MB | none | 68.4 ms | 266, 15.0 fps, 0 stalls |

- **The chunked compile does not inflate the process.** During the 27 s cold compile the footprint peaked at
  54 MB (the single program's compile reached 2,464 MB); the uncapped run's 148 MB peak is playback (frame cache,
  audio windows, the displayed images), ~170–190 MB of engine under the caps. Under
  the 4 GB allowance the process kept ≥ 1.9 GB of headroom, and ≥ 1.6 GB with a 300 MB host app on top: no kill,
  no critical pressure. **The per-process limit of the 4 GB phones is not a risk for the chunked models**; the
  Lingua gate's 3.5 GB memory floor can stay.
- **Speed under the cap.** B equals A, so the ballast itself costs nothing. C and D were slower (94–100 ms per face
  against 65) with the four runs back to back, four compiles in three minutes. C2, the same capped cold run after a
  four-minute cool-down, came back at 68.4 ms and 15.0 fps with the thermal state nominal at both ends: the slow
  runs were heat, the cap costs nothing. (§5.3 rule 6 again: no compile in the same minute as a measurement.)
- **What the emulation cannot show.** It caps this process; a real 4 GB phone also has 2 GB less for everything
  else (the ANE compiler service's own working set during the compile, the compressor, other apps). D's one
  warning at a 1,430 MB footprint is this phone's system-wide pressure with 1.3 GB of ballast resident, not a
  number a 4 GB phone reproduces. The remaining check for that tier is one real device (an iPhone 13 or SE 2022)
  through the Lingua app's cold compile and a call.

---

### 5.6 Toward 25 fps: four levers that stack (2026-11-02, Mac M5 Pro; phone pending)

Goal: a 30 % faster frame than the 2026-11-01 baseline, on the way to 25 fps on the A14 (40 ms per frame). All
numbers below are the M5 Pro's Neural Engine, measured with baseline and candidate alternating in one session
(another user's Core ML test was running on the same Mac part of the time). The phone was not connected, so the
A14 effect of each lever is still to measure; `run_phone_25fps.sh` in the session scratchpad does it on one
install (UNet A/B, then the Lipsync tab A/B).

| Lever | What it is | Mac effect |
|---|---|---|
| **SPLIT_EINSUM_V2** attention | ml-stable-diffusion's ANE attention that chunks the query sequence (512); here it applies to the 1,024-token 32×32 level | UNet 19.5 → 16.3–16.7 ms (−16 %), gate identical (same numerics) |
| **Fold the time embedding** (`--fold-time-embedding`) | MuseTalk runs t = 0 only, but coremltools kept the embedding MLP and all 22 resnets' `time_emb_proj` 1×1 convs (12–14 per chunk on constant inputs). Replaced by their values: 25.8M parameters fewer (−29 MB int8; the A14, which expands int8 to fp16, reads ~52 MB fewer weight bytes per frame) | UNet 16.63 → 16.43 ms; gate mae 0.01309 / max_abs 1.030 (was 0.01311 / 1.035). Expected to matter more on the A14 |
| **TAESD in the compose stage, on the GPU** (`LipsyncEngine.taesdUnits = .cpuAndGPU`, `deferDecode`) | the Neural Engine stage runs the UNet alone; the compose thread decodes (4–6 ms on the GPU) and blends while the next UNet runs | ANE stage −1.8 ms (TAESD) − the RGB conversion; composed frames unchanged (0.176/255 vs 0.175/255 against Python) |
| **Two Neural Engine workers** (`LipsyncPipeline.inferWorkers = 2`) | the next frame's chunk 1 is queued on the engine while the previous frame's outputs are converted and its chunk 2 submitted; frames still leave the compose stage in order | 16.2–16.3 → 15.5–15.6 ms/frame wall |

Full pipeline over 400 frames (`bench-mac --lipsync-check`, UNet + TAESD + blend + CGImage, free-running):

| Configuration | ms/frame wall (rounds) | vs shipped |
|---|---|---|
| Shipped (SPLIT_EINSUM chunks, TAESD on the ANE stage) | 21.0, 21.2, 21.0, 21.3, 21.8, 21.5, 21.4, 21.4, 21.0, 20.8 | 1.00× |
| V2 | 18.2, 18.6 | 1.15× |
| V2 + TAESD on the GPU in the compose stage | 16.5, 17.0 | 1.26× |
| V2 folded + TAESD on the GPU | 16.6, 16.8, 16.5, 16.2, 16.3 | 1.29× |
| **V2 folded + TAESD on the GPU + 2 workers** | **15.5, 15.6, 15.6, 15.7** | **1.34–1.35×** (against the shipped runs of the same sessions, 21.0 / 20.8 / 21.0 / 21.3) |
| Same with 3 workers | 15.3, 15.5 | 1.37×, at three frames in flight (46 ms per frame through the ANE stage): not worth the latency |
| Shipped chunks + TAESD on the GPU + 2 workers (no new model pack) | 18.6, 18.7 | 1.13× |

The paced session (`bench-mac --lipsync-session`, a 60 s WAV, 1,440 frames at 24 fps) with the full candidate:
1,440 generated, 1,439 shown, 0 skipped, 0 late, 0 stalls, clock start 0.23 s (shipped: same counts, 0.29 s).

Rejected on the way: deep-feature reuse across frames and the V2 chunk size (§3.5). Not yet tried: output backings
for the 13 boundary tensors, TAESD fused into chunk 2 (moot once it runs on the GPU).

What to expect on the A14 (estimates until the phone run): V2's gain was −16 % on the M5 Pro; Apple reports larger
V2 gains on iPhones. The fold removes ~52 MB of per-frame weight reads there (~1.5 ms at the A14's ~34 GB/s). TAESD
on the ANE stage is 5.1 ms on the A14 (§5.2), off the critical path once it moves to the GPU, as long as the
compose stage (~9–10 ms) plus the GPU decode stays under the UNet. From the 2026-11-01 ANE stage of 62–64 ms in
calls, these give roughly 62 → 45–48 ms (21–22 fps) before the second worker, whose gain depends on how long the
A14's CPU takes between the chunk calls.

Reproduce:

```bash
.venv-ct/bin/python scripts/ios/export_coreml_ane_chunks.py --apple-repo <ml-stable-diffusion> --cut 0 --variants int8w \
    --attention SPLIT_EINSUM_V2 --fold-time-embedding --prefix unet_anev2fc0
# compile both chunks with `xcrun coremlcompiler compile`, add them to a bundle's meta.json as unet_anev2fc0_int8w_b1_chunk{1,2}
.build/release/bench-mac --resources <bundle> --lipsync-check <bundle>/avatar --lipsync-frames 400 \
    --lipsync-taesd-units GPU --defer-decode --infer-workers 2
# phone: MuseTalkBench --lipsync --exit --unet unet_anev2fc0_int8w_b1_chunked --taesd-gpu --defer-decode --infer-workers 2
```

## 6. The harness app (`ios/MuseTalkBench/`)

```
ios/MuseTalkBench/
  Package.swift                       HarnessCore (library) + bench-mac (CLI); iOS 17 / macOS 14; Swift 5
  Sources/HarnessCore/
    Meta.swift                        meta.json / avatar.json models (BundleMeta, ModelMeta with chunk_of/chunk_index, TensorFile …)
    Tensors.swift                     flat fp16/fp32 tensor files, MLMultiArray construction and conversion
    Metrics.swift                     diff statistics (torch.quantile-compatible), timing stats, monotonic clock
    DeviceInfo.swift                  device model, OS, memory footprint / available memory, free disk, thermal state
    Report.swift                      the JSON report (UNet runs, sustained buckets, chain, Whisper, options)
    Harness.swift                     BenchOptions, model loading, gate scoring, timing, MLComputePlan placement, sustained/chain/whisper phases
    Lipsync.swift                     AvatarBundle, window gather (any length), BlendPlan/compose, LipsyncEngine, LipsyncPipeline, StagedUNet
    LipsyncSession.swift              FrameBuffer + one spoken turn: speak(waveform:playbackURL:), pre-roll, the AVAudioPlayer clock,
                                      currentFrame() for a display link, stats line, Result (LIPSYNC_RESULT); what the Lingua module wraps
    AudioInput.swift                  loadWaveform16k: any audio file -> mono float32 16 kHz (AVAudioConverter)
    TarArchive.swift                  streaming ustar reader for the phone bundles and model packs; BundleError
    RemoteBundle.swift                download with SHA-256, fetchAvatarBundle, fetchModelPack (manifest -> HarnessBundle-shaped folder)
  Sources/bench-mac/BenchMac.swift    CLI: --resources --out --frames --sustained --chain --accuracy-frames --units --models
                                      --no-gpu --no-b4 --no-whisper --no-plan --no-reload --no-unet --taesd-units --settle
                                      --phase-units --lipsync-check <avatar dir> --lipsync-frames N
                                      --lipsync-session <avatar dir> --wav <file> [--preroll S]   (a whole turn, silent, on the Mac)
                                      --unpack-tar <tar> <dir>                                   (the tar reader)
  App/project.yml                     xcodegen spec (bundle id com.lingua.musetalkbench, team 563AETFKT9, HarnessBundle as a folder reference)
  App/MuseTalkBench.xcodeproj         generated, committed
  App/Sources/
    MuseTalkBenchApp.swift            @main; tabs; autorun / autostart from launch arguments
    ContentView.swift                 TabView: Bench (options, run, log, share JSON) and Lipsync
    BenchViewModel.swift              runs the harness; --autorun flags; prints [bench] lines and BENCH_JSON_BEGIN/END; saves to Documents
    LipsyncPlayer.swift               thin wrapper over LipsyncSession: engine load, optional bundle/WAV download, CADisplayLink, prints LIPSYNC_RESULT
                                      flags: --lipsync [--exit] [--avatar-url <tar> [--avatar-sha256 <hex>] [--avatar-id <id>]]
                                             [--wav <path|url>] [--preroll <s>] [--silent]
    LipsyncView.swift                 the image, status, stats, Play/Stop
  HarnessBundle/                      GENERATED, gitignored: compiled models, gate captures, Whisper segment, avatar/
  README.md                           build, run, phone steps, flags
```

- **Bundle** (`scripts/ios/prepare_harness_bundle.py`, 0.83 GB with the chunks + 46 MB avatar): `xcrun
  coremlcompiler compile` of each manifest model into `.mlmodelc`; the 128 gate frames as flat fp16 binaries
  (`unet_latents_f16.bin`, `unet_audio_f16.bin` with the positional encoding subtracted because the graph adds it,
  `unet_ref_f16.bin` = the server's fp16 outputs); a 30 s Whisper segment (`whisper_waveform_f32.bin`,
  `whisper_mel_f16.bin`, `whisper_features_ref_f16.bin`); `meta.json` describing every model's I/O, the gate
  thresholds and the Mac reference figures shown next to the device's. `--models` picks other manifest entries.
- **Bench report**, per UNet (chunk groups are reported as one UNet named `<parent>_chunked`) and compute-unit
  setting: load time (first, cached), first-prediction latency, per-file gate statistics, per-frame timing (mean,
  p50/p90/p99, max, frames over 40/50 ms), `MLComputePlan` placement (ops and estimated cost per device, iOS 17.4+),
  then sustained buckets with thermal state, the UNet → TAESD chain (with a warm-up call so the decoder's first-call
  compile is not in the mean), and the Whisper phase with its feature error. Console shows every phase with
  footprint, available memory, free disk, memory-pressure and thermal events.
- **Phone protocol** (Mac side; phone unlocked, Developer Mode on):

  ```bash
  cd ios/MuseTalkBench/App
  xcodebuild -project MuseTalkBench.xcodeproj -scheme MuseTalkBench -configuration Release \
    -destination 'id=<udid>' -derivedDataPath DerivedData -allowProvisioningUpdates build
  xcrun devicectl device install app --device <udid> DerivedData/Build/Products/Release-iphoneos/MuseTalkBench.app
  # timing harness
  xcrun devicectl device process launch --device <udid> --console --terminate-existing com.lingua.musetalkbench \
    --autorun --units CPU_AND_NE --phase-units CPU_AND_NE --frames 200 --sustained 30 --accuracy-frames 128 \
    --no-b4 --no-gpu --no-reload --settle 10 --exit
  # streaming lipsync view, bundled avatar and clip
  xcrun devicectl device process launch --device <udid> --console --terminate-existing com.lingua.musetalkbench \
    --lipsync --exit
  # streaming lipsync view from a downloaded phone bundle and a TTS WAV (presigned S3 URLs; what Lingua will do)
  xcrun devicectl device process launch --device <udid> --console --terminate-existing com.lingua.musetalkbench \
    --lipsync --exit --avatar-url "$(aws s3 presign s3://lingua-musetalk-s3-storage/avatars/ios/v1/<avatar_id>.tar)" \
    --avatar-sha256 <sha256 from the sidecar> --avatar-id <avatar_id> --wav "$(aws s3 presign s3://…/clip.wav)" --preroll 0.4
  ```

  The JSON report also lands in the app's Documents folder (Files → On My iPhone → MuseTalk Bench, or
  `devicectl device copy from --domain-type appDataContainer`). The user's iPhone 12 Pro is UDID
  `00008101-00123DA43A51003A`.
- **Gotchas**: an iOS bundle must not contain a top-level folder named `Resources` (codesign refuses the app; the
  folder is `HarnessBundle`); inside `async` Swift `MLModel.prediction(from:)` resolves to the async overload, so
  timed calls go through a synchronous helper; the iOS Simulator has no Neural Engine (CPU numbers only, and ALL in
  the simulator returns garbage); `log collect --device` needs root and zsh's `log` builtin shadows `/usr/bin/log`.

---

## 7. Repository map

### 7.1 Scripts (`scripts/ios/`)

| Script | Purpose |
|---|---|
| `bench_torch_baseline.py` | PyTorch (MPS/CPU) per-frame baseline; saves the 16-frame reference set |
| `export_coreml.py` | UNet (diffusers layout) and TAESD → Core ML; fp16 / int8w / int8wa; batch 1/4/8 |
| `export_coreml_ane.py` | UNet in Apple's ANE layout from MuseTalk weights; fp16 / int8w; batch 1/4 |
| `export_coreml_ane_chunks.py` | the two-chunk fp16-boundary UNet for iOS (`--cut`, `--check`) |
| `chunk_coreml.py` | `bisect_model` chunks (kept for the record; rejected on the phone) |
| `export_coreml_precision.py` | fp32-island variants for the error attribution |
| `palettize_coreml.py` | 6/4-bit palettized variants |
| `export_whisper_coreml.py` | log-mel and Whisper-tiny encoder → Core ML |
| `bench_coreml.py` | Mac ANE timing, numerics, sustained and back-to-back turn tests |
| `bench_latency.py` | load, first frame, jitter, blend cost |
| `bench_whisper_coreml.py` | front-end timing and the all-Core-ML chain check |
| `profile_coreml.py` | `MLComputePlan` cost by op type / block |
| `validate_unet_coreml_gate.py` | the repo gate on a Core ML UNet (`--units`, `--pytorch-fp32`) |
| `render_evidence.py` | labelled video evidence of each build vs the PyTorch reference |
| `review_ane_vs_fp16.py` | the r5-style A/B review package (arms A/N/B, blind + labelled clips, metrics, FaceMesh) |
| `prepare_harness_bundle.py` | builds `HarnessBundle/` for the app |
| `prepare_avatar_bundle.py` | builds `HarnessBundle/avatar/` for the Lipsync tab; `--phone-only` writes just what the phone needs |
| `pack_phone_bundle.py` | phone avatar bundle → `<avatar_id>.tar` (plain ustar) + sidecar json for `avatars/ios/v1/` |
| `pack_model_pack.py` | the lipsync path's compiled models → one tar each + trimmed `meta.json` + `manifest.json` for `musetalk/ios/<version>/` |

### 7.2 Documents and results (`docs/`)

- `ios_unet_bench_2026-10-30.md`: the measurement log, §1–12 (setup, conversion, baseline, ANE results, evidence,
  turns, latency, Whisper, phone projection, levers, product gaps, reproduce, gate + attribution, harness, iPhone,
  review).
- `ios_unet_bench/*.json`: every measurement (`baseline_torch_*`, `coreml_*`, `gate_*`, `profile_*`, `latency_*`,
  `turns_*`, `whisper_coreml_*`, `chunk_*`, `harness_mac_*`, `harness_iphone12pro_*`, `lipsync_iphone12pro_*`).
- `ios_unet_bench/evidence/`: stills and metrics for codex_smoke, Japanese, Latina.
- `ios_unet_bench/review_ane_vs_fp16/`: README, `review_sheet.csv`, `key/blind_key.json`, `metrics.{json,md}`, stills.
- This file.
- Published for the phone (bucket `lingua-musetalk-s3-storage`, 2026-10-31): `musetalk/ios/v1/` = the model pack
  (`manifest.json`, `meta.json`, `unet_anec0_int8w_b1_chunk1/2.mlmodelc.tar` 346 + 509 MB, `taesd_fp16_b1`,
  `whisper_encoder_fp16`, `whisper_logmel_fp32` tars, the three gate capture bins; 879 MB) and `avatars/ios/v1/` =
  phone avatar bundles (`<avatar_id>.tar` + `.json`; Latina talking so far). Served by presigned URLs (or a
  CloudFront origin on that bucket, not set up yet). `musetalk/ios/test/omnivoice_broad-0084.wav` is the 9.15 s
  OmniVoice test clip. The Lingua plan is `docs/musetalk-on-device-renderer-plan.md` in the Lingua repo.

### 7.3 Generated, not in git

`models/` (weights: `musetalkV15/unet.pth`, `taesd/`, `whisper/`, `sd-vae/`), `models/ios_bench/coreml/*.mlpackage`
and `manifest.json`, `models/ios_bench/evidence/*.mp4`, `ios/MuseTalkBench/HarnessBundle/`, `.build/`,
`App/DerivedData/`, `experiments/video_validation/ane_vs_fp16/`, `.venv-ct`, `.venv-ios`, `.venv-mp`, the avatar
caches (from `s3://lingua-musetalk-s3-storage/avatars/v15/`).

---

## 8. Reproduce (short form; full commands in the bench doc §10)

```bash
cd ~/code/musetalk-ios-bench
# environment and weights: bench doc §10 (uv venv .venv-ct; torch 2.7.0, torchvision 0.22.0, coremltools 9.0; hf download …)
git clone --depth 1 https://github.com/apple/ml-stable-diffusion /tmp/ml-stable-diffusion
.venv-ct/bin/python scripts/ios/export_coreml_ane_chunks.py --apple-repo /tmp/ml-stable-diffusion --cut 0 --variants int8w --check
.venv-ct/bin/python scripts/ios/export_coreml.py --batches 1 --variants fp16          # for taesd_fp16_b1
.venv-ct/bin/python scripts/ios/export_whisper_coreml.py
.venv-ct/bin/python scripts/ios/prepare_harness_bundle.py                               # chunks + TAESD + Whisper + gate frames
.venv-ct/bin/python scripts/ios/prepare_avatar_bundle.py --avatar /path/to/latina_guided_20260925_talking_84c5bc80b8 \
    --audio data/audio/ai-assistant.mpga
cd ios/MuseTalkBench && swift build -c release
.build/release/bench-mac --resources HarnessBundle --out ../../docs/ios_unet_bench/harness_mac_m5pro.json   # Mac sanity
.build/release/bench-mac --resources HarnessBundle --lipsync-check HarnessBundle/avatar                       # frame loop check
# phone: §6 above
```

---

## 9. Open items, in order

1. **Human blind pass** of the review package (`docs/ios_unet_bench/review_ane_vs_fp16/README.md`), unblind, apply
   the acceptance rule. The one question is the Japanese character's lip aperture.
2. If the Japanese result needs fixing: the only known lever is moving the error-carrying 1,280-channel middle blocks
   to the GPU as a separate model (Core ML will not split it; bench doc §11.2), worth it only on phones whose GPU can
   carry it; not an A14.
3. **Newer phone**: an iPhone 15 Pro or later run (int8 weights stay int8 on the ANE; 30–40 ms projected) decides
   whether real time at 20 fps is reachable without a GPU split. The A14 is settled: compute-bound at 54–66 ms per
   frame, 16.4 fps shown, 13–15 fps when warm (§5.2, §5.4).
4. **Streaming audio**: replace the bundled clip with a ring buffer fed by the microphone or a TTS stream; the window
   gather already works per frame, the encoder needs the 30 s segment re-run as audio arrives (28 ms on the A14) or
   the server can send the windows with the audio.
5. **Avatar download**: fetch a character's cache from S3 and convert on device (or publish the phone bundle format
   from the character factory); multiple avatars; the pose router / idle-live switching from `webrtc_pose_router.py`.
6. **Thermal budget and the few-ms levers**: the warm phone loses up to 45 % of its Neural Engine speed within 15 s
   (§5.4), which outweighs the remaining levers (output backings for the 13 boundary tensors, TAESD fused into
   chunk 2, the 10 ms CPU stage in Metal or Accelerate, already hidden by the pipeline). A product build should
   compile at install time, not before a call, and measure a 5-minute call, not a 30 s burst.
7. **Lingua integration**: a native module around `HarnessCore` (the Lingua app is Expo 53 / React Native 0.79 with no
   on-device ML today; it consumes WebRTC from the server). The Lipsync tab is the reference for what the module
   does: the phone renders and displays its own frames, no WebRTC and no server in the loop. **Planned 2026-10-31**
   in the Lingua repo, `docs/musetalk-on-device-renderer-plan.md`: the Lingua client already holds the TTS WAV
   before uploading it to the worker, so device mode is a local branch at that point (no new server transport); an
   Expo module + view over `HarnessCore`, a `musetalk.renderer` flag (server / device / auto, with a server kill
   switch), phone avatar bundles exported server-side from the v15 cache, and a per-turn latency timeline recorded
   in both modes. Phase 0 there asks this repo for: the player logic (`FrameBuffer`, pre-roll, display link, audio
   clock) moved from the harness app into `HarnessCore`, WAV → 16 kHz resampling, > 30 s audio segmenting, and a
   pre-roll parameter. **Phase 0, this repo's half, done 2026-10-31**: `LipsyncSession` (+ `AudioInput`,
   `TarArchive`, `RemoteBundle`), `--phone-only` export, `pack_phone_bundle.py`, `pack_model_pack.py`, the model
   pack and the Latina bundle on S3, the harness playing a downloaded bundle with a downloaded OmniVoice WAV on the
   iPhone 12 Pro: 17.2 fps shown, 0 stalls, clock start 0.79 s after `speak()` warm (§5.2). Phase 1 note: the
   0.4 s pre-roll is 10 frames, which the A14 takes 0.62 s to make; 0.2 s would start ~0.3 s sooner at the cost of
   skipping from the first second (the clock outruns generation either way).

---

## 10. Changelog (branch `claude/ios-unet-bench`, base `d8a3e8f` on main)

| Commit | Date | What |
|---|---|---|
| `4e02e8c` | 2026-10-30 | PyTorch baseline of the per-frame path on an M5 Pro; Core ML export and bench scripts |
| `b17e52e` | 2026-10-30 | UNet export through `torch.export` + `run_decompositions` |
| `67bea46` | 2026-10-30 | Neural Engine results on the M5 Pro, turn test, evidence renderer |
| `c12c367` | 2026-10-30 | Results doc, back-to-back turn tests, codex_smoke evidence metrics and stills |
| `ddbe36e` | 2026-10-30 | Evidence on the Japanese and Latina 9:16 characters |
| `c12d609` | 2026-10-30 | Load time, per-frame jitter and blend cost of the int8w build |
| `22656d0` | 2026-10-30 | Whisper-tiny front end in Core ML, validated through the UNet |
| `3e6ff8c` | 2026-10-30 | Repo gate on the Core ML UNet, ANE error attribution, Apple-layout and palettized levers |
| `18d105e` | 2026-10-30 | Xcode timing harness (HarnessCore + bench-mac + SwiftUI app) with the repo gate on device |
| `3ce20ad` | 2026-10-30 | iPhone 12 Pro measured; two-chunk fp16-boundary UNet (59.6 ms/frame on the A14 ANE) |
| `bbe9967` | 2026-10-30 | Labelled A/B quality review package for the Neural Engine UNet (r5 protocol) |
| `c77437a` | 2026-10-30 | Streaming lipsync view (avatar bundle export, Swift frame loop, paced player) |
| `6ea7532` | 2026-10-30 | Review package rendered for six production characters; metrics and findings in the bench doc §12.4 |
| `79e3535` | 2026-10-31 | This document |
| `9d4904e` | 2026-10-31 | Bottleneck pinpointing: two-stage lipsync pipeline (CPU stage off the critical path, Mac 23.6 → 20.8 ms); batch-4 two-chunk UNet exported for the bandwidth-vs-compute test on the A14; bench doc §12.5 |
| `8f05b0e` | 2026-10-31 | Phone runs: batch 4 vs batch 1 on the iPhone 12 Pro (54.2 vs 56.4 ms: compute-bound), warm-start sustained run (54 → 78 ms), the lipsync view played on the phone (16.4 fps shown, 0 stalls); fixes: `Stats` public init (app build), `FrameBuffer` self-deadlock, skip-ahead aims `clock + lead`; results in §5.2, §5.4 and bench doc §12.5 |
| `704f4a2`, `11b8884` | 2026-10-31 | Two more lipsync runs (17.1 / 16.6 fps), audio route and volume reported; open item 7 points at the Lingua plan. PR #5 merged to main as `fad1f08` |
| — | 2026-10-31 | Lingua Phase 0, this repo's half: `LipsyncSession` + `AudioInput` + `TarArchive` + `RemoteBundle` in HarnessCore; windows for any-length audio; `prepare_avatar_bundle.py --phone-only`; `pack_phone_bundle.py`, `pack_model_pack.py`; model pack and the Latina bundle on S3; harness `--avatar-url/--wav/--preroll`; bench-mac `--lipsync-session`, `--unpack-tar`; phone plays the S3 bundle with an OmniVoice WAV at 17.2 fps |
| `704d6c5`, next | 2026-11-02 | Branch `claude/ios-25fps` (from main `a5e75f9`): TAESD decodable in the compose stage on the GPU, UNet picked by name, `fastPrediction` option, `inferWorkers`; the chunk exporter's `--fold-time-embedding`, `--v2-chunk-size`, `--prefix`; `check_deep_reuse.py`; Mac results in §5.6 (1.34× on the full pipeline) |
