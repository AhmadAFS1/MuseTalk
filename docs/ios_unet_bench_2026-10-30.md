# MuseTalk on an Apple Neural Engine: baseline, Core ML conversion, evidence (2026-10-30)

**Question.** Can the MuseTalk v1.5 live path (one UNet step + decoder per frame) run on an iPhone in real
time for one user, at the quality of the GPU servers? The servers reach ~400 fps aggregate on an RTX 4070
SUPER / 3090 with TensorRT INT8 engines, but that is 15–20 streams at 20 fps each, ~2.5 ms of GPU per frame.

**Verdict.** On this Mac's M5 Pro Neural Engine, the same ANE generation as the iPhone 17 Pro's, the whole
per-frame path runs at **28.9 ms/frame in fp16 and 25.3 ms/frame with int8 weights**, against a 50 ms budget at
the server's 20 fps (40 ms at 25 fps). A 10 s turn renders in 6.3–7.2 s of inference, chunk by chunk. Both
builds are visually indistinguishable from the PyTorch reference on the two 9:16 production characters
(Japanese, Latina) and a smoke avatar (face-crop error 0.16–0.30/255, 46–54 dB PSNR over 250 frames each;
§5). **Correction, same day (§11):** on the repo's own UNet gate (`scripts/validate_unet_backend.py`: production
captures, fp16 PyTorch reference, per-file mae ≤ 0.01 and max_abs ≤ 0.5) both builds **fail**, mae 0.013 and
max_abs up to 1.0, about 4× the error of the shipping TensorRT INT8 engines. The cause is the Neural Engine's own
fp16 arithmetic in the conv/linear layers: the identical Core ML model run on this Mac's GPU scores mae 0.0004,
the same as fp32. Earlier sections quote the 16-frame max-abs figure (0.37–0.41) that led to the wrong "passes
the gate" claim; §11 has the real numbers. Naive int8 activations are
faster (20.6 ms) but fail the gate. The audio front end (log-mel + Whisper-tiny encoder) is in Core ML too: ~13 ms
per 30 s turn on the Neural Engine, and the all-Core-ML chain sits as close to the reference as the UNet alone
(§6.2). **No iPhone was measured**: the only phone available is an iPhone 12 Pro,
whose A14 Neural Engine and 34 GB/s memory bus put it well below real time. Expected on real devices, unmeasured:
iPhone 17 Pro ~25–30 ms/frame, iPhone 15/16 Pro ~30–40 ms (real time at 20 fps, thin at 25), iPhone 12 Pro
~50–80 ms (not real time). "Same exact quality and speed" as the server is not on the table: ~10× slower per
frame (one stream's worth) and a different, gated numeric path.

Branch `claude/ios-unet-bench`, worktree `~/code/musetalk-ios-bench`. Nothing here touches the server path.

## 1. Setup

| Item | Value |
|---|---|
| Host | Apple M5 Pro, 24 GB, macOS 27.0.1, Xcode 27 |
| Baseline venv | `.venv-ios`: Python 3.12.12, torch 2.15.0 (MPS), diffusers 0.42.1 |
| Export venv | `.venv-ct`: torch 2.7.0 (coremltools' newest tested torch), coremltools 9.0, diffusers 0.42.1 |
| UNet | HF `TMElyralab/MuseTalk` `musetalkV15/unet.pth` (fp32, 3.4 GB; 849.9M params), `musetalk.json` |
| Decoders | TAESD (`madebyollin/taesd`, the server's decoder); SD-VAE `stabilityai/sd-vae-ft-mse` for the stock reference |
| Audio features | Whisper-tiny encoder, repo `AudioProcessor` windows (50×384 per frame at the 50 Hz / 25 fps timeline) |
| Avatar | `codex_smoke` from S3 `avatars/v15` (600 frames, 512×512, cached latents, masks, coords) |
| Audio | `data/audio/ai-assistant.mpga` (17.8 s; the first 10 s = 250 frames at 25 fps are used) |

Scripts (all under `scripts/ios/`): `bench_torch_baseline.py`, `export_coreml.py`, `bench_coreml.py`,
`render_evidence.py`, `bench_latency.py`, `export_whisper_coreml.py`, `bench_whisper_coreml.py`, and for §11
`validate_unet_coreml_gate.py`, `profile_coreml.py`, `export_coreml_ane.py`, `export_coreml_precision.py`,
`palettize_coreml.py`. Results JSON under `docs/ios_unet_bench/`. Core ML models and videos are under
`models/ios_bench/` (not in git).

## 2. What was converted

The Core ML UNet model is the whole per-frame UNet step the server runs: inputs `sample` [B,8,32,32] (masked
face latent + reference latent) and `audio` [B,50,384] (raw Whisper window); the sinusoidal positional encoding
and the fixed timestep 0 are inside the graph; output `pred_latents` [B,4,32,32]. Attention uses diffusers'
explicit `AttnProcessor` (matmul + softmax), target iOS 17, fp16 compute.

- `torch.jit.trace` fails in coremltools 9 on 592 `aten::Int` shape casts (`height*width` reshapes and
  `batch//heads` in attention) that it cannot fold; `torch.export` + `run_decompositions({})` specializes them.
- Variants: **fp16** (1,622 MB), **int8w** (int8 per-channel linear weights, fp16 activations, 812 MB),
  **int8wa** (int8 weights and per-tensor activations, calibrated on 16 real frames, 813 MB). One model per batch
  size 1, 4, 8 (int8wa at 8 was not built; the variant fails quality at 1 and 4).
- TAESD decoder [B,4,32,32] → [B,3,256,256] in [0,1], fp16, 2.4 MB, with the server's `(x/2+0.5).clamp`.
- **Audio front end** (`export_whisper_coreml.py`, §6.2). MuseTalk's conditioning is the Whisper-tiny *encoder*
  (`musetalk/utils/audio_processor.py`): `whisper_logmel_fp32`, waveform [1,480000] (one 30 s segment) → log-mel
  [1,80,3000], the `WhisperFeatureExtractor` math written as a conv1d STFT so it converts without the complex
  dialect (fp32, 0.7 MB, matches the extractor to 2e-5); `whisper_encoder_fp16`, log-mel → `features`
  [1,1500,5,384], the 4-layer encoder with its five hidden states stacked exactly as
  `AudioProcessor.encode_whisper_feature` does (equal to `transformers`' `output_hidden_states` to 0.0; fp16,
  15.7 MB, `torch.jit.trace` works here). The per-frame 50×384 gather (`build_audio_prompts`) is indexing, not
  a model.

## 3. PyTorch baseline on this Mac (`baseline_torch_m5pro.json`)

Real avatar latents and real Whisper windows; median of 30 iterations after 5 warm-ups; step = PE + UNet + TAESD.

| Device | dtype | batch | UNet ms/frame | TAESD ms/frame | step ms/frame | single-stream fps |
|---|---|---|---|---|---|---|
| MPS (GPU) | fp16 | 1 | 47.6 | 3.4 | 51.0 | 19.6 |
| MPS | fp16 | 4 | 21.9 | 2.2 | 24.1 | 41.5 |
| MPS | fp16 | 8 | 18.8 | 2.1 | 20.9 | 47.9 |
| MPS | fp16 | 16 | 17.2 | 2.0 | 19.2 | 52.2 |
| MPS | fp32 | 4 | 33.1 | 4.7 | 37.8 | 26.5 |
| CPU | fp32 | 1 | 203.6 | 75.0 | 277.9 | 3.6 |
| CPU | fp32 | 4 | 141.4 | 40.7 | 182.6 | 5.5 |

Whisper-tiny encoder: 3.5 ms per 30 s audio segment on MPS fp16 (per turn, not per frame). Plain PyTorch on
the Mac GPU is already real time at batch ≥ 4; it is not the phone path, but it is the "stock MuseTalk on a Mac"
number.

## 4. Core ML on the Neural Engine (`coreml_m5pro.json`)

`compute_units=CPU_AND_NE`, 100 timed predictions after 20 warm-ups; `cpu` = CPU_ONLY, `all` = CPU+GPU+ANE
(20 runs each). Placement is Core ML's compute plan (ops per device).

| Model | MB | ANE ms/frame | CPU-only | all | placement | latent max-abs vs fp32 | face err /255 (16 fr) |
|---|---|---|---|---|---|---|---|
| UNet fp16, batch 1 | 1,622 | **27.1** | 73.4 | 18.8 | 1,778 ops ANE | 0.37 | 0.43 |
| UNet int8w, batch 1 | 812 | **23.5** | 73.7 | 22.3 | 1,778 ops ANE | 0.41 | 0.48 |
| UNet int8wa, batch 1 | 813 | 19.0 | 88.0 | 18.3 | 3,074 ops ANE | **1.68 (fails)** | 1.63, worst pixel 83 |
| UNet fp16, batch 4 | 1,622 | 314 | 61.6 | 9.3 | **1,778 ops CPU** | 0.23 | 0.22 |
| UNet int8w, batch 4 | 812 | 314 | 61.6 | 10.2 | **1,778 ops CPU** | 0.20 | 0.25 |
| UNet fp16, batch 8 (short run) | 1,622 | 320 | 60.7 | 8.6 | **1,778 ops CPU** | – | – |
| UNet int8w, batch 8 (short run) | 812 | 315 | 60.3 | 9.7 | **1,778 ops CPU** | – | – |
| TAESD fp16, batch 1 / 4 / 8 | 2.4 | 1.52 / 1.43 / 1.40 | 12–14 | 1.3–1.9 | 83 ops ANE | – | 0.08 |

Reading it:

- **Batch 1 is the Neural Engine path.** Every UNet op lands on the ANE; the model loads in 5–7 s (first-load
  compile is cached by Core ML afterwards).
- **Batch 4 is rejected by the ANE compiler**: every op falls to the CPU, where the CPU_AND_NE path is 5× slower
  than CPU_ONLY (a single-threaded fallback), while `all` runs it on the Mac GPU at 9–10 ms/frame. Batching
  therefore needs a graph change (§7), not more hardware.
- **int8 weights buy 13% on this Mac** (23.5 vs 27.1 ms) because the M5 Pro's bus (~150 GB/s) is not the limit
  here: the fp16 model streams 1.62 GB in 27 ms (60 GB/s), the int8w model 0.81 GB in 23.5 ms (35 GB/s), so
  int8w is compute-bound on this ANE. On a phone bus (34–77 GB/s) the halved traffic is what keeps fp16-class
  speed possible (§6).
- **int8 activations (naive, per-tensor) fail**: 4× the latent error of fp16, visible as a 138/255 worst pixel
  in the 250-frame render. The same lesson as the server's r5 work: only a layer-selective recipe passed there.
- The `all` setting is faster than ANE-only on this Mac (18.8 ms fp16 batch 1) because Core ML spreads the graph
  over the GPU and ANE; phone GPUs are far weaker, so ANE-only is the number that transfers.

## 5. Quality gate and video evidence (`evidence/evidence_metrics.json`)

Same 10 s turn (250 frames) rendered through six paths with the repo's own audio windows, cached latents and
`get_image_blending`, i.e. `realtime_inference.py`'s loop minus preprocessing. Reference = PyTorch fp32 UNet +
fp32 TAESD (the production decoder). Errors are mean absolute pixel error in 0–255 over the 256×256 face crop.

| Path | render ms/frame | face err vs ref | worst pixel | PSNR (face) | full-frame err |
|---|---|---|---|---|---|
| fp32 + SD-VAE (stock MuseTalk decoder) | 504.7 (CPU) | 1.03 | 40 | 40.2 dB | 0.14 |
| fp32 + TAESD (reference) | – | 0 | 0 | – | 0 |
| MPS fp16 + TAESD (baseline) | 127.9* | 0.01 | 2 | 68.2 dB | 0.001 |
| **Core ML fp16 (ANE) + Core ML TAESD** | **28.9** | **0.16** | 32 | 53.3 dB | 0.02 |
| **Core ML int8w (ANE) + Core ML TAESD** | **25.3** | **0.17** | 30 | 53.0 dB | 0.02 |
| Core ML int8wa (ANE) + Core ML TAESD | 20.6 | 0.56 | 138 | 42.5 dB | 0.08 |

\* includes first-call kernel compilation; the baseline table above is the timing source for MPS.

The two passing Core ML builds differ from the reference by 6× less than the decoder choice itself (TAESD vs
SD-VAE: 1.03/255), which the server already accepted. Videos (`models/ios_bench/evidence/`): one full-frame mp4
per path with the audio muxed in, `grid_face_crops.mp4` (six face crops side by side), and
`full_frame_ref_vs_coreml.mp4` (reference vs fp16 vs int8w). Stills at 2 / 5 / 8 s and the ×8 difference heatmap
of Core ML fp16 against the reference are in `docs/ios_unet_bench/evidence/`.

![face grid at 5 s](ios_unet_bench/evidence/face_grid_5s.png)

![full frame at 5 s](ios_unet_bench/evidence/full_frame_5s.png)

What this does and does not validate: the Core ML UNet and decoder reproduce the PyTorch pipeline's output on
a real avatar with real speech, frame for frame, within the gate. It does not validate lip-sync against the
audio beyond "identical to PyTorch" (the PyTorch pipeline is the ground truth here), and it is not a labelled
human review of the kind the server's r5 went through.

### 5.1 Production characters (9:16): Japanese and Latina

The same render on the two prepared talking caches the server docs use for decoder comparisons,
`japanese_realtime_talking_7d94520b7f` and `latina_guided_20260925_talking_84c5bc80b8` (482 frames each, 512×832
portrait), same audio, same 10 s. Metrics are against each character's own fp32 + TAESD reference.

**Japanese** (`evidence/japanese_realtime_talking_7d94520b7f/`):

| Path | render ms/frame | face err vs ref | worst pixel | PSNR (face) | full-frame err |
|---|---|---|---|---|---|
| fp32 + SD-VAE | 521.1 (CPU) | 0.79 | 80 | 42.7 dB | 0.14 |
| MPS fp16 + TAESD | 107.9* | 0.01 | 2 | 67.6 dB | 0.002 |
| **Core ML fp16 (ANE)** | **28.8** | **0.28** | 94 | 46.7 dB | 0.05 |
| **Core ML int8w (ANE)** | **25.2** | **0.30** | 97 | 46.1 dB | 0.05 |
| Core ML int8wa (ANE) | 20.7 | 0.74 | 156 | 39.2 dB | 0.13 |

![Japanese face grid at 5 s](ios_unet_bench/evidence/japanese_realtime_talking_7d94520b7f/face_grid_5s.png)

![Japanese full frame at 5 s](ios_unet_bench/evidence/japanese_realtime_talking_7d94520b7f/full_frame_5s.png)

The ×8 difference heatmap shows the fp16 Core ML error confined to the lip region the UNet repaints, with the
rest of the face at zero; the per-pixel worst case (94/255) is a single lip-edge pixel in one frame, while the
mean stays a third of the decoder's own TAESD-vs-SD-VAE difference (0.79).

**Latina** (`evidence/latina_guided_20260925_talking_84c5bc80b8/`):

| Path | render ms/frame | face err vs ref | worst pixel | PSNR (face) | full-frame err |
|---|---|---|---|---|---|
| fp32 + SD-VAE | 500.2 (CPU) | 0.74 | 68 | 43.1 dB | 0.13 |
| MPS fp16 + TAESD | 108.1* | 0.01 | 2 | 67.9 dB | 0.002 |
| **Core ML fp16 (ANE)** | **28.8** | **0.17** | 28 | 54.1 dB | 0.03 |
| **Core ML int8w (ANE)** | **25.1** | **0.18** | 31 | 54.0 dB | 0.03 |
| Core ML int8wa (ANE) | 20.6 | 0.65 | 109 | 41.9 dB | 0.11 |

![Latina face grid at 5 s](ios_unet_bench/evidence/latina_guided_20260925_talking_84c5bc80b8/face_grid_5s.png)

![Latina full frame at 5 s](ios_unet_bench/evidence/latina_guided_20260925_talking_84c5bc80b8/full_frame_5s.png)

Across the three characters the Core ML fp16 and int8w builds stay within 0.16–0.30/255 of the PyTorch
reference on the face crop (46–54 dB), always below the 0.74–1.03/255 that the production decoder itself
differs from the stock SD-VAE by; int8wa is 0.56–0.74/255 with 109–156/255 worst pixels, consistently the
worst. Speed does not depend on the character: 28.8–28.9 ms/frame fp16 and 25.1–25.3 ms/frame int8w on all
three, end to end on the Neural Engine.

## 6. Back-to-back turns (`turns_*_m5pro.json`)

The real-time question in the product's own terms: 12 consecutive 10 s turns at 25 fps (250 frames each),
rendered in chunks of 1 frame through the Core ML UNet and the Core ML TAESD on the Neural Engine, back to
back with no pause (2 minutes of continuous speech).

| Build | first chunk ready | 250 frames rendered in | real-time factor | ms/frame turn 1 → 12 | drift |
|---|---|---|---|---|---|
| int8w, batch 1 | 25 ms (25.1–26.1) | 6.34–6.37 s | **1.57×** every turn | 25.36 → 25.49 | +0.5% |
| fp16, batch 1 | 29 ms (28.7–29.2) | 7.20–7.26 s | **1.38×** every turn | 28.91 → 29.04 | +0.4% |

Every one of the 24 turns finished well inside its own 10 s, with no slowdown over 2 minutes on this Mac. At
the server's 20 fps the same turn is 200 frames, 5.1 s (int8w) / 5.8 s (fp16): 1.96× / 1.72× real time. Phone
thermals are not represented here (§7).

### 6.1 What a streaming client feels: load, first frame, jitter, blend (`latency_int8w_b1_m5pro.json`)

`scripts/ios/bench_latency.py`, int8w batch 1 on the Neural Engine, Latina avatar (512×832, face box 215×332):

| Stage | Measured | Note |
|---|---|---|
| `.mlpackage` → `.mlmodelc` compile | 0.22 s (UNet), 0.03 s (TAESD) | build-time step (Xcode does it); not paid on device |
| Load compiled UNet, first time in a process | **6.7 s** | Neural Engine specialisation of the 812 MB program; the OS caches it per model, so this is app launch / character load, not utterance start |
| Load again in the same process | 0.05 s | |
| First prediction after load | 37.5 ms; second 23.8 ms | one warm-up frame |
| Per-frame UNet, 750 frames | mean 23.7, p99 24.3, max 26.9 ms | |
| Per-frame TAESD | mean 1.6, p99 1.8, max 2.2 ms | |
| Per-frame UNet + TAESD | **mean 25.3, p99 26.1, max 28.5 ms** | 0 of 750 frames over 40 ms (25 fps budget), 0 over 50 ms (20 fps) |
| Resize face to box + `get_image_blending` (CPU) | 0.07 + 0.81 ms (p99 0.98, max 2.5) | runs on the CPU in parallel with the ANE; a phone CPU is a few × slower, still a few ms |

So on this ANE the inference side of the first frame is ~30 ms after the first Whisper window exists, and the
per-frame cost is flat (max − p50 = 3.3 ms over 30 s of frames). The latency a user would see for the first
frame is set by what is upstream of the UNet, not by it: TTS first-audio latency plus MuseTalk's audio look-ahead
(the 50×384 window for frame *i* spans ±2 frames, so the first frame needs ~80–100 ms of audio after its own
timestamp) plus the Whisper-tiny encoder (12 ms on the ANE per 30 s segment, §6.2). Nothing here is measured
on a phone (§7).

### 6.2 Audio front end in Core ML (`whisper_coreml_m5pro.json`)

`scripts/ios/bench_whisper_coreml.py`, Latina avatar, `ai-assistant.mpga`. Timing is per 30 s segment, which
is one call per TTS turn (the server encodes a turn's audio in one go, then windows it per frame):

| Model | CPU | Neural Engine | GPU | placement with CPU_AND_NE |
|---|---|---|---|---|
| log-mel fp32 | 1.28 ms | 1.24 ms (stays on the CPU) | 1.47 ms | 18 of 18 ops CPU |
| encoder fp16 | 18.4 ms | **12.2 ms** | **3.1 ms** | 100 of 100 ops Neural Engine |

Feature error against fp32 PyTorch, max / mean abs over the 1500×5×384 stack (Whisper features have outlier
channels up to |249|, so the max is dominated by them):

| encoder on | HF log-mel | Core ML log-mel |
|---|---|---|
| Neural Engine | 9.46 / 0.0043 | 9.21 / 0.0043 |
| GPU | 2.17 / 0.0007 | 1.96 / 0.0007 |

The log-mel model is not the error source; the Neural Engine's fp16 accumulation is. What counts is the effect
after the UNet. The same 250 frames through the Core ML int8w UNet + Core ML TAESD, against the fp32 reference
(fp32 Whisper → fp32 UNet → fp32 TAESD):

| Features into the int8w UNet | composed face crop (as §5) | raw 256² decoder output | latent vs fp32: max / p99.9 / share > 0.5 |
|---|---|---|---|
| PyTorch fp32 features | 0.177/255, 54.0 dB, worst px 31 | 0.571/255, 48.8 dB, worst 38 | 1.143 / 0.117 / 2.6e-5 |
| **all Core ML: log-mel → ANE encoder** | 0.178/255, 53.9 dB, worst px 31 | 0.572/255, 48.8 dB, worst 38 | 1.176 / 0.118 / 2.9e-5 |
| Core ML vs PyTorch features, same UNet | 0.015/255, 66.3 dB, worst px 8 | 0.044/255, 61.7 dB, worst 9 | 0.191 |

The Core ML front end moves the output by 0.015/255 on the composed face, an order of magnitude under the int8
UNet's own distance from fp32, which does not change. Video: `coreml_full_chain_int8w.mp4` next to the other
Latina renders in `models/ios_bench/evidence/latina_guided_20260925_talking_84c5bc80b8/`; stills
`evidence/latina_guided_20260925_talking_84c5bc80b8/full_chain_vs_ref_{2,5,8}s.png` (reference left, all
Core ML right).

Two things this run shows that §5 did not: the raw decoder output is ~0.57/255 from fp32, three times the
composed-crop figure, because composition pastes only the blended mouth region into an otherwise identical
frame; and over 250 frames the int8w UNet's worst single latent element is 1.14 from fp32 (p99.9 = 0.12, 26 per
million over 0.5), where the 16-frame figure in §5 was 0.41. Whether that is a pass or a fail of the r5 gate is
not decided here: the gate has to be run on its own frame set, with its own statistic.

## 7. From this Mac to a phone

Per frame, the UNet reads its whole weight set once (the ANE has no cache that holds 0.8–1.6 GB), so a phone's
memory bus sets a floor under the frame time regardless of compute:

| Device | ANE | memory bus | floor fp16 (1.62 GB/frame) | floor int8w (0.81 GB) |
|---|---|---|---|---|
| iPhone 12 Pro (A14) | 11 TOPS fp16, no int8 | LPDDR4X ~34 GB/s | 48 ms | 24 ms |
| iPhone 15 Pro (A17 Pro) | 35 TOPS, int8 | LPDDR5 ~51 GB/s | 32 ms | 16 ms |
| iPhone 16 Pro (A18 Pro) | 35 TOPS | LPDDR5X ~60 GB/s | 27 ms | 13.5 ms |
| iPhone 17 Pro (A19 Pro) | 16-core, same generation as M5 | LPDDR5X ~77 GB/s | 21 ms | 10.6 ms |
| this Mac (M5 Pro) | 16-core | ~150 GB/s | measured 27.1 ms | measured 23.5 ms |

On the Mac the int8w model is compute-bound at 23.5 ms. On an iPhone 17 Pro the same ANE generation should land
near that, 25–30 ms with phone clocks and thermals, i.e. real time with margin. On the A17 Pro / A18 Pro ANE
(older, and by Apple's Stable Diffusion benchmarks roughly 1.3–1.5× slower per op than the newest generation)
expect 30–40 ms: real time at 20 fps, borderline at 25. The iPhone 12 Pro's ANE (about a third of the compute,
no int8, 34 GB/s) will not reach 20 fps; 50–80 ms/frame is the expectation. All of these are projections from
one Mac measurement and Apple's published ratios; a real device run is still the missing step.

Other phone-only factors: sustained-load throttling (a call is minutes of continuous turns), a ~2 GB memory
footprint on an app that also holds the avatar's frames and pose caches, and battery.

## 8. Where the remaining speed is (ordered by value)

*(Written before §11; items 1, 2 and 4 were then measured. See §11 for what held up.)*

1. **Ship int8 weights, batch 1, on the ANE** — done here, halves memory traffic. It does not pass the repo
   gate (§11.1), but the int8 weights are not why: they add 0.0003 to a 0.0128 mae that the ANE's arithmetic
   already causes.
2. **Make batching run on the ANE.** Today batch > 1 is rejected and falls to the CPU. The likely cause is the
   attention layout at batch×heads = 32 (`baddbmm` over [32,1024,1024]); Apple's Stable Diffusion conversion
   solved the same problem with a split-einsum attention (conv-1×1 + chunked einsums) and, on iOS 18+, the fused
   `scaled_dot_product_attention` op. Batch 4 would cut weight traffic per frame 4× (the fp16 floor on an iPhone
   15 Pro drops from 32 to 8 ms) at the cost of 160 ms of extra latency at 25 fps, well under the server's 2 s
   prebuffer. This is the lever for the A17/A18 phones. *Done in §11.3 with Apple's layout: batch 4 runs
   entirely on the ANE, and batch 1 got 17–20% faster as a side effect.*
3. **Fuse TAESD and the BGR/uint8 post-processing into the UNet model** so a frame is one Core ML call and one
   buffer copy (saves ~1–2 ms and the fp16 round trip).
4. **Mixed-bit palettization of insensitive layers** (Apple's MBP recipe, 4–6 bit, decompressed in flight by the
   ANE) to cut bytes below int8 without the activation error; re-gate per layer like r5.
5. **Layer-selective int8 activations** (the r5 method, keeping the `up` blocks in fp16): worth ~15–20% more
   compute throughput, only if a real device turns out compute-bound after 1–2. Deferred.
6. **Pipeline:** Whisper-tiny via Core ML (WhisperKit-class), blend in Metal/Accelerate, a dedicated inference
   thread with double-buffered inputs so the UNet never waits on the decoder or the blend.

None of these are needed to hit 20 fps on an iPhone 17 Pro-class ANE if the projection holds; 2 is what makes
the 15/16 Pro comfortable.

## 9. What a product build still needs

What exists is two Core ML models (`unet_int8w_b1`, `taesd_fp16_b1`) and Python harnesses that drive them on a
Mac; it is not a runnable iOS pipeline. Still missing, in dependency order:

1. **Audio features on device: models done (§6.2).** `whisper_logmel_fp32` and `whisper_encoder_fp16` exist and
   are validated through the UNet. What remains is Swift for the 30 s segmenting, the trim/pad and the per-frame
   50×384 gather (`encode_whisper_feature` / `build_audio_prompts`, pure indexing). Having the server send the
   windows with the TTS audio is still an option that removes all of this from the phone.
2. A native module in the Lingua app running UNet → TAESD → blend per frame, loading the UNet once at character
   select (6.7 s on this ANE, §6.1) and pacing frames to the audio clock.
3. The avatar pose caches (3–6 poses × 160–350 MB) downloaded per character; the pose router, idle/live
   switching and A/V sync logic ported from `webrtc_tracks.py` / `webrtc_pose_router.py`; TTS stays remote.
4. A labelled video review of the int8w build like the one r5 went through.
5. The device measurement this doc could not do: per-frame p99 and a 10-minute thermal run on an iPhone 15 Pro
   or newer. On a phone the per-frame floor is memory bandwidth (§7), and sustained ANE + CPU load throttles in
   a way this fan-cooled Mac never showed.

## 10. Reproduce

```bash
cd ~/code/musetalk-ios-bench
uv venv --python python3.12 .venv-ct && uv pip install --python .venv-ct/bin/python torch==2.7.0 diffusers transformers accelerate safetensors numpy psutil coremltools==9.0 scipy opencv-python-headless librosa soundfile einops
.venv-ct/bin/hf download TMElyralab/MuseTalk musetalkV15/unet.pth musetalkV15/musetalk.json --local-dir models
.venv-ct/bin/hf download madebyollin/taesd --local-dir models/taesd
.venv-ct/bin/hf download openai/whisper-tiny config.json generation_config.json preprocessor_config.json model.safetensors --local-dir models/whisper
.venv-ct/bin/hf download stabilityai/sd-vae-ft-mse config.json diffusion_pytorch_model.safetensors --local-dir models/sd-vae
aws s3 cp s3://lingua-musetalk-s3-storage/avatars/v15/codex_smoke.tar.gz /tmp/ && tar -xzf /tmp/codex_smoke.tar.gz -C /tmp/
.venv-ct/bin/python scripts/ios/bench_torch_baseline.py --avatar-latents /tmp/codex_smoke/latents.pt --audio data/audio/ai-assistant.mpga
.venv-ct/bin/python scripts/ios/export_coreml.py --batches 1,4,8 --variants fp16,int8w,int8wa
.venv-ct/bin/python scripts/ios/bench_coreml.py --compare --check
.venv-ct/bin/python scripts/ios/bench_coreml.py --turn-model unet_int8w_b1 --turns 12 --turn-seconds 10 --turn-fps 25
.venv-ct/bin/python scripts/ios/render_evidence.py --avatar /tmp/codex_smoke --audio data/audio/ai-assistant.mpga --seconds 10 --fps 25
.venv-ct/bin/python scripts/ios/bench_latency.py --unet unet_int8w_b1 --avatar /tmp/latina_guided_20260925_talking_84c5bc80b8 --frames 750 --out docs/ios_unet_bench/latency_int8w_b1_m5pro.json
.venv-ct/bin/python scripts/ios/export_whisper_coreml.py
.venv-ct/bin/python scripts/ios/bench_whisper_coreml.py --avatar /tmp/latina_guided_20260925_talking_84c5bc80b8 --audio data/audio/ai-assistant.mpga --seconds 10 --fps 25
```

(The baseline above was measured with torch 2.15.0 in `.venv-ios`; the export needs torch ≤ 2.7 for coremltools 9.)

§11 additionally needs `torchvision==0.22.0` in `.venv-ct` (the gate script's import chain) and a clone of
`https://github.com/apple/ml-stable-diffusion` (imported, not installed):

```bash
uv pip install --python .venv-ct/bin/python torch==2.7.0 torchvision==0.22.0
git clone --depth 1 https://github.com/apple/ml-stable-diffusion /tmp/ml-stable-diffusion
.venv-ct/bin/python scripts/ios/validate_unet_coreml_gate.py --model unet_int8w_b1          # repo gate on a Core ML model
.venv-ct/bin/python scripts/ios/validate_unet_coreml_gate.py --model unet_fp16_b1 --units ALL  # same model on the GPU
.venv-ct/bin/python scripts/ios/validate_unet_coreml_gate.py --model pytorch_fp32             # fp32 anchor
.venv-ct/bin/python scripts/ios/profile_coreml.py --model unet_int8w_b1
.venv-ct/bin/python scripts/ios/export_coreml_ane.py --apple-repo /tmp/ml-stable-diffusion --batches 1,4 --variants fp16,int8w --exporter export
.venv-ct/bin/python scripts/ios/export_coreml_precision.py --variants gn32,sm32,mm32,gnsm32,region_early,region_mid,region_late
.venv-ct/bin/python scripts/ios/palettize_coreml.py --model unet_ane_fp16_b1 --nbits 6,4
```

§12, the on-device harness (Xcode 27, `brew install xcodegen` only to regenerate the project):

```bash
.venv-ct/bin/python scripts/ios/chunk_coreml.py --model unet_ane_int8w_b1 --check   # coremltools bisect (fp32 boundary; CPU on the phone)
.venv-ct/bin/python scripts/ios/export_coreml_ane_chunks.py --apple-repo /tmp/ml-stable-diffusion --cut 0 --check  # torch cut, fp16 boundary
.venv-ct/bin/python scripts/ios/prepare_harness_bundle.py                      # -> ios/MuseTalkBench/HarnessBundle (chunks: 0.83 GB)
cd ios/MuseTalkBench && swift build -c release
.build/release/bench-mac --resources HarnessBundle --out ../../docs/ios_unet_bench/harness_mac_m5pro.json --frames 300 --sustained 20
cd App && xcodebuild -project MuseTalkBench.xcodeproj -scheme MuseTalkBench -configuration Release \
    -destination 'id=<iphone udid>' -derivedDataPath DerivedData -allowProvisioningUpdates build
xcrun devicectl device install app --device <udid> DerivedData/Build/Products/Release-iphoneos/MuseTalkBench.app
xcrun devicectl device process launch --device <udid> --console --terminate-existing com.lingua.musetalkbench \
    --autorun --frames 200 --sustained 30 --exit                                 # phone unlocked, app in front
```

## 11. Speed and quality for an A14-class phone, measured where this Mac can measure (2026-10-30, later)

The question behind this section: can the on-device build match the Vast.ai workers' output on an iPhone 12
Pro, at the same quality and speed. Three things were measured: the Core ML build against the repo's own UNet
gate (the one the server's TensorRT engines are gated with, so the two are judged on identical inputs), where
the Neural Engine's error comes from, and what each speed lever actually buys.

### 11.1 Quality: the repo gate (`gate_*_m5pro.json`)

`scripts/ios/validate_unet_coreml_gate.py` runs `scripts/validate_unet_backend.py`'s statistics on the
production captures in `calibration/unet_portable_bs8/` (16 files, 128 frames, avatar `chin_h3_japanese_new`
among others; inputs as the server's UNet saw them, reference = the fp16 PyTorch production outputs). The
server's numbers come from `docs/fps_comparisons/*/gate/*.json` on its larger 176-file set. Gate: per-file
mae ≤ 0.01 and max_abs ≤ 0.5.

| UNet | runs on | mae mean / max | max_abs mean / max | gate |
|---|---|---|---|---|
| PyTorch fp32 (anchor: fp32 vs the fp16 reference) | Mac CPU | 0.00038 / 0.00048 | 0.015 / 0.047 | pass |
| **server TensorRT fp16** (`gunet_srcfp16_main`) | 4070S | 0.0018 / 0.0025 | 0.109 / 0.414 | pass |
| **server TensorRT INT8, shipping recipe** (`gunet_srcg50_main`) | 4070S | 0.0032 / 0.0044 | 0.253 / 0.777 | mae pass, max_abs over on its set |
| server TensorRT INT8 (`gunet_srcblkA8_main`) | 4070S | 0.0050 / 0.0071 | 0.449 / 1.284 | mae pass, max_abs over |
| Core ML fp16 (same model as below) | **Mac GPU** | 0.00041 / 0.00053 | 0.017 / 0.055 | pass, = fp32 |
| Core ML fp16 | Mac CPU | 0.0064 / 0.0080 | 0.236 / 0.470 | pass |
| Core ML fp16 | **Neural Engine** | 0.0128 / 0.0184 | 0.521 / 1.001 | **fail (mae, 16 of 16 files)** |
| Core ML int8 weights (the "phone build" of §4–6) | Neural Engine | 0.0131 / 0.0187 | 0.544 / 1.027 | fail |
| Core ML Apple layout fp16 / int8w (§11.3) | Neural Engine | 0.0128 / 0.0184, 0.0131 / 0.0187 | 0.52 / 1.00 | fail |
| Core ML Apple layout, 6-bit palettized | Neural Engine | 0.0186 / 0.0232 | 0.565 / 1.004 | fail |
| Core ML Apple layout, 4-bit palettized | Neural Engine | 0.0565 / 0.0750 | 1.47 / 2.46 | fail, visibly (2.3/255) |

So the device build is not at the server's quality by the server's own measure: 4× the shipping INT8 engines'
mae, 7× TensorRT fp16's, and over the 0.01 line on every file. In pixels this is the 0.18/255 composed-face /
0.57/255 raw-face distance of §5–6 (54 dB), which no one has seen in a video, but it is a real, measured gap, and
it is not the int8 weights: fp16 and int8w score the same.

### 11.2 Where the Neural Engine's error comes from

The identical fp16 model is exact on the GPU and 30× worse on the ANE, so the error is in how the ANE executes
fp16, not in the conversion. To find which ops, `export_coreml_precision.py` holds one op class in fp32 (Core ML
then runs those ops on the CPU, splitting the graph) and the gate is rerun:

| fp32 island | ops moved off the ANE | gate mae mean / max | change |
|---|---|---|---|
| none (baseline fp16) | 0 | 0.0128 / 0.0184 | — |
| GroupNorm statistics (`reduce_mean, sub, square, sqrt, real_div`) | 610 of 1,961 | 0.0126 / 0.0181 | none |
| softmax | 101 | 0.0128 / 0.0184 | none |
| attention matmuls + softmax | 224 | 0.0128 / 0.0184 | none |
| GroupNorm + softmax | 711 | 0.0126 / 0.0182 | none |
| GroupNorm rewritten as chained short reductions, all on the ANE (`--groupnorm hier`) | 0 | 0.0128 / 0.0184 | none |
| first third of the 345 conv/linear/matmul ops (conv_in, down blocks 0–1) in fp32 | 722 | 0.0127 / 0.0180 | none |
| last third (up blocks 2–3, conv_out) in fp32 | 634 | 0.0119 / 0.0175 | −7% |
| middle third (down 2–3, mid block, up 0–1: the 1,280-channel layers) in fp32 | compiler put *all* 1,999 ops on the CPU | 0.0060 / 0.0076 (pass) | = the CPU-only row of §11.1, so not a hybrid result |
| every conv/linear with ≥ 10,000 accumulation terms (the 1,280-channel 3×3 convs) in fp32 | compiler put all 1,834 ops on the CPU | 0.0060 / 0.0075 (pass) | CPU-only again, not a hybrid |
| every conv/linear with ≥ 5,000 terms in fp32 | all 1,872 ops on the CPU | 0.0047 / 0.0052 (pass) | " |

Normalisation and attention are not it. The error sits in the convolution and linear layers themselves, i.e. in
the ANE's multiply-accumulate precision (the GPU accumulates fp16 products in fp32; the ANE evidently does not,
or not fully). The depth rows point at the middle of the network, the 1,280-channel layers whose accumulations
are the longest (up to 11,520 terms per output), which is what fp16 accumulation would predict; the first third
being exact buys nothing and the last third 7%. The test that would confirm it, those layers in fp32 with the
rest on the ANE, could not be run as a hybrid: whenever the fp32 islands sit in the middle of the graph, Core
ML's planner gives up on splitting and schedules the whole model on the CPU (three attempts, all 100% CPU). A
real hybrid would have to be two Core ML models, the outer blocks on the ANE and the middle on the GPU, with the
activations handed across; on an A14 the GPU half alone would cost more than the whole budget.

That rules out every cheap fix: there is no op class to reformulate, and int8 weights, layout and batching do not
change it. What is left is a product decision on the 0.013 mae (§11.4), a manual two-model ANE/GPU split for
phones with a GPU fast enough to carry the 1,280-channel middle (not an A14), or a newer ANE if a later
generation accumulates differently (unknown; untested).

### 11.3 Speed: where the time goes and what the levers bought (`profile_*`, `coreml_ane_layout_m5pro.json`, `coreml_gn_pal_m5pro.json`)

Compute-plan cost profile of the int8w batch-1 UNet (all ops on the ANE): conv 35.7%, linear 24.0%,
**reshape + transpose + split 15.9%**, softmax 6.9%, matmul 2.8%, GroupNorm pieces ~7%. The 16% of layout
shuffling is the signature of running a [B, S, C] transformer on hardware that wants [B, C, 1, S].

Apple's `ml-stable-diffusion` UNet is that re-layout (Conv2d everywhere, LayerNormANE, split-einsum attention);
it loads MuseTalk's weights directly (0 missing / 0 unexpected keys) and reproduces the diffusers UNet to
1.6e-5 in fp32 (`export_coreml_ane.py`). Measured on the ANE, batch-1 per frame unless noted:

| UNet build | weight MB per frame | ms/frame on this ANE | vs §4 | ops on ANE | gate mae |
|---|---|---|---|---|---|
| diffusers layout fp16 (§4) | 1,624 | 27.1 | — | 1,778 / 1,778 | 0.0128 |
| diffusers layout int8w (§4) | 812 | 23.5 | — | all | 0.0131 |
| **Apple layout fp16** | 1,624 | **22.6** | −17% | 3,049 / 3,049 | 0.0128 |
| **Apple layout int8w** | 812 | **18.9** | −20% | all | 0.0131 |
| Apple layout fp16, batch 4 | 406 | 25.1 | batch now runs on the ANE | all | 0.0128 |
| Apple layout int8w, batch 4 | 203 | 20.1 | " | all | 0.0131 |
| Apple layout, 6-bit palettized | 610 | 18.6 | | all | 0.0186 |
| Apple layout, 4-bit palettized | 408 | 17.9 | | all | 0.0565 |
| Apple layout + chained-reduction GroupNorm | 1,624 | 23.0 | +0.4 | all | 0.0128 |

Readings: the layout alone is worth 17–20% on this ANE at identical numerics. Batch 4 is now accepted by the ANE
compiler (the diffusers-layout model was not), but on this Mac it is *slower* per frame than batch 1: here the
UNet is compute-bound and batching only adds latency. Its value is on a phone bus, where 203 MB of weights per
frame instead of 812 MB turns a 24 ms floor (iPhone 12 Pro, 34 GB/s) into 6 ms. Palettization to 6 bits saves
25% of bytes for a 45% rise in gate error and a visible 0.7/255; 4 bits is out. Int8 weights remain the right
compression: free in quality, half the bytes.

### 11.4 What it means for an iPhone 12 Pro (A14), still unmeasured

- **Speed.** With the Apple-layout int8w build the Mac ANE does 18.9 ms + 1.5 ms TAESD. The A14's Neural
  Engine is roughly a third to a half of this one per op and its bus a quarter; with batch 4 removing the
  bandwidth floor the frame time is set by compute: **~40–60 ms/frame is the projection**, against the 50 ms
  budget at the server's 20 fps. That is on the line, not comfortably inside it, and the 12 Pro has no fan.
  Only a device run settles it (§9 item 5).
- **Quality.** The ANE cannot run this UNet at the server's numeric quality (§11.1–11.2). The choice is
  explicit: accept the 0.013-mae ANE output after the same labelled human review r5's INT8 recipe went through
  (the pixel distance, 0.18/255 composed, is far below what such a review has ever failed), or move the
  error-carrying layers to the GPU at a speed cost no A14 can pay. "Same quality as Vast" in the gate's terms is
  not available on an A14 Neural Engine with this network.

### 11.5 Order of work from here

1. Device measurement with the Apple-layout int8w batch-1 and batch-4 models (the harness of §12; the
   12 Pro calibrates the projection for the 14/15 Pro).
2. Labelled review of the ANE output against the fp16 reference on the production characters, as r5 did,
   to decide whether the 0.013 gate miss is acceptable for the product.
3. If a hybrid is wanted: a manual two-model split (outer blocks on the ANE, the 1,280-channel middle on the
   GPU), since Core ML will not split it for us (§11.2); only worth building for phones whose GPU can carry the
   middle in the budget, measured on the device, not here.
4. Only then the pipeline work of §9 (window gather, native module, pose logic).

## 12. The on-device harness (2026-10-30, late)

### 12.1 What it is (`ios/MuseTalkBench/`)

A Swift package `HarnessCore` plus two front ends that run the same code: `bench-mac`, a command-line runner
for this Mac, and an iOS app (`App/`, xcodegen project, SwiftUI) that bundles the models and inputs. The
bundle (`HarnessBundle/`, 1.66 GB, gitignored) is assembled by `scripts/ios/prepare_harness_bundle.py`:
`xcrun coremlcompiler` output of `unet_ane_int8w_b1` and `_b4`, `taesd_fp16_b1`, `whisper_encoder_fp16`,
`whisper_logmel_fp32`; the 128 gate frames of §11.1 as flat fp16 binaries (positional encoding removed, the
server's fp16 outputs as reference); a 30 s Whisper segment with fp32 torch features; and the Mac gate
figures for display next to the device's. Per UNet model and compute-unit setting the report has load time
(first and cached), first-prediction latency, the §11.1 gate statistics per capture file, per-frame timing
(mean, p50/p90/p99, max, frames over 40/50 ms), and the `MLComputePlan` placement (ops and estimated cost
per device). Then a sustained run (back-to-back frames for N seconds in 5 s buckets with the thermal state),
the UNet → TAESD chain per frame, and the Whisper front end with its feature error. The JSON goes to the
app's Documents folder, the Xcode / `devicectl` console, and the share sheet; `--autorun … --exit` makes it
scriptable from the Mac. README in the folder has the commands.

Two things learned building it: an iOS bundle must not contain a top-level folder named `Resources`
(codesign then refuses the whole app), and inside an `async` Swift function `MLModel.prediction(from:)`
resolves to the asynchronous overload, so the timed calls go through a synchronous helper.

### 12.2 Validation on this Mac (`harness_mac_m5pro.json`)

The harness must reproduce the Python bench before a phone number means anything. It does:

| run | Python bench (§11) | Swift harness |
|---|---|---|
| `unet_ane_int8w_b1`, CPU_AND_NE, ms/frame | 18.9 | 18.7 (p99 19.0), 100% ops on the ANE |
| same, gate mae mean / max_abs max | 0.0131 / 1.03 | 0.0131 / 1.04, FAIL on every file |
| `unet_ane_int8w_b4`, CPU_AND_NE, ms/frame | 20.1 | 20.1, 100% ANE, gate 0.0131 |
| Whisper encoder, ANE / GPU, ms per 30 s | 12.2 / 3.1 | 11.1 / 2.2; feature error 9.44 / 2.17 max |
| log-mel, CPU, ms per 30 s | 1.24 | 1.1 |
| sustained 20 s, ms/frame first → last bucket | — | 18.2 → 18.4, thermal nominal |
| UNet → TAESD chain, ALL, ms/frame | 25.2 (§6.1, TAESD on CPU_AND_NE) | 23.7 (UNet 18.6 + TAESD 5.2 when the planner picks its own unit) |

Two results the Python bench had not produced. Under ALL (Core ML's choice) the planner puts the batch-4
model entirely on the Mac GPU: 13.0 ms/frame, and it then **passes the gate (mae 0.0034, max_abs 0.35)**,
since the GPU accumulates in fp32. And the int8w model gated on the GPU (CPU_AND_GPU, 64 frames) scores mae
0.0032 / max_abs 0.23: the int8 weights alone cost what the shipping TensorRT INT8 engine costs (0.0032), so
the whole remaining gap on the ANE is the ANE's arithmetic. On a phone, the GPU is not an option at this
size (§11.4), but it is now measured rather than inferred.

Simulator (iPhone 17, iOS 26.2): the app runs end to end on the Mac CPU (no Neural Engine there);
CPU_AND_NE gives mae 0.0074 (fp32 accumulate), while ALL in the simulator returns garbage (mae 0.67), a
known simulator GPU-path artifact, not a device result.

### 12.3 iPhone 12 Pro, measured (2026-10-30 night; `harness_iphone12pro_*.json`)

Device: iPhone13,3 (A14, 6 GB; 5.58 GB visible to the app, 3.07 GB allowance per process), iOS 26.2, no
low-power mode. Six of the first eight launches were killed by the system (signal 9, no crash report) and
the kill took most of the evening to pin down, so the result comes with its cause.

**The single 850M-parameter program does not survive on an iPhone.** The first launch after each install
loaded (24–28 s, the Neural Engine compile) and ran; every later instantiation died during the load. The
instrumented run showed why: during the compile the app's footprint reached **2,464 MB with 607 MB left**
before the allowance (the int8 weights are expanded to fp16 inside the process, the A14 ANE having no int8
weight path), and the phone had **1,040 MB of free disk**, so the 1.6 GB compiled program could never be
cached and every launch recompiled. Apple's ml-stable-diffusion README states the rule for the same network:
chunking "is **required** for Neural Engine deployment on iOS and iPadOS if weights are not quantized to
6-bits or less". With the single program, the phone measured once, on a cold launch: **51.8 ms/frame** (p99
54.2), gate mae 0.0137 on 32 frames; the next launch's 128-frame pass gave **54.3 ms/frame**, gate mae
0.0132 / max_abs 1.02, 100% of 3,049 ops on the ANE, before the kill.

**Chunking, two ways.** `coremltools.models.utils.bisect_model` (Apple's chunk script) cut the program at
420 + 393 MB, bit-identical to the whole on this Mac's ANE, but it declares the 13 boundary tensors fp32,
and the A14 compiler then dropped **both chunks to the CPU** (placement `["CPU": 3066]`, 870 ms/frame
under CPU_AND_NE; under ALL the planner chose the GPU: 388 ms/frame, 1.95 GB resident). Cutting the network
in PyTorch at the mid/up boundary (`scripts/ios/export_coreml_ane_chunks.py --cut 0`: chunk 1 = time
embedding, `conv_in`, down blocks, mid block, 344M parameters, 330 MB int8; chunk 2 = up blocks and
`conv_out`, 506M, 486 MB; boundary = the 4×4 hidden state plus 12 residuals, 3.4 MB, all fp16) is exact
against the diffusers UNet in fp32 (1.25e-5) and bit-identical to the single program on the Mac ANE, and
**the A14 compiler keeps all 3,056 ops on the Neural Engine**, with the process footprint at 53 MB after
both compiles. The compile is 12–32 s per chunk; with disk freed to 12 GB the second compile in the same
process still happened for the other compute-unit setting, so the cache is keyed per setting.

| iPhone 12 Pro, chunked int8w UNet (`unet_anec0_int8w_b1_chunk1/2`) | ms/frame | note |
|---|---|---|
| CPU_AND_NE, 200 frames after a 128-frame gate pass | **59.6** (p50 59.6, p99 61.8, max 62.1) | 100% ANE, footprint 64 MB |
| ALL, same, immediately after the compile | 130.5 | transient: the next ALL load ran at 54 |
| sustained 30 s under ALL, 509 frames | **58.9** mean; buckets 54.5 → 57.2 → 59.0 → 60.2 → 61.2 → 62.4 | thermal "serious" throughout, from the earlier runs |
| chain UNet → TAESD (TAESD placed by Core ML) | **59.7** = 54.6 + 5.1 | no slow-down from alternating programs |
| chain with TAESD on the CPU | 81.5 = 55.1 + 26.4 | keep the decoder off the CPU |
| gate on 128 production frames | mae 0.01317 / max 0.01876, max_abs 1.03, FAIL | identical to the single program and to this Mac's ANE (0.0131) |
| Whisper encoder, 30 s segment | 28.1 (ANE), log-mel 3.7 (CPU) | feature error mean 0.0043, as on the Mac ANE |
| first prediction after a compile | 65–78 | |
| **warm relaunch** (compiled chunks cached, 9.6 GB free): load **0.59 s** (0.17 + 0.42), first prediction 61.5, then **54.7** (p99 56.3) over 50 frames | | the product's launch path |

**Reading.** At 20 fps the budget is 50 ms per frame; the A14 does the UNet in 52–55 ms when cool and
59–62 ms when hot, plus 5 ms for the decoder. That is 16–17 fps sustained, not 20: the §11.4 projection
(40–60 ms) was right and the phone sits at its upper end. The chunk hand-off costs at most a few ms (the
cool single program was 52, the first hot chunked bucket 54.5). Quality is exactly the Mac ANE's: the
0.013 gate miss travels with the Neural Engine, not with the device generation. An iPhone 14 Pro (A16,
~1.6× the A14's ANE) would land near 35 ms by the same arithmetic; 15 Pro and later also keep the int8
weights as int8 on the ANE, which removes the 2.4 GB compile peak that made chunking mandatory here.

What the product build must do, learned the hard way: ship the UNet as two chunks with fp16 boundaries
(not `bisect_model`'s fp32 ones), pin `MLComputeUnits.cpuAndNeuralEngine` (ALL was 2× slower on the
first pass and sends the decoder wherever it likes), load once per session and never hold two
instances, expect 30–60 s of compile on first launch per setting with ~1.7 GB of cache writes, and check
free disk before compiling. The CPU_AND_GPU accuracy anchor and the batch-4 model were not measured on
the phone (both were killed at load with the single program; batch 4 is pointless on an A14 anyway).

### 12.4 Labelled quality review of the Neural Engine output (`docs/ios_unet_bench/review_ane_vs_fp16/`)

The 0.013 gate miss of §11.1 and §12.3 is a number; whether it is visible is a human verdict, and r5 (the INT8
TensorRT recipe the servers run) was accepted on exactly such a verdict: labelled A/B clips per identity with a
mouth zoom and a raw-face difference panel, a per-identity metrics table, and a review of the whole set
(`docs/fps_comparisons/4070s_400fps_20260928/README.md` §8.1–8.5, `scripts/video_ab_round.py`,
`scripts/quality_ab_metrics.py`). `scripts/ios/review_ane_vs_fp16.py` builds the same package for the phone
chain, on this Mac, because the phone's Neural Engine output is bit-identical to this Mac's on the gate frames.

**Arms**, per character and audio clip, sharing the avatar cache, the Whisper windows and the blend:

| arm | what | why |
|---|---|---|
| A, reference | PyTorch fp32 UNet (CPU) + torch TAESD fp32 | the gate's reference is the server's fp16 PyTorch; fp32 differs from it by ~0.0004 latent mae, a thirtieth of B's |
| N, noise arm | PyTorch fp16 UNet (Mac GPU) + torch TAESD fp32 | the server's own precision: the size of difference shipping already accepts, in every metric. It is r5's "±1 LSB noise arm" |
| B, candidate | Core ML int8 UNet in two fp16-boundary chunks on the Neural Engine + Core ML TAESD | the iPhone chain, bit-identical to the 12 Pro's output |
| Btd (metrics only) | B's latents through torch TAESD fp32 | separates the UNet's error from the decoder's |

**Characters**: six production avatars, as r5 had six identities: `japanese_realtime_talking`, `latina_guided`,
`en_01_maya`, `en_03_marcus`, `es_01_valeria`, `es_03_mateo` (the last four pulled from
`s3://lingua-musetalk-s3-storage/avatars/v15/`). **Audio**: `ai-assistant.mpga` (TTS) and `eng.wav` (a real
recording), 10 s each at 25 fps: 12 clips, 3,000 frames per arm.

**Per clip**: `A.mp4`, `B.mp4`, `N.mp4`; the labelled `<clip>_ab.mp4` (A | B full frame, 3× nearest mouth zoom
A | B, the raw 256 px generated face |A−B| ×8, burned-in labels and a metrics line); a blind `<clip>_blind.mp4`
(LEFT | RIGHT with the sides randomised per clip, mouth zoom, no difference panel, no metrics; the assignment only
in `key/blind_key.json`); the raw generated faces and latents of A and B, and the FaceMesh landmarks of all arms
(`experiments/video_validation/ane_vs_fp16/<identity>__<audio>/`, gitignored). In `docs/…/review_ane_vs_fp16/`:
`review_sheet.csv` (one row per clip in a shuffled blind order, columns for the worse side and a 0–3 severity for
lip sync, mouth artefacts, flicker, seam, overall), `key/blind_key.json`, `metrics.json` and `metrics.md`,
`stills/` (a labelled frame at 3 s and 7 s per clip), and `README.md` with the reviewer's steps.

**Metrics** are quality_ab_metrics' where they can be had on this Mac, computed on raw composed frames (no codec
noise): latent mae / max_abs vs A; raw-face mae and PSNR; face-box and mouth-ROI PSNR and SSIM; flicker ratio
B/A (mean |f(t)−f(t−1)|) in the mouth ROI and face box; mouth Laplacian-variance sharpness ratio; Lab shift in the
face box; and the FaceMesh inner-lip aperture series per arm (mediapipe 0.10.14 in `.venv-mp`, the same tracker
configuration as the h3 workflow via `scripts/quality_ab_facemesh.py`): Pearson A vs B, lag of maximum
cross-correlation, mean |Δ| px. r5's thresholds apply where they exist: aperture corr ≥ 0.97, lag 0, mean |Δ| ≤
0.5 px, flicker ratio ≤ 1.05, sharpness ratio ≥ 0.95. SyncNet (GPU, server) and the chin-geometry gates (they need
the h3 workflow's generated landmarks) are not reproduced.

**Acceptance rule** (r5's "every identity acceptable, no systematic artefact", made countable): after the blind
pass and unblinding, B is accepted when no clip has a severity ≥ 2 attributed to B in any column, every identity
has at least one clip where B was not called worse overall, and B is called worse overall at most three times more
often than A over the 12 clips. One failing identity → accepted with a reservation for that character; otherwise
rejected, with the labelled clips and the B / N / Btd columns saying whether the UNet precision or the decoder
carries the difference.

**Rendered 2026-10-30 23:00–23:27** on this Mac: 12 clips, 3,000 frames per arm, FaceMesh tracked every frame of
every arm (0 misses). UNet cost per frame: A 152–175 ms (fp32 CPU), N 91–102 ms (fp16 GPU, batch 1), B 19.6–20.0 ms
(Neural Engine chunks). `metrics.md`, with the fp16 noise arm N beside B in every column:

| identity | audio | latent mae B / N | raw face mae /255 B / N | face PSNR dB B / N | mouth PSNR dB B / N | mouth SSIM B / N | flicker mouth B / N | sharpness mouth B / N | lip corr B / N | lip mean\|Δ\| px B / N | lag B | r5 gates B |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| en_01_maya | ai-assistant | 0.0125 / 0.0003 | 0.639 / 0.012 | 53.0 / 73.5 | 47.0 / 67.9 | 0.9970 / 0.9999 | 0.995 / 1.000 | 1.000 / 1.000 | 0.9982 / 1.0000 | 0.24 / 0.03 | 0 | 6/6 |
| en_01_maya | eng | 0.0122 / 0.0003 | 0.622 / 0.012 | 53.8 / 73.5 | 47.7 / 67.8 | 0.9970 / 0.9999 | 0.995 / 1.000 | 0.999 / 1.000 | 0.9989 / 1.0000 | 0.20 / 0.03 | 0 | 6/6 |
| en_03_marcus | ai-assistant | 0.0118 / 0.0003 | 0.568 / 0.015 | 54.2 / 71.6 | 47.3 / 65.4 | 0.9965 / 0.9999 | 0.992 / 1.000 | 0.975 / 1.000 | 0.9989 / 1.0000 | 0.21 / 0.04 | 0 | 6/6 |
| en_03_marcus | eng | 0.0115 / 0.0003 | 0.556 / 0.015 | 54.2 / 71.5 | 47.2 / 65.2 | 0.9963 / 0.9999 | 0.992 / 1.000 | 0.977 / 1.000 | 0.9986 / 1.0000 | 0.23 / 0.04 | 0 | 6/6 |
| es_01_valeria | ai-assistant | 0.0132 / 0.0003 | 0.594 / 0.010 | 53.3 / 74.3 | 47.4 / 68.8 | 0.9971 / 0.9999 | 1.003 / 1.000 | 0.994 / 1.000 | 0.9986 / 1.0000 | 0.21 / 0.03 | 0 | 6/6 |
| es_01_valeria | eng | 0.0130 / 0.0003 | 0.583 / 0.010 | 53.6 / 74.3 | 47.8 / 68.8 | 0.9971 / 0.9999 | 1.003 / 1.000 | 0.993 / 1.000 | 0.9976 / 1.0000 | 0.24 / 0.03 | 0 | 6/6 |
| es_03_mateo | ai-assistant | 0.0132 / 0.0003 | 0.747 / 0.015 | 52.9 / 72.0 | 45.9 / 65.8 | 0.9962 / 0.9999 | 0.995 / 1.000 | 0.986 / 1.000 | 0.9952 / 0.9999 | 0.28 / 0.04 | 0 | 6/6 |
| es_03_mateo | eng | 0.0132 / 0.0003 | 0.748 / 0.015 | 52.7 / 71.9 | 45.5 / 65.6 | 0.9960 / 0.9999 | 0.995 / 1.000 | 0.983 / 1.000 | 0.9968 / 1.0000 | 0.25 / 0.03 | 0 | 6/6 |
| japanese | ai-assistant | 0.0168 / 0.0003 | 0.762 / 0.014 | 47.1 / 73.0 | 39.5 / 66.6 | 0.9919 / 0.9999 | 1.001 / 1.000 | 0.983 / 1.000 | 0.9632 / 1.0000 | 0.90 / 0.04 | 0 | 4/6 |
| japanese | eng | 0.0159 / 0.0003 | 0.724 / 0.011 | 48.1 / 74.1 | 40.5 / 67.6 | 0.9928 / 0.9999 | 1.006 / 1.000 | 0.989 / 1.000 | 0.9557 / 1.0000 | 0.78 / 0.04 | 0 | 4/6 |
| latina | ai-assistant | 0.0114 / 0.0003 | 0.572 / 0.011 | 55.1 / 74.0 | 48.6 / 67.8 | 0.9968 / 0.9999 | 0.996 / 1.000 | 0.993 / 1.000 | 0.9992 / 1.0000 | 0.19 / 0.03 | 0 | 6/6 |
| latina | eng | 0.0114 / 0.0003 | 0.562 / 0.011 | 55.3 / 74.1 | 48.8 / 67.9 | 0.9968 / 0.9999 | 0.995 / 1.000 | 0.997 / 1.000 | 0.9992 / 1.0000 | 0.21 / 0.03 | 0 | 6/6 |

**What the numbers say before anyone watches a clip.**

- **Five of the six characters** sit together: latent mae 0.0114–0.0132, raw generated face 0.56–0.75/255, face
  PSNR 52.7–55.3 dB, mouth PSNR 45.5–48.8 dB, mouth SSIM 0.996–0.997, flicker ratios 0.992–1.006, mouth sharpness
  0.975–1.000, lip aperture correlation 0.9952–0.9992 at lag 0 with a mean |Δ| of 0.19–0.28 px. All six r5
  thresholds pass on all ten of their clips. Against the record, that is **r4's accepted band** (lip correlation
  0.9989–0.9994, Δ 0.13–0.24 px, mouth PSNR 47.7–50.1 dB, "every perceptual gate passes") and short of r5's
  (0.9997–0.9998, 0.08–0.14 px, 51–53 dB).
- **The Japanese character is the outlier on both audios**: latent mae 0.0159–0.0168 (its gate mae was also the
  highest of the set), face PSNR 47–48 dB, mouth PSNR 39.5–40.5 dB, lip correlation 0.956–0.963 and a mean |Δ| of
  0.78–0.90 px (p95 3.3 px): the two aperture thresholds fail, flicker and sharpness pass. Its stills show nothing
  at full frame; in the 3× zoom the difference is at the teeth and the lip edge. B's latents through the torch
  decoder give the same mouth PSNR, so it is the UNet's error, not the decoder's.
- **The noise arm calibrates all of it**: fp16 on the GPU differs from fp32 by 0.0003 in latent mae, 0.010–0.015/255
  in the face, 65–69 dB mouth PSNR, correlation ≥ 0.9999 and 0.03–0.04 px of aperture, on every clip including the
  Japanese one. The server's own precision loss is 50× smaller than the Neural Engine's in the face and 5–20× in
  the lip metrics; whatever a reviewer sees in B is the ANE, not fp16.

So the human review has one question to settle, on one character first: does the Japanese clip pair read as a
lip-sync difference at severity ≥ 2 in the blind pass? If not, B meets r5's bar as r4 did; if it does, the rule
gives "accepted with a reservation" for that character, and the next engineering step is the fp32-on-GPU
treatment of the error-carrying middle blocks (§11.2) measured on that face. The sheet, key and clips are ready;
nothing in this section is a verdict.
