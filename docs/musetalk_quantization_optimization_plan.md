# MuseTalk Quantization And Backend Optimization Plan

Date: 2026-05-25

## Goal

Improve MuseTalk throughput by applying quantization and backend acceleration to
the parts of the pipeline where lower precision neural inference can actually
reduce end-to-end generation time.

The plan is intentionally staged. The safest path is not to quantize every model
at once. The right path is:

1. Measure the current end-to-end baseline.
2. Quantize the VAE decoder first.
3. Validate visual quality.
4. Quantize or compile the UNet second.
5. Only target Whisper if audio preparation is a measured bottleneck.
6. Keep composition and encoding optimization separate from model quantization.

## Current Starting Point

Based on the current docs and runtime notes:

- The app already uses FP16 for important model paths.
- The working TensorRT path is focused on VAE decode.
- The preferred VAE acceleration path is stagewise TensorRT, not one monolithic
  exported VAE graph.
- Existing benchmarks suggest TensorRT VAE decode gives roughly a 20 percent
  gain over PyTorch FP16 in the tested setup.
- That gain is useful, but insufficient by itself for a large end-to-end
  throughput jump because the remaining cost is split across UNet, VAE decode,
  composition, encoding, memory movement, and scheduler overhead.

## Optimization Principles

1. **Optimize measured bottlenecks only**
   - Do not quantize a model just because it exists.
   - Quantize where the profiler shows meaningful time.

2. **Use real calibration data**
   - Random tensors are not enough for INT8 calibration.
   - Calibration should use real MuseTalk latents, real audio features, and real
     batch sizes from actual jobs.

3. **Prefer mixed precision over full INT8 at first**
   - Keep fragile or quality-sensitive stages in FP16.
   - Use INT8 only where visual and sync quality remain stable.

4. **Validate end-to-end, not only isolated model fps**
   - A faster VAE benchmark is useful.
   - The real success metric is generated video throughput and latency.

5. **Separate model optimization from media pipeline optimization**
   - TensorRT and quantization help UNet/VAE/Whisper.
   - They do not directly fix CPU composition or ffmpeg encode time.

## Candidate Ranking

| Priority | Target | Why | Main risk | Recommendation |
| --- | --- | --- | --- | --- |
| 1 | VAE decoder | Known bottleneck and already has TensorRT stagewise backend | Visual artifacts | Start here |
| 2 | MuseTalk UNet | Core per-frame generation model | Lip-sync drift and latent errors | Do after VAE |
| 3 | Whisper | Audio prep model | Audio alignment degradation | Only if prep is bottleneck |
| 4 | VAE encoder | Avatar prep only for cached avatars | Prep quality issues | Not a live-throughput priority |
| 5 | Face detection/parsing | Avatar prep only | Bad crops/masks | Avoid for now |
| 6 | Positional encoding | Too small | Little or no speedup | Do not prioritize |
| 7 | SyncNet | Not core live inference | No runtime benefit | Do not prioritize |

## Phase 0: Establish A Baseline

Before changing precision, capture a clean baseline.

Required measurements:

- End-to-end fps for generated output.
- First-frame or first-segment latency.
- UNet time per batch.
- VAE decode time per batch.
- Whisper/audio preparation time.
- Tensor H2D/D2H transfer time if available.
- CPU composition time.
- ffmpeg encode/mux time.
- Queueing and scheduler wait time.

Recommended batch coverage:

```text
BS1, BS2, BS4, BS8, BS16, BS32
```

Recommended test clips:

- Short audio, 3 to 5 seconds.
- Medium audio, 15 to 30 seconds.
- Longer audio, 60 seconds.
- At least two avatars with different face sizes and motion.

Baseline output artifacts to save:

```text
baseline timing logs
baseline generated videos
baseline generated face crops
baseline UNet outputs if practical
baseline VAE decoder outputs
```

Acceptance for moving forward:

- Reproducible timings across multiple runs.
- Clear percentage split between UNet, VAE decode, compose, encode, and audio
  prep.
- A saved visual reference set for quality comparison.

## Phase 1: Capture Calibration Data

INT8 quantization needs representative tensors.

For VAE decoder calibration, capture:

```text
pred_latents
batch size
avatar id or avatar type
frame index
output face crop reference
```

For UNet calibration, capture:

```text
latent_batch
audio_feature_batch
timesteps
positional encoding output if separate
UNet predicted latents
batch size
```

For Whisper calibration, only if needed, capture:

```text
audio feature inputs
Whisper encoder hidden states
audio duration
language or phonetic variety
```

Calibration corpus requirements:

- Use real avatars and real audio.
- Include different speakers.
- Include quiet, loud, fast, and slow speech.
- Include different mouth motion intensity.
- Include the actual fixed batch sizes used by the scheduler.

Do not calibrate only on random tensors. Random tensors may produce a fast
engine that is inaccurate on real MuseTalk distributions.

## Phase 2: VAE Decoder Mixed INT8/FP16

This is the first quantization target.

Why:

- VAE decode is a known hot path.
- TensorRT stagewise infrastructure already exists.
- VAE decode runs for every generated frame batch.
- Reducing VAE time directly improves the GPU generation loop.

Recommended implementation strategy:

1. Keep the current stagewise TensorRT structure.
2. Add INT8 calibration for selected decoder stages.
3. Keep fragile normalization or output-sensitive stages in FP16 at first.
4. Compare each stage against the FP16 baseline.
5. Promote more stages to INT8 only when quality remains stable.

Suggested precision order:

```text
current: FP16 PyTorch / FP16 TensorRT stagewise
step 1: INT8 internal conv-heavy stages, FP16 boundaries
step 2: mixed INT8/FP16 with normalization kept FP16
step 3: broader INT8 only if visual quality remains stable
```

Validation checks:

- Side-by-side generated face crops.
- Full-frame blended output.
- Mouth-region comparison.
- Color and skin texture stability.
- Flicker across consecutive frames.
- Edge blending around the mask.
- End-to-end HLS/WebRTC playback.

Suggested metrics:

```text
VAE isolated latency
end-to-end fps
mean absolute pixel error on generated crop
SSIM or perceptual similarity if available
manual visual review
```

Quality gate:

- No obvious mouth artifacts.
- No obvious color shift.
- No frame-to-frame flicker introduced by quantization.
- No unstable mask boundary artifacts.

Performance gate:

- VAE decode should improve meaningfully beyond the current FP16 TensorRT
  stagewise result.
- End-to-end fps should improve, not just isolated VAE fps.

## Phase 3: UNet TensorRT Or Mixed INT8/FP16

The UNet is the second major target.

Why:

- It is the main audio-conditioned latent generator.
- It runs per generated batch.
- After VAE decode improves, UNet may become the next dominant GPU cost.

Recommended strategy:

1. First attempt stable TensorRT/compiled FP16 UNet if current runtime supports
   it reliably.
2. If FP16 compilation is stable, add INT8 calibration for selected blocks.
3. Keep attention, normalization, and sensitive output areas in FP16 until
   proven safe.
4. Validate predicted latents and final generated video.

Calibration inputs:

```text
latent_batch
audio_feature_batch
timesteps
```

Validation outputs:

```text
UNet predicted latents
VAE decoded face crops
final blended video frames
```

Risk areas:

- Audio-lip alignment.
- Mouth openness.
- Teeth and tongue detail.
- Jitter across frames.
- Identity preservation.
- Latent distribution shift that only becomes visible after VAE decode.

Quality gate:

- Lip-sync must look equivalent to baseline.
- Mouth shapes must remain stable across fast phoneme changes.
- No increased jitter in the mouth region.
- No systematic identity drift.

Performance gate:

- UNet isolated latency improves.
- End-to-end generation improves after accounting for scheduler and VAE time.
- No new first-request compilation penalty in production paths.

## Phase 4: Whisper Quantization Only If Needed

Whisper should not be the first quantization target unless profiling shows audio
preparation is limiting throughput or first-response latency.

Useful when:

- Many short audio requests arrive concurrently.
- First-token or first-segment latency matters more than steady-state fps.
- Audio preprocessing dominates queue time.
- GPU time is available but audio prep is blocking job scheduling.

Risks:

- Worse phoneme representation.
- Degraded speech timing.
- Subtle lip-sync errors downstream.

Recommendation:

- Leave Whisper FP16 unless the baseline proves it is a bottleneck.
- If optimized, compare Whisper feature outputs and final video sync, not only
  Whisper encoder speed.

## Phase 5: Non-model Bottlenecks

Quantization will not fix every bottleneck. Once UNet and VAE are faster, these
areas may become dominant:

- CPU composition.
- Generated crop resize and blend.
- CPU/GPU tensor copies.
- ffmpeg encode and mux.
- HLS segment creation.
- WebRTC frame pacing.
- Scheduler queueing.

Potential non-quantization work:

- Move composition to GPU if CPU blend becomes dominant.
- Reduce unnecessary CPU/GPU transfers.
- Keep tensors in GPU memory until absolutely necessary.
- Batch compose work more efficiently.
- Tune ffmpeg settings for latency and throughput.
- Ensure NVENC is used when available and beneficial.
- Avoid per-request cold starts and repeated avatar loads.

These should be tracked separately from quantization experiments so the impact
of each change is clear.

## Acceptance Criteria

An optimization should only be considered successful if it passes both speed and
quality gates.

Speed criteria:

- Isolated target model latency improves.
- End-to-end generation fps improves.
- First playable output latency does not regress.
- Throughput under concurrent sessions improves or remains stable.
- No major new memory pressure or engine build delay appears during live use.

Quality criteria:

- Lip-sync remains comparable to FP16 baseline.
- No obvious visual artifacts.
- No new flicker.
- No systematic color shift.
- No identity degradation.
- No mask-edge instability.

Operational criteria:

- Engines are warmed before live traffic.
- Batch sizes match scheduler behavior.
- Fallback to FP16/PyTorch remains available.
- Logs identify which backend and precision were used.
- A bad quantized engine can be disabled quickly through config.

## Recommended Experiment Order

1. Re-run current FP16 PyTorch and TensorRT stagewise baselines.
2. Add calibration capture for VAE `pred_latents`.
3. Build a mixed INT8/FP16 VAE decoder experiment.
4. Compare VAE output against FP16 baseline.
5. Measure end-to-end fps and first-output latency.
6. If VAE quality is stable, expand INT8 coverage stage by stage.
7. Profile again to see whether UNet is now the main GPU bottleneck.
8. Attempt stable FP16 UNet TensorRT/compiled execution.
9. Add UNet INT8 only after FP16 UNet backend is stable.
10. Revisit Whisper only if audio prep remains material.

## Rollback Plan

Every quantization experiment should have a fast rollback path.

Recommended runtime switches:

```text
VAE_BACKEND=pyTorch_fp16
VAE_BACKEND=trt_stagewise_fp16
VAE_BACKEND=trt_stagewise_int8_mixed

UNET_BACKEND=pyTorch_fp16
UNET_BACKEND=trt_fp16
UNET_BACKEND=trt_int8_mixed
```

The exact variable names can follow the existing project naming style, but the
principle is important: precision and backend should be configurable without
code edits.

## Final Recommendation

The best next optimization is not full-pipeline quantization. It is targeted
mixed precision:

1. VAE decoder stagewise TensorRT INT8/FP16 first.
2. UNet TensorRT or mixed INT8/FP16 second.
3. Whisper only if audio preparation is measured as a bottleneck.
4. Composition and encoding handled as separate pipeline optimization tracks.

This gives the highest chance of real throughput improvement without damaging
the visual quality that makes the generated talking head usable.

