# Next Bottleneck Plan: Same-Quality VAE Late-Block Optimization

Date: 2026-06-11

## Executive Summary

The next bottleneck to attack is the `256 x 256` VAE decoder late path,
specifically:

1. `decoder_up_block_3`
2. `decoder_up_block_2`

The lower-resolution ROI experiment proved that reducing spatial decoder work
can move throughput, but `224 x 224` was visibly blurry and only improved the
closest RTX 3090 VAE-only `256` baseline by about `10-12%`. So the next
production-quality optimization should keep the generated ROI at `256 x 256`
and reduce the cost of the high-resolution VAE decoder blocks directly.

Expected gain range:

| Scenario | Likely aggregate gain | RTX 3090 optimized-path estimate | RTX 6000 Ada optimized-path estimate | Confidence |
| --- | ---: | ---: | ---: | --- |
| Conservative `decoder_up_block_3` cleanup | `+5-9%` | `75-78 fps` | `118-124 fps` | Medium |
| Target late-block mixed precision/fusion | `+12-21%` | `81-87 fps` | `126-138 fps` | Medium-low |
| Aggressive block 2 + 3 rewrite/partition | `+25-33%` | `90-96 fps` | `140-151 fps` | Low |

Capacity read:

- On RTX 3090, a real `+12-21%` model-path improvement could make `4`
  concurrent `20 fps` streams plausible, but only if tail intervals also come
  down. Aggregate FPS alone is not enough.
- On RTX 6000 Ada, aggregate throughput is already above `100 fps`, but strict
  capacity is tail-latency limited. A VAE late-block win could make `5` smooth
  streams plausible, but this must be proven with `C5/C6` WebRTC ramps.
- The biggest risk is visual quality. Whole-stage INT8 for
  `decoder_up_block_3` already failed quality review, so the next attempt must
  be finer-grained than flipping the entire block to INT8.

## Evidence

### Current Throughput Baselines

Recent validated serving profiles:

| Hardware / profile | Strict smooth 20 fps capacity | Saturated aggregate plateau | Notes |
| --- | ---: | ---: | --- |
| RTX 3090, historical `256`, VAE INT8 + PyTorch UNet | about `3` streams | about `65-67 fps` | closest VAE-only comparison |
| RTX 3090, historical `256`, VAE INT8 + TRT UNet split8 | about `3` streams | about `71-72 fps` | strongest local optimized-path reference |
| RTX 3090, current `224`, VAE INT8 + PyTorch UNet | `3` streams | about `72-74 fps` | blurry; reverted |
| RTX 5000 Ada, `256`, VAE INT8 + PyTorch UNet | `3` streams | about `70-72 fps` | no validated local TRT UNet artifact |
| RTX 6000 Ada, `256`, VAE INT8 + PyTorch UNet | `4` streams | about `108-111 fps` | current Ada VAE-only baseline |
| RTX 6000 Ada, `256`, VAE INT8 + TRT UNet split8 | `4` streams | about `112-114 fps` | current Ada optimized baseline |

The `224` ROI run is useful because it isolates the value of reducing VAE
spatial work:

| Concurrency | Historical 3090 `256` VAE-only | Current 3090 `224` VAE-only | Change |
| ---: | ---: | ---: | ---: |
| 4 | `66.7 fps` | `74.1 fps` | `+11.1%` |
| 6 | `65.9 fps` | `73.2 fps` | `+11.1%` |
| 8 | `65.0 fps` | `72.7 fps` | `+11.8%` |

That gain is real, but the blur makes it unsuitable for the production quality
path.

### Current Model-Side Timing

The most useful current local timing reference is the optimized RTX 3090
`256 x 256` path with VAE INT8 plus TRT UNet split8:

| Metric | Mean |
| --- | ---: |
| `avg_gpu_batch` | `0.185s` |
| `avg_unet` | `0.051s` |
| `avg_vae` | `0.132s` |
| `avg_compose` | `0.061s` |

VAE decode is about `71%` of the measured model-side GPU batch time in this
reference. UNet has already been improved by the split8 TRT runtime, so VAE is
again the biggest model-side stage.

The newer `224` run showed the same direction even without TRT UNet:

| Metric | Mean |
| --- | ---: |
| `avg_gpu_batch` | `0.1863s` |
| `avg_unet` | `0.0624s` |
| `avg_vae` | `0.1220s` |
| `avg_compose` | `0.0692s` |

The exact timings differ by run/backend, but the conclusion is stable: VAE is
still the largest model-side slice.

### VAE Stage Breakdown

Stage-level timing from the five-stage VAE INT8 path plus TRT UNet split8:

| Stage | Avg time | Share of VAE stage timing |
| --- | ---: | ---: |
| `decoder_pre` | `0.0004s` | `0.3%` |
| `decoder_mid_block` | `0.0035s` | `2.8%` |
| `decoder_up_block_0` | `0.0051s` | `4.1%` |
| `decoder_up_block_1` | `0.0193s` | `15.6%` |
| `decoder_up_block_2` | `0.0312s` | `25.2%` |
| `decoder_up_block_3` | `0.0600s` | `48.4%` |
| `decoder_postprocess` | `0.0045s` | `3.6%` |

Combined, `decoder_up_block_2` and `decoder_up_block_3` account for about
`73.6%` of the remaining VAE stage timing. That is the target.

### Why `decoder_up_block_3` Is First

`decoder_up_block_3` is a Diffusers `UpDecoderBlock2D`. In this VAE it contains:

- three `ResnetBlock2D` children
- no upsampler
- group norm, SiLU, two `3 x 3` convolutions, and residual add per ResNet
- one shortcut convolution in the first ResNet because channels change from
  `256` to `128`

Focused batch-16 profiling showed:

| Child | Avg time |
| --- | ---: |
| `resnets.0` | `0.02798s` |
| `resnets.1` | `0.01879s` |
| `resnets.2` | `0.01880s` |
| full block | `0.06577s` |

The cost is mostly real high-resolution convolution at
`[batch, 128, 256, 256]`. It is not primarily Python overhead.

### Why Whole-Stage INT8 Is Not The Plan

Previous `decoder_up_block_3` INT8 attempts:

| Experiment | Batch | Result |
| --- | ---: | --- |
| `onnx_qdq`, `minmax` | `16` | failed during TensorRT build with GPU OOM/assertion |
| `onnx_qdq`, `minmax` | `8` | built, `mae=0.01905`, `max_abs=0.09082`; visual quality rejected |
| `onnx_qdq`, `entropy2` | `8` | built, same `mae=0.01905`, `max_abs=0.09082`; visual quality rejected |

Conclusion: do not promote whole-stage INT8 for `decoder_up_block_3`. The next
attempt should be conv-level, sub-block-level, or a different FP16 fusion/backend
strategy.

## Throughput Estimate Model

Use the optimized RTX 3090 reference as the easiest local estimate:

```text
baseline aggregate fps ~= 72 fps
baseline avg_gpu_batch ~= 0.185s
decoder_up_block_3 ~= 0.0600s
decoder_up_block_2 ~= 0.0312s
```

Approximate model:

```text
saved_time = block_time * percent_reduction
new_gpu_batch = baseline_avg_gpu_batch - saved_time
estimated_fps = baseline_fps * baseline_avg_gpu_batch / new_gpu_batch
```

This is intentionally approximate. The real WebRTC result includes scheduler
packing, GPU utilization, compose overlap, encoder behavior, and tail latency.
Still, this model is useful because it prevents unrealistic expectations.

## Estimated Gains

### Single-Target `decoder_up_block_3`

| `decoder_up_block_3` improvement | Saved time | Estimated model-turn improvement | RTX 3090 plateau estimate | RTX 6000 Ada plateau estimate |
| ---: | ---: | ---: | ---: | ---: |
| `15%` | `0.0090s` | `+5.1%` | `75-76 fps` | `118-120 fps` |
| `25%` | `0.0150s` | `+8.8%` | `78-79 fps` | `122-124 fps` |
| `40%` | `0.0240s` | `+14.9%` | `82-83 fps` | `129-131 fps` |
| `50%` | `0.0300s` | `+19.4%` | `85-86 fps` | `134-136 fps` |

Interpretation:

- `15-25%` is a reasonable first target if we only improve selected convs or
  reduce per-block overhead.
- `40-50%` probably requires a stronger backend/fusion result or a deeper
  rewrite. It should not be assumed.

### Combined `decoder_up_block_3` + `decoder_up_block_2`

| Scenario | Saved from block 3 | Saved from block 2 | Total saved | Estimated gain | RTX 3090 plateau estimate | RTX 6000 Ada plateau estimate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Conservative | `15% = 0.0090s` | `10% = 0.0031s` | `0.0121s` | `+7.0%` | `76-77 fps` | `120-122 fps` |
| Target A | `25% = 0.0150s` | `20% = 0.0062s` | `0.0212s` | `+12.9%` | `81-82 fps` | `126-129 fps` |
| Target B | `40% = 0.0240s` | `25% = 0.0078s` | `0.0318s` | `+20.8%` | `86-87 fps` | `135-138 fps` |
| Aggressive | `50% = 0.0300s` | `50% = 0.0156s` | `0.0456s` | `+32.7%` | `95-96 fps` | `148-151 fps` |

Interpretation:

- The realistic near-term goal is `Target A`: roughly `+13%` aggregate throughput
  while preserving `256 x 256` quality.
- `Target B` would be a strong win and could materially change RTX 3090 serving
  capacity.
- The aggressive case is a stretch and should not be planned into product
  capacity until measured.

## Optimization Strategy

### Phase 0: Refresh The Baseline

Goal: make sure every A/B test starts from the real optimized `256` path.

Actions:

1. Start from `256 x 256`; do not re-enable `MUSETALK_GENERATED_ROI_SIZE`.
2. Use five-stage VAE INT8:

```text
MUSETALK_VAE_BACKEND=trt_stagewise
MUSETALK_TRT_STAGEWISE_PRECISION=int8_mixed
MUSETALK_TRT_STAGEWISE_INT8_FRONTEND=onnx_qdq
MUSETALK_TRT_STAGEWISE_INT8_STAGES=decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=8,16
HLS_SCHEDULER_FIXED_BATCH_SIZES=8,16
HLS_SCHEDULER_MAX_BATCH=16
```

3. If the local workspace has a validated UNet TRT split8 artifact, enable it.
   If not, keep UNet PyTorch and document that the run is VAE-only.
4. Run a quick C4/C6/C8 WebRTC baseline before changing VAE internals.

Required log evidence:

```text
VAE decode backend active: tensorrt_stagewise_int8_mixed
UNet backend active: tensorrt_unet_multi
HLS GPU scheduler started (... fixed_batch_sizes=[8, 16] ...)
```

If the UNet TRT artifact is missing, expected evidence is:

```text
VAE decode backend active: tensorrt_stagewise_int8_mixed
UNet backend: PyTorch
```

### Phase 1: Build A Late-Block Sensitivity Map

Goal: identify which subparts of `decoder_up_block_3` can change precision or
backend without visible quality loss.

Candidate units:

- `decoder_up_block_3.resnets.0.conv1`
- `decoder_up_block_3.resnets.0.conv2`
- `decoder_up_block_3.resnets.0.conv_shortcut`
- `decoder_up_block_3.resnets.1.conv1`
- `decoder_up_block_3.resnets.1.conv2`
- `decoder_up_block_3.resnets.2.conv1`
- `decoder_up_block_3.resnets.2.conv2`

Testing order:

1. Single conv at a time.
2. Pair convs inside the same ResNet.
3. Whole `resnets.1` or `resnets.2`, because they have no channel-changing
   shortcut.
4. `resnets.0` last, because it is largest and contains the shortcut/channel
   transition.

Candidate backend treatments:

- INT8 QDQ for selected convolutions only, with group norm, activation, residual
  adds, and output conversion left FP16.
- FP16 TensorRT tactic/fusion changes for the full block.
- Split the block into smaller TensorRT sub-engines if that enables safer mixed
  precision.
- Keep output scale and residual-add boundaries conservative.

Do not:

- Quantize the entire `decoder_up_block_3` as one stage again.
- Treat VAE crop-level MAE alone as sufficient. Previous whole-stage INT8 had
  measurable error and visible quality loss.

### Phase 2: Quality Gates

Each candidate must pass numerical and visual gates before any WebRTC load test.

Numerical gates:

| Gate | Target |
| --- | ---: |
| VAE crop `mae_max` vs current safe baseline | ideally `<= 0.010`; reject near previous `0.019` |
| VAE crop `mae_mean` | no worse than the current promoted five-stage profile by more than small noise |
| Mouth-region MAE | must not regress materially versus baseline |
| Max absolute error | investigate any concentrated high-error mouth/teeth/edge region |

Visual gates:

- side-by-side decoded crop sheet
- mouth-region crop sheet
- blended full-frame contact sheet
- short lipsync video, same avatar/audio as the load tests
- close watch for lip edge harshness, teeth texture, inner-mouth shimmer, cheek
  color shifts, and blend-boundary texture changes

Required artifacts per candidate:

```text
tmp/vae_late_block_experiments/<candidate>/metrics.json
tmp/vae_late_block_experiments/<candidate>/comparison_sheet.png
tmp/vae_late_block_experiments/<candidate>/mouth_sheet.png
tmp/vae_late_block_experiments/<candidate>/sample.mp4
```

### Phase 3: Performance Gates

Only visually acceptable candidates proceed to WebRTC.

Microbench gates:

| Gate | Minimum useful result |
| --- | ---: |
| `decoder_up_block_3` isolated reduction | at least `15%` |
| Full VAE decode reduction | at least `7%` |
| No extra engine-launch overhead | sub-stage split must not erase the gain |
| Batch support | must support the warmed serving buckets, normally `8,16` |

WebRTC gates:

| Test | Purpose |
| --- | --- |
| C4/C6/C8 | compare with existing RTX 3090 and RTX 5000 docs |
| C10/C12 | stress plateau and tail behavior |
| C5/C6 on RTX 6000 Ada | check whether strict smooth capacity can move beyond `4` |

Required metrics:

- completed / failed sessions
- average live-ready time
- average segment interval
- max segment interval
- aggregate fps
- peak VRAM
- `avg_gpu_batch`
- `avg_unet`
- `avg_vae`
- `avg_compose`
- queue depth / tail interval read

### Phase 4: Move To `decoder_up_block_2`

Only start this phase after at least one `decoder_up_block_3` candidate passes
quality.

Why block 2 second:

- It is the second-largest VAE stage at about `25%` of VAE stage timing.
- It includes the expensive upsample transition into the high-resolution path.
- Its previous whole-stage INT8 quality was acceptable enough to be promoted,
  but further gains may require finer-grained work around the upsampler and
  high-resolution residual convs.

Block 2 candidate units:

- upsampler conv path
- residual convs after the upsample transition
- FP16 fusion/tactic tuning for the upsample+conv transition
- selected conv-only INT8 if the current whole-stage INT8 leaves FP16 islands or
  suboptimal tactics

Expected incremental gain after block 3:

- modest case: `+3-5%`
- target case: `+5-8%`
- aggressive case: `+10%+`

### Phase 5: Scheduling And Tail Cleanup

This is not the first optimization target, but it becomes important if block
timing improves and strict capacity still does not move.

Questions to answer:

- Is max interval still high when average interval improves?
- Are WebRTC streams waiting on a shared model turn, compose callback, or queue
  admission?
- Can UNet of one job and VAE of another overlap on separate CUDA streams, or do
  the VAE convolutions saturate the GPU enough that overlap only increases tail?
- Does smaller model-turn time reduce queue spikes naturally?

Potential work:

- scheduler fairness tuning after faster VAE
- admission control based on measured tail intervals, not only aggregate fps
- optional CUDA stream overlap experiment, gated by utilization/tail metrics

## Implementation Sketch

The safest implementation direction is to make the VAE stagewise runtime more
granular for late decoder blocks:

1. Keep the current promoted stage list unchanged for production.
2. Add an experimental late-block override profile, for example:

```text
MUSETALK_TRT_STAGEWISE_LATE_BLOCK_EXPERIMENT=up3_conv_selective_int8
MUSETALK_TRT_STAGEWISE_UP3_INT8_CONVS=resnets.1.conv1,resnets.1.conv2
```

3. Export/build a candidate engine for that profile into an isolated cache dir:

```text
models/tensorrt/vae_late_block_experiments/up3_resnets1_int8_20260611
```

4. Validate against captured real scheduler latents before enabling in API.
5. Run the same WebRTC harness only after validation passes.

The exact env names can change during implementation. The important product
constraint is that candidate artifacts remain opt-in and cannot silently replace
the proven five-stage INT8 profile.

## Acceptance Criteria

A candidate can be considered useful if it satisfies all of these:

1. Keeps `256 x 256` generated ROI.
2. Preserves visual quality on the reference avatar and at least one additional
   avatar with a different face box / mouth shape.
3. Reduces full VAE decode timing by at least `7%`.
4. Improves WebRTC aggregate FPS by at least `5%` at C4/C6/C8.
5. Does not increase peak VRAM enough to remove the `8,16` bucket profile from
   the target GPU.
6. Does not worsen max interval/tail enough to erase the practical serving gain.

A candidate should be rejected if:

- mouth/teeth detail regresses visibly
- skin texture becomes harsh or color-shifted
- block-level speed improves but full WebRTC throughput does not
- it only works at batch `8` and fails batch `16`, unless the serving profile is
  explicitly changed
- it requires lowering ROI resolution

## Recommended Next Work Order

1. Refresh the `256` optimized baseline on the current machine.
2. Re-enable stage and child timing for a short profiling-only run.
3. Build the `decoder_up_block_3` conv sensitivity harness.
4. Test one-conv and one-ResNet candidates with visual sheets.
5. Promote the first visually clean candidate to a C4/C6/C8 WebRTC A/B test.
6. If `decoder_up_block_3` cannot pass quality above `+5%` aggregate, stop that
   branch and try FP16 fusion/tactic work before more INT8.
7. After a block 3 win lands, repeat the process for `decoder_up_block_2`.

## 2026-06-11 Implementation Pass: Split Up-Block Substage INT8

Implemented an opt-in VAE stagewise runtime split for decoder up blocks:

```text
MUSETALK_TRT_STAGEWISE_SPLIT_UP_BLOCKS=3
```

When enabled, `decoder_up_block_3` is executed as individual substages:

```text
decoder_up_block_3_resnet_0
decoder_up_block_3_resnet_1
decoder_up_block_3_resnet_2
```

The production/default stage layout is unchanged unless the env var is set.
The experiment helper now also accepts:

```text
scripts/experiment_vae_decoder_int8.py --split-up-block 3
```

New helper scripts:

- `scripts/benchmark_vae_stagewise_decode.py`: same-latent isolated VAE decode
  benchmark and PyTorch comparison.
- `scripts/record_webrtc_session.py`: records a WebRTC session to MP4 for
  visual review.

### Candidate Matrix

All candidates used the existing safe five-stage INT8 profile plus one or more
`decoder_up_block_3` substages, batch `8`, `onnx_qdq`, `minmax`, and calibration
from `calibration/vae_decoder`.

Baseline for this table is the same safe five-stage INT8 path with the full
`decoder_up_block_3` left FP16.

| Candidate | Avg VAE decode | Decode FPS | Speedup vs safe5 | MAE vs PyTorch | Max abs | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| safe5, whole `up_block_3` FP16 | `0.06214s` | `128.75` | baseline | `0.00504` | `0.12866` | current safe reference |
| + `decoder_up_block_3_resnet_0` INT8 | `0.05829s` | `137.24` | `+6.59%` | `0.00723` | `0.14478` | best balance; keep testing |
| + `decoder_up_block_3_resnet_1` INT8 | `0.05982s` | `133.74` | `+3.88%` | `0.00931` | `0.12109` | smaller speedup, close to MAE gate |
| + `decoder_up_block_3_resnet_2` INT8 | `0.05994s` | `133.48` | `+3.67%` | `0.01251` | `0.13135` | reject for now; MAE too high |
| + `resnet_0` and `resnet_1` INT8 | `0.05561s` | `143.86` | `+11.74%` | `0.01125` | `0.13965` | speed is interesting, quality risk too high |

Decision from this pass:

- `decoder_up_block_3_resnet_0` is the only candidate worth carrying forward.
- `resnet_0 + resnet_1` proves there is more available speed, but it crosses the
  preferred mean-error threshold and should not be promoted without deeper
  visual work.
- `resnet_2` is not attractive: lower speedup than `resnet_0` and higher error.

### WebRTC Video Review

Generated same-resolution `256 x 256` WebRTC reference artifacts:

```text
results/vae_late_block_webrtc_compare_20260611/baseline_safe5_webrtc.mp4
results/vae_late_block_webrtc_compare_20260611/candidate_up3_resnet0_int8_webrtc.mp4
results/vae_late_block_webrtc_compare_20260611/side_by_side_baseline_vs_up3_resnet0_int8.mp4
results/vae_late_block_webrtc_compare_20260611/crop_contact_safe5_vs_up3_resnet0_int8.png
results/vae_late_block_webrtc_compare_20260611/side_by_side_frame_8s.png
```

The side-by-side video is useful for playback review, but it is not a
pixel-perfect comparison because the two WebRTC recordings start from different
idle/source offsets. The crop contact sheet is the cleaner quality artifact.

Quality read:

- The `resnet_0` candidate does not show the obvious blur seen in the `224` ROI
  experiment, because it keeps the generated ROI at `256 x 256`.
- The visible difference is much subtler: low-level texture and tonal changes,
  with risk concentrated around the eyes, lip edge, teeth/inner-mouth detail,
  and skin texture.
- The higher max absolute error means this should stay experimental until it is
  checked on more avatars and a frame-aligned comparison path.

### WebRTC Load Test

Live server settings for the candidate load test:

```text
MUSETALK_VAE_BACKEND=trt_stagewise
MUSETALK_TRT_STAGEWISE_PRECISION=int8_mixed
MUSETALK_TRT_STAGEWISE_INT8_FRONTEND=onnx_qdq
MUSETALK_TRT_STAGEWISE_INT8_STAGES=decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2,decoder_up_block_3_resnet_0
MUSETALK_TRT_STAGEWISE_SPLIT_UP_BLOCKS=3
HLS_SCHEDULER_FIXED_BATCH_SIZES=8
HLS_SCHEDULER_MAX_BATCH=8
```

Server log evidence:

```text
VAE decode backend active: tensorrt_stagewise_int8_mixed
UNet backend: PyTorch
HLS GPU scheduler started (... fixed_batch_sizes=[8] ...)
VAE decode timing backend=tensorrt_stagewise_int8_mixed calls=1000 avg_total=0.0570s max_total=0.0671s
```

The current workspace did not have a validated local TRT UNet split8 artifact,
so these WebRTC numbers are VAE INT8 plus PyTorch UNet.

| Concurrency | Completed | Failed | Avg live-ready | Avg frame interval | Approx aggregate FPS | Max frame interval | Avg GPU util | Peak VRAM |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 4 | 0 | `2.781s` | `0.056s` | `71.4` | `0.184s` | `74.65%` | `8134 MB` |
| 6 | 6 | 0 | `3.962s` | `0.083s` | `72.3` | `0.430s` | `72.15%` | `8134 MB` |
| 8 | 8 | 0 | `5.280s` | `0.113s` | `70.8` | `1.215s` | `72.15%` | `8134 MB` |

Artifacts:

```text
tmp/load_tests/load_test_webrtc_3090_up3_resnet0_int8_20_20_c4_bs8_20260611.json
tmp/load_tests/load_test_webrtc_3090_up3_resnet0_int8_20_20_c4_bs8_20260611_detailed.json
tmp/load_tests/load_test_webrtc_3090_up3_resnet0_int8_20_20_c6_c8_bs8_20260611.json
tmp/load_tests/load_test_webrtc_3090_up3_resnet0_int8_20_20_c6_c8_bs8_20260611_detailed.json
```

Load-test read:

- The isolated VAE decode improvement is real: `+6.59%` for batch-8 VAE decode.
- The WebRTC aggregate plateau stayed around `71-72 fps`, and tail intervals are
  still not smooth at `C4+`.
- This means the candidate reduces the VAE slice but does not by itself change
  practical strict-smooth capacity on this RTX 3090 run.
- The logged live path remains roughly `0.040-0.047s` UNet plus `0.057s` VAE per
  batch-8 model turn, with compose around `0.041-0.047s`.

### Current Decision

Do not promote `decoder_up_block_3_resnet_0` to the default runtime yet.

Reasons:

- Only batch `8` has been built and tested.
- The production-like `8,16` bucket profile has not been validated for this
  candidate.
- The visual review used one avatar and the WebRTC side-by-side is not
  frame-aligned.
- WebRTC throughput did not clearly move the serving plateau despite the VAE
  microbench gain.
- Max absolute error is higher than the safe five-stage reference.

Next work:

1. Build and benchmark the `resnet_0` candidate for batch `16`.
2. Re-run C4/C6/C8 with an `8,16` bucket profile if batch `16` builds cleanly.
3. Add a frame-aligned quality comparison path for WebRTC or use deterministic
   offline full-frame exports from identical generated frames.
4. Test at least one additional avatar with a different face box and mouth
   shape.
5. If quality risk remains, try a more conservative split inside `resnet_0`
   instead of quantizing the whole ResNet substage.

## Bottom Line

The next high-ROI optimization is not more ROI downscaling. It is making the
current `256 x 256` VAE decoder cheaper, starting with
`decoder_up_block_3`. The first split-substage implementation found a viable
candidate in `decoder_up_block_3_resnet_0`: it improves isolated batch-8 VAE
decode by `+6.59%` without the blur from ROI downscaling. It is not enough by
itself to move practical WebRTC capacity yet, so the next gate is batch-16 plus
better frame-aligned quality review before moving on to `decoder_up_block_2`.
