# MuseTalk Pipeline Component Analysis and Optimization Targets — RTX 4070 SUPER

Date: 2026-09-22

## Executive Summary

Every component of the live serving path was profiled in isolation on this host.
The conclusion matches the historical 3090 / RTX 6000 Ada documents on *where*
the time goes, and disagrees with them on *what to do about it*.

VAE decode is still the bottleneck: **76% of the GPU batch**. Years of work on
this stage have gone into TensorRT + INT8 quantization of the existing SD-VAE
decoder, which reached `142.5 ms -> 82.9 ms` at batch 8 (`1.72x`). That line of
attack is close to exhausted — `decoder_up_block_3` INT8 has been rejected twice
on visual quality, and batch-16 INT8 builds OOM.

The measured result here is that **replacing the decoder entirely beats
quantizing it by an order of magnitude**. A distilled tiny decoder (TAESD,
`1.2M` params vs `49.5M`) decodes the same latents in `12.4 ms` instead of
`82.9 ms`, and end-to-end throughput goes from `47.6 fps` to `209 fps` at
batch 8 — with *lower* peak VRAM.

| Config | GPU ms / batch-8 | Aggregate fps | Peak VRAM | Streams @20fps |
| --- | ---: | ---: | ---: | ---: |
| TRT-UNet + SD-VAE FP16 | `168.0` | `47.6` | `2.77 GB` | `2.4` |
| TRT-UNet + VAE INT8 (shipping)¹ | `~108` | `~74` | n/a² | `~3.7` |
| **TRT-UNet + TAESD** | **`38.3`** | **`209.0`** | **`2.09 GB`** | **`10.5`** |

¹ composed from separately measured stages; see *VRAM ceiling* below.
² the INT8 VAE and the TRT UNet could not be co-resident on this 12 GB card.

That is the `8-10 concurrent stream` target, reached on a 12 GB RTX 4070 SUPER
rather than a 24 GB 3090.

The cost is quality: TAESD's deviation inside the blended region is
`mae=0.0112` on the reference avatar `shared_indian_20260915` (160 real frames),
versus `mae=0.0050` for the shipping INT8 path. A labelled A/B video is attached
for the visual verdict:

```text
docs/musetalk_4070s_decoder_comparison_indian_20260922.mp4
```

Two further findings are independent of the TAESD decision and worth taking
regardless:

- **`40%` of every decoded frame is thrown away.** The decoder always emits
  `256 x 256`, but only rows `~104..256` are ever blended, and the result is
  then downscaled to the `~169 x 204` face bbox. Cropping the late decoder
  blocks is a `1.47x` VAE win at `mae=0.0115`, and stacks with everything else.
- **`torch.compile` was never tried and is nearly free.** `1.54x` on the VAE
  decoder at `mae=0.0001` — 50x more accurate than the shipping INT8 path, and
  it needs no calibration corpus, no engine build, and no artifact management.

## Host

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4070 SUPER, `12282 MiB`, sm_89 (Ada) |
| Driver / CUDA | `595.84` / `13.2` |
| Torch | `2.5.1+cu121` |
| TensorRT | `10.3.0` (torch_tensorrt `2.5.0`) |
| ModelOpt | `0.23.2` |
| Venv | `/workspace/.venvs/musetalk_trt_stagewise` |

Relevant to the plan: **sm_89 has native FP8 tensor cores**, which sm_86
(RTX 3090) does not. Every quantization result in the historical documents was
produced on hardware that could not do FP8. This is confirmed available here
(see *Untried lever: FP8* below) and has never been attempted on this project.

## Component Map

The live path is `api_server.py` -> `scripts/hls_gpu_scheduler.py`
(`_run_generation_batch`) -> UNet -> VAE -> compose -> encode. Measured per
batch of 8 frames:

| Stage | Where | ms @ bs8 | Share of GPU batch |
| --- | --- | ---: | ---: |
| Batch assembly (gather + pinned staging) | `hls_gpu_scheduler.py:1309-1375` | `<1` | ~1% |
| H2D copy | `hls_gpu_scheduler.py:1390-1400` | `~0.5` | ~0.5% |
| **UNet forward (TRT FP16)** | `musetalk/models/unet.py`, `scripts/trt_runtime.py` | **`25.4`** | **23%** |
| **VAE decode (INT8 5-stage)** | `musetalk/models/vae.py:209`, `trt_runtime.py` | **`82.9`** | **76%** |
| VAE postprocess (GPU->CPU uint8 NHWC BGR) | `musetalk/models/vae.py:166-182` | `0.18` | 0.2% |
| Compose / alpha blend (CPU, threaded) | `api_avatar.py:1096`, `blending.py:218` | `2.7` | overlapped |
| H.264 encode | `hls_gpu_scheduler.py:1997` | — | separate thread |

Out of the per-batch loop, once per request:

| Stage | Measured |
| --- | --- |
| Whisper mel + encoder | `0.88 s` for `8 s` of audio (~`9x` realtime) |
| Model load | `7.5 s` |
| Avatar prep | offline / cached |

### Where the VAE decode time actually goes

PyTorch FP16, batch 8, total `142.9 ms`:

| Stage | Resolution | ms | Share |
| --- | --- | ---: | ---: |
| `conv_in` | `32²` | `0.064` | 0.0% |
| `mid_block` | `32²` | `4.284` | 3.0% |
| `up_block_0` | `32² -> 64²` | `7.313` | 5.1% |
| `up_block_1` | `64² -> 128²` | `30.559` | 21.4% |
| `up_block_2` | `128² -> 256²` | `45.126` | 31.6% |
| `up_block_3` | `256²` | `51.203` | 35.8% |
| `norm_out + act + conv_out` | `256²` | `4.344` | 3.0% |

`up_block_2 + up_block_3` = `67%` of the decoder. This reproduces the
2026-06-11 3090 finding exactly. The cost is real high-resolution convolution at
`[8, 128..512, 256, 256]`, not Python overhead.

## Finding 1 — 40% of every decoded frame is discarded

The decoder always produces `256 x 256`. `compose_frame()` then does:

```python
res_frame_resized = cv2.resize(res_frame, (x2 - x1, y2 - y1))   # api_avatar.py:1124
return get_image_blending_with_plan(ori_frame, res_frame_resized, compose_plan)
```

Mapping the blend plan's `face_src_slice` and non-zero alpha back into the
decoder's output frame, across every cycle frame of two prepared avatars:

| Avatar | Frame | Face bbox | Decoded | Downscale area ratio | Rows actually blended |
| --- | --- | --- | --- | ---: | --- |
| `codex_smoke` | `512x512` | `169x204` | `256x256` | `0.528` | `110..256` (`55.7%`) |
| `shared_indian_20260915` | `672x384` | `155x228` | `256x256` | `0.539` | `111..256` (`55.5%`) |

Both waste the same way, for a structural reason: MuseTalk only regenerates the
lower face, so the blend mask never covers the top of the ROI. The decoder
spends `~40%` of `up_block_1/2/3` on rows that are discarded, and then the
survivors are downscaled to roughly half the decoded pixel count.

Cropping the feature map before the late blocks (latent-aligned to rows
`104..256`, plus a convolution halo) measured:

| Variant | ms @ bs8 | Speedup | mae_roi | max_abs_roi |
| --- | ---: | ---: | ---: | ---: |
| full decode | `142.5` | `1.00x` | — | — |
| crop after `up_block_1`, halo 8 | `96.6` | `1.47x` | `0.0115` | `0.0322` |
| crop after `up_block_2`, halo 8 | `101.5` | `1.40x` | `0.0035` | `0.0308` |
| crop after `up_block_3`, halo 8 | `118.5` | `1.20x` | `0.0041` | `0.0459` |
| crop after `up_block_1` + `torch.compile` | `77.6` | `1.84x` | `0.0115` | `0.0327` |

The residual error does **not** shrink as the halo grows (`8 -> 16 -> 24 -> 40 px`
all land near the same value), which identifies the source: **GroupNorm**
statistics are computed over the spatial extent, so cropping shifts them. It is
a global tone shift, not an edge artifact — there is a single tile, so there are
no seams. Worth noting that `max_abs = 0.032` is **4x tighter than the INT8 path
already in production** (`0.140`).

## Finding 2 — `torch.compile` was never tried, and it beats several shipping optimizations

Lossless-ish levers, VAE decoder, batch 8:

| Lever | ms | Speedup | mae | max_abs |
| --- | ---: | ---: | ---: | ---: |
| NCHW FP16 (baseline) | `142.5` | `1.00x` | — | — |
| `cudnn.benchmark=True` | `142.6` | `1.00x` | `0` | `0` |
| `channels_last` (NHWC) | `149.4` | `0.95x` | `0.00013` | `0.0017` |
| **`torch.compile(mode="max-autotune")`** | **`92.3`** | **`1.54x`** | `0.00010` | `0.0010` |

Two things to record:

- **`channels_last` is a regression here** (`0.95x`). This is worth writing down
  because it is the standard first move for FP16 convnets. The diffusers VAE
  decoder is GroupNorm-heavy, and eager GroupNorm in NHWC costs more than the
  convolutions gain.
- **`torch.compile` gives `1.54x` for one line of code**, at `mae=0.0001` —
  roughly 50x more accurate than the shipping INT8 decoder, with no calibration
  corpus, no ONNX/QDQ frontend, no `.plan` cache, and no per-GPU artifact
  rebuild. Inductor fuses GroupNorm+SiLU+residual, which is precisely what the
  memory-bound `256²` late blocks need.

Same lever on the UNet, batch 8:

| Backend | ms | Speedup | mae | max_abs |
| --- | ---: | ---: | ---: | ---: |
| PyTorch FP16 | `40.4` | `1.00x` | — | — |
| `channels_last` | `36.8` | `1.10x` | `0.00025` | `0.0024` |
| `torch.compile` | `35.3` | `1.15x` | `0.00029` | `0.0039` |
| `channels_last` + `torch.compile` | `30.5` | `1.33x` | `0.00028` | `0.0044` |
| **TRT FP16 bs8 (shipping)** | **`25.5`** | **`1.58x`** | `0.0014` | `0.0117` |

The existing TRT UNet artifact is the best UNet option and should stay. Note it
is `849.9M` params — larger than SD 1.5's UNet.

## Finding 3 — Replacing the decoder beats quantizing it

Full comparison at batch 8. Error is measured **only inside the region
`compose_frame()` actually pastes**, using real post-UNet latents produced from
`data/audio/yongen.wav` through the real Whisper and UNet path — not `randn`.

| VAE decode backend | ms | vs FP16 | vs INT8 | mae_roi | max_abs_roi | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| PyTorch FP16 | `142.5` | `1.00x` | `0.58x` | — | — | reference |
| `torch.compile` FP16 | `92.3` | `1.54x` | `0.90x` | `0.00010` | `0.0010` | **untried** |
| late-crop @`up_block_1` | `96.6` | `1.47x` | `0.86x` | `0.0115` | `0.0322` | **new** |
| **INT8 5-stage** | **`82.9`** | **`1.72x`** | **`1.00x`** | `0.0050` | `0.140` | **shipping** |
| late-crop + compile | `77.6` | `1.84x` | `1.07x` | `0.0115` | `0.0327` | **new** |
| **TAESD** | **`12.4`** | **`11.5x`** | **`6.7x`** | `0.0157` | `0.177` | **new** |
| TAESD + compile | `6.8` | `20.8x` | `12.2x` | `0.0157` | `0.177` | **new** |

TAESD (`madebyollin/taesd`, `AutoencoderTiny`) is a distilled decoder for the
same SD 1.x latent space that `sd-vae-ft-mse` uses, so it is a drop-in for
MuseTalk's latents. Correct convention, established by sweep:

```python
img = (taesd.decode(z).sample / 2 + 0.5).clamp(0, 1)   # z = post-UNet latent, unscaled
```

Getting that wrong costs `16x` in reported error (`mae 0.0157 -> 0.255`), which
is worth flagging because it looks like catastrophic quality loss rather than a
scaling bug.

### Measured end-to-end, 200 frames of real audio

| Backend | UNet | VAE | Post | Compose | GPU/batch | End-to-end fps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TRT-UNet + SD-VAE FP16 | `25.1` | `146.0` | `0.8` | `3.5` | `171.9` | `45.4` |
| TRT-UNet + TAESD | `24.4` | `13.0` | `0.3` | `3.2` | `37.6` | `194.3` |

### Capacity and VRAM

| Config | bs | UNet ms | VAE ms | Total | fps | Peak VRAM | Streams @20fps | @25fps |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| TRT-UNet + SD-VAE FP16 | 8 | `25.4` | `142.7` | `168.0` | `47.6` | `2.77 GB` | `2.4` | `1.9` |
| | 16 | `52.1` | `284.5` | `336.5` | `47.5` | `3.77 GB` | `2.4` | `1.9` |
| | 32 | `104.8` | `570.6` | `675.4` | `47.4` | `5.77 GB` | `2.4` | `1.9` |
| **TRT-UNet + TAESD** | **8** | `25.9` | `12.4` | **`38.3`** | **`209.0`** | **`2.09 GB`** | **`10.5`** | **`8.4`** |
| | 16 | `52.4` | `26.9` | `79.3` | `201.9` | `2.40 GB` | `10.1` | `8.1` |
| | 32 | `105.2` | `56.0` | `161.2` | `198.5` | `3.02 GB` | `9.9` | `7.9` |

TAESD also *reduces* VRAM, and reduces how fast VRAM grows with batch
(`+0.31 GB` per 8 frames vs `+1.00 GB`). The SD-VAE decoder's `256² x 512`
activations are the memory hog, not the weights. On a 12 GB card this matters as
much as the speed.

These are model-path numbers with no scheduler, encoder, WebRTC pacing or tail
latency. Historically the real WebRTC capacity has landed well below the
aggregate-fps implied capacity, so `10.5` should be read as headroom, not a
promise.

## Finding 4 — VRAM ceiling on 12 GB, and the UNet batch-16 gap

Two constraints specific to this host:

1. **The TRT UNet engine and the INT8 stagewise VAE could not be built
   co-resident.** Loading the TRT UNet and then building the INT8 VAE stages in
   one process fails with `Error Code 2: OutOfMemory` during TensorRT context
   creation. The 2026-07-04 document records the same class of failure on a
   24 GB 3090 when loading both bs8 and bs16 UNet engines. Choosing TAESD
   removes the entire INT8 VAE engine set and sidesteps this.
2. **The UNet TRT artifact is static batch-8 and split for larger batches**, so
   there is no amortization: `bs16` costs exactly `2x bs8` (`52.4` vs `25.9 ms`).
   PyTorch shows real amortization (`5.07 ms/frame` at bs8 -> `4.44` at bs16),
   and the 2026-07-04 3090 run measured a native bs16 engine at `44.9 ms`
   vs `51.8 ms` for split-8. A native bs16 engine is worth roughly `13%` of
   UNet time — but only matters once the VAE stops dominating.

## Untried lever: FP8 on Ada

Confirmed available on this host and never attempted on this project:

| Check | Result |
| --- | --- |
| `torch.float8_e4m3fn` cast | OK |
| `torch._scaled_mm` FP8 GEMM | OK |
| `trt.BuilderFlag.FP8` (TensorRT 10.3) | settable |
| `modelopt.torch.quantization.FP8_DEFAULT_CFG` | available |

Why this matters: every INT8 rejection in the history was a *dynamic range*
failure. `decoder_up_block_3` INT8 was rejected twice on visible colour/texture
shift at `mae=0.019`. FP8 E4M3 keeps an exponent, so it handles the wide
activation range of the high-resolution decoder blocks far better than INT8's
uniform scale. If the SD-VAE decoder is kept for a quality tier, FP8 is the
right next attempt for `up_block_2/3` — not another INT8 calibration algorithm.

## Ranked Optimization Targets

| # | Target | Lever | Measured / expected | Quality risk | Effort |
| --- | --- | --- | --- | --- | --- |
| 1 | VAE decode (76%) | TAESD tiny decoder | `82.9 -> 12.4 ms`, `47.6 -> 209 fps` | **medium-high** (`mae 0.0157`) | low — drop-in |
| 2 | VAE decode | `torch.compile` max-autotune | `1.54x`, stacks with 1 and 3 | negligible (`mae 0.0001`) | trivial |
| 3 | VAE decode | late-block spatial crop | `1.47x`, stacks | low (`max_abs 0.032`) | medium |
| 4 | UNet (23%, 65% after 1) | FP8 TRT on sm_89 | untested, est. `1.3-1.8x` | unknown | medium-high |
| 5 | UNet | native bs16 TRT engine | `~13%` of UNet | low (gate exists) | medium |
| 6 | VAE (quality tier) | FP8 on `up_block_2/3` | untested | unknown | medium-high |
| 7 | Compose / postprocess | — | `2.7 ms` and `0.18 ms` | — | **do not bother** |

Items 1, 2 and 3 are independent and multiply. Item 7 is listed to close it out:
the 2026-07-03 ROI-shrink work already took compose to `0.33 ms/frame`, and the
GPU->CPU handoff is `0.18 ms`. Together they are under `2%` of the batch. The
CPU-side hot path is done; there is nothing left there worth taking.

## Recommended Next Round

The decision that needs a human is **item 1**, because it trades measurable
quality for a `4.4x` throughput change. Everything else is gated on it, since
optimizing the SD-VAE decoder is wasted work if it is being replaced.

1. **Watch `docs/musetalk_4070s_decoder_comparison_20260922.mp4`** and rule on
   TAESD. Full-frame and mouth-zoom, labelled, same audio and avatar, 200 frames.
2. If TAESD is acceptable — land it behind `MUSETALK_VAE_DECODER=taesd`, stack
   `torch.compile` (item 2), then re-run the real WebRTC C4/C8/C10 harness,
   which is the number that actually governs capacity.
3. If TAESD is not acceptable as-is — **fine-tune it before discarding it.**
   Distilling TAESD's decoder against the SD-VAE decoder on this project's own
   face-crop distribution is a few GPU-hours and directly targets the gap. The
   generic checkpoint is trained on all of LAION; MuseTalk only ever decodes
   lower-face crops, which is a far narrower problem.
4. If the quality tier must stay on SD-VAE — take items 2 + 3 (`1.84x` combined,
   `max_abs 0.033`, already better than shipping INT8 on both axes), then FP8.

## Reproduction

Probes written for this analysis (scratchpad, not repo state):

| Script | Produces |
| --- | --- |
| `profile_components.py` | full component map, VAE per-submodule breakdown |
| `probe_vae_levers.py` | memory format / cudnn / compile levers, ROI probe |
| `roi_waste.py` | decoded-vs-displayed geometry per avatar |
| `probe_crop_decode.py` | latent-crop and late-crop variants |
| `taesd_convention.py` | TAESD input-scaling / output-range sweep |
| `probe_decoders_real.py` | all decoders on real post-UNet latents |
| `probe_unet.py`, `probe_unet2.py` | UNet backends, FP8 capability |
| `e2e_compare.py` | 200-frame end-to-end runs + videos |
| `capacity.py` | throughput vs batch size and peak VRAM |

Existing in-repo benchmark used for the shipping INT8 number:

```bash
python scripts/benchmark_vae_stagewise_decode.py \
  --label int8_safe5_bs8_4070s --batch-size 8 --iters 40 \
  --calibration-dir ./calibration/vae_decoder \
  --cache-dir ./models/tensorrt/stagewise_int8_onnx_qdq_cache \
  --int8-stages decoder_pre,decoder_mid_block,decoder_up_block_0,decoder_up_block_1,decoder_up_block_2 \
  --output-json /tmp/vae_int8_bs8_4070s.json
# -> avg_decode_s = 0.0829, decode_fps = 96.5, mae = 0.0050, max_abs = 0.140
```

---

# Implementation round 1 — 2026-09-22

Two levers landed. They are orthogonal: one raises how many streams a GPU can
serve, the other raises how many avatars can stay resident to serve them.

| Lever | Metric moved | Before | After | Factor | Quality |
| --- | --- | ---: | ---: | ---: | --- |
| TAESD decoder backend | VAE decode @bs8 | `138.4 ms` | `6.83 ms` | **`20.2x`** | `mae_roi 0.0157` |
| | end-to-end aggregate | `47.6 fps` | `209.0 fps` | **`4.4x`** | (see video) |
| | peak VRAM @bs8 | `2.77 GB` | `2.09 GB` | `0.75x` | — |
| Avatar cycle dedup | host RAM per avatar | `673.9 MB` | `339.4 MB` | **`1.99x`** | **bit-identical** |
| | avatar load time | `1.4 s` | `1.2 s` | `1.17x` | — |

## 1. TAESD decoder backend

New module `scripts/vae_fast_decoder.py`, dispatched from the existing single
integration point in `scripts/trt_runtime.py::load_vae_trt_decoder`, so every
caller (`avatar_manager_parallel.py`, `benchmark_pipeline.py`,
`validate_vae_backend.py`) picks it up unchanged.

```bash
MUSETALK_VAE_BACKEND=taesd     # the whole switch; unset to roll back
```

| Variable | Default | Purpose |
| --- | --- | --- |
| `MUSETALK_VAE_BACKEND=taesd` | — | select the backend |
| `MUSETALK_TAESD_COMPILE` | `1` | `torch.compile` the decoder (`138 -> 6.8 ms`; eager is `~12.4 ms`) |
| `MUSETALK_TAESD_COMPILE_MODE` | `max-autotune` | inductor mode |
| `MUSETALK_TAESD_WARMUP_BATCHES` | `8` | batches to compile/warm at load |
| `MUSETALK_TAESD_LOCAL_DIR` | vendored | weights path |
| `MUSETALK_TAESD_TIMING` | `0` | periodic decode timing log |

Weights are vendored at `models/taesd/` (**4.7 MB**, vs `320 MB` for `sd-vae`),
so a cold start needs no network. Load is `0.1 s`; warmup with compile is `3.8 s`.

Validated through the real path (`load_vae_trt_decoder` -> `set_decode_backend`
-> `decode_latents_tensor` -> `decode_latents`), on real post-UNet latents from
`data/audio/yongen.wav`:

- output contract: `(8, 3, 256, 256)` fp16, range `[0.005, 1.000]`, and the
  NumPy path returns `(8, 256, 256, 3)` uint8 BGR — all asserted.
- deviation vs SD-VAE FP16 inside the blended ROI: `mae 0.0157`, `max_abs 0.177`.

**The one thing to get right:** TAESD consumes the *scaled* latent the UNet
emits and returns `[-1, 1]`; it does **not** want the `1/scaling_factor`
division the `AutoencoderKL` path applies. Getting this wrong costs ~16x in
reported error (`mae 0.0157 -> 0.255`) and looks like catastrophic quality loss
rather than a scaling bug. This is why `_raw_decode` carries a comment and
ignores the `scaling_factor` argument explicitly rather than silently.

### Quality — reference avatar `shared_indian_20260915`

Quality is a real trade, not free. Measured against the SD-VAE FP16 reference on
**160 real frames** of the production avatar, using the rows the blend mask
actually pastes (`111..256` for this avatar):

| Decoder | mae (blended ROI) | max_abs (blended ROI) | VAE @bs8 |
| --- | ---: | ---: | ---: |
| shipping INT8 5-stage | `0.0050` | `0.140` | `82.9 ms` |
| **TAESD** | **`0.0112`** | **`0.356`** | **`6.9 ms`** |

The mae is the comparable figure; `max_abs` is over 160 frames here versus 8 for
the INT8 number, so it samples ~20x more opportunities for a worst pixel.

End-to-end on this avatar, same audio, same UNet (TRT FP16 bs8):

| | UNet | VAE | compose | e2e |
| --- | ---: | ---: | ---: | ---: |
| SD-VAE FP16 | `25.87 ms` | `141.11 ms` | `3.09 ms` | `46.7 fps` |
| **TAESD** | `23.73 ms` | **`6.90 ms`** | `2.10 ms` | **`241.1 fps`** |

`5.16x` — higher than the `4.4x` on `codex_smoke` because this avatar's frame is
`384x672` rather than `512x512`, so compose is cheaper and the VAE dominates more.

**Labelled A/B for the visual verdict** (full frame + nearest-neighbour mouth
zoom, so the zoom shows real decoder output rather than interpolation):

```text
docs/musetalk_4070s_decoder_comparison_indian_20260922.mp4
```

The mouth shape and teeth track closely. The visible differences are that
TAESD's lips read slightly more saturated and the stubble/skin microtexture is
marginally softer.

An earlier A/B on `codex_smoke` is retained at
`docs/musetalk_4070s_decoder_comparison_20260922.mp4`
(`mae 0.0157 / max_abs 0.177` over 8 frames, `47.6 -> 209 fps`).

## 2. Avatar cycle deduplication

A prepared avatar cycle is forward+reverse, so `cycle[i]` and `cycle[N-1-i]` are
byte-identical files on disk. The production load path
(`_load_existing_materials`) read all `N` and decoded all `N`. Verified by md5
across every cycle position:

| Avatar | cycle | unique frames | unique masks |
| --- | ---: | ---: | ---: |
| `codex_smoke` | 600 | **300** | **300** |
| `shared_indian_20260915` | 50 | **1** | **1** |

The second case is a still-portrait avatar holding 50 copies of one frame, one
mask and one compose plan.

`scripts/api_avatar.py` now hashes the encoded bytes *before* decoding
(`_read_imgs_dedup`) and decodes each distinct image once, returning a
full-length cycle over shared buffers. Compose plans — the two alpha arrays, the
largest per-position allocation after the frame — are shared across positions
with the same `(shape, bbox, mask, crop_box)`.

```bash
MUSETALK_AVATAR_DEDUP_CYCLE=0   # rollback; default is 1
```

Hashing before decoding matters. Deduplicating *after* load only returned
`91 MB` of the predicted `333 MB`, because the peak already allocated every
buffer and glibc retains the freed arena. Skipping the decode moves the real
number:

| | marginal RSS per avatar | 3 avatars resident |
| --- | ---: | ---: |
| `MUSETALK_AVATAR_DEDUP_CYCLE=0` | `673.9 MB` | `2021.8 MB` |
| `MUSETALK_AVATAR_DEDUP_CYCLE=1` | **`339.4 MB`** | **`1018.3 MB`** |

**This is lossless, and proven so** — composed frames are bit-identical at
positions 0/1/299/300/599 (including the mirror boundary and the endpoint), max
abs diff `0`. Sharing is safe because `compose_frame()` copies before blending;
a mutation-safety assertion confirms the shared source buffer is untouched after
a compose.

### Residency accounting was over-reporting

`_numpy_sequence_nbytes` summed `nbytes` per element with no dedup by storage,
and `_compose_plan_sequence_nbytes` counted `alpha` but **not** `alpha_u8` —
which is the array the default fixed-point blend path actually uses. Both are
fixed. This matters beyond bookkeeping: `AvatarCache` budgets admissions from
`estimate_memory_usage_bytes()`, so an inflated figure caps resident avatars
below what the RAM holds.

## Capacity

Model-path only — no scheduler, encoder, WebRTC pacing or tail latency. Real
WebRTC capacity has historically landed well below implied aggregate fps, so
read the stream counts as headroom, not a promise.

| | before | after |
| --- | ---: | ---: |
| aggregate fps @bs8 | `47.6` | `209.0` |
| concurrent streams @20 fps | `2.4` | `10.5` |
| concurrent streams @25 fps | `1.9` | `8.4` |
| peak VRAM @bs8 | `2.77 GB` | `2.09 GB` |
| host RAM per resident avatar | `673.9 MB` | `339.4 MB` |
| resident avatars in 24 GB host | `~36` | **`~72`** |
| resident avatars in 32 GB host | `~48` | **`~96`** |

The binding constraint for *avatars* is host RAM, not VRAM: an avatar's
GPU-resident part is only its latent cycle (`9.4 MB` for `codex_smoke`), against
`339 MB` of host frames/masks/plans. VRAM would hold hundreds.

## Not done

- **Late-block spatial crop** (`1.47x`, `max_abs 0.032`). Superseded for the
  TAESD path — it optimises the SD-VAE decoder that TAESD replaces. Still the
  right lever if a quality tier keeps SD-VAE.
- **FP8 on sm_89.** Confirmed available (`torch._scaled_mm`, TRT 10.3
  `BuilderFlag.FP8`, ModelOpt `FP8_DEFAULT_CFG`). Now the biggest remaining
  lever, because after TAESD the UNet is ~65% of the batch.
- **Native bs16 UNet TRT engine.** The bs8 artifact is split for larger batches
  and does not amortise (`bs16` costs exactly `2x bs8`); a native engine is
  worth ~13% of UNet time.
- **Frame storage on disk.** Dedup halves resident frames; storing the cycle as
  `N/2` files would halve the avatar's disk footprint too, and is a prep-side
  change.

## Verification commands

All harnesses are retained under `docs/fps_comparisons/4070s_20260922/`.

```bash
V=/workspace/.venvs/musetalk_trt_stagewise/bin/python
D=docs/fps_comparisons/4070s_20260922

# TAESD: loads through load_vae_trt_decoder, asserts the output contract,
# measures deviation vs SD-VAE on real post-UNet latents, and times both.
$V $D/validate_taesd_backend.py
#   -> backend loaded: name=taesd
#   -> blended ROI: mae=0.01565  max_abs=0.1772
#   -> speed @bs8: SD-VAE 138.38 ms -> TAESD 6.83 ms  (20.2x)

# Avatar dedup: marginal RSS with 3 avatars resident, both ways.
for d in 0 1; do MUSETALK_AVATAR_DEDUP_CYCLE=$d $V $D/avatar_dedup_rss.py; done
#   -> dedup=0  marginal_per_avatar=673.9 MB
#   -> dedup=1  marginal_per_avatar=339.4 MB

# Avatar dedup correctness: composed frames bit-identical + mutation safety.
$V $D/verify_dedup_correctness.py
#   -> composed-frame max abs diff ...: [0, 0, 0, 0, 0]
#   -> PASS: composed frames bit-identical

# Per-avatar residency audit (what is duplicated, and by how much).
$V $D/avatar_residency_audit.py
```

---

# Cross-avatar validation and why this works

## Measured across every prepared avatar

Same audio, same UNet (TRT FP16 bs8), 200 frames each. Quality is measured over
160 real frames per avatar, restricted to the rows the blend mask actually
pastes.

| Avatar | Frame | Cycle | Motion | VAE before | VAE after | e2e before | e2e after | Speedup | mae (ROI) |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `shared_indian_20260915` | `384x672` | 50 | static | `141.11` | `6.90` | `46.7` | `241.1` | **`5.16x`** | `0.0112` |
| `ltx23_direct_recheck_seed190` | `480x832` | 480 | moving | `141.88` | `6.86` | `44.2` | `224.5` | **`5.08x`** | `0.0113` |
| `ltx23_native_flf_seed191` | `512x832` | 482 | moving | `142.07` | `6.85` | `44.4` | `222.1` | **`5.00x`** | — |
| `ltx23_closeup_smile_direct` | `480x832` | 290 | moving | `142.12` | `6.87` | `45.5` | `225.1` | **`4.95x`** | `0.0125` |
| `codex_smoke` | `512x512` | 600 | moving | `~142` | `~6.9` | `47.6` | `209.0` | `4.40x` | `0.0157`* |

\* measured over 8 frames, not 160; the others are directly comparable to each other.

Moving-avatar A/B video (480-frame cycle, face bbox travels `41x51 px`):

```text
docs/musetalk_4070s_decoder_comparison_moving_20260922.mp4
```

## Why the VAE decode was the bottleneck

The SD-VAE decoder upsamples `32x32` latents to `256x256` RGB in four stages.
Convolution cost scales with `spatial_area x in_ch x out_ch x k²`, and this
decoder keeps **512 channels until 128², and 128–256 channels at 256²**. The
result is that essentially all the work happens in the last two blocks:

| Block | Resolution | Channels | ms @bs8 | Share |
| --- | --- | ---: | ---: | ---: |
| `mid_block` + `up_block_0` | `32²` | 512 | `11.6` | 8% |
| `up_block_1` | `64² -> 128²` | 512 | `30.6` | 21% |
| `up_block_2` | `128² -> 256²` | 512 -> 256 | `45.1` | **32%** |
| `up_block_3` | `256²` | 256 -> 128 | `51.2` | **36%** |

`up_block_3.resnets.0` alone is `22.0 ms` — one ResNet block, because it runs
`3x3` convolutions over `[8, 256, 256, 256]`. That is ~465 GFLOPs per call, and
its `[8, 512, 256, 256]` fp16 activations are `512 MB` per tensor, so it is
bandwidth-bound as well as FLOP-bound: GroupNorm, SiLU and the residual add each
stream that through HBM.

This is why the historical TensorRT/INT8 campaign plateaued. INT8 makes the same
FLOPs cheaper by at most ~2x against FP16, and it delivered `1.72x`
(`142.5 -> 82.9 ms`). There was no more headroom on that axis.

## Why TAESD is 20x faster

TAESD is **not** a quantized, compiled or pruned SD-VAE. It is a different,
much smaller network trained to approximate the same latent -> RGB mapping:

| | SD-VAE decoder | TAESD decoder |
| --- | ---: | ---: |
| Parameters | `49.5M` | `1.22M` (**40x fewer**) |
| Channels at `256²` | 128–256 | 64 |
| Channels at `128²` | 512 | 64 |
| Weights on disk | `320 MB` | `4.7 MB` |

The decisive number is the channel count at high resolution. Convolution cost
goes as `in_ch x out_ch`, so running 64 channels where SD-VAE runs 512 is a
**~64x** arithmetic reduction on the most expensive layer, before any
implementation detail. The speedup is not a systems trick — **the work was
removed, not executed faster.** That is why it beats a multi-year quantization
effort whose ceiling was ~2x.

The last `~1.8x` (`12.4 -> 6.85 ms`) is `torch.compile`. At this size the
network is kernel-launch-bound rather than compute-bound, and Inductor fuses the
many small ops into far fewer kernels.

Why the accuracy loss is tolerable *here specifically*, when a general
image-generation pipeline could not accept it:

1. MuseTalk only repaints the lower face — the blend mask uses rows `111..256`
   of the `256x256` decode on every avatar measured.
2. The decode is downscaled to the face bbox (`155x228` to `213x330`), i.e. to
   ~53% of the decoded pixel count, before compositing.
3. It is then alpha-blended into a real photographic frame, so the surrounding
   context is untouched ground truth.

A distilled decoder loses high-frequency microtexture first. Two of the three
steps above are low-pass operations, so most of what TAESD gives up is discarded
before it reaches the screen.

## Will it reproduce across every avatar?

Three different answers, and the distinction matters.

### Uniform — the VAE saving is avatar-independent

`141.1–142.1 ms -> 6.85–6.90 ms`, **under 1% variance across five avatars** with
frame sizes from `384x672` to `512x832` and cycles from 50 to 600 frames.

This is structural, not luck. Every avatar's face crop is resized to `256x256`
before VAE encoding (`scripts/api_avatar.py:907`, `musetalk/models/vae.py:40`
`resized_img=256`), so the decoder **always** sees `(B,4,32,32) -> (B,3,256,256)`
no matter what the avatar looks like. Avatar resolution, cycle length, framing
and content never reach it. This part will reproduce on any avatar.

### Predictable — end-to-end speedup varies `4.4x` to `5.2x`

Once the VAE is nearly free, the remaining per-frame cost is the UNet (fixed)
plus compose, and **compose scales with full-frame size and bbox area**
(`2.1 ms` at `384x672`, `4.7 ms` at `480x832`, `9.7 ms` before optimisation on
the largest). So a larger delivery frame means a smaller overall multiplier. The
rule: bigger frames -> lower e2e speedup, bounded below by the UNet.

### Highly variable — the RAM saving depends on cycle structure

| Avatar | Cycle | Unique | Host RAM now | Deduped | Saving |
| --- | ---: | ---: | ---: | ---: | ---: |
| `codex_smoke` | 600 | 300 | `558.6M` | `279.3M` | `2.00x` |
| `ltx23_native_flf_seed191` | 482 | 240 | `798.9M` | `397.8M` | `2.01x` |
| `ltx23_direct_recheck_seed190` | 480 | 229 | `759.6M` | `362.4M` | `2.10x` |
| `ltx23_talking_endpoint_override` | 480 | 229 | `759.6M` | `362.4M` | `2.10x` |
| `ltx23_closeup_smile_direct` | 290 | 133 | `458.9M` | `210.5M` | `2.18x` |
| `shared_indian_20260915` | 50 | **1** | `47.5M` | `0.9M` | **`50.00x`** |
| **all six resident** | | | **`3383M`** | **`1613M`** | **`2.10x`** |

Moving avatars land at `2.0–2.2x` because the cycle is forward+reverse. A static
portrait collapses to one frame. An avatar whose cycle was *not* built
forward+reverse would get no benefit — the dedup is content-addressed, so it
takes whatever is actually there and never makes things worse.

### Quality is the part to keep watching

`mae 0.0112 / 0.0113 / 0.0125` across three avatars is consistent, but these are
three renders of two synthetic people in similar lighting. A distilled decoder
gives up high-frequency detail first, so the avatars most likely to show it are
ones this set does not contain: heavy stubble or beard, glasses, strong
specular highlights, visible skin texture, or high-contrast makeup. **Run the
A/B on a new avatar class before assuming the number holds**, and judge it on
the video rather than the mae.

The numeric gate is cheap to re-run per avatar:

```bash
V=/workspace/.venvs/musetalk_trt_stagewise/bin/python
$V docs/fps_comparisons/4070s_20260922/e2e_indian.py --avatar <avatar_id>
```
