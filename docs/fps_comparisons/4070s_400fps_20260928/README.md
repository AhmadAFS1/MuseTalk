# MuseTalk 400 fps attempt, session 2: layer-selective INT8 UNet (2026-09-28, 19:40–21:05 UTC)

**Status (session 3, resumed by the user 21:15 UTC): 400 fps reached with 100% chin. The quality work is ongoing (§8).**
- **r4 = 414.9 fps**, measured in the six-stream full-recipe harness (engine set `srcblkA8`: srcmix + the
  `blkA_thr_8e-06` blocks).
  - Latent error is 6× lower than r3's, and the pixel-level change is about half of r3's, with no added flicker.
  - Videos and metrics: `experiments/video_validation/r4_srcblkA8_int8sel_taesdtrt_chin/`.
- Session 2 (§1–§7 below) had stopped at the user's request before any candidate was measured.

The previous session had already reached **350.2 fps** with 100% chin (round r2, FP16-noise quality). Its
400 attempt reached **415.6 fps** (round r3) but fails the UNet gate by ~15×. See
`experiments/video_validation/README.md` and `docs/fps_comparisons/4070s_300fps_impl_20260928/`. This session
tried to reach 400 fps at close to r2 quality by quantizing only the UNet layers that tolerate INT8.

## 1. Starting point and budget

The previous transcript was recovered from `~/.claude/projects/-workspace/866169db-….jsonl`. It had ended cleanly at
18:49 UTC with the r2/r3 summary; the user's verdicts on the r2/r3 videos and on the landmark bar were still open.

Box at 19:41 UTC: GPU idle, 17 GB RAM available, 14 GB disk free, power limit 220 W (= the maximum, so it cannot be
raised).

The six-stream harness (`scripts/chin_multistream_render.py`, full recipe) is **GPU-bound**. GPU busy is 99.9%
and the workers are 58% idle. Per 16-frame job:

| Run | Job total | UNet | TAESD decode + post + D2H | fps |
|---|---|---|---|---|
| r2 / srcmix (`Te_srcmix_taesdtrt_n6`) | 45.59 ms | 37.43 ms | 8.16 ms | 350.2 |
| r3 / srcv1 (`Tf_srcv1_taesdtrt_n6`) | 38.37 ms | 30.34 ms | 8.03 ms | 415.6 |

400 fps means a job of at most 40.0 ms. With the decoder unchanged, the UNet must be at most ~31.9 ms in the
harness, which is about **≤ 31.5 ms** as a sum of per-block engine times (the harness adds ~0.1–0.4 ms). That is
a saving of about **6 ms from srcmix**.

The exact staged-crop TAESD (0.345 vs 0.486 ms/frame, bit-exact) cannot be used: the chin tracker reads the whole
generated face (`scripts/vae_fast_decoder.py:22`, plan §8). The UNet therefore has to provide the saving.

## 2. Method: a fake-quant study that predicts the engines

New tool: `scripts/int8_layer_study.py`.
- It runs the eager FP16 UNet with modelopt fake quantization: per-channel INT8 weights and per-tensor INT8
  inputs, the same Q/DQ TensorRT uses.
- It scores against the repo gate metric (`scripts/validate_unet_backend.py`): per capture file, mae and max_abs
  of the UNet output vs the captured FP16 eager `pred_latents`. Gate: max-over-files mae ≤ 0.01 and
  max_abs ≤ 0.5.
- Corpus: `calibration/unet_multi_avatar_20260928`.
  - 32 main-split bs8 files for calibration.
  - `main_eval`: 88 other main files.
  - `holdout`: all 96 files (unseen avatars).
  - `quick`: 6 main + 6 holdout files, used for single-layer sensitivity.
- Candidates: 254 Conv2d/Linear layers, 82.8 GMAC/frame. Excluded:
  - `conv_in`, `conv_out`, the time embedding and `time_emb_proj` (tiny or constant-folded);
  - `down_blocks.0.resnets.0` (it is in the per-source prefix cache).

**Validation of the method.** A replica of the r3 whole-block recipe reproduces the real TensorRT r3 engine:

| | mae_mean (main) | mae_max (main) | max_abs (main) |
|---|---|---|---|
| Fake-quant replica (171 layers, 65.7% of MACs) | 0.0300 | 0.0374 | 2.375 |
| TensorRT v1 engine, repo gate (previous session) | 0.0299 | 0.0382 | 2.389 |

Recipes can therefore be screened in seconds, not in 10–12 minute engine builds. For reference, re-running eager
against the captured reference scores mae_max 0.00033 / 0.00030 and max_abs 0.044 / 0.059 (main / holdout).

## 3. Findings

### 3.1 Where the MACs are

| By category | Share | By block | Share |
|---|---|---|---|
| res_conv1 | 29.0% | up2 | 23.9% |
| res_conv2 | 17.9% | up1 | 22.7% |
| ff_in (GEGLU proj) | 15.4% | up3 | 19.9% |
| upsample conv | 10.3% | down1 / down2 | 9.5% each |
| ff_out | 7.7% | down0 (rest) | 7.7% |
| attn1 q/k/v | 5.8% | up0 | 3.9% |
| res_shortcut | 2.7% | mid / down3 | 1.8% / 1.1% |
| proj_in, proj_out, attn1_out, attn2_q, attn2_out | 1.9% each | | |
| downsample / attn2 k/v | 0.9% / 0.6% | | |

### 3.2 Single-layer sensitivity

`study.json` → `sens`. Only one layer is INT8; the value is the added output MSE on the quick set.

- **Two activation-range rules were tried:** max calibration, and a per-layer MSE-optimal clip (`amax_mse`: the
  clip that minimizes the layer's own output MSE). The MSE clip cuts single-layer error 2–4× for most layers.
  It is not always better end to end: for the `up2` upsampler, max gives 1.7e-5 and mse gives 3.4e-5. Recipes
  therefore pick max or mse per layer, whichever has the lower single-layer error.
- **Least sensitive:** every `ff_in`, the attention q/k/v and out projections, the `up1` upsampler (4.6% of MACs,
  4e-6), and nearly everything in `down1`, `down2`, `up0`, `mid` and `down3`.
- **Most sensitive** (MSE clip):

  | Layer(s) | Added MSE |
  |---|---|
  | `up1.resnets.2.conv1` | 2.0e-4 (max_abs 1.07) |
  | `up3` ResNet convs | 1.4e-5 to 1.1e-4 each |
  | `up1.attentions.2.proj_out` | 3.1e-5 |
  | `ff_out` in `up1` / `up2` / `up3` | 1e-5 to 4e-5 |
  | `up2.resnets.2` convs | 1.5e-5 to 1.7e-5 |

- **By block:** summed sensitivity is up3 6.8e-4, up1 5.1e-4, up2 1.4e-4, down0 8.1e-5, down1 2.9e-5, down2 1.9e-5,
  up0 4.2e-6, mid 3.2e-7 and down3 1.5e-7.
- **Implication:** the ResNet convs, the largest MAC share, are sensitive, not robust. This is the opposite of
  the usual assumption.

### 3.3 Combined recipes (fake-quant, full eval splits)

"Threshold" recipes take every layer whose best single-layer MSE is ≤ T. `down3` and `mid` stay INT8 (all layers,
max calibration) as in srcmix. "MACs" includes `down3`/`mid` (~0.03).

| Recipe | Layers | MACs | main mae_max / max_abs | holdout mae_max / max_abs |
|---|---|---|---|---|
| r2 FP16 TRT engine (reference, repo gate) | – | – | 0.0025 / 0.39 | 0.0021 / 0.24 |
| **Large layers only** (≥ 0.5% of MACs each), all blocks | | | | |
| thr_1e-06 | 41 | 0.268 | 0.00179 / 0.493 | 0.00172 / 0.695 |
| thr_2e-06 | 49 | 0.354 | 0.00295 / 0.564 | 0.00277 / 0.474 |
| thr_4e-06 | 58 | 0.446 | 0.00439 / 1.370 | 0.00413 / 1.128 |
| thr_8e-06 | 64 | 0.530 | 0.00588 / 1.021 | 0.00547 / 0.812 |
| thr_2e-05 (T = 1.6e-5) | 69 | 0.592 | 0.00817 / 1.192 | 0.00863 / 1.123 |
| **Large layers, no down0** (built as engines) | | | | |
| nd0_thr_4e-06 | 56 | 0.426 | 0.00436 / 0.968 | 0.00395 / 1.173 |
| nd0_thr_8e-06 | 61 | 0.499 | 0.00557 / 0.808 | 0.00511 / 1.274 |
| **All layers of down1–up2** minus a blacklist, audio K/V excluded (§3.4) | | | | |
| blkA_thr_2e-06 | 128 | 0.427 | 0.00354 / 0.475 | 0.00457 / 1.135 |
| blkA_thr_4e-06 | 139 | 0.516 | 0.00503 / 0.834 | 0.00544 / 1.263 |
| blkA_thr_8e-06 (engine built, not timed) | 146 | 0.593 | 0.00634 / 1.176 | 0.00668 / 1.895 |
| blkA_thr_2e-05 (T = 1.6e-5) | 151 | 0.629 | 0.00823 / 1.319 | 0.00776 / 2.225 |
| r3 replica, for comparison | 171 | 0.657 | 0.0374 / 2.375 | 0.0409 / 2.024 |

- **mae:** at similar coverage, layer-selective INT8 has **4–6× lower mae** than r3. Every recipe up to 0.63 of
  MACs passes the mae ≤ 0.01 gate.
- **max_abs:** every recipe beyond ~0.3 of MACs fails the max_abs ≤ 0.5 gate. max_abs is a single worst latent
  element, it is noisy (not even monotonic in the table), and the FP16 engine already sits at 0.39–0.43. How much
  it matters has to come from the pixel-level quality tool and the video. That was not reached (§6).
- **Naming:** the recipe names ending `thr_2e-05` are really T = 1.6e-5, an artefact of `:.0e` formatting.

### 3.4 Audio-input layers must stay FP16 (a production-relevant trap)

The first all-layers recipes (`blk_thr_*`) had holdout mae 0.014–0.020, while main was 0.004–0.011. A per-category
ablation (`recipes_abl.json`) found the cause: only removing `attn2.to_k` / `attn2.to_v` (the cross-attention
projections of the audio features) fixes it, taking holdout mae from 0.0144 to 0.0043. Removing any other category
leaves 0.0143–0.0170.

Cause:
- Calibration used only the TTS clips `A_af_heart_dense` and `B_am_michael`.
- Half the holdout uses `C_repo_eng`, a real human recording.
- The per-tensor audio amax from TTS clips the real-speech features.

These layers are 0.6% of MACs, so they stay FP16 in every recipe. More generally, INT8 calibration must include
real speech, and the holdout's `C_repo_eng` half is the generalization check.

### 3.5 Where INT8 speed actually is (per-block TensorRT timing)

New tool: `scripts/bench_stagewise_blocks.py`. It captures each block engine into a CUDA graph and times the sets
interleaved; values are the median ms per bs16 call. Results: `blocks/existing_sets.json`, `blocks/pool_up01.json`,
`blocks/recipes_r1.json`.

| Block | FP16 (srcmix) | All-INT8 | INT8 pool | nd0_thr_4e-06 | nd0_thr_8e-06 |
|---|---|---|---|---|---|
| down0rest | 4.37 | 4.41 (v1 `down0`) | **0** | – (FP16) | – (FP16) |
| down1 | 3.10 | 2.13 | 0.97 | 2.55 | 2.42 |
| down2 | 2.99 | 1.70 | 1.29 | 2.09 | 2.14 |
| down3 / mid | 0.24 / 0.48 | (INT8 in srcmix) | – | 0.25 / 0.50 | 0.25 / 0.47 |
| up0 | 1.30 | 0.72 (`pool_up01`) | 0.58 | 0.90 | 0.92 |
| up1 | 6.90 | **3.54** (`pool_up01`) | **3.35** | 5.23 | 4.61 |
| up2 | 7.31 | 4.61 | 2.69 | 6.31 | 6.12 |
| up3 | 10.21 | 8.38 | 1.79 | **10.71** | **10.74** |
| tail | 0.10 | 0.10 | – | – | – |
| **UNet sum** | **37.02** | 30.25 (srcv1) | ~10.7 total | **33.02** | **32.14** |

- **`up1` is the biggest untapped pool** (−3.35 ms). r3 never quantized it.
- **`down0` gains nothing from INT8.** Its 1024-token attention dominates.
- **`up3` gets slower with partial INT8** (+0.5 ms): three isolated INT8 `ff_in` GEMMs at 1024 tokens cost more in
  Q/DQ than they save. Keep `up3` FP16 unless it is quantized wholesale.
- **Small layers matter.** `down2` with all its large layers INT8 (2.14 ms) is still 0.44 ms slower than all-INT8
  `down2` (1.70 ms). The gap is the small attention projections, whose single-layer errors are ~1e-8 to 1e-7. That
  motivated the `blkA` all-layers-minus-blacklist shape.
- **Projection, not measured:**
  - nd0_thr_8e-06 with `up3` from srcmix: 32.14 − 10.74 + 10.21 = **31.61 ms**. That saves 5.4 ms, gives a
    harness job of ≈ 40.1 ms and ≈ 399 fps.
  - blkA_thr_8e-06 covers more small layers, so it should be a little faster. Its engines are built (`up3`
    excluded) but were never timed.

### 3.6 Exact SmoothQuant for `ff_out`: not adopted

New module: `scripts/unet_int8_smooth.py`. It folds a per-channel power-of-two scale into the GEGLU value rows and
the `ff.net.2` columns (α = 0.5). Results are in `int8_study/study_smooth05.json`.
- **Not bit-exact after all.** The FP16 output changed by up to 0.0116 (subnormal rounding of the rescaled
  weights). That is within FP16 noise, but it breaks the "bit-identical" claim.
- **Small gain.** Single-layer `ff_out` error fell only ~1.5–2×, on layers holding ~3% of MACs. For example,
  `up1.attentions.0` went 2.8e-6 → 1.3e-6 and `up2.attentions.1` went 1.2e-5 → 8.9e-6.
- **Status:** the builder applies it only when a recipe carries `smooth_ff`. No exported recipe does.

### 3.7 Other notes

- **The previous-session levers still stand:**
  - concurrency is slower on the 220 W cap;
  - bs32 and TAESD bs2/bs4 give nothing;
  - FP8 is dead;
  - attention already runs as fused MHA at ~5% of engine time (plan §8). `scripts/profile_stagewise_block.py` was
    written for that question and never run, so it is not needed.
- **INT8 TAESD, held in reserve.** The 2026-09-27 probe measured the staged crop at 0.147 vs 0.345 ms/frame in
  FP16, at 46.3 dB PSNR (`4070s_300fps_20260927/taesd_probe/p9_int8.json`). At full height that suggests ~4 ms
  per 16 frames. Decoder error goes straight to pixels and into chin tracking, so it would need the same
  per-layer treatment and a video verdict.
- **TAESD optimization level, not tried.** The TAESD TRT engine is built at optimization level 3
  (`MUSETALK_TAESD_TRT_OPT_LEVEL` default); the UNet uses 5. A level-5 rebuild is a possible small FP16-only gain.
- **Build cost.** INT8 builds take 30–185 s per block and ~11–12 min per 7–8-block set. Host RSS peaks at
  ~11.8 GB (minimum MemAvailable 8.3 GB). All runs went through `scripts/box_guard.sh`, and the oom_kill counter
  stayed at 12.

## 4. Session log (UTC)

| Time | Step |
|---|---|
| 19:40 | Recovered the previous state from memory, the transcript and repo docs. Box check. |
| 19:49 | Study inventory and baseline: fake-quant replica matches the r3 engine. |
| 19:55 | Per-layer MSE-optimal clip and single-layer sensitivity (254 layers × 2 calibration rules, 5 min). |
| 19:58 | Threshold recipes (large layers, all blocks). |
| 20:02 | Per-block timing of the existing sets (srcfp16, srcmix, srcv1, v1). |
| 20:06 | `pool_up01` build (all-INT8 `up0`/`up1`): `up1` 6.90 → 3.54 ms. |
| 20:08–20:40 | Exported and built `nd0_thr_4e-06` and `nd0_thr_8e-06`. Timed at 20:41. |
| 20:45 | `ff_out` smoothing study: not exact, small gain. |
| 20:47–20:51 | `blk_*` recipes, audio K/V ablation, `blkA_*` recipes. |
| 20:52 | Started `blkA_thr_8e-06` then `blkA_thr_4e-06` builds. Wrote the LSQ stage and the validation driver meanwhile. |
| ~21:05 | User said stop. `blkA_thr_8e-06` had finished (rc 0). The `blkA_thr_4e-06` build was killed, the GPU lease released and the GPU left idle. |

## 5. Files from this session

Code, uncommitted:

| File | Status |
|---|---|
| `scripts/int8_layer_study.py` | New. Stages `inventory`, `baseline`, `amax_mse`, `sens`, `recipes` and `export` were used. `errdist` (where the worst latent errors sit), `lsq` (learned per-layer activation ranges, trained end to end on the output MSE with straight-through rounding) and `tune` (coordinate descent on the ranges) are **written but never run**. |
| `scripts/build_unet_stagewise.py` | Modified. New `--int8-recipe <json>`: blocks that hold a recipe layer are built INT8 with Q/DQ only on those layers, and input amax is pinned to the recipe's values. It also applies a recipe's optional `smooth_ff`. Default behaviour is unchanged. |
| `scripts/bench_stagewise_blocks.py` | New. Used. Per-block engine timing, and it accepts partial sets. |
| `scripts/unet_int8_smooth.py` | New. Used in the study only. Not bit-exact (§3.6). |
| `scripts/assemble_stagewise_set.py` | New. **Never run.** Symlinks overlay blocks onto a base set and merges manifests, like the srcmix/srcv1 assembly. |
| `scripts/profile_stagewise_block.py` | New. **Never run**, and not needed (§3.7). |
| `docs/fps_comparisons/4070s_400fps_20260928/validate_candidate.sh` | New. **Never run.** Runs T (throughput), Q (quality capture + `quality_ab_metrics.py`) and V (video capture) for one engine set, using the r2/r3 commands. |

Results in this folder:
- `int8_study/study.json`: inventory, baseline, amax_mse, sens (max and mse), every evaluated recipe with per-layer
  amax, and the splits.
- `int8_study/study_smooth05.json`: the same study with `ff_out` smoothing.
- `int8_study/recipes_*.json`: layer lists fed to the recipes stage.
- `int8_study/recipe_<name>.json`: exported builder recipes, for `nd0_thr_4e-06`, `nd0_thr_8e-06`, `blkA_thr_4e-06`
  and `blkA_thr_8e-06`.
- `int8_study/recipes_perblock.json`: 30 per-block, per-level recipes for a block-level knapsack. Never evaluated.
- `blocks/`: the build scripts, build logs and per-block timing JSON.

Engine directories, in `/workspace/MuseTalk/models/` via the `models` symlink. All are partial sets that hold only
the rebuilt blocks, so none is loadable as a full UNet until it is assembled with srcmix's prefix, `down0rest`,
`up3` and tail and then finalized:

| Directory | Size | Contents |
|---|---|---|
| `tensorrt_unet_stagewise_sm89_pool_up01` | 412 MB | all-INT8 `up0`/`up1` (speed probe only) |
| `tensorrt_unet_stagewise_sm89_nd0_thr_4e-06` | 1.2 GB | `down1`–`up3` recipe blocks |
| `tensorrt_unet_stagewise_sm89_nd0_thr_8e-06` | 1.1 GB | `down1`–`up3` recipe blocks |
| `tensorrt_unet_stagewise_sm89_blkA_thr_8e-06` | 917 MB | `down1`–`up2` recipe blocks, **not timed** |
| `tensorrt_unet_stagewise_sm89_blkA_thr_4e-06` | empty | build killed |

Disk was at 11 GB free at the stop. Deleting any of these is the user's call.

None of the recipe engines has been through the repo UNet gate (`validate_unet_backend.py`) or the builder's
finalize/probe step. Their TensorRT error is predicted by fake-quant only (§2).

## 6. If this is resumed (only when the user asks)

1. **Time the built set:**
   `scripts/bench_stagewise_blocks.py --root models/tensorrt_unet_stagewise_sm89_srcmix --root models/tensorrt_unet_stagewise_sm89_blkA_thr_8e-06`.
   The `blkA_thr_8e-06` sum plus srcmix `down0rest`/`up3`/tail must be ≤ ~31.5 ms.
2. **If it is short:**
   - run `int8_layer_study.py --stage lsq --lsq-recipe blkA_thr_8e-06` (untested; check its GPU memory first);
   - or pick per-block coverage levels from `recipes_perblock.json` against the per-block times in §3.5;
   - then re-export and rebuild.
3. **Assemble and finalize:**
   - `scripts/assemble_stagewise_set.py --base models/tensorrt_unet_stagewise_sm89_srcmix --overlay <recipe root>:down1,down2,down3,mid,up0,up1,up2 --out models/tensorrt_unet_stagewise_sm89_<name>`;
   - then `build_unet_stagewise.py --variant srccache --root <out> --blocks prefix` to finalize;
   - then the repo UNet gate on main and holdout (commands in `4070s_300fps_impl_20260928/unet_fp16/run_srccache.sh`).
4. **Validate:** run `docs/fps_comparisons/4070s_400fps_20260928/validate_candidate.sh <label> <root> T Q V`, then
   compose the labelled A/B videos the way r3 was composed:
   `scripts/video_ab_round.py --round r4_<name> --run-dir docs/fps_comparisons/4070s_400fps_20260928/chin_multistream/V_<label> --quality-label <label> --label-b '<what B is>' --fps-b '<measured fps>'`.
   It writes to `experiments/video_validation/r4_<name>/`; add a row to that README's rounds table.

## 7. Decisions still open for the user

Carried over from the previous session:
- **Videos:** the verdict on the r2 (350 fps) and r3 (415.6 fps) videos.
- **Landmark bar:** keep the strict 0.05/0.15 px bar or adopt the calibrated 0.10/0.35 px one.

New from this session:
- **max_abs gate for INT8:** every INT8 recipe with enough coverage for 400 fps fails the repo's max_abs ≤ 0.5
  latent gate, even at mae 0.004–0.007. Should the gate for INT8 candidates be the pixel-level quality tool and the
  video instead?

## 8. Session 3 (2026-09-28, from 21:15 UTC): 400 fps measured, then quality work

The user asked to continue the goal. This session is the original 300/350 fps session, resumed.

### 8.1 blkA_thr_8e-06 timed, assembled and measured: 414.9 fps (round r4)

**Per-block timing** (`blocks/blkA_thr_8e-06.json`, interleaved with srcmix and nd0_thr_8e-06):
- The `blkA_thr_8e-06` blocks `down1`–`up2` sum to **15.34 ms**, against 22.13 ms for srcmix, which is FP16 except for INT8 `down3`/`mid`.
- With srcmix's `down0rest` 4.37 + `up3` 10.18 + tail 0.11 ms, the UNet is **30.0 ms per bs16 call**. That is
  inside the ~31.5 ms 400 fps budget and on par with srcv1 (30.25 ms).
- The more small layers `blkA` covers, the more `down1`/`down2`/`up0`/`up1` speed up: 2.16 / 1.76 / 0.75 / 4.18 ms,
  against 2.38 / 2.13 / 0.90 / 4.56 ms for `nd0_thr_8e-06`.

**Set and gate** (`gate/run_assemble_gate.sh srcblkA8 ...`):
- `models/tensorrt_unet_stagewise_sm89_srcblkA8` = srcmix prefix, `down0rest`, `up3` and tail, plus `blkA_thr_8e-06`
  `down1`–`up2`, finalized.
- Repo UNet gate:

  | Split | mae_max | max_abs |
  |---|---|---|
  | main | 0.00706 | 1.28 |
  | holdout | 0.00614 | 1.80 |

  The fake-quant prediction was 0.0063 / 1.18 and 0.0067 / 1.90. The mae gate passes; max_abs fails, as predicted.

**Full recipe** (`validate_candidate.sh srcblkA8 ... T Q V`, six streams, TRT TAESD + 100% chin + refined seam):
- **Throughput: 415.6 / 414.1 fps, median 414.9**, in two 62 s runs of 25,920 frames each. GPU busy 99.9%.
- **Quality**, 6/6 identities vs the accepted renders on raw frames. The full table is in
  `experiments/video_validation/r4_*/README.md`; `compare_quality.py` builds it.

  | Metric | r4 | r3 | r2 |
  |---|---|---|---|
  | Lip correlation | 0.9989–0.9994 | 0.997–0.9986 | ≥ 0.9998 |
  | Aperture delta (px) | 0.13–0.24 | 0.23–0.37 | 0.04–0.08 |
  | Mouth flicker ratio | 0.998–1.003 | 1.017–1.025 | ~1.000 |
  | Landmark deviation mean / p99 (px) | 0.105–0.171 / 0.37–0.65 | 0.20–0.36 / 0.65–1.01 | 0.036–0.094 / 0.11–0.47 |
  | Face / mouth PSNR (dB) | 54.9–57.6 / 47.7–50.1 | — | 60.8–64.2 / 55.1–57.7 |

  The synthetic ±1 LSB noise arm gives 56.9–57.8 / 51.5–51.9 dB. Every perceptual gate passes. The landmark gates
  fail at both bars.
- **Videos:** round r4, 6 A/B clips plus a mosaic.

### 8.2 Quality recovery at the same speed: small gains only

All numbers below are fake-quant. Recovery runs use `int8_layer_study.py --stage wa` and `--stage recover` (new).
- **Which side dominates** (`wa`):

  | INT8 side (blkA8) | Latent MSE |
  |---|---|
  | Weights only | 7.9e-6 |
  | Activations only (per-tensor) | 1.13e-4 |
  | Both | 1.2e-4 |

  Activation rounding is ~14× the weight error.
- **Learned input ranges** (LSQ, 12 epochs over calib): main MSE −17%, holdout unchanged.
  - The old `lsq` stage could not run: modelopt's diffusers `Attention` wrapper routes SDPA through a forward-only
    ONNX-export op (`FP8SDPA`). `recover` switches that off for training.
- **Per-channel bias corrections:** unstable at lr 2e-4. At lr 2e-5, −10–17%.
- **Learned per-channel weight scales:** NaN in FP16.
- **SmoothQuant-style per-input-channel smoothing** (fusable into the preceding GroupNorm/SiLU/LayerNorm):
  worse at every α. MSE 1.17e-4 at α 0.3, 1.9e-4 at 0.5 and 5.7e-4 at 0.7. There are no channel outliers to
  migrate.
- **Quantization-aware LoRA** (rank 16 on every INT8 layer, merged into the FP16 weights before export, so no
  speed cost):
  - it diverges at lr ≥ 2e-5 (Adam moves the whole low-rank update coherently);
  - at 1e-6 it reaches main 9.3e-5 / holdout 9.9e-5, i.e. −20% / −15%.
- **Conclusion:** the INT8 error here is per-tensor activation rounding noise, not a systematic offset. Recovery at
  fixed coverage buys ≤ 20%. The lever is which layers are INT8 per millisecond saved.

### 8.3 Where the FP16 time is (`profile/srcblkA8_profile.json`)

- **up3 (10.3 ms):**
  - pointwise and norm kernels 38%;
  - convs 32%, near tensor-core peak (`resnets.0.conv1` 960→320 runs at ~55 TMAC/s);
  - attention and the GEGLU/FF GEMMs 30%.
- **down0rest:** attention and GEGLU 45%, pointwise 39%, convs 16%. That is why INT8 on its convs buys nothing.
- **`up2.upsamplers.0.conv`** (FP16 in blkA8): 1.08 ms, the single largest conv outside `up3`.
- **The rest:** there is no cheap FP16-only speed left in the UNet. The pointwise/norm share needs custom fusion.

### 8.4 Choosing INT8 layers by error per MAC (recipes `gmac_*`)

- **Method:** greedy by best single-layer error / MACs, over `down1`–`up2`. `down0`, `up3` and audio K/V stay FP16;
  `down3`/`mid` stay all-INT8.
- **Result:** at equal MACs, the real combined error is much lower than the threshold recipes.

  | Recipe | MACs | Main MSE | Holdout MSE |
  |---|---|---|---|
  | blkA_thr_8e-06 (r4) | 0.593 | 1.22e-4 | 1.22e-4 |
  | gmac_0.62 | 0.620 | 9.7e-5 | 9.4e-5 |
  | gmac_0.59 | 0.590 | 7.0e-5 | 7.6e-5 |
  | gmac_0.55 | 0.527 | 4.6e-5 | 4.9e-5 |
  | gmac_0.50 | 0.498 | 3.2e-5 | 3.9e-5 |
  | blkA_thr_2e-06 | 0.427 | 3.5e-5 | 4.1e-5 |

- **Why (gmac_0.59 vs blkA8):** the main difference is `up2.upsamplers.0.conv`. It has 4.6% of MACs at 1.7e-5, 1.08 ms in FP16, and is
  now INT8. In exchange, ~18 small res-shortcut / proj / ff_out layers go back to FP16.

### 8.5 Round r5: gmac_0.50, 400.9 fps at ~3× lower error than r4

**Per-block timing** (`blocks/gmac_sets.json`, 9 interleaved rounds). Times for `down1`–`up2`:

| Set | ms |
|---|---|
| srcmix (FP16 except INT8 down3/mid) | 22.19 |
| blkA8 | 15.41 |
| gmac_0.59 | 15.22 (faster than blkA8 at 1.7× lower error) |
| gmac_0.50 | 16.49 |

`gmac_0.55` is partial: its build was killed by the box_guard RAM watchdog at 22:43 (§8.6).

**Selection.** `select_combo.py blocks/gmac_sets.json 31.45` uses per-block fake-quant errors (recipes
`pb_<set>_<block>`), which add up to within ~10% of the combined error. The best combination is `gmac_0.50` in
every block: UNet 31.20 ms, predicted MSE 3.8e-5, vs 1.2e-4 for r4.

**Set.** `models/tensorrt_unet_stagewise_sm89_srcg50` = srcmix prefix, `down0rest`, `up3` and tail, plus the
`gmac_0.50` blocks. Repo UNet gate:

| Split | mae_max | max_abs |
|---|---|---|
| main | 0.00444 | 0.78 |
| holdout | 0.00392 | 1.26 |

For comparison, r4 is 0.0071 / 1.28 and r2 is 0.0025 / 0.39.

**Full recipe:**
- **Throughput: 401.8 / 400.1 fps, median 400.9**, in 64 s runs.
- **Quality**, 6/6 identities, raw frames vs accepted:
  - lip corr 0.9997–0.9998;
  - aperture delta 0.08–0.14 px;
  - flicker 0.999–1.001;
  - landmarks 0.064–0.126 / 0.20–0.47 px, calibrated bar 4/6 (r2 5/6, r4 0/6);
  - mean mouth PSNR 51.0–53.4 dB, the ±1 LSB-noise level; worst frames are 43–47 dB, 4–8 dB below the noise arm;
  - aperture and landmark changes are about 1.5–2× the noise arm.
- **Composition:** `gmac_0.50` has 117 INT8 layers in total, 99 beyond r2's `down3`/`mid`. It is a strict subset of
  blkA8 (29 of blkA8's layers are back in FP16; `up2.upsamplers.0.conv` stays FP16). It ties for the lowest
  predicted error with a `down2`-from-`gmac_0.55` combination.
- **Sustained throughput** (`T_srcg50_sustained`, 5 × 64 s): 404.0 → 400.9 → 400.5 → 400.1 → 399.96 fps as the GPU
  warms (64 → 67 °C, SM 2475 → 2460 MHz). r5 sustains ≈400 with no margin.
  - A faster mix of the already-built blocks, `down1` and `up0` from `gmac_0.59`, predicts ≈ −0.28 ms/call (≈ +3 fps)
    at +21% latent error. It is not measured.
- **Like-for-like baseline** (`T_baseline_pair`, back to back with `T_srcg50_pair`): pre-change backends 252.6 / 251.4
  fps, all clips bit-identical to the accepted-recipe renders, against r5 401.4 / 400.1. That is a 1.59× speedup.
  - The baseline pair ran with `--compare-accepted`, about 1 s of CPU per worker; the GPU was 99.9% busy in both
    runs.
- **Videos:**
  - `experiments/video_validation/r5_srcg50_int8gmac_taesdtrt_chin/`;
  - `focus_before_r2_r5/` (BEFORE | r2 | r5 at native resolution);
  - `lineage_all_rounds/` (BEFORE | r2 | r3 | r4 | r5).
- **Fairness of those videos:** all columns are rebuilt from raw frames with `scripts/video_lineage.py`, bit-exact
  to the recorded render SHAs on 30/30 columns, and encoded once.

### 8.6 Incident: one build killed by the RAM watchdog (22:43 UTC)

- **What happened:** a CPU-side lineage-video test render, about 2 GB, ran while the `gmac_0.55` engine build was at
  its RAM peak (10.7 GB RSS). MemAvailable fell to 2.95 GB, and box_guard's watchdog killed the build at its
  3 GB floor.
- **Impact:** nothing else was affected; the kernel oom_kill counter stayed at 12. `gmac_0.55` has `up1`/`up2`
  missing and was not needed.
- **Rule:** no CPU-heavy work while an engine build runs.

### 8.7 Stability: 15 concurrent streams on r5 (2026-09-29, `T_srcg50_n15_stability`)

**Setup.** 15 streams in the full-recipe harness (TRT TAESD, 100% chin, refined seam) with the r5 engines, 10
consecutive ~63 s timed windows (10.5 min, 252,000 frames). Only the six avatars in `avatar_diversity_20260927`
have every chin-recipe asset, so three avatars ran 3 streams each and three ran 2.
- Each stream is its own session: its own worker process, FaceMesh tracker and chin state.
- Streams on the same avatar share that avatar's read-only arena in `/dev/shm` (~590 MB) and its GPU caches.
- 15 *distinct* avatars would need ~8.8 GB of arenas. They don't fit next to the 8.9 GB SoulX state in `/dev/shm`.

The full per-window table is in `stability_n15.md`, made by `stability_report.py`.

| Measure | Result |
|---|---|
| Aggregate fps per window | 403.9, 399.8, 400.3, 399.9, 400.0, 399.7, 399.6, 399.7, 400.0, 399.6 (median 399.8) |
| Per-stream fps, clip by clip (900 clips of 240 frames) | min 26.56, p1 26.59, median 26.67, max 27.10. Every stream stays above the 20 fps real-time target; 15 × 20 needs 300 aggregate. |
| Fairness | All 15 streams finish each 63 s window within 0.28–0.29 s of each other |
| Per-stream variation across windows | CV 0.31% |
| Output correctness | Bit-exact determinism: every clip of an avatar hashes identically across all its streams and loops |
| GPU | SM 2475 → 2460 MHz, 204–206 W, 64 → 67 °C, 4.0 GB memory, busy 99.9% |
| Host memory | MemAvailable 7.63 → 5.79 GB (box_guard floor 3 GB). Workers are flat (+0–6 MiB). |
| Errors | None. `oom_kill` unchanged. |

**One thing to watch: FaceMesh helper memory.**
- Each stream's chin-tracker subprocess (mediapipe, in the unchanged workflow code) grew 76–169 MiB over the 10
  windows.
- The growth slows down: about 17.6 MiB per window per helper in the first half, 7.4 in the second, with dips.
  The 6-stream sustained run shows the same shape (17.0 → 7.2).
- That looks like heap warm-up settling rather than an unbounded leak, but 10 minutes does not prove a plateau.
- **Before long-running 15-stream serving,** run an hour-long soak (CPU-only is enough:
  `--backend replay --streams 15`). If it doesn't level off, recycle each tracker subprocess every N clips.

**Harness fix.** Almost every harness run, including the pre-change baseline, recorded a `stop_warning` "worker N
exited".
- **Cause:** a race in the harness's own shutdown. A worker closes its pipe right after sending its final "bye".
- **Fix:** `Collector.wait_for` now treats end-of-file after "bye" as a clean exit.
- **Check:** a CPU-only replay run afterwards has no warning and stays bit-identical to the accepted renders. The
  timed runs above used the old code; their data is unaffected.
