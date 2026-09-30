# Reproducing the 350 / 400 fps MuseTalk results (RTX 4070 SUPER, 100% chin recipe)

These scripts rebuild and re-measure the published results from a clean state: the TensorRT engines, the
correctness gates, the harness throughput, quality, video and stability runs, and the labelled videos.

| Round | What | Aggregate fps (6 streams, same harness) | Published engine set |
|---|---|---|---|
| BEFORE | shipping TensorRT FP16 bs8 `.ts` UNet + compiled TAESD | 252.0 | `models/tensorrt_unet_sm89_bs8_local/unet_trt.ts` |
| r2 | stagewise FP16 bs16 UNet + source-prefix cache + INT8 `down3`/`mid` + TensorRT TAESD | 350.2 (1.39×) | `models/tensorrt_unet_stagewise_sm89_srcmix` |
| **r5** | r2 + INT8 on 117 UNet layers chosen by error per MAC (`gmac_0.50`) | **≈400 sustained** (1.59×) | `models/tensorrt_unet_stagewise_sm89_srcg50` |

- **Quality evidence and sign-off:** `experiments/video_validation/` (`signoff_r2_r5/`, `focus_before_r2_r5/`,
  `lineage_all_rounds/`, `r5_*/README.md`).
- **Engineering record:** `docs/fps_comparisons/4070s_300fps_impl_20260928/` (r1–r3) and
  `docs/fps_comparisons/4070s_400fps_20260928/README.md` (r4, r5, the INT8 study, the stability test).

## Steps

Each step is a script that is safe to re-run. Every heavy step runs under `scripts/box_guard.sh`: GPU lease, RAM
watchdog (kills below 3 GB available) and the pause file. Logs, JSON and quality runs go to
`docs/fps_comparisons/repro_400fps/` (`MUSETALK_REPRO_OUT`). Everything is tagged `repro_<set>`, so re-measuring a
published set never overwrites the published records.

| Step | Script | Time | Needs free RAM | What it does |
|---|---|---|---|---|
| 0a | `05_fetch_inputs.sh [--engines]` | ~1–3 min | – | Restores the inputs git does not hold from checksum-pinned S3 bundles: the six harness avatars and the UNet capture corpus (and with `--engines` the published r5/r2 engines + TAESD TRT). Inputs already here are verified against the bundles, not overwritten. Needs the runtime secret's bucket and read access (`set -a; . /workspace/.musetalk-runtime.env; set +a`). |
| 0 | `00_check.sh [--deep]` | ~1 min | – | Prerequisites: GPU, power limit, versions against `requirements/constraints-cu121.txt`, weights and avatars byte-exact against `render.json`, the corpus against its manifest, the runtime env file, `/dev/shm`, disk, tools, and that the package is committed. |
| 1 | `10_build_engines.sh [--set r5\|r2] [--hardware-compat ampere_plus] [--dry-run]` | ~20–25 min (~45 min with `ampere_plus`) | 14 GB | Builds the set from scratch into `models/tensorrt_unet_stagewise_sm89_<set>` (`..._ampere_plus_<set>` with `--hardware-compat ampere_plus`: the served, GPU-portable variant). Order: FP16 blocks, then INT8 blocks, then `prefix` (finalizes). Then the TensorRT TAESD engine (hardware-compatible too with that flag), then a per-block ONNX comparison with the published set. |
| 2 | `20_gate.sh [root]` | ~6 min | 8 GB | Repo UNet gate (main + holdout), `forward_cached == forward` bit-exactness, and TAESD load + probe plus the published G-TAESD gate. |
| 3 | `30_benchmark.sh [root] [BENCH T SUST PAIR Q V N15]` | ~50 min for all | 12–14 GB | Per-block times vs the published set; 6-stream throughput; sustained throughput; like-for-like BEFORE pair; quality metrics; video capture; the 15-stream stability run. |
| 4 | `40_videos.sh [root] [r2 capture]` | ~5 min | 6 GB | BEFORE \| r2 \| the set at native resolution: bit-exact, encoded once, with diff row, metrics and verdicts. |
| opt. | `50_derive_recipe.sh` | ~10 min | 8 GB | Re-derives the INT8 recipe: fake-quant sensitivity study, greedy error per MAC, export. |

**Commands.** A full reproduction on a free box:

```bash
scripts/repro_400fps/05_fetch_inputs.sh                 # avatars + corpus from S3 (needs the runtime secret env)
scripts/repro_400fps/00_check.sh
scripts/repro_400fps/10_build_engines.sh                 # r5; add --set r2 for the 350 fps set
scripts/repro_400fps/20_gate.sh
scripts/repro_400fps/30_benchmark.sh
scripts/repro_400fps/40_videos.sh
```

To re-measure the published engines without building (records are tagged `repro_srcg50`):

```bash
scripts/repro_400fps/20_gate.sh models/tensorrt_unet_stagewise_sm89_srcg50
scripts/repro_400fps/30_benchmark.sh models/tensorrt_unet_stagewise_sm89_srcg50 T SUST N15
```

## Expected results (published r5, this box)

| Measurement | Published value | Notes |
|---|---|---|
| UNet per bs16 call, sum of block engines | 31.17 ms | `blocks/gmac_sets.json`; srcmix is 36.87 ms. The 31.20 ms in the record was `select_combo.py`'s prediction. |
| Repo UNet gate, main / holdout (mae_max / max_abs) | 0.0044 / 0.78 and 0.0039 / 1.26 | mae passes 0.01. max_abs fails 0.5, as every INT8 set with enough coverage for 400 fps does; judged on pixels and video instead. |
| `forward_cached == forward` | bit-exact, including permuted rows | |
| G-TAESD (shared by r2–r5) | FAIL on max: 5 LSB vs the 3 LSB bar; mean 0.066 passes | Used from r2 on; your decision is pending. |
| T: six streams, 2 × 64 s | 401.8 / 400.1 fps | |
| SUST: five consecutive 64 s windows | 404.0 → 400.9 → 400.5 → 400.1 → 399.96 | The GPU warms from 64 to 67 °C and the SM clock drops from 2475 to 2460 MHz at the 220 W cap. There is no margin above 400. |
| PAIR: BEFORE / r5, back to back | 252.6 / 251.4 vs 401.4 / 400.1 | 1.59× |
| N15: 15 streams over 6 avatars, 10 × 63 s | aggregate 399.6–403.9 (median 399.8); per-stream clips min 26.56 fps, median 26.67; bit-exact; no errors | `stability_n15.md`. Watch the FaceMesh helper memory: +76–169 MiB per helper over 10 min, slowing down. |
| Quality vs BEFORE, 6 avatars | lip corr 0.9997–0.9998; aperture delta 0.08–0.14 px; landmarks 0.064–0.126 / 0.20–0.47 px; mean mouth PSNR 51–53 dB | All quality-tool gates pass except the landmark gate. At the strict bar (FaceMesh noise floor) it fails 6/6; at the proposed bar it passes 4/6. |

**Status fields.** `all_clips_match_accepted` is `false` for r2 and r5 by design: their outputs differ from the
pre-change renders. `deterministic_per_identity` must be `true`.

**Harness shutdown fix.** The harness was fixed on 2026-09-29 for a shutdown race, so new runs record no
`stop_warning`. The published runs all carried one, which is benign. Their `harness_code_sha256` for
`chin_multistream_render.py` is the pre-fix one.

## A rebuild is close, not identical

- **ONNX identical:** step 1 checks every block's ONNX hash against the published manifest; expect 11/11 MATCH.
  Checked on 2026-09-29 with a partial fresh-root build (`tail` FP16 and `up0` INT8 recipe): both MATCH the
  published r5 manifest, and the engine bytes differ, as expected.
  The recipe pins every INT8 layer's input range, and weight ranges are per-channel max, so the calibration data
  cannot change the r5 network.
- **Engines differ:** the plan files and their tactics differ, because TensorRT tactic timing varies (the builder
  notes up to ~5% per block). The published blocks were also built cold in four separate roots with their own
  timing caches (base FP16, srcfp16, mixed, gmac_0.50), while the package builds one root with one cache. A leftover
  seed cache (`docs/fps_comparisons/4070s_300fps_20260927/unet_probe/tt16_timing_cache.bin`) would change tactics;
  `00_check.sh` warns about it.
- **Speed:** with no margin above 400, a rebuilt r5 may measure 398–402 fps. `30_benchmark.sh BENCH` times the
  rebuilt blocks interleaved with the published set, so tactic drift shows directly.
- **Portability:** engines are specific to the GPU model, driver (595.84 here), TensorRT and torch. Build on the
  machine that serves. To copy the published sets instead, use `cp -L` or `rsync -L`: `srcg50` and `srcmix` are
  symlink farms into `/workspace/MuseTalk/models`.

## Fresh machine: what the scripts need and don't create

The layout is fixed. The repo can live anywhere; the paths below are hard-coded in the harness, the quality tool and
the (unchangeable) chin workflow.

**1. This repo, committed.** Branch `main` (the `perf/300fps-4070s` work was merged in PR #2), including:
- this package;
- `scripts/build_unet_stagewise.py` (`--variant srccache`, `--int8-recipe`) and `scripts/unet_stagewise_trt.py`;
- `scripts/vae_fast_decoder.py`;
- the harness `scripts/chin_multistream*`;
- `scripts/quality_ab_metrics.py`, `video_lineage.py` and `video_signoff.py`;
- `scripts/int8_layer_study.py` and `bench_stagewise_blocks.py`;
- the recipe `docs/fps_comparisons/4070s_400fps_20260928/int8_study/recipe_gmac_0.50.json`.

`00_check.sh` verifies these are tracked.

**2. The main venv** `/workspace/.venvs/musetalk_trt_stagewise`, with torch 2.5.1+cu121, torch_tensorrt 2.5.0,
tensorrt-cu12 10.3.0, nvidia-modelopt 0.23.2, diffusers 0.30.2 and onnx 1.17.0 (pinned in
`requirements/constraints-cu121.txt`):

```bash
scripts/install_musetalk.sh --matrix cu121 --with-legacy-int8
```

**3. The FaceMesh venv** `/workspace/SoulX-FlashHead/.venv` (mediapipe 0.10.9). The chin workflow's tracker
hard-codes this path. Either:

```bash
scripts/install_musetalk.sh --with-chin-tools --chin-venv /workspace/SoulX-FlashHead/.venv
```

or symlink that path to your chin-tools venv.

**4. Weights.** `models/musetalkV15/unet.pth` (3,400,074,924 bytes), SD-VAE, Whisper and TAESD, from
`install_musetalk.sh`. `00_check.sh` checks the TAESD files by sha256 against the renders.

**5. The six prepared avatars** `/workspace/experiments/avatar_diversity_20260927/<id>/` (763 MB).
- `05_fetch_inputs.sh` restores them byte-exact from S3 (`trt-artifacts/repro-inputs/avatar-diversity-20260927/`).
  They cannot be regenerated: the portraits, the MiniMax H3 source videos and the Kokoro speech were made with
  external tools.
- Every harness and quality check compares against their `render.json` hashes; `00_check.sh` verifies them
  (`--deep` includes the mp4s).

**6. The UNet capture corpus** `calibration/unet_multi_avatar_20260928`: 352 main + 96 holdout bs8 captures,
218 MB. `05_fetch_inputs.sh` restores it from S3 (it is also inside the r5 engine bundle). Rebuilding it
(`scripts/build_unet_multi_avatar_corpus.py`) needs the 14 source avatars its manifest names.

**7. The runtime env file** `.runtime/musetalk_trt_local_sm89.env`. Every harness run reads it. The first harness
step installs the recorded values from `scripts/repro_400fps/musetalk_trt_local_sm89.env` when it is absent, and
`00_check.sh` compares an existing one with them.

**8. The BEFORE engine** `models/tensorrt_unet_sm89_bs8_local/unet_trt.ts` (2.1 GB). Only the PAIR step needs it.
Build it with `scripts/unet_engine_store.py` (docs/STARTUP.md §6; ~7 min, ~10 GB RAM). A rebuilt one is not the
published engine, so its BEFORE numbers are "like-for-like" only approximately.

**9. The r2 video capture** for the BEFORE | r2 | set videos. Its raw arrays are gitignored. Without it, build r2
(`10_build_engines.sh --set r2`), capture it (`30_benchmark.sh models/tensorrt_unet_stagewise_sm89_r2 Q V`) and pass
`docs/fps_comparisons/repro_400fps/chin_multistream/V_repro_r2` to `40_videos.sh`.

**10. Host.**
- **GPU:** RTX 4070 SUPER at its 220 W power limit (every number above; a lower cap lowers them).
- **Memory and disk:** ≥ 16 GB free RAM for the builds and the 15-stream run, ≥ 5 GB free `/dev/shm` (the harness
  maps a ~0.6 GB arena per avatar there; in Docker use `--shm-size=8g` or more), and ≥ 4 GB free disk (a fresh set
  is ~1.1 GB of plans).
- **Tools and CPU:** ffmpeg with libx264, flock and setsid, and a writable `/workspace`. The harness uses about 5
  cores of the 32 threads here.

## Serving these engines

**Which set is served.** The default recipe r5 serves `models/tensorrt_unet_stagewise_ampere_plus_r5`: the same
recipe and the same ONNX per block as the published `..._srcg50` set, built with TensorRT hardware compatibility
`AMPERE_PLUS` so one set of plans runs on every GPU of compute capability 8.0+ (RTX 3090 included). Rebuild it with
`10_build_engines.sh --hardware-compat ampere_plus`; `20_gate.sh` and `30_benchmark.sh` then gate and measure it with
the hardware-compatible TAESD engine automatically (they read the set's `hardware_compatibility_level`). The
measured cost against the sm_89-only set is in `docs/trt_artifacts/README.md`.

On a fresh instance `scripts/vast_onstart.sh` restores that set from the pinned S3 bundle
(`configs/trt_bundles/ampere-plus-r5-srcg50-int8.json`), and the resolver points the server at it only when the
restore verified and the GPU is in the bundle's range. A set rebuilt by `10_build_engines.sh` is a different set (its
engine hashes differ): serve it through an overrides file (`docs/STARTUP.md` §5), not through the recipe, or publish
it as a new bundle with its own descriptor.

The live WebRTC test ran with these engines on 2026-09-29 (`docs/fps_comparisons/live15_r5_20260929/README.md`):
10 concurrent calls pass with margin and 15 is at the knee; the limit is the server's single event loop, not the GPU.
The server path still calls `forward`, not `forward_cached`, so it recomputes the per-source prefix (~1.8% of UNet
time) until per-avatar prefix caching is wired into the scheduler.
