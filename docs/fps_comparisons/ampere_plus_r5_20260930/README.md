# r5 built for every Ampere+ GPU (TensorRT `AMPERE_PLUS`), 2026-09-30

**Why.** The r5 engines (`models/tensorrt_unet_stagewise_sm89_srcg50` + TAESD TRT `6111388248264a4ef2ae`) are
TensorRT plans for the GPU they were built on: they load only on an RTX 4070 SUPER (sm_89). Workers often run on
RTX 3090s (sm_86), so the default recipe needs engines that load on any GPU. TensorRT's hardware compatibility
level `AMPERE_PLUS` builds one set of plans that loads on every GPU of compute capability 8.0 or newer with the
same TensorRT version.

**Result.** Same accuracy, 23% less throughput on the RTX 4070 SUPER. The default recipe therefore uses these
portable engines only where no GPU-specific bundle fits: `configs/recipes/r5.env` lists
`bundle:rtx4070super-r5-srcg50-int8|ampere-plus-r5-srcg50-int8`, the first candidate that fits a host wins.

| | sm_89 set (published r5) | AMPERE_PLUS set (this build) |
|---|---|---|
| Loads on | RTX 4070 SUPER only | any GPU of compute capability 8.0-9.0, TensorRT 10.3.0 (RTX 3090, 4070 SUPER, 4090, A-series, L40S, A100, H100) |
| ONNX per block | - | identical to the sm_89 set (11/11 hashes match: same graph, same INT8 scales) |
| UNet per bs16 call (BENCH, interleaved, this GPU) | 31.19 ms | 40.59 ms (+30%) |
| Full recipe, 6 streams (T: 2 x >= 60 s) | 401.8 / 400.1 fps (published) | 307.7 / 307.1 fps (-23%) |
| Repo UNet gate main, mae / max_abs | 0.0044 / 0.78 | 0.0047 / 0.735 |
| Repo UNet gate holdout, mae / max_abs | 0.0039 / 1.26 | 0.0042 / 1.257 |
| `forward_cached == forward` (125 frames, permuted rows) | PASS | PASS |
| Probe vs eager FP16, rel_l2 | 0.0034 | 0.0039 |
| G-TAESD (3584 frames) | FAIL: max 5 LSB (bar 3), mean 0.066 | FAIL: max 5 LSB, mean 0.066 (identical) |
| Engines on disk | 1,060 MiB + TAESD 3.2 MiB | 1,103 MiB + TAESD 3.4 MiB |
| Build time on this GPU | ~15 min (seeded timing cache) | 40 min (no seed: every tactic timed) |

Per block (ms, sm_89 -> AMPERE_PLUS): down0rest 4.37 -> 5.35, down1 2.48 -> 3.17, down2 1.79 -> 2.14,
down3 0.24 -> 0.32, mid 0.50 -> 0.55, up0 0.83 -> 1.00, up1 4.43 -> 6.16, up2 6.23 -> 8.66, up3 10.21 -> 13.10,
tail 0.10 -> 0.13. FP16 and INT8 blocks alike lose 22-39%: in hardware-compatible mode TensorRT excludes the
architecture-specific (sm_89) kernels and runs Ampere-generic ones.

The two sets differ from each other on the probe by rel_l2 0.0040 (max_abs 0.063): the same size as each set's INT8
error against eager, i.e. ordinary tactic-to-tactic variation. `all_clips_match_accepted` is false for both, as for
every INT8 set (it compares against the pre-INT8 accepted renders).

## What was built and how

```bash
scripts/repro_400fps/10_build_engines.sh --hardware-compat ampere_plus     # = the three steps below + TAESD
B="python scripts/build_unet_stagewise.py --batch 16 --opt-level 5 --variant srccache \
   --root models/tensorrt_unet_stagewise_ampere_plus_r5 --calib-dir calibration/unet_multi_avatar_20260928 \
   --hardware-compat ampere_plus"
$B --blocks down0rest,up3,tail                                                         # 10 min
$B --blocks down1,down2,down3,mid,up0,up1,up2 \
   --int8-recipe docs/fps_comparisons/4070s_400fps_20260928/int8_study/recipe_gmac_0.50.json   # 29 min, peak RSS 10.8 GB
$B --blocks prefix                                                                     # 1 min, finalises
MUSETALK_TAESD_TRT_BATCH=8 MUSETALK_TAESD_TRT_HW_COMPAT=ampere_plus python scripts/vae_fast_decoder.py build  # 40 s
MUSETALK_REPRO_OUT=docs/fps_comparisons/ampere_plus_r5_20260930 scripts/repro_400fps/20_gate.sh models/tensorrt_unet_stagewise_ampere_plus_r5
MUSETALK_REPRO_OUT=docs/fps_comparisons/ampere_plus_r5_20260930 scripts/repro_400fps/30_benchmark.sh models/tensorrt_unet_stagewise_ampere_plus_r5 BENCH T
```

Code that makes the set portable:
- `scripts/unet_stagewise_trt.py`: `build_engine_from_onnx(..., hardware_compat="ampere_plus")` sets
  `IBuilderConfig.hardware_compatibility_level = AMPERE_PLUS`; the manifest records `hardware_compatibility_level`;
  the loader then accepts any device of compute capability >= 8.0 instead of the build GPU's exact one.
- `scripts/vae_fast_decoder.py`: `MUSETALK_TAESD_TRT_HW_COMPAT=ampere_plus` builds both TAESD plans the same way; the
  engine key then omits the GPU name and compute capability (the key is `512bfd629a5e1f4f2e40` on every GPU). The
  exported ONNX that the key hashes is device-independent: a CPU export and a GPU export are byte-identical.
- Load checks on another GPU model: the UNet and TAESD probes stay bit-exact on the build GPU; on any other GPU the
  plans may round differently, so both loaders accept a relative-L2 bound of 0.01 against the recorded probe output
  instead (manifest `probe.cross_gpu_rel_l2_max`, TAESD meta `probe.cross_gpu_rel_l2_max` + `*.probe_fp16.pt`).
  2.5x the tactic-to-tactic difference above; a wrong or broken engine is O(1). Plan sha256 checks are unchanged.

## Served through the real launcher

`scripts/run_musetalk_server.sh` with no `MUSETALK_RECIPE` on this box, `/health` reached both times:
- default: recipe r5, bundle `rtx4070super-r5-srcg50-int8` (the first candidate fits this GPU and is restored),
  `TAESD TRT backend: key=6111388248264a4ef2ae ... probe=exact hw_compat=none`, `UNet backend active:
  tensorrt_unet_stagewise`;
- with a recipe file that lists only the portable candidate: bundle `ampere-plus-r5-srcg50-int8`,
  `TAESD TRT backend: key=512bfd629a5e1f4f2e40 ... probe=exact hw_compat=ampere_plus`, `UNet backend active:
  tensorrt_unet_stagewise`, 0 resolver warnings.

## Not measured here

- **No sm_86 (RTX 3090) run yet.** This box has only the 4070 SUPER. That the plans load there is TensorRT's
  hardware-compatibility guarantee plus the loader checks above; the fps on a 3090 is unknown (the 3090 has
  similar tensor throughput, 1.9x the memory bandwidth and 1/8 of the L2). First boot of a 3090 worker shows it:
  `scripts/vast_onstart.sh` logs `r5 engine bundle ampere-plus-r5-srcg50-int8 ready`, then
  `Recipe verification passed: vae=taesd_trt unet=trt_stagewise`, and the server log records the probe status.
- **A 3090-native bundle** (sm_86 plans, likely faster than the portable ones there, as on the 4070 SUPER) needs a
  build on a 3090: `10_build_engines.sh` there, then a bundle + descriptor like `rtx4070super-r5-srcg50-int8` and
  a place in the candidate list before the portable one (`docs/trt_artifacts/README.md`).
