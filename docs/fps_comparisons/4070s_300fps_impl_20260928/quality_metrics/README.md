# Quality A/B metrics: the quality half of the 300 fps goal

28 September 2026. Tool: [`scripts/quality_ab_metrics.py`](../../../../scripts/quality_ab_metrics.py), with the FaceMesh helper [`scripts/quality_ab_facemesh.py`](../../../../scripts/quality_ab_facemesh.py). The tool compares a candidate render (B) against the accepted pre-change render (A) of the same identity and audio. It covers lip sync, flicker, face masking and seam, chin alignment, and overall image quality. Every comparison writes a JSON file (all numbers plus per-frame series) and a short markdown table (A, B, delta, threshold, result) to `runs/`. Tables grouped by identity are in `per_identity/`. [`calibration_summary.md`](calibration_summary.md) holds every calibration comparison.

Tags: **[M]** measured here, **[D]** a decision or definition from the plan or the task, **[I]** inferred.

## Summary

1. **The tool reproduces the accepted ground truth exactly [M].** For all 6 diversity identities it rebuilds the pre-encode refined frames from `faces.npz` + `generated_landmarks.npy` using the unchanged `chin.py` (sha `fd753e7d…`). The frames' SHA-256 equals `render.json` `raw_refined_sha256` in all 6. Other checks that match exactly:
   - protected-lip rows against `pixel_checks.json`;
   - warped blend alphas against `mask_samples.npz`;
   - the recomputed `chin_delta` against `chin_delta.npy`;
   - FaceMesh on `refined_raw.mp4` against validate_stage's `refined_landmarks.npy`, and against chin_fps_validation's `taesd_final_landmarks.npy` (japanese, latina);
   - re-tracking the generated faces with the render's Tracker procedure against `generated_landmarks.npy`.

   So a candidate only needs to supply its **generated faces** (a 240×256×256×3 uint8 `faces.npz`, 47 MB). The tool re-derives g, chin_delta, the refined compose and pixel_checks with the accepted code.
2. **Self comparisons are perfect [M]** (18 of 18): correlation 1, lag 0, 0 px deviation, identical SHA, ratio 1.000. FaceMesh is deterministic.
3. **Codec noise alone exceeds the proposed G-TRACK thresholds [M].** Two crf18 encodes of the same raw frames, or raw frames against their accepted crf18 encode, give:
   - jaw+lip landmark deviation of **0.24–0.35 px mean and 0.65–1.05 px p99**, against a gate of 0.05/0.15;
   - chin-target error shifts of −0.03 to +0.10 px, against a margin of 0.05.

   **E1 gates must therefore run on raw pre-encode frames.** The tool enforces this: under `--profile e1`, encoded input downgrades the landmark and chin gates to report-only and gives the verdict INCOMPLETE.
4. **The proposed G-TRACK thresholds (0.05 mean / 0.15 p99) sit at FaceMesh's own sensitivity floor [M].** In the synthetic test, noise is added to the generated faces, then g is re-tracked and the frame recomposed. Results:
   - Sparse ±1 LSB on 20% of samples (mean |Δ| 0.2 LSB, the G-TAESD mean bound) gives 0.034–0.083 px mean and 0.10–0.43 px p99. G-TRACK passes on 3 of 6 identities.
   - iid ±1 LSB gives 0.045–0.088 / 0.13–0.34. It passes on 1 of 6.
   - iid ±3 LSB passes on 0 of 6.

   Every other gate passes these perturbations easily, apart from the seam-ring flicker ratio of 1.073 at ±3 LSB on one identity. The thresholds are kept as specified [D]. A calibrated alternative, **mean ≤ 0.10 / p99 ≤ 0.35 px**, is reported next to them as report-only (`chin.landmark_dev_calibrated_proposal`). It passes every 1-LSB floor and still fails codec noise and every known visible change. **Choosing between the two is for you to decide, not this tool.**
5. **Sensitivity [M].** The known visible differences all fail at least one gate:
   - Accepted refined vs standard (chin100 on/off, 6 identities): landmark deviation 0.63–1.10 px mean and 2.4–5.0 px p99. The chin shifts by 0.5–1.8 px. Seam-band per-frame max p99 is 66–121 RGB levels. Aperture mean |Δ| is 0.42–0.80 px. Mouth sharpness ratio falls to 0.87 in the worst case.
   - Accepted INT8 vs TAESD (japanese, latina): mouth sharpness ratio 0.79–0.84 (TAESD is softer), aperture mean |Δ| 0.62–0.65 px, landmark deviation 0.47–0.50 / 1.38–1.41 px, face PSNR 39.7–39.9 dB, mouth PSNR 34.7–35.1 dB.
   - Lip correlation (≥0.989) and chin error do **not** separate these pairs. Aperture |Δ|, landmark deviation, sharpness and the seam band do.
   - A temporal shift of 1 or 2 frames is caught by the lag gate; see [Temporal shift](#temporal-shift-sensitivity).
6. **SyncNet [M, uncalibrated]** separates matched audio from mismatched audio. On the H3 source (mouth not driven by this audio), confidence is 0.04–0.05. On MuseTalk renders it is 0.17–0.66. Refined vs standard confidence differs by −0.022 to +0.003. INT8 vs TAESD differs by −0.006 / +0.003. Treat these values as relative only.

## Running it

```bash
cd /workspace/MuseTalk-perf300
PY=/workspace/.venvs/musetalk_trt_stagewise/bin/python          # FaceMesh runs in /workspace/SoulX-FlashHead/.venv via the helper
ID=/workspace/experiments/avatar_diversity_20260927/black_woman

# Candidate that saved only its generated faces (recommended for E1 levers: exact raw path)
$PY scripts/quality_ab_metrics.py pair --identity-dir $ID --a dir=$ID \
    --b faces=/path/to/candidate_faces.npz,label=trt_taesd --profile e1 --name black_woman__trt_taesd

# Candidate with its own render_stage-style directory (faces.npz, generated_landmarks.npy, chin_delta.npy, refined_raw.mp4, render.json)
$PY scripts/quality_ab_metrics.py pair --identity-dir $ID --a dir=$ID --b dir=/path/to/cand_dir --profile e1 --name ...

# Encoded-only candidate (lossy arms; E1 landmark/chin gates are not valid on encoded frames)
$PY scripts/quality_ab_metrics.py pair --identity-dir $ID --a dir=$ID --b video=/path/b.mp4,g=/path/b_generated_landmarks.npy \
    --frames video --profile lossy --name ...

# Optional SyncNet (GPU, small): add --syncnet and run under the guard
scripts/box_guard.sh run --min-avail-gb 6 --wait-min 90 --label qm_syncnet -- $PY scripts/quality_ab_metrics.py pair ... --syncnet

$PY scripts/quality_ab_metrics.py calibrate        # all calibrations below, CPU only, ~27 min, peak RSS 2.6 GB
$PY scripts/quality_ab_metrics.py report           # rebuild calibration_summary.* and per_identity/*.md from runs/*.json
scripts/box_guard.sh run --min-avail-gb 6 --wait-min 90 --label qm_syncnet -- $PY scripts/quality_ab_metrics.py syncnet-calibrate
```

- **Arm keys:** `dir=`, `compose=refined|standard|source` (`source` is the earlier SourceMask correction), `video=`, `faces=`, `g=`, `chin_delta=`, `raw=` (a .npy or .npz of T×H×W×3 BGR uint8 pre-encode frames), `retrack=1`, `label=`.
- **Identity:** `--identity-dir` (source.mp4, source_landmarks.npy, cache.pt, masks.npz, speech.wav), or `--source --source-landmarks --cache --masks --audio`.
- **Exit code:** 0 only on PASS.
- **Python API** for in-memory candidates: `Identity.from_dir`, `Arm(...)` with `.frames` / `.g_arr` set, then `compare(ident, A, B, mode="raw")`. See `shift_sensitivity` in this README.
- **Memory:** about 1.8 GB RSS per raw pair. The tool waits when MemAvailable is below 4 GB (box_guard's definition) and aborts after 15 minutes.

## Metrics

All metrics use all 240 frames unless noted. A and B always share the same regions in each frame, built from the source landmarks plus both arms' tracked landmarks.

| # | Metric | Definition |
|---|---|---|
| 1 | Inner-lip aperture | Mean of FaceMesh distances 13–14, 82–87, 312–317, 81–178, 311–402, on the arm's final frames tracked with the workflow's exact FaceMesh config. Reported in px and in eye spans (the source `chin.axes` span). Per arm: stats and the full series. |
| 1 | A vs B | Pearson of the normalised series. Lag = argmax of Pearson(a[t], b[t+k]) for k ∈ [−6, 6], with ties going to the smallest \|k\|. Mean and p95 \|Δ aperture\| px. |
| 1 | SyncNet (optional) | LatentSync 16-frame pixel SyncNet (`models/syncnet/latentsync_syncnet.pt`, `configs/training/syncnet.yaml`). Identity face box → 256² RGB [−1, 1], lower half, 16 frames. Mel windows of 52 frames (`musetalk/data/audio.py`) at offsets ±8 frames, windows every 2 frames. Confidence = max − median of the mean cosine-similarity curve; offset = its argmax. Uncalibrated: trained at 25 fps, and this video is 24 fps. |
| 2 | Flicker | Per arm, the mean RGB \|f(t) − f(t−1)\| and the second difference \|f(t+1) − 2f(t) + f(t−1)\|. Measured in: the mouth ROI (lip box of source + both arms, padded 0.12/0.10 eye spans); the chin/jaw band (lower jaw polylines 58…152…288, ±0.08 eye spans); the seam ring; the face box. Also the temporal power above 6 Hz per pixel in fixed union mouth and jaw boxes. Reported as the ratio B/A. |
| 3 | Seam / masking | Per frame, max and mean \|A − B\| inside the feather band (0 < α < 255 of either arm's effective full-frame alpha), along the ring (50% alpha contour dilated 3 px) and outside the mask. The alpha matches render_stage's mask_samples: RefinedMask.current(g, delta) placed in the face box, then warp_roi. Reported as mean, p99 and max over frames, plus series. |
| 3 | Protected lips | Raw mode only, per arm. Max RGB difference between the arm's compose and its **own** `chin.standard` compose, inside the generated-lip hull dilated by max(4, round(0.06 span)). These are exactly the render_stage pixel_checks semantics, and the rows are verified identical to `pixel_checks.json`. |
| 4 | Chin target | validate_stage.py: error = (q[152] − g[152])·down(source), with q = FaceMesh on the final frames and g = the arm's generated landmarks. Frames 24–215. Target abs error, positive excess, lower lip to chin, source chin error, jaw step, all per arm. |
| 4 | Landmark deviation | Euclidean distance between tracked A and B over JAW(21) + LIPS(20) + inner lips(20) = 61 points, all frames: mean, p99, max, jaw and lip means. The generated-in-render deviation (the taesd_trt agent's G-TRACK definition) is also reported, report-only. |
| 5 | Overall | PSNR (per-frame mean capped at 100 dB, worst frame and global) and SSIM (Gaussian 11×11 σ 1.5 on luma, mean and worst) of A vs B, for the full frame, face box and mouth ROI. Mouth and face sharpness = mean Laplacian variance, ratio B/A. Colour = mean Lab of the face box per arm, ΔE76 of the means, and mean per-pixel ΔE76. |

## PASS thresholds

| Gate | Threshold | Source | Applies |
|---|---|---|---|
| `lip.aperture_corr` | ≥ 0.97 | plan L3 / task [D] | all |
| `lip.xcorr_lag_frames` | = 0 | task [D] | all |
| `lip.mean_abs_delta_px` | ≤ 0.5 px | plan L3 [D] | all |
| `flicker.mouth_ratio`, `flicker.jaw_ratio`, `flicker.ring_ratio` | B/A ≤ 1.05 (first difference) | plan L6 / task [D]; ring is the same criterion applied to the seam | all |
| `flicker.hf_*_ratio` | ≤ 1.05 | report only | all |
| `chin.target_error_mean_px` | B ≤ A + 0.05 px (e1), + 0.20 px (lossy) | G-TRACK / L4 / task [D] | raw frames for e1 |
| `chin.landmark_dev_mean_px`, `_p99_px` | ≤ 0.05 / ≤ 0.15 px | G-TRACK / task [D] | e1 and exact, raw frames only; report for lossy |
| `chin.landmark_dev_calibrated_proposal` | ≤ 0.10 / ≤ 0.35 px | this calibration [I], proposed | report only |
| `frames.raw_pre_encode` | frames mode = raw | this calibration (codec floor) [M] | e1 and exact |
| `seam.protected_lip_A`, `_B` | = 0, each arm against its own standard compose | pixel_checks semantics [D] | raw only |
| `overall.mouth_sharpness_ratio` | ≥ 0.95 (the 1.05 upper edge of the L5 band is reported) | L5 / task [D] | all |
| `overall.face_psnr_db` | report only | task [D] | all |
| `track.missing_face_frames` | = 0 | | all |
| `exact.frames_sha_identical` | pre-encode SHA identical | G-EXACT [D] | exact |

Verdicts:
- **PASS:** every applicable gate passes.
- **FAIL:** any applicable gate fails.
- **INCOMPLETE:** a required gate could not run (for example, protected lips or the e1 raw requirement on encoded input).

Self-video comparisons are INCOMPLETE for that reason alone.

## Calibration results

Range across identities (min .. max); [`calibration_summary.md`](calibration_summary.md) has every row, and `runs/<identity>__<case>.{json,md}` has the details:

| Category | n | Ap. corr | Lag | Ap. \|Δ\| px | Flk mouth | Flk jaw | Flk ring | Band max p99 | Chin B−A px | LM mean px | LM p99 px | PSNR face | PSNR mouth | SSIM mouth | Sharp B/A | ΔE76 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| self (video, raw, faces-only re-track) | 18 | 1.000 | 0 | 0 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 | 0 | 100 (identical) | 100 | 1.000 | 1.000 | 0 |
| codec (raw vs crf18; crf18 fast vs medium) | 12 | 0.998–0.999 | 0 | 0.23–0.38 | 0.985–1.000 | 0.92–1.01 | 0.93–1.01 | 22–30 | −0.03…+0.10 | 0.24–0.35 | 0.65–1.05 | 37.6–41.6 | 37.3–41.1 | 0.980–0.985 | 0.99–1.40 | 0.03–0.94 |
| synthetic sparse ±1 LSB, 20% (mean 0.2 LSB) | 6 | 1.000 | 0 | 0.03–0.07 | 1.001–1.002 | 1.000–1.002 | 1.001–1.004 | 2–6 | −0.005…+0.012 | 0.034–0.083 | 0.10–0.43 | 60.9–61.9 | 55.7–56.0 | 0.999 | 1.01–1.02 | 0.000 |
| synthetic iid ±1 LSB | 6 | 1.000 | 0 | 0.04–0.07 | 1.003–1.008 | 1.000–1.003 | 1.004–1.014 | 3–6 | −0.012…+0.013 | 0.045–0.088 | 0.13–0.34 | 56.9–57.8 | 51.5–51.9 | 0.997 | 1.04–1.07 | 0.000–0.001 |
| synthetic iid ±3 LSB | 6 | 1.000 | 0 | 0.07–0.12 | 1.020–1.044 | 1.004–1.015 | 1.020–1.073 | 4–11 | −0.014…+0.019 | 0.072–0.123 | 0.21–0.47 | 51.1–52.1 | 45.3–45.8 | 0.991–0.993 | 1.21–1.35 | 0.001–0.002 |
| refined vs standard (chin100 on/off) | 12 | 0.989–0.995 | 0 | 0.42–0.80 | 0.994–1.002 | 0.70–0.89 | 0.90–1.03 | 66–121 | −1.84…−0.48 | 0.63–1.10 | 2.4–5.0 | 32.8–39.7 | 37.7–50.3 | 0.974–0.996 | 0.87–1.00 | 0.05–0.23 |
| INT8 vs TAESD (accepted pair, lossy, video) | 2 | 0.992–0.993 | 0 | 0.62–0.65 | 0.92–0.95 | 0.99–1.00 | 0.96–0.99 | 27–29 | −0.06…−0.04 | 0.47–0.50 | 1.38–1.41 | 39.7–39.9 | 34.7–35.1 | 0.970–0.972 | 0.79–0.84 | 0.02–0.32 |

Other exact checks [M] ([`calibration_checks.json`](calibration_checks.json)):
- Re-encoding the raw frames with backend.encode's settings (crf18, preset fast, 2 threads) decodes bit-identically to the accepted `refined_raw.mp4` for all 6 identities. The accepted MP4s are therefore a deterministic function of the raw frames.
- FaceMesh tracking of `taesd_pipelined/aligned_raw.mp4` equals chin_fps_validation's saved landmarks, and its TAESD chin error of 1.0675 px equals `quality.json`.

What the calibration means:
- **Lip sync.** Correlation reached 0.989 even for chin100 on/off, and lag was always 0 (both arms use the same audio features). The correlation gate is therefore a coarse catastrophe detector. The discriminating lip metric is mean |Δ aperture|: about 0.05 px for E1-like noise, 0.23–0.38 px for codec noise, and 0.42–0.80 px for the known visible changes, against a gate of 0.5.
- **Flicker.** Adding noise raises the flicker ratios (up to 1.044 mouth and 1.073 ring at ±3 LSB), so the 1.05 gate responds to noise. The chin100 correction itself adds jaw-band temporal change: the standard/refined ratio is 0.70–0.89. Comparing standard (A) against refined (B) would give 1.12–1.42 and fail.
- **Chin.** Encoding moves the validate_stage chin error by up to 0.10 px, which is why e1 needs raw frames. E1-like noise moves it by at most 0.019 px. The gate is one-sided: refined vs standard passes it because the standard compose sits closer to the generated chin. The landmark deviation and seam-band metrics catch that case instead.
- **Sharpness.** Laplacian variance rises with codec noise (raw vs crf18 gives 1.26–1.40) and with additive noise (1.21–1.35 at ±3 LSB). So the ≥ 0.95 gate cannot catch artificial sharpening; the report-only upper edge of 1.05 flags it. It does catch the TAESD softening (0.79–0.84) and the chin-warp softening below the lips (0.87 worst).
- **PSNR and SSIM** give the scale. The codec floor is about 38–41 dB face. E1-like noise is at least 51 dB. Known visible changes are 33–40 dB.

### SyncNet
Per arm, from [`syncnet_calibration.json`](syncnet_calibration.json):

| Identity | Arm | Confidence | Offset (frames) | Cos-sim at 0 |
|---|---|---:|---:|---:|
| black_man_short_beard | refined / standard | 0.607 / 0.630 | 2 / 2 | 0.164 / 0.149 |
| black_woman | refined / standard | 0.518 / 0.516 | 3 / 3 | 0.143 / 0.133 |
| east_asian_man_goatee | refined / standard | 0.575 / 0.594 | 2 / 2 | 0.188 / 0.182 |
| middle_eastern_man_full_beard | refined / standard | 0.571 / 0.590 | 2 / 2 | 0.139 / 0.132 |
| south_asian_woman | refined / standard | 0.580 / 0.577 | 3 / 3 | 0.140 / 0.140 |
| white_man_clean_shaven | refined / standard | 0.622 / 0.629 | 1 / 1 | 0.476 / 0.467 |
| japanese | INT8 / TAESD / H3 source | 0.179 / 0.173 / 0.042 | −1 / −1 / 5 | 0.302 / 0.299 / 0.108 |
| latina | INT8 / TAESD / H3 source | 0.656 / 0.658 / 0.047 | 2 / 2 / −2 | 0.195 / 0.196 / 0.113 |

Offset k is the audio window, starting k frames after the video window, that matches best. The constant +1 to +3 offset of MuseTalk outputs is a property of the uncalibrated setup (the 24/25 fps window and the crop), not a measured A/V error. Compare only arms of the same identity.

### Temporal shift sensitivity
SHIFT_PLACEHOLDER

## Caveats

- **Landmark metrics are not perceptual scores.** FaceMesh in tracking mode reacts to sub-LSB input changes at the 0.03–0.09 px level (see above). Always pair these gates with the video review (plan L8).
- **Masks, ROIs and chin geometry use identity inputs** (cache boxes, masks.npz, source landmarks). Protected-lip and raw-SHA checks need `faces.npz`. Encoded-only candidates get every other metric.
- **Mixed modes.** The `raw_vs_video` codec comparisons are calibration only. For a candidate, analyse both arms the same way: both raw, or both encoded with the identical encoder.
- **Tool version.** The calibration runs record `tool_sha256` 93ea052e…. The final file adds only report-side code: the floors table, the SyncNet wording, and the report-only calibrated-proposal gate with `would_pass`. No metric computation changed.
- **Resource use.** CPU only (FaceMesh plus NumPy/OpenCV at 2 threads, `nice`), not under the GPU lease. SyncNet ran under box_guard: 15 s, peak group RSS 2.9 GB, oom_kill unchanged.
