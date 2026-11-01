# Labelled quality review: Neural Engine UNet vs the PyTorch reference

12 clips, 6 production characters (en_01_maya_abe451d3_talking_f6cf8be7fb, en_03_marcus_19eb1ea8_talking_ada35a4a4d, es_01_valeria_aed74c50_talking_003d585d96, es_03_mateo_884600e1_talking_75d88cda6c, japanese_realtime_talking_7d94520b7f, latina_guided_20260925_talking_84c5bc80b8), 10 s each at 25 fps.
Arm A is the PyTorch fp32 UNet with TAESD (the reference the repo's gate uses, in fp32); arm B is the two-chunk
Core ML int8 UNet on the Neural Engine with Core ML TAESD: the chain the iPhone runs, bit-identical to the
iPhone 12 Pro's output on the gate frames. Both arms share the avatar cache, the Whisper windows and the blend.
A third arm, N (PyTorch fp16 on the Mac GPU, the server's precision), is not in the clips but is in
`metrics.md`: it shows how large a difference the shipping server already accepts, metric by metric.

This follows how r5 (the INT8 TensorRT recipe) was accepted: labelled A/B clips per identity with a mouth zoom
and a raw-face difference panel, a per-identity metrics table, and a human verdict on the whole set.

## What the reviewer does

1. **Blind pass.** Open the clips in `review_sheet.csv` order (`blind_video` column; paths are relative to the repo
   root, clips live under `experiments/video_validation/ane_vs_fp16/`). Each shows LEFT | RIGHT full frame with
   audio and a 3x mouth zoom. Which side is A or B is randomised per clip and recorded only in `key/blind_key.json`;
   do not open the key before finishing the sheet. Watch each clip twice, once at the mouth zoom.
2. For each clip fill the sheet: for lip sync, mouth artefacts (teeth, blur, ghosting, texture), flicker, and
   seam/blend, write the worse side (`L`, `R`, or `none`) and a severity: 0 = cannot tell, 1 = slight (visible
   when looking for it), 2 = clear, 3 = unacceptable. Then an overall worse side and severity, and notes.
3. **Unblind** with `key/blind_key.json` and count, per identity, the verdicts that fell on B.
4. **Labelled pass.** Open the `_ab.mp4` clips (A left, B right, labels, mouth zoom, |A-B| x8 face panel and the
   metrics line) for every clip where B was called worse, to see what the difference is.

## Acceptance rule (r5's: every identity acceptable, no systematic artefact)

B is **accepted** when, after unblinding, (a) no clip has a severity ≥ 2 attributed to B in any column, (b) every
identity has at least one clip where B was not called worse overall, and (c) across all clips B is called worse
overall at most 3 times more often than A (with 12 clips, side picks under indistinguishability split
about evenly; a run of B-worse calls is the signal). If (a) fails on one identity only, B is **accepted with a
reservation** for that character. Otherwise B is **rejected** and the labelled clips plus `metrics.md` say which
component (UNet precision: compare the B and N columns; decoder: compare `B_vs_A` with `Btd_vs_A` in
`metrics.json`) carries the difference.

## Files

- `review_sheet.csv` — blank sheet, one row per clip in blind order.
- `key/blind_key.json` — LEFT/RIGHT assignment per clip (open after the blind pass).
- `metrics.md`, `metrics.json` — per-clip metrics for B and the fp16 noise arm N against A; r5's thresholds
  (aperture corr ≥ 0.97, lag 0, mean |Δ| ≤ 0.5 px, flicker ratio ≤ 1.05, sharpness ratio ≥ 0.95).
- `stills/` — one labelled frame per clip at 3 s and 7 s.
- Clips and raw material: `experiments/video_validation/ane_vs_fp16/<identity>__<audio>/` (A.mp4, B.mp4, N.mp4,
  `*_ab.mp4`, `*_blind.mp4`, faces_A/B.npz, latents_*.npy, landmarks_*.npy). Rebuild with
  `scripts/ios/review_ane_vs_fp16.py`.
