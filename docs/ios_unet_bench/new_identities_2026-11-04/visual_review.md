# Independent quality review: 30 new static identities

The frozen student fails all 60 clips in both fp32 (S) and fp16 (S16). The full-model fp16 control N passes all 60. Every one of the 30 identities fails both held-out clips under each student precision. This is a complete negative quality result with a passing precision control, not an incomplete evaluation or a demonstrated deployment-ready improvement.

## Numerical and provenance checks

Read all 60 original per-clip metrics and the original review summary under `.runtime/new_identity_audit/raw/out/review/`. Verified exactly `ni30_01`–`ni30_30` × `ai-assistant`/`eng`, 250 frames at 25 fps, and source dimensions 512×896. All A/N/S/S16 landmark coverage counts are 250; all 180 scored arm/clip metric trees are finite, and all aperture comparisons contain 250 valid paired frames. There are no control failures or unscored clips.

Student SHA256 is `a05d2d98f69b431252458d3b25f72e395d557ba693821baee91d11f5fe5a96cd`; full teacher SHA256 is `7ebf6c98c181e20838e4c0054e96e944ac60d5d692cc01db42839fe11b787007`. Summary, prepared manifest, and job provenance agree; the prepared-manifest byte hash matches the summary and its full input record. Review code is `604ffac76177096a3ab236b1a66f35026553e523`, CUDA RTX 3090, torch 2.5.1+cu121, diffusers 0.30.2.

| Metric across 60 clips | N full fp16 | S student fp32 | S16 student fp16 |
|---|---:|---:|---:|
| r5 passes | 60/60 | 0/60 | 0/60 |
| Latent MAE range (mean) | 0.00037–0.00049 (0.0004155) | 0.08540–0.15087 (0.1176962) | 0.08539–0.15086 (0.1176950) |
| Lip correlation range (mean) | 0.99764–0.99999 (0.9997248) | 0.78306–0.96502 (0.9089885) | 0.78241–0.96513 (0.9091232) |
| Mean absolute aperture error, px: range (mean) | 0.0161–0.1680 (0.0536033) | 0.7846–4.5852 (1.7144083) | 0.7867–4.5869 (1.7140067) |
| Mouth PSNR, dB: range (mean) | 61.94–66.05 (64.9687) | 22.73–31.41 (26.6065) | 22.73–31.41 (26.6065) |
| Mouth sharpness ratio: range (mean) | 0.9996–1.0009 (1.0002267) | 0.4540–1.1089 (0.7766367) | 0.4540–1.1090 (0.7765700) |
| Mouth flicker ratio: range (mean) | 0.9996–1.0002 (0.9999350) | 0.6646–1.2363 (0.8620983) | 0.6646–1.2363 (0.8620433) |
| Mouth error flicker: range (mean) | 0.0316–0.0804 (0.0412517) | 2.0932–4.3386 (3.0071300) | 2.0933–4.3385 (3.0071350) |

Both student arms fail lip correlation and mean aperture error on **60/60** clips. Each also fails sharpness on **52/60**, zero-lag on **13/60**, mouth flicker on **2/60**, and face flicker on **2/60**. Lag distribution is 47 zero, 11 at −1 frame and two at +1 frame. N has zero lag and passes every gate on every clip. Student mean aperture is below A in 46 clips and above A in 14; its signed mean bias is approximately −0.792 px (S) / −0.793 px (S16). A flicker ratio below one is not a quality pass: reduced variation can accompany blurred or damped mouth motion.

Notable extremes:

- Lowest lip correlation: `ni30_11__ai-assistant`, S 0.78306 / S16 0.78241.
- Largest aperture error: `ni30_10__eng`, S 4.5852 px / S16 4.5869 px.
- Lowest sharpness ratio: `ni30_06__ai-assistant`, 0.4540 in both arms.
- Lowest mouth PSNR and largest mouth error flicker: `ni30_01__ai-assistant`, 22.73 dB and about 4.3386 / 4.3385.
- Largest latent MAE: `ni30_29__ai-assistant`, 0.15087 / 0.15086.

## Limited visual inspection

Viewed six existing worker A|S stills with `view_image`; these are selected frames, not a complete temporal/video review. Selection uses the authored character metadata and includes the first and last identities, glasses, varied ages and skin-tone descriptions:

- `ni30_01__ai-assistant_3s.jpg`: student shows a different mouth opening/teeth band and softer lip detail; the zoom exposes a clear shape mismatch.
- `ni30_06__ai-assistant_3s.jpg`: student mouth, teeth and surrounding lower-face texture are visibly smoother, with reduced edge detail, consistent with the low sharpness score.
- `ni30_11__ai-assistant_3s.jpg`: A is nearly closed while the student mouth remains visibly open at the same frame; lip contour/closure differs.
- `ni30_13__eng_7s.jpg`: the glasses and upper face remain aligned, while the student lips/teeth are blurrier and differently shaped. This supports a mouth-region difference rather than a whole-frame pairing/registration mistake in this inspected sample.
- `ni30_29__ai-assistant_3s.jpg`: mouth aperture shape and visible teeth differ; student lower-face detail is softer.
- `ni30_30__eng_3s.jpg`: student opening is smaller/more rounded with different tooth visibility and softened lip edges.

The same left-A/right-S labeling and mouth zoom are present in all six inspected stills. Static head/body and matching framing are apparent. The inspected stills are JPEG previews, and the 3× mouth zoom uses nearest-neighbor enlargement, so block edges/compression should not be treated as independent model defects. The amplified difference panels are diagnostic visualizations, not natural-looking output. Still inspection cannot independently establish flicker, acoustic synchronization or frame lag; those claims here come from the full numerical sequences. No S16 video frames were visually inspected in this review, so the visual observations apply to S only; S16's similarly poor quality is established numerically.

## Interpretation and artifact limits

The near-identical S/S16 metrics show that the large quality gap is already present in student fp32. These measurements do not isolate whether architecture capacity, data coverage, training duration or optimization choices caused it. No retraining or checkpoint selection was performed on this new set.

This evaluates 30 newly generated static frontal identities on two held-out clips. It does not establish universal performance, moving-head/pose-transition quality, multilingual coverage, personality quality, or demographic fairness. Skin-tone/age/presentation groups are authored sample descriptors. No historical shipped-chain D baseline exists for these identities. CUDA/PyTorch S16 is not quantized Core ML or iPhone ANE quality evidence, and the clips' encoded 25 fps is not a new phone throughput measurement.

r5 measures agreement with the full-model reference. A passing full-fp16 control establishes close precision agreement with A, not absolute ground-truth speech synchronization or a claim that A has no artifacts.

All metrics were computed from raw composed frames. Worker-delivery A|S videos are already CRF26 presentation copies; later local CRF30/768px videos with AAC 32 kbps presentation audio are additional presentation compression. Original S16 MP4 files are retained byte-for-byte. Raw unencoded landmarks were excluded from the original tar; the separately verified, read-only aperture-curve supplement preserves their derived 250-frame series without changing any scored metric or archive file.

No raw or tracked files were modified during this review. This note is ignored local evidence only.
