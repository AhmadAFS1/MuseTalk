# Round 1: trained bk_tiny — FAIL

**None of the student's 12 clips passes r5**; the full fp16 control passes **12/12**. The student is
also worse than today's phone-chain D metrics on every clip. This rejection is supported by r5 failures
independently of the additional zero-margin D comparison.

## Training and selection

- Code: `d8826442c5d1bd8772185a7fc564eddd98faf19f`, branch `25fps-training-lip-network`.
- One RTX 3090; environment: `{'device': 'NVIDIA GeForce RTX 3090', 'fp16_device': 'NVIDIA GeForce RTX 3090', 'torch': '2.5.1+cu121', 'diffusers': '0.30.2'}`.
- bk_tiny, 318.2M parameters, initialized strictly from the full teacher's retained tensors.
- 30,000 steps, **116.7 minutes**, batch 8, gradient checkpointing, LR 1e-4 with 500-step warmup and cosine decay.
- Ten training characters (2,400 frames), two validation characters (256 fixed selection pairs), nine training
  audio clips (3,256 windows). All 14 gate identities and six review identities, including their other poses,
  are excluded from training/selection. Both review audios remain held out.
- Selection uses validation only: **step 28,250**, validation MAE **0.07585432**;
  corresponding 128-frame gate MAE **0.09673169**, max absolute error **4.85269**.
  The initial gate MAE was 0.30561042; teacher fp16 gate MAE was 0.00027554.
- All 121 evaluations, including step 0, are in [train/log.jsonl](train/log.jsonl).
- Best checkpoint SHA256: `9bce3e82144b65d20cdb5b022c290ac0ebe34dcefae8beccad64054c7473e724`.
- Local, untracked weights/config: `models/ios_bench/distilled/round1_bk_tiny_{best,last}.pt` and
  `models/ios_bench/distilled/round1_bk_tiny.json`.

## Held-out video review

Six avatars × two clips, 250 frames per clip at 25 fps, matching the existing D baseline's 10-second protocol.
A = full fp32 with TF32 off; N = full fp16; S = student fp32; S16 = student fp16.
N passes r5 on 12/12; S and S16 pass on 0/12. The gate requires aperture correlation ≥0.97, zero lag,
mean aperture error ≤0.5 px, mouth/face flicker ratio ≤1.05, and sharpness ratio ≥0.95.

| Student metric | Range across 12 clips | Mean |
|---|---:|---:|
| Lip aperture correlation | 0.53769–0.96255 | 0.85306 |
| Mean lip aperture error (px) | 1.34410–3.29190 | 2.00797 |
| Mouth PSNR (dB) | 26.62000–31.35000 | 28.70417 |
| Mouth sharpness ratio | 0.75420–1.04270 | 0.92087 |

Japanese clips additionally lag by 1–2 frames. Every clip fails aperture correlation and aperture error;
several also fail sharpness. Full metrics and D comparisons are in [review/summary.json](review/summary.json)
and [review/summary.md](review/summary.md), with each clip's detailed metrics in its subdirectory.
The six D comparisons use the zero deterioration margins declared before training. N's D comparisons are
informational; some D ratio measurements are slightly below the reference's 1.0 and therefore even the N
control can miss a zero-margin ratio bound. The student's rejection does not depend on those small differences.

## Selected comparisons

- [en_01_maya_abe451d3_talking_f6cf8be7fb / eng](review/en_01_maya_abe451d3_talking_f6cf8be7fb__eng/en_01_maya_abe451d3_talking_f6cf8be7fb__eng_AS.mp4)
- [en_03_marcus_19eb1ea8_talking_ada35a4a4d / ai-assistant](review/en_03_marcus_19eb1ea8_talking_ada35a4a4d__ai-assistant/en_03_marcus_19eb1ea8_talking_ada35a4a4d__ai-assistant_AS.mp4)
- [japanese_realtime_talking_7d94520b7f / ai-assistant](review/japanese_realtime_talking_7d94520b7f__ai-assistant/japanese_realtime_talking_7d94520b7f__ai-assistant_AS.mp4)
- [es_03_mateo_884600e1_talking_75d88cda6c / ai-assistant](review/es_03_mateo_884600e1_talking_75d88cda6c__ai-assistant/es_03_mateo_884600e1_talking_75d88cda6c__ai-assistant_AS.mp4)

These presentation videos were re-encoded to H.264 CRF 26 at unchanged resolution, frame count, timing and audio
for compact sharing. Numerical metrics were measured from the original raw frames, before video compression.
The complete source archive remains local under `.runtime/mtboot-distill/dt-r1-20261103a/results.tar`.
[The curation manifest](curation_manifest.json) records source/delivery hashes and the unchanged checkpoint hash.

## Rental and next step

Instance **59051256** (`mtboot-dt-r1-20261103a`) used stock launch, worker revision 4e2b02e, one RTX 3090,
$0.2747743056/h, and a pre-create four-hour expiry timer. Create requests: **1**. Results and weights were
pulled before cleanup. The provider reports **absent** and the timer **inactive**; see
[rental_receipt.json](rental_receipt.json). Estimated spend: **$0.7629**
(time plus quoted-rate container transfer; not invoice-confirmed).

The permitted second attempt changes only the step cap to 100,000, keeping bk_tiny and the 150-minute limit.
It starts fresh from the same initialization and seed; it does not alter or reset the round-1 resume manifest.
Expected actual training is about 38,000–40,000 steps. See [the round-2 plan](../round2/plan.md).
Phone inference is measured after the final training attempt. The historical untrained 24-fps sweep is not
claimed as a trained 25-fps result. Lingua remains unchanged.
