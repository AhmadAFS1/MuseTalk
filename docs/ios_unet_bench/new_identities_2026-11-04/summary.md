<!-- generated_by: musetalk-new-identity-summary-v1 -->
# Thirty new faces: full model versus trained student

Sixty frozen 10-second clips from 30 newly generated static portraits. A is the full fp32 reference; N is full fp16, S student fp32 and S16 student fp16.

| Arm | Strict verdict | Clips PASS / FAIL / INCOMPLETE | Identities with both clips passing |
|---|---|---|---|
| N | PASS | 60 / 0 / 0 of 60 | 30/30 |
| S | FAIL | 0 / 60 / 0 of 60 | 0/30 |
| S16 | FAIL | 0 / 60 / 0 of 60 | 0/30 |

Observed metrics: 60/60; available A | S comparisons: 60/60.

Full finite landmark coverage and all required numeric metrics are required. Known failed gates remain recorded even when a clip is INCOMPLETE.

## Control and student failures

- student failure with passing control: 60/60.

## Limits

- Static generated portraits: source head/body remain frozen; moving-source poses were not tested.
- These 30 authored identities and two clips cannot establish quality for every future character.
- No historical D baseline exists for these identities; D non-inferiority was not evaluated.
- S16 is PyTorch fp16 on the reported device, not an iPhone ANE or quantized Core ML measurement.
- Authored skin-tone, age and presentation groups are descriptive small samples, not a fairness proof or inferred ethnicity.
- The earlier round-2 12-clip student quality failure remains part of the record.

## Frozen models and provenance

- Teacher SHA256: 7ebf6c98c181e20838e4c0054e96e944ac60d5d692cc01db42839fe11b787007.
- Student SHA256: a05d2d98f69b431252458d3b25f72e395d557ba693821baee91d11f5fe5a96cd.
- Review code revision: 604ffac76177096a3ab236b1a66f35026553e523.
- Prepared manifest SHA256: 3f1fbdb0b7e8a6e6ae0433aad84a0b679d580441d5f7af47c5e69dc684578e8a.
- The checkpoint was frozen before these identities; this review does not train or select a model.
- Numerical metrics precede video compression. A | S videos compare the two fp32 arms.

## Metrics across 60 clips

| Arm | Metric | Finite n | Min | Max | Mean | Worst clip |
|---|---|---:|---:|---:|---:|---|
| N | lip correlation | 60 | 0.998 | 1.000 | 1.000 | ni30_23__ai-assistant |
| N | mean lip error, px | 60 | 0.016 | 0.168 | 0.054 | ni30_19__ai-assistant |
| N | lip lag, frames | 60 | 0.000 | 0.000 | 0.000 | ni30_01__ai-assistant |
| N | mouth PSNR, dB | 60 | 61.940 | 66.050 | 64.969 | ni30_08__eng |
| N | mouth flicker ratio | 60 | 1.000 | 1.000 | 1.000 | ni30_08__ai-assistant |
| N | face flicker ratio | 60 | 1.000 | 1.000 | 1.000 | ni30_02__ai-assistant |
| N | mouth sharpness ratio | 60 | 1.000 | 1.001 | 1.000 | ni30_26__ai-assistant |
| N | mouth error flicker | 60 | 0.032 | 0.080 | 0.041 | ni30_08__eng |
| N | latent MAE | 60 | 0.000 | 0.000 | 0.000 | ni30_08__ai-assistant |
| S | lip correlation | 60 | 0.783 | 0.965 | 0.909 | ni30_11__ai-assistant |
| S | mean lip error, px | 60 | 0.785 | 4.585 | 1.714 | ni30_10__eng |
| S | lip lag, frames | 60 | -1.000 | 1.000 | -0.150 | ni30_03__ai-assistant |
| S | mouth PSNR, dB | 60 | 22.730 | 31.410 | 26.607 | ni30_01__ai-assistant |
| S | mouth flicker ratio | 60 | 0.665 | 1.236 | 0.862 | ni30_11__ai-assistant |
| S | face flicker ratio | 60 | 0.678 | 1.224 | 0.863 | ni30_11__ai-assistant |
| S | mouth sharpness ratio | 60 | 0.454 | 1.109 | 0.777 | ni30_06__ai-assistant |
| S | mouth error flicker | 60 | 2.093 | 4.339 | 3.007 | ni30_01__ai-assistant |
| S | latent MAE | 60 | 0.085 | 0.151 | 0.118 | ni30_29__ai-assistant |
| S16 | lip correlation | 60 | 0.782 | 0.965 | 0.909 | ni30_11__ai-assistant |
| S16 | mean lip error, px | 60 | 0.787 | 4.587 | 1.714 | ni30_10__eng |
| S16 | lip lag, frames | 60 | -1.000 | 1.000 | -0.150 | ni30_03__ai-assistant |
| S16 | mouth PSNR, dB | 60 | 22.730 | 31.410 | 26.607 | ni30_01__ai-assistant |
| S16 | mouth flicker ratio | 60 | 0.665 | 1.236 | 0.862 | ni30_11__ai-assistant |
| S16 | face flicker ratio | 60 | 0.678 | 1.224 | 0.863 | ni30_11__ai-assistant |
| S16 | mouth sharpness ratio | 60 | 0.454 | 1.109 | 0.777 | ni30_06__ai-assistant |
| S16 | mouth error flicker | 60 | 2.093 | 4.338 | 3.007 | ni30_01__ai-assistant |
| S16 | latent MAE | 60 | 0.085 | 0.151 | 0.118 | ni30_29__ai-assistant |

## Per identity, clip and arm

Worst bounded gate ranks fractional threshold violations for correlation, aperture error, mouth/face flicker and sharpness. Lag is separate. This navigation aid does not change any acceptance gate.

| Identity / audio | Arm | r5 | Correlation | Mean lip error, px | Lag | Worst bounded gate | A \| S video |
|---|---|---|---:|---:|---:|---|---|
| ni30_01 / ai-assistant | N | PASS | 1.000 | 0.039 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_01 / ai-assistant | S | FAIL | 0.908 | 1.887 | 0 | mean lip error, px: 1.887 (required ≤0.5) | [play](review/ni30_01__ai-assistant/ni30_01__ai-assistant_AS.mp4) |
| ni30_01 / ai-assistant | S16 | FAIL | 0.908 | 1.889 | 0 | mean lip error, px: 1.889 (required ≤0.5) | — |
| ni30_01 / eng | N | PASS | 1.000 | 0.030 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_01 / eng | S | FAIL | 0.921 | 1.998 | 0 | mean lip error, px: 1.998 (required ≤0.5) | [play](review/ni30_01__eng/ni30_01__eng_AS.mp4) |
| ni30_01 / eng | S16 | FAIL | 0.921 | 1.997 | 0 | mean lip error, px: 1.997 (required ≤0.5) | — |
| ni30_02 / ai-assistant | N | PASS | 1.000 | 0.032 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_02 / ai-assistant | S | FAIL | 0.862 | 1.604 | 0 | mean lip error, px: 1.604 (required ≤0.5) | [play](review/ni30_02__ai-assistant/ni30_02__ai-assistant_AS.mp4) |
| ni30_02 / ai-assistant | S16 | FAIL | 0.862 | 1.602 | 0 | mean lip error, px: 1.602 (required ≤0.5) | — |
| ni30_02 / eng | N | PASS | 1.000 | 0.023 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_02 / eng | S | FAIL | 0.913 | 1.239 | 0 | mean lip error, px: 1.239 (required ≤0.5) | [play](review/ni30_02__eng/ni30_02__eng_AS.mp4) |
| ni30_02 / eng | S16 | FAIL | 0.913 | 1.234 | 0 | mean lip error, px: 1.234 (required ≤0.5) | — |
| ni30_03 / ai-assistant | N | PASS | 1.000 | 0.030 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_03 / ai-assistant | S | FAIL | 0.802 | 1.384 | -1 | mean lip error, px: 1.384 (required ≤0.5) | [play](review/ni30_03__ai-assistant/ni30_03__ai-assistant_AS.mp4) |
| ni30_03 / ai-assistant | S16 | FAIL | 0.803 | 1.380 | -1 | mean lip error, px: 1.380 (required ≤0.5) | — |
| ni30_03 / eng | N | PASS | 1.000 | 0.023 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_03 / eng | S | FAIL | 0.867 | 1.279 | -1 | mean lip error, px: 1.279 (required ≤0.5) | [play](review/ni30_03__eng/ni30_03__eng_AS.mp4) |
| ni30_03 / eng | S16 | FAIL | 0.867 | 1.280 | -1 | mean lip error, px: 1.280 (required ≤0.5) | — |
| ni30_04 / ai-assistant | N | PASS | 1.000 | 0.028 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_04 / ai-assistant | S | FAIL | 0.841 | 2.130 | 1 | mean lip error, px: 2.130 (required ≤0.5) | [play](review/ni30_04__ai-assistant/ni30_04__ai-assistant_AS.mp4) |
| ni30_04 / ai-assistant | S16 | FAIL | 0.841 | 2.128 | 1 | mean lip error, px: 2.128 (required ≤0.5) | — |
| ni30_04 / eng | N | PASS | 1.000 | 0.022 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_04 / eng | S | FAIL | 0.895 | 1.927 | 1 | mean lip error, px: 1.927 (required ≤0.5) | [play](review/ni30_04__eng/ni30_04__eng_AS.mp4) |
| ni30_04 / eng | S16 | FAIL | 0.895 | 1.930 | 1 | mean lip error, px: 1.930 (required ≤0.5) | — |
| ni30_05 / ai-assistant | N | PASS | 1.000 | 0.026 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_05 / ai-assistant | S | FAIL | 0.889 | 1.596 | 0 | mean lip error, px: 1.596 (required ≤0.5) | [play](review/ni30_05__ai-assistant/ni30_05__ai-assistant_AS.mp4) |
| ni30_05 / ai-assistant | S16 | FAIL | 0.889 | 1.595 | 0 | mean lip error, px: 1.595 (required ≤0.5) | — |
| ni30_05 / eng | N | PASS | 1.000 | 0.026 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_05 / eng | S | FAIL | 0.939 | 1.395 | 0 | mean lip error, px: 1.395 (required ≤0.5) | [play](review/ni30_05__eng/ni30_05__eng_AS.mp4) |
| ni30_05 / eng | S16 | FAIL | 0.939 | 1.395 | 0 | mean lip error, px: 1.395 (required ≤0.5) | — |
| ni30_06 / ai-assistant | N | PASS | 1.000 | 0.039 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_06 / ai-assistant | S | FAIL | 0.913 | 3.499 | 0 | mean lip error, px: 3.499 (required ≤0.5) | [play](review/ni30_06__ai-assistant/ni30_06__ai-assistant_AS.mp4) |
| ni30_06 / ai-assistant | S16 | FAIL | 0.913 | 3.499 | 0 | mean lip error, px: 3.499 (required ≤0.5) | — |
| ni30_06 / eng | N | PASS | 1.000 | 0.041 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_06 / eng | S | FAIL | 0.872 | 3.831 | 0 | mean lip error, px: 3.831 (required ≤0.5) | [play](review/ni30_06__eng/ni30_06__eng_AS.mp4) |
| ni30_06 / eng | S16 | FAIL | 0.872 | 3.834 | 0 | mean lip error, px: 3.834 (required ≤0.5) | — |
| ni30_07 / ai-assistant | N | PASS | 1.000 | 0.046 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_07 / ai-assistant | S | FAIL | 0.949 | 0.870 | 0 | mean lip error, px: 0.870 (required ≤0.5) | [play](review/ni30_07__ai-assistant/ni30_07__ai-assistant_AS.mp4) |
| ni30_07 / ai-assistant | S16 | FAIL | 0.949 | 0.873 | 0 | mean lip error, px: 0.873 (required ≤0.5) | — |
| ni30_07 / eng | N | PASS | 1.000 | 0.033 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_07 / eng | S | FAIL | 0.963 | 0.895 | 0 | mean lip error, px: 0.895 (required ≤0.5) | [play](review/ni30_07__eng/ni30_07__eng_AS.mp4) |
| ni30_07 / eng | S16 | FAIL | 0.963 | 0.897 | 0 | mean lip error, px: 0.897 (required ≤0.5) | — |
| ni30_08 / ai-assistant | N | PASS | 1.000 | 0.096 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_08 / ai-assistant | S | FAIL | 0.951 | 1.127 | 0 | mean lip error, px: 1.127 (required ≤0.5) | [play](review/ni30_08__ai-assistant/ni30_08__ai-assistant_AS.mp4) |
| ni30_08 / ai-assistant | S16 | FAIL | 0.951 | 1.140 | 0 | mean lip error, px: 1.140 (required ≤0.5) | — |
| ni30_08 / eng | N | PASS | 1.000 | 0.106 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_08 / eng | S | FAIL | 0.962 | 1.191 | 0 | mean lip error, px: 1.191 (required ≤0.5) | [play](review/ni30_08__eng/ni30_08__eng_AS.mp4) |
| ni30_08 / eng | S16 | FAIL | 0.962 | 1.188 | 0 | mean lip error, px: 1.188 (required ≤0.5) | — |
| ni30_09 / ai-assistant | N | PASS | 1.000 | 0.025 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_09 / ai-assistant | S | FAIL | 0.886 | 1.474 | -1 | mean lip error, px: 1.474 (required ≤0.5) | [play](review/ni30_09__ai-assistant/ni30_09__ai-assistant_AS.mp4) |
| ni30_09 / ai-assistant | S16 | FAIL | 0.886 | 1.472 | -1 | mean lip error, px: 1.472 (required ≤0.5) | — |
| ni30_09 / eng | N | PASS | 1.000 | 0.021 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_09 / eng | S | FAIL | 0.930 | 1.185 | 0 | mean lip error, px: 1.185 (required ≤0.5) | [play](review/ni30_09__eng/ni30_09__eng_AS.mp4) |
| ni30_09 / eng | S16 | FAIL | 0.930 | 1.184 | 0 | mean lip error, px: 1.184 (required ≤0.5) | — |
| ni30_10 / ai-assistant | N | PASS | 0.999 | 0.129 | 0 | lip correlation: 0.999 (required ≥0.97) | — |
| ni30_10 / ai-assistant | S | FAIL | 0.905 | 4.091 | 0 | mean lip error, px: 4.091 (required ≤0.5) | [play](review/ni30_10__ai-assistant/ni30_10__ai-assistant_AS.mp4) |
| ni30_10 / ai-assistant | S16 | FAIL | 0.906 | 4.086 | 0 | mean lip error, px: 4.086 (required ≤0.5) | — |
| ni30_10 / eng | N | PASS | 1.000 | 0.126 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_10 / eng | S | FAIL | 0.894 | 4.585 | 0 | mean lip error, px: 4.585 (required ≤0.5) | [play](review/ni30_10__eng/ni30_10__eng_AS.mp4) |
| ni30_10 / eng | S16 | FAIL | 0.895 | 4.587 | 0 | mean lip error, px: 4.587 (required ≤0.5) | — |
| ni30_11 / ai-assistant | N | PASS | 1.000 | 0.041 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_11 / ai-assistant | S | FAIL | 0.783 | 2.553 | -1 | mean lip error, px: 2.553 (required ≤0.5) | [play](review/ni30_11__ai-assistant/ni30_11__ai-assistant_AS.mp4) |
| ni30_11 / ai-assistant | S16 | FAIL | 0.782 | 2.564 | -1 | mean lip error, px: 2.564 (required ≤0.5) | — |
| ni30_11 / eng | N | PASS | 1.000 | 0.042 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_11 / eng | S | FAIL | 0.813 | 2.346 | -1 | mean lip error, px: 2.346 (required ≤0.5) | [play](review/ni30_11__eng/ni30_11__eng_AS.mp4) |
| ni30_11 / eng | S16 | FAIL | 0.813 | 2.345 | -1 | mean lip error, px: 2.345 (required ≤0.5) | — |
| ni30_12 / ai-assistant | N | PASS | 1.000 | 0.053 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_12 / ai-assistant | S | FAIL | 0.920 | 1.230 | 0 | mean lip error, px: 1.230 (required ≤0.5) | [play](review/ni30_12__ai-assistant/ni30_12__ai-assistant_AS.mp4) |
| ni30_12 / ai-assistant | S16 | FAIL | 0.920 | 1.223 | 0 | mean lip error, px: 1.223 (required ≤0.5) | — |
| ni30_12 / eng | N | PASS | 1.000 | 0.047 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_12 / eng | S | FAIL | 0.945 | 1.202 | 0 | mean lip error, px: 1.202 (required ≤0.5) | [play](review/ni30_12__eng/ni30_12__eng_AS.mp4) |
| ni30_12 / eng | S16 | FAIL | 0.946 | 1.197 | 0 | mean lip error, px: 1.197 (required ≤0.5) | — |
| ni30_13 / ai-assistant | N | PASS | 1.000 | 0.030 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_13 / ai-assistant | S | FAIL | 0.951 | 0.872 | 0 | mean lip error, px: 0.872 (required ≤0.5) | [play](review/ni30_13__ai-assistant/ni30_13__ai-assistant_AS.mp4) |
| ni30_13 / ai-assistant | S16 | FAIL | 0.951 | 0.872 | 0 | mean lip error, px: 0.872 (required ≤0.5) | — |
| ni30_13 / eng | N | PASS | 1.000 | 0.027 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_13 / eng | S | FAIL | 0.952 | 1.037 | 0 | mean lip error, px: 1.037 (required ≤0.5) | [play](review/ni30_13__eng/ni30_13__eng_AS.mp4) |
| ni30_13 / eng | S16 | FAIL | 0.952 | 1.036 | 0 | mean lip error, px: 1.036 (required ≤0.5) | — |
| ni30_14 / ai-assistant | N | PASS | 1.000 | 0.048 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_14 / ai-assistant | S | FAIL | 0.939 | 1.048 | 0 | mean lip error, px: 1.048 (required ≤0.5) | [play](review/ni30_14__ai-assistant/ni30_14__ai-assistant_AS.mp4) |
| ni30_14 / ai-assistant | S16 | FAIL | 0.939 | 1.048 | 0 | mean lip error, px: 1.048 (required ≤0.5) | — |
| ni30_14 / eng | N | PASS | 1.000 | 0.043 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_14 / eng | S | FAIL | 0.965 | 0.785 | 0 | mean lip error, px: 0.785 (required ≤0.5) | [play](review/ni30_14__eng/ni30_14__eng_AS.mp4) |
| ni30_14 / eng | S16 | FAIL | 0.965 | 0.787 | 0 | mean lip error, px: 0.787 (required ≤0.5) | — |
| ni30_15 / ai-assistant | N | PASS | 1.000 | 0.052 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_15 / ai-assistant | S | FAIL | 0.937 | 1.358 | 0 | mean lip error, px: 1.358 (required ≤0.5) | [play](review/ni30_15__ai-assistant/ni30_15__ai-assistant_AS.mp4) |
| ni30_15 / ai-assistant | S16 | FAIL | 0.937 | 1.354 | 0 | mean lip error, px: 1.354 (required ≤0.5) | — |
| ni30_15 / eng | N | PASS | 1.000 | 0.050 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_15 / eng | S | FAIL | 0.949 | 1.423 | 0 | mean lip error, px: 1.423 (required ≤0.5) | [play](review/ni30_15__eng/ni30_15__eng_AS.mp4) |
| ni30_15 / eng | S16 | FAIL | 0.949 | 1.422 | 0 | mean lip error, px: 1.422 (required ≤0.5) | — |
| ni30_16 / ai-assistant | N | PASS | 1.000 | 0.022 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_16 / ai-assistant | S | FAIL | 0.902 | 1.560 | 0 | mean lip error, px: 1.560 (required ≤0.5) | [play](review/ni30_16__ai-assistant/ni30_16__ai-assistant_AS.mp4) |
| ni30_16 / ai-assistant | S16 | FAIL | 0.903 | 1.560 | 0 | mean lip error, px: 1.560 (required ≤0.5) | — |
| ni30_16 / eng | N | PASS | 1.000 | 0.016 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_16 / eng | S | FAIL | 0.925 | 1.719 | 0 | mean lip error, px: 1.719 (required ≤0.5) | [play](review/ni30_16__eng/ni30_16__eng_AS.mp4) |
| ni30_16 / eng | S16 | FAIL | 0.925 | 1.719 | 0 | mean lip error, px: 1.719 (required ≤0.5) | — |
| ni30_17 / ai-assistant | N | PASS | 1.000 | 0.025 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_17 / ai-assistant | S | FAIL | 0.887 | 2.053 | 0 | mean lip error, px: 2.053 (required ≤0.5) | [play](review/ni30_17__ai-assistant/ni30_17__ai-assistant_AS.mp4) |
| ni30_17 / ai-assistant | S16 | FAIL | 0.887 | 2.056 | 0 | mean lip error, px: 2.056 (required ≤0.5) | — |
| ni30_17 / eng | N | PASS | 1.000 | 0.023 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_17 / eng | S | FAIL | 0.927 | 1.569 | 0 | mean lip error, px: 1.569 (required ≤0.5) | [play](review/ni30_17__eng/ni30_17__eng_AS.mp4) |
| ni30_17 / eng | S16 | FAIL | 0.927 | 1.569 | 0 | mean lip error, px: 1.569 (required ≤0.5) | — |
| ni30_18 / ai-assistant | N | PASS | 1.000 | 0.035 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_18 / ai-assistant | S | FAIL | 0.947 | 1.099 | 0 | mean lip error, px: 1.099 (required ≤0.5) | [play](review/ni30_18__ai-assistant/ni30_18__ai-assistant_AS.mp4) |
| ni30_18 / ai-assistant | S16 | FAIL | 0.946 | 1.103 | 0 | mean lip error, px: 1.103 (required ≤0.5) | — |
| ni30_18 / eng | N | PASS | 1.000 | 0.035 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_18 / eng | S | FAIL | 0.942 | 1.378 | 0 | mean lip error, px: 1.378 (required ≤0.5) | [play](review/ni30_18__eng/ni30_18__eng_AS.mp4) |
| ni30_18 / eng | S16 | FAIL | 0.943 | 1.368 | 0 | mean lip error, px: 1.368 (required ≤0.5) | — |
| ni30_19 / ai-assistant | N | PASS | 0.999 | 0.168 | 0 | lip correlation: 0.999 (required ≥0.97) | — |
| ni30_19 / ai-assistant | S | FAIL | 0.873 | 2.882 | -1 | mean lip error, px: 2.882 (required ≤0.5) | [play](review/ni30_19__ai-assistant/ni30_19__ai-assistant_AS.mp4) |
| ni30_19 / ai-assistant | S16 | FAIL | 0.874 | 2.868 | -1 | mean lip error, px: 2.868 (required ≤0.5) | — |
| ni30_19 / eng | N | PASS | 0.999 | 0.149 | 0 | lip correlation: 0.999 (required ≥0.97) | — |
| ni30_19 / eng | S | FAIL | 0.862 | 2.791 | -1 | mean lip error, px: 2.791 (required ≤0.5) | [play](review/ni30_19__eng/ni30_19__eng_AS.mp4) |
| ni30_19 / eng | S16 | FAIL | 0.864 | 2.781 | -1 | mean lip error, px: 2.781 (required ≤0.5) | — |
| ni30_20 / ai-assistant | N | PASS | 1.000 | 0.044 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_20 / ai-assistant | S | FAIL | 0.897 | 1.513 | 0 | mean lip error, px: 1.513 (required ≤0.5) | [play](review/ni30_20__ai-assistant/ni30_20__ai-assistant_AS.mp4) |
| ni30_20 / ai-assistant | S16 | FAIL | 0.897 | 1.511 | 0 | mean lip error, px: 1.511 (required ≤0.5) | — |
| ni30_20 / eng | N | PASS | 1.000 | 0.034 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_20 / eng | S | FAIL | 0.885 | 1.537 | -1 | mean lip error, px: 1.537 (required ≤0.5) | [play](review/ni30_20__eng/ni30_20__eng_AS.mp4) |
| ni30_20 / eng | S16 | FAIL | 0.886 | 1.531 | -1 | mean lip error, px: 1.531 (required ≤0.5) | — |
| ni30_21 / ai-assistant | N | PASS | 1.000 | 0.028 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_21 / ai-assistant | S | FAIL | 0.870 | 1.524 | 0 | mean lip error, px: 1.524 (required ≤0.5) | [play](review/ni30_21__ai-assistant/ni30_21__ai-assistant_AS.mp4) |
| ni30_21 / ai-assistant | S16 | FAIL | 0.869 | 1.526 | 0 | mean lip error, px: 1.526 (required ≤0.5) | — |
| ni30_21 / eng | N | PASS | 1.000 | 0.025 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_21 / eng | S | FAIL | 0.860 | 1.516 | -1 | mean lip error, px: 1.516 (required ≤0.5) | [play](review/ni30_21__eng/ni30_21__eng_AS.mp4) |
| ni30_21 / eng | S16 | FAIL | 0.860 | 1.517 | -1 | mean lip error, px: 1.517 (required ≤0.5) | — |
| ni30_22 / ai-assistant | N | PASS | 1.000 | 0.026 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_22 / ai-assistant | S | FAIL | 0.920 | 1.537 | 0 | mean lip error, px: 1.537 (required ≤0.5) | [play](review/ni30_22__ai-assistant/ni30_22__ai-assistant_AS.mp4) |
| ni30_22 / ai-assistant | S16 | FAIL | 0.920 | 1.537 | 0 | mean lip error, px: 1.537 (required ≤0.5) | — |
| ni30_22 / eng | N | PASS | 1.000 | 0.021 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_22 / eng | S | FAIL | 0.965 | 1.111 | 0 | mean lip error, px: 1.111 (required ≤0.5) | [play](review/ni30_22__eng/ni30_22__eng_AS.mp4) |
| ni30_22 / eng | S16 | FAIL | 0.965 | 1.107 | 0 | mean lip error, px: 1.107 (required ≤0.5) | — |
| ni30_23 / ai-assistant | N | PASS | 0.998 | 0.163 | 0 | lip correlation: 0.998 (required ≥0.97) | — |
| ni30_23 / ai-assistant | S | FAIL | 0.930 | 0.989 | 0 | mean lip error, px: 0.989 (required ≤0.5) | [play](review/ni30_23__ai-assistant/ni30_23__ai-assistant_AS.mp4) |
| ni30_23 / ai-assistant | S16 | FAIL | 0.930 | 0.990 | 0 | mean lip error, px: 0.990 (required ≤0.5) | — |
| ni30_23 / eng | N | PASS | 0.998 | 0.159 | 0 | lip correlation: 0.998 (required ≥0.97) | — |
| ni30_23 / eng | S | FAIL | 0.933 | 1.032 | 0 | mean lip error, px: 1.032 (required ≤0.5) | [play](review/ni30_23__eng/ni30_23__eng_AS.mp4) |
| ni30_23 / eng | S16 | FAIL | 0.931 | 1.043 | 0 | mean lip error, px: 1.043 (required ≤0.5) | — |
| ni30_24 / ai-assistant | N | PASS | 1.000 | 0.043 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_24 / ai-assistant | S | FAIL | 0.924 | 1.752 | 0 | mean lip error, px: 1.752 (required ≤0.5) | [play](review/ni30_24__ai-assistant/ni30_24__ai-assistant_AS.mp4) |
| ni30_24 / ai-assistant | S16 | FAIL | 0.924 | 1.756 | 0 | mean lip error, px: 1.756 (required ≤0.5) | — |
| ni30_24 / eng | N | PASS | 1.000 | 0.040 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_24 / eng | S | FAIL | 0.943 | 1.692 | 0 | mean lip error, px: 1.692 (required ≤0.5) | [play](review/ni30_24__eng/ni30_24__eng_AS.mp4) |
| ni30_24 / eng | S16 | FAIL | 0.943 | 1.686 | 0 | mean lip error, px: 1.686 (required ≤0.5) | — |
| ni30_25 / ai-assistant | N | PASS | 1.000 | 0.023 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_25 / ai-assistant | S | FAIL | 0.922 | 1.197 | 0 | mean lip error, px: 1.197 (required ≤0.5) | [play](review/ni30_25__ai-assistant/ni30_25__ai-assistant_AS.mp4) |
| ni30_25 / ai-assistant | S16 | FAIL | 0.922 | 1.197 | 0 | mean lip error, px: 1.197 (required ≤0.5) | — |
| ni30_25 / eng | N | PASS | 1.000 | 0.018 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_25 / eng | S | FAIL | 0.948 | 1.290 | 0 | mean lip error, px: 1.290 (required ≤0.5) | [play](review/ni30_25__eng/ni30_25__eng_AS.mp4) |
| ni30_25 / eng | S16 | FAIL | 0.948 | 1.287 | 0 | mean lip error, px: 1.287 (required ≤0.5) | — |
| ni30_26 / ai-assistant | N | PASS | 1.000 | 0.028 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_26 / ai-assistant | S | FAIL | 0.931 | 2.016 | 0 | mean lip error, px: 2.016 (required ≤0.5) | [play](review/ni30_26__ai-assistant/ni30_26__ai-assistant_AS.mp4) |
| ni30_26 / ai-assistant | S16 | FAIL | 0.930 | 2.017 | 0 | mean lip error, px: 2.017 (required ≤0.5) | — |
| ni30_26 / eng | N | PASS | 1.000 | 0.023 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_26 / eng | S | FAIL | 0.965 | 1.432 | 0 | mean lip error, px: 1.432 (required ≤0.5) | [play](review/ni30_26__eng/ni30_26__eng_AS.mp4) |
| ni30_26 / eng | S16 | FAIL | 0.965 | 1.432 | 0 | mean lip error, px: 1.432 (required ≤0.5) | — |
| ni30_27 / ai-assistant | N | PASS | 1.000 | 0.027 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_27 / ai-assistant | S | FAIL | 0.887 | 2.602 | 0 | mean lip error, px: 2.602 (required ≤0.5) | [play](review/ni30_27__ai-assistant/ni30_27__ai-assistant_AS.mp4) |
| ni30_27 / ai-assistant | S16 | FAIL | 0.887 | 2.600 | 0 | mean lip error, px: 2.600 (required ≤0.5) | — |
| ni30_27 / eng | N | PASS | 1.000 | 0.024 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_27 / eng | S | FAIL | 0.927 | 2.306 | 0 | mean lip error, px: 2.306 (required ≤0.5) | [play](review/ni30_27__eng/ni30_27__eng_AS.mp4) |
| ni30_27 / eng | S16 | FAIL | 0.927 | 2.307 | 0 | mean lip error, px: 2.307 (required ≤0.5) | — |
| ni30_28 / ai-assistant | N | PASS | 1.000 | 0.074 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_28 / ai-assistant | S | FAIL | 0.919 | 1.218 | 0 | mean lip error, px: 1.218 (required ≤0.5) | [play](review/ni30_28__ai-assistant/ni30_28__ai-assistant_AS.mp4) |
| ni30_28 / ai-assistant | S16 | FAIL | 0.918 | 1.222 | 0 | mean lip error, px: 1.222 (required ≤0.5) | — |
| ni30_28 / eng | N | PASS | 1.000 | 0.045 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_28 / eng | S | FAIL | 0.949 | 0.985 | 0 | mean lip error, px: 0.985 (required ≤0.5) | [play](review/ni30_28__eng/ni30_28__eng_AS.mp4) |
| ni30_28 / eng | S16 | FAIL | 0.950 | 0.972 | 0 | mean lip error, px: 0.972 (required ≤0.5) | — |
| ni30_29 / ai-assistant | N | PASS | 0.999 | 0.132 | 0 | lip correlation: 0.999 (required ≥0.97) | — |
| ni30_29 / ai-assistant | S | FAIL | 0.859 | 1.690 | 0 | mean lip error, px: 1.690 (required ≤0.5) | [play](review/ni30_29__ai-assistant/ni30_29__ai-assistant_AS.mp4) |
| ni30_29 / ai-assistant | S16 | FAIL | 0.858 | 1.698 | 0 | mean lip error, px: 1.698 (required ≤0.5) | — |
| ni30_29 / eng | N | PASS | 0.998 | 0.138 | 0 | lip correlation: 0.998 (required ≥0.97) | — |
| ni30_29 / eng | S | FAIL | 0.866 | 1.685 | 0 | mean lip error, px: 1.685 (required ≤0.5) | [play](review/ni30_29__eng/ni30_29__eng_AS.mp4) |
| ni30_29 / eng | S16 | FAIL | 0.866 | 1.688 | 0 | mean lip error, px: 1.688 (required ≤0.5) | — |
| ni30_30 / ai-assistant | N | PASS | 1.000 | 0.106 | 0 | lip correlation: 1.000 (required ≥0.97) | — |
| ni30_30 / ai-assistant | S | FAIL | 0.862 | 2.176 | -1 | mean lip error, px: 2.176 (required ≤0.5) | [play](review/ni30_30__ai-assistant/ni30_30__ai-assistant_AS.mp4) |
| ni30_30 / ai-assistant | S16 | FAIL | 0.863 | 2.162 | -1 | mean lip error, px: 2.162 (required ≤0.5) | — |
| ni30_30 / eng | N | PASS | 0.999 | 0.150 | 0 | lip correlation: 0.999 (required ≥0.97) | — |
| ni30_30 / eng | S | FAIL | 0.900 | 1.975 | -1 | mean lip error, px: 1.975 (required ≤0.5) | [play](review/ni30_30__eng/ni30_30__eng_AS.mp4) |
| ni30_30 / eng | S16 | FAIL | 0.899 | 1.994 | -1 | mean lip error, px: 1.994 (required ≤0.5) | — |

## Authored group descriptions

Labels come from the character brief. Counts and averages describe this small set only; they are not inferred ethnicity, a fairness proof or support for universal character quality.

### Skin tone group

| Authored group | Identities | N / S / S16 clip passes | N / S / S16 identities with both passing | S mean corr / mean px |
|---|---:|---|---|---|
| deep_brown | 3 | 6 / 0 / 0 of 6 | 3 / 0 / 0 of 3 | 0.894 / 3.039 |
| deep_tan | 3 | 6 / 0 / 0 of 6 | 3 / 0 / 0 of 3 | 0.932 / 1.357 |
| fair | 3 | 6 / 0 / 0 of 6 | 3 / 0 / 0 of 3 | 0.899 / 1.186 |
| light | 3 | 6 / 0 / 0 of 6 | 3 / 0 / 0 of 3 | 0.918 / 1.569 |
| light_medium | 3 | 6 / 0 / 0 of 6 | 3 / 0 / 0 of 3 | 0.933 / 1.218 |
| medium | 3 | 6 / 0 / 0 of 6 | 3 / 0 / 0 of 3 | 0.928 / 2.260 |
| medium_brown | 3 | 6 / 0 / 0 of 6 | 3 / 0 / 0 of 3 | 0.905 / 1.418 |
| medium_tan | 3 | 6 / 0 / 0 of 6 | 3 / 0 / 0 of 3 | 0.926 / 1.659 |
| very_deep_brown | 3 | 6 / 0 / 0 of 6 | 3 / 0 / 0 of 3 | 0.879 / 1.707 |
| very_fair | 3 | 6 / 0 / 0 of 6 | 3 / 0 / 0 of 3 | 0.876 / 1.731 |

### Age band

| Authored group | Identities | N / S / S16 clip passes | N / S / S16 identities with both passing | S mean corr / mean px |
|---|---:|---|---|---|
| 18–34 | 10 | 20 / 0 / 0 of 20 | 10 / 0 / 0 of 10 | 0.901 / 1.563 |
| 35–54 | 10 | 20 / 0 / 0 of 20 | 10 / 0 / 0 of 10 | 0.922 / 1.673 |
| 55+ | 10 | 20 / 0 / 0 of 20 | 10 / 0 / 0 of 10 | 0.904 / 1.908 |

### Gender presentation

| Authored group | Identities | N / S / S16 clip passes | N / S / S16 identities with both passing | S mean corr / mean px |
|---|---:|---|---|---|
| androgynous | 2 | 4 / 0 / 0 of 4 | 2 / 0 / 0 of 2 | 0.912 / 1.700 |
| feminine | 14 | 28 / 0 / 0 of 28 | 14 / 0 / 0 of 14 | 0.902 / 1.669 |
| masculine | 14 | 28 / 0 / 0 of 28 | 14 / 0 / 0 of 14 | 0.915 / 1.762 |

[Open the searchable local gallery](gallery.html). Private S16 links appear only when explicitly supplied from retained ignored local files.
