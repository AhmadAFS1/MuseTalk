# New static-avatar r5 review: FAIL

Thirty held-out generated portraits × two audio clips, 250 frames at 25 fps per clip. Source head and body are
frozen; only generated lips/face regions change. Manifest SHA256: `3f1fbdb0b7e8a6e6ae0433aad84a0b679d580441d5f7af47c5e69dc684578e8a`.
Student checkpoint SHA256: `a05d2d98f69b431252458d3b25f72e395d557ba693821baee91d11f5fe5a96cd`. All arms share the source cache, audio windows, blend and fp32 TAESD.
A is full fp32 with TF32 off; N is full fp16; S is student fp32; S16 is student fp16 on NVIDIA GeForce RTX 3090.

Acceptance uses the unchanged r5 thresholds against A: aperture correlation ≥ 0.97,
lag 0, mean aperture error ≤ 0.5 px, mouth/face
flicker ratios ≤ 1.05, mouth sharpness ratio ≥ 0.95.
The complete manifest set, finite metrics and all 250 finite FaceMesh frames are required independently for each arm.

| Arm | Independent verdict | r5 clips passing |
|---|---|---|
| N | PASS | 60/60 |
| S | FAIL | 0/60 |
| S16 | FAIL | 0/60 |

- Static generated portraits: source head/body remain frozen; this does not test moving-source poses.
- Results characterize these 30 identities and two clips; they cannot prove quality for all future characters.
- No shipped-chain D baseline was measured for these identities; D non-inferiority is not evaluated.
- S16 is PyTorch fp16 on the reported device, not Core ML quantization or iPhone ANE quality evidence.

## Reasons by arm

### N: PASS

- Every expected clip passes.

### S: FAIL

- ni30_01__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_01__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_01__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_01__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_01__eng: r5 aperture_corr_ge_0.97
- ni30_01__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_02__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_02__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_02__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_02__eng: r5 aperture_corr_ge_0.97
- ni30_02__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_03__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_03__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_03__ai-assistant: r5 aperture_lag_0
- ni30_03__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_03__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_03__eng: r5 aperture_corr_ge_0.97
- ni30_03__eng: r5 aperture_lag_0
- ni30_03__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_04__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_04__ai-assistant: r5 aperture_lag_0
- ni30_04__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_04__eng: r5 aperture_corr_ge_0.97
- ni30_04__eng: r5 aperture_lag_0
- ni30_04__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_05__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_05__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_05__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_05__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_05__eng: r5 aperture_corr_ge_0.97
- ni30_05__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_06__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_06__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_06__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_06__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_06__eng: r5 aperture_corr_ge_0.97
- ni30_06__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_07__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_07__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_07__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_07__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_07__eng: r5 aperture_corr_ge_0.97
- ni30_07__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_08__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_08__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_08__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_08__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_08__eng: r5 aperture_corr_ge_0.97
- ni30_08__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_09__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_09__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_09__ai-assistant: r5 aperture_lag_0
- ni30_09__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_09__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_09__eng: r5 aperture_corr_ge_0.97
- ni30_09__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_10__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_10__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_10__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_10__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_10__eng: r5 aperture_corr_ge_0.97
- ni30_10__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_11__ai-assistant: r5 flicker_mouth_ratio_le_1.05
- ni30_11__ai-assistant: r5 flicker_face_ratio_le_1.05
- ni30_11__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_11__ai-assistant: r5 aperture_lag_0
- ni30_11__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_11__eng: r5 flicker_mouth_ratio_le_1.05
- ni30_11__eng: r5 flicker_face_ratio_le_1.05
- ni30_11__eng: r5 aperture_corr_ge_0.97
- ni30_11__eng: r5 aperture_lag_0
- ni30_11__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_12__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_12__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_12__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_12__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_12__eng: r5 aperture_corr_ge_0.97
- ni30_12__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_13__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_13__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_13__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_13__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_13__eng: r5 aperture_corr_ge_0.97
- ni30_13__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_14__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_14__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_14__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_14__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_14__eng: r5 aperture_corr_ge_0.97
- ni30_14__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_15__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_15__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_15__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_15__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_15__eng: r5 aperture_corr_ge_0.97
- ni30_15__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_16__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_16__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_16__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_16__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_16__eng: r5 aperture_corr_ge_0.97
- ni30_16__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_17__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_17__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_17__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_17__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_17__eng: r5 aperture_corr_ge_0.97
- ni30_17__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_18__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_18__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_18__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_18__eng: r5 aperture_corr_ge_0.97
- ni30_18__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_19__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_19__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_19__ai-assistant: r5 aperture_lag_0
- ni30_19__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_19__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_19__eng: r5 aperture_corr_ge_0.97
- ni30_19__eng: r5 aperture_lag_0
- ni30_19__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_20__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_20__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_20__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_20__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_20__eng: r5 aperture_corr_ge_0.97
- ni30_20__eng: r5 aperture_lag_0
- ni30_20__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_21__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_21__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_21__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_21__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_21__eng: r5 aperture_corr_ge_0.97
- ni30_21__eng: r5 aperture_lag_0
- ni30_21__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_22__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_22__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_22__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_22__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_22__eng: r5 aperture_corr_ge_0.97
- ni30_22__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_23__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_23__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_23__eng: r5 aperture_corr_ge_0.97
- ni30_23__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_24__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_24__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_24__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_24__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_24__eng: r5 aperture_corr_ge_0.97
- ni30_24__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_25__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_25__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_25__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_25__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_25__eng: r5 aperture_corr_ge_0.97
- ni30_25__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_26__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_26__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_26__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_26__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_26__eng: r5 aperture_corr_ge_0.97
- ni30_26__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_27__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_27__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_27__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_27__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_27__eng: r5 aperture_corr_ge_0.97
- ni30_27__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_28__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_28__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_28__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_28__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_28__eng: r5 aperture_corr_ge_0.97
- ni30_28__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_29__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_29__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_29__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_29__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_29__eng: r5 aperture_corr_ge_0.97
- ni30_29__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_30__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_30__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_30__ai-assistant: r5 aperture_lag_0
- ni30_30__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_30__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_30__eng: r5 aperture_corr_ge_0.97
- ni30_30__eng: r5 aperture_lag_0
- ni30_30__eng: r5 aperture_mean_abs_delta_le_0.5px

### S16: FAIL

- ni30_01__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_01__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_01__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_01__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_01__eng: r5 aperture_corr_ge_0.97
- ni30_01__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_02__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_02__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_02__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_02__eng: r5 aperture_corr_ge_0.97
- ni30_02__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_03__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_03__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_03__ai-assistant: r5 aperture_lag_0
- ni30_03__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_03__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_03__eng: r5 aperture_corr_ge_0.97
- ni30_03__eng: r5 aperture_lag_0
- ni30_03__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_04__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_04__ai-assistant: r5 aperture_lag_0
- ni30_04__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_04__eng: r5 aperture_corr_ge_0.97
- ni30_04__eng: r5 aperture_lag_0
- ni30_04__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_05__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_05__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_05__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_05__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_05__eng: r5 aperture_corr_ge_0.97
- ni30_05__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_06__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_06__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_06__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_06__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_06__eng: r5 aperture_corr_ge_0.97
- ni30_06__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_07__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_07__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_07__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_07__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_07__eng: r5 aperture_corr_ge_0.97
- ni30_07__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_08__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_08__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_08__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_08__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_08__eng: r5 aperture_corr_ge_0.97
- ni30_08__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_09__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_09__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_09__ai-assistant: r5 aperture_lag_0
- ni30_09__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_09__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_09__eng: r5 aperture_corr_ge_0.97
- ni30_09__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_10__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_10__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_10__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_10__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_10__eng: r5 aperture_corr_ge_0.97
- ni30_10__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_11__ai-assistant: r5 flicker_mouth_ratio_le_1.05
- ni30_11__ai-assistant: r5 flicker_face_ratio_le_1.05
- ni30_11__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_11__ai-assistant: r5 aperture_lag_0
- ni30_11__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_11__eng: r5 flicker_mouth_ratio_le_1.05
- ni30_11__eng: r5 flicker_face_ratio_le_1.05
- ni30_11__eng: r5 aperture_corr_ge_0.97
- ni30_11__eng: r5 aperture_lag_0
- ni30_11__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_12__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_12__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_12__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_12__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_12__eng: r5 aperture_corr_ge_0.97
- ni30_12__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_13__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_13__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_13__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_13__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_13__eng: r5 aperture_corr_ge_0.97
- ni30_13__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_14__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_14__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_14__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_14__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_14__eng: r5 aperture_corr_ge_0.97
- ni30_14__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_15__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_15__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_15__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_15__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_15__eng: r5 aperture_corr_ge_0.97
- ni30_15__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_16__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_16__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_16__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_16__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_16__eng: r5 aperture_corr_ge_0.97
- ni30_16__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_17__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_17__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_17__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_17__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_17__eng: r5 aperture_corr_ge_0.97
- ni30_17__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_18__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_18__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_18__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_18__eng: r5 aperture_corr_ge_0.97
- ni30_18__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_19__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_19__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_19__ai-assistant: r5 aperture_lag_0
- ni30_19__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_19__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_19__eng: r5 aperture_corr_ge_0.97
- ni30_19__eng: r5 aperture_lag_0
- ni30_19__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_20__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_20__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_20__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_20__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_20__eng: r5 aperture_corr_ge_0.97
- ni30_20__eng: r5 aperture_lag_0
- ni30_20__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_21__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_21__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_21__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_21__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_21__eng: r5 aperture_corr_ge_0.97
- ni30_21__eng: r5 aperture_lag_0
- ni30_21__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_22__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_22__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_22__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_22__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_22__eng: r5 aperture_corr_ge_0.97
- ni30_22__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_23__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_23__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_23__eng: r5 aperture_corr_ge_0.97
- ni30_23__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_24__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_24__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_24__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_24__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_24__eng: r5 aperture_corr_ge_0.97
- ni30_24__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_25__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_25__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_25__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_25__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_25__eng: r5 aperture_corr_ge_0.97
- ni30_25__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_26__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_26__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_26__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_26__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_26__eng: r5 aperture_corr_ge_0.97
- ni30_26__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_27__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_27__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_27__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_27__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_27__eng: r5 aperture_corr_ge_0.97
- ni30_27__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_28__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_28__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_28__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_28__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_28__eng: r5 aperture_corr_ge_0.97
- ni30_28__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_29__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_29__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_29__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_29__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_29__eng: r5 aperture_corr_ge_0.97
- ni30_29__eng: r5 aperture_mean_abs_delta_le_0.5px
- ni30_30__ai-assistant: r5 mouth_sharpness_ratio_ge_0.95
- ni30_30__ai-assistant: r5 aperture_corr_ge_0.97
- ni30_30__ai-assistant: r5 aperture_lag_0
- ni30_30__ai-assistant: r5 aperture_mean_abs_delta_le_0.5px
- ni30_30__eng: r5 mouth_sharpness_ratio_ge_0.95
- ni30_30__eng: r5 aperture_corr_ge_0.97
- ni30_30__eng: r5 aperture_lag_0
- ni30_30__eng: r5 aperture_mean_abs_delta_le_0.5px


## Per clip (D is not evaluated)

| clip | arm | latent mae | lip corr | lip mean abs delta px | lag | mouth PSNR dB | mouth flicker ratio | error flicker mouth | mouth sharpness | r5 | vs D |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ni30_01__ai-assistant | N | 0.0004 | 1.0000 | 0.039 | 0 | 64.98 | 1.000 | 0.040 | 1.000 | pass | not_evaluated |
| ni30_01__ai-assistant | S | 0.1495 | 0.9080 | 1.887 | 0 | 22.73 | 0.868 | 4.339 | 0.754 | FAIL | not_evaluated |
| ni30_01__ai-assistant | S16 | 0.1495 | 0.9075 | 1.889 | 0 | 22.73 | 0.868 | 4.338 | 0.754 | FAIL | not_evaluated |
| ni30_01__eng | N | 0.0004 | 1.0000 | 0.030 | 0 | 65.06 | 1.000 | 0.040 | 1.000 | pass | not_evaluated |
| ni30_01__eng | S | 0.1417 | 0.9210 | 1.998 | 0 | 23.54 | 0.832 | 3.473 | 0.800 | FAIL | not_evaluated |
| ni30_01__eng | S16 | 0.1416 | 0.9213 | 1.997 | 0 | 23.54 | 0.832 | 3.473 | 0.800 | FAIL | not_evaluated |
| ni30_02__ai-assistant | N | 0.0004 | 0.9999 | 0.032 | 0 | 65.72 | 1.000 | 0.034 | 1.000 | pass | not_evaluated |
| ni30_02__ai-assistant | S | 0.0891 | 0.8615 | 1.604 | 0 | 30.92 | 0.857 | 2.850 | 0.976 | FAIL | not_evaluated |
| ni30_02__ai-assistant | S16 | 0.0891 | 0.8623 | 1.602 | 0 | 30.92 | 0.857 | 2.850 | 0.975 | FAIL | not_evaluated |
| ni30_02__eng | N | 0.0004 | 1.0000 | 0.023 | 0 | 65.66 | 1.000 | 0.035 | 1.000 | pass | not_evaluated |
| ni30_02__eng | S | 0.0854 | 0.9130 | 1.239 | 0 | 31.41 | 0.818 | 2.344 | 0.905 | FAIL | not_evaluated |
| ni30_02__eng | S16 | 0.0854 | 0.9132 | 1.234 | 0 | 31.41 | 0.818 | 2.343 | 0.905 | FAIL | not_evaluated |
| ni30_03__ai-assistant | N | 0.0004 | 0.9999 | 0.030 | 0 | 65.33 | 1.000 | 0.037 | 1.000 | pass | not_evaluated |
| ni30_03__ai-assistant | S | 0.1184 | 0.8022 | 1.384 | -1 | 25.96 | 1.014 | 3.625 | 0.781 | FAIL | not_evaluated |
| ni30_03__ai-assistant | S16 | 0.1184 | 0.8034 | 1.380 | -1 | 25.96 | 1.014 | 3.625 | 0.781 | FAIL | not_evaluated |
| ni30_03__eng | N | 0.0004 | 1.0000 | 0.023 | 0 | 65.35 | 1.000 | 0.037 | 1.000 | pass | not_evaluated |
| ni30_03__eng | S | 0.1112 | 0.8671 | 1.279 | -1 | 26.96 | 0.936 | 3.039 | 0.789 | FAIL | not_evaluated |
| ni30_03__eng | S16 | 0.1112 | 0.8670 | 1.280 | -1 | 26.95 | 0.936 | 3.039 | 0.788 | FAIL | not_evaluated |
| ni30_04__ai-assistant | N | 0.0004 | 1.0000 | 0.028 | 0 | 65.32 | 1.000 | 0.037 | 1.000 | pass | not_evaluated |
| ni30_04__ai-assistant | S | 0.1198 | 0.8407 | 2.130 | 1 | 28.55 | 0.845 | 3.465 | 1.109 | FAIL | not_evaluated |
| ni30_04__ai-assistant | S16 | 0.1198 | 0.8413 | 2.128 | 1 | 28.55 | 0.845 | 3.465 | 1.109 | FAIL | not_evaluated |
| ni30_04__eng | N | 0.0004 | 1.0000 | 0.022 | 0 | 65.37 | 1.000 | 0.037 | 1.000 | pass | not_evaluated |
| ni30_04__eng | S | 0.1210 | 0.8953 | 1.927 | 1 | 28.62 | 0.797 | 2.862 | 1.024 | FAIL | not_evaluated |
| ni30_04__eng | S16 | 0.1209 | 0.8951 | 1.930 | 1 | 28.62 | 0.797 | 2.862 | 1.023 | FAIL | not_evaluated |
| ni30_05__ai-assistant | N | 0.0004 | 1.0000 | 0.026 | 0 | 65.45 | 1.000 | 0.036 | 1.000 | pass | not_evaluated |
| ni30_05__ai-assistant | S | 0.1072 | 0.8888 | 1.596 | 0 | 25.32 | 0.895 | 3.301 | 0.629 | FAIL | not_evaluated |
| ni30_05__ai-assistant | S16 | 0.1072 | 0.8889 | 1.595 | 0 | 25.32 | 0.895 | 3.302 | 0.629 | FAIL | not_evaluated |
| ni30_05__eng | N | 0.0004 | 1.0000 | 0.026 | 0 | 65.63 | 1.000 | 0.035 | 1.000 | pass | not_evaluated |
| ni30_05__eng | S | 0.0981 | 0.9388 | 1.395 | 0 | 26.71 | 0.852 | 2.767 | 0.644 | FAIL | not_evaluated |
| ni30_05__eng | S16 | 0.0981 | 0.9389 | 1.395 | 0 | 26.70 | 0.852 | 2.767 | 0.644 | FAIL | not_evaluated |
| ni30_06__ai-assistant | N | 0.0004 | 1.0000 | 0.039 | 0 | 64.72 | 1.000 | 0.043 | 1.001 | pass | not_evaluated |
| ni30_06__ai-assistant | S | 0.1269 | 0.9128 | 3.499 | 0 | 24.26 | 0.920 | 3.132 | 0.454 | FAIL | not_evaluated |
| ni30_06__ai-assistant | S16 | 0.1269 | 0.9131 | 3.499 | 0 | 24.27 | 0.920 | 3.132 | 0.454 | FAIL | not_evaluated |
| ni30_06__eng | N | 0.0004 | 0.9999 | 0.041 | 0 | 64.52 | 1.000 | 0.045 | 1.001 | pass | not_evaluated |
| ni30_06__eng | S | 0.1221 | 0.8718 | 3.831 | 0 | 24.34 | 0.861 | 2.694 | 0.482 | FAIL | not_evaluated |
| ni30_06__eng | S16 | 0.1221 | 0.8721 | 3.834 | 0 | 24.34 | 0.861 | 2.694 | 0.482 | FAIL | not_evaluated |
| ni30_07__ai-assistant | N | 0.0004 | 0.9998 | 0.046 | 0 | 64.99 | 1.000 | 0.040 | 1.000 | pass | not_evaluated |
| ni30_07__ai-assistant | S | 0.1054 | 0.9485 | 0.870 | 0 | 31.13 | 0.879 | 2.419 | 0.767 | FAIL | not_evaluated |
| ni30_07__ai-assistant | S16 | 0.1054 | 0.9485 | 0.873 | 0 | 31.13 | 0.879 | 2.419 | 0.767 | FAIL | not_evaluated |
| ni30_07__eng | N | 0.0004 | 0.9999 | 0.033 | 0 | 65.06 | 1.000 | 0.040 | 1.000 | pass | not_evaluated |
| ni30_07__eng | S | 0.1036 | 0.9628 | 0.895 | 0 | 31.13 | 0.823 | 2.093 | 0.729 | FAIL | not_evaluated |
| ni30_07__eng | S16 | 0.1036 | 0.9629 | 0.897 | 0 | 31.13 | 0.823 | 2.093 | 0.729 | FAIL | not_evaluated |
| ni30_08__ai-assistant | N | 0.0005 | 0.9995 | 0.096 | 0 | 62.20 | 1.000 | 0.076 | 1.000 | pass | not_evaluated |
| ni30_08__ai-assistant | S | 0.1190 | 0.9507 | 1.127 | 0 | 27.95 | 0.831 | 3.504 | 0.603 | FAIL | not_evaluated |
| ni30_08__ai-assistant | S16 | 0.1189 | 0.9508 | 1.140 | 0 | 27.95 | 0.831 | 3.504 | 0.603 | FAIL | not_evaluated |
| ni30_08__eng | N | 0.0005 | 0.9995 | 0.106 | 0 | 61.94 | 1.000 | 0.080 | 1.000 | pass | not_evaluated |
| ni30_08__eng | S | 0.1161 | 0.9619 | 1.191 | 0 | 27.90 | 0.775 | 3.044 | 0.571 | FAIL | not_evaluated |
| ni30_08__eng | S16 | 0.1161 | 0.9615 | 1.188 | 0 | 27.91 | 0.775 | 3.043 | 0.571 | FAIL | not_evaluated |
| ni30_09__ai-assistant | N | 0.0004 | 1.0000 | 0.025 | 0 | 65.67 | 1.000 | 0.035 | 1.000 | pass | not_evaluated |
| ni30_09__ai-assistant | S | 0.1299 | 0.8860 | 1.474 | -1 | 25.75 | 0.895 | 3.308 | 0.682 | FAIL | not_evaluated |
| ni30_09__ai-assistant | S16 | 0.1299 | 0.8860 | 1.472 | -1 | 25.75 | 0.895 | 3.308 | 0.682 | FAIL | not_evaluated |
| ni30_09__eng | N | 0.0004 | 1.0000 | 0.021 | 0 | 65.53 | 1.000 | 0.036 | 1.000 | pass | not_evaluated |
| ni30_09__eng | S | 0.1204 | 0.9303 | 1.185 | 0 | 27.09 | 0.851 | 2.745 | 0.652 | FAIL | not_evaluated |
| ni30_09__eng | S16 | 0.1204 | 0.9305 | 1.184 | 0 | 27.09 | 0.851 | 2.745 | 0.651 | FAIL | not_evaluated |
| ni30_10__ai-assistant | N | 0.0005 | 0.9993 | 0.129 | 0 | 63.18 | 1.000 | 0.060 | 1.000 | pass | not_evaluated |
| ni30_10__ai-assistant | S | 0.1509 | 0.9050 | 4.091 | 0 | 25.29 | 0.699 | 3.495 | 0.609 | FAIL | not_evaluated |
| ni30_10__ai-assistant | S16 | 0.1509 | 0.9057 | 4.086 | 0 | 25.29 | 0.699 | 3.495 | 0.609 | FAIL | not_evaluated |
| ni30_10__eng | N | 0.0005 | 0.9995 | 0.126 | 0 | 62.98 | 1.000 | 0.063 | 1.000 | pass | not_evaluated |
| ni30_10__eng | S | 0.1484 | 0.8941 | 4.585 | 0 | 25.59 | 0.665 | 2.982 | 0.627 | FAIL | not_evaluated |
| ni30_10__eng | S16 | 0.1484 | 0.8948 | 4.587 | 0 | 25.59 | 0.665 | 2.982 | 0.626 | FAIL | not_evaluated |
| ni30_11__ai-assistant | N | 0.0004 | 0.9999 | 0.041 | 0 | 65.36 | 1.000 | 0.037 | 1.000 | pass | not_evaluated |
| ni30_11__ai-assistant | S | 0.1291 | 0.7831 | 2.553 | -1 | 24.30 | 1.236 | 3.104 | 0.981 | FAIL | not_evaluated |
| ni30_11__ai-assistant | S16 | 0.1291 | 0.7824 | 2.564 | -1 | 24.30 | 1.236 | 3.104 | 0.981 | FAIL | not_evaluated |
| ni30_11__eng | N | 0.0004 | 0.9999 | 0.042 | 0 | 65.16 | 1.000 | 0.039 | 1.000 | pass | not_evaluated |
| ni30_11__eng | S | 0.1203 | 0.8129 | 2.346 | -1 | 25.11 | 1.113 | 2.748 | 1.010 | FAIL | not_evaluated |
| ni30_11__eng | S16 | 0.1203 | 0.8131 | 2.345 | -1 | 25.11 | 1.114 | 2.748 | 1.010 | FAIL | not_evaluated |
| ni30_12__ai-assistant | N | 0.0004 | 0.9998 | 0.053 | 0 | 65.22 | 1.000 | 0.038 | 1.000 | pass | not_evaluated |
| ni30_12__ai-assistant | S | 0.1079 | 0.9199 | 1.230 | 0 | 28.24 | 0.861 | 2.881 | 0.647 | FAIL | not_evaluated |
| ni30_12__ai-assistant | S16 | 0.1079 | 0.9204 | 1.223 | 0 | 28.24 | 0.861 | 2.882 | 0.647 | FAIL | not_evaluated |
| ni30_12__eng | N | 0.0004 | 0.9998 | 0.047 | 0 | 65.23 | 1.000 | 0.038 | 1.000 | pass | not_evaluated |
| ni30_12__eng | S | 0.1045 | 0.9452 | 1.202 | 0 | 28.83 | 0.805 | 2.396 | 0.655 | FAIL | not_evaluated |
| ni30_12__eng | S16 | 0.1045 | 0.9457 | 1.197 | 0 | 28.83 | 0.805 | 2.396 | 0.655 | FAIL | not_evaluated |
| ni30_13__ai-assistant | N | 0.0004 | 0.9999 | 0.030 | 0 | 65.30 | 1.000 | 0.037 | 1.000 | pass | not_evaluated |
| ni30_13__ai-assistant | S | 0.1001 | 0.9512 | 0.872 | 0 | 28.29 | 0.876 | 2.916 | 0.842 | FAIL | not_evaluated |
| ni30_13__ai-assistant | S16 | 0.1001 | 0.9514 | 0.872 | 0 | 28.30 | 0.876 | 2.916 | 0.842 | FAIL | not_evaluated |
| ni30_13__eng | N | 0.0004 | 1.0000 | 0.027 | 0 | 65.36 | 1.000 | 0.037 | 1.000 | pass | not_evaluated |
| ni30_13__eng | S | 0.0977 | 0.9518 | 1.037 | 0 | 28.79 | 0.814 | 2.475 | 0.778 | FAIL | not_evaluated |
| ni30_13__eng | S16 | 0.0977 | 0.9521 | 1.036 | 0 | 28.79 | 0.814 | 2.475 | 0.778 | FAIL | not_evaluated |
| ni30_14__ai-assistant | N | 0.0005 | 0.9998 | 0.048 | 0 | 63.65 | 1.000 | 0.055 | 1.000 | pass | not_evaluated |
| ni30_14__ai-assistant | S | 0.1353 | 0.9388 | 1.048 | 0 | 23.50 | 0.884 | 3.710 | 0.785 | FAIL | not_evaluated |
| ni30_14__ai-assistant | S16 | 0.1353 | 0.9389 | 1.048 | 0 | 23.50 | 0.884 | 3.710 | 0.785 | FAIL | not_evaluated |
| ni30_14__eng | N | 0.0004 | 0.9999 | 0.043 | 0 | 63.38 | 1.000 | 0.058 | 1.000 | pass | not_evaluated |
| ni30_14__eng | S | 0.1208 | 0.9648 | 0.785 | 0 | 24.81 | 0.850 | 3.104 | 0.781 | FAIL | not_evaluated |
| ni30_14__eng | S16 | 0.1208 | 0.9651 | 0.787 | 0 | 24.81 | 0.850 | 3.104 | 0.781 | FAIL | not_evaluated |
| ni30_15__ai-assistant | N | 0.0004 | 0.9998 | 0.052 | 0 | 64.43 | 1.000 | 0.045 | 1.000 | pass | not_evaluated |
| ni30_15__ai-assistant | S | 0.1039 | 0.9372 | 1.358 | 0 | 28.53 | 0.789 | 2.692 | 0.811 | FAIL | not_evaluated |
| ni30_15__ai-assistant | S16 | 0.1039 | 0.9374 | 1.354 | 0 | 28.53 | 0.788 | 2.692 | 0.811 | FAIL | not_evaluated |
| ni30_15__eng | N | 0.0004 | 0.9999 | 0.050 | 0 | 64.54 | 1.000 | 0.044 | 1.000 | pass | not_evaluated |
| ni30_15__eng | S | 0.0989 | 0.9488 | 1.423 | 0 | 28.93 | 0.765 | 2.208 | 0.827 | FAIL | not_evaluated |
| ni30_15__eng | S16 | 0.0989 | 0.9487 | 1.422 | 0 | 28.93 | 0.765 | 2.208 | 0.827 | FAIL | not_evaluated |
| ni30_16__ai-assistant | N | 0.0004 | 1.0000 | 0.022 | 0 | 65.03 | 1.000 | 0.040 | 1.001 | pass | not_evaluated |
| ni30_16__ai-assistant | S | 0.1176 | 0.9024 | 1.560 | 0 | 24.82 | 0.930 | 3.449 | 0.712 | FAIL | not_evaluated |
| ni30_16__ai-assistant | S16 | 0.1176 | 0.9026 | 1.560 | 0 | 24.82 | 0.930 | 3.449 | 0.712 | FAIL | not_evaluated |
| ni30_16__eng | N | 0.0004 | 1.0000 | 0.016 | 0 | 65.22 | 1.000 | 0.038 | 1.001 | pass | not_evaluated |
| ni30_16__eng | S | 0.1086 | 0.9254 | 1.719 | 0 | 25.78 | 0.881 | 2.858 | 0.709 | FAIL | not_evaluated |
| ni30_16__eng | S16 | 0.1086 | 0.9254 | 1.719 | 0 | 25.78 | 0.881 | 2.858 | 0.709 | FAIL | not_evaluated |
| ni30_17__ai-assistant | N | 0.0004 | 1.0000 | 0.025 | 0 | 65.27 | 1.000 | 0.038 | 1.000 | pass | not_evaluated |
| ni30_17__ai-assistant | S | 0.1042 | 0.8873 | 2.053 | 0 | 28.75 | 0.730 | 2.922 | 0.594 | FAIL | not_evaluated |
| ni30_17__ai-assistant | S16 | 0.1042 | 0.8866 | 2.056 | 0 | 28.75 | 0.730 | 2.922 | 0.594 | FAIL | not_evaluated |
| ni30_17__eng | N | 0.0004 | 1.0000 | 0.023 | 0 | 65.26 | 1.000 | 0.038 | 1.000 | pass | not_evaluated |
| ni30_17__eng | S | 0.0982 | 0.9266 | 1.569 | 0 | 29.28 | 0.692 | 2.516 | 0.569 | FAIL | not_evaluated |
| ni30_17__eng | S16 | 0.0982 | 0.9265 | 1.569 | 0 | 29.28 | 0.692 | 2.517 | 0.569 | FAIL | not_evaluated |
| ni30_18__ai-assistant | N | 0.0004 | 0.9999 | 0.035 | 0 | 65.08 | 1.000 | 0.040 | 1.000 | pass | not_evaluated |
| ni30_18__ai-assistant | S | 0.1135 | 0.9468 | 1.099 | 0 | 27.21 | 0.898 | 3.049 | 0.962 | FAIL | not_evaluated |
| ni30_18__ai-assistant | S16 | 0.1135 | 0.9465 | 1.103 | 0 | 27.21 | 0.898 | 3.050 | 0.963 | FAIL | not_evaluated |
| ni30_18__eng | N | 0.0004 | 0.9999 | 0.035 | 0 | 65.01 | 1.000 | 0.040 | 1.000 | pass | not_evaluated |
| ni30_18__eng | S | 0.1090 | 0.9419 | 1.378 | 0 | 27.86 | 0.830 | 2.710 | 0.929 | FAIL | not_evaluated |
| ni30_18__eng | S16 | 0.1089 | 0.9427 | 1.368 | 0 | 27.85 | 0.830 | 2.710 | 0.930 | FAIL | not_evaluated |
| ni30_19__ai-assistant | N | 0.0004 | 0.9991 | 0.168 | 0 | 64.99 | 1.000 | 0.040 | 1.000 | pass | not_evaluated |
| ni30_19__ai-assistant | S | 0.1377 | 0.8729 | 2.882 | -1 | 25.00 | 0.856 | 3.507 | 0.816 | FAIL | not_evaluated |
| ni30_19__ai-assistant | S16 | 0.1376 | 0.8741 | 2.868 | -1 | 25.00 | 0.856 | 3.507 | 0.816 | FAIL | not_evaluated |
| ni30_19__eng | N | 0.0004 | 0.9991 | 0.149 | 0 | 64.85 | 1.000 | 0.042 | 1.000 | pass | not_evaluated |
| ni30_19__eng | S | 0.1287 | 0.8620 | 2.791 | -1 | 25.47 | 0.830 | 2.969 | 0.844 | FAIL | not_evaluated |
| ni30_19__eng | S16 | 0.1287 | 0.8640 | 2.781 | -1 | 25.47 | 0.829 | 2.969 | 0.844 | FAIL | not_evaluated |
| ni30_20__ai-assistant | N | 0.0004 | 0.9999 | 0.044 | 0 | 66.05 | 1.000 | 0.032 | 1.000 | pass | not_evaluated |
| ni30_20__ai-assistant | S | 0.1324 | 0.8965 | 1.513 | 0 | 23.27 | 1.018 | 3.505 | 0.621 | FAIL | not_evaluated |
| ni30_20__ai-assistant | S16 | 0.1324 | 0.8965 | 1.511 | 0 | 23.27 | 1.017 | 3.505 | 0.621 | FAIL | not_evaluated |
| ni30_20__eng | N | 0.0004 | 0.9999 | 0.034 | 0 | 66.00 | 1.000 | 0.032 | 1.000 | pass | not_evaluated |
| ni30_20__eng | S | 0.1231 | 0.8852 | 1.537 | -1 | 24.62 | 0.991 | 2.807 | 0.651 | FAIL | not_evaluated |
| ni30_20__eng | S16 | 0.1231 | 0.8859 | 1.531 | -1 | 24.62 | 0.991 | 2.807 | 0.651 | FAIL | not_evaluated |
| ni30_21__ai-assistant | N | 0.0004 | 1.0000 | 0.028 | 0 | 65.65 | 1.000 | 0.035 | 1.000 | pass | not_evaluated |
| ni30_21__ai-assistant | S | 0.1401 | 0.8700 | 1.524 | 0 | 22.91 | 0.843 | 3.829 | 0.898 | FAIL | not_evaluated |
| ni30_21__ai-assistant | S16 | 0.1401 | 0.8695 | 1.526 | 0 | 22.91 | 0.842 | 3.829 | 0.898 | FAIL | not_evaluated |
| ni30_21__eng | N | 0.0004 | 0.9999 | 0.025 | 0 | 65.48 | 1.000 | 0.036 | 1.000 | pass | not_evaluated |
| ni30_21__eng | S | 0.1355 | 0.8598 | 1.516 | -1 | 23.68 | 0.794 | 3.253 | 0.851 | FAIL | not_evaluated |
| ni30_21__eng | S16 | 0.1355 | 0.8596 | 1.517 | -1 | 23.68 | 0.794 | 3.253 | 0.851 | FAIL | not_evaluated |
| ni30_22__ai-assistant | N | 0.0004 | 1.0000 | 0.026 | 0 | 65.84 | 1.000 | 0.033 | 1.000 | pass | not_evaluated |
| ni30_22__ai-assistant | S | 0.1242 | 0.9203 | 1.537 | 0 | 25.61 | 0.924 | 2.851 | 0.750 | FAIL | not_evaluated |
| ni30_22__ai-assistant | S16 | 0.1242 | 0.9203 | 1.537 | 0 | 25.61 | 0.923 | 2.850 | 0.750 | FAIL | not_evaluated |
| ni30_22__eng | N | 0.0004 | 1.0000 | 0.021 | 0 | 65.89 | 1.000 | 0.033 | 1.000 | pass | not_evaluated |
| ni30_22__eng | S | 0.1143 | 0.9650 | 1.111 | 0 | 26.90 | 0.870 | 2.381 | 0.747 | FAIL | not_evaluated |
| ni30_22__eng | S16 | 0.1143 | 0.9651 | 1.107 | 0 | 26.90 | 0.870 | 2.381 | 0.746 | FAIL | not_evaluated |
| ni30_23__ai-assistant | N | 0.0004 | 0.9976 | 0.163 | 0 | 64.80 | 1.000 | 0.042 | 1.000 | pass | not_evaluated |
| ni30_23__ai-assistant | S | 0.1047 | 0.9297 | 0.989 | 0 | 26.88 | 0.978 | 2.687 | 1.075 | FAIL | not_evaluated |
| ni30_23__ai-assistant | S16 | 0.1047 | 0.9303 | 0.990 | 0 | 26.88 | 0.978 | 2.687 | 1.075 | FAIL | not_evaluated |
| ni30_23__eng | N | 0.0004 | 0.9980 | 0.159 | 0 | 64.76 | 1.000 | 0.042 | 1.000 | pass | not_evaluated |
| ni30_23__eng | S | 0.0987 | 0.9327 | 1.032 | 0 | 27.84 | 0.886 | 2.316 | 1.014 | FAIL | not_evaluated |
| ni30_23__eng | S16 | 0.0987 | 0.9312 | 1.043 | 0 | 27.84 | 0.886 | 2.316 | 1.013 | FAIL | not_evaluated |
| ni30_24__ai-assistant | N | 0.0004 | 0.9999 | 0.043 | 0 | 64.94 | 1.000 | 0.041 | 1.000 | pass | not_evaluated |
| ni30_24__ai-assistant | S | 0.1164 | 0.9242 | 1.752 | 0 | 27.93 | 0.836 | 2.793 | 0.826 | FAIL | not_evaluated |
| ni30_24__ai-assistant | S16 | 0.1164 | 0.9242 | 1.756 | 0 | 27.93 | 0.836 | 2.794 | 0.826 | FAIL | not_evaluated |
| ni30_24__eng | N | 0.0004 | 0.9999 | 0.040 | 0 | 64.90 | 1.000 | 0.041 | 1.000 | pass | not_evaluated |
| ni30_24__eng | S | 0.1121 | 0.9431 | 1.692 | 0 | 28.68 | 0.795 | 2.286 | 0.769 | FAIL | not_evaluated |
| ni30_24__eng | S16 | 0.1121 | 0.9434 | 1.686 | 0 | 28.68 | 0.795 | 2.286 | 0.769 | FAIL | not_evaluated |
| ni30_25__ai-assistant | N | 0.0004 | 1.0000 | 0.023 | 0 | 65.81 | 1.000 | 0.034 | 1.000 | pass | not_evaluated |
| ni30_25__ai-assistant | S | 0.0926 | 0.9217 | 1.197 | 0 | 29.41 | 0.866 | 2.622 | 0.689 | FAIL | not_evaluated |
| ni30_25__ai-assistant | S16 | 0.0926 | 0.9217 | 1.197 | 0 | 29.41 | 0.866 | 2.622 | 0.689 | FAIL | not_evaluated |
| ni30_25__eng | N | 0.0004 | 1.0000 | 0.018 | 0 | 65.87 | 1.000 | 0.033 | 1.000 | pass | not_evaluated |
| ni30_25__eng | S | 0.0877 | 0.9476 | 1.290 | 0 | 30.05 | 0.827 | 2.230 | 0.706 | FAIL | not_evaluated |
| ni30_25__eng | S16 | 0.0877 | 0.9479 | 1.287 | 0 | 30.05 | 0.827 | 2.230 | 0.706 | FAIL | not_evaluated |
| ni30_26__ai-assistant | N | 0.0004 | 1.0000 | 0.028 | 0 | 65.89 | 1.000 | 0.033 | 1.000 | pass | not_evaluated |
| ni30_26__ai-assistant | S | 0.1285 | 0.9307 | 2.016 | 0 | 24.88 | 0.882 | 3.406 | 0.785 | FAIL | not_evaluated |
| ni30_26__ai-assistant | S16 | 0.1285 | 0.9305 | 2.017 | 0 | 24.88 | 0.882 | 3.406 | 0.785 | FAIL | not_evaluated |
| ni30_26__eng | N | 0.0004 | 1.0000 | 0.023 | 0 | 65.75 | 1.000 | 0.034 | 1.000 | pass | not_evaluated |
| ni30_26__eng | S | 0.1164 | 0.9647 | 1.432 | 0 | 26.06 | 0.831 | 2.895 | 0.732 | FAIL | not_evaluated |
| ni30_26__eng | S16 | 0.1164 | 0.9648 | 1.432 | 0 | 26.06 | 0.831 | 2.895 | 0.732 | FAIL | not_evaluated |
| ni30_27__ai-assistant | N | 0.0004 | 1.0000 | 0.027 | 0 | 65.15 | 1.000 | 0.039 | 1.000 | pass | not_evaluated |
| ni30_27__ai-assistant | S | 0.1348 | 0.8870 | 2.602 | 0 | 24.68 | 0.871 | 3.399 | 0.819 | FAIL | not_evaluated |
| ni30_27__ai-assistant | S16 | 0.1348 | 0.8875 | 2.600 | 0 | 24.68 | 0.871 | 3.400 | 0.819 | FAIL | not_evaluated |
| ni30_27__eng | N | 0.0004 | 1.0000 | 0.024 | 0 | 65.28 | 1.000 | 0.038 | 1.000 | pass | not_evaluated |
| ni30_27__eng | S | 0.1266 | 0.9273 | 2.306 | 0 | 25.35 | 0.822 | 2.822 | 0.875 | FAIL | not_evaluated |
| ni30_27__eng | S16 | 0.1266 | 0.9269 | 2.307 | 0 | 25.35 | 0.822 | 2.822 | 0.875 | FAIL | not_evaluated |
| ni30_28__ai-assistant | N | 0.0004 | 0.9996 | 0.074 | 0 | 64.61 | 1.000 | 0.044 | 1.000 | pass | not_evaluated |
| ni30_28__ai-assistant | S | 0.1289 | 0.9192 | 1.218 | 0 | 26.51 | 0.851 | 3.331 | 0.815 | FAIL | not_evaluated |
| ni30_28__ai-assistant | S16 | 0.1289 | 0.9183 | 1.222 | 0 | 26.51 | 0.851 | 3.331 | 0.815 | FAIL | not_evaluated |
| ni30_28__eng | N | 0.0004 | 0.9999 | 0.045 | 0 | 64.65 | 1.000 | 0.044 | 1.000 | pass | not_evaluated |
| ni30_28__eng | S | 0.1214 | 0.9487 | 0.985 | 0 | 27.46 | 0.804 | 2.775 | 0.770 | FAIL | not_evaluated |
| ni30_28__eng | S16 | 0.1214 | 0.9500 | 0.972 | 0 | 27.46 | 0.804 | 2.775 | 0.770 | FAIL | not_evaluated |
| ni30_29__ai-assistant | N | 0.0004 | 0.9986 | 0.132 | 0 | 64.49 | 1.000 | 0.045 | 1.000 | pass | not_evaluated |
| ni30_29__ai-assistant | S | 0.1509 | 0.8592 | 1.690 | 0 | 23.20 | 0.940 | 3.791 | 0.920 | FAIL | not_evaluated |
| ni30_29__ai-assistant | S16 | 0.1509 | 0.8580 | 1.698 | 0 | 23.20 | 0.940 | 3.791 | 0.920 | FAIL | not_evaluated |
| ni30_29__eng | N | 0.0004 | 0.9984 | 0.138 | 0 | 64.43 | 1.000 | 0.046 | 1.000 | pass | not_evaluated |
| ni30_29__eng | S | 0.1409 | 0.8657 | 1.685 | 0 | 24.12 | 0.897 | 3.180 | 0.854 | FAIL | not_evaluated |
| ni30_29__eng | S16 | 0.1409 | 0.8655 | 1.688 | 0 | 24.12 | 0.897 | 3.180 | 0.854 | FAIL | not_evaluated |
| ni30_30__ai-assistant | N | 0.0004 | 0.9996 | 0.106 | 0 | 64.54 | 1.000 | 0.044 | 1.000 | pass | not_evaluated |
| ni30_30__ai-assistant | S | 0.1085 | 0.8619 | 2.176 | -1 | 25.57 | 0.854 | 4.021 | 0.771 | FAIL | not_evaluated |
| ni30_30__ai-assistant | S16 | 0.1085 | 0.8631 | 2.162 | -1 | 25.57 | 0.854 | 4.021 | 0.771 | FAIL | not_evaluated |
| ni30_30__eng | N | 0.0004 | 0.9990 | 0.150 | 0 | 64.34 | 1.000 | 0.047 | 1.000 | pass | not_evaluated |
| ni30_30__eng | S | 0.1031 | 0.8998 | 1.975 | -1 | 26.13 | 0.795 | 3.550 | 0.820 | FAIL | not_evaluated |
| ni30_30__eng | S16 | 0.1031 | 0.8987 | 1.994 | -1 | 26.13 | 0.795 | 3.550 | 0.820 | FAIL | not_evaluated |
