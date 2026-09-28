Range across identities per calibration category (min .. max):

| Category | n | Ap. corr | Lag | Ap. abs-delta px | Flk mouth | Flk jaw | Flk ring | Band max p99 | Chin B-A px | LM mean px | LM p99 px | PSNR face | PSNR mouth | SSIM mouth | Sharp B/A | dE76 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| self | 18 | 1.000 | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0.000 | 0.000 | 0.000 | 100.000 | 100.000 | 1.000 | 1.000 | 0.000 |
| codec | 12 | 0.998 .. 0.999 | 0 | 0.234 .. 0.380 | 0.985 .. 1.000 | 0.921 .. 1.007 | 0.931 .. 1.005 | 21.610 .. 30.000 | -0.033 .. 0.099 | 0.238 .. 0.353 | 0.654 .. 1.050 | 37.578 .. 41.562 | 37.254 .. 41.072 | 0.980 .. 0.985 | 0.986 .. 1.402 | 0.025 .. 0.941 |
| synthetic_sparse1lsb20pct | 6 | 1.000 .. 1.000 | 0 | 0.032 .. 0.073 | 1.001 .. 1.002 | 1.000 .. 1.002 | 1.001 .. 1.004 | 2.000 .. 5.610 | -0.005 .. 0.012 | 0.034 .. 0.083 | 0.098 .. 0.431 | 60.861 .. 61.908 | 55.702 .. 56.048 | 0.999 .. 0.999 | 1.012 .. 1.021 | 0.000 .. 0.000 |
| synthetic_noise1lsb | 6 | 1.000 .. 1.000 | 0 | 0.044 .. 0.073 | 1.003 .. 1.008 | 1.000 .. 1.003 | 1.004 .. 1.014 | 3.000 .. 6.000 | -0.012 .. 0.013 | 0.045 .. 0.088 | 0.128 .. 0.335 | 56.884 .. 57.818 | 51.543 .. 51.914 | 0.997 .. 0.997 | 1.039 .. 1.065 | 0.000 .. 0.001 |
| synthetic_noise3lsb | 6 | 1.000 .. 1.000 | 0 | 0.068 .. 0.122 | 1.020 .. 1.044 | 1.004 .. 1.015 | 1.020 .. 1.073 | 4.000 .. 11.000 | -0.014 .. 0.019 | 0.072 .. 0.123 | 0.212 .. 0.472 | 51.129 .. 52.058 | 45.302 .. 45.773 | 0.991 .. 0.993 | 1.211 .. 1.346 | 0.001 .. 0.002 |
| sensitivity_refined_vs_standard | 12 | 0.989 .. 0.995 | 0 | 0.422 .. 0.804 | 0.994 .. 1.002 | 0.702 .. 0.894 | 0.902 .. 1.028 | 65.610 .. 120.610 | -1.843 .. -0.483 | 0.633 .. 1.101 | 2.426 .. 5.008 | 32.792 .. 39.712 | 37.708 .. 50.254 | 0.974 .. 0.996 | 0.869 .. 0.996 | 0.054 .. 0.232 |
| sensitivity_int8_vs_taesd | 2 | 0.992 .. 0.993 | 0 | 0.621 .. 0.650 | 0.924 .. 0.950 | 0.993 .. 0.998 | 0.962 .. 0.986 | 26.610 .. 28.610 | -0.061 .. -0.038 | 0.473 .. 0.499 | 1.381 .. 1.411 | 39.653 .. 39.901 | 34.681 .. 35.082 | 0.970 .. 0.972 | 0.791 .. 0.839 | 0.020 .. 0.317 |

Every comparison:

| Comparison | Verdict | Ap. corr | Lag | Ap. abs-delta px | Flk mouth B/A | Flk jaw B/A | Flk ring B/A | Band max p99 | Lip A | Lip B | Chin A px | Chin B px | LM mean px | LM p99 px | PSNR face | PSNR mouth | SSIM mouth | Sharp B/A | dE76 | Failed gates |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| black_man_short_beard self_video | INCOMPLETE | 1.000 | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | n/a | n/a | 1.727 | 1.727 | 0.000 | 0.000 | 100.000 | 100.000 | 1.000 | 1.000 | 0.000 | - |
| black_man_short_beard self_raw | PASS | 1.000 | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0 | 1.720 | 1.720 | 0.000 | 0.000 | 100.000 | 100.000 | 1.000 | 1.000 | 0.000 | - |
| black_man_short_beard codec_raw_vs_crf18 | INCOMPLETE | 0.998 | 0 | 0.380 | 0.985 | 0.921 | 0.931 | 26.610 | 0 | n/a | 1.720 | 1.727 | 0.331 | 0.987 | 37.984 | 37.466 | 0.982 | 1.264 | 0.941 | - |
| black_man_short_beard codec_crf18fast_vs_crf18medium | INCOMPLETE | 0.998 | 0 | 0.369 | 1.000 | 1.007 | 0.999 | 22.610 | n/a | n/a | 1.727 | 1.707 | 0.324 | 0.965 | 40.778 | 40.270 | 0.981 | 0.988 | 0.034 | - |
| black_man_short_beard faces_only_retracked_raw | PASS | 1.000 | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0 | 1.720 | 1.720 | 0.000 | 0.000 | 100.000 | 100.000 | 1.000 | 1.000 | 0.000 | - |
| black_man_short_beard synthetic_faces_sparse1lsb20pct_raw | FAIL | 1.000 | 0 | 0.062 | 1.002 | 1.000 | 1.002 | 3.000 | 0 | 0 | 1.720 | 1.722 | 0.056 | 0.169 | 60.982 | 55.736 | 0.999 | 1.012 | 0.000 | chin.landmark_dev_mean_px, chin.landmark_dev_p99_px |
| black_man_short_beard synthetic_faces_noise1lsb_raw | FAIL | 1.000 | 0 | 0.073 | 1.006 | 1.001 | 1.007 | 5.000 | 0 | 0 | 1.720 | 1.725 | 0.069 | 0.211 | 57.063 | 51.591 | 0.997 | 1.039 | 0.000 | chin.landmark_dev_mean_px, chin.landmark_dev_p99_px |
| black_man_short_beard synthetic_faces_noise3lsb_raw | FAIL | 1.000 | 0 | 0.122 | 1.030 | 1.007 | 1.042 | 7.000 | 0 | 0 | 1.720 | 1.717 | 0.113 | 0.340 | 51.404 | 45.379 | 0.992 | 1.211 | 0.001 | chin.landmark_dev_mean_px, chin.landmark_dev_p99_px |
| black_man_short_beard refined_vs_standard_video | FAIL | 0.990 | 0 | 0.804 | 1.001 | 0.798 | 0.934 | 92.000 | n/a | n/a | 1.727 | 0.474 | 0.806 | 3.219 | 35.761 | 39.338 | 0.977 | 0.921 | 0.057 | lip.mean_abs_delta_px, overall.mouth_sharpness_ratio |
| black_man_short_beard refined_vs_standard_raw | FAIL | 0.992 | 0 | 0.716 | 1.000 | 0.782 | 0.917 | 93.610 | 0 | 0 | 1.720 | 0.474 | 0.749 | 3.155 | 36.857 | 46.947 | 0.993 | 0.869 | 0.060 | lip.mean_abs_delta_px, chin.landmark_dev_mean_px, chin.landmark_dev_p99_px, overall.mouth_sharpness_ratio |
| black_woman self_video | INCOMPLETE | 1.000 | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | n/a | n/a | 1.670 | 1.670 | 0.000 | 0.000 | 100.000 | 100.000 | 1.000 | 1.000 | 0.000 | - |
| black_woman self_raw | PASS | 1.000 | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0 | 1.623 | 1.623 | 0.000 | 0.000 | 100.000 | 100.000 | 1.000 | 1.000 | 0.000 | - |
| black_woman codec_raw_vs_crf18 | INCOMPLETE | 0.998 | 0 | 0.284 | 0.994 | 0.937 | 0.980 | 24.000 | 0 | n/a | 1.623 | 1.670 | 0.288 | 0.790 | 38.190 | 37.872 | 0.983 | 1.371 | 0.885 | - |
| black_woman codec_crf18fast_vs_crf18medium | INCOMPLETE | 0.998 | 0 | 0.274 | 1.000 | 0.998 | 0.997 | 24.000 | n/a | n/a | 1.670 | 1.706 | 0.288 | 0.756 | 41.060 | 40.822 | 0.981 | 0.986 | 0.044 | - |
| black_woman faces_only_retracked_raw | PASS | 1.000 | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0 | 1.623 | 1.623 | 0.000 | 0.000 | 100.000 | 100.000 | 1.000 | 1.000 | 0.000 | - |
| black_woman synthetic_faces_sparse1lsb20pct_raw | PASS | 1.000 | 0 | 0.043 | 1.002 | 1.001 | 1.002 | 3.000 | 0 | 0 | 1.623 | 1.621 | 0.048 | 0.143 | 61.674 | 56.048 | 0.999 | 1.020 | 0.000 | - |
| black_woman synthetic_faces_noise1lsb_raw | FAIL | 1.000 | 0 | 0.062 | 1.006 | 1.001 | 1.006 | 4.000 | 0 | 0 | 1.623 | 1.622 | 0.062 | 0.183 | 57.637 | 51.914 | 0.997 | 1.060 | 0.001 | chin.landmark_dev_mean_px, chin.landmark_dev_p99_px |
| black_woman synthetic_faces_noise3lsb_raw | FAIL | 1.000 | 0 | 0.104 | 1.035 | 1.007 | 1.034 | 6.610 | 0 | 0 | 1.623 | 1.609 | 0.102 | 0.297 | 51.928 | 45.773 | 0.991 | 1.322 | 0.001 | chin.landmark_dev_mean_px, chin.landmark_dev_p99_px |
| black_woman refined_vs_standard_video | FAIL | 0.993 | 0 | 0.549 | 0.997 | 0.894 | 0.984 | 67.000 | n/a | n/a | 1.670 | 0.765 | 0.725 | 2.720 | 37.480 | 38.449 | 0.976 | 0.960 | 0.141 | lip.mean_abs_delta_px |
| black_woman refined_vs_standard_raw | FAIL | 0.994 | 0 | 0.481 | 0.994 | 0.864 | 0.965 | 70.220 | 0 | 0 | 1.623 | 0.690 | 0.658 | 2.571 | 39.712 | 42.347 | 0.992 | 0.925 | 0.139 | chin.landmark_dev_mean_px, chin.landmark_dev_p99_px, overall.mouth_sharpness_ratio |
| east_asian_man_goatee self_video | INCOMPLETE | 1.000 | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | n/a | n/a | 2.682 | 2.682 | 0.000 | 0.000 | 100.000 | 100.000 | 1.000 | 1.000 | 0.000 | - |
| east_asian_man_goatee self_raw | PASS | 1.000 | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0 | 2.602 | 2.602 | 0.000 | 0.000 | 100.000 | 100.000 | 1.000 | 1.000 | 0.000 | - |
| east_asian_man_goatee codec_raw_vs_crf18 | INCOMPLETE | 0.999 | 0 | 0.236 | 0.990 | 0.948 | 0.968 | 27.610 | 0 | n/a | 2.602 | 2.682 | 0.255 | 0.695 | 38.199 | 37.488 | 0.985 | 1.402 | 0.907 | - |
| east_asian_man_goatee codec_crf18fast_vs_crf18medium | INCOMPLETE | 0.999 | 0 | 0.234 | 1.000 | 1.002 | 0.995 | 23.610 | n/a | n/a | 2.682 | 2.649 | 0.239 | 0.654 | 41.142 | 40.216 | 0.982 | 0.990 | 0.033 | - |
| east_asian_man_goatee faces_only_retracked_raw | PASS | 1.000 | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0 | 2.602 | 2.602 | 0.000 | 0.000 | 100.000 | 100.000 | 1.000 | 1.000 | 0.000 | - |
| east_asian_man_goatee synthetic_faces_sparse1lsb20pct_raw | PASS | 1.000 | 0 | 0.035 | 1.001 | 1.000 | 1.003 | 3.000 | 0 | 0 | 2.602 | 2.603 | 0.040 | 0.120 | 61.316 | 55.765 | 0.999 | 1.017 | 0.000 | - |
| east_asian_man_goatee synthetic_faces_noise1lsb_raw | FAIL | 1.000 | 0 | 0.045 | 1.005 | 1.000 | 1.009 | 3.000 | 0 | 0 | 2.602 | 2.604 | 0.052 | 0.153 | 57.403 | 51.612 | 0.997 | 1.053 | 0.001 | chin.landmark_dev_mean_px, chin.landmark_dev_p99_px |
| east_asian_man_goatee synthetic_faces_noise3lsb_raw | FAIL | 1.000 | 0 | 0.078 | 1.026 | 1.005 | 1.049 | 5.610 | 0 | 0 | 2.602 | 2.602 | 0.087 | 0.245 | 51.715 | 45.396 | 0.992 | 1.294 | 0.001 | chin.landmark_dev_mean_px, chin.landmark_dev_p99_px |
| east_asian_man_goatee refined_vs_standard_video | FAIL | 0.991 | 0 | 0.561 | 1.002 | 0.864 | 0.912 | 112.050 | n/a | n/a | 2.682 | 0.864 | 0.922 | 3.808 | 32.792 | 39.322 | 0.980 | 0.996 | 0.155 | lip.mean_abs_delta_px |
| east_asian_man_goatee refined_vs_standard_raw | FAIL | 0.991 | 0 | 0.561 | 1.001 | 0.855 | 0.902 | 113.270 | 0 | 0 | 2.602 | 0.759 | 0.924 | 3.856 | 33.345 | 48.301 | 0.996 | 0.986 | 0.151 | lip.mean_abs_delta_px, chin.landmark_dev_mean_px, chin.landmark_dev_p99_px |
| middle_eastern_man_full_beard self_video | INCOMPLETE | 1.000 | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | n/a | n/a | 2.082 | 2.082 | 0.000 | 0.000 | 100.000 | 100.000 | 1.000 | 1.000 | 0.000 | - |
| middle_eastern_man_full_beard self_raw | PASS | 1.000 | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0 | 2.037 | 2.037 | 0.000 | 0.000 | 100.000 | 100.000 | 1.000 | 1.000 | 0.000 | - |
| middle_eastern_man_full_beard codec_raw_vs_crf18 | INCOMPLETE | 0.998 | 0 | 0.279 | 0.988 | 0.946 | 0.941 | 30.000 | 0 | n/a | 2.037 | 2.082 | 0.353 | 1.050 | 37.578 | 37.254 | 0.982 | 1.387 | 0.941 | - |
| middle_eastern_man_full_beard codec_crf18fast_vs_crf18medium | INCOMPLETE | 0.998 | 0 | 0.270 | 1.000 | 1.006 | 1.005 | 30.000 | n/a | n/a | 2.082 | 2.085 | 0.338 | 1.031 | 40.056 | 39.897 | 0.980 | 0.995 | 0.025 | - |
| middle_eastern_man_full_beard faces_only_retracked_raw | PASS | 1.000 | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0 | 2.037 | 2.037 | 0.000 | 0.000 | 100.000 | 100.000 | 1.000 | 1.000 | 0.000 | - |
| middle_eastern_man_full_beard synthetic_faces_sparse1lsb20pct_raw | FAIL | 1.000 | 0 | 0.073 | 1.001 | 1.000 | 1.001 | 5.610 | 0 | 0 | 2.037 | 2.049 | 0.083 | 0.431 | 60.861 | 55.702 | 0.999 | 1.016 | 0.000 | chin.landmark_dev_mean_px, chin.landmark_dev_p99_px |
| middle_eastern_man_full_beard synthetic_faces_noise1lsb_raw | FAIL | 1.000 | 0 | 0.067 | 1.003 | 1.001 | 1.004 | 6.000 | 0 | 0 | 2.037 | 2.050 | 0.088 | 0.335 | 56.884 | 51.543 | 0.997 | 1.052 | 0.000 | chin.landmark_dev_mean_px, chin.landmark_dev_p99_px |
| middle_eastern_man_full_beard synthetic_faces_noise3lsb_raw | FAIL | 1.000 | 0 | 0.103 | 1.020 | 1.004 | 1.020 | 11.000 | 0 | 0 | 2.037 | 2.033 | 0.123 | 0.472 | 51.129 | 45.302 | 0.993 | 1.295 | 0.001 | chin.landmark_dev_mean_px, chin.landmark_dev_p99_px |
| middle_eastern_man_full_beard refined_vs_standard_video | FAIL | 0.990 | 0 | 0.616 | 1.001 | 0.702 | 0.910 | 120.610 | n/a | n/a | 2.082 | 0.544 | 1.101 | 5.008 | 33.504 | 39.186 | 0.979 | 0.942 | 0.063 | lip.mean_abs_delta_px, overall.mouth_sharpness_ratio |
| middle_eastern_man_full_beard refined_vs_standard_raw | FAIL | 0.989 | 0 | 0.612 | 1.001 | 0.714 | 0.905 | 116.050 | 0 | 0 | 2.037 | 0.504 | 1.060 | 4.986 | 34.349 | 50.254 | 0.996 | 0.876 | 0.054 | lip.mean_abs_delta_px, chin.landmark_dev_mean_px, chin.landmark_dev_p99_px, overall.mouth_sharpness_ratio |
| south_asian_woman self_video | INCOMPLETE | 1.000 | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | n/a | n/a | 2.455 | 2.455 | 0.000 | 0.000 | 100.000 | 100.000 | 1.000 | 1.000 | 0.000 | - |
| south_asian_woman self_raw | PASS | 1.000 | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0 | 2.356 | 2.356 | 0.000 | 0.000 | 100.000 | 100.000 | 1.000 | 1.000 | 0.000 | - |
| south_asian_woman codec_raw_vs_crf18 | INCOMPLETE | 0.998 | 0 | 0.306 | 0.996 | 0.926 | 0.978 | 25.000 | 0 | n/a | 2.356 | 2.455 | 0.260 | 0.743 | 38.462 | 37.842 | 0.984 | 1.275 | 0.823 | - |
| south_asian_woman codec_crf18fast_vs_crf18medium | INCOMPLETE | 0.999 | 0 | 0.246 | 1.000 | 0.999 | 0.993 | 24.000 | n/a | n/a | 2.455 | 2.450 | 0.238 | 0.666 | 41.328 | 40.702 | 0.982 | 0.995 | 0.035 | - |
| south_asian_woman faces_only_retracked_raw | PASS | 1.000 | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0 | 2.356 | 2.356 | 0.000 | 0.000 | 100.000 | 100.000 | 1.000 | 1.000 | 0.000 | - |
| south_asian_woman synthetic_faces_sparse1lsb20pct_raw | PASS | 1.000 | 0 | 0.032 | 1.002 | 1.000 | 1.003 | 2.000 | 0 | 0 | 2.356 | 2.351 | 0.034 | 0.098 | 61.908 | 56.029 | 0.999 | 1.017 | 0.000 | - |
| south_asian_woman synthetic_faces_noise1lsb_raw | PASS | 1.000 | 0 | 0.044 | 1.007 | 1.001 | 1.009 | 3.000 | 0 | 0 | 2.356 | 2.344 | 0.045 | 0.128 | 57.818 | 51.877 | 0.997 | 1.055 | 0.001 | - |
| south_asian_woman synthetic_faces_noise3lsb_raw | FAIL | 1.000 | 0 | 0.068 | 1.037 | 1.004 | 1.048 | 4.000 | 0 | 0 | 2.356 | 2.352 | 0.072 | 0.212 | 52.058 | 45.723 | 0.992 | 1.299 | 0.001 | chin.landmark_dev_mean_px, chin.landmark_dev_p99_px |
| south_asian_woman refined_vs_standard_video | INCOMPLETE | 0.994 | 0 | 0.431 | 1.000 | 0.887 | 1.018 | 74.220 | n/a | n/a | 2.455 | 1.252 | 0.734 | 3.189 | 37.016 | 37.708 | 0.974 | 0.985 | 0.114 | - |
| south_asian_woman refined_vs_standard_raw | FAIL | 0.995 | 0 | 0.437 | 0.997 | 0.861 | 0.995 | 70.220 | 0 | 0 | 2.356 | 1.124 | 0.704 | 3.154 | 38.740 | 40.981 | 0.989 | 0.967 | 0.116 | chin.landmark_dev_mean_px, chin.landmark_dev_p99_px |
| white_man_clean_shaven self_video | INCOMPLETE | 1.000 | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | n/a | n/a | 1.164 | 1.164 | 0.000 | 0.000 | 100.000 | 100.000 | 1.000 | 1.000 | 0.000 | - |
| white_man_clean_shaven self_raw | PASS | 1.000 | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0 | 1.133 | 1.133 | 0.000 | 0.000 | 100.000 | 100.000 | 1.000 | 1.000 | 0.000 | - |
| white_man_clean_shaven codec_raw_vs_crf18 | INCOMPLETE | 0.998 | 0 | 0.306 | 0.990 | 0.936 | 0.948 | 22.610 | 0 | n/a | 1.133 | 1.164 | 0.302 | 0.874 | 38.417 | 37.919 | 0.985 | 1.362 | 0.913 | - |
| white_man_clean_shaven codec_crf18fast_vs_crf18medium | INCOMPLETE | 0.998 | 0 | 0.284 | 1.000 | 1.005 | 0.998 | 21.610 | n/a | n/a | 1.164 | 1.162 | 0.285 | 0.847 | 41.562 | 41.072 | 0.983 | 0.996 | 0.047 | - |
| white_man_clean_shaven faces_only_retracked_raw | PASS | 1.000 | 0 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 0 | 0 | 1.133 | 1.133 | 0.000 | 0.000 | 100.000 | 100.000 | 1.000 | 1.000 | 0.000 | - |
| white_man_clean_shaven synthetic_faces_sparse1lsb20pct_raw | FAIL | 1.000 | 0 | 0.048 | 1.002 | 1.002 | 1.004 | 3.000 | 0 | 0 | 1.133 | 1.137 | 0.055 | 0.193 | 61.159 | 55.809 | 0.999 | 1.021 | 0.000 | chin.landmark_dev_mean_px, chin.landmark_dev_p99_px |
| white_man_clean_shaven synthetic_faces_noise1lsb_raw | FAIL | 1.000 | 0 | 0.067 | 1.008 | 1.003 | 1.014 | 3.610 | 0 | 0 | 1.133 | 1.138 | 0.072 | 0.227 | 57.211 | 51.670 | 0.997 | 1.065 | 0.001 | chin.landmark_dev_mean_px, chin.landmark_dev_p99_px |
| white_man_clean_shaven synthetic_faces_noise3lsb_raw | FAIL | 1.000 | 0 | 0.102 | 1.044 | 1.015 | 1.073 | 5.610 | 0 | 0 | 1.133 | 1.152 | 0.103 | 0.306 | 51.561 | 45.496 | 0.991 | 1.346 | 0.002 | flicker.ring_ratio, chin.landmark_dev_mean_px, chin.landmark_dev_p99_px |
| white_man_clean_shaven refined_vs_standard_video | INCOMPLETE | 0.994 | 0 | 0.486 | 1.001 | 0.826 | 1.028 | 66.000 | n/a | n/a | 1.164 | 0.681 | 0.656 | 2.426 | 36.348 | 38.680 | 0.980 | 0.995 | 0.232 | - |
| white_man_clean_shaven refined_vs_standard_raw | FAIL | 0.995 | 0 | 0.422 | 0.999 | 0.808 | 1.000 | 65.610 | 0 | 0 | 1.133 | 0.632 | 0.633 | 2.461 | 37.743 | 43.736 | 0.994 | 0.980 | 0.230 | chin.landmark_dev_mean_px, chin.landmark_dev_p99_px |
| japanese int8_vs_taesd_video | FAIL | 0.992 | 0 | 0.650 | 0.950 | 0.998 | 0.986 | 26.610 | n/a | n/a | 1.128 | 1.068 | 0.473 | 1.381 | 39.653 | 34.681 | 0.972 | 0.839 | 0.317 | lip.mean_abs_delta_px, overall.mouth_sharpness_ratio |
| latina int8_vs_taesd_video | FAIL | 0.993 | 0 | 0.621 | 0.924 | 0.993 | 0.962 | 28.610 | n/a | n/a | 1.052 | 1.014 | 0.499 | 1.411 | 39.901 | 35.082 | 0.970 | 0.791 | 0.020 | lip.mean_abs_delta_px, overall.mouth_sharpness_ratio |

SyncNet (LatentSync 16-frame pixel model; uncalibrated, relative only). Offset k = the audio window starting k frames after the video window matches best (k > 0: lips lead the audio). Same model, crop and audio for every arm:

| Identity | Arm | Confidence (max - median) | Offset | Cos-sim at 0 | Delta conf vs first arm |
|---|---|---:|---:|---:|---:|
| black_man_short_beard | refined_mp4 | 0.6072 | 2 | 0.1641 | 0.0000 |
| black_man_short_beard | standard_mp4 | 0.6295 | 2 | 0.1492 | 0.0223 |
| black_woman | refined_mp4 | 0.5175 | 3 | 0.1428 | 0.0000 |
| black_woman | standard_mp4 | 0.5157 | 3 | 0.1328 | -0.0017 |
| east_asian_man_goatee | refined_mp4 | 0.5752 | 2 | 0.1876 | 0.0000 |
| east_asian_man_goatee | standard_mp4 | 0.5943 | 2 | 0.1819 | 0.0191 |
| middle_eastern_man_full_beard | refined_mp4 | 0.5707 | 2 | 0.1392 | 0.0000 |
| middle_eastern_man_full_beard | standard_mp4 | 0.5896 | 2 | 0.1320 | 0.0190 |
| south_asian_woman | refined_mp4 | 0.5797 | 3 | 0.1395 | 0.0000 |
| south_asian_woman | standard_mp4 | 0.5768 | 3 | 0.1399 | -0.0029 |
| white_man_clean_shaven | refined_mp4 | 0.6218 | 1 | 0.4755 | 0.0000 |
| white_man_clean_shaven | standard_mp4 | 0.6290 | 1 | 0.4674 | 0.0071 |
| japanese | approved_INT8_chin100 | 0.1790 | -1 | 0.3016 | 0.0000 |
| japanese | TAESD_chin100 | 0.1728 | -1 | 0.2991 | -0.0062 |
| japanese | H3_source | 0.0422 | 5 | 0.1081 | -0.1368 |
| latina | approved_INT8_chin100 | 0.6555 | 2 | 0.1947 | 0.0000 |
| latina | TAESD_chin100 | 0.6580 | 2 | 0.1961 | 0.0025 |
| latina | H3_source | 0.0473 | -2 | 0.1134 | -0.6082 |
