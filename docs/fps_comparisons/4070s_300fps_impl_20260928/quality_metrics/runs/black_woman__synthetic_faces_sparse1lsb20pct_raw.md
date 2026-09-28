### black_woman__synthetic_faces_sparse1lsb20pct_raw — PASS (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_sparse1lsb20pct` (refined); identity `black_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0872 | 0.0872 | 1.0000 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.7519 | 9.7443 | 0.0426 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.3186 | 7.3332 | 1.0020 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.1249 | 4.1290 | 1.0010 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.9748 | 3.9827 | 1.0020 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.5302 | 9.5621 | 1.0034 | report |  |
| Mouth-box >6 Hz temporal power | 1714.2 | 1715.9 | 1.0010 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 4.0 | 0.0745 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.6 / 3.0 | 0.0909 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 6.0 | 0.0007 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.6228 | 1.6208 | -0.0020 | <= A (1.6228) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.2968 | 0.3122 | 0.0154 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0475 | <= 0.05 | PASS |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1426 | <= 0.15 | PASS |
| PSNR A vs B, dB mean frame (cap 100) | - | full 68.88 / face 61.67 | mouth 56.05; worst face 59.84 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9999 / face 0.9996 | mouth 0.9986; worst mouth 0.9984 | report |  |
| Mouth sharpness (Laplacian var) | 21.3647 | 21.7859 | 1.0197 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 42.2931 | 42.2929 | 0.0003 | report |  |
