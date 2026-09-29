### south_asian_woman__synthetic_faces_sparse1lsb20pct_raw — PASS (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_sparse1lsb20pct` (refined); identity `south_asian_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.1053 | 0.1053 | 1.0000 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 10.1771 | 10.1762 | 0.0321 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.2735 | 7.2889 | 1.0021 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.4892 | 4.4898 | 1.0001 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.2686 | 3.2781 | 1.0029 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.7756 | 9.8097 | 1.0035 | report |  |
| Mouth-box >6 Hz temporal power | 1973.5 | 1974.9 | 1.0007 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 3.0 | 0.0733 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 3.0 | 0.0893 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 2.0 | 0.0002 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.3558 | 2.3505 | -0.0053 | <= A (2.3558) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.8213 | 0.8234 | 0.0021 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0335 | <= 0.05 | PASS |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.0984 | <= 0.15 | PASS |
| PSNR A vs B, dB mean frame (cap 100) | - | full 70.61 / face 61.91 | mouth 56.03; worst face 60.89 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9999 / face 0.9996 | mouth 0.9986; worst mouth 0.9982 | report |  |
| Mouth sharpness (Laplacian var) | 30.7581 | 31.2931 | 1.0174 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.2190 | 45.2187 | 0.0003 | report |  |
