### white_man_clean_shaven__synthetic_faces_sparse1lsb20pct_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_sparse1lsb20pct` (refined); identity `white_man_clean_shaven`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0891 | 0.0891 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.7785 | 8.7759 | 0.0479 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 6.5109 | 6.5271 | 1.0025 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.2089 | 4.2156 | 1.0016 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 2.3009 | 2.3100 | 1.0039 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 8.5685 | 8.6052 | 1.0043 | report |  |
| Mouth-box >6 Hz temporal power | 1310.3 | 1312.1 | 1.0013 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 3.0 | 0.0798 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 2.0 | 0.0908 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 6.0 | 0.0008 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.1328 | 1.1374 | 0.0046 | <= A (1.1328) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.6628 | 1.7067 | 0.0440 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0555 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1926 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 68.81 / face 61.16 | mouth 55.81; worst face 58.61 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9999 / face 0.9995 | mouth 0.9985; worst mouth 0.9982 | report |  |
| Mouth sharpness (Laplacian var) | 22.9896 | 23.4746 | 1.0211 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 53.1347 | 53.1344 | 0.0004 | report |  |
