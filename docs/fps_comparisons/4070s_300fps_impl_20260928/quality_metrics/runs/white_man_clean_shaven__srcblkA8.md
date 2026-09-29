### white_man_clean_shaven__srcblkA8 — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcblkA8` (refined); identity `white_man_clean_shaven`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0891 | 0.0896 | 0.9993 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.7785 | 8.8210 | 0.1344 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 6.5089 | 6.5252 | 1.0025 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.2082 | 4.2047 | 0.9992 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 2.3009 | 2.3041 | 1.0014 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 8.5655 | 8.5932 | 1.0032 | report |  |
| Mouth-box >6 Hz temporal power | 1310.3 | 1320.6 | 1.0078 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 6.6 / 9.0 | 0.0989 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 6.0 / 9.0 | 0.1131 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 9.0 | 0.0019 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.1328 | 1.1389 | 0.0061 | <= A (1.1328) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.6628 | 1.6293 | -0.0335 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.1158 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.4101 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.1158 / 0.4101 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (over) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 65.23 / face 57.58 | mouth 50.06; worst face 49.53 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9999 / face 0.9994 | mouth 0.9978; worst mouth 0.9918 | report |  |
| Mouth sharpness (Laplacian var) | 22.9765 | 23.0789 | 1.0045 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 53.1347 | 53.1333 | 0.0080 | report |  |
