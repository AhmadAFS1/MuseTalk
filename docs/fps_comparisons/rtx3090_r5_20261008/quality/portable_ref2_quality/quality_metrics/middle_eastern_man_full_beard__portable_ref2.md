### middle_eastern_man_full_beard__portable_ref2 — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `portable_ref2` (refined); identity `middle_eastern_man_full_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0740 | 0.0739 | 0.9996 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.0131 | 7.9980 | 0.1184 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 10.0659 | 10.0655 | 1.0000 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 9.3016 | 9.3026 | 1.0001 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 5.2550 | 5.2546 | 0.9999 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 13.1109 | 13.1066 | 0.9997 | report |  |
| Mouth-box >6 Hz temporal power | 2938.6 | 2927.9 | 0.9963 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 8.6 / 11.0 | 0.1303 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 5.6 / 8.0 | 0.1607 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 14.0 | 0.0014 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.0367 | 2.0695 | 0.0328 | <= A (2.0367) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.8397 | 1.8652 | 0.0255 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.1245 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.4220 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.1245 / 0.4220 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (over) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 64.21 / face 57.51 | mouth 50.94; worst face 52.92 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9999 / face 0.9995 | mouth 0.9982; worst mouth 0.9959 | report |  |
| Mouth sharpness (Laplacian var) | 22.8807 | 22.8429 | 0.9983 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.1161 | 45.1160 | 0.0030 | report |  |
