### middle_eastern_man_full_beard__stagewise16_taesdtrt — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `stagewise16_taesdtrt` (refined); identity `middle_eastern_man_full_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0740 | 0.0740 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.0131 | 8.0031 | 0.0721 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 10.0665 | 10.0647 | 0.9998 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 9.3012 | 9.3038 | 1.0003 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 5.2546 | 5.2540 | 0.9999 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 13.1126 | 13.1112 | 0.9999 | report |  |
| Mouth-box >6 Hz temporal power | 2938.6 | 2935.8 | 0.9990 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 6.0 / 6.0 | 0.0762 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 4.0 / 6.0 | 0.0948 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 9.0 | 0.0009 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.0367 | 2.0574 | 0.0207 | <= A (2.0367) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.8397 | 1.8157 | -0.0240 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0914 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.3850 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0914 / 0.3850 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (over) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 67.54 / face 60.85 | mouth 55.12; worst face 55.71 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9999 / face 0.9997 | mouth 0.9990; worst mouth 0.9981 | report |  |
| Mouth sharpness (Laplacian var) | 22.8321 | 22.8299 | 0.9999 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.1161 | 45.1159 | 0.0025 | report |  |
