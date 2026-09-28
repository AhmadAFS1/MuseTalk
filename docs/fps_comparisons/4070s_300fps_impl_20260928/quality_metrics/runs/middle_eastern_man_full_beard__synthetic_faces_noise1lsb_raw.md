### middle_eastern_man_full_beard__synthetic_faces_noise1lsb_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_noise1lsb` (refined); identity `middle_eastern_man_full_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0740 | 0.0740 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.0131 | 8.0059 | 0.0673 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 10.0660 | 10.1011 | 1.0035 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 9.3013 | 9.3128 | 1.0012 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 5.2543 | 5.2735 | 1.0037 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 13.1122 | 13.1876 | 1.0057 | report |  |
| Mouth-box >6 Hz temporal power | 2938.6 | 2944.9 | 1.0021 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 6.0 / 9.0 | 0.2053 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 3.0 | 0.2314 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 9.0 | 0.0009 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.0367 | 2.0496 | 0.0129 | <= A (2.0367) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.8397 | 1.8462 | 0.0065 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0875 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.3348 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 63.59 / face 56.88 | mouth 51.54; worst face 54.19 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9998 / face 0.9991 | mouth 0.9974; worst mouth 0.9969 | report |  |
| Mouth sharpness (Laplacian var) | 22.8357 | 24.0265 | 1.0521 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.1161 | 45.1159 | 0.0003 | report |  |
