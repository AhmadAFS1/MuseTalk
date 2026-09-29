### middle_eastern_man_full_beard__srcblkA8 — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcblkA8` (refined); identity `middle_eastern_man_full_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0740 | 0.0740 | 0.9991 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.0131 | 8.0100 | 0.1768 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 10.0653 | 10.0516 | 0.9986 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 9.2993 | 9.2989 | 1.0000 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 5.2559 | 5.2577 | 1.0003 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 13.1112 | 13.0990 | 0.9991 | report |  |
| Mouth-box >6 Hz temporal power | 2938.6 | 2923.3 | 0.9948 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 16.0 / 20.0 | 0.1915 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 12.0 / 14.0 | 0.2327 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 32.0 | 0.0022 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.0367 | 2.0609 | 0.0242 | <= A (2.0367) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.8397 | 1.7695 | -0.0702 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.1713 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.5871 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.1713 / 0.5871 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (over) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 61.56 / face 54.86 | mouth 47.93; worst face 45.58 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9998 / face 0.9992 | mouth 0.9972; worst mouth 0.9898 | report |  |
| Mouth sharpness (Laplacian var) | 22.8605 | 22.7957 | 0.9972 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.1161 | 45.1153 | 0.0036 | report |  |
