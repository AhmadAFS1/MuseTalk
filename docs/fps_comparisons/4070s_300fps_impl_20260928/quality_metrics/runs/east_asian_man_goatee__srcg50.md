### east_asian_man_goatee__srcg50 — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcg50` (refined); identity `east_asian_man_goatee`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0879 | 0.0879 | 0.9998 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.0220 | 9.0236 | 0.0841 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 8.9463 | 8.9398 | 0.9993 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 6.3069 | 6.3057 | 0.9998 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.1939 | 3.1948 | 1.0003 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 12.1163 | 12.1131 | 0.9997 | report |  |
| Mouth-box >6 Hz temporal power | 2592.2 | 2588.5 | 0.9986 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 7.0 / 9.0 | 0.0798 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 7.0 / 9.0 | 0.0929 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 8.0 | 0.0013 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.6024 | 2.6053 | 0.0029 | <= A (2.6024) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.1348 | 0.1251 | -0.0097 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0699 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.2202 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0699 / 0.2202 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 67.06 / face 59.68 | mouth 52.33; worst face 55.87 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9999 / face 0.9996 | mouth 0.9986; worst mouth 0.9977 | report |  |
| Mouth sharpness (Laplacian var) | 26.3635 | 26.2907 | 0.9972 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 54.8736 | 54.8728 | 0.0063 | report |  |
