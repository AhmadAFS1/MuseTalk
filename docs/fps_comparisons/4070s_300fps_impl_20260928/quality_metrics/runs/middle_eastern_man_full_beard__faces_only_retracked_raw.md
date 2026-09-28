### middle_eastern_man_full_beard__faces_only_retracked_raw — PASS (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_only` (refined); identity `middle_eastern_man_full_beard`; pre-encode/decoded frames SHA-identical: **True**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0740 | 0.0740 | 1.0000 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.0131 | 8.0131 | 0.0000 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 10.0672 | 10.0672 | 1.0000 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 9.3036 | 9.3036 | 1.0000 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 5.2537 | 5.2537 | 1.0000 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 13.1140 | 13.1140 | 1.0000 | report |  |
| Mouth-box >6 Hz temporal power | 2938.6 | 2938.6 | 1.0000 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 0.0 / 0.0 | 0.0000 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 0.0 / 0.0 | 0.0000 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 0.0 | 0.0000 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.0367 | 2.0367 | 0.0000 | <= A (2.0367) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.8397 | 1.8397 | 0.0000 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0000 | <= 0.05 | PASS |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.0000 | <= 0.15 | PASS |
| PSNR A vs B, dB mean frame (cap 100) | - | full 100.00 / face 100.00 | mouth 100.00; worst face 100.00 | report-only | REPORT |
| SSIM A vs B mean | - | full 1.0000 / face 1.0000 | mouth 1.0000; worst mouth 1.0000 | report |  |
| Mouth sharpness (Laplacian var) | 22.8263 | 22.8263 | 1.0000 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.1161 | 45.1161 | 0.0000 | report |  |
