### middle_eastern_man_full_beard__srcg50 — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcg50` (refined); identity `middle_eastern_man_full_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0740 | 0.0740 | 0.9997 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.0131 | 8.0092 | 0.1057 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 10.0656 | 10.0710 | 1.0005 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 9.3001 | 9.3008 | 1.0001 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 5.2552 | 5.2545 | 0.9999 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 13.1106 | 13.1246 | 1.0011 | report |  |
| Mouth-box >6 Hz temporal power | 2938.6 | 2933.9 | 0.9984 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 8.0 / 9.0 | 0.1274 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 6.0 / 7.0 | 0.1576 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 15.0 | 0.0014 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.0367 | 2.0565 | 0.0198 | <= A (2.0367) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.8397 | 1.8141 | -0.0256 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.1263 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.4672 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.1263 / 0.4672 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (over) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 64.34 / face 57.63 | mouth 51.02; worst face 53.00 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9999 / face 0.9995 | mouth 0.9982; worst mouth 0.9965 | report |  |
| Mouth sharpness (Laplacian var) | 22.8504 | 22.8132 | 0.9984 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 45.1161 | 45.1155 | 0.0031 | report |  |
