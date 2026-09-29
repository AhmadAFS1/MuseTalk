### middle_eastern_man_full_beard__refined_vs_standard_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined` (refined), B = `B_standard` (standard); identity `middle_eastern_man_full_beard`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0740 | 0.0721 | 0.9890 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.0131 | 7.8074 | 0.6124 | <= 0.5 | FAIL |
| Flicker mouth mean abs(dt) RGB | 10.0556 | 10.0637 | 1.0008 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 9.2539 | 6.6085 | 0.7141 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 6.8355 | 6.1876 | 0.9052 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 13.0990 | 13.0979 | 0.9999 | report |  |
| Mouth-box >6 Hz temporal power | 2951.1 | 2714.0 | 0.9197 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 116.0 / 122.0 | 2.9153 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 91.2 / 109.0 | 4.0674 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 63.0 | 0.0047 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 2.0367 | 0.5037 | -1.5330 | <= A (2.0367) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.8397 | 0.9854 | -0.8543 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 1.0604 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 4.9862 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 41.05 / face 34.35 | mouth 50.25; worst face 30.35 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9923 / face 0.9643 | mouth 0.9961; worst mouth 0.9716 | report |  |
| Mouth sharpness (Laplacian var) | 23.0430 | 20.1752 | 0.8755 | >= 0.95 | FAIL |
| Face Lab L* mean; delta = dE76 of means | 45.1161 | 45.1577 | 0.0543 | report |  |
