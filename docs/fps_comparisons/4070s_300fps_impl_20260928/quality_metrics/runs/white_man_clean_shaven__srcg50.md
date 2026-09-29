### white_man_clean_shaven__srcg50 — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcg50` (refined); identity `white_man_clean_shaven`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0891 | 0.0892 | 0.9997 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.7785 | 8.7838 | 0.0994 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 6.5107 | 6.5073 | 0.9995 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.2088 | 4.2084 | 0.9999 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 2.3011 | 2.3004 | 0.9997 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 8.5682 | 8.5729 | 1.0005 | report |  |
| Mouth-box >6 Hz temporal power | 1310.3 | 1311.5 | 1.0009 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 4.0 / 6.0 | 0.0661 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 3.6 / 4.0 | 0.0747 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 11.0 | 0.0014 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.1328 | 1.1305 | -0.0023 | <= A (1.1328) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.6628 | 1.6959 | 0.0332 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0848 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.2727 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0848 / 0.2727 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 67.66 / face 60.02 | mouth 52.91; worst face 53.02 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9999 / face 0.9996 | mouth 0.9985; worst mouth 0.9964 | report |  |
| Mouth sharpness (Laplacian var) | 22.9892 | 22.9187 | 0.9969 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 53.1347 | 53.1327 | 0.0054 | report |  |
