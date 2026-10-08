### white_man_clean_shaven__portable_ref1 — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `portable_ref1` (refined); identity `white_man_clean_shaven`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0891 | 0.0892 | 0.9997 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.7785 | 8.7798 | 0.0958 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 6.5099 | 6.5068 | 0.9995 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.2086 | 4.2071 | 0.9996 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 2.3011 | 2.3008 | 0.9999 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 8.5671 | 8.5598 | 0.9992 | report |  |
| Mouth-box >6 Hz temporal power | 1310.3 | 1311.1 | 1.0006 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 4.6 / 7.0 | 0.0702 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 3.6 / 4.0 | 0.0801 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 9.0 | 0.0016 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.1328 | 1.1332 | 0.0004 | <= A (1.1328) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.6628 | 1.6533 | -0.0095 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0871 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.2575 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0871 / 0.2575 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 67.43 / face 59.80 | mouth 52.86; worst face 50.71 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9999 / face 0.9996 | mouth 0.9985; worst mouth 0.9948 | report |  |
| Mouth sharpness (Laplacian var) | 22.9842 | 22.9771 | 0.9997 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 53.1347 | 53.1325 | 0.0051 | report |  |
