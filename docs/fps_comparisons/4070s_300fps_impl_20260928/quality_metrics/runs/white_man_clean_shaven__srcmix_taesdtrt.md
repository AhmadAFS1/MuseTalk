### white_man_clean_shaven__srcmix_taesdtrt — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcmix_taesdtrt` (refined); identity `white_man_clean_shaven`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0891 | 0.0891 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.7785 | 8.7746 | 0.0535 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 6.5105 | 6.5118 | 1.0002 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.2089 | 4.2102 | 1.0003 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 2.3010 | 2.3013 | 1.0001 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 8.5681 | 8.5715 | 1.0004 | report |  |
| Mouth-box >6 Hz temporal power | 1310.3 | 1311.8 | 1.0011 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 3.0 | 0.0415 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 2.0 | 0.0477 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 5.0 | 0.0008 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.1328 | 1.1338 | 0.0010 | <= A (1.1328) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.6628 | 1.6598 | -0.0029 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0550 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1783 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0550 / 0.1783 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 70.84 / face 63.21 | mouth 57.18; worst face 58.76 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 1.0000 / face 0.9997 | mouth 0.9992; worst mouth 0.9985 | report |  |
| Mouth sharpness (Laplacian var) | 22.9885 | 23.0069 | 1.0008 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 53.1347 | 53.1338 | 0.0046 | report |  |
