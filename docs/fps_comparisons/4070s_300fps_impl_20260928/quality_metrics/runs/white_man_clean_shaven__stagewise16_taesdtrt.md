### white_man_clean_shaven__stagewise16_taesdtrt — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `stagewise16_taesdtrt` (refined); identity `white_man_clean_shaven`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0891 | 0.0890 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.7785 | 8.7683 | 0.0516 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 6.5109 | 6.5115 | 1.0001 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.2093 | 4.2086 | 0.9999 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 2.3010 | 2.3004 | 0.9998 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 8.5686 | 8.5729 | 1.0005 | report |  |
| Mouth-box >6 Hz temporal power | 1310.3 | 1311.4 | 1.0008 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 3.0 | 0.0420 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 2.0 | 0.0482 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 5.0 | 0.0008 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.1328 | 1.1333 | 0.0005 | <= A (1.1328) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.6628 | 1.6668 | 0.0040 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0564 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1860 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0564 / 0.1860 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 70.80 / face 63.18 | mouth 57.22; worst face 59.60 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 1.0000 / face 0.9997 | mouth 0.9992; worst mouth 0.9988 | report |  |
| Mouth sharpness (Laplacian var) | 22.9902 | 23.0131 | 1.0010 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 53.1347 | 53.1339 | 0.0045 | report |  |
