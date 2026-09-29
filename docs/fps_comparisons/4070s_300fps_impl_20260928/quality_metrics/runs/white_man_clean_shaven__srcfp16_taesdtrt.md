### white_man_clean_shaven__srcfp16_taesdtrt — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `srcfp16_taesdtrt` (refined); identity `white_man_clean_shaven`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0891 | 0.0890 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.7785 | 8.7689 | 0.0552 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 6.5108 | 6.5115 | 1.0001 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.2087 | 4.2099 | 1.0003 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 2.3011 | 2.3010 | 1.0000 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 8.5683 | 8.5722 | 1.0005 | report |  |
| Mouth-box >6 Hz temporal power | 1310.3 | 1312.3 | 1.0015 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 4.0 | 0.0417 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 2.0 | 0.0476 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 5.0 | 0.0008 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.1328 | 1.1352 | 0.0024 | <= A (1.1328) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.6628 | 1.6712 | 0.0084 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0563 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.1789 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0563 / 0.1789 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 70.82 / face 63.20 | mouth 57.19; worst face 56.27 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 1.0000 / face 0.9997 | mouth 0.9992; worst mouth 0.9979 | report |  |
| Mouth sharpness (Laplacian var) | 22.9880 | 23.0147 | 1.0012 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 53.1347 | 53.1339 | 0.0046 | report |  |
