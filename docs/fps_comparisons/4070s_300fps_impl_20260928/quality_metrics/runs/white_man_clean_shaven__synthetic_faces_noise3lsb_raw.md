### white_man_clean_shaven__synthetic_faces_noise3lsb_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_noise3lsb` (refined); identity `white_man_clean_shaven`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0891 | 0.0890 | 0.9997 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.7785 | 8.7681 | 0.1023 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 6.5094 | 6.7973 | 1.0442 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.2075 | 4.2699 | 1.0148 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 2.3007 | 2.4685 | 1.0729 | <= 1.05 | FAIL |
| Flicker mouth 2nd-diff | 8.5665 | 9.1831 | 1.0720 | report |  |
| Mouth-box >6 Hz temporal power | 1310.3 | 1345.5 | 1.0268 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 5.6 / 6.0 | 0.4757 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 3.0 | 0.5467 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 9.0 | 0.0018 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.1328 | 1.1517 | 0.0189 | <= A (1.1328) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.6628 | 1.6426 | -0.0201 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.1029 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.3056 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 59.24 / face 51.56 | mouth 45.50; worst face 50.89 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9995 / face 0.9972 | mouth 0.9910; worst mouth 0.9892 | report |  |
| Mouth sharpness (Laplacian var) | 22.9851 | 30.9490 | 1.3465 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 53.1347 | 53.1335 | 0.0018 | report |  |
