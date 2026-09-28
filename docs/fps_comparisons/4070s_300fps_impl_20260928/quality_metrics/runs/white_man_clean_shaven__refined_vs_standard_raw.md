### white_man_clean_shaven__refined_vs_standard_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined` (refined), B = `B_standard` (standard); identity `white_man_clean_shaven`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0891 | 0.0876 | 0.9949 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.7785 | 8.6269 | 0.4219 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 6.5016 | 6.4957 | 0.9991 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.1885 | 3.3828 | 0.8076 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 2.4696 | 2.4699 | 1.0001 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 8.5550 | 8.5275 | 0.9968 | report |  |
| Mouth-box >6 Hz temporal power | 1310.3 | 1285.1 | 0.9807 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 65.6 / 70.0 | 2.0110 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 60.0 / 65.0 | 2.8487 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 44.0 | 0.0205 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.1328 | 0.6324 | -0.5004 | <= A (1.1328) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.6628 | 0.9417 | -0.7210 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.6333 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 2.4607 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 45.16 / face 37.74 | mouth 43.74; worst face 32.95 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9967 / face 0.9845 | mouth 0.9941; worst mouth 0.9700 | report |  |
| Mouth sharpness (Laplacian var) | 22.9459 | 22.4896 | 0.9801 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 53.1347 | 52.9098 | 0.2305 | report |  |
