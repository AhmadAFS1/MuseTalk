### white_man_clean_shaven__synthetic_faces_noise1lsb_raw — FAIL (profile e1, frames raw, 240 frames)

A = `A_refined_raw` (refined), B = `B_faces_noise1lsb` (refined); identity `white_man_clean_shaven`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0891 | 0.0891 | 0.9999 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.7785 | 8.7771 | 0.0667 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 6.5105 | 6.5642 | 1.0082 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.2080 | 4.2208 | 1.0030 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 2.3010 | 2.3326 | 1.0137 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 8.5678 | 8.6857 | 1.0138 | report |  |
| Mouth-box >6 Hz temporal power | 1310.3 | 1316.6 | 1.0047 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 3.6 / 5.0 | 0.1996 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 2.0 / 2.0 | 0.2299 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 8.0 | 0.0011 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.1328 | 1.1384 | 0.0056 | <= A (1.1328) + 0.05 | PASS |
| Chin positive excess p95 (px) | 1.6628 | 1.6614 | -0.0014 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0723 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.2272 | <= 0.15 | FAIL |
| PSNR A vs B, dB mean frame (cap 100) | - | full 64.88 / face 57.21 | mouth 51.67; worst face 55.35 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9998 / face 0.9989 | mouth 0.9967; worst mouth 0.9960 | report |  |
| Mouth sharpness (Laplacian var) | 22.9840 | 24.4838 | 1.0653 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 53.1347 | 53.1340 | 0.0008 | report |  |
