### white_man_clean_shaven__refined_vs_standard_video — INCOMPLETE (profile e1, frames video, 240 frames)

A = `A_refined` (refined), B = `B_standard` (standard); identity `white_man_clean_shaven`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0883 | 0.0871 | 0.9937 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 8.6922 | 8.5786 | 0.4857 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 6.4397 | 6.4472 | 1.0012 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 3.9237 | 3.2406 | 0.8259 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 2.3114 | 2.3756 | 1.0278 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 8.5366 | 8.5364 | 1.0000 | report |  |
| Mouth-box >6 Hz temporal power | 1274.7 | 1256.9 | 0.9860 | <= 1.05 (report) | REPORT |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 66.0 / 69.0 | 2.9843 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 63.0 / 68.0 | 3.6852 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 45.0 | 1.0416 | report |  |
| Protected-lip max RGB change vs own standard | n/a | n/a | n/a | == 0 vs own standard compose | NOT-RUN |
| Chin-target abs error mean (px) | 1.1638 | 0.6811 | -0.4827 | <= A (1.1638) + 0.05 | REPORT |
| Chin positive excess p95 (px) | 1.8524 | 0.9147 | -0.9377 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.6556 | <= 0.05 | REPORT |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 2.4256 | <= 0.15 | REPORT |
| PSNR A vs B, dB mean frame (cap 100) | - | full 41.43 / face 36.35 | mouth 38.68; worst face 32.50 | report-only | REPORT |
| SSIM A vs B mean | - | full 0.9798 / face 0.9690 | mouth 0.9796; worst mouth 0.9576 | report |  |
| Mouth sharpness (Laplacian var) | 31.2530 | 31.0857 | 0.9946 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 52.4477 | 52.2220 | 0.2318 | report |  |
