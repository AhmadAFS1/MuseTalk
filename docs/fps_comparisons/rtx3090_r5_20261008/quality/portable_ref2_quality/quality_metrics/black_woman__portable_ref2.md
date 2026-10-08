### black_woman__portable_ref2 — FAIL (profile e1, frames raw, 240 frames)

A = `A` (refined), B = `portable_ref2` (refined); identity `black_woman`; pre-encode/decoded frames SHA-identical: **False**

| Metric | A | B | Delta / ratio | Threshold | Result |
|---|---:|---:|---:|---|---|
| Lip aperture mean (eye spans); delta = Pearson A vs B | 0.0872 | 0.0873 | 0.9998 | >= 0.97 | PASS |
| Aperture xcorr lag (frames) | - | - | 0 | == 0 | PASS |
| Aperture mean abs delta (px) | 9.7519 | 9.7626 | 0.1014 | <= 0.5 | PASS |
| Flicker mouth mean abs(dt) RGB | 7.3183 | 7.3265 | 1.0011 | <= 1.05 | PASS |
| Flicker jaw mean abs(dt) RGB | 4.1248 | 4.1267 | 1.0005 | <= 1.05 | PASS |
| Flicker ring mean abs(dt) RGB | 3.9752 | 3.9751 | 1.0000 | <= 1.05 | PASS |
| Flicker mouth 2nd-diff | 9.5297 | 9.5438 | 1.0015 | report |  |
| Mouth-box >6 Hz temporal power | 1714.2 | 1716.3 | 1.0012 | <= 1.05 (report) | REPORT (ok) |
| Seam band abs(A-B): per-frame max p99 / max; delta = mean | - | 4.6 / 6.0 | 0.0752 | report |  |
| Seam ring abs(A-B): per-frame max p99 / max; delta = mean | - | 3.0 / 4.0 | 0.0919 | report |  |
| Outside-mask abs(A-B): max; delta = mean | - | 8.0 | 0.0012 | report |  |
| Protected-lip max RGB change vs own standard | 0 | 0 | n/a | == 0 vs own standard compose | PASS |
| Chin-target abs error mean (px) | 1.6228 | 1.6166 | -0.0062 | <= A (1.6228) + 0.05 | PASS |
| Chin positive excess p95 (px) | 0.2968 | 0.2977 | 0.0009 | report |  |
| Jaw+lip landmark dev A vs B mean (px) | - | - | 0.0866 | <= 0.05 | FAIL |
| Jaw+lip landmark dev A vs B p99 (px) | - | - | 0.2593 | <= 0.15 | FAIL |
| Landmark dev vs calibrated proposal (mean/p99) | - | - | 0.0866 / 0.2593 | mean <= 0.1 and p99 <= 0.35 (proposed, report) | REPORT (ok) |
| PSNR A vs B, dB mean frame (cap 100) | - | full 66.79 / face 59.59 | mouth 52.76; worst face 54.46 | report-only | REPORT (ok) |
| SSIM A vs B mean | - | full 0.9999 / face 0.9996 | mouth 0.9984; worst mouth 0.9937 | report |  |
| Mouth sharpness (Laplacian var) | 21.3644 | 21.3501 | 0.9993 | >= 0.95 | PASS |
| Face Lab L* mean; delta = dE76 of means | 42.2931 | 42.2931 | 0.0054 | report |  |
