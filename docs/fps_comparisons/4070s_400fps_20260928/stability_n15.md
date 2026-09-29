# Stability: T_srcg50_n15_stability

15 concurrent streams over 6 avatars, 10 timed windows, backend stagewise16_taesdtrt, pack 16.

| Window | Wall s | Aggregate fps | Stream fps min / median / max | SM MHz | Power W | GPU °C | GPU mem MiB | MemAvailable min GB | Worker RSS max MiB | FaceMesh RSS max MiB |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 62.4 | 403.9 | 26.93 / 26.93 / 26.93 | 2475.0 | 206.36 | 64.0 | 3998.0 | 7.63 | 693 | 278 |
| 1 | 63.0 | 399.8 | 26.65 / 26.65 / 26.65 | 2460.0 | 203.62 | 67.0 | 3998.0 | 7.06 | 693 | 314 |
| 2 | 63.0 | 400.3 | 26.69 / 26.69 / 26.69 | 2460.0 | 204.865 | 67.0 | 3998.0 | 6.78 | 693 | 331 |
| 3 | 63.0 | 399.9 | 26.66 / 26.66 / 26.66 | 2460.0 | 204.005 | 67.0 | 3998.0 | 6.52 | 693 | 336 |
| 4 | 63.0 | 400.0 | 26.66 / 26.66 / 26.66 | 2460.0 | 204.15 | 67.0 | 3998.0 | 6.41 | 693 | 364 |
| 5 | 63.0 | 399.7 | 26.65 / 26.65 / 26.65 | 2460.0 | 204.035 | 67.0 | 3998.0 | 6.14 | 693 | 376 |
| 6 | 63.1 | 399.6 | 26.64 / 26.64 / 26.64 | 2460.0 | 203.64999999999998 | 66.0 | 3998.0 | 6.09 | 693 | 386 |
| 7 | 63.0 | 399.7 | 26.65 / 26.65 / 26.65 | 2460.0 | 203.74 | 67.0 | 3998.0 | 5.96 | 693 | 396 |
| 8 | 63.0 | 400.0 | 26.67 / 26.67 / 26.67 | 2460.0 | 204.63 | 67.0 | 3998.0 | 5.82 | 693 | 385 |
| 9 | 63.1 | 399.6 | 26.64 / 26.64 / 26.64 | 2460.0 | 203.695 | 67.0 | 3998.0 | 5.79 | 693 | 410 |

## Across windows

- Aggregate fps: first 403.9, last 399.6, min 399.6, median 399.8, max 403.9.
- Per-stream fps measured clip by clip (240 frames between consecutive clip completions, 900 clips): min 26.56, p1 26.59, p5 26.61, median 26.67, max 27.10.
- Slowest clip of any stream: **26.56 fps** against a 20 fps real-time target (every stream stayed above it); aggregate needed for 15 x 20 fps = 300.
- Fairness: all streams finish each window within 0.28-0.29 s of each other (the GPU issuer serves every stream's frames in one shared bs16 queue).
- Per-stream fps variation across windows (coefficient of variation): max 0.31%, median 0.31%.
- Worker private memory (RssAnon) change, first to last window: +0.0 to +5.9 MiB; FaceMesh helper: +75.8 to +168.9 MiB.
- MemAvailable minimum per window: 5.79-7.63 GB.
- Bit-exact determinism (every clip of an avatar hashes identically across streams and loops): True.
- All windows timed >= min: True; GPU busy fraction median 0.9990.
- Run status: complete; shutdown note: RuntimeError('worker 5 exited (exitcode None)').
