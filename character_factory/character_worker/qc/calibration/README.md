# QC calibration record (strict profile)

Generated 2026-10-25T00:10:30Z by `qc/calibrate.py` on the RTX 4070 SUPER box, face backend = Haar cascade (no MediaPipe venv here).

| Clip | Pose | Result | Hard failures | Warnings | median frame diff | mouth motion ratio |
| --- | --- | --- | --- | --- | ---: | ---: |
| idle/source.mp4 | idle | pass | - | - | 1.1878 | 0.9376 |
| smiling/source.mp4 | smiling | pass | - | - | 0.8017 | 1.0052 |
| black_man_short_beard/source.mp4 | talking | pass | - | - | 1.5066 | 1.3347 |
| black_woman/source.mp4 | talking | pass | - | - | 2.0746 | 1.2043 |
| east_asian_man_goatee/source.mp4 | talking | pass | - | - | 1.8056 | 1.2646 |
| middle_eastern_man_full_beard/source.mp4 | talking | pass | - | - | 2.2257 | 1.3088 |
| south_asian_woman/source.mp4 | talking | pass | - | - | 1.815 | 1.2909 |
| white_man_clean_shaven/source.mp4 | talking | pass | - | - | 1.5288 | 1.232 |
| accepted-japanese | talking | pass | - | - | 1.6314 | 1.1994 |
| quiet-japanese | talking | pass | - | - | 1.1172 | 1.0698 |
| short-japanese | talking | FAIL | anchor.first_frame_ssim | - | 1.5125 | 1.3807 |
| accepted-latina | talking | pass | - | - | 1.3824 | 1.1716 |
| quiet-latina | talking | pass | - | - | 1.0881 | 1.0816 |
| short-latina | talking | FAIL | anchor.first_frame_ssim | - | 1.3167 | 1.5617 |
| chinese-bob-idle | idle | pass | - | - | 1.0885 | 0.8638 |
| chinese-bob-talking | talking | pass | - | - | 1.6338 | 1.2733 |
| chinese-bob-smiling | smiling | pass | - | - | 0.8475 | 0.9889 |
| musetalk-japanese | talking | pass | - | - | 1.5897 | 1.0994 |
| musetalk-latina | talking | pass | - | - | 1.3916 | 1.3093 |

The two `short-*` clips fail `anchor.first_frame_ssim` by design: they are the historical `certify` deliveries whose frame 0 is raw H3 output, not the shared anchor that the three-pose pipeline places at frame 0 (SSIM 0.993 on every three-pose clip).

Smallest margins (negative = a reference violates the threshold):

- `talking:anchor.first_frame_ssim`: margin -0.0309 (value 0.9391, threshold 0.97, short-latina)
- `idle:contract.frames`: margin 0 (value 240, threshold 240, idle/source.mp4)
- `idle:audio.presence`: margin 0 (value 0, threshold 0, idle/source.mp4)
- `smiling:contract.frames`: margin 0 (value 158, threshold 158, smiling/source.mp4)
- `smiling:audio.presence`: margin 0 (value 0, threshold 0, smiling/source.mp4)
- `talking:contract.frames`: margin 0 (value 240, threshold 240, black_man_short_beard/source.mp4)
- `talking:audio.presence`: margin 0 (value 1, threshold 1, black_man_short_beard/source.mp4)
- `talking:audio.clip_fraction`: margin 0.001 (value 0.0, threshold 0.001, black_man_short_beard/source.mp4)
- `idle:face.detection_fraction`: margin 0.015 (value 1.0, threshold 0.985, idle/source.mp4)
- `smiling:face.detection_fraction`: margin 0.015 (value 1.0, threshold 0.985, smiling/source.mp4)
- `talking:face.detection_fraction`: margin 0.015 (value 1.0, threshold 0.985, black_man_short_beard/source.mp4)
- `idle:anchor.first_frame_ssim`: margin 0.0227 (value 0.9927, threshold 0.97, chinese-bob-idle)
