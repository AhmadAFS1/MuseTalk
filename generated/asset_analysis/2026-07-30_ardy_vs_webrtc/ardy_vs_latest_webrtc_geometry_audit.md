# ARDY versus latest MuseTalk WebRTC geometry audit

Date: 2026-07-30

## Conclusion

The ARDY canonical switch geometry is not the primary cause of the visible
inconsistency. The mounted certified ARDY delivery has exact shared 15-frame
opening and ending handles across all ten assets. Its face, body, camera, and
background are therefore pixel-identical at every ordered switch.

The larger inconsistency appears after motion transfer:

1. `speaking_direct_v2` is the main outlier. Its Indian-tutor derivative has the
   largest facial-shape drift, and the latest ARDY specification also contains
   much faster head motion than the other MVP poses.
2. `active_listening_empathetic_v1` has a smaller but still visible eyebrow and
   face-shape difference.
3. `light_smile` is the closest geometric match to neutral.
4. Body framing at the beginning of the delivered Indian-tutor clips differs by
   only roughly 1-4 pixels. Body position is not the dominant problem.

## Provenance gap

The latest WebRTC showcase uses the four-pose Segmind run documented in:

`generated/downloads/indian_tutor_essential_six_v1/manifest.json`

That manifest names this ARDY source:

`/workspace/ardy/outputs/pose_delivery_mvp_four_v2`

and records delivery-manifest SHA-256:

`e9a36be379cebc35d36b373d572f4f845d40e4b76e04767df35dd434d863c00c`

That exact directory and manifest are not in the mounted ARDY checkout. The
checkout instead contains the July 27 ten-pose delivery:

`outputs/pose_delivery_cleaneyes_extra_slow_blink40_v1`

whose manifest SHA-256 is:

`aceac2cb7e45aed53851d6a01ab3cc314bfb1d107e98fb72c2cdfb366fcda765`

The July 29 ARDY commit updates the four relevant specifications and renderer
policy but does not commit newly rendered videos. Therefore, the exact source
videos submitted to the latest Segmind run cannot be independently compared
frame-for-frame from this checkout.

## Independent ARDY boundary validation

All ten MP4s in the mounted certified delivery were decoded again:

- assets: 10
- ordered non-self switches: 90
- unique decoded opening handles: 1
- unique decoded ending handles: 1
- opening handle equals ending handle: yes
- every ordered last-to-first switch is exact: yes
- shared decoded 15-frame handle SHA-256 using OpenCV BGR bytes:
  `0cfd98f74e90d549cf8045dfb5dd5249e6523e85c5106cdbdf9022bb0bb5a4c5`

The latest four specifications also all use:

- the same `authored_neutral` base;
- the same camera position, look-at point, field of view, and resolution;
- the same 0.5-second / 15-frame boundary;
- zero root translation drift;
- exact matching opening and ending motion blocks.

This rules out a different ARDY camera or canonical body position as the source
of the switch mismatch.

## Latest ARDY specification motion

The July 29 specifications were composed in memory without rendering. This
measures their intended skeleton motion directly.

| Pose | Maximum head excursion | Maximum head step | Maximum head displacement | First keyframe after handle |
| --- | ---: | ---: | ---: | ---: |
| neutral | 0.897° | 0.052°/frame | 0.688 cm | ambient only |
| listener + empathy | 5.870° | 0.420°/frame | 1.459 cm | 0.35 s |
| speaking direct v2 | 6.860° | **1.918°/frame** | 1.800 cm | **0.18 s** |
| light smile | 1.953° | 0.130°/frame | 0.476 cm | 0.45 s |

`speaking_direct_v2` is the clear temporal outlier:

- its maximum head step is 4.6 times the listener/empathy value;
- its maximum shoulder step is about 0.415°/frame versus 0.133°/frame for
  listener/empathy;
- at 20 fps, decimation of a 30 fps motion can make the largest head beat appear
  as roughly a 3° displayed-frame change.

This does not break the canonical handle, but it can make the speaking pose feel
like a geometry jump shortly after switching into it.

## Indian-tutor showcase input geometry

The physical pose videos consumed by MuseTalk were analyzed with the same
68-point DWPose face model used by the MuseTalk preprocessing stack. Similarity
alignment removed global face translation, rotation, and scale before comparing
feature geometry.

### Facial shape versus neutral first frame

| Pose | Whole face | Brows | Eyes | Nose | Mouth |
| --- | ---: | ---: | ---: | ---: | ---: |
| listener / empathy | 2.23 px | 3.40 px | 1.50 px | 1.58 px | 2.22 px |
| speaking direct v2 | **2.79 px** | **4.36 px** | 1.47 px | **2.33 px** | **2.66 px** |
| light smile | 1.80 px | 2.32 px | 1.48 px | 1.32 px | 1.94 px |

The eye shape itself is comparatively stable. Most of the visible upper-face
difference comes from eyebrow position plus small changes in gaze/opening. The
nose and mouth differences in speaking direct are also above the normal
within-handle landmark variation.

### Body/framing offset versus neutral first frame

| Pose | Nose center delta | Shoulder center delta | Shoulder-width delta |
| --- | --- | --- | ---: |
| listener / empathy | +4.17, -2.08 px | +1.04, -2.08 px | -2.06 px |
| speaking direct v2 | +2.08, -4.17 px | +1.04, -2.08 px | -2.06 px |
| light smile | 0.00, -2.08 px | 0.00, -1.04 px | -4.16 px |

These body offsets are small relative to the 720×1280 frame. The perceived
inconsistency is primarily facial identity/perspective drift, not a large torso
or camera displacement.

`active_listening` and `empathetic_head_tilt` in the current MuseTalk pose set
are the exact same MP4, so those two logical IDs cannot differ geometrically.

## Recorded WebRTC evidence

In the recommended two-frame WebRTC capture, transitions into speaking direct
carry the largest observed facial movement:

- smile to speaking: 6.90 px aligned face change and 28.03 px nose travel over
  the measured 400 ms transition/motion window;
- listener to speaking: 3.40 px aligned face change and 8.59 px nose travel;
- all other measured silent cuts were 2.81 px or lower.

The 400 ms window includes both the switch and intended incoming motion, so it
is not a pure boundary hash. It nevertheless confirms that speaking direct is
the visible outlier in the real WebRTC result.

## Root-cause assessment

| Potential cause | Finding |
| --- | --- |
| Different ARDY canonical face/body frame | Not present in certified delivery |
| Different ARDY camera or root position | Not present in latest specifications |
| ARDY interior pose amplitude | Speaking v2 is substantially faster/stronger |
| Segmind identity/feature regeneration | Present, strongest in speaking v2 |
| Large derivative torso/framing mismatch | Not present; only small offsets |
| MuseTalk crossfade | Masks the mismatch but does not create the source drift |

## Recommended corrections

1. Restore or copy the exact `pose_delivery_mvp_four_v2` source batch so its
   recorded hashes can be verified against the MuseTalk provenance manifest.
2. Render the four latest ARDY poses as four separate MP4s in one certified
   library build. This preserves separate files while guaranteeing a common
   canonical handle.
3. Reduce the sharpest `speaking_direct_v2` head beats so maximum head velocity
   is closer to the listener/empathy range, ideally no more than about
   0.5° per 30 fps source frame.
4. After motion transfer, validate the generated Indian-tutor files rather than
   trusting source certification:
   - exact decoded boundary hashes;
   - aligned face RMSE by region;
   - nose and shoulder center offsets;
   - motion velocity after conversion to the 20 fps WebRTC cadence.
5. Keep the two-frame crossfade as a short safety layer until the generated
   Indian-tutor handles themselves pass.
