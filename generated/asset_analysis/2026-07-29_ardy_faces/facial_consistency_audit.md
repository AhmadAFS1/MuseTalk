# ARDY facial consistency audit

Date: 2026-07-29

## Scope

The certified ARDY source directory referenced by the delivery manifest,
`/workspace/ardy/outputs/pose_delivery_mvp_four_v2`, is not mounted on this
MuseTalk instance. Therefore, the original ARDY videos could not be
independently decoded here.

This audit covers all six logical assets delivered to and consumed by MuseTalk:

- `neutral_resting`
- `active_listening`
- `empathetic_head_tilt`
- `speaking_direct`
- `nod_agree`
- `light_smile`

The delivery manifest reports that the original ARDY references passed with a
shared 15-frame decoded boundary handle. The Segmind motion-transfer derivatives
do not preserve that equality.

## Conclusion

The delivered MuseTalk assets show the same person and retain broadly consistent
facial identity, but their facial features are **not mathematically identical**.

- `active_listening` and `empathetic_head_tilt` are aliases of the exact same
  MP4. They match byte-for-byte and frame-for-frame.
- The remaining five physical videos have zero exact first-frame matches among
  their 10 unordered pairs.
- They have zero exact decoded final-to-first matches among all 20 directed
  switches.
- The eyebrows have the largest systematic geometry difference.
- Eye geometry is comparatively stable, although eye openness and gaze vary as
  part of the motion.
- Nose and mouth geometry have smaller but measurable pose-dependent drift.

## Asset identity

| Logical asset | Frames | SHA-256 prefix | Notes |
| --- | ---: | --- | --- |
| neutral | 300 | `f0853c17dda96ccf` | New four-pose derivative |
| listener | 240 | `21c77b017e196169` | New merged listener/empathy derivative |
| empathy | 240 | `21c77b017e196169` | Exact alias of listener |
| speaking | 300 | `d2a57eef4c39b4eb` | New speaking-direct-v2 derivative |
| nod | 88 | `e87a06220e754659` | Retained compatibility derivative |
| smile | 120 | `59597dd4bfe36824` | New four-pose derivative |

The 88-frame `nod_agree` here is not the certified 84-frame ARDY production nod
described by the prior ARDY audit. It cannot be used to independently validate
that source asset.

## Facial landmark method

The first and last 15 frames plus 2-fps samples were analyzed with MuseTalk's
68-point DWPose face model. Detector confidence was approximately `0.970`.
Similarity alignment removed global head translation, rotation, and scale before
regional geometry was compared.

Normal motion/detector variation across each asset's first 15 frames had a median
whole-face aligned RMSE of `1.21` to `1.36` pixels.

## First-frame geometry versus neutral

| Asset | Whole face | Jaw | Brows | Eyes | Nose | Mouth |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| listener / empathy | 2.23 px | 2.10 | 3.40 | 1.50 | 1.58 | 2.22 |
| speaking | 2.79 px | 2.67 | 4.36 | 1.47 | 2.33 | 2.66 |
| nod derivative | 2.50 px | 2.27 | 3.45 | 1.33 | 1.51 | 2.96 |
| smile | 1.80 px | 1.68 | 2.32 | 1.48 | 1.32 | 1.94 |

The eye residuals are close to the within-handle baseline, so there is no strong
evidence that the intrinsic eye shape changed substantially. The eyebrows exceed
that baseline most clearly. Nose and mouth differences are real but subtler and
partly expression-related.

The detected outer-eye distance varies from `191.67` to `193.79` pixels across
the first frames, approximately a 1.1% range. Normalized facial proportions also
vary slightly:

- nose width: `0.3369` to `0.3480` of outer-eye distance;
- nose height: `0.5160` to `0.5377`;
- mouth width: `0.5979` to `0.6304`;
- face width: `1.5052` to `1.5325`;
- face height: `1.3223` to `1.3587`.

## Transition implications

The largest aligned directed boundary differences are:

1. neutral to speaking: `2.77` px whole face, `4.25` px brows;
2. neutral to nod derivative: `2.59` px whole face, `3.48` px brows;
3. speaking to neutral: `2.54` px whole face, `3.80` px brows;
4. smile to speaking: `2.29` px whole face, `3.30` px brows.

These are subtle identity/feature shifts rather than a different person. They
are large enough to explain why a mathematically smooth temporal blend can still
feel slightly synthetic around the eyebrow, eye-position, and nose areas.

## Visual references

- `first_frames.jpg`: first decoded frame of each physical asset
- `last_frames.jpg`: last decoded frame of each physical asset

## Recommended source-level validation

Mount or copy the certified ARDY delivery into this instance and repeat:

1. verify the expected 84-frame production nod;
2. hash every decoded frame in the first and last 15-frame handles;
3. run the same aligned 68-point landmark audit on those exact ARDY frames;
4. compare ARDY landmarks with the Segmind outputs to measure how much drift the
   motion-transfer stage introduces.

If the ARDY handles are exact but the derivatives differ, the corrective action
belongs in the avatar generation/normalization stage rather than the ARDY pose
library.
