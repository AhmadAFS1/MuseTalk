# Thirty new identities: frozen-model generalization review

Create 30 new fictional people, then compare the existing full MuseTalk model with the already trained
round-2 bk_tiny student on **60 held-out clips**. This measures behavior on unseen faces. It does not train a
new model or overturn the previous result: round 2 failed r5 on all 12 earlier review clips.

## Identities and source images

[characters.json](characters.json) defines the stable IDs **ni30_01–ni30_30** and a complete, standalone
generation prompt for each. Generate a new image for every ID without reference images, previous avatars or
celebrity references. Do not reuse one face with a different hairstyle, color or outfit.

The set contains 14 feminine, 14 masculine and two androgynous presentations, ages 21–77. Ten descriptive
skin-tone groups each contain three people, spanning very fair to very deep brown. These are design
descriptions, not inferred ethnicities. Face width, cheekbones, jaw and chin, nose shape, lip proportions,
eyebrows, hair texture, age lines, glasses and short facial hair vary across the set. Three people wear
clear-lens glasses; three have short facial hair with fully visible lip contours.

Personality concepts are fictional writing choices, independent of skin tone or facial anatomy. Everyday
wardrobe and a slight expression in the eyes convey each concept. They are not personality judgments made
from an image. All portraits use the same capture conditions:

- One adult, square PNG, frontal head and shoulders, level camera, direct eye contact. The generator was
  asked for 1024 × 1024 and returned 1254 × 1254; native source files are preserved. Production preparation
  resizes with aspect ratio preserved and pads to 512 × 896 before caches and evaluation hashes are frozen.
- The whole head, both cheeks, jaw, chin and shoulders remain visible.
- Naturally closed, relaxed lips, no teeth, no broad smile or pursing; mouth and eyes unobscured.
- Soft even studio lighting, natural skin texture, plain pale warm-gray background, no text or logos.

Store portraits at `generated/new_identities_2026-11-04/portraits/{id}.png`. ni30_01 is reserved for the
35-year-old woman with deep brown skin, broad square jaw, high cheekbones, broad rounded nose, full lips and
short natural tightly coiled hair, wearing a teal cardigan. Its fictional concept is a calm, patient biology
teacher. Each other identity has its own independently specified face and concept.

Before model inference, inspect the portraits for duplicate-looking identities, malformed anatomy, pose,
resolution, framing and lip visibility. Regenerate only to correct those source defects, keeping the same ID
and recorded prompt. Freeze the accepted source-image SHA256 hashes before reviewing model output. Do not
replace a source because its student result looks worse. Record any unresolved source failure under its ID;
the required denominator remains 30 identities and 60 clips.

## Static-portrait scope

These are **static portraits**, not newly generated videos with natural head motion. Avatar preparation may
derive the input cache from repeated frames of the same portrait. Each prepared cache, frame sequence,
mask, coordinate set and latent input must then be shared unchanged by every inference arm.

This setup tests unseen facial geometry, appearance and audio-driven lip synthesis with a fixed source
pose. It does not establish quality under moving heads, changing expressions, pose transitions or a complete
animated-source character-creation workflow. Report the actual source/preparation method and cache hashes.
Do not describe still-image repetition as new motion footage or claim motion coverage from it.

## Models and separation from training

Freeze these files before generation or inference:

| Model | SHA256 | Role |
|---|---|---|
| Full MuseTalk teacher `unet.pth` | `7ebf6c98c181e20838e4c0054e96e944ac60d5d692cc01db42839fe11b787007` | Reference and fp16 control |
| `round2_bk_tiny_best.pt` | `a05d2d98f69b431252458d3b25f72e395d557ba693821baee91d11f5fe5a96cd` | Frozen student, selected at step 35,750 by the earlier validation MAE |

The existing training source revision is `d8826442c5d1bd8772185a7fc564eddd98faf19f`; record the actual review
runner revision separately. **All 30 identities and both review audios remain test-only.** Do not train on
their portraits, prepared latents, generated outputs or audio, and do not select or revise a checkpoint from
these results. No further distillation is part of this task.

## Sixty-clip protocol

For every ID, render the first 10 seconds of both existing held-out clips at **25 fps, 250 frames per clip**:

| Audio | SHA256 |
|---|---|
| `data/audio/ai-assistant.mpga` | `98638b2f39c6135098ecbab92122f1eea39a52e622a17c27b9b5f885e0ccf81c` |
| `data/audio/eng.wav` | `654dcbce843d70451d1123f7649a58bee11bb9dec9a7e835c05b1e367efb2078` |

Use clip IDs `{id}__ai-assistant` and `{id}__eng`. The exact 60 IDs must be present in the result manifest.
Every arm receives the same cached avatar inputs and Whisper windows, uses the same blending and regions,
and decodes with the same torch TAESD fp32 model. Record decoder/config hashes and runtime versions.

| Arm | UNet | Precision and purpose |
|---|---|---|
| A | Full teacher | fp32, TF32 disabled; reference |
| N | Full teacher | fp16, including positional encoding; full-model precision control |
| S | Frozen round-2 student | fp32, TF32 disabled; primary comparison with the previous distillation review |
| S16 | Frozen round-2 student | fp16, including positional encoding; separate student precision result |

This is 60 clips per arm: 240 individual renders. Produce **60 labelled A | S side-by-side videos**, with
synchronized audio, a mouth zoom, stable left/right labels and clip IDs. Retain S16 metrics and renders for
all 60; add A | S16 comparisons when presenting fp16 behavior. Individual render timings are runtime
observations, not iPhone speed measurements.

Compute metrics on raw composed frames before encoding. Save per-clip metrics, aperture series, landmark
coverage and enough hashes to trace each result to its source portrait, prepared cache, model and audio.
Sharing videos may be compressed afterwards; label that compression and preserve raw-frame metric provenance.

## Frozen scoring and interpretation

Apply the existing r5 thresholds to N, S and S16 **against A**, without changing the thresholds after seeing
outputs:

| Check | Required per clip |
|---|---:|
| Lip aperture correlation | ≥0.97 |
| Lip aperture lag | 0 frames |
| Mean absolute aperture error | ≤0.5 px |
| Mouth flicker ratio | ≤1.05 |
| Face flicker ratio | ≤1.05 |
| Mouth sharpness ratio | ≥0.95 |

Report r5 pass/fail for each arm and all 60 clips. Report aperture-frame coverage for A and each candidate;
missing or insufficient landmarks, undefined correlation, incomplete frames or an absent clip must remain
visible as unscored/incomplete, never silently omitted or counted as a pass. Any unscored required check keeps
the aggregate INCOMPLETE, with known failures listed alongside it. Complete coverage with a failed check yields
FAIL. A student arm passes this new-set r5 review only
when **all 60 clips pass every required check** with valid coverage. N must be reported alongside it so that
control failures and preparation defects can be distinguished from a student regression.

Also report latent MAE/max error, mouth PSNR/SSIM, aperture error/correlation, flicker and sharpness ranges
and worst clip IDs. Latent MAE, PSNR and SSIM are descriptive here; no new acceptance thresholds are invented
for them. Keep S and S16 verdicts separate rather than choosing whichever looks better on a given identity.

**There is no historical D baseline for these new identities.** The original six-avatar D measurements
cannot support a no-worse-than-D claim for this set. Use a dedicated new-set r5 verdict. The old
`review_distilled_unet.py` verdict expects historical D clips and would correctly be incomplete for unknown
identities; do not fabricate D entries or rename new clips to old identities to obtain a pass.

An optional Core ML/phone arm may measure the exported, quantized student with these same inputs. Record
its exact model hash, compute units, device and precision, and score its actual outputs against A. Plain
torch fp16 is not a measurement of quantized ANE quality. A comparison against today's full on-device chain
would require rendering a **new** matched baseline for all of these faces; label it as newly measured rather
than reusing the old D numbers. Optional device work does not alter the frozen r5 thresholds or S/S16 verdicts.

## Execution record

Keep generation prompts and accepted portrait hashes, preparation/cache hashes, frozen model/audio hashes,
runner revision, device/runtime details, all 60 clip statuses and the comparison videos together under this
set's record. Distinguish generation/preparation failures, control failures and student quality failures.
The final report must state the static-source limitation and preserve the previous round-2 quality rejection.

This document prepares the manifest and protocol only. Portrait generation is tracked by the parent task;
preparation, any additional GPU validation and optional phone runs are separate execution steps. No rental,
training job, device operation, commit or push is performed by preparing this plan.
