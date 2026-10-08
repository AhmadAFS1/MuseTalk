# Canonical contact-sample extraction (CPU only)

`quality_contact_sheets.py` creates a plan, or extracts paired review PNGs from
existing canonical quality captures. It does not change inference, compute new
quality metrics, inspect images automatically, or approve a release. Full frames
decode the existing lossy review MP4s: these are **not** the pre-encode raw frames
used by the e1 quality gates. Native 512×896 full-frame pixels are preserved; crops
are cut and placed side by side without resizing. The exact 256×256 generated-face
NPZ pixel hashes are separately checked against capture metadata.

```bash
python3 scripts/repro_3090/quality_contact_sheets.py \
  --run /actual/copied/portable_ref1_quality/report.json \
  --run /actual/copied/portable_ref2_quality/report.json \
  --run /actual/copied/native_quality/report.json \
  --inputs /actual/copied/quality-inputs-v1.json \
  --fixture-root /actual/copied/review_fixtures \
  --out /actual/new/contact_review --extract
```

Supply one to three actual quality runs. Omit `--extract` for a plan only. Outputs
are exclusive-create: choose a fresh directory for every run. Copying an entire
capture directory is necessary because raw face/landmark arrays, videos, reports,
and backend identities are checked. The input manifest must be the exact one
referenced by those runs. A local copy works: only selected review-fixture files
are freshly checked, not absent remote models/calibration. The fixture root must
contain one directory per canonical identity, each with these exact pinned files:

- `source.mp4`
- `refined_raw.mp4`
- `speech.wav`
- `render.json`

Their SHA-256 **and byte counts** must match the input manifest. The accepted A raw
frame hash must also match `render.json`. Missing files produce `MISSING_SOURCES`
with exact paths; they are not replaced by source-only or another avatar's pixels.

Every avatar gets common-frame columns: accepted A, then supplied runs in CLI
order. Selection is the union of frame 0, frame 120, frame 239, each arm's peak
reported aperture, each run's largest reported seam-band/ring mean pixel error,
and largest absolute chin error within the original [24,216) gate window. Ties use
the first frame. These are actual saved per-frame series; the tool does not claim
the seam selection is the worst full-frame PSNR/SSIM/landmark error, whose full
series the canonical reports do not retain.

Mouth and jaw/chin crops use the union of the canonical metric report's
`temporal_hf_power_gt_6hz` mouth/jaw rectangles across supplied runs. The identical
crop coordinates are used for every column. They are review rectangles, not a new
face detector or changed blending mask. `ffprobe` verifies each video has 240
native-resolution frames at 24fps, aligned to frame zero. Both exact rational
frame times and observed video timestamps are recorded, along with decoded frame,
source, video, metric, capture, engine and output hashes.

`index.html` shows the full-frame and crop samples with column labels, reasons,
frame indices and original speech playback. PNGs retain their exact native pixel
dimensions even if the browser scales their display. `review.json` is the full
evidence index. FFmpeg/FFprobe must already be installed; no package installation
or GPU work is performed. CPU tests cover indices, tie-breaking, strict hashes,
missing sources, crops, PNG bytes and video timing/shape constraints.

## Silence requires separate audio evidence

No silence labels are inferred from aperture or invented timestamps. Without
audio-backed reviewed intervals, each avatar retains `MISSING_AUDIO_REVIEW`.
Optional `--silence-evidence FILE` accepts this schema (example shape only; do not
use invented values as actual annotations):

```json
{
  "schema": "canonical_audio_silence_v1",
  "fps": 24,
  "avatars": {
    "EXACT_CANONICAL_IDENTITY": {
      "audio_sha256": "ACTUAL_SHA256_OF_REVIEWED_SPEECH_WAV",
      "review_method": "reviewed_audio_intervals",
      "reviewer": "Actual reviewer identity",
      "evidence_note": "How the actual audio interval was reviewed",
      "intervals": [{"start_frame": 0, "end_frame_exclusive": 1}]
    }
  }
}
```

Intervals must be within [0,240), with one to eight nonempty intervals per supplied
avatar. Their midpoint frame is selected; the audio hash must match the frozen
fixture. Unannotated avatars remain explicitly missing. This is recorded review
evidence, not automatic voice-activity detection.

`PLAN_READY` and `EXTRACTED_REVIEW_ONLY` mean only that evidence processing worked.
Every result retains `visual_review_status=NOT_PERFORMED` and `release_ready=false`.
Sparse sheets cannot prove temporal quality, lip sync, full-clip acceptance, or
production 16×3 pose compatibility. Any actual visual/audio acceptance must be
recorded separately by a reviewer who inspected the resulting pixels and audio.
