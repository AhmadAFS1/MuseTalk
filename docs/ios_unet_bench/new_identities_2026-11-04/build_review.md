# New-identity validation tooling review

The new protocol adds 30 newly generated identities and 60 held-out clips without changing the original
six-avatar review's default behavior or thresholds. Existing student/teacher weights are frozen; no training
or checkpoint selection occurs.

- Preparation calls the unchanged production `APIAvatar` constructor with v15/jaw parsing and SD-VAE in an
  isolated working directory. Native portraits are preserved, resized without cropping and padded to 512×896.
  Accepted portrait hashes, normalized images, cache files, configuration, models and seeds are recorded.
- `new-static-r5` requires the exact frozen 30-identity/two-audio manifest, complete 250-frame clips at 25 fps,
  finite metrics and full FaceMesh coverage. N, S and S16 receive independent verdicts against A. Old D metrics
  are not loaded or reused. Static-pose and CUDA-versus-ANE limitations are explicit.
- The detached box job verifies the immutable code and model hashes, resumes matching inputs only, reserves
  time for packaging/pull/cleanup, and requires 60 A|S comparisons plus 60 original S16 videos. It contains no
  training or rental creation. Existing EC2 controller/drive rental guards remain unchanged.
- Cross-review fixed completed-render reuse after cache changes: the resume record now also binds the
  prepared manifest and model dependencies. A second finding fixed accepted portrait hashes being declared
  but not compared with source bytes before preparation.

Validation passed: 22 tiny tests covering preparation, real reviewer-manifest integration, strict resume,
input changes, corrupted caches, normalization, source-hash mismatch, exact clip sets, incomplete/finite
metrics, independent arm verdicts and legacy D behavior. Shell/Python syntax, metadata-only job/transport
dry-runs and offline relay quoting/guard/archive mocks passed. No full model was allocated on the Mac.

All 30 generated sources passed a lightweight local input check: exactly one 468-landmark face after
normalization. This is not a lipsync quality pass. GPU preparation and 60-clip inference subsequently completed on the one
additional validation-only rental explicitly approved by the user; the final results are in README.md.

Before any inference, slow checkout/package transport required setup recovery on the same rental. The
delivery-encoding allowance was reduced from 15 to five minutes; the 20-minute pull/cleanup reserve and
original four-hour expiry remain fixed. The exact two-thread CRF26 command encoded one existing 250-frame
1024×1298 comparison in 2.41 seconds on the Mac (about 144 seconds for 60 sequential encodes). This supports
the allowance but does not guarantee worker CPU timing. The earlier frozen request is preserved, and the
wrapper, checkout revision and replacement request must change together. Model weights, portraits, audio,
preparation/scoring code and thresholds remain unchanged; no model inference preceded this adjustment.

Final artifact verification passed: all 30 identities, 60 A | S comparisons, 60 unchanged original S16 files,
60 unchanged per-clip metrics and 60 curve JSONs have verified coverage/provenance. All 180 aperture
aggregate recomputations exactly matched the original scores. Public media uses 768px H.264 CRF30 with
32 kbps AAC presentation audio; original test audio and raw-frame metrics are unchanged. The combined
old/new report package is approximately 48.9 MB, below 50 MB. Six representative original stills received
independent visual inspection; this is not an all-video temporal review. See visual_review.md.
