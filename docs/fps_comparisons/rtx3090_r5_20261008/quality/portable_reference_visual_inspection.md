# Portable Q1 visual reference inspection

Actual pixel inspection completed for all 117 final-v2 sheets: 39 frame pairs
across six canonical fixtures, each inspected as a full 1024×896 paired image and
native-resolution mouth/jaw crops. Left is accepted A; right is portable Q1.
Exact file names, SHA-256 hashes, dimensions, frame indices, selection reasons,
source bindings and per-avatar observations are in
`portable_reference_visual_inspection.json`.

This is reference characterization, not a native-engine approval. No obvious new
gross identity change, detached beard, torn jaw boundary or displaced tooth row
was visible in these selected Q1 pairs. Both arms retain visible mouth-region
softness and simplified teeth. The goatee/full-beard fixtures also have visibly
smoothed moustache detail relative to their sharper outer beard. These shared
artifacts are recorded, not silently called flawless.

| Canonical fixture | Inspected frame indices | Notable shared observations |
|---|---|---|
| black_man_short_beard | 0, 15, 120, 125, 183, 239 | Soft mouth edges; simplified bright tooth patch at 120; short-beard/chin silhouette remains corresponding. |
| black_woman | 0, 4, 120, 182, 183, 239 | Similar open-mouth shapes at 182/183; smooth lower-mouth detail; no apparent new chin-boundary tear. |
| east_asian_man_goatee | 0, 112, 120, 130, 172, 182, 239 | Smoothed grey moustache in both; simplified teeth; goatee remains attached and similarly positioned. |
| middle_eastern_man_full_beard | 0, 15, 120, 151, 171, 209, 239 | Blurred central moustache/mouth detail contrasts with sharper outer beard in both; matching dark opening at 171. |
| south_asian_woman | 0, 3, 69, 120, 182, 239 | Same blink at 69; similar peak opening at 182; soft teeth/mouth and some uneven chin shading are shared. |
| white_man_clean_shaven | 0, 15, 120, 145, 174, 184, 239 | Similar tooth rows/openings; frame 145 nearly closed in both, but is **not** labelled silence. |

The viewed full-frame sheets decode existing lossy review MP4s. Minute lip,
tooth-edge, hair or colour differences cannot be confidently assigned to the
engine rather than encoding from this evidence. This is not a raw pixel-equality
claim. All original UNet, TAESD and six e1 verdicts remain **FAIL**.

Audio was attempted through the available audio transport, which returned
“audio content omitted because you do not support audio input.” No speech was
heard by this reviewer. All six silence annotations and audio lip-sync judgments
therefore remain missing. The original audio files were not modified.

No supported tool provided continuous timed video consumption to this reviewer.
Selected and adjacent stills were inspected, but they cannot establish flicker,
shimmer, jaw wobble, natural motion or lip-sync timing. Continuous full-clip review,
native candidate comparison, production 16×3 poses, and live pose/speech
transitions remain unassessed. `visual_parity_approval=NOT_ISSUED`,
`candidate_evaluated=false`, and `release_ready=false`.
