export const meta = {
  name: 'verify-lineage-videos',
  description: 'Independently verify the BEFORE/r2/r3/r4 lineage comparison videos (labels, layout, sync, metrics, visual quality) and adversarially check the evidence is not misleading',
  phases: [
    { title: 'Verify', detail: 'one agent per identity: extract frames, view them, check labels/metrics/audio, describe visible differences' },
    { title: 'Critic', detail: 'skeptic: look for anything misleading or unfair in how the evidence was produced' },
  ],
}

const ROOT = '/workspace/MuseTalk-perf300'
const OUT = ROOT + '/experiments/video_validation/lineage_before_r2_r3_r4'
const SCR = '/tmp/claude-0/-workspace/866169db-0af2-4cde-9a98-79a398efeda7/scratchpad'
const QR = ROOT + '/docs/fps_comparisons/4070s_300fps_impl_20260928/quality_metrics/runs'
const IDS = args.ids

const CHECK_SCHEMA = {
  type: 'object',
  properties: {
    identity: { type: 'string' },
    checks: { type: 'array', items: { type: 'object', properties: {
      check: { type: 'string' }, pass: { type: 'boolean' }, detail: { type: 'string' } }, required: ['check', 'pass', 'detail'] } },
    visual_observations: { type: 'string', description: 'what a viewer sees comparing r2/r3/r4 against BEFORE: lips, teeth, jaw/chin, seam, beard, sharpness, any artefact; cite frame numbers' },
    problems: { type: 'array', items: { type: 'object', properties: {
      severity: { type: 'string', enum: ['blocker', 'major', 'minor'] }, description: { type: 'string' }, suggested_fix: { type: 'string' } },
      required: ['severity', 'description', 'suggested_fix'] } },
  },
  required: ['identity', 'checks', 'visual_observations', 'problems'],
}

const common = `Context: MuseTalk (talking-head lip-sync) throughput work on one RTX 4070 SUPER. The user asked for video evidence of the new
implementations vs the previous working ones. ${OUT}/<identity>_lineage.mp4 shows 4 columns, left to right:
  BEFORE = the accepted pre-optimization render (TensorRT FP16 bs8 UNet + compiled TAESD, 100% chin, single stream 148-171 fps),
  r2 = 350.2 fps aggregate (stagewise FP16 UNet + source-prefix cache + INT8 down3/mid + TRT TAESD),
  r3 = 415.6 fps aggregate (r2 + broad INT8 PTQ on 6 UNet blocks),
  r4 NEW = 414.9 fps aggregate (r2 + layer-selective INT8 on 146 layers).
Rows: labels | full frame (512x896 scaled to 400x700) | 3x mouth zoom (nearest) | raw 256 px generated face (BEFORE column) or
|face - BEFORE face| x8 (round columns, black = identical) | cyan metrics lines from the quality tool vs BEFORE.
Every column was rebuilt from raw pre-encode frames and the whole canvas encoded once (libx264 crf 14, 24 fps, audio = the identity's
speech.wav). ${OUT}/lineage_report.json records bit-exactness checks (BEFORE vs render.json, each round vs the harness's recorded
raw_refined_sha256). Quality-tool JSONs: ${QR}/<identity>__<label>.json with labels srcmix_taesdtrt (r2), srcv1_taesdtrt (r3), srcblkA8 (r4).
Tools: ffmpeg/ffprobe (4.4), python3 with numpy, /workspace/.venvs/musetalk_trt_stagewise/bin/python (numpy, cv2). View PNGs with the Read tool.
Rules: read-only on the repo (write only under ${SCR}/verify_*), CPU only, no GPU jobs, keep RAM small (extract individual frames, never
decode whole videos into memory at once), do not print secrets, do not touch /dev/shm. Report facts you verified; if unsure, say so.`

phase('Verify')
const results = await pipeline(IDS, id => agent(`${common}

Your identity: ${id}. Verify ${OUT}/${id}_lineage.mp4 end to end:
1. ffprobe: resolution, frame count (expect 240), fps 24, duration ~10 s, an audio stream whose duration matches the video within 0.2 s,
   pixel format playable in VS Code/Chromium (yuv420p, H.264 High).
2. Extract at least 6 frames to ${SCR}/verify_${id}/ (e.g. frames 1, 48, 96, 144, 192, 240 plus the two stills frames listed in
   lineage_report.json identities.${id}.stills_frames; ffmpeg -vf "select=eq(n\\,K)" -vframes 1). View every PNG with Read.
   Check: the four column labels and fps numbers are correct and legible and not clipped/overlapping; the frame counter increments;
   the mouth zoom row actually shows the mouth in all four columns; the r2 diff panel is near-black, r3 shows broad differences,
   r4 differences concentrate around the lips; the BEFORE column row 3 shows the generated face.
3. Cross-check the cyan metrics text you read in the frames against ${QR}/${id}__{srcmix_taesdtrt,srcv1_taesdtrt,srcblkA8}.json
   (gates: lip.aperture_corr, lip.mean_abs_delta_px, flicker.mouth_ratio, flicker.jaw_ratio, chin.landmark_dev_mean_px,
   chin.landmark_dev_p99_px; overall.psnr face_box/mouth_roi mean_frame_db_capped) - flag any mismatch.
4. Check lineage_report.json for ${id}: BEFORE raw_matches_render_json true and every round bit_exact_vs_harness_render true.
5. Independently measure, from the extracted frames, the mean absolute difference of each round's mouth-zoom cell vs the BEFORE cell
   (crop by column: each column is 400 px wide) at 2+ frames, and confirm the ordering matches the metrics (r2 smallest).
6. Describe honestly what a viewer would notice between BEFORE and r2/r3/r4 in the mouth zoom (teeth, lip edges, inner mouth,
   beard/stubble texture, chin line, seam), citing frame numbers. Do not overstate; if nothing is visible at normal viewing, say so.
Return the structured result.`, { label: `verify:${id}`, phase: 'Verify', schema: CHECK_SCHEMA }))

phase('Critic')
const critic = await agent(`${common}

You are the skeptic. Try to find anything that would make this video evidence misleading or unfair to the BEFORE column or flattering
to the new rounds. Investigate concretely (read ${ROOT}/scripts/video_lineage.py and ${ROOT}/scripts/video_ab_round.py, the report JSON,
and the per-round READMEs in ${ROOT}/experiments/video_validation/r2_srcmix_taesdtrt_chin, r3_srcv1_int8_taesdtrt_chin,
r4_srcblkA8_int8sel_taesdtrt_chin). Questions to answer with evidence:
- Is BEFORE truly the accepted pre-change render (compare render.json / faces.npz provenance under
  /workspace/experiments/avatar_diversity_20260927/<id>/)? Are all columns using the same source frames, audio, chin code (chin.py sha)?
- Are the fps labels attributable to measured runs (docs/fps_comparisons/4070s_300fps_impl_20260928/chin_multistream/*.json and
  docs/fps_comparisons/4070s_400fps_20260928/chin_multistream/T_srcblkA8.json 'aggregate_fps_per_repeat'), and is the BEFORE fps
  (single-stream 148-171) a fair statement of what the pre-change pipeline did (see render.json of the identities)?
- Is the encoding identical for all columns (single encode)? Is the x8 diff amplification applied identically? Does scaling hide anything
  (e.g. full frame downscaled; mouth zoom nearest)? Is the frame alignment between columns identical (same frame index)?
- The older per-round A/B videos (video_ab_round.py) used the stored accepted refined_raw.mp4 (~1.9 Mbit/s) against crf-12 candidates:
  quantify with ffprobe whether that older comparison was biased, so the user knows which videos to trust.
Return a concise list of verified facts, any problems (with severity), and concrete fixes. Default to skepticism.`,
  { label: 'critic', phase: 'Critic', schema: {
    type: 'object',
    properties: {
      verified_facts: { type: 'array', items: { type: 'string' } },
      problems: { type: 'array', items: { type: 'object', properties: {
        severity: { type: 'string', enum: ['blocker', 'major', 'minor'] }, description: { type: 'string' }, suggested_fix: { type: 'string' } },
        required: ['severity', 'description', 'suggested_fix'] } },
    },
    required: ['verified_facts', 'problems'] } })

return { results: results.filter(Boolean), critic }
