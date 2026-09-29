export const meta = {
  name: 'verify-final-lineage',
  description: 'Second-pass verification of the final lineage_all_rounds (BEFORE/r2-r5) and focus_before_r2_r5 videos, their READMEs and claims',
  phases: [
    { title: 'Verify', detail: 'video checks per identity pair, README number audit, fairness skeptic' },
  ],
}

const ROOT = '/workspace/MuseTalk-perf300'
const V = ROOT + '/experiments/video_validation'
const SCR = '/tmp/claude-0/-workspace/866169db-0af2-4cde-9a98-79a398efeda7/scratchpad'
const QR = ROOT + '/docs/fps_comparisons/4070s_300fps_impl_20260928/quality_metrics/runs'

const RES = {
  type: 'object',
  properties: {
    scope: { type: 'string' },
    checks: { type: 'array', items: { type: 'object', properties: {
      check: { type: 'string' }, pass: { type: 'boolean' }, detail: { type: 'string' } }, required: ['check', 'pass', 'detail'] } },
    visual_observations: { type: 'string' },
    problems: { type: 'array', items: { type: 'object', properties: {
      severity: { type: 'string', enum: ['blocker', 'major', 'minor'] }, description: { type: 'string' }, suggested_fix: { type: 'string' } },
      required: ['severity', 'description', 'suggested_fix'] } },
  },
  required: ['scope', 'checks', 'visual_observations', 'problems'],
}

const common = `Context: MuseTalk talking-head throughput work on one RTX 4070 SUPER. The user asked for video evidence of the new implementations vs the
previous working ones. A first review found the evidence bit-exact but asked for presentation fixes; those were applied and the final sets rendered:
  ${V}/lineage_all_rounds/<id>_lineage.mp4  (5 columns: BEFORE | r2 350.2 | r3 415.6 | r4 414.9 | r5 NEW 400.9 fps; 400 px columns; crf 12)
  ${V}/focus_before_r2_r5/<id>_lineage.mp4  (3 columns: BEFORE | r2 (previous working) | r5 NEW; native 512 px; crf 10)
Each has <id>_stills.png (claimed lossless from the raw canvas) and lineage_report.json. Rows: labels (name, fps with 'Nx BEFORE' speedup over
BEFORE's like-for-like 252.0 fps 6-stream harness number) | full frame | lip-following mouth zoom (factor printed) | face+neck region of the composited
output: BEFORE pixels, rounds show 0 where identical else 40 + 8*|diff| per channel | metrics lines with colour-coded verdicts (green pass,
orange fail); BEFORE column lists thresholds. Generator: ${ROOT}/scripts/video_lineage.py. Quality JSONs: ${QR}/<id>__<label>.json with
labels srcmix_taesdtrt (r2), srcv1_taesdtrt (r3), srcblkA8 (r4), srcg50 (r5). Throughput JSONs: ${ROOT}/docs/fps_comparisons/4070s_300fps_impl_20260928/chin_multistream/Te_srcmix_taesdtrt_n6.json,
Tf_srcv1_taesdtrt_n6.json, Ta_baseline_n6.json; ${ROOT}/docs/fps_comparisons/4070s_400fps_20260928/chin_multistream/T_srcblkA8.json, T_srcg50.json,
T_baseline_pair.json, T_srcg50_pair.json (field aggregate_fps_per_repeat / median_aggregate_fps).
Tools: ffmpeg/ffprobe 4.4, python3 + numpy, /workspace/.venvs/musetalk_trt_stagewise/bin/python (numpy, cv2). View PNGs with Read.
Rules: read-only on the repo (write only under ${SCR}/verify2_*), CPU only, no GPU jobs, keep RAM small (extract single frames), no secrets,
do not touch /dev/shm. Report only verified facts; say when unsure.`

const pairs = [['black_man_short_beard', 'black_woman'], ['east_asian_man_goatee', 'middle_eastern_man_full_beard'], ['south_asian_woman', 'white_man_clean_shaven']]

phase('Verify')
const jobs = pairs.map(p => () => agent(`${common}

Your identities: ${p.join(', ')}. For BOTH final sets and both identities:
1. ffprobe: H.264 High yuv420p, 240 frames, 24 fps, 10 s, audio present and matching length.
2. Extract >= 4 frames per video (include the two stills frames from lineage_report.json stills_frames_1based; ffmpeg select n = frame-1) to
   ${SCR}/verify2_${p[0]}/ and view them. Check: labels/fps/speedups correct and legible (speedup = fps/252.0, e.g. r5 400.9 -> 1.59x); the
   printed zoom factor matches CW/box width (lineage_report mouth_box_size_wh; CW 400 or 512); lips fully inside the zoom in every viewed
   frame; dividers do not cross the title; the verdict lines' colours and PASS/FAIL agree with the quality JSONs (strict = mean<=0.05 and
   p99<=0.15; calibrated = mean<=0.10 and p99<=0.35; 'all other gates' = no failing gate outside chin.landmark_dev*); metric values match JSONs.
3. Verify the diff panel semantics on one frame: rebuild is expensive, so instead check that in the encoded panel r2 shows sparse dim speckle
   (not pure black) and that r5 is denser than r2 and sparser than r3/r4 around the mouth - measure fraction of panel pixels > 20.
4. Verify the stills PNGs are lossless relative to the canvas: compare a still's full-frame cell against the same frame extracted from the mp4
   - the PNG should be sharper (no codec noise) and its BEFORE-vs-r2 full-frame cell difference should be ~0 outside the face region.
5. Describe honestly what a viewer sees comparing BEFORE, r2 and r5 at native resolution (focus set), citing frames.`,
  { label: `verify2:${p[0].split('_')[0]}+${p[1].split('_')[0]}`, phase: 'Verify', schema: RES }))

jobs.push(() => agent(`${common}

You are the README/claims auditor. Read ${V}/README.md (rounds table + notes), ${V}/lineage_all_rounds/README.md, ${V}/focus_before_r2_r5/README.md,
${V}/r5_srcg50_int8gmac_taesdtrt_chin/README.md and ${ROOT}/docs/fps_comparisons/4070s_400fps_20260928/README.md section 8.
Re-derive every number they state from the source JSONs (throughput JSONs above; quality JSONs over all 6 identities; lineage_report.json
bit-exact flags; ${ROOT}/docs/fps_comparisons/4070s_400fps_20260928/gate/gunet_srcg50_{main,holdout}.json and gunet_srcblkA8_*; the TAESD gate
${ROOT}/docs/fps_comparisons/4070s_300fps_impl_20260928/taesd_trt/gate_taesd_trt.json). Flag every number/range/count that is wrong, rounded
misleadingly, or unsupported (give the correct value and the file). Also flag any claim of 'bit-exact', 'lossless', 'like-for-like' that the
evidence does not support.`, { label: 'verify2:readme-audit', phase: 'Verify', schema: RES }))

jobs.push(() => agent(`${common}

You are the skeptic for the FINAL sets. The first review's fairness issues were: fps baseline mode, missing verdicts, diff panel hiding 1-LSB
changes, inaccurate zoom label, BEFORE overstated as 'accepted'. Check each is actually fixed in the final videos/READMEs (not just claimed),
and look for NEW problems introduced by the changes: e.g. does the 40+8x diff floor exaggerate small changes in a way that misleads (it applies
equally to all rounds - is that stated?); does the lip-following zoom introduce jitter or misalignment between columns (it must be the same crop
per frame for all columns - verify from the code and by checking a static background feature in two adjacent columns); does the face+neck crop
include the chin-warp region; are the like-for-like fps numbers (BEFORE 252.0, r5 400.7-400.9) measured under comparable conditions
(compare T_baseline_pair.json and T_srcg50_pair.json: timestamps, GPU clocks/power, busy fraction, mem_available); is r5 >= 400 robust
(per-repeat values) or marginal - say so plainly. Return verified facts as checks and any problems.`, { label: 'verify2:skeptic', phase: 'Verify', schema: RES }))

const out = await parallel(jobs)
return out.filter(Boolean)
