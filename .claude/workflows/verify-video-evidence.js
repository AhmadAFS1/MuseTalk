export const meta = {
  name: 'verify-video-evidence',
  description: 'Independently verify MuseTalk before/after comparison videos (bit-exactness, labels, metrics, verdicts, fairness) and audit the README claims against the source JSONs',
  whenToUse: 'After scripts/video_lineage.py or scripts/video_signoff.py produced a new round of comparison videos, before showing them to the user for sign-off.',
  phases: [
    { title: 'Verify', detail: 'per identity pair: ffprobe, extracted frames viewed, labels/metrics/verdicts vs JSON; README auditor; fairness skeptic' },
  ],
}

// args (all optional; defaults = the r2/r5 sets published 2026-09-29):
//   { sets: [{dir, what}], ids: [..6 avatar ids..], readmes: [paths], quality_labels: {label: round}, throughput_jsons: [paths] }
const ROOT = '/workspace/MuseTalk-perf300'
const V = ROOT + '/experiments/video_validation'
const A = args || {}
const SETS = A.sets || [
  { dir: V + '/focus_before_r2_r5', what: '3 columns BEFORE | r2 | r5 at native 512 px, crf 10, diff row + metrics + verdicts' },
  { dir: V + '/lineage_all_rounds', what: '5 columns BEFORE | r2 | r3 | r4 | r5 at 400 px, crf 12' },
  { dir: V + '/signoff_r2_r5', what: 'plain BEFORE | r2 | r5 side by side + clean single clips + reel, crf 14' },
]
const IDS = A.ids || ['black_man_short_beard', 'black_woman', 'east_asian_man_goatee', 'middle_eastern_man_full_beard',
  'south_asian_woman', 'white_man_clean_shaven']
const READMES = A.readmes || [V + '/README.md', V + '/lineage_all_rounds/README.md', V + '/focus_before_r2_r5/README.md',
  V + '/r5_srcg50_int8gmac_taesdtrt_chin/README.md', ROOT + '/docs/fps_comparisons/4070s_400fps_20260928/README.md']
const QL = A.quality_labels || { srcmix_taesdtrt: 'r2', srcv1_taesdtrt: 'r3', srcblkA8: 'r4', srcg50: 'r5' }
const TJ = A.throughput_jsons || [
  ROOT + '/docs/fps_comparisons/4070s_300fps_impl_20260928/chin_multistream/Te_srcmix_taesdtrt_n6.json',
  ROOT + '/docs/fps_comparisons/4070s_400fps_20260928/chin_multistream/T_srcg50.json',
  ROOT + '/docs/fps_comparisons/4070s_400fps_20260928/chin_multistream/T_baseline_pair.json',
  ROOT + '/docs/fps_comparisons/4070s_400fps_20260928/chin_multistream/T_srcg50_pair.json',
  ROOT + '/docs/fps_comparisons/4070s_400fps_20260928/chin_multistream/T_srcg50_sustained.json',
]
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

const common = `Context: MuseTalk talking-head throughput work on one RTX 4070 SUPER. The user signs off on quality from comparison videos.
Video sets to verify:
${SETS.map(s => `  ${s.dir}: ${s.what}`).join('\n')}
Each set has a report JSON (lineage_report.json or signoff_report.json) with per-column bit-exactness against recorded render hashes.
Generators: ${ROOT}/scripts/video_lineage.py, ${ROOT}/scripts/video_signoff.py. Quality JSONs: ${QR}/<id>__<label>.json with labels
${Object.entries(QL).map(([k, v]) => `${k} (${v})`).join(', ')}. Throughput JSONs: ${TJ.join(', ')} (aggregate_fps_per_repeat, median_aggregate_fps).
The like-for-like BEFORE baseline is the pre-change backends in the same six-stream harness (T_baseline_pair).
Tools: ffmpeg/ffprobe, python3 + numpy, /workspace/.venvs/musetalk_trt_stagewise/bin/python (numpy, cv2). View PNGs with Read.
Rules: read-only on the repo (write only under your own scratch dir in /tmp), CPU only, no GPU jobs, keep RAM small (extract single
frames), never print secrets, do not touch /dev/shm. Report only verified facts; say when unsure.`

const pairs = []
for (let i = 0; i < IDS.length; i += 2) pairs.push(IDS.slice(i, i + 2))

phase('Verify')
const jobs = pairs.map(p => () => agent(`${common}

Your identities: ${p.join(', ')}. For every set and both identities:
1. ffprobe: codec/profile/pix_fmt playable in a browser, frame count, fps, duration, audio present and the same length.
2. Extract >= 4 frames per video (include any stills frames the report lists; ffmpeg select n = frame-1), view them, and check that
   labels, fps and speedups are correct and legible, the lips stay inside the mouth zoom, dividers don't cross text, and verdict
   colours and PASS/FAIL agree with the quality JSONs. The strict landmark gate is mean<=0.05 and p99<=0.15; the proposed bar is
   <=0.10/<=0.35; "other gates" means no failing gate outside chin.landmark_dev*.
3. Confirm the report's bit-exact flags are true and spot-check one hash against the harness JSON it names.
4. Describe honestly what a viewer sees comparing BEFORE with each round, citing frames.`,
  { label: `verify:${p.map(x => x.split('_')[0]).join('+')}`, phase: 'Verify', schema: RES }))

jobs.push(() => agent(`${common}

You are the README/claims auditor. Read ${READMES.join(', ')}. Re-derive every number, range and count they state from the source JSONs.
Flag anything that is wrong, rounded misleadingly or unsupported, giving the correct value and the file. Also flag any "bit-exact",
"lossless" or "like-for-like" claim that the evidence doesn't support.`, { label: 'verify:readme-audit', phase: 'Verify', schema: RES }))

jobs.push(() => agent(`${common}

You are the fairness skeptic. Look for anything that makes the evidence flatter the new rounds. Examples: unequal encodes, baselines
measured in a different mode, verdict colours that follow a lenient bar, diff panels that hide or exaggerate changes, crops that
misalign columns, throughput numbers taken under different conditions (compare timestamps, clocks, power, busy fraction), or claims
of margins that the per-repeat values don't support. Return the verified facts as checks, plus any problems.`,
  { label: 'verify:skeptic', phase: 'Verify', schema: RES }))

const out = (await parallel(jobs)).filter(Boolean)
log(`${out.length} verifier reports; problems: ${out.reduce((n, r) => n + r.problems.length, 0)}`)
return out
