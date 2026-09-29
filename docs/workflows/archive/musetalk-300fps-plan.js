export const meta = {
  name: 'musetalk-300fps-plan',
  description: 'Design competing 300 fps plans from measured evidence, score and synthesize, adversarially verify, then revise into the final plan',
  phases: [
    { title: 'Design', detail: '4 independent plans from different angles' },
    { title: 'Synthesize', detail: 'score plans, graft best ideas into one plan' },
    { title: 'Verify', detail: '3 adversarial lenses on the synthesized plan' },
    { title: 'Revise', detail: 'apply verified corrections into the final plan' },
  ],
}

const S = '/tmp/claude-0/-workspace/866169db-0af2-4cde-9a98-79a398efeda7/scratchpad/throughput300'

const BASE = `
## Context
You are helping plan how to raise a MuseTalk v1.5 real-time talking-head server (/workspace/MuseTalk) on ONE
RTX 4070 SUPER (12 GB, 220 W cap) + Ryzen 9 7950X (16C/32T) + 30 GB RAM from today's state to ~300 fps
aggregate (15 concurrent WebRTC streams x 20 fps), stretch 400 fps (20 streams). The user asked for a full
feasibility analysis and a FULL PLAN ONLY (no implementation this turn).

A measurement/reading workflow has already run. START by reading ${S}/EVIDENCE_DIGEST.md (condensed, tagged
[M]=measured now, [D]=doc, [I]=inferred). Deep details are in ${S}/wf1_result.json (JSON: readers[] with
summary/per_frame_costs/levers/constraints/open_questions; probes.{unet,taesd,cpu} with measurements/conclusions)
and probe JSON/scripts under ${S}/unet_probe, taesd_probe, cpu_probe, audio_tts. You may read any repo file to
ground file paths, function names, env flags and line numbers (key files: MuseTalk/scripts/hls_gpu_scheduler.py,
api_server.py, scripts/trt_runtime.py, scripts/vae_fast_decoder.py, scripts/webrtc_tracks.py,
scripts/webrtc_native_vp8.py, scripts/api_avatar.py, musetalk/utils/blending.py,
character_factory/h3_avatar_workflow/chin.py, experiments/chin_fps_validation_20260927/run.py, load_test_webrtc.py).

## Hard guardrails for you
- STRICTLY READ-ONLY and LIGHTWEIGHT: do not run anything on the GPU, do not start servers, do not run heavy
  Python processes (the box has 30 GB RAM shared with a user's live test server and other sessions; earlier
  probes in this study likely contributed to an OOM kill of the user's server). Reading files, grep, git log/show,
  and trivial arithmetic in python3 are fine. Do not write files anywhere except optionally under ${S}/plan/.
- Every quantitative claim you make must cite its source tag ([M] probe measurement, [D] doc path, [I] your
  inference with the arithmetic shown). Do not invent measurements.

## User constraints the plan must honor
- REQUIRED output recipe: TAESD decoder + MuseTalk native avatar encoder + 100% chin alignment + refined seam +
  expressive H3 source. Output quality regressions are judged by the user on labelled same-speed comparison video
  (full frame + mouth zoom) plus numeric gates; lossy levers are ALLOWED to be attempted ("attempt, then gate
  visually") but each must sit behind a one-line env flag rollback, and every implementation round must end with a
  labelled comparison video. 224/192 ROI was rejected as blurry. 20 fps per stream (no lowering fps).
- The live server and other sessions share this box; plan steps that build engines or load-test must include
  resource hygiene (GPU lease / quiet-box check, RAM headroom, disk space) since disk has ~1.2-1.4 GB free and a
  UNet engine artifact is 2.2 GB. Deleting files is the user's decision — list candidates, don't assume.
`

const PLAN_SHAPE = `
## What to produce (markdown, as your final text)
A complete, self-contained plan (not only your angle — but lead with your angle's priorities), containing:
1. Verdict on 300 fps and on 400 fps feasibility on this exact machine, with confidence and the binding ceilings.
2. A per-frame BUDGET TABLE: today vs after each phase, for GPU ms/frame (UNet, TAESD, other), live-path
   efficiency (fraction of GPU ceiling actually delivered), CPU cores, RAM, VRAM — each cell sourced.
3. Phases with numbered work items. For each item: what exactly changes (file/function/flag), rollback flag,
   expected gain (ms/frame or fps, sourced), quality risk, the gate that must pass, effort, dependencies.
4. A projected fps waterfall (cumulative) with low/expected/high, clearly separating measured vs estimated.
5. The validation protocol: how "300 fps / 15 streams" is proven (live WebRTC load test design, TTS off-box,
   pre-synthesized audio, metrics & pass criteria, quiet-box hygiene), plus quality gates and the video deliverable.
6. Risks, kill criteria, and explicitly CLOSED levers (so nobody retries them) with evidence.
7. Decisions the user must make (disk cleanup candidates, duty-cycle assumption, codec/transport fps, training
   compute for INT8-QAT/distillation, etc.).
Be concrete and quantitative. Length: as long as needed, but no filler.
`

const ANGLES = [
  { key: 'exact-first', prompt: `Your angle: EXACT-FIRST SYSTEMS ENGINEER. Push as far as possible with bit-exact or
gate-passing FP16 levers only (engine rebuilds, CUDA graphs, native bs16, ONNX-parser path, TAESD TRT staged exact
crop with per-avatar crop rows, removing syncs, async double-buffering, zero quality change). Quantify exactly how far
lossless gets (is 300 reached with margin? is 400?) and only then position lossy levers as optional. Be rigorous about
double counting between overlapping levers (e.g., CUDA graphs vs bs16+graph measurement, ONNX-path vs bs16).` },
  { key: 'serving-first', prompt: `Your angle: SERVING ARCHITECT, RISK-FIRST. History says live WebRTC always landed below
model-path capacity and strict smooth 20 fps never exceeded 4 streams on any GPU. Treat the serving path as the real
blocker: design the target process architecture (GPU process + media worker processes + FaceMesh/chin pool, shared
memory rings, ordered per-stream actors), non-blocking handoff, async double-buffered GPU pipeline, idle-frame cache,
encoder fix (NVENC patch is dead code; 12-session NVENC cap; VP8 1-thread), deadline-aware scheduling, RAM/thread
budgets (server hit 10.5 GB RSS and >1000 threads at 10 sessions, then OOM). Make "measure the live path first" Phase 0.` },
  { key: 'model-accel', prompt: `Your angle: MODEL ACCELERATION FOR 400 fps. Plan the quality-risky GPU levers the user is
willing to attempt: mixed-precision INT8 UNet (per-block sensitivity sweep; up1 and 1024-token transformers likely FP16;
multi-avatar calibration from MUSETALK_UNET_CALIBRATION_CAPTURE; SmoothQuant; QAT or distillation to recover), UNet
block pruning + distillation (SoulX precedent: decoder distill worked, generator pruning broke lip sync), INT8 TAESD,
a TRT 10.9+ upgrade venv; build logistics under the disk/RAM/VRAM limits (11 GB host RSS for a bs16 build, 2.2 GB
artifacts), training feasibility on a 12 GB card vs a rented GPU, and the gates (capture mae, SyncNet as relative
diagnostic only, labelled video). Still include the lossless and serving work as prerequisites.` },
  { key: 'capacity-product', prompt: `Your angle: CAPACITY & PRODUCT ENGINEER. The user's real goal is "15-20 concurrent
20 fps streams on one server". Separate GPU demand (simultaneous speakers only) from encode/transport demand (all
sessions x playback fps, maybe 30 fps transport), skip GPU work for discarded frames (exact_silence / raw idle),
duty-cycle-aware admission with p99 simultaneous speakers, burst absorption via run-ahead, per-session RAM/VRAM/CPU
costs for 15-20 sessions (idle caches, chin caches 8.5 MB/source frame, NVENC VRAM), SLOs (first-frame latency,
held-frame budget), and the load-test protocol that proves it. Still include the GPU and serving work needed.` },
]

const SYNTH_SCHEMA = {
  type: 'object',
  properties: {
    scores: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          plan: { type: 'string' },
          evidence_fidelity: { type: 'number' }, feasibility: { type: 'number' }, risk_handling: { type: 'number' },
          sequencing: { type: 'number' }, user_constraints: { type: 'number' }, measurement_rigor: { type: 'number' },
          notes: { type: 'string' },
        },
        required: ['plan', 'evidence_fidelity', 'feasibility', 'risk_handling', 'sequencing', 'user_constraints', 'measurement_rigor', 'notes'],
      },
    },
    winner: { type: 'string' },
    grafts: { type: 'array', items: { type: 'string' }, description: 'ideas taken from non-winning plans' },
    plan_markdown: { type: 'string' },
  },
  required: ['scores', 'winner', 'grafts', 'plan_markdown'],
}

const VERIFY_SCHEMA = {
  type: 'object',
  properties: {
    verdict: { type: 'string', description: 'overall: is the plan sound under your lens? 3-6 sentences' },
    issues: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          claim: { type: 'string', description: 'the exact plan claim or item being challenged' },
          problem: { type: 'string' },
          severity: { type: 'string', enum: ['critical', 'major', 'minor'] },
          correction: { type: 'string', description: 'what the plan should say/do instead' },
          evidence: { type: 'string', description: 'source path / measurement / arithmetic that supports the challenge' },
        },
        required: ['claim', 'problem', 'severity', 'correction', 'evidence'],
      },
    },
    missing: { type: 'array', items: { type: 'string' }, description: 'things the plan omits that it needs' },
  },
  required: ['verdict', 'issues', 'missing'],
}

const LENSES = [
  { key: 'gpu-arithmetic', prompt: `Lens: GPU ARITHMETIC & CLAIMS AUDITOR. Recompute every GPU ms/frame and fps
projection in the plan from the measurements in ${S}/wf1_result.json and the probe JSON files (e.g.
${S}/unet_probe/*.json, ${S}/taesd_probe/*.json). Check for double counting between levers that overlap (CUDA graph
gain vs the bs16+graph number; ONNX-parser gain measured at bs8 as a sum of per-block engines vs bs16; TAESD 0.739 vs
0.86 baselines; sustained power-capped clocks vs isolated runs), unit errors, and any number with no source. Flag
optimism: a claim that cannot be traced to evidence should be downgraded to an estimate with a range. Also check the
INT8 quality numbers are represented honestly.` },
  { key: 'serving-resources', prompt: `Lens: SERVING / CPU / RAM / VRAM FEASIBILITY SKEPTIC. Try to break the plan's
claim that the live path can deliver ~300 fps. Check: host RAM at 15-20 sessions (server was 7-10.5 GB RSS at 10
sessions and OOMed at 0.8 GB free; plus idle yuv caches ~165 MB per clip, chin caches ~2 GB per pose untrimmed,
FaceMesh ~31 MB per stream graph, media worker processes); thread counts; VRAM (bs16 engine + TRT TAESD + cudagraph
pools + NVENC ~200 MB/session + whisper); NVENC 12-session cap; IPC/shared-memory costs of a process split; ordering
+ one-frame lookahead of the chin filter across batches; GIL on the GPU-feeding thread (spin-wait, enqueue slowdown
under Python load); event-loop saturation; the 30 fps vs 20 fps transport question. Say which live-fps projections
are unjustified and what measurement must precede them.` },
  { key: 'completeness-constraints', prompt: `Lens: COMPLETENESS & USER-CONSTRAINT CRITIC. What is missing or wrong
relative to what the user needs and has required? Check: TAESD + native encoder + 100% chin + refined seam preserved
everywhere; every lossy lever flagged with a one-line env rollback and a labelled video gate; chin path exactness gates
(verify_baseline.py, SHA equality) carried into the live integration; disk/RAM/GPU hygiene steps before builds and
load tests (and not disrupting the user's live server or other sessions); closed levers not re-proposed (FP8, 2nd CUDA
stream, bs>16, ROI 224, DeepCache, source-only caching, latent-space crop); decisions for the user listed; kill
criteria present; validation protocol would actually prove "15 streams x 20 fps smooth" (not just aggregate fps);
the plan explains what the user's "160 fps" figure is versus live reality. Also check sequencing: are dependencies
ordered so each phase can be measured independently?` },
]

phase('Design')
const plans = await parallel(ANGLES.map(a => () =>
  agent(`${BASE}\n\n${a.prompt}\n${PLAN_SHAPE}`, { label: `design:${a.key}`, phase: 'Design' })
    .then(t => t ? { key: a.key, text: t } : null)))
const good = plans.filter(Boolean)
log(`${good.length}/4 plans produced`)

phase('Synthesize')
const synth = await agent(`${BASE}

You are the JUDGE and SYNTHESIZER. Below are ${good.length} independent plans. First score each (1-10) on
evidence_fidelity (numbers traceable to [M]/[D]/[I] sources, no inventions), feasibility, risk_handling, sequencing,
user_constraints, measurement_rigor. Pick a winner. Then write ONE final plan (plan_markdown) that starts from the
winner and grafts the best ideas from the others. Resolve contradictions between plans by going back to the evidence
files. Required structure of plan_markdown:
# MuseTalk on RTX 4070 SUPER: path to 300 fps (15 x 20 fps streams)
## 1. Verdict  (300 fps: yes/no + confidence; 400 fps; the binding ceilings in order)
## 2. Where the "160 fps" number comes from vs what live WebRTC delivers today
## 3. Budget: today vs target  (GPU ms/frame table; live efficiency; CPU cores; RAM; VRAM)
## 4. The plan  (Phase 0 prerequisites & baseline measurement; then phases; numbered items with
      file/function/flag, rollback flag, expected gain [sourced], risk, gate, effort, dependencies)
## 5. Projected fps waterfall  (cumulative low/expected/high; measured vs estimated marked)
## 6. Validation protocol  (live load test design + pass criteria; quality gates; labelled video)
## 7. Risks and kill criteria
## 8. Closed levers — do not retry  (with evidence)
## 9. Decisions needed from you
## 10. Appendix: evidence index  (paths to probe JSON / scripts / docs)
Write for the user (a hands-on engineer who knows this codebase) and for whoever executes the phases later
(possibly another agent), so each item must be actionable on its own.

${good.map(p => `\n\n======== PLAN: ${p.key} ========\n${p.text}`).join('')}`,
  { label: 'synthesize', phase: 'Synthesize', schema: SYNTH_SCHEMA })

if (!synth) return { plans: good, synth: null }
log(`winner: ${synth.winner}; grafts: ${synth.grafts.length}`)

phase('Verify')
const reviews = await parallel(LENSES.map(l => () =>
  agent(`${BASE}

You are an ADVERSARIAL VERIFIER. Your job is to find what is wrong, unsupported, or missing in the plan below —
default to flagging when a claim cannot be traced to evidence. Only flag real problems; do not nitpick wording.
${l.prompt}

======== PLAN UNDER REVIEW ========
${synth.plan_markdown}`, { label: `verify:${l.key}`, phase: 'Verify', schema: VERIFY_SCHEMA })
    .then(r => r ? { lens: l.key, ...r } : null)))
const revs = reviews.filter(Boolean)

phase('Revise')
const final = await agent(`${BASE}

You are the FINAL EDITOR. Below is a synthesized plan and ${revs.length} adversarial reviews. For each review issue,
check it against the evidence files yourself; apply it if it holds (fix numbers, add missing items, tighten claims),
reject it if the evidence contradicts it. Keep the plan's section structure. Keep every number sourced. Make the
verdict honest: state clearly which parts of 300 fps are measured, which are projected, and what single measurement
would most change the conclusion. Return:
- plan_markdown: the final plan
- changes_applied: list of changes you made (one line each, citing which lens)
- rejected: issues you rejected and why
- residual_uncertainties: the top uncertainties that remain

======== SYNTHESIZED PLAN ========
${synth.plan_markdown}

======== REVIEWS ========
${JSON.stringify(revs, null, 1)}`,
  { label: 'revise', phase: 'Revise', schema: {
    type: 'object',
    properties: {
      plan_markdown: { type: 'string' },
      changes_applied: { type: 'array', items: { type: 'string' } },
      rejected: { type: 'array', items: { type: 'string' } },
      residual_uncertainties: { type: 'array', items: { type: 'string' } },
    },
    required: ['plan_markdown', 'changes_applied', 'rejected', 'residual_uncertainties'],
  } })

return { scores: synth.scores, winner: synth.winner, grafts: synth.grafts, reviews: revs, final }
