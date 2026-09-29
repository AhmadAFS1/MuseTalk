export const meta = {
  name: 'review-repro-scripts',
  description: 'Audit scripts/repro_400fps (the MuseTalk 350/400 fps reproduction package) for correctness and completeness against what was actually run, without GPU work',
  whenToUse: 'After editing scripts/repro_400fps/*, or before handing the package to someone who will rebuild the engines on a new machine.',
  phases: [
    { title: 'Review', detail: 'static + dry-run check, history cross-check, fresh-machine completeness critic' },
  ],
}

const ROOT = '/workspace/MuseTalk-perf300'
const RP = ROOT + '/scripts/repro_400fps'
const SCHEMA = {
  type: 'object',
  properties: {
    scope: { type: 'string' },
    verified: { type: 'array', items: { type: 'string' } },
    problems: { type: 'array', items: { type: 'object', properties: {
      severity: { type: 'string', enum: ['blocker', 'major', 'minor'] }, file: { type: 'string' },
      description: { type: 'string' }, suggested_fix: { type: 'string' } }, required: ['severity', 'file', 'description', 'suggested_fix'] } },
  },
  required: ['scope', 'verified', 'problems'],
}
const common = `Package under review: ${RP}/ (README.md, lib.sh, 00_check.sh, 10_build_engines.sh, 20_gate.sh, 30_benchmark.sh, 40_videos.sh,
50_derive_recipe.sh). It must rebuild and re-measure the published MuseTalk results on an RTX 4070 SUPER: r5 = stagewise bs16
TensorRT UNet, variant srccache, INT8 recipe gmac_0.50 (117 layers) + TensorRT TAESD, measured in the six-stream full-recipe harness
(scripts/chin_multistream_render.py, preset stagewise16_taesdtrt); r2 = srcmix. The record of what was actually run is in
${ROOT}/docs/fps_comparisons/4070s_400fps_20260928/ (README.md §8, gate/run_assemble_gate.sh, validate_candidate.sh, blocks/*.sh,
int8_study/, chin_multistream/*.json) and ${ROOT}/docs/fps_comparisons/4070s_300fps_impl_20260928/unet_fp16/run_srccache.sh.
Engine manifests: ${ROOT}/models/tensorrt_unet_stagewise_sm89_{srcg50,srcmix,gmac_0.50}/bs16/manifest.json.
Rules: read-only; no GPU jobs, no engine builds, no harness runs (a --dry-run and bash -n are fine); never print secrets; small RAM.`

phase('Review')
const jobs = [
  () => agent(`${common}

Static check. Run bash -n on every script and 10_build_engines.sh --dry-run. For every command, confirm that each flag exists in the
target script's argparse (read the scripts: build_unet_stagewise.py, validate_unet_backend.py, vae_fast_decoder.py,
chin_multistream_render.py, quality_ab_metrics.py, video_lineage.py, int8_layer_study.py, stability_report.py, compare_quality.py),
that the paths exist, and that functions from lib.sh are used correctly (quoting, set -u, return codes, die paths).`,
    { label: 'review:static', phase: 'Review', schema: SCHEMA }),
  () => agent(`${common}

History cross-check. Compare every step in the package with what produced the published engines and numbers (the manifests'
build_flags, int8_calibration, build_log, variant; the builder commands in blocks/build_recipes_r2.sh and gate/run_assemble_gate.sh;
the harness args recorded in chin_multistream/*.json 'args'). Flag any flag, order, batch size, opt level, recipe path, preset or harness
argument that differs, and whether building every block into one fresh root (instead of assembling symlinks as the published set did)
changes anything. Check that 20_gate.sh matches the published gate commands.`, { label: 'review:history', phase: 'Review', schema: SCHEMA }),
  () => agent(`${common}

Fresh-machine completeness critic. Imagine a new RTX 4070 SUPER box with only this repo checked out. List everything the package
needs that it neither creates nor checks, with how to obtain it: venvs and exact packages (torch/TensorRT/modelopt/diffusers
versions from the manifests), weights, the six prepared avatars and their provenance, the calibration corpus (it is a symlink in this
checkout), the FaceMesh venv, the seed timing cache, disk and RAM. Check whether 00_check.sh catches each item and whether README.md
tells the reader. Also check that the README's expected-results table matches the source JSONs.`,
    { label: 'review:fresh-machine', phase: 'Review', schema: SCHEMA }),
]
const out = (await parallel(jobs)).filter(Boolean)
log(`${out.length} reviews; problems: ${out.reduce((n, r) => n + r.problems.length, 0)}`)
return out
