export const meta = {
  name: 'vast-template-audit',
  description: 'Audit whether the unchanged Vast onstart template yields the full MuseTalk r5 pipeline (4 audits + adversarial verify)',
  phases: [
    { title: 'Audit', detail: 'template trace, runtime inventory, venv parity, secrets/control plane' },
    { title: 'Verify', detail: 'one adversarial verifier per audit, refuting each gap/risk/blocker' },
  ],
}

const SCRATCH = '/tmp/claude-0/-workspace/866169db-0af2-4cde-9a98-79a398efeda7/scratchpad/vast_audit'

const TEMPLATE = `#!/usr/bin/env bash
set -Eeuo pipefail
mkdir -p /workspace
BOOT_LOG=/workspace/bootstrap.log
exec > >(tee -a "$BOOT_LOG") 2>&1
trap 'echo "[bootstrap] failed at line $LINENO with exit $?"' ERR
LOCK_FILE=/tmp/musetalk-onstart.lock
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[bootstrap] another onstart is already running; exiting"
  exit 0
fi
echo "[bootstrap] start $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
REPO_DIR=/workspace/MuseTalk
STAGE_DIR="$(mktemp -d /workspace/MuseTalk.stage.XXXXXX)"
REPO_URL="https://github.com/AhmadAFS1/MuseTalk.git"
REPO_BRANCH="main"
cleanup() { rm -rf "$STAGE_DIR"; }
trap cleanup EXIT
if ! command -v git >/dev/null 2>&1; then
  apt-get update -y
  apt-get install -y git ca-certificates
fi
echo "[bootstrap] cloning git repo checkout"
rm -rf "$STAGE_DIR"
git clone --depth 1 --branch "$REPO_BRANCH" "$REPO_URL" "$STAGE_DIR"
echo "[bootstrap] publishing git checkout"
rm -rf "$REPO_DIR"
mv "$STAGE_DIR" "$REPO_DIR"
cd "$REPO_DIR"
git config --global --add safe.directory "$REPO_DIR" || true
git remote -v
git status --short --branch
export MUSETALK_AWS_SECRET_ID="arn:aws:secretsmanager:us-east-1:<acct>:secret:lingua/musetalk-worker-runtime-<suffix>"
export MUSETALK_AWS_SECRET_REGION="us-east-1"
export MUSETALK_SECRETS_STRICT="true"
export MUSETALK_SECRETS_VERIFY_S3="1"
(the same four exports repeated once more)
export AWS_ACCESS_KEY_ID="<redacted: the bootstrap IAM user's key>"
export AWS_SECRET_ACCESS_KEY="<redacted>"
export AWS_DEFAULT_REGION="us-east-1"
SETUP_CLEAN=1 \\
SETUP_FULL_STACK=1 \\
STARTUP_TIMEOUT_SECONDS=1800 \\
PROFILE=throughput_record \\
PORT=8000 \\
bash scripts/vast_onstart.sh`

const CONTEXT = `CONTEXT
/workspace/MuseTalk is the MuseTalk repo on a shared RTX 4070 SUPER box. Local main is e0d0985, two commits ahead of
origin/main (49e58d4) and NOT pushed yet. Those two commits:
- make recipe r5 the default (no MUSETALK_RECIPE needed) and skip the old .ts UNet build by default;
- add pinned S3 engine bundles: configs/recipes/r5.env's group r5_engines requires
  bundle:rtx4070super-r5-srcg50-int8|ampere-plus-r5-srcg50-int8 (ordered candidates; the first that fits the host
  and is restored wins; descriptors in configs/trt_bundles/*.json; the resolver scripts/musetalk_host_profile.py
  bundle-check and scripts/vast_onstart.sh restore_r5_bundle apply the same host rule);
- the rtx4070super bundle is exact-GPU (RTX 4070 SUPER only) and is already in S3; the ampere-plus bundle
  (TensorRT hardware compatibility AMPERE_PLUS; compute capability 8.0-9.0, TensorRT 10.3.0, >= 8 GB VRAM) is built
  locally in /workspace/MuseTalk/tmp/trt_bundle/ but NOT uploaded to S3 yet (publisher login expired).
The user runs Vast.ai instances, mostly RTX 3090s, with this onstart template, which they do not want to change
(AWS key values redacted here on purpose):
<template>
${TEMPLATE}
</template>
The user's goal: keep this template unchanged and have a Vast instance end up with everything this machine has
for the MuseTalk pipeline to run: serving (live WebRTC calls with recipe r5) and avatar creation
(POST /avatars/prepare). Evaluate HEAD (e0d0985) as "what the clone gets after the push", and say where
origin/main today differs.

HARD RULES (violating any is a failure)
- Read-only. Do not edit, create, move or delete anything anywhere, except scratch files under ${SCRATCH}/<your label>/.
- No GPU work, no servers, no pip/apt installs, no model downloads, no git commit/push/fetch that changes refs.
  Do not start/stop coturn or any process; do not bind ports; do not run the installer or vast_onstart.sh itself.
- Never print or copy secret values. For /workspace/.musetalk-runtime.env, /workspace/MuseTalk/.env.webrtc-turn.local,
  /workspace/.lingua-control-plane.env and ~/.aws/*: key names only. Do not try to recover the redacted keys.
- /workspace/MuseTalk/character_factory is user-owned and being edited by another pipeline: read only if needed.
  Never touch /dev/shm/soulx-lfs-state-20260919.
- Cite evidence as path:line for every claim. Say "unknown" when you cannot establish something; do not guess.

OUTPUT
Return checks, each with item, status (ok | gap | risk | blocker | info), detail, evidence, and fix (what would
close a gap WITHOUT changing the template, if anything; say "needs a template change" only if nothing in the repo,
the AWS secret or S3 could do it).`

const AUDITS = [
  { key: 'template_trace', prompt: `${CONTEXT}

YOUR DIMENSION: trace the template end to end against the repo at HEAD for three hosts: an RTX 3090 (sm_86, 24 GB),
an RTX 4070 SUPER (sm_89, 12 GB) and a Tesla T4 (sm_75). Cover, with path:line evidence:
1. The template itself: depth-1 clone of main; rm -rf of /workspace/MuseTalk on EVERY container start (what is lost:
   models/ weights, .runtime/ state incl. trt_artifacts stamps, restored engines, results/ avatars, TURN env);
   lock and ERR trap behavior.
2. scripts/vast_onstart.sh main() with the env the template sets (SETUP_CLEAN=1, SETUP_FULL_STACK=1,
   STARTUP_TIMEOUT_SECONDS=1800, PROFILE=throughput_record, PORT=8000, MUSETALK_AWS_SECRET_ID/REGION,
   MUSETALK_SECRETS_STRICT=true, MUSETALK_SECRETS_VERIFY_S3=1, bootstrap AWS_* creds):
   - the install step: exact installer flags onstart passes; what install_musetalk.sh --clean does (does it delete
     /workspace/.venvs/musetalk_trt_stagewise? re-download weights?), and roughly how long/how many GB;
   - post-setup validation; the secrets bootstrap: which variables it exports, whether the runtime user's AWS keys
     replace the template's bootstrap keys in the environment, and which credentials scripts/trt_artifact_bundle.py
     then uses for the r5 restore (it builds its S3 client from env);
   - refresh_recipe_after_secrets; TURN autogen (which Vast-provided env it reads; what happens if absent);
   - restore_r5_bundle per GPU (which candidates fit; URI from TRT_ARTIFACT_S3_BUCKET; disk needed; what happens
     while the ampere-plus object is missing from S3, given MUSETALK_R5_BUNDLE_RESTORE defaults to required);
   - provision_engines (.ts provisioning off for r5); vast_server_ctl.sh start (PORT, STARTUP_TIMEOUT_SECONDS, log
     dir, the verify-log expectation after /health).
3. Restarts: onstart runs on every container start; list what is redone each time (clean venv install, weights,
   bundle download) with sizes/time where the code or docs state them.
4. What differs if a user boots now, before the push (origin/main 49e58d4: default recipe fast, .ts build) and
   before the ampere-plus upload.
Conclude per GPU: is r5 served, with which bundle, and what blocks it.` },
  { key: 'runtime_inventory', prompt: `${CONTEXT}

YOUR DIMENSION: inventory every runtime dependency of (1) the live server: api_server.py and what it imports under
scripts/ and musetalk/, serving WebRTC calls with recipe r5, and (2) avatar creation: POST /avatars/prepare (DWPose,
face detection, face parsing, SD-VAE encode, masks, S3 upload). Find each file, directory or absolute path they open:
models/* (musetalkV15, sd-vae, whisper, taesd, dwpose, face-parse-bisent, s3fd, syncnet, ...), configs/*, assets,
idle/pose clips, fonts, results/ (avatars), .runtime/* (native VP8, resolved env), Kokoro voices/cache, and any
path outside the repo (e.g. /workspace/SoulX-FlashHead, /workspace/experiments, /workspace/.venvs,
/workspace/run-musetalk-local-trt.sh). For each: tracked in git (git ls-files), produced by scripts/install_musetalk.sh
or download_weights.sh for the template's install groups (read vast_onstart.sh for the flags; is Kokoro on by
default?), restored from S3 (engine bundle at boot; avatars lazily via scripts/avatar_s3_store.py), generated at boot,
or LOCAL-ONLY on this machine (a gap for Vast). Compare with what exists here (ls models/, .runtime/, results/).
Only flag experiment leftovers (e.g. models/tensorrt_unet_sm89_bs8_local, .runtime/musetalk_trt_local_sm89.env) if
the live server or the prepare path actually reads them under recipe r5. Also decide whether the character_factory
avatar-creation pipeline (Codex + H3 MiniMax, ComfyUI venv /workspace/.venvs/comfy-h3) is part of the MuseTalk
worker's runtime or a separate tool whose outputs reach workers through S3 (evidence).` },
  { key: 'venv_parity', prompt: `${CONTEXT}

YOUR DIMENSION: compare this machine's serving venv /workspace/.venvs/musetalk_trt_stagewise (list
site-packages/*.dist-info names and versions; do not import heavy modules) with what a fresh
install_musetalk.sh --clean installs for the template (read scripts/vast_onstart.sh for the exact installer flags
SETUP_FULL_STACK=1 implies; confirm the matrix auto rule a 3090 and a 4070 SUPER get), resolved from
requirements/*.in against requirements/constraints-<matrix>.txt. Report:
1. distributions present here but not in a fresh install, and whether serving or avatar-prep code imports any of
   them (grep for imports; include lazy imports inside functions);
2. version mismatches between this venv and the constraints for packages r5 depends on: torch 2.5.1+cu121,
   tensorrt-cu12 10.3.0 (+ -libs/-bindings), torch_tensorrt 2.5.0, onnx 1.17.0, diffusers, transformers, aiortc, av,
   numpy, opencv, and the CUDA runtime wheels;
3. whether the TAESD TensorRT engine key (scripts/vae_fast_decoder.py taesd_trt_fingerprint hashes the ONNX that
   torch.onnx.export produces from models/taesd at startup) would reproduce on a fresh install: which package
   versions the export depends on and whether they are pinned; note the bundle's key is 512bfd629a5e1f4f2e40
   (ampere-plus) / 6111388248264a4ef2ae (rtx4070super);
4. the install-time GPU self-test and import smoke (scripts/musetalk_selftest.py and install_musetalk.sh): anything
   that could fail or take long on a 3090 or a 4070 SUPER.` },
  { key: 'secrets_controlplane', prompt: `${CONTEXT}

YOUR DIMENSION: what a Vast worker needs from the AWS secret lingua/musetalk-worker-runtime and from IAM, given the
template. Read docs/musetalk_worker_secrets.md, scripts/bootstrap_aws_secrets.py, scripts/vast_onstart.sh
(bootstrap_runtime_secrets, refresh_recipe_after_secrets), scripts/worker_control_plane.py, api_server.py
(registration, heartbeats, capacity), scripts/avatar_s3_store.py, scripts/trt_artifact_bundle.py (S3 client region,
bucket, credentials). Report:
1. keys the secret must contain for r5 serving + avatar prepare + control-plane registration (S3 bucket/region,
   runtime AWS keys, LINGUA_* registration, LINGUA_WORKER_DEFAULT_CAPACITY, TURN-related if any), which the
   template provides itself, and which keys this box's /workspace/.musetalk-runtime.env and
   /workspace/.lingua-control-plane.env contain (KEY NAMES ONLY);
2. IAM: what the bootstrap user needs (only secretsmanager:GetSecretValue?) and what the runtime user needs
   (s3:GetObject on trt-artifacts/* covering trt-artifacts/ampere-plus/...; avatars read AND write for prepare);
3. a live metadata check with the runtime credentials: load /workspace/.musetalk-runtime.env into the environment
   of a python subprocess (never print values) and call head_object on (a) the rtx4070super bundle key and (b) the
   ampere-plus bundle key, both from configs/trt_bundles/*.json, bucket from TRT_ARTIFACT_S3_BUCKET; print only
   status codes / ContentLength / whether metadata sha256 matches the descriptor. Expect (b) to be 404 until upload;
4. capacity: what the control plane does with LINGUA_WORKER_DEFAULT_CAPACITY (code), the value the docs' example
   shows, and what docs/fps_comparisons/live15_r5_20260929/README.md validated (10 calls pass, 15 at the knee, on
   the 4070 SUPER bundle). Never print the configured value; report only whether the key is present locally.` },
]

const FINDINGS = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    checks: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          item: { type: 'string' },
          status: { type: 'string', enum: ['ok', 'gap', 'risk', 'blocker', 'info'] },
          detail: { type: 'string' },
          evidence: { type: 'string' },
          fix: { type: 'string' },
        },
        required: ['item', 'status', 'detail', 'evidence'],
      },
    },
  },
  required: ['summary', 'checks'],
}

const VERDICTS = {
  type: 'object',
  properties: {
    verdicts: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          item: { type: 'string' },
          verdict: { type: 'string', enum: ['confirmed', 'refuted', 'uncertain'] },
          corrected_status: { type: 'string', enum: ['ok', 'gap', 'risk', 'blocker', 'info'] },
          reason: { type: 'string' },
          evidence: { type: 'string' },
        },
        required: ['item', 'verdict', 'corrected_status', 'reason', 'evidence'],
      },
    },
    missed: { type: 'array', items: { type: 'string' } },
  },
  required: ['verdicts'],
}

const results = await pipeline(
  AUDITS,
  (a) => agent(a.prompt, { label: `audit:${a.key}`, phase: 'Audit', schema: FINDINGS }),
  async (audit, a) => {
    if (!audit) return { key: a.key, audit: null, verification: null }
    const flagged = audit.checks.filter((c) => c.status !== 'ok' && c.status !== 'info')
    const oks = audit.checks.filter((c) => c.status === 'ok')
    log(`${a.key}: ${audit.checks.length} checks, ${flagged.length} flagged for verification`)
    const vprompt = `${CONTEXT}

YOU ARE AN ADVERSARIAL VERIFIER for the audit dimension "${a.key}". Below are the auditor's findings.
For EACH finding with status gap, risk or blocker: try hard to REFUTE it from the code and the machine state. Re-read
the cited lines and the code paths around them; look for handling, defaults, fallbacks or later steps the auditor
missed; check claims about sizes and timings against what the code/docs actually say. Verdict: refuted (the claim is
wrong or overstated), confirmed (it holds, with your own evidence), or uncertain (cannot be established either way).
Give the corrected status. Then look at the "ok" findings and flag up to 3 that you believe are actually wrong (add
them as verdicts too). Finally list in "missed" any serious problem in this dimension the auditor did not report
(max 5, each with path:line evidence).

FLAGGED FINDINGS:
${JSON.stringify(flagged, null, 1)}

OK FINDINGS (spot-check):
${JSON.stringify(oks.map((c) => ({ item: c.item, detail: c.detail, evidence: c.evidence })), null, 1)}`
    const verification = await agent(vprompt, { label: `verify:${a.key}`, phase: 'Verify', schema: VERDICTS })
    return { key: a.key, audit, verification }
  },
)
return { results }
