#!/usr/bin/env bash
# box_job.sh: distil the bk_tiny MuseTalk UNet on a rented RTX 3090, review it on video, leave one results.tar.
#
# Why: 25 fps on the iPhone 12 Pro needs a smaller UNet, and bk_tiny is fast enough untrained
# (ios_coreml_implementation.md 5.8). Distilling it from the full UNet takes thousands of CUDA steps; the Mac did
# not finish 30 in 8 minutes. drive.sh rents the box through the EC2 controller (stock Vast template + the locked
# bootstrap at 4e2b02e). That bootstrap already installed what training needs, and this job reuses it:
# the locked venv /workspace/.venvs/musetalk_locked_<contract artifact id> (torch cu121, diffusers 0.30.2) and
# /workspace/MuseTalk/models. The training checkout is a separate clone so the bootstrap's tree stays untouched.
#
# Steps. A step whose marker in state/ matches is skipped, so re-running the job resumes it:
#   boot      wait until the bootstrap has exited, then hold its on-start lock so that no re-run of it can start
#             the server again in the middle of the job
#   stop      stop the bootstrap's :8000 API server: its own vast_server_ctl.sh stop, then only api_server.py
#             processes running from /workspace/MuseTalk; nothing else on the box is touched
#   clone     AhmadAFS1/MuseTalk at DISTILL_SHA into MuseTalk/ (blob-filtered; generated/, assets/,
#             character_factory/, third_party_wheels/ and uploads/ left out)
#   deps      import both scripts with the locked venv; pip-install only the missing modules, into overlay/ (the
#             venv stays as installed). Check that this diffusers builds a UNet with mid_block_type=None (bk_small
#             and bk_tiny need it; 0.30 has it). Install a newer diffusers into overlay/ only if that check fails.
#   models    link musetalkV15, whisper, taesd (+ sd-vae) from the bootstrap; fetch from public Hugging Face repos only
#             what is missing (unet.pth is missing unless the box was created with MUSETALK_SKIP_EAGER_UNET=0)
#   inputs    gate .bin files into the clone's HarnessBundle; downloads from inputs/urls.tsv, one
#             "<dest path under inputs/>\t<presigned GET>" per line: a phone bundle .tar keeps only latents_f16.bin and
#             avatar.json (checked against its .json sidecar when the list has one), a v15 avatar .tar.gz is
#             extracted whole. The URL list is deleted after the downloads, whether they succeeded or not, and no URL
#             is ever logged.
#   facemesh  a small venv with mediapipe 0.10.14 for the review's FaceMesh lip aperture (--facemesh-python)
#   train     scripts/ios/distill_small_unet.py --resume (its <out>/ckpt.pt). Its --max-minutes counts training time
#             across resumes, so it gets DISTILL_MAX_MINUTES, or less when the lease must keep DISTILL_RESERVE_MINUTES
#             for the review, the pull and cleanup; `timeout` is the backstop.
#   review    scripts/ios/review_distilled_unet.py on the review avatars x the held-out clips
#   package   results.tar: job.log, STATUS, job.env, out/ (log.jsonl, train/review logs, best/last weights, json,
#             review summary and videos; not ckpt.pt's optimizer state nor the Whisper window cache) and anything the review wrote under the clone's experiments/video_validation
#             or docs/ios_unet_bench. Built after a failure too.
# Flags are passed to the two Python scripts only if their --help lists them, because both scripts are still changing.
#
# Files in DISTILL_ROOT (/workspace/distill): job.log (everything), STATUS (one line: the current state),
# DONE ("ok" or "failed <step> rc=<n>"; written once results.tar is final), results.tar + results.tar.sha256,
# job.env (settings, pushed by drive.sh; the caller's environment wins), lease_deadline_utc (from drive.sh), out/, state/.
#
# Settings (job.env or environment; defaults):
#   DISTILL_SHA (required, 40 hex)    DISTILL_VARIANT=bk_tiny          DISTILL_MAX_MINUTES=150
#   DISTILL_STEPS=100000 (the time cap ends the run first; the LR cosine follows whichever ends first)
#   DISTILL_RESERVE_MINUTES=60        DISTILL_EXTRA_ARGS (appended, word-split, e.g. "--batch 16 --val-count 4")
#   DISTILL_TRAIN_GLOB=$ROOT/inputs/bundles/*/latents_f16.bin (comma-separated globs; add $ROOT/inputs/avatars/*/latents.pt
#                                     to train on the review avatars' server caches too)
#   REVIEW_AVATARS (comma ids; empty: every extracted avatar)        REVIEW_SECONDS=10   REVIEW_MAX_MINUTES=40
#   REVIEW_EXTRA_ARGS                 FACEMESH=1                       BOOT_WAIT_MINUTES=40
#   RESULTS_INCLUDE_CKPT=0 (1 also packs the optimizer checkpoint)   FORCE=0 (1 re-runs a job whose DONE says ok)
# Local test only: DISTILL_ROOT, DISTILL_BOOT_REPO, DISTILL_CLONE, BOX_PY.
#
# Usage on the box (drive.sh start does this):  setsid -f nohup bash /workspace/distill/box_job.sh
# Locally:  bash box_job.sh --dry-run   (prints the commands; runs only read-only probes such as the scripts' --help)
set -Eeuo pipefail
umask 077

DRY=0
case "${1:-}" in
  --dry-run) DRY=1 ;;
  "") ;;
  *) echo "usage: box_job.sh [--dry-run]" >&2; exit 2 ;;
esac

ROOT="${DISTILL_ROOT:-/workspace/distill}"
BOOT_REPO="${DISTILL_BOOT_REPO:-/workspace/MuseTalk}"
CLONE="${DISTILL_CLONE:-$ROOT/MuseTalk}"
REPO_URL=https://github.com/AhmadAFS1/MuseTalk.git
INPUTS="$ROOT/inputs"
OUT="$ROOT/out"
STATE="$ROOT/state"
OVERLAY="$ROOT/overlay"
LOG="$ROOT/job.log"
ONSTART_LOCK=/tmp/musetalk-onstart.lock
CONTRACT=configs/dependencies/rtx3090-cp310-cu121-screening.json
# download_weights.sh's pin, so a TAESD fetched here is the one the bootstrap would have fetched.
TAESD_REPO=madebyollin/taesd
TAESD_REV=614f76814bbe30edbe2e627ace1c2234c81a2c0e
GATE_FILES="unet_latents_f16.bin:2097152 unet_audio_f16.bin:4915200 unet_ref_f16.bin:1048576"
SPARSE_PATTERNS='/*
!/generated/
!/assets/
!/character_factory/
!/third_party_wheels/
!/uploads/'
JOB_KEYS="RUN_LABEL DISTILL_SHA DISTILL_VARIANT DISTILL_MAX_MINUTES DISTILL_STEPS DISTILL_RESERVE_MINUTES DISTILL_EXTRA_ARGS
DISTILL_TRAIN_GLOB REVIEW_AVATARS REVIEW_SECONDS REVIEW_MAX_MINUTES REVIEW_EXTRA_ARGS FACEMESH BOOT_WAIT_MINUTES
RESULTS_INCLUDE_CKPT DIFFUSERS_FALLBACK"
# import name -> pip requirement, for modules the scripts may need that the server venv might lack
PIP_NAMES="cv2=opencv-python-headless PIL=pillow skimage=scikit-image sklearn=scikit-learn yaml=pyyaml einops=einops
librosa=librosa soundfile=soundfile scipy=scipy imageio=imageio imageio_ffmpeg=imageio-ffmpeg tqdm=tqdm
omegaconf=omegaconf lpips=lpips av=av coremltools=coremltools==9.0"
CURRENT=setup
LOCKED=0
MP_PY=""

log() { printf '[box_job] [%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "ERROR: $*"; exit 2; }
# In a dry run a failed precondition is reported and the walk goes on, so every step's commands get printed.
soft_die() { if (( DRY )); then log "(dry-run) would stop here: $*"; else die "$*"; fi; }
run() { log "+ $(printf '%q ' "$@")"; if (( ! DRY )); then "$@"; fi; }
redact() { sed -E 's#https?://[^[:space:]"]+#[URL]#g'; }
status() {
  log "STATUS $*"
  if (( ! DRY )); then printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" > "$ROOT/STATUS.tmp" && mv -f "$ROOT/STATUS.tmp" "$ROOT/STATUS"; fi
}
marked() { [[ -f $STATE/$1.done && "$(cat "$STATE/$1.done")" == "$2" ]]; }
mark() { if (( ! DRY )); then printf '%s\n' "$2" > "$STATE/$1.done"; fi; }
# The --help text of a script, and whether it lists a flag / gives the flag a value.
has_flag() { grep -qE -- "(^|[^A-Za-z0-9-])$2([^A-Za-z0-9-]|\$)" <<<"$1"; }
takes_value() { grep -qE -- "\[?$2[ =][A-Z_]+" <<<"$1"; }
epoch_of() { date -u -d "$1" +%s 2>/dev/null || date -u -j -f '%Y-%m-%dT%H:%M:%SZ' "$1" +%s 2>/dev/null; }
count() { local n=0 f; for f in "$@"; do if [[ -e $f ]]; then n=$((n + 1)); fi; done; echo "$n"; }

load_job_env() {
  local f="$ROOT/job.env" line key value re='^([A-Z][A-Z0-9_]*)=([A-Za-z0-9._,:/=+* -]*)$'
  [[ -f $f ]] || return 0
  # Parsed, not sourced: only known keys and plain values, so the file cannot run code.
  while IFS= read -r line || [[ -n $line ]]; do
    [[ -z $line || $line == \#* ]] && continue
    [[ $line =~ $re ]] || die "job.env: unreadable line for ${line%%=*}"
    key=${BASH_REMATCH[1]} value=${BASH_REMATCH[2]}
    case " $(tr '\n' ' ' <<<"$JOB_KEYS") " in *" $key "*) ;; *) die "job.env: unknown key $key" ;; esac
    if [[ -z ${!key+x} ]]; then export "$key=$value"; fi
  done < "$f"
}

settings() {
  load_job_env
  : "${DISTILL_SHA:=}" "${DISTILL_VARIANT:=bk_tiny}" "${DISTILL_MAX_MINUTES:=150}" "${DISTILL_STEPS:=100000}"
  : "${DISTILL_RESERVE_MINUTES:=60}" "${DISTILL_EXTRA_ARGS:=}" "${DISTILL_TRAIN_GLOB:=$INPUTS/bundles/*/latents_f16.bin}"
  : "${REVIEW_AVATARS:=}" "${REVIEW_SECONDS:=10}" "${REVIEW_MAX_MINUTES:=40}" "${REVIEW_EXTRA_ARGS:=}"
  : "${FACEMESH:=1}" "${BOOT_WAIT_MINUTES:=40}" "${RESULTS_INCLUDE_CKPT:=0}" "${DIFFUSERS_FALLBACK:=diffusers==0.31.0}"
  : "${FORCE:=0}"
  [[ $DISTILL_SHA =~ ^[a-f0-9]{40}$ ]] || die "DISTILL_SHA must be a full 40-hex commit (job.env)"
  case $DISTILL_VARIANT in bk_base|bk_small|bk_tiny) ;; *) die "DISTILL_VARIANT must be bk_base, bk_small or bk_tiny" ;; esac
  local n
  for n in DISTILL_MAX_MINUTES DISTILL_STEPS DISTILL_RESERVE_MINUTES REVIEW_MAX_MINUTES BOOT_WAIT_MINUTES; do
    [[ ${!n} =~ ^[0-9]+$ ]] || die "$n must be a whole number"
  done
  log "settings: sha=$DISTILL_SHA variant=$DISTILL_VARIANT max_minutes=$DISTILL_MAX_MINUTES steps=$DISTILL_STEPS" \
      "reserve=$DISTILL_RESERVE_MINUTES review_avatars=${REVIEW_AVATARS:-all} facemesh=$FACEMESH dry_run=$DRY"
}

# ---------------------------------------------------------------------------------------------------------- steps

step_boot() {
  CURRENT=boot
  status "running boot: waiting for the bootstrap to exit"
  if (( DRY )); then log "+ exec 7>$ONSTART_LOCK; flock -w $(( BOOT_WAIT_MINUTES * 60 )) 7  (held until the job exits)"; return 0; fi
  # The bootstrap holds this lock (fd 9) for its whole run, and its daemons do not inherit it. Holding it here makes
  # any later on-start attempt exit at once ("Another bootstrap owns the lock") instead of restarting the server.
  exec 7>"$ONSTART_LOCK"
  flock -w $(( BOOT_WAIT_MINUTES * 60 )) 7 || die "the bootstrap is still running after $BOOT_WAIT_MINUTES min"
  log "bootstrap exited; its last line: $(grep '^\[bootstrap\]' /workspace/bootstrap.log 2>/dev/null | tail -n 1 | cut -c1-200)"
}

step_stop_server() {
  CURRENT=stop
  status "running stop: the bootstrap's :8000 server"
  local ctl="$BOOT_REPO/scripts/vast_server_ctl.sh" sig pid cwd cmd
  if [[ -f $ctl ]]; then
    run env DRAIN_TIMEOUT_SECONDS=20 bash "$ctl" stop || log "vast_server_ctl.sh stop failed; checking processes"
  fi
  if (( DRY )); then log "+ (then TERM/KILL only api_server.py processes whose cwd or command line is under $BOOT_REPO)"; return 0; fi
  # Whatever the PID file missed: only the API server chain of the bootstrap's checkout.
  for sig in TERM KILL; do
    for pid in $(pgrep -f 'api_server\.py' || true); do
      cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || true)
      cmd=$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)
      [[ $cwd == "$BOOT_REPO" || $cwd == "$BOOT_REPO/"* || $cmd == *"$BOOT_REPO/"* ]] || continue
      log "stopping bootstrap API process $pid ($sig)"
      kill -s "$sig" "$pid" 2>/dev/null || true
    done
    sleep 5
  done
  if (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null; then
    die "something still accepts connections on :8000 after the server stop; not training on a shared GPU"
  fi
  # One query before training starts; never while the GPU job runs.
  log "GPU after the stop: $(nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || echo unknown)"
}

step_python() {
  CURRENT=python
  local artifact=""
  if [[ -z ${BOX_PY:-} ]]; then
    artifact=$(python3 -I -c 'import json,re,sys; v=json.load(open(sys.argv[1]))["artifact_id"]; assert re.fullmatch("[a-f0-9]{64}", v); print(v)' \
      "$BOOT_REPO/$CONTRACT" 2>/dev/null || true)
    BOX_PY="/workspace/.venvs/musetalk_locked_${artifact}/bin/python"
    if [[ -z $artifact || ! -x $BOX_PY ]]; then
      BOX_PY=$(ls -1dt /workspace/.venvs/musetalk_locked_*/bin/python 2>/dev/null | head -n 1 || true)
    fi
  fi
  [[ -n ${BOX_PY:-} && -x $BOX_PY ]] || soft_die "no locked venv python (did the bootstrap's install finish?)"
  export PYTHONPATH="$OVERLAY${PYTHONPATH:+:$PYTHONPATH}" PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_TELEMETRY=1
  log "python: ${BOX_PY:-none} (overlay $OVERLAY)"
  if (( ! DRY )); then
    "$BOX_PY" -c 'import torch; assert torch.cuda.is_available(), "no CUDA"; print("torch", torch.__version__, torch.cuda.get_device_name(0))' \
      || die "the locked venv has no working CUDA torch"
  fi
}

step_clone() {
  CURRENT=clone
  if [[ -d $CLONE/.git ]] && [[ "$(git -C "$CLONE" rev-parse HEAD 2>/dev/null)" == "$DISTILL_SHA" ]]; then
    log "clone: $CLONE already at $DISTILL_SHA"
    return 0
  fi
  status "running clone: $DISTILL_SHA"
  if [[ -d $CLONE/.git ]]; then
    { git -C "$CLONE" diff --quiet && git -C "$CLONE" diff --cached --quiet; } || soft_die "tracked edits in $CLONE; left unchanged"
    run git -C "$CLONE" -c core.hooksPath=/dev/null fetch --depth 1 --filter=blob:none origin "$DISTILL_SHA"
    run git -C "$CLONE" -c core.hooksPath=/dev/null checkout --detach "$DISTILL_SHA"
  else
    local stage="$CLONE.stage"
    run rm -rf "$stage"
    run git init -q "$stage"
    run git -C "$stage" remote add origin "$REPO_URL"
    run git -C "$stage" config core.sparseCheckout true
    run git -C "$stage" config core.sparseCheckoutCone false
    log "+ sparse-checkout patterns: $(tr '\n' ' ' <<<"$SPARSE_PATTERNS")"
    if (( ! DRY )); then printf '%s\n' "$SPARSE_PATTERNS" > "$stage/.git/info/sparse-checkout"; fi
    run git -C "$stage" -c core.hooksPath=/dev/null fetch --depth 1 --filter=blob:none origin "$DISTILL_SHA"
    run git -C "$stage" -c core.hooksPath=/dev/null checkout --detach "$DISTILL_SHA"
    run mv -T "$stage" "$CLONE"
  fi
  if (( ! DRY )); then
    [[ "$(git -C "$CLONE" rev-parse HEAD)" == "$DISTILL_SHA" ]] || die "clone HEAD differs from $DISTILL_SHA"
  fi
}

# Prints the import names the two scripts (and the modules they import lazily) cannot find.
PY_IMPORT_PROBE='
import importlib, importlib.util, sys
root = sys.argv[1]
sys.path[:0] = [root, root + "/scripts/ios", root + "/scripts"]
missing = {n for n in ("numpy", "torch", "cv2", "diffusers", "transformers", "librosa", "PIL", "einops")
           if importlib.util.find_spec(n) is None}
for script in ("distill_small_unet", "review_distilled_unet"):
    if importlib.util.find_spec(script) is not None:
        try:
            importlib.import_module(script)
        except ModuleNotFoundError as e:
            missing.add((e.name or "?").split(".")[0])
print(" ".join(sorted(missing)))
'

PY_MID_BLOCK_CHECK='
import torch, diffusers
from diffusers import UNet2DConditionModel
m = UNet2DConditionModel(sample_size=8, in_channels=4, out_channels=4, layers_per_block=1, block_out_channels=(32, 64),
                         down_block_types=("CrossAttnDownBlock2D", "DownBlock2D"), up_block_types=("UpBlock2D", "CrossAttnUpBlock2D"),
                         mid_block_type=None, cross_attention_dim=16, attention_head_dim=8, norm_num_groups=8)
assert m.mid_block is None, "mid_block_type=None was not honoured"
with torch.no_grad():
    y = m(torch.randn(1, 4, 8, 8), 0, encoder_hidden_states=torch.randn(1, 3, 16)).sample
assert tuple(y.shape) == (1, 4, 8, 8)
print("diffusers", diffusers.__version__, "builds mid_block_type=None")
'

pip_overlay() {
  # The uv-built locked venv may have no pip: fall back to the image's /venv/main pip (same CPython 3.10), into overlay/.
  local pip=()
  if "$BOX_PY" -m pip --version >/dev/null 2>&1; then pip=("$BOX_PY" -m pip)
  elif /venv/main/bin/python -m pip --version >/dev/null 2>&1; then pip=(/venv/main/bin/python -m pip)
  elif command -v uv >/dev/null 2>&1; then pip=(uv pip --python "$BOX_PY")
  else soft_die "no pip for the overlay install"; return 0
  fi
  run "${pip[@]}" install --disable-pip-version-check --no-deps --target "$OVERLAY" "$@"
}

step_deps() {
  CURRENT=deps
  status "running deps"
  local missing="" previous="" name spec specs i
  run mkdir -p "$OVERLAY"
  for i in 1 2 3 4 5 6; do
    missing=$("$BOX_PY" -c "$PY_IMPORT_PROBE" "$CLONE" 2>>"$OUT/deps.log") || soft_die "importing the scripts failed (see out/deps.log)"
    if [[ -z $missing ]]; then log "deps: both scripts and their lazy imports resolve"; break; fi
    [[ $missing != "$previous" ]] || soft_die "still missing after the overlay install: $missing"
    log "missing modules: $missing"
    specs=()
    for name in $missing; do
      spec=$(tr ' ' '\n' <<<"$PIP_NAMES" | sed -n "s/^$name=//p")
      [[ -n $spec ]] || soft_die "missing module $name is not a known pip package; add it to PIP_NAMES"
      [[ -n $spec ]] && specs+=("$spec")
    done
    (( ${#specs[@]} )) && pip_overlay "${specs[@]}"
    previous=$missing
    if (( DRY )); then break; fi
  done
  if ! "$BOX_PY" -c "$PY_MID_BLOCK_CHECK" 2>>"$OUT/deps.log"; then
    log "this diffusers cannot build mid_block_type=None; overlaying $DIFFUSERS_FALLBACK"
    pip_overlay "$DIFFUSERS_FALLBACK"
    if (( ! DRY )); then "$BOX_PY" -c "$PY_MID_BLOCK_CHECK" || die "mid_block_type=None still fails with $DIFFUSERS_FALLBACK"; fi
  fi
  if ! command -v ffmpeg >/dev/null 2>&1; then
    log "ffmpeg missing (the review writes mp4s)"
    run apt-get update -y
    run env DEBIAN_FRONTEND=noninteractive apt-get install -y ffmpeg
  fi
}

hf_get() {  # repo file revision local_dir (public repos only)
  log "+ hf_hub_download $1 $2${3:+ @$3} -> $4"
  if (( DRY )); then return 0; fi
  "$BOX_PY" -c 'import sys; from huggingface_hub import hf_hub_download
print(hf_hub_download(repo_id=sys.argv[1], filename=sys.argv[2], revision=sys.argv[3] or None, local_dir=sys.argv[4]))' "$@"
}

link_dir() {  # target link
  if [[ -L $2 && "$(readlink "$2")" == "$1" ]]; then return 0; fi
  [[ ! -e $2 || -L $2 ]] || soft_die "$2 exists and is not a link; left unchanged"
  run ln -sfn "$1" "$2"
}

step_models() {
  CURRENT=models
  status "running models"
  local src="$BOOT_REPO/models" own="$ROOT/models" dst="$CLONE/models" f
  run mkdir -p "$dst" "$own"
  # The teacher needs unet.pth. r5 workers skip it unless created with MUSETALK_SKIP_EAGER_UNET=0 (drive.sh does that).
  if [[ -s $src/musetalkV15/unet.pth && -s $src/musetalkV15/musetalk.json ]]; then
    link_dir "$src/musetalkV15" "$dst/musetalkV15"
  else
    log "models: no unet.pth from the bootstrap; fetching musetalkV15 from TMElyralab/MuseTalk"
    for f in musetalkV15/musetalk.json musetalkV15/unet.pth; do
      [[ -s $own/$f ]] || hf_get TMElyralab/MuseTalk "$f" "" "$own"
    done
    link_dir "$own/musetalkV15" "$dst/musetalkV15"
  fi
  if [[ -s $src/whisper/config.json && -s $src/whisper/pytorch_model.bin && -s $src/whisper/preprocessor_config.json ]]; then
    link_dir "$src/whisper" "$dst/whisper"
  else
    for f in config.json pytorch_model.bin preprocessor_config.json; do
      [[ -s $own/whisper/$f ]] || hf_get openai/whisper-tiny "$f" "" "$own/whisper"
    done
    link_dir "$own/whisper" "$dst/whisper"
  fi
  # render_evidence.taesd_torch loads <repo>/models/taesd with AutoencoderTiny.from_pretrained.
  if [[ -s $src/taesd/config.json && -s $src/taesd/diffusion_pytorch_model.safetensors ]]; then
    link_dir "$src/taesd" "$dst/taesd"
  else
    for f in config.json diffusion_pytorch_model.safetensors; do
      [[ -s $own/taesd/$f ]] || hf_get "$TAESD_REPO" "$f" "$TAESD_REV" "$own/taesd"
    done
    link_dir "$own/taesd" "$dst/taesd"
  fi
  if [[ -d $src/sd-vae ]]; then link_dir "$src/sd-vae" "$dst/sd-vae"; fi
}

# Extracts one downloaded archive (stdlib only, run with -I: the archives are data, not code).
#   latents <bundle.tar> <out dir> <sidecar.json>   only latents_f16.bin + avatar.json; sidecar bytes/sha256 checked
#   full <avatar.tar.gz> <parent dir> <avatar id>   everything, under <parent>/<avatar id>/, which must hold latents.pt
PY_EXTRACT='
import hashlib, json, shutil, sys, tarfile
from pathlib import Path, PurePosixPath

FRAME = 8 * 32 * 32 * 2  # one [8, 32, 32] fp16 latent

def parts(member):
    p = PurePosixPath(member.name)
    if p.is_absolute() or any(x in ("", ".", "..") for x in p.parts):
        sys.exit("unsafe member name in archive")
    return p.parts

def copy(tar, member, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    src = tar.extractfile(member)
    with src, open(dest, "wb") as f:
        shutil.copyfileobj(src, f)

mode, archive = sys.argv[1], Path(sys.argv[2])
if mode == "latents":
    out, sidecar = Path(sys.argv[3]), Path(sys.argv[4])
    if sidecar.is_file():
        meta = json.loads(sidecar.read_text())
        h = hashlib.sha256()
        with archive.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        if archive.stat().st_size != int(meta["bytes"]) or h.hexdigest() != meta["sha256"]:
            sys.exit("bundle differs from its sidecar (bytes or sha256)")
    tmp = out.with_name(out.name + ".part")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    with tarfile.open(archive, "r:") as tar:
        for m in tar:
            p = parts(m)
            if m.isfile() and len(p) == 1 and p[0] in ("latents_f16.bin", "avatar.json"):
                copy(tar, m, tmp / p[0])
    lat = tmp / "latents_f16.bin"
    size = lat.stat().st_size if lat.is_file() else 0
    if size == 0 or size % FRAME:
        sys.exit("bundle has no [N, 8, 32, 32] fp16 latents_f16.bin")
    if (tmp / "avatar.json").is_file():
        shape = json.loads((tmp / "avatar.json").read_text()).get("latents", {}).get("shape")
        if shape and list(shape) != [size // FRAME, 8, 32, 32]:
            sys.exit("latents_f16.bin does not match the shape in avatar.json")
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    print(f"{size // FRAME} latent frames")
elif mode == "full":
    parent, stem = Path(sys.argv[3]), sys.argv[4]
    stage = parent / f".{stem}.stage"
    shutil.rmtree(stage, ignore_errors=True)
    stage.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as tar:
        for m in tar:
            p = parts(m)
            if p[0] != stem:
                sys.exit("archive root is not the avatar id")
            if m.isdir():
                stage.joinpath(*p).mkdir(parents=True, exist_ok=True)
            elif m.isfile():
                copy(tar, m, stage.joinpath(*p))
            else:
                sys.exit("archive holds a link or a special file")
    if not (stage / stem / "latents.pt").is_file():
        sys.exit("avatar archive has no latents.pt")
    shutil.rmtree(parent / stem, ignore_errors=True)
    (stage / stem).rename(parent / stem)
    shutil.rmtree(stage)
    print(sum(1 for _ in (parent / stem).rglob("*")), "entries")
else:
    sys.exit("unknown mode")
'

fetch_one() {  # dest url; never logs the URL. Called in a || list, so errexit is off here: every step checks itself.
  local dest=$1 url=$2 path="$INPUTS/$1" kind=file dir stem
  case $dest in *.tar.gz) kind=avatar ;; *.tar) kind=bundle ;; esac
  dir=$(dirname "$path")
  stem=$(basename "$path")
  stem=${stem%.tar.gz}
  stem=${stem%.tar}
  case $kind in
    bundle) if [[ -s $dir/$stem/latents_f16.bin ]]; then log "have $dest"; return 0; fi ;;
    avatar) if [[ -f $dir/$stem/.extracted ]]; then log "have $dest"; return 0; fi ;;
    *) if [[ -s $path ]]; then log "have $dest"; return 0; fi ;;
  esac
  if (( DRY )); then log "+ curl -K - (URL on stdin, never logged) -o $path.part, then keep as $kind"; return 0; fi
  mkdir -p "$dir" || return 1
  # The URL goes to curl on stdin as a config line: not in argv (ps), not in the log.
  printf 'url = "%s"\n' "$url" \
    | curl -fsS --retry 5 --retry-delay 3 --retry-all-errors --connect-timeout 20 --max-time 3600 -K - -o "$path.part" \
      2> >(redact >&2) || { rm -f "$path.part"; return 1; }
  mv -f "$path.part" "$path" || return 1
  case $kind in
    bundle) python3 -I -c "$PY_EXTRACT" latents "$path" "$dir/$stem" "$dir/$stem.json" || return 1; rm -f "$path" ;;
    avatar) python3 -I -c "$PY_EXTRACT" full "$path" "$dir" "$stem" || return 1; rm -f "$path"; touch "$dir/$stem/.extracted" ;;
  esac
  log "fetched $dest"
}

fetch_all() {
  local list=$1 n=0 bad=0 dest url re_dest='^[A-Za-z0-9][A-Za-z0-9._/-]*$' re_url='^https://[^"\\[:space:]]+$'
  log "downloads: $(grep -c . "$list" || true) entries in the URL list"
  while IFS=$'\t' read -r dest url || [[ -n $dest ]]; do
    n=$((n + 1))
    [[ -z $dest || $dest == \#* ]] && continue
    if ! [[ $dest =~ $re_dest ]] || [[ $dest == *..* || $dest == urls.tsv || $dest == harness/* ]]; then
      log "line $n: refused destination"; bad=$((bad + 1)); continue
    fi
    if ! [[ $url =~ $re_url ]]; then log "line $n ($dest): malformed URL (not shown)"; bad=$((bad + 1)); continue; fi
    fetch_one "$dest" "$url" || { log "line $n: download or extraction failed: $dest"; bad=$((bad + 1)); }
  done < "$list"
  url=""
  run rm -f "$list"
  (( bad == 0 )) || soft_die "$bad of $n downloads failed; the URL list is deleted: drive.sh presign + push, then start again"
}

step_inputs() {
  CURRENT=inputs
  status "running inputs"
  local hb="$CLONE/ios/MuseTalkBench/HarnessBundle" spec name size have
  run mkdir -p "$hb"
  for spec in $GATE_FILES; do
    name=${spec%%:*} size=${spec##*:} have=""
    if [[ -f $INPUTS/harness/$name ]]; then have=$(wc -c < "$INPUTS/harness/$name" | tr -d ' '); fi
    [[ $have == "$size" ]] || soft_die "gate file inputs/harness/$name is ${have:-missing} bytes, expected $size (drive.sh push)"
    run cp -f "$INPUTS/harness/$name" "$hb/$name"
  done
  if [[ -f $INPUTS/urls.tsv ]]; then
    fetch_all "$INPUTS/urls.tsv"
  elif ! marked inputs ok; then
    log "no URL list; using the inputs already on the box"
  fi
  local bundles
  # shellcheck disable=SC2046  # the globs expand here
  bundles=$(count $(tr ',' ' ' <<<"$DISTILL_TRAIN_GLOB"))
  log "inputs: $bundles training latent files, $(count "$INPUTS"/avatars/*/latents.pt) review avatars," \
      "$(count "$INPUTS"/audio/train/*) training clips, $(count "$INPUTS"/audio/review/*) review clips"
  (( bundles > 0 )) || soft_die "no training latents match $DISTILL_TRAIN_GLOB"
  mark inputs ok
}

step_facemesh() {
  CURRENT=facemesh
  if [[ $FACEMESH != 1 ]]; then log "FACEMESH=0: the review runs without FaceMesh lip aperture"; return 0; fi
  local venv="$ROOT/venv-mp"
  if [[ -x $venv/bin/python ]] && "$venv/bin/python" -c 'import mediapipe, cv2' 2>/dev/null; then MP_PY="$venv/bin/python"; return 0; fi
  status "running facemesh: mediapipe 0.10.14 venv"
  # The local .venv-mp's versions: mediapipe 0.10.14 with numpy 1.26 and opencv-contrib 4.10.
  if run python3 -m venv "$venv" \
     && run "$venv/bin/python" -m pip install --disable-pip-version-check -q mediapipe==0.10.14 numpy==1.26.4 opencv-contrib-python==4.10.0.84 \
     && { (( DRY )) || "$venv/bin/python" -c 'import mediapipe, cv2'; }; then
    MP_PY="$venv/bin/python"
  else
    log "WARNING: no mediapipe venv; the review runs without FaceMesh lip aperture"
  fi
}

TRAIN_TOTAL=0
TRAIN_WALL=0
train_budget() {  # <out dir>
  # distill_small_unet.py counts --max-minutes across resumes, so it gets the total (TRAIN_TOTAL): DISTILL_MAX_MINUTES,
  # or less when the lease must keep DISTILL_RESERVE_MINUTES for the review, the pull and cleanup. TRAIN_WALL is what
  # is left of it from now, for the timeout backstop. Minutes already trained come from log.jsonl's evals, and only
  # when ckpt.pt exists (without it the script starts over).
  local now lease trained=0 wall
  now=$(date -u +%s)
  if [[ -f $1/ckpt.pt && -f $1/log.jsonl ]]; then
    trained=$(grep -o '"elapsed_min": [0-9.]*' "$1/log.jsonl" | awk '{ if ($2 > m) m = $2 } END { print int(m) }' || true)
  fi
  wall=$(( DISTILL_MAX_MINUTES - ${trained:-0} ))
  lease=$(cat "$ROOT/lease_deadline_utc" 2>/dev/null || true)
  if [[ -n $lease ]] && lease=$(epoch_of "$lease"); then
    if (( (lease - now) / 60 - DISTILL_RESERVE_MINUTES < wall )); then wall=$(( (lease - now) / 60 - DISTILL_RESERVE_MINUTES )); fi
  else
    log "no lease deadline on the box (drive.sh push and start write it); capped by DISTILL_MAX_MINUTES only"
  fi
  TRAIN_WALL=$wall
  TRAIN_TOTAL=$(( ${trained:-0} + wall ))
}

newest_ckpt() {
  ls -1t "$1"/*ckpt*.pt "$1"/*resume*.pt 2>/dev/null | head -n 1 || true
}

step_train() {
  CURRENT=train
  local fp="$DISTILL_SHA $DISTILL_VARIANT" out="$OUT/train" script=scripts/ios/distill_small_unet.py help ckpt rc
  if marked train "$fp"; then log "train: done earlier for $fp"; return 0; fi
  run mkdir -p "$out"
  train_budget "$out"
  if (( TRAIN_WALL < 5 )); then
    if [[ -s $out/${DISTILL_VARIANT}_best.pt ]]; then log "train: time budget spent; reviewing the best weights so far"; mark train "$fp"; return 0; fi
    soft_die "no time left for training ($TRAIN_WALL min) and no weights yet"
  fi
  status "running train: $DISTILL_VARIANT, up to $TRAIN_WALL more min ($TRAIN_TOTAL in all)"
  [[ -f $CLONE/$script ]] || soft_die "$script is not in $DISTILL_SHA"
  help=$(cd "$CLONE" && "$BOX_PY" "$script" --help 2>/dev/null || true)
  [[ -n $help ]] || log "WARNING: $script --help printed nothing; passing the base flags only"
  local args=("$script" --variant "$DISTILL_VARIANT")
  # --latents (comma-separated globs) is the current name; --avatars the first version's
  if has_flag "$help" --latents; then args+=(--latents "$DISTILL_TRAIN_GLOB"); else args+=(--avatars "$DISTILL_TRAIN_GLOB"); fi
  args+=(--audio "$INPUTS/audio/train/*" --steps "$DISTILL_STEPS" --device cuda --out "$out")
  if has_flag "$help" --max-minutes; then args+=(--max-minutes "$TRAIN_TOTAL")
  else log "WARNING: $script has no --max-minutes; only the timeout backstop caps it"; fi
  ckpt=$(newest_ckpt "$out")
  if [[ -n $ckpt ]] && has_flag "$help" --resume; then
    if takes_value "$help" --resume; then args+=(--resume "$ckpt"); else args+=(--resume); fi
    log "train: resuming from $ckpt"
  elif [[ -n $ckpt ]]; then
    log "WARNING: $ckpt exists but $script has no --resume; training starts over"
  fi
  local extra=()
  read -r -a extra <<<"$DISTILL_EXTRA_ARGS"
  args+=(${extra[@]+"${extra[@]}"})
  # The backstop leaves the script 10 minutes past its own cap for the last eval and save.
  log "+ (cd $CLONE && timeout -s INT -k 120 $(( (TRAIN_WALL + 10) * 60 )) $(printf '%q ' "$BOX_PY" "${args[@]}")) | tee -a $out/train.log"
  if (( DRY )); then return 0; fi
  set +e
  (cd "$CLONE" && exec timeout -s INT -k 120 $(( (TRAIN_WALL + 10) * 60 )) "$BOX_PY" "${args[@]}") 2>&1 | tee -a "$out/train.log"
  rc=${PIPESTATUS[0]}
  set -e
  if (( rc == 124 )) && [[ -s $out/${DISTILL_VARIANT}_best.pt ]]; then
    log "train: stopped by the timeout backstop; keeping the best weights so far"
  elif (( rc != 0 )); then
    die "distill_small_unet.py exited $rc"
  fi
  mark train "$fp"
}

step_review() {
  CURRENT=review
  local out="$OUT/review" script=scripts/ios/review_distilled_unet.py weights help flag id dirs="" clips=() f rc minutes
  local student_json="$OUT/train/${DISTILL_VARIANT}.json"
  weights="$OUT/train/${DISTILL_VARIANT}_best.pt"
  [[ -s $weights ]] || weights="$OUT/train/${DISTILL_VARIANT}_last.pt"
  local fp="$DISTILL_SHA $DISTILL_VARIANT"
  if [[ -f $weights ]]; then fp="$fp $(cksum < "$weights" | cut -d' ' -f1)"; fi
  if marked review "$fp"; then log "review: done earlier"; return 0; fi
  status "running review"
  [[ -f $CLONE/$script ]] || soft_die "$script is not in $DISTILL_SHA"
  [[ -s $weights ]] || soft_die "no trained weights in $OUT/train"
  if [[ -n $REVIEW_AVATARS ]]; then
    for id in ${REVIEW_AVATARS//,/ }; do
      [[ -f $INPUTS/avatars/$id/latents.pt ]] || soft_die "review avatar $id was not downloaded"
      dirs="${dirs:+$dirs,}$INPUTS/avatars/$id"
    done
  else
    for f in "$INPUTS"/avatars/*/latents.pt; do [[ -f $f ]] && dirs="${dirs:+$dirs,}$(dirname "$f")"; done
  fi
  for f in "$INPUTS"/audio/review/*; do if [[ -f $f ]]; then clips+=("$f"); fi; done
  [[ -n $dirs && ${#clips[@]} -gt 0 ]] || soft_die "the review needs avatars (inputs/avatars) and clips (inputs/audio/review)"
  help=$(cd "$CLONE" && "$BOX_PY" "$script" --help 2>/dev/null || true)
  local args=("$script")
  has_flag "$help" --variant && args+=(--variant "$DISTILL_VARIANT")
  for flag in --weights --student --student-weights --checkpoint --ckpt; do
    if has_flag "$help" "$flag"; then args+=("$flag" "$weights"); break; fi
  done
  [[ ${#args[@]} -gt 1 && " ${args[*]} " == *" $weights "* ]] || soft_die "$script --help lists no weights flag (--weights/--student/--checkpoint)"
  if has_flag "$help" --student-json; then
    [[ -f $student_json ]] || soft_die "no $student_json from the training run"
    args+=(--student-json "$student_json")
  fi
  args+=(--avatars "$dirs")
  # --audio takes the clips as separate arguments (nargs="+"); review_mixed_units' --audios one comma-separated string
  if has_flag "$help" --audios; then args+=(--audios "$(IFS=,; echo "${clips[*]}")"); else args+=(--audio "${clips[@]}"); fi
  has_flag "$help" --seconds && args+=(--seconds "$REVIEW_SECONDS")
  has_flag "$help" --device && args+=(--device cuda)
  [[ -n $MP_PY ]] && has_flag "$help" --facemesh-python && args+=(--facemesh-python "$MP_PY")
  has_flag "$help" --out && args+=(--out "$out")
  has_flag "$help" --videos-dir && args+=(--videos-dir "$out/videos")
  has_flag "$help" --docs-dir && args+=(--docs-dir "$out/docs")
  local extra=()
  read -r -a extra <<<"$REVIEW_EXTRA_ARGS"
  args+=(${extra[@]+"${extra[@]}"})
  minutes=$REVIEW_MAX_MINUTES
  run mkdir -p "$out"
  if (( ! DRY )); then touch "$STATE/review.started"; fi
  log "+ (cd $CLONE && timeout -s INT -k 60 $(( minutes * 60 )) $(printf '%q ' "$BOX_PY" "${args[@]}")) | tee -a $out/review.log"
  if (( DRY )); then return 0; fi
  set +e
  (cd "$CLONE" && exec timeout -s INT -k 60 $(( minutes * 60 )) "$BOX_PY" "${args[@]}") 2>&1 | tee -a "$out/review.log"
  rc=${PIPESTATUS[0]}
  set -e
  (( rc == 0 )) || die "review_distilled_unet.py exited $rc"
  mark review "$fp"
}

PY_SUMMARY='
import json, pathlib, sys
root, state, rc, step = pathlib.Path(sys.argv[1]), sys.argv[2], int(sys.argv[3]), sys.argv[4]
evals = []
log = root / "out" / "train" / "log.jsonl"
if log.is_file():
    for line in log.read_text().splitlines():
        try:
            evals.append(json.loads(line))
        except ValueError:
            pass
key = "val_mae" if any("val_mae" in e for e in evals) else "blend_mae"  # what the script selects on
best = min((e for e in evals if e.get(key) is not None), key=lambda e: e[key], default=None)
files = sorted(str(p.relative_to(root)) for p in (root / "out").rglob("*") if p.is_file())
summary = {"result": "ok" if rc == 0 else "failed", "step": step, "rc": rc,
           "settings": dict(l.split("=", 1) for l in (root / "job.env").read_text().splitlines() if "=" in l) if (root / "job.env").is_file() else {},
           "markers": {p.stem: p.read_text().strip() for p in pathlib.Path(state).glob("*.done")},
           "evals": len(evals), "first_eval": evals[0] if evals else None, "last_eval": evals[-1] if evals else None,
           "best_eval": best, "files": files}
(root / "out" / "job_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
'

package() {
  local rc=$1 result
  if (( rc == 0 )); then result=ok; else result="failed $CURRENT rc=$rc"; fi
  if (( DRY )); then log "+ (cd $ROOT && tar -cf results.tar job.log STATUS job.env out [the review's files under the clone]); DONE <- $result"; return 0; fi
  set +e
  mkdir -p "$OUT"
  python3 -I -c "$PY_SUMMARY" "$ROOT" "$STATE" "$rc" "$CURRENT" || log "WARNING: no job_summary.json"
  local extra=() p
  # Anything the review wrote to its own default places in the clone (if it ignored the dirs passed to it).
  if [[ -f $STATE/review.started ]]; then
    for p in experiments/video_validation docs/ios_unet_bench; do
      [[ -d $CLONE/$p ]] || continue
      while IFS= read -r f; do extra+=("${f#"$CLONE"/}"); done < <(find "$CLONE/$p" -type f -newer "$STATE/review.started")
    done
  fi
  local excludes=()
  excludes=(--exclude=audio_windows.pt)
  [[ $RESULTS_INCLUDE_CKPT == 1 ]] || excludes+=(--exclude='*ckpt*.pt' --exclude='*resume*.pt')
  local have=() n from_clone=()
  for n in job.log STATUS job.env out; do if [[ -e $ROOT/$n ]]; then have+=("$n"); fi; done
  if (( ${#extra[@]} )); then from_clone=(-C "$CLONE" "${extra[@]}"); fi
  if tar -cf "$ROOT/results.tar.part" ${excludes[@]+"${excludes[@]}"} --transform 's,^\(experiments\|docs\)/,repo/\1/,' \
       -C "$ROOT" "${have[@]}" ${from_clone[@]+"${from_clone[@]}"}; then
    mv -f "$ROOT/results.tar.part" "$ROOT/results.tar"
    (cd "$ROOT" && sha256sum results.tar > results.tar.sha256)
    log "results.tar: $(wc -c < "$ROOT/results.tar" | tr -d ' ') bytes, ${#extra[@]} files from the clone"
  else
    log "ERROR: packaging results.tar failed"
    result="$result; results.tar failed"
  fi
  printf '%s\n' "$result" > "$ROOT/DONE"
}

finish() {
  local rc=$1
  (( LOCKED )) || return 0
  trap - EXIT
  if (( rc == 0 )); then CURRENT=package; status "packaging"; else status "failed $CURRENT rc=$rc: packaging what exists"; fi
  package "$rc"
  if (( rc == 0 )); then status "done: results.tar ready (drive.sh pull)"; else status "failed $CURRENT rc=$rc: results.tar has the logs"; fi
  exit "$rc"
}

main() {
  mkdir -p "$ROOT" "$STATE" "$OUT" "$INPUTS"
  if (( ! DRY )); then
    exec >>"$LOG" 2>&1
    exec 9>"$ROOT/.lock"
    flock -n 9 || { log "another box_job.sh holds $ROOT/.lock; exiting"; exit 0; }
  fi
  log "start (pid $$)"
  settings
  if [[ -f $ROOT/DONE && "$(head -n 1 "$ROOT/DONE")" == ok && $FORCE != 1 ]]; then
    log "DONE says ok; nothing to do (FORCE=1 re-runs the remaining steps)"
    exit 0
  fi
  if (( ! DRY )); then rm -f "$ROOT/DONE" "$ROOT/results.tar" "$ROOT/results.tar.sha256"; fi
  LOCKED=1
  trap 'finish $?' EXIT
  trap 'exit 129' HUP
  trap 'exit 130' INT
  trap 'exit 143' TERM
  step_boot
  step_stop_server
  step_python
  step_clone
  step_deps
  step_models
  step_inputs
  step_facemesh
  step_train
  step_review
  CURRENT=package
}

main "$@"
