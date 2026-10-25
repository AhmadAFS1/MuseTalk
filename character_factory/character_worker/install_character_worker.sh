#!/usr/bin/env bash
# Idempotent installer for the character-creation worker. Safe to re-run; each step checks before acting.
#
#  1. apt prerequisites (root only)
#  2. MuseTalk server venv with avatar-prep (mmcv/mmdet/mmpose) and, by default, the MediaPipe chin-tools venv
#     -> scripts/install_musetalk.sh --check decides: 0 skip, 11 repair in place, 10 clean install
#  3. runtime secret from AWS Secrets Manager -> $ROOT/runtime.env (mode 600), sourced for the rest
#  4. pinned MiniMax H3 runtime: minimax-h3-workflows at the locked commit, ./install.sh with the
#     absolute paths the batch code expects, lib/h3 helpers copied next to the models
#  5. dependency verification and $ROOT/versions.json
#
# Flags: --plan (print, change nothing)  --check (read-only verification, exit 0/1)
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
WORKSPACE="${WORKSPACE:-/workspace}"
ROOT="${CHARACTER_WORKER_ROOT:-$WORKSPACE/character_worker}"
LOCK_JSON="$SCRIPT_DIR/locks/character_worker.lock.json"
VENV_PATH="${VENV_PATH:-$WORKSPACE/.venvs/musetalk_trt_stagewise}"
CHIN_VENV="${MUSETALK_CHIN_TOOLS_VENV:-$WORKSPACE/.venvs/musetalk_chin_tools}"
CHIN_TOOLS="${CHARACTER_WORKER_CHIN_TOOLS:-1}"
SKIP_H3="${CHARACTER_WORKER_SKIP_H3:-0}"
SKIP_MUSETALK="${CHARACTER_WORKER_SKIP_MUSETALK:-0}"
RUNTIME_ENV="$ROOT/runtime.env"
LOG_DIR="$ROOT/logs"
MODE=install
case "${1:-}" in
  --plan) MODE=plan ;;
  --check) MODE=check ;;
  "") ;;
  *) echo "usage: $0 [--plan|--check]" >&2; exit 2 ;;
esac
mkdir -p "$ROOT" "$LOG_DIR"

log() { printf '[install] [%s] %s\n' "$(date -u '+%H:%M:%S')" "$*"; }
die() { log "ERROR: $*"; exit 1; }
step_status() {  # keep /health informative during the long install
  [[ -n "${CHARACTER_WORKER_STATUS_FILE:-}" && -f "$CHARACTER_WORKER_STATUS_FILE" ]] || return 0
  python3 - "$CHARACTER_WORKER_STATUS_FILE" "$1" <<'PY' || true
import json, sys, time
path, message = sys.argv[1:]
data = json.load(open(path)); data["message"] = message; data["updated_utc"] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
json.dump(data, open(path + ".tmp", "w"), indent=2); import os; os.replace(path + ".tmp", path)
PY
}

mapfile -t LOCKV < <(python3 - "$LOCK_JSON" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); h = d["h3_workflows"]; p = h["install_paths"]
print(h["repo"]); print(h["commit"]); print(p["H3_RUNTIME"]); print(p["H3_COMFY"]); print(p["H3_VENV"]); print(p["H3_MODEL_DIR"])
print(" ".join(h["lib_copies"])); print(d["recipe"]["name"]); print(h["models_lock_sha256_prefix"])
PY
)
H3_REPO="${LOCKV[0]}"; H3_COMMIT="${LOCKV[1]}"
export H3_RUNTIME="${H3_RUNTIME:-${LOCKV[2]}}" H3_COMFY="${H3_COMFY:-${LOCKV[3]}}" H3_VENV="${H3_VENV:-${LOCKV[4]}}" H3_MODEL_DIR="${H3_MODEL_DIR:-${LOCKV[5]}}"
H3_LIBS="${LOCKV[6]}"; RECIPE="${LOCKV[7]}"; MODELS_LOCK_PREFIX="${LOCKV[8]}"
H3_CHECKOUT="${H3_WORKFLOWS_DIR:-$WORKSPACE/minimax-h3-workflows}"
H3_STAMP="$ROOT/h3_install.stamp"

if [[ "$MODE" == plan ]]; then
  cat <<PLAN
character worker install plan
  1. apt: git curl ca-certificates ffmpeg python3.10 python3.10-venv python3.10-dev build-essential libgl1 libglib2.0-0 libsndfile1 jq (root only)
  2. MuseTalk: bash $REPO_ROOT/scripts/install_musetalk.sh --venv $VENV_PATH --with-avatar-prep$( [[ "$CHIN_TOOLS" == 1 ]] && echo " --with-chin-tools --chin-venv $CHIN_VENV")  (skip=$SKIP_MUSETALK)
  3. secret: MUSETALK_AWS_SECRET_ID=${MUSETALK_AWS_SECRET_ID:-<unset>} -> $RUNTIME_ENV
  4. H3: $H3_REPO @ $H3_COMMIT -> $H3_CHECKOUT; ./install.sh with H3_RUNTIME=$H3_RUNTIME H3_COMFY=$H3_COMFY H3_VENV=$H3_VENV H3_MODEL_DIR=$H3_MODEL_DIR; copy $H3_LIBS -> $H3_RUNTIME  (skip=$SKIP_H3)
  5. verify imports, write $ROOT/versions.json (recipe $RECIPE)
PLAN
  exit 0
fi

# ---------------------------------------------------------------- 1. apt
if [[ "$MODE" == install && "$(id -u)" == 0 ]] && command -v apt-get >/dev/null; then
  missing=0
  for tool in git curl ffmpeg ffprobe python3.10 jq; do command -v "$tool" >/dev/null || missing=1; done
  python3.10 -c 'import ensurepip' 2>/dev/null || missing=1
  if (( missing )); then
    step_status "installing apt prerequisites"
    export DEBIAN_FRONTEND=noninteractive
    apt-get -o DPkg::Lock::Timeout=600 update
    apt-get -o DPkg::Lock::Timeout=600 install -y --no-install-recommends \
      git curl ca-certificates ffmpeg python3.10 python3.10-venv python3.10-dev build-essential libgl1 libglib2.0-0 libsndfile1 jq
  fi
fi
for tool in git curl ffmpeg ffprobe python3.10; do command -v "$tool" >/dev/null || die "missing prerequisite: $tool"; done

# ---------------------------------------------------------------- 2. MuseTalk
mt_args=(--venv "$VENV_PATH" --with-avatar-prep)
if [[ "$CHIN_TOOLS" == 1 ]]; then mt_args+=(--with-chin-tools --chin-venv "$CHIN_VENV"); fi
if [[ "$SKIP_MUSETALK" != 1 ]]; then
  rc=0
  bash "$REPO_ROOT/scripts/install_musetalk.sh" --check "${mt_args[@]}" > "$LOG_DIR/musetalk_check.log" 2>&1 || rc=$?
  case "$rc" in
    0) log "MuseTalk install check: OK" ;;
    10|11)
      [[ "$MODE" == install ]] || die "MuseTalk install check exited $rc (needs install); run without --check"
      extra=(); (( rc == 10 )) && extra=(--clean)
      step_status "installing the MuseTalk venv (avatar prep$( [[ "$CHIN_TOOLS" == 1 ]] && echo ", chin tools")) - several minutes"
      log "MuseTalk install ($( (( rc == 10 )) && echo clean || echo repair)): scripts/install_musetalk.sh ${mt_args[*]} ${extra[*]:-}"
      ( cd "$REPO_ROOT" && bash scripts/install_musetalk.sh "${mt_args[@]}" "${extra[@]}" ) 2>&1 | tee "$LOG_DIR/musetalk_install.log" | grep -E "^\[install_musetalk|DONE|ERROR|WARN" || true
      bash "$REPO_ROOT/scripts/install_musetalk.sh" --check "${mt_args[@]}" > "$LOG_DIR/musetalk_check.log" 2>&1 || die "MuseTalk install check still fails after install; see $LOG_DIR/musetalk_install.log"
      ;;
    *) die "MuseTalk install check exited $rc; see $LOG_DIR/musetalk_check.log" ;;
  esac
fi
[[ -x "$VENV_PATH/bin/python" ]] || die "MuseTalk venv python missing: $VENV_PATH/bin/python"

# ---------------------------------------------------------------- 3. runtime secret
if [[ -n "${MUSETALK_AWS_SECRET_ID:-}" && "$MODE" == install ]]; then
  step_status "loading the runtime secret from AWS Secrets Manager"
  tmp="$(mktemp "$ROOT/.runtime.XXXXXX.env")"; chmod 600 "$tmp"
  secret_args=(--output "$tmp")
  case "${MUSETALK_SECRETS_VERIFY_S3:-1}" in 1|true|yes|on) secret_args+=(--verify-s3) ;; esac
  if ( cd "$REPO_ROOT" && "$VENV_PATH/bin/python" scripts/bootstrap_aws_secrets.py "${secret_args[@]}" ); then
    mv "$tmp" "$RUNTIME_ENV"; chmod 600 "$RUNTIME_ENV"
    log "runtime secret written to $RUNTIME_ENV ($(grep -c '^export ' "$RUNTIME_ENV") keys; names only are logged)"
  else
    rm -f "$tmp"
    case "${MUSETALK_SECRETS_STRICT:-true}" in true|1|yes|on) die "AWS Secrets Manager bootstrap failed (strict)" ;; esac
    log "WARNING: secret bootstrap failed; continuing with the existing $RUNTIME_ENV if any"
  fi
fi
if [[ -f "$RUNTIME_ENV" ]]; then
  set -a; # shellcheck disable=SC1090
  source "$RUNTIME_ENV"; set +a
fi
if [[ -z "${AVATAR_S3_BUCKET:-}" && "${CHARACTER_WORKER_ALLOW_NO_S3:-0}" != 1 ]]; then
  die "AVATAR_S3_BUCKET is not set (template env or runtime secret); the worker cannot publish caches. Set CHARACTER_WORKER_ALLOW_NO_S3=1 for a dry install."
fi

# ---------------------------------------------------------------- 4. H3 runtime
clone_exact() {  # clone_exact <url> <commit> <dest>; token-authenticated without the token on the command line
  local url="$1" ref="$2" dest="$3"
  if [[ -d "$dest/.git" ]]; then
    if [[ "$(git -C "$dest" rev-parse HEAD)" != "$ref" ]]; then
      log "updating $dest to $ref"
      git -C "$dest" fetch --depth 1 origin "$ref" || git -C "$dest" fetch origin
      git -C "$dest" checkout --detach -q "$ref"
    fi
    return 0
  fi
  local token="${H3_WORKFLOWS_GIT_TOKEN:-${GITHUB_TOKEN:-}}" askpass=""
  if [[ -n "$token" ]]; then
    askpass="$(mktemp "$ROOT/.askpass.XXXXXX")"; chmod 700 "$askpass"
    printf '#!/usr/bin/env bash\ncase "$1" in *sername*) echo x-access-token ;; *) echo "%s" ;; esac\n' "$token" > "$askpass"
  fi
  local stage; stage="$(mktemp -d "${dest}.clone.XXXXXX")"
  (
    export GIT_TERMINAL_PROMPT=0
    [[ -n "$askpass" ]] && export GIT_ASKPASS="$askpass"
    git init -q "$stage" && git -C "$stage" remote add origin "$url" \
      && git -C "$stage" fetch --depth 1 origin "$ref" && git -C "$stage" checkout -q --detach FETCH_HEAD
  ) || { rm -rf "$stage" "${askpass:-/nonexistent}"; die "could not fetch $url @ $ref (private repo: set H3_WORKFLOWS_GIT_TOKEN or GITHUB_TOKEN)"; }
  [[ -n "$askpass" ]] && rm -f "$askpass"
  mv -T "$stage" "$dest"
}

h3_fast_check() {  # sizes + CUDA + package pins, no 61 GB hashing
  [[ -x "$H3_VENV/bin/python" && -f "$H3_CHECKOUT/scripts/doctor.py" ]] || return 1
  ( cd "$H3_CHECKOUT" && H3_RUNTIME="$H3_RUNTIME" H3_COMFY="$H3_COMFY" H3_VENV="$H3_VENV" H3_MODEL_DIR="$H3_MODEL_DIR" "$H3_VENV/bin/python" scripts/doctor.py ) > "$LOG_DIR/h3_doctor.log" 2>&1
}

if [[ "$SKIP_H3" != 1 ]]; then
  if [[ "$MODE" == install ]]; then
    step_status "fetching minimax-h3-workflows @ ${H3_COMMIT:0:12}"
    clone_exact "$H3_REPO" "$H3_COMMIT" "$H3_CHECKOUT"
  fi
  [[ -d "$H3_CHECKOUT" ]] || die "H3 workflows checkout missing: $H3_CHECKOUT"
  [[ "$(git -C "$H3_CHECKOUT" rev-parse HEAD)" == "$H3_COMMIT" ]] || die "H3 workflows checkout is not at the pinned commit $H3_COMMIT"
  [[ "$(sha256sum "$H3_CHECKOUT/locks/models.lock.json" | cut -c1-16)" == "$MODELS_LOCK_PREFIX" ]] || die "models.lock.json differs from the pinned lock; refusing to install unverified weights"
  want_stamp="$H3_COMMIT:$MODELS_LOCK_PREFIX"
  if [[ "$(cat "$H3_STAMP" 2>/dev/null || true)" == "$want_stamp" ]] && h3_fast_check; then
    log "H3 runtime present and verified (stamp matches; doctor OK)"
  else
    [[ "$MODE" == install ]] || die "H3 runtime not installed/verified (stamp or doctor failed); see $LOG_DIR/h3_doctor.log"
    step_status "installing the pinned H3 runtime (ComfyUI, venv, 61 GB of weights with SHA-256 checks) - this is the long step"
    log "H3 install: $H3_CHECKOUT/install.sh (H3_RUNTIME=$H3_RUNTIME H3_COMFY=$H3_COMFY H3_VENV=$H3_VENV H3_MODEL_DIR=$H3_MODEL_DIR)"
    mkdir -p "$H3_RUNTIME"
    ( cd "$H3_CHECKOUT" && H3_PYTHON="${H3_PYTHON:-python3.10}" ./install.sh ) 2>&1 | tee "$LOG_DIR/h3_install.log" | grep -vE "^(DOWNLOAD|VERIFIED) " || true
    h3_fast_check || die "H3 doctor failed after install; see $LOG_DIR/h3_doctor.log and $LOG_DIR/h3_install.log"
    printf '%s\n' "$want_stamp" > "$H3_STAMP"
  fi
  # the batch stages import these next to the models and read extra_model_paths_h3.yaml
  for name in $H3_LIBS; do
    src="$H3_CHECKOUT/lib/h3/$name"; dst="$H3_RUNTIME/$name"
    if [[ ! -f "$dst" ]] || ! cmp -s "$src" "$dst"; then
      [[ "$MODE" == install ]] || die "H3 helper $dst missing or stale"
      cp "$src" "$dst"
    fi
  done
  if [[ ! -f "$H3_RUNTIME/extra_model_paths_h3.yaml" || "$MODE" == install ]]; then
    "$H3_VENV/bin/python" - "$H3_MODEL_DIR" "$H3_RUNTIME/extra_model_paths_h3.yaml" <<'PY'
import sys, yaml
from pathlib import Path
models, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
cfg = {'minimax_h3': {'base_path': str(models), 'diffusion_models': 'diffusion_models', 'unet': 'diffusion_models',
                      'text_encoders': 'text_encoders', 'vae': 'vae', 'loras': 'loras'}}
text = yaml.safe_dump(cfg, sort_keys=False)
if not out.exists() or out.read_text() != text:
    out.write_text(text)
PY
  fi
  "$H3_VENV/bin/python" -c 'import PIL, yaml' || die "the comfy-h3 venv lacks Pillow/PyYAML"
fi

# ---------------------------------------------------------------- 5. verify + versions
"$VENV_PATH/bin/python" -c 'import cv2, numpy, boto3, PIL' || die "MuseTalk venv lacks cv2/numpy/boto3/Pillow"
QC_BACKEND=haar
if [[ -x "$CHIN_VENV/bin/python" ]] && "$CHIN_VENV/bin/python" -c 'import mediapipe, cv2, numpy' 2>/dev/null; then QC_BACKEND=mediapipe; fi
python3 - "$ROOT/versions.json" "$REPO_ROOT" "$H3_CHECKOUT" "$RECIPE" "$QC_BACKEND" "$VENV_PATH" "$H3_VENV" "$SKIP_H3" <<'PY'
import json, os, subprocess, sys, time
out, repo, h3, recipe, qc_backend, venv, h3_venv, skip_h3 = sys.argv[1:]
def rev(path):
    try: return subprocess.check_output(['git', '-C', path, 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: return None
def gpu():
    try: return subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,compute_cap', '--format=csv,noheader'], text=True).strip()
    except Exception: return None
data = {"installed_utc": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), "recipe": recipe, "musetalk_commit": rev(repo),
        "h3_workflows_commit": rev(h3) if skip_h3 != "1" else None, "qc_face_backend": qc_backend, "musetalk_venv": venv,
        "h3_venv": h3_venv, "gpu": gpu(), "instance_id": os.environ.get("VAST_INSTANCE_ID"), "hostname": os.uname().nodename}
json.dump(data, open(out + ".tmp", "w"), indent=2); os.replace(out + ".tmp", out)
print(json.dumps(data))
PY
log "install complete (qc face backend: $QC_BACKEND)"
