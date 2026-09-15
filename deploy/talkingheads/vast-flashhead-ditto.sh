#!/usr/bin/env bash
# Companion to an independently installed MuseTalk on Ubuntu 22.04 x86_64.
# Installs SoulX + Ditto only. Host driver >=570.26; Python 3.10.
# Default: wait for MuseTalk on-start completion before using apt.
set -Eeuo pipefail
umask 022
WORKSPACE="${WORKSPACE:-/workspace}"
SOULX_REF="${SOULX_REF:-3f95daee841bf6758ffbe0dba1a8f6e4ebd82c50}"
DITTO_REF="${DITTO_REF:-c3e47eee2e626500017a0556b470d6d4182f85e8}"
SOULX_URL="${SOULX_URL:-https://github.com/AhmadAFS1/SoulX-FlashHead.git}"
DITTO_URL="${DITTO_URL:-https://github.com/antgroup/ditto-talkinghead.git}"
MIN_FREE_GIB="${MIN_FREE_GIB:-65}"
SOULX_MODELS="${SOULX_MODELS:-all}" # all or lite
[[ "$SOULX_MODELS" == all || "$SOULX_MODELS" == lite ]] || { echo 'SOULX_MODELS must be all or lite'; exit 2; }
[[ "$MIN_FREE_GIB" =~ ^[0-9]+$ ]] || exit 2
if [[ "${1:-}" == --plan ]]; then
  printf 'Workspace: %s\nSoulX: %s (%s models)\nDitto: %s\nMinimum free disk for companion: %s GiB\nMuseTalk: existing installation; wait for completion, then install companions\n' "$WORKSPACE" "$SOULX_REF" "$SOULX_MODELS" "$DITTO_REF" "$MIN_FREE_GIB"
  exit 0
fi
[[ $# == 0 ]] || { echo 'Usage: vast-flashhead-ditto.sh [--plan]'; exit 2; }
[[ $EUID == 0 ]] || { echo 'Run as root in a Vast container.'; exit 1; }
[[ "$(uname -m)" == x86_64 ]] || { echo 'This installer requires x86_64.'; exit 1; }
mkdir -p "$WORKSPACE/logs/flashhead-ditto" "$WORKSPACE/.flashhead-ditto" "$WORKSPACE/.venvs"
exec 9>"$WORKSPACE/.flashhead-ditto/install.lock"
flock -n 9 || { echo 'Another install is running.'; exit 1; }
exec > >(tee -a "$WORKSPACE/logs/flashhead-ditto/install.log") 2>&1
trap 'echo "FAILED at line $LINENO; inspect $WORKSPACE/logs/flashhead-ditto/install.log"' ERR
export WORKSPACE PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HOME="$WORKSPACE/.cache/flashhead-ditto/huggingface"
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=120 HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export MAX_JOBS="${MAX_JOBS:-2}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export TMPDIR="$WORKSPACE/.flashhead-ditto/tmp"
mkdir -p "$TMPDIR"
# Avoid the base image injecting its Python packages into these environments.
unset PYTHONPATH PYTHONHOME PIP_TARGET PIP_PREFIX PIP_USER VIRTUAL_ENV
export PIP_CONFIG_FILE=/dev/null
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee "$WORKSPACE/.flashhead-ditto/gpu-host.csv"
for driver in $(nvidia-smi --query-gpu=driver_version --format=csv,noheader); do
  dpkg --compare-versions "$driver" ge 570.26 || { echo 'SoulX cu128 requires host driver >=570.26. Choose a newer Vast host.'; exit 1; }
done
for capability in $(nvidia-smi --query-gpu=compute_cap --format=csv,noheader); do
  case "$capability" in
    8.*|9.*) ;;
    *) echo "GPU compute capability $capability is outside this pinned Ampere/Ada/Hopper stack; choose compatible versions before installing."; exit 1 ;;
  esac
done
# A completed install may restart with less free disk; individual stages are fingerprinted.
fingerprint="$(sha256sum "$0" | cut -d ' ' -f1):$SOULX_REF:$DITTO_REF:$SOULX_MODELS"
if [[ "$(cat "$WORKSPACE/.flashhead-ditto/complete" 2>/dev/null || true)" != "$fingerprint" ]]; then
  free_kib=$(df -Pk "$WORKSPACE" | awk 'NR==2 {print $4}')
  required_gib=$MIN_FREE_GIB
  if [[ -f "$WORKSPACE/.flashhead-ditto/soulx" || -f "$WORKSPACE/.flashhead-ditto/ditto" ]]; then
    required_gib=20 # allow resuming after earlier stages consumed disk
  fi
  (( free_kib >= required_gib * 1024 * 1024 )) || { echo "Need $required_gib GiB free to continue; allocate 200 GB on Vast."; exit 1; }
fi
# Do not compete for apt locks with the user's MuseTalk bootstrap. Its fd-9
# flock can be inherited by the background server, so waiting on it can hang
# forever. Instead use the final marker emitted by scripts/vast_onstart.sh.
WAIT_FOR_MUSETALK="${WAIT_FOR_MUSETALK:-1}"
MUSETALK_LOG="${MUSETALK_LOG:-$WORKSPACE/onstart.log}"
MUSETALK_WAIT_SECONDS="${MUSETALK_WAIT_SECONDS:-14400}"
[[ "$WAIT_FOR_MUSETALK" =~ ^[01]$ && "$MUSETALK_WAIT_SECONDS" =~ ^[0-9]+$ ]] || exit 2
musetalk_state() {
  [[ -f "$MUSETALK_LOG" ]] || { echo missing; return; }
  awk '
    /VAST_ONSTART BEGIN:/ {state="running"}
    /VAST_ONSTART FAILED:/ {state="failed"}
    /VAST_ONSTART COMPLETE:/ {state="complete"}
    END {print state ? state : "unknown"}
  ' "$MUSETALK_LOG"
}
wait_for_musetalk() {
  local started=$SECONDS state
  while true; do
    state=$(musetalk_state)
    case "$state" in
      complete) echo 'MuseTalk on-start completed; installing companion dependencies.'; return ;;
      failed) echo "MuseTalk reports failure; inspect $MUSETALK_LOG before retrying."; return 1 ;;
    esac
    (( SECONDS - started < MUSETALK_WAIT_SECONDS )) || { echo "Timed out waiting for $MUSETALK_LOG. Check MuseTalk bootstrap logs."; return 1; }
    echo "Waiting for MuseTalk bootstrap ($state); elapsed $((SECONDS-started))s."
    sleep 15
  done
}
if [[ "$WAIT_FOR_MUSETALK" == 1 ]]; then wait_for_musetalk; fi
# Recheck free space after MuseTalk has consumed its installation space.
free_kib=$(df -Pk "$WORKSPACE" | awk 'NR==2 {print $4}')
if [[ "$(cat "$WORKSPACE/.flashhead-ditto/complete" 2>/dev/null || true)" != "$fingerprint" ]]; then
  (( free_kib >= required_gib * 1024 * 1024 )) || { echo "Need $required_gib GiB free after MuseTalk installation."; exit 1; }
fi
export DEBIAN_FRONTEND=noninteractive
# MuseTalk has finished apt work. The timeout also tolerates other apt installs.
apt-get -o DPkg::Lock::Timeout=600 update
apt-get -o DPkg::Lock::Timeout=600 install -y --no-install-recommends \
  python3.10 python3.10-venv python3.10-dev git curl ca-certificates \
  build-essential ffmpeg libgl1 libglib2.0-0 libsndfile1
PYTHON_BIN="$(command -v python3.10)"
export PYTHON_BIN
clone_exact() {
  local url="$1" ref="$2" dest="$3"
  if [[ -e "$dest" ]]; then
    [[ -d "$dest/.git" || -f "$dest/.git" ]] || { echo "Not a Git checkout: $dest"; return 1; }
    [[ "$(git -C "$dest" rev-parse HEAD)" == "$ref" ]] || { echo "Existing $dest is on a different commit; use a fresh WORKSPACE."; return 1; }
    [[ -z "$(git -C "$dest" status --porcelain --untracked-files=no)" ]] || { echo "Tracked edits in $dest; refusing to install over them."; return 1; }
  else
    (
      clone_dir=$(mktemp -d "${dest}.clone.XXXXXX")
      trap 'rm -rf -- "$clone_dir"' EXIT
      git init "$clone_dir"
      git -C "$clone_dir" remote add origin "$url"
      git -C "$clone_dir" fetch --depth=1 origin "$ref"
      git -C "$clone_dir" checkout --detach FETCH_HEAD
      mv -T "$clone_dir" "$dest"
    )
  fi
}
clone_exact "$SOULX_URL" "$SOULX_REF" "$WORKSPACE/SoulX-FlashHead"
clone_exact "$DITTO_URL" "$DITTO_REF" "$WORKSPACE/ditto-talkinghead"
SOUL="$WORKSPACE/SoulX-FlashHead/.venv"
DITTO="$WORKSPACE/.venvs/ditto"
new_env() {
  "$PYTHON_BIN" -m venv "$1"
  "$1/bin/python" -m pip install 'pip==25.2' 'setuptools<81' wheel
}
# Mark a stage complete only after it succeeds. Failed stages can be rerun.
stage() {
  local name="$1"; shift
  local marker="$WORKSPACE/.flashhead-ditto/$name"
  if [[ "$(cat "$marker" 2>/dev/null || true)" == "$fingerprint" ]]; then
    echo "Already installed: $name"
  else
    "$@"
    printf '%s\n' "$fingerprint" > "$marker"
  fi
}
install_soul() {
  new_env "$SOUL"
  "$SOUL/bin/python" -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  # Keep the ABI/runtime versions aligned while resolving the fork requirements.
  printf 'torch==2.7.1\ntorchvision==0.22.1\ntorchaudio==2.7.1\nnumpy==2.2.6\nhuggingface-hub==0.36.0\n' > "$WORKSPACE/.flashhead-ditto/soul-constraints.txt"
  "$SOUL/bin/python" -m pip install -c "$WORKSPACE/.flashhead-ditto/soul-constraints.txt" \
    -r "$WORKSPACE/SoulX-FlashHead/requirements-webrtc.txt"
  local abi
  abi=$("$SOUL/bin/python" -c 'import torch; print(str(torch._C._GLIBCXX_USE_CXX11_ABI).upper())')
  "$SOUL/bin/python" -m pip install --no-deps \
    "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.0.post2/flash_attn-2.8.0.post2%2Bcu12torch2.7cxx11abi${abi}-cp310-cp310-linux_x86_64.whl"
}
install_ditto() {
  new_env "$DITTO"
  "$DITTO/bin/python" -m pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
  "$DITTO/bin/python" -m pip install --extra-index-url https://pypi.nvidia.com \
    tensorrt==8.6.1 cuda-python==12.6.2.post1 numpy==2.0.1 librosa==0.10.2.post1 \
    tqdm filetype imageio==2.36.1 opencv-python-headless==4.10.0.84 \
    scikit-image==0.25.0 cython imageio-ffmpeg==0.5.1 colored polygraphy \
    huggingface-hub==0.36.0
}
stage soulx install_soul
stage ditto install_ditto
# local_dir downloads avoid an additional full Git-LFS or HF blob-cache copy.
export SOULX_MODELS
"$SOUL/bin/python" - <<'PY'
import os
from huggingface_hub import snapshot_download
from pathlib import Path
w = Path(os.environ['WORKSPACE'])
patterns = ['Model_Lite/*', 'VAE_LTX/*', '*.json']
if os.environ['SOULX_MODELS'] == 'all':
    patterns += ['Model_Pro/*', 'VAE_Wan/*']
for repo, revision, dest, allow in [
    ('Soul-AILab/SoulX-FlashHead-1_3B', '59119b6c681230c3eeee157e224ae1941746711e', w/'SoulX-FlashHead/models/SoulX-FlashHead-1_3B', patterns),
    ('facebook/wav2vec2-base-960h', '22aad52d435eb6dbaf354bdad9b0da84ce7d6156', w/'SoulX-FlashHead/models/wav2vec2-base-960h', ['*.json', 'pytorch_model.bin']),
    ('digital-avatar/ditto-talkinghead', 'e4a2f60328ee7c32af585ac4b3cce299e4c8e254', w/'ditto-talkinghead/checkpoints', ['ditto_cfg/*', 'ditto_onnx/*', 'ditto_pytorch/*', 'ditto_trt_Ampere_Plus/*']),
]:
    snapshot_download(repo, revision=revision, local_dir=str(dest), allow_patterns=allow, max_workers=4)
PY
# Verify imports on every run. GPU arithmetic is optional so installation
# does not allocate GPU tensors while MuseTalk serves requests.
export RUN_GPU_SMOKE="${RUN_GPU_SMOKE:-0}"
[[ "$RUN_GPU_SMOKE" =~ ^[01]$ ]] || exit 2
for env_dir in "$SOUL" "$DITTO"; do
  "$env_dir/bin/python" - <<'PY'
import os, torch
print('torch', torch.__version__, 'CUDA build', torch.version.cuda)
if os.environ['RUN_GPU_SMOKE'] == '1':
    assert torch.cuda.is_available(), 'CUDA unavailable'
    x = torch.ones(8, device='cuda')
    assert (x+x).sum().item() == 16
    print('GPU arithmetic check:', torch.cuda.get_device_name())
else:
    print('GPU arithmetic not run; model inference remains unverified.')
PY
  "$env_dir/bin/python" -m pip freeze > "$WORKSPACE/.flashhead-ditto/$(basename "$env_dir")-$(basename "$(dirname "$env_dir")")-freeze.txt"
done
(cd "$WORKSPACE/SoulX-FlashHead"; "$SOUL/bin/python" -c 'import flash_attn, aiortc; from flash_head.src.modules.flash_head_model import flash_attention; import soulx_rtc.server')
(cd "$WORKSPACE/ditto-talkinghead"; "$DITTO/bin/python" -c 'from core.utils.tensorrt_utils import TRTWrapper'; "$DITTO/bin/python" inference.py --help)
printf '%s\n' "$fingerprint" > "$WORKSPACE/.flashhead-ditto/complete"
echo 'FLASHHEAD + DITTO INSTALL COMPLETE. No inference server was started.'
echo 'See FLASHHEAD_DITTO_COMPANION.md for launch commands and GPU validation.'
df -h "$WORKSPACE"
