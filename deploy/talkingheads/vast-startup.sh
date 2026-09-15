#!/usr/bin/env bash
# Ubuntu 22.04 x86_64, CUDA 12.1 devel, Python 3.10; host driver >=570.26.
# Installs three isolated runtimes. Does not launch GPU servers automatically.
set -Eeuo pipefail
umask 022
WORKSPACE="${WORKSPACE:-/workspace}"
MUSETALK_REF="${MUSETALK_REF:-e8e5de56bc9a2f97e9ca0e87757ac3545a550f8e}"
SOULX_REF="${SOULX_REF:-503378690b871dcff215f9a33a0656730b6221b4}"
DITTO_REF="${DITTO_REF:-c3e47eee2e626500017a0556b470d6d4182f85e8}"
MUSETALK_URL="${MUSETALK_URL:-https://github.com/AhmadAFS1/MuseTalk.git}"
SOULX_URL="${SOULX_URL:-https://github.com/AhmadAFS1/SoulX-FlashHead.git}"
DITTO_URL="${DITTO_URL:-https://github.com/antgroup/ditto-talkinghead.git}"
MIN_FREE_GIB="${MIN_FREE_GIB:-100}"
SOULX_MODELS="${SOULX_MODELS:-all}" # all or lite
[[ "$SOULX_MODELS" == all || "$SOULX_MODELS" == lite ]] || { echo 'SOULX_MODELS must be all or lite'; exit 2; }
[[ "$MIN_FREE_GIB" =~ ^[0-9]+$ ]] || exit 2
if [[ "${1:-}" == --plan ]]; then
  printf 'Workspace: %s\nMuseTalk: %s\nSoulX: %s (%s models)\nDitto: %s\nMinimum free disk before installation: %s GiB\nRecommended allocation: 200 GB\n' "$WORKSPACE" "$MUSETALK_REF" "$SOULX_REF" "$SOULX_MODELS" "$DITTO_REF" "$MIN_FREE_GIB"
  exit 0
fi
[[ $# == 0 ]] || { echo 'Usage: vast-startup.sh [--plan]'; exit 2; }
[[ $EUID == 0 ]] || { echo 'Run as root in a Vast container.'; exit 1; }
[[ "$(uname -m)" == x86_64 ]] || { echo 'This installer requires x86_64.'; exit 1; }
mkdir -p "$WORKSPACE/logs/talkingheads" "$WORKSPACE/.talkingheads" "$WORKSPACE/.venvs"
exec 9>"$WORKSPACE/.talkingheads/install.lock"
flock -n 9 || { echo 'Another install is running.'; exit 1; }
exec > >(tee -a "$WORKSPACE/logs/talkingheads/install.log") 2>&1
trap 'echo "FAILED at line $LINENO; inspect $WORKSPACE/logs/talkingheads/install.log"' ERR
export WORKSPACE PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
export HF_HOME="$WORKSPACE/.cache/huggingface"
export HF_HUB_DOWNLOAD_TIMEOUT=120 HF_HUB_ETAG_TIMEOUT=120 HF_XET_CHUNK_CACHE_SIZE_BYTES=0
export MAX_JOBS="${MAX_JOBS:-4}"
# Avoid the base image injecting its Python packages into these environments.
unset PYTHONPATH PYTHONHOME
nvidia-smi
for driver in $(nvidia-smi --query-gpu=driver_version --format=csv,noheader); do
  dpkg --compare-versions "$driver" ge 570.26 || { echo 'SoulX cu128 requires host driver >=570.26. Choose a newer Vast host.'; exit 1; }
done
nvcc --version | grep -q 'release 12.1' || { echo 'Use a CUDA 12.1 devel image for MuseTalk avatar preparation.'; exit 1; }
# A completed install may restart with less free disk; individual stages are fingerprinted.
fingerprint="$(sha256sum "$0" | cut -d ' ' -f1):$MUSETALK_REF:$SOULX_REF:$DITTO_REF:$SOULX_MODELS"
if [[ "$(cat "$WORKSPACE/.talkingheads/complete" 2>/dev/null || true)" != "$fingerprint" ]]; then
  free_kib=$(df -Pk "$WORKSPACE" | awk 'NR==2 {print $4}')
  required_gib=$MIN_FREE_GIB
  if [[ -f "$WORKSPACE/.talkingheads/musetalk" || -f "$WORKSPACE/.talkingheads/soulx" || -f "$WORKSPACE/.talkingheads/ditto" ]]; then
    required_gib=20 # allow resuming after earlier stages consumed disk
  fi
  (( free_kib >= required_gib * 1024 * 1024 )) || { echo "Need $required_gib GiB free to continue; allocate 200 GB on Vast."; exit 1; }
fi
export DEBIAN_FRONTEND=noninteractive
apt-get update
apt-get install -y --no-install-recommends python3.10 python3.10-venv python3.10-dev \
  git git-lfs curl ca-certificates build-essential ffmpeg libgl1 libglib2.0-0 \
  libsm6 libxext6 libxrender1 libsndfile1 espeak-ng coturn
apt-get clean
PYTHON_BIN="$(command -v python3.10)"
export PYTHON_BIN
clone_exact() {
  local url="$1" ref="$2" dest="$3"
  if [[ -e "$dest" ]]; then
    [[ -d "$dest/.git" || -f "$dest/.git" ]] || { echo "Not a Git checkout: $dest"; return 1; }
    [[ "$(git -C "$dest" rev-parse HEAD)" == "$ref" ]] || { echo "Existing $dest is on a different commit; use a fresh WORKSPACE."; return 1; }
    [[ -z "$(git -C "$dest" status --porcelain --untracked-files=no)" ]] || { echo "Tracked edits in $dest; refusing to install over them."; return 1; }
  else
    git init "$dest"
    git -C "$dest" remote add origin "$url"
    git -C "$dest" fetch --depth=1 origin "$ref"
    git -C "$dest" checkout --detach FETCH_HEAD
  fi
}
clone_exact "$MUSETALK_URL" "$MUSETALK_REF" "$WORKSPACE/MuseTalk"
clone_exact "$SOULX_URL" "$SOULX_REF" "$WORKSPACE/SoulX-FlashHead"
clone_exact "$DITTO_URL" "$DITTO_REF" "$WORKSPACE/ditto-talkinghead"
MUSE="$WORKSPACE/.venvs/musetalk_trt_stagewise"
SOUL="$WORKSPACE/SoulX-FlashHead/.venv"
DITTO="$WORKSPACE/.venvs/ditto"
new_env() {
  "$PYTHON_BIN" -m venv "$1"
  "$1/bin/python" -m pip install 'pip==25.2' 'setuptools<81' wheel
}
# Mark a stage complete only after it succeeds. Failed stages can be rerun.
stage() {
  local name="$1"; shift
  local marker="$WORKSPACE/.talkingheads/$name"
  if [[ "$(cat "$marker" 2>/dev/null || true)" == "$fingerprint" ]]; then
    echo "Already installed: $name"
  else
    "$@"
    printf '%s\n' "$fingerprint" > "$marker"
  fi
}
install_muse() {
  WHEELHOUSE_ENABLED=0 REPO_ROOT="$WORKSPACE/MuseTalk" \
    bash "$WORKSPACE/MuseTalk/setup_musetalk.sh" --full-stack --skip-apt \
      --python-bin "$PYTHON_BIN" --venv-path "$MUSE"
}
install_soul() {
  new_env "$SOUL"
  "$SOUL/bin/python" -m pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128
  # Keep the ABI/runtime versions aligned while resolving the fork requirements.
  printf 'torch==2.7.1\ntorchvision==0.22.1\ntorchaudio==2.7.1\nnumpy==2.2.6\nhuggingface-hub==0.36.0\n' > "$WORKSPACE/.talkingheads/soul-constraints.txt"
  "$SOUL/bin/python" -m pip install -c "$WORKSPACE/.talkingheads/soul-constraints.txt" \
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
stage musetalk install_muse
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
# Verify GPU computation and imports on every run, including resumed installs.
for env_dir in "$MUSE" "$SOUL" "$DITTO"; do
  "$env_dir/bin/python" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA unavailable'
x = torch.ones(8, device='cuda')
assert (x+x).sum().item() == 16
print(torch.__version__, torch.cuda.get_device_name())
PY
  "$env_dir/bin/python" -m pip freeze > "$WORKSPACE/.talkingheads/$(basename "$env_dir")-$(basename "$(dirname "$env_dir")")-freeze.txt"
done
(cd "$WORKSPACE/MuseTalk"; "$MUSE/bin/python" -c 'import tensorrt, mmcv._ext, aiortc; from musetalk.utils.preprocessing import get_landmark_and_bbox')
(cd "$WORKSPACE/SoulX-FlashHead"; "$SOUL/bin/python" -c 'import flash_attn, aiortc; from flash_head.src.modules.flash_head_model import flash_attention; import soulx_rtc.server')
(cd "$WORKSPACE/ditto-talkinghead"; "$DITTO/bin/python" -c 'from core.utils.tensorrt_utils import TRTWrapper'; "$DITTO/bin/python" inference.py --help)
printf '%s\n' "$fingerprint" > "$WORKSPACE/.talkingheads/complete"
echo 'INSTALL COMPLETE. See the companion README for launch and GPU inference checks.'
df -h "$WORKSPACE"
