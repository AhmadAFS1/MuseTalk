#!/usr/bin/env bash
# CPU image build only. Runtime boot never invokes dependency installation.
set -Eeuo pipefail
REPO_ROOT="${REPO_ROOT:-/opt/character-runtime/app}"
WORKSPACE="${WORKSPACE:-/workspace}"
[[ -e "$WORKSPACE/MuseTalk" ]] || ln -s "$REPO_ROOT" "$WORKSPACE/MuseTalk"
mkdir -p "$WORKSPACE/.venvs" "$WORKSPACE/character_worker/logs"
# The dependency diagnostic pruned openmim's download/index dependencies after
# compiling MMCV. Remove the build-only CLI too: runtime mmpose uses mmengine,
# mmcv and mmdet directly, and the complete runtime must pass pip check.
/opt/musetalk/venv/bin/python -m pip uninstall -y openmim
# Keep the dependency base's venv at its original absolute prefix.
[[ -e "$WORKSPACE/.venvs/musetalk_trt_stagewise" ]] || ln -s /opt/musetalk/venv "$WORKSPACE/.venvs/musetalk_trt_stagewise"
python3.10 -m venv "$WORKSPACE/.venvs/musetalk_chin_tools"
"$WORKSPACE/.venvs/musetalk_chin_tools/bin/python" -m pip install --no-cache-dir \
  -r "$REPO_ROOT/requirements/chin-tools.in" -c "$REPO_ROOT/requirements/constraints-chin-tools.txt"
cd "$WORKSPACE/minimax-h3-workflows"
H3_PYTHON=python3.10 H3_MODEL_PROFILE=character-fl2va ./install.sh --environment-only --profile character-fl2va
cd "$REPO_ROOT"
# Assert the reused base has the unchanged MuseTalk preparation requirement set.
/opt/musetalk/venv/bin/python -m pip check
/opt/musetalk/venv/bin/python -c 'import torch, cv2, numpy, fastapi, multipart; from mmcv.ops import get_compiling_cuda_version; assert str(get_compiling_cuda_version()).startswith("12.1")'
