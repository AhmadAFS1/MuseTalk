#!/usr/bin/env bash
set -e

# ========= CONFIG =========
VENV_DIR="/content/py310"
REPO_DIR="/content/MuseTalk"
MODELS_DIR="$REPO_DIR/models"
# ==========================

echo "[1/7] Updating apt and installing system packages..."
apt-get update -y
apt-get install -y python3.10 python3.10-venv ffmpeg aria2 build-essential git

echo "[2/7] Creating / activating Python 3.10 venv..."
if [ ! -d "$VENV_DIR" ]; then
  python3.10 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip

echo "[3/7] Installing PyTorch stack (2.0.1 / 0.15.2 / 2.0.2)..."
pip install "torch==2.0.1" "torchvision==0.15.2" "torchaudio==2.0.2"

echo "[4/7] Cloning MuseTalk repo if needed..."
if [ ! -d "$REPO_DIR/.git" ]; then
  cd /content
  rm -rf "$REPO_DIR"
  git clone https://github.com/AhmadAFS1/MuseTalk.git
else
  echo "Repo already exists at $REPO_DIR, skipping clone."
fi

cd "$REPO_DIR"

echo "[5/7] Installing Python requirements..."
pip install -r requirements.txt

echo "[5.1/7] Pinning huggingface_hub for cached_download compatibility..."
pip install "huggingface_hub==0.20.3"

echo "[6/7] Installing MM family packages (mmengine, mmcv, mmdet, mmpose)..."
pip install --no-cache-dir -U openmim

# mmengine
mim install mmengine

# mmcv
mim install "mmcv==2.0.1"

# mmdet
mim install "mmdet==3.1.0"

# chumpy needed by mmpose; install explicitly with older build behavior
pip install "chumpy==0.70" --no-build-isolation

# mmpose (try mim first, fall back to pip)
if ! mim install "mmpose==1.1.0"; then
  echo "mim install mmpose failed, trying pip..."
  pip install "mmpose==1.1.0"
fi

echo "[7/7] Ensuring models directory exists and cloning Hugging Face model repos..."
mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

# Each of these will clone under MuseTalk/models/
# If already present, skip with a message instead of failing.
if [ ! -d "sd-vae-ft-mse" ]; then
  git clone https://huggingface.co/stabilityai/sd-vae-ft-mse
else
  echo "sd-vae-ft-mse already exists, skipping clone."
fi

if [ ! -d "whisper-tiny" ]; then
  git clone https://huggingface.co/openai/whisper-tiny
else
  echo "whisper-tiny already exists, skipping clone."
fi

if [ ! -d "DWPose" ]; then
  git clone https://huggingface.co/yzd-v/DWPose
else
  echo "DWPose already exists, skipping clone."
fi

if [ ! -d "LatentSync" ]; then
  git clone https://huggingface.co/ByteDance/LatentSync
else
  echo "LatentSync already exists, skipping clone."
fi

cd "$REPO_DIR"

echo "[7+.1/7] Downloading any remaining model weights via download_weights.sh..."
chmod +x download_weights.sh
bash download_weights.sh

echo
echo "==============================================="
echo "MuseTalk + Python 3.10 environment is READY 🎉"
echo
echo "To launch the app next, run in a new cell:"
echo
echo "%%bash"
echo "source $VENV_DIR/bin/activate"
echo "cd $REPO_DIR"
echo "export FFMPEG_PATH=/usr/bin/ffmpeg"
echo "export MPLBACKEND=Agg"
echo "python app.py --use_float16"
echo "==============================================="

cd "$MODELS_DIR"

# Replace/overwrite LatentSync → syncnet
rm -rf "syncnet"
mv "LatentSync" "syncnet"

# Replace/overwrite DWPose → dwpose
rm -rf "dwpose"
mv "DWPose" "dwpose"

# Replace/overwrite sd-vae → sd-vae-ft-mse
rm -rf "sd-vae"
mv "sd-vae-ft-mse" "sd-vae"

# Replace/overwrite whisper → whisper-tiny
rm -rf "whisper"
mv "whisper-tiny" "whisper"

cd "$REPO_DIR"

pip install "huggingface_hub==0.30.2"

