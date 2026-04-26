# MuseTalk Real-Time API

Multi-user, concurrent, real-time avatar video generation with smart GPU/VRAM management.

---

## 🚀 Quickstart

### 1. **Clone To The Vast Workspace**

```bash
git clone <your-musetalk-repo-url> /workspace/MuseTalk
cd /workspace/MuseTalk
```

---

### 2. **Create The Current TRT-Stagewise Venv**

```bash
bash scripts/setup_trt_stagewise_server_env.sh --clean
```

---

This creates the current single-venv runtime at:

- repo: `/workspace/MuseTalk`
- venv: `/workspace/.venvs/musetalk_trt_stagewise`
- backend: TRT-stagewise inference
- current validated package family:
  - `torch==2.5.1+cu121`
  - `torch_tensorrt==2.5.0`
  - `tensorrt==10.3.0`

If this same node must also run avatar preparation in-process, use the slower
full-stack path instead:

```bash
bash scripts/setup_trt_stagewise_server_env.sh --clean --full-stack
```

That path adds:

- `mmcv==2.1.0` with `mmcv._ext`
- `mmengine==0.10.4`
- `mmdet==3.2.0`
- `mmpose==1.3.1`

---

### 3. **Start The API Server**

```bash
bash scripts/run_trt_stagewise_server.sh --profile baseline
```

This short command is the intended baseline launcher. It sets the current
runtime defaults internally.

For the Vast background wrapper:

```bash
PORT=8000 bash scripts/vast_onstart.sh
```

That wrapper now defaults to `throughput_record` when `PROFILE` is unset. Use
`PROFILE=baseline` explicitly if you want the lower-VRAM startup shape.

---

## 🧪 Sample API Requests

### **Prepare an Avatar**

```bash
curl -X POST "http://localhost:8000/avatars/prepare?avatar_id=test_avatar&batch_size=20&bbox_shift=5" \
  -F "video_file=@data/video/ai_test_default_moving_vid.mp4"
```

If a previous failed preparation left partial files behind, force a clean
rebuild:

```bash
curl -X POST "http://localhost:8000/avatars/prepare?avatar_id=test_avatar&batch_size=20&bbox_shift=5&force_recreate=true" \
  -F "video_file=@data/video/ai_test_default_moving_vid.mp4"
```

---

### **Generate a Video**

```bash
curl -X POST "http://localhost:8000/generate?avatar_id=test_avatar&batch_size=2&fps=25" \
  -F "audio_file=@data/audio/response-2.mpga"
```

---

### **Check Generation Status**

```bash
curl http://localhost:8000/generate/<request_id>/status
```

---

### **Download Generated Video**

```bash
curl -O -J http://localhost:8000/generate/<request_id>/download
```

---

### **List Prepared Avatars**

```bash
curl http://localhost:8000/avatars/list
```

---

### **Evict Avatar from VRAM Cache**

```bash
curl -X DELETE http://localhost:8000/avatars/test_avatar
```

---

### **Get System Stats**

```bash
curl http://localhost:8000/stats
```

---

## 🖥️ Interactive API Docs

Open in your browser:

```
http://localhost:8000/docs
```

---

## 📝 Notes

- Always activate your venv before running the server or scripts:
  ```bash
  source /workspace/.venvs/musetalk_trt_stagewise/bin/activate
  ```
- Place your test files in `data/video/` and `data/audio/` as shown in the sample requests.
- The server supports concurrent requests and will manage VRAM automatically.
- All generated videos and intermediate files are stored in the `results/` directory.
- For Vast.ai boot automation on a serving-only CUDA 12.1.1 node, prefer:
  ```bash
  SETUP_CLEAN=1 PROFILE=throughput_record PORT=8000 bash scripts/vast_onstart.sh
  ```
- Only add `SETUP_FULL_STACK=1` when that same node must handle
  `/avatars/prepare`; on the current `torch==2.5.1+cu121` stack, fresh
  full-stack boots may source-build `mmcv` because OpenMMLab does not publish a
  matching prebuilt wheel index for `cu121/torch2.5.x`
- `download_weights.sh` now includes the S3FD face-detector weight used by
  avatar preparation, so `/avatars/prepare` should not need to download that
  checkpoint at runtime
- Optional S3 avatar asset persistence is available through environment vars:
  ```bash
  AVATAR_S3_ENABLED=1
  AVATAR_S3_BUCKET=<your-bucket>
  AVATAR_S3_PREFIX=avatars                  # optional, default: avatars
  AVATAR_S3_ENDPOINT_URL=<custom-endpoint>  # optional (MinIO / R2 / etc)
  AWS_REGION=us-east-1                      # optional
  ```
  With this enabled, `/avatars/prepare` uploads prepared avatar artifacts to S3
  and new instances lazily restore missing avatar assets by `avatar_id`.

---

## 🛠️ Troubleshooting

- **Missing dependencies:**  
  Re-run `bash scripts/setup_trt_stagewise_server_env.sh --clean --full-stack`.
- **CUDA errors:**  
  Make sure the node uses a CUDA 12.1 toolkit and that `python -c "import torch; print(torch.version.cuda)"` reports `12.1`.
- **Avatar prep fails with `<urlopen error [Errno -2] Name or service not known>`:**
  The S3FD face-detector weight is missing locally. Re-run `bash ./download_weights.sh` and verify `models/face_detection/s3fd.pth`.
- **Permission errors:**  
  Ensure you have write access to the `uploads/` and `results/` directories.
- **Warnings during successful avatar prep:**
  Current upstream warnings around `torch.load(weights_only=False)` and some MMDetection/MMEngine deprecations are noisy but were observed during a successful end-to-end `test_avatar` preparation on the validated CUDA 12.1 stack.

---

## 🧹 Clean Shutdown

To stop the server and clean up resources:

```bash
CTRL+C
```

The server will automatically clear the avatar cache and release VRAM.

---

## 📬 Contact

For issues or questions, open an issue on your repository or contact the maintainer.
