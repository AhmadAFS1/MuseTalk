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
bash scripts/setup_trt_stagewise_server_env.sh --clean --full-stack
```

---

This creates the current single-venv runtime at:

- repo: `/workspace/MuseTalk`
- venv: `/workspace/.venvs/musetalk_trt_stagewise`
- backend: TRT-stagewise inference with avatar-prep support in the same venv

---

### 3. **Start The API Server**

```bash
bash scripts/run_trt_stagewise_server.sh --profile baseline
```

---

## 🧪 Sample API Requests

### **Prepare an Avatar**

```bash
curl -X POST "http://localhost:8000/avatars/prepare" \
  -F "avatar_id=test_avatar" \
  -F "video_file=@data/video/ai_test_default_moving_vid.mp4" \
  -F "batch_size=20" \
  -F "bbox_shift=5"
```

---

### **Generate a Video**

```bash
curl -X POST "http://localhost:8000/generate" \
  -F "avatar_id=test_avatar" \
  -F "audio_file=@data/audio/response-2.mpga" \
  -F "batch_size=2" \
  -F "fps=25"
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
- For Vast.ai boot automation on a CUDA 12.1.1 node, prefer:
  ```bash
  SETUP_CLEAN=1 SETUP_FULL_STACK=1 PROFILE=baseline PORT=8000 bash scripts/vast_onstart.sh
  ```

---

## 🛠️ Troubleshooting

- **Missing dependencies:**  
  Re-run `bash scripts/setup_trt_stagewise_server_env.sh --clean --full-stack`.
- **CUDA errors:**  
  Make sure the node uses a CUDA 12.1 toolkit and that `python -c "import torch; print(torch.version.cuda)"` reports `12.1`.
- **Permission errors:**  
  Ensure you have write access to the `uploads/` and `results/` directories.

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
