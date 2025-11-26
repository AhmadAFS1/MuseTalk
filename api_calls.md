# MuseTalk Real-Time API

Multi-user, concurrent, real-time avatar video generation with smart GPU/VRAM management.

---

## 🚀 Quickstart

### 1. **Clone & Setup**

```bash
git clone <your-musetalk-repo-url>
cd MuseTalk
```

---

### 2. **Create and Activate Python Virtual Environment**

```bash
# Create venv (Python 3.10 recommended)
python3.10 -m venv /content/py310

# Activate venv
source /content/py310/bin/activate

# Upgrade pip
pip install --upgrade pip
```

---

### 3. **Install Requirements**

```bash
pip install -r requirements.txt
```

If you see errors about missing packages (e.g., `cv2`), install them individually:

```bash
pip install opencv-python fastapi uvicorn python-multipart torch torchvision torchaudio
```

---

### 4. **Download Model Weights**

```bash
bash download_weights.sh
```

---

### 5. **Start the API Server**

```bash
source /content/py310/bin/activate
python api_server.py --host 0.0.0.0 --port 8000
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
  source /content/py310/bin/activate
  ```
- Place your test files in `data/video/` and `data/audio/` as shown in the sample requests.
- The server supports concurrent requests and will manage VRAM automatically.
- All generated videos and intermediate files are stored in the `results/` directory.

---

## 🛠️ Troubleshooting

- **Missing dependencies:**  
  Install them with `pip install -r requirements.txt` or manually as needed.
- **CUDA errors:**  
  Make sure you have the correct PyTorch version for your CUDA driver.
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
