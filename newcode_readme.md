---

# MuseTalk API Backend Architecture

A comprehensive guide to understanding the backend flow of the MuseTalk Real-Time API, detailing how avatar preparation and video generation work under the hood.

---

## 📋 Table of Contents

- [System Overview](#system-overview)
- [Core Components](#core-components)
- [API Flow: Prepare an Avatar](#api-flow-prepare-an-avatar)
- [API Flow: Generate a Video](#api-flow-generate-a-video)
- [API Flow: Session Streaming](#api-flow-session-streaming)
- [API Flow: WebRTC Streaming](#api-flow-webrtc-streaming)
- [WebRTC Implementation](#webrtc-implementation)
- [Session API Reference](#session-api-reference)
- [Component Deep Dive](#component-deep-dive)
- [Memory Management](#memory-management)
- [Performance Optimizations](#performance-optimizations)
- [File Structure](#file-structure)

---

## 🏗️ System Overview

The MuseTalk API is built on a **parallel, multi-user architecture** that enables concurrent video generation on a single GPU. The system is designed around three key principles:

1. **Smart Caching**: Avatars are cached in VRAM with TTL-based and LRU eviction.
2. **GPU Memory Budgeting**: Concurrent requests are managed through memory allocation tracking.
3. **Asynchronous Processing**: Non-blocking inference with thread pools.

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Server                         │
│                    (api_server.py)                          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ParallelAvatarManager                          │
│          (avatar_manager_parallel.py)                       │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  • Model Loading (VAE, UNet, PE, Whisper)            │  │
│  │  • Thread Pool Executor                              │  │
│  │  • Request Tracking                                  │  │
│  └──────────────────────────────────────────────────────┘  │
└───┬─────────────────────────┬───────────────────────────────┘
    │                         │
    ▼                         ▼
┌─────────────────┐   ┌──────────────────────────┐
│  AvatarCache    │   │  GPUMemoryManager        │
│ (avatar_cache.py)│   │(concurrent_gpu_manager.py)│
│                 │   │                          │
│ • LRU eviction  │   │ • Memory budgeting       │
│ • TTL cleanup   │   │ • Concurrent allocation  │
│ • Hit/Miss stats│   │ • Batch size tracking    │
└─────────────────┘   └──────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                     APIAvatar                               │
│                  (api_avatar.py)                            │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  • Face detection & landmark extraction              │  │
│  │  • VAE encoding of video frames                     │  │
│  │  • Mask generation for blending                      │  │
│  │  • Audio-driven inference                            │  │
│  │  • Frame blending & video composition                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 🧩 Core Components

### 1. **[api_server.py](api_server.py)** - FastAPI Application
The main entry point that handles HTTP requests and routes them to the appropriate manager methods.

**Key Features:**
- RESTful endpoints for avatar management and video generation.
- File upload handling (video/audio).
- CORS middleware for cross-origin requests.
- Health checks and statistics endpoints.

### 2. **[avatar_manager_parallel.py](scripts/avatar_manager_parallel.py)** - Orchestration Layer
Manages all avatars, coordinates GPU resources, and handles concurrent inference requests.

**Key Features:**
- Single-instance model loading (VAE, UNet, PE, Whisper).
- Thread pool for parallel inference.
- Integration with `AvatarCache` and `GPUMemoryManager`.
- Request tracking and status management.

### 3. **[api_avatar.py](scripts/api_avatar.py)** - Avatar Processing Engine
A **completely rewritten** version of [`realtime_inference.py`](scripts/realtime_inference.py), designed for server/API usage without user prompts.

**Key Differences from Original:**
- ✅ No global variables - all models passed explicitly.
- ✅ No interactive prompts - raises exceptions instead.
- ✅ API-friendly initialization (`preparation` flag).
- ✅ Proper exception handling for server context.
- ✅ Thread-safe frame processing.

### 4. **[avatar_cache.py](scripts/avatar_cache.py)** - Smart Caching System
Implements LRU + TTL-based caching with automatic cleanup.

**Key Features:**
- OrderedDict for LRU tracking.
- Background cleanup thread.
- Memory usage tracking.
- Hit/miss statistics.

### 5. **[concurrent_gpu_manager.py](scripts/concurrent_gpu_manager.py)** - GPU Memory Allocator
Manages GPU memory budget to prevent OOM errors during concurrent inference.

**Key Features:**
- Per-batch memory allocation tracking.
- Context manager for safe allocation/release.
- Blocking allocation when memory insufficient.

### 6. **[session_manager.py](scripts/session_manager.py)** - Session Orchestration
Manages per-user streaming sessions for real-time chunk delivery.

**Key Features:**
- Per-session chunk queues for SSE delivery.
- Session TTL tracking and background cleanup.
- Active stream tracking to prevent duplicate streams.

---

## 🎬 API Flow: Prepare an Avatar

### Endpoint
```http
POST /avatars/prepare
```

### Request
```http
Content-Type: multipart/form-data

avatar_id: "test_avatar"
video_file: [binary MP4 file]
batch_size: 20
bbox_shift: 5
force_recreate: false
```

### Backend Flow

1. **File Upload & Validation**: Saves the uploaded video to the server.
2. **Avatar Preparation**: Creates or loads an avatar, extracting frames, landmarks, and generating latents.
3. **Caching**: Stores the avatar in memory for faster future access.

---

## 🎥 API Flow: Generate a Video

### Endpoint
```http
POST /generate
```

### Request
```http
Content-Type: multipart/form-data

avatar_id: "test_avatar"
audio_file: [binary audio file]
batch_size: 2
fps: 25
```

### Backend Flow

1. **Audio Upload**: Saves the uploaded audio file to the server.
2. **Avatar Retrieval**: Loads the avatar from cache or disk.
3. **Inference**: Generates video frames conditioned on audio embeddings.
4. **Video Composition**: Combines frames and audio into a final video.

---

## 📡 API Flow: Session Streaming

This flow powers the Call-Annie style experience: a persistent WebView player that receives live chunks while your app uploads audio to a session.

### High-Level Flow

1. **Create a Session**: `POST /sessions/create` returns `session_id` and a `player_url`.
2. **Load the Player**: Your app opens `player_url` (WebView or browser). The player connects to SSE events.
3. **Stream Audio**: Your app uploads audio to `POST /sessions/{session_id}/stream`.
4. **Chunk Delivery**: The server emits chunk events over SSE and the player auto-plays them.
5. **Cleanup**: Sessions expire after TTL or are deleted explicitly.

### SSE Event Payloads

Each SSE event is JSON and includes one of:

- `{"event": "chunk", "url": "/chunks/{request_id}/chunk_0001.mp4", "index": 0, "total_chunks": 20, "duration": 2, "creation_time": "..."}`
- `{"event": "complete", "total_chunks": 20}`
- `{"event": "error", "message": "..." }`

---

## 📡 API Flow: WebRTC Streaming

WebRTC runs in parallel to SSE/MSE and provides lower-latency playback with direct media tracks. The server uses `aiortc` and keeps an idle loop playing until live audio arrives.

### High-Level Flow

1. **Create WebRTC Session**: `POST /webrtc/sessions/create` returns `session_id`, `player_url`, and `ice_servers`.
2. **Open the WebRTC Player**: `GET /webrtc/player/{session_id}` loads the HTML player which performs SDP offer/answer + ICE exchange.
3. **Stream Audio**: `POST /webrtc/sessions/{session_id}/stream` uploads audio; the server pushes live frames into the WebRTC video track.
4. **Idle ↔ Live Switching**: the video track stays alive and switches between idle video frames and live inference frames.
5. **Cleanup**: sessions close on TTL or explicit delete.

### WebRTC Endpoints (summary)

```http
POST /webrtc/sessions/create?avatar_id=test_avatar&user_id=user_123
POST /webrtc/sessions/{session_id}/offer
POST /webrtc/sessions/{session_id}/ice
POST /webrtc/sessions/{session_id}/stream
GET  /webrtc/player/{session_id}
DELETE /webrtc/sessions/{session_id}
```

---

## 🧠 WebRTC Implementation

### 1) WebRTC Session Manager (`scripts/webrtc_manager.py`)

- Builds `RTCConfiguration` from `WEBRTC_STUN_URLS` / `WEBRTC_TURN_URLS` and exposes `ice_servers` to the client.
- Each session stores a peer connection, a switchable video track, a silence audio track, senders, timestamps, and a shared sync clock for audio alignment.
- Background cleanup evicts expired sessions; connection `closed` triggers immediate teardown.

### 2) WebRTC Tracks (`scripts/webrtc_tracks.py`)

- **`SwitchableVideoStreamTrack`**: single video track that always emits frames. It plays idle frames by default, and switches to live frames when `start_live()` is called. `end_live()` drains the queue and falls back to idle without replacing the track. It also supports a higher `playback_fps` by duplicating source frames.
- **`IdleVideoStreamTrack`**: loops the prepared avatar MP4 via PyAV and outputs `yuv420p` frames at the configured FPS.
- **`SilenceAudioStreamTrack`**: emits 20 ms silent audio frames to keep the audio m-line alive during idle.
- **`VideoSyncClock`**: shared clock driven by live video frames. It tracks source-frame time so audio can align to real video progress (not just wall-clock time).
- **`SyncedAudioStreamTrack`**: high-quality audio path used for live streaming. It converts audio to PCM (FFmpeg + soxr if available), preloads into memory, and paces frames using PTS timing. Playback begins when `signal_start()` is called and waits for the video clock to start. It then throttles or skips audio to stay within drift bounds (lead/lag).

Other tracks (`LiveVideoStreamTrack`, `FileAudioStreamTrack`, `ToneAudioStreamTrack`) exist for debugging/experiments but are not used in the current WebRTC flow.
Audio sync knobs (env vars):
- `WEBRTC_AUDIO_PREBUFFER_SECONDS`: delay before audio starts (default `0.2`).
- `WEBRTC_AUDIO_MAX_LEAD_SECONDS`: max audio lead before it sleeps (default `0.08`).
- `WEBRTC_AUDIO_MAX_LAG_SECONDS`: max audio lag before it skips frames (default `0.12`).

### 3) WebRTC Player (`templates/webrtc_player.py`)

- Creates an `RTCPeerConnection` using the session’s ICE servers.
- Adds `recvonly` transceivers for video and audio.
- Uses a separate `<audio>` element plus WebAudio routing to avoid Safari autoplay issues.
- Requires a user tap to start audio (Safari policy).
- Includes a debug overlay (getStats) showing audio bytes/packets and ICE state.

### 4) WebRTC Endpoints (`api_server.py`)

- **Create** (`/webrtc/sessions/create`): validates avatar, resolves idle MP4, and initializes a WebRTC session with ICE servers.
- **Offer** (`/webrtc/sessions/{id}/offer`): sets remote description, attaches the video + silence-audio tracks, creates an answer, and waits for ICE gathering.
- **ICE** (`/webrtc/sessions/{id}/ice`): adds incoming ICE candidates.
- **Stream** (`/webrtc/sessions/{id}/stream`): saves the audio file, creates `SyncedAudioStreamTrack` tied to the session's `VideoSyncClock`, signals audio start after a small prebuffer, and pushes live frames into `SwitchableVideoStreamTrack`. The audio track then stays aligned to actual video progress.
- **Player** (`/webrtc/player/{id}`): returns the HTML player.

---

## 🧾 Session API Reference

### Create Session
```http
POST /sessions/create?avatar_id=test_avatar&user_id=user_123&batch_size=2&fps=15&chunk_duration=2
```

**Response**
```json
{
  "session_id": "s0m3s3ss10n",
  "player_url": "/player/session/s0m3s3ss10n",
  "avatar_id": "test_avatar",
  "user_id": "user_123",
  "config": {
    "batch_size": 2,
    "fps": 15,
    "chunk_duration": 2
  },
  "expires_in_seconds": 3600
}
```

### Start Session Stream (Upload Audio)
```http
POST /sessions/{session_id}/stream
Content-Type: multipart/form-data

audio_file: [binary audio file]
```

Notes:
- Only one active stream is allowed per session. A second call returns 409 while streaming.

**Response**
```json
{
  "request_id": "test_avatar_req_ab12cd34",
  "session_id": "s0m3s3ss10n",
  "status": "streaming",
  "message": "Stream started. WebView will receive chunks automatically."
}
```

### Receive Streaming Events (SSE)
```http
GET /sessions/{session_id}/events
```

The player consumes this endpoint directly to receive chunks as they are generated.

### Session Status
```http
GET /sessions/{session_id}/status
```

### Delete Session
```http
DELETE /sessions/{session_id}
```

### Session Statistics
```http
GET /sessions/stats
```

### Session Player (WebView)
```http
GET /player/session/{session_id}
```

This HTML player auto-connects to the session SSE stream and plays chunks as they arrive.
Platform note:
- Use `/player/session/{session_id}` for Android (Chrome/WebView).
- Use `/hls/player/{session_id}` for iOS (Safari/WKWebView).

### Minimal Mobile Player (Optional)
```http
GET /player/mobile?session_id={session_id}
```

---

## 🔍 Component Deep Dive

### AvatarCache - Smart Caching System

**Purpose:** Reduce avatar loading time by keeping frequently-used avatars in memory.

**Key Mechanisms:**
- **LRU Eviction**: Removes least recently used avatars when memory is full.
- **TTL-based Cleanup**: Evicts stale avatars after a set time.
- **Access Tracking**: Updates access time and usage statistics for each avatar.

---

## 💾 Memory Management

### Avatar Preparation Phase
```
GPU Memory Usage (batch_size=20):
├── VAE Encoder: ~2GB
├── Face Parser: ~1GB
├── Latent Storage: ~500MB (saved to disk)
└── Peak: ~3.5GB
```

### Inference Phase
```
GPU Memory Usage (batch_size=2):
├── Models (persistent): ~5GB
├── Avatar Cache: ~500MB per avatar
├── Inference (transient): ~3GB
└── Total: ~8.5GB
```

---

## ⚡ Performance Optimizations

1. **Cyclic Frame Lists**: Smooths first/last frame transitions.
2. **Background Frame Blending**: Runs blending in parallel with inference.
3. **Batch Processing**: Processes multiple frames simultaneously.
4. **Float16 Inference**: Reduces memory usage by 50%.

---

## 📁 File Structure

```
MuseTalk/
├── api_server.py                      # FastAPI application
├── scripts/
│   ├── api_avatar.py                  # API-friendly avatar
│   ├── avatar_manager_parallel.py     # Orchestration layer
│   ├── avatar_cache.py                # Smart caching
│   ├── concurrent_gpu_manager.py      # Memory management
│   ├── session_manager.py             # Session tracking and cleanup
│   ├── webrtc_manager.py              # WebRTC session manager
│   ├── webrtc_tracks.py               # WebRTC tracks (video + audio)
│   └── realtime_inference.py          # Original CLI version
├── templates/
│   ├── session_player.py              # Session player HTML
│   ├── webrtc_player.py               # WebRTC player HTML
├── uploads/                           # Temporary uploads
├── results/                           # Outputs
└── models/                            # Pre-trained weights
```

---

## 🔗 References

- **Original MuseTalk:** [realtime_inference.py](scripts/realtime_inference.py)
- **Technical Report:** https://arxiv.org/abs/2410.10122
- **Model Weights:** https://huggingface.co/TMElyralab/MuseTalk
