# MuseTalk API Backend Architecture

A comprehensive guide to understanding the backend flow of the MuseTalk Real-Time API, detailing how avatar preparation and video generation work under the hood.

---

## 📋 Table of Contents

- [System Overview](#system-overview)
- [Core Components](#core-components)
- [API Flow: Prepare an Avatar](#api-flow-prepare-an-avatar)
- [API Flow: Generate a Video](#api-flow-generate-a-video)
- [API Flow: Session Streaming](#api-flow-session-streaming)
- [API Flow: HLS Streaming](#api-flow-hls-streaming)
- [HLS Player & Sync](#hls-player--sync)
- [API Flow: WebRTC Streaming](#api-flow-webrtc-streaming)
- [WebRTC Implementation](#webrtc-implementation)
- [Session API Reference](#session-api-reference)
- [Component Deep Dive](#component-deep-dive)
- [Memory Management](#memory-management)
- [GPU Throttling Analysis](#gpu-throttling-analysis)
- [Performance Optimizations](#performance-optimizations)
- [Load Testing](#load-testing)
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
- `{"event": "error", "message": "..."}`

---

## 📺 API Flow: HLS Streaming

This flow powers the iOS HLS path: a persistent HLS player that loops idle and switches to live segments when audio is uploaded.

### High-Level Flow

1. **Create HLS Session**: `POST /hls/sessions/create` returns `session_id`, `player_url`, and `manifest_url`.
2. **Load HLS Player**: Your app opens `/hls/player/{session_id}` (Safari/WKWebView) or loads the manifest directly.
3. **Stream Audio**: `POST /hls/sessions/{session_id}/stream` uploads audio and triggers chunk generation.
4. **Live Playback**: The server appends `.ts` segments to `live.m3u8`; the player reveals live once it has buffered enough.
5. **Return to Idle**: When live ends, the player holds the last live frame and then resumes idle at a continuity point.

### HLS Endpoints (summary)

```http
POST /hls/sessions/create?avatar_id=test_avatar&playback_fps=30&musetalk_fps=10&batch_size=2&segment_duration=1&part_duration=0&hls_server_timing=true
GET  /hls/sessions/{session_id}/index.m3u8
GET  /hls/sessions/{session_id}/live.m3u8
POST /hls/sessions/{session_id}/stream
GET  /hls/sessions/{session_id}/status
GET  /hls/player/{session_id}
DELETE /hls/sessions/{session_id}
```

Notes:
- `hls_server_timing` is per-session; if true, the server computes offsets without client input.
- `start_offset_seconds` on `/stream` is only used when `hls_server_timing=false`.

---

## 🎛️ HLS Player & Sync

This section explains how the HLS player keeps idle and live head positions aligned when switching.

### Player Architecture (`templates/hls_player.py`)

- **Dual video layers**: idle (VOD playlist) + live (event playlist).
- **Hold-frame canvas**: captures the last live frame to avoid flicker during transitions.
- **Reveal gate**: live is only shown after a decoded frame and a minimum buffered lead (`LIVE_PREBUFFER_SECONDS`).
- **Idle continuity**: on live end, the player seeks idle to a continuity point before swapping layers.

### Server-Authoritative Timing (No Idle Freeze)

To avoid a visible jump while still letting idle play, the server computes the offset for the *expected live reveal time*:

1. **Session init**
   - Record `idle_start_monotonic` and `idle_start_wall_time`.
   - Compute `idle_duration_seconds` from the idle HLS manifest.
   - Cache `idle_cycle_frames` from the avatar cycle.
2. **Stream start**
   - `expected_delay = segment_duration * HLS_LIVE_STARTUP_SEGMENTS + HLS_LIVE_PREBUFFER_SECONDS`
   - `idle_elapsed = (now - idle_start_monotonic) % idle_duration_seconds`
   - `idle_at_reveal = (idle_elapsed + expected_delay) % idle_duration_seconds`
   - Map to cycle frames:
     `offset_frames = round((idle_at_reveal / idle_duration_seconds) * idle_cycle_frames)`
   - Convert to generation time:
     `start_offset_seconds = offset_frames / generation_fps`
3. **Generation**
   - The offset is passed into `inference_streaming`, aligning the first visible live frame to the idle head at reveal time.

### Per-Session Override

- `hls_server_timing=true` (default): server computes offsets.
- `hls_server_timing=false`: client can send `start_offset_seconds`.
- The `/stream` response includes a `timing` object for debugging.
- `/hls/sessions/{id}/status` exposes `idle_duration_seconds`, `idle_elapsed_seconds`, and `hls_server_timing`.

### Caveats

- The avatar cycle is ping-pong, while idle HLS loops forward; perfect long-run alignment is not possible.
- If `idle_duration_seconds` is unknown, the offset falls back to `0.0`.
- When `playback_fps != musetalk_fps`, offsets are approximate; match FPS for best continuity.

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

- Creates an `RTCPeerConnection` using the session's ICE servers.
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

## 🔥 GPU Throttling Analysis

### Observed Symptom

Earlier load testing showed **GPU utilization hitting 100% and segment delivery slowing down even at concurrency=1**. After the shared HLS scheduler work and the later chunk-boundary / encode-path fixes, the current picture is better:

- `concurrency=1` is now healthy under the latest test profile
- `concurrency=2` is close to realtime but still near the throttle threshold
- `concurrency=3` still shows clear throughput throttling

This remains a **compute-bound bottleneck**, not a memory bottleneck, but the bottleneck now appears mainly under shared load rather than in the single-stream baseline.

### Root Cause Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   CUDA Default Stream                           │
│                  (all work serializes here)                     │
│                                                                 │
│  Session A          Session B                                   │
│  batch_size=2       batch_size=2                                │
│                                                                 │
│  ┌─────┐┌─────┐   ┌─────┐┌─────┐                              │
│  │Whsp ││UNet │   │Whsp ││UNet │                              │
│  └──┬──┘└──┬──┘   └──┬──┘└──┬──┘                              │
│     │      │         │      │                                   │
│  ===▼======▼=========▼======▼=== (serialized on GPU)           │
│  [A:Whsp][A:UNet][A:VAE][B:Whsp][B:UNet][B:VAE][A:Whsp]...   │
│                                                                 │
│  Python GIL acquired between every kernel launch ──────────►   │
└─────────────────────────────────────────────────────────────────┘
```

### Root Causes (Ranked by Impact)

| # | Cause | Component | Severity |
|---|-------|-----------|----------|
| 1 | **Single CUDA stream serialization** | `avatar_manager_parallel.py` | 🔴 Critical |
| 2 | **batch_size=2 under-utilizes GPU SMs** | `api_avatar.py` | 🔴 Critical |
| 3 | **Python GIL contention between CUDA calls** | `avatar_manager_parallel.py` | 🟡 High |
| 4 | **No compute-aware admission control** | `concurrent_gpu_manager.py` | 🟡 High |
| 5 | **Many sequential GPU kernels per frame** | `api_avatar.py` | 🟠 Medium |

### Cause 1: Single CUDA Stream Serialization

`ParallelAvatarManager` uses a **ThreadPoolExecutor** for inference. However, all CUDA operations submitted from any thread execute on the **default CUDA stream**, which is serial. Two threads submitting GPU work don't run in parallel — they queue behind each other.

```
ThreadPoolExecutor (looks parallel)
├── Thread 1: session_A inference → CUDA kernels ──┐
├── Thread 2: session_B inference → CUDA kernels ──┤
│                                                   │
│   Reality on GPU:                                 ▼
│   [A kernel 1][A kernel 2][B kernel 1][B kernel 2]  ← serial
```

**Impact**: At concurrency=2, wall time nearly doubles. GPU shows 100% because it is constantly busy with small operations plus Python overhead between them, but no actual parallelism occurs.

### Cause 2: Small Batch Size Under-Utilizes GPU

At `batch_size=2`, each UNet/VAE forward pass processes only 2 frames. Modern GPUs have thousands of CUDA cores; a batch of 2 leaves most streaming multiprocessors (SMs) idle during each kernel, but the per-kernel launch overhead and memory transfer costs are still paid in full.

```
GPU SM Utilization at Various Batch Sizes:
├── batch_size=2:  ~15-25% SM occupancy per kernel
├── batch_size=4:  ~30-50% SM occupancy per kernel
├── batch_size=8:  ~60-80% SM occupancy per kernel
└── batch_size=16: ~85-95% SM occupancy per kernel
```

The paradox: GPU utilization reads 100% (always busy), but **throughput is low** because each kernel is inefficient.

### Cause 3: Python GIL Contention

Between every CUDA kernel launch, the Python Global Interpreter Lock (GIL) must be acquired to set up the next operation. With a ThreadPoolExecutor, multiple inference threads contend for the GIL, adding latency between GPU operations.

```
Thread 1: [CUDA kernel]──[acquire GIL]──[Python setup]──[release GIL]──[CUDA kernel]
Thread 2:                 [wait GIL ↓ ]──[acquire GIL]──[Python setup]──[release GIL]──[wait GIL]
                          └─ wasted time ─┘
```

### Cause 4: Memory Gate Without Compute Gate

`GPUMemoryManager` tracks VRAM allocation but has **no concept of compute budget**. When VRAM is available, it admits concurrent requests. But each admitted request adds more serialized GPU work, driving utilization to 100% without improving throughput.

```
GPUMemoryManager check: "8GB free? Admit both sessions!"
                              ↓
GPU reality: both sessions serialize on the same stream
                              ↓
Result: 100% utilization, ~0% parallelism
```

### Cause 5: Per-Frame Sequential Pipeline

Each frame goes through a multi-step pipeline where each step waits for the previous one:

```
Per-frame pipeline (all sequential on GPU):
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│ Whisper embed │──▶│  UNet fwd    │──▶│  VAE decode  │──▶│  Face blend  │
│   (~2ms)      │   │  (~15ms)     │   │   (~5ms)     │   │   (~3ms)     │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
                                                                    │
                                                          ~25ms per frame
                                                          × 15 fps = 375ms/s
                                                          Only 2.67× realtime
```

At `batch_size=2`, ~8 batch iterations are required per 1-second segment. Each iteration has pipeline stall points where the GPU waits for Python to set up the next step.

### HLS Streaming Amplification

For HLS streaming specifically, the problem compounds:

```
Per 1-second HLS segment (at 15fps, batch_size=2):
├── 15 frames ÷ 2 per batch = ~8 batch iterations
├── Each iteration: Whisper + UNet + VAE + Blend = ~25ms × 2 frames
├── Plus Python overhead between batches: ~5ms × 8 = 40ms
├── Total GPU time: ~240ms per segment
│
│   With 2 concurrent sessions (serialized):
├── Total GPU time: ~480ms per 1-second segment
├── Segment delivery interval: 480ms (just under real-time!)
└── Any additional overhead → segments arrive LATE → throttling
```

---

### Recommended Fixes

#### Fix 1: Inference Compute Semaphore (P0 — Quick Win)

Add a compute-aware semaphore so only one session runs GPU inference at a time. This prevents GPU thrashing from context switching and actually improves throughput under contention.

**File:** `scripts/concurrent_gpu_manager.py`

```python
import asyncio
import threading

class GPUMemoryManager:
    def __init__(self, ...):
        # ...existing memory tracking...

        # Compute semaphore: limit concurrent GPU inference
        self._max_concurrent_inference = 1  # start with 1, tune up if headroom
        self._inference_semaphore = threading.Semaphore(self._max_concurrent_inference)
        self._async_inference_semaphore = asyncio.Semaphore(self._max_concurrent_inference)

    def acquire_inference_slot(self) -> bool:
        """Block until an inference slot is available (sync context)."""
        return self._inference_semaphore.acquire(timeout=120)

    def release_inference_slot(self):
        self._inference_semaphore.release()

    async def acquire_inference_slot_async(self):
        """Await until an inference slot is available (async context)."""
        await self._async_inference_semaphore.acquire()

    def release_inference_slot_async(self):
        self._async_inference_semaphore.release()
```

**Usage in `avatar_manager_parallel.py`:**
```python
await self.gpu_manager.acquire_inference_slot_async()
try:
    result = await loop.run_in_executor(self._executor, self._run_inference, ...)
finally:
    self.gpu_manager.release_inference_slot_async()
```

#### Fix 2: Increase Effective Batch Size (P0 — Biggest Throughput Win)

Increase the default batch size from 2 to 4–8. This is the single highest-impact change.

```
Throughput improvement (approximate):
├── batch_size=2:  ~2.7× realtime (barely keeps up)
├── batch_size=4:  ~4.5× realtime (comfortable headroom)
├── batch_size=8:  ~6.0× realtime (supports 2 concurrent sessions)
└── batch_size=16: ~7.5× realtime (diminishing returns)
```

For concurrent sessions, dynamically scale batch size based on available VRAM:

```python
def _get_effective_batch_size(self) -> int:
    """Scale batch size based on available VRAM."""
    free_mem = self.gpu_manager.get_free_memory_mb()
    return min(int(free_mem / 1500), 8)  # ~1.5GB per batch unit
```

#### Fix 3: Dedicated CUDA Streams (P1 — Advanced)

Create per-session CUDA streams to overlap memory transfers with compute:

```python
import torch

class GPUMemoryManager:
    def __init__(self, ...):
        # ...existing code...
        self._cuda_streams = [torch.cuda.Stream() for _ in range(2)]
        self._stream_idx = 0

    def get_cuda_stream(self) -> torch.cuda.Stream:
        stream = self._cuda_streams[self._stream_idx % len(self._cuda_streams)]
        self._stream_idx += 1
        return stream
```

In the inference path:
```python
cuda_stream = gpu_manager.get_cuda_stream()
with torch.cuda.stream(cuda_stream):
    result = unet(latents, timesteps, encoder_hidden_states)
    frames = vae.decode(result)
cuda_stream.synchronize()
```

#### Fix 4: Cross-Session Batch Aggregation (P1 — Best Scaling)

Instead of each session running `batch_size=2` independently, aggregate frames from multiple sessions into a single larger batch:

```
Current (serialized, small batches):
Session A: [batch 2][batch 2][batch 2][batch 2] → Session B: [batch 2][batch 2][batch 2][batch 2]

Proposed (aggregated):
Combined:  [batch 4 (2×A + 2×B)][batch 4][batch 4][batch 4]
           ↓                                         ↓
     Better SM utilization                  Half the iterations
```

#### Fix Priority Matrix

| Priority | Fix | Effort | Throughput Impact | Latency Impact |
|----------|-----|--------|-------------------|----------------|
| 🔴 P0 | Increase `batch_size` to 4–8 | Trivial | +70–120% | Slight increase per-segment |
| 🔴 P0 | Inference semaphore | Low | Prevents thrashing | Queues instead of contends |
| 🟡 P1 | Cross-session batch aggregation | Medium | +100–200% at concurrency≥2 | Amortized |
| 🟡 P1 | Dedicated CUDA streams | Medium | +20–40% overlap | Reduced stalls |
| 🟢 P2 | ProcessPoolExecutor (bypass GIL) | High | +10–15% | Complex to implement |

---

## ⚡ Performance Optimizations

1. **Cyclic Frame Lists**: Smooths first/last frame transitions.
2. **Background Frame Blending**: Runs blending in parallel with inference.
3. **Batch Processing**: Processes multiple frames simultaneously.
4. **Float16 Inference**: Reduces memory usage by 50%.
5. **Strict Chunk Boundary Slicing**: Prevents oversized HLS segments when the shared scheduler compose buffer crosses a segment boundary.
6. **NVENC-First HLS Encoding**: Prefers hardware-backed HLS chunk encoding, with `libx264` fallback when NVENC is unavailable.

---

## 🧪 Load Testing

### Running the Load Test

```bash
python load_test.py --base-url http://localhost:8000 \
                    --avatar-id test_avatar \
                    --audio-file ./data/audio/ai-assistant.mpga \
                    --ramp 1,2,3,4,5,6 \
                    --hold-seconds 120
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--base-url` | `http://localhost:8000` | API server URL |
| `--avatar-id` | `test_avatar` | Pre-prepared avatar ID |
| `--audio-file` | `./data/audio/ai-assistant.mpga` | Audio file to stream |
| `--concurrency` | (none) | Single-stage shortcut; overrides `--ramp` |
| `--ramp` | `1,2,3,4,5` | Comma-separated concurrency levels |
| `--hold-seconds` | `30` | Cool-down seconds between stages |
| `--segment-duration` | `1.0` | HLS segment duration in seconds |
| `--playback-fps` | `30` | Player-side FPS |
| `--musetalk-fps` | `15` | MuseTalk generation FPS |
| `--batch-size` | `2` | Inference batch size |

Latest validated throughput runs used:

- `--segment-duration 1.0`
- `--playback-fps 30`
- `--musetalk-fps 15`
- `--batch-size 4`

Server-side shared-batch sizing is controlled by the `HLS_SCHEDULER_MAX_BATCH` environment variable. This is read when the backend starts, so change it before launching the API server and restart the server after editing it.

### What the Load Test Measures

For each concurrency level the test:
1. Creates N HLS sessions simultaneously.
2. Uploads audio to all sessions at the same instant (coordinated start).
3. Polls session status and fetches `.ts` segments from `live.m3u8`.
4. Records per-session metrics: create latency, stream start latency, time to `live_ready`, segment fetch intervals.
5. Detects throttling when `max_segment_interval > 2 × segment_duration`.

### Interpreting Results

```json
{
  "concurrency": 2,
  "completed": 2,
  "failed": 0,
  "avg_time_to_live_ready_s": 3.456,
  "avg_segment_interval_s": 1.234,
  "max_segment_interval_s": 2.891,
  "wall_time_s": 45.2,
  "errors": []
}
```

| Metric | Healthy | Throttled |
|--------|---------|-----------|
| `avg_segment_interval_s` | ≤ `segment_duration` × 1.2 | > `segment_duration` × 1.5 |
| `max_segment_interval_s` | ≤ `segment_duration` × 2.0 | > `segment_duration` × 2.0 |
| `avg_time_to_live_ready_s` | < 5s | > 10s |

### Latest Validated Results (March 13, 2026)

Using:

- `segment_duration=1.0`
- `playback_fps=30`
- `musetalk_fps=15`
- `batch_size=4`
- `LIVE_MAX_CONCURRENT_GENERATIONS=6`
- `HLS_SCHEDULER_MAX_BATCH=20`

Observed results:

| Concurrency | Completed | Avg `live_ready` | Avg segment interval | Max segment interval | Wall time | Interpretation |
|-------------|-----------|------------------|----------------------|----------------------|-----------|----------------|
| `1` | `1/1` | `1.53s` | `0.81s` | `1.08s` | `15.0s` | Healthy |
| `2` | `2/2` | `2.35s` | `1.62s` | `2.16s` | `29.2s` | Near realtime, slight throttle alert |
| `3` | `3/3` | `2.89s` | `2.43s` | `2.74s` | `44.4s` | Throttled |
| `4` | `4/4` | `3.36s` | `3.26s` | `3.72s` | `59.7s` | Heavily throttled |
| `5` | `5/5` | `4.12s` | `4.11s` | `4.71s` | `76.6s` | Saturated |
| `6` | `6/6` | `4.85s` | `4.94s` | `5.39s` | `90.8s` | Deeply saturated |

These runs are materially better than the earlier scheduler-era baseline and show that the backend is now healthy for one stream, close for two streams, and overloaded from three concurrent streams onward. Higher concurrency is now mostly a stability test rather than a realtime-capacity test.

### High-Concurrency `batch_size=2` Follow-Up (March 13, 2026)

These higher-concurrency runs used:

- `segment_duration=1.0`
- `playback_fps=30`
- `musetalk_fps=15`
- `batch_size=2`
- `LIVE_MAX_CONCURRENT_GENERATIONS=9`
- `HLS_SCHEDULER_MAX_BATCH=20`

Observed results:

| Concurrency | Batch Size | Avg `live_ready` | Avg segment interval | Max segment interval | Wall time | Interpretation |
|-------------|------------|------------------|----------------------|----------------------|-----------|----------------|
| `4` | `2` | `4.67s` | `3.21s` | `3.79s` | `59.2s` | Heavily throttled |
| `6` | `2` | `5.50s` | `4.90s` | `5.72s` | `90.3s` | Deeply saturated |
| `7` | `2` | `19.07s` | `5.74s` | `7.04s` | `120.0s` | Startup fairness collapse |
| `8` | `2` | `6.63s` | `6.58s` | `7.70s` | `123.0s` | Deeply saturated |

Key takeaways:

- dropping per-stream `batch_size` from `4` to `2` did not materially change aggregate throughput at `concurrency=4` or `6`
- this suggests the current shared HLS path is limited more by aggregate GPU/model throughput and scheduler fill strategy than by the per-stream batch size alone
- smaller per-stream batches may improve flexibility and admission headroom, but they do not by themselves create more realtime capacity
- at `concurrency=7`, startup fairness degrades sharply even though all sessions still complete, which points to scheduler behavior under overload as a separate issue from steady-state throughput

### Adding GPU Monitoring

To correlate throttling with GPU metrics, add `nvidia-smi` polling to the load test. See the `poll_gpu_stats` helper below for sampling `utilization.gpu` and `memory.used` during each stage, and include `gpu_peak_util_pct`, `gpu_avg_util_pct`, and `gpu_peak_mem_mb` in the `StageReport`.

---

## 📁 File Structure

```
MuseTalk/
├── api_server.py                      # FastAPI application
├── load_test.py                       # HLS concurrent session load tester
├── scripts/
│   ├── api_avatar.py                  # API-friendly avatar
│   ├── avatar_manager_parallel.py     # Orchestration layer
│   ├── avatar_cache.py                # Smart caching
│   ├── concurrent_gpu_manager.py      # Memory management
│   ├── session_manager.py             # Session tracking and cleanup
│   ├── hls_session_manager.py         # HLS session + playlist manager
│   ├── webrtc_manager.py              # WebRTC session manager
│   ├── webrtc_tracks.py               # WebRTC tracks (video + audio)
│   └── realtime_inference.py          # Original CLI version
├── templates/
│   ├── session_player.py              # Session player HTML
│   ├── hls_player.py                  # HLS player HTML
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
