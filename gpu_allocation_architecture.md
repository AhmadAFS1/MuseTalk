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
- [HLS Scheduler Deep Dive](#hls-scheduler-deep-dive)
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
- Per-avatar load locks to prevent duplicate cold loads.

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

**Important Note:** The HLS scheduler bypasses this module's lease system by using `_memory_bucket=1`. The scheduler runs one batch at a time in a serial loop and only needs a single minimal lease. The full memory manager remains relevant for SSE and non-HLS inference paths.

### 6. **[session_manager.py](scripts/session_manager.py)** - Session Orchestration
Manages per-user streaming sessions for real-time chunk delivery.

**Key Features:**
- Per-session chunk queues for SSE delivery.
- Session TTL tracking and background cleanup.
- Active stream tracking to prevent duplicate streams.

### 7. **[hls_gpu_scheduler.py](scripts/hls_gpu_scheduler.py)** - HLS GPU Stream Scheduler
The shared GPU scheduler that batches work across all active HLS streams.

**Key Features:**
- Single scheduler thread drives all HLS generation.
- Batches frames from multiple streams into one GPU forward pass.
- Separate prep, compose, and encode worker pools.
- Startup fairness: new streams get a small initial slice before warmed streams.
- Round 3 fill: always fills remaining GPU capacity across all schedulable jobs.

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

### VRAM Budget for Batched HLS Scheduling (RTX 3090, 24GB)
```
Model weights (UNet+VAE+PE+Whisper):    ~2.5GB (always resident)
Avatar cache (1-3 loaded avatars):       ~1-2GB
PyTorch allocator overhead:              ~1GB
Available for batch activations:         ~18-19GB

Per-frame activation cost during UNet+VAE forward pass: ~250-400MB
Max safe combined batch size: 18GB ÷ 0.35GB ≈ 48-52 frames

Current default max_combined_batch_size=32 uses ~11GB for activations.
This leaves ~7-8GB of headroom for avatar cache and overhead.
```

---

## 🔥 GPU Throttling Analysis

### Observed Symptom

Earlier load testing showed **GPU utilization hitting 100% and segment delivery slowing down even at concurrency=1**. After the shared HLS scheduler work and the later chunk-boundary / encode-path fixes, the current picture is better:

- `concurrency=1` is now healthy under the latest test profile
- `concurrency=2` is close to realtime but still near the throttle threshold
- `concurrency=3` still shows clear throughput throttling

This remains a **compute-bound bottleneck**, not a memory bottleneck, but the bottleneck now appears mainly under shared load rather than in the single-stream baseline.

### The Hardware Throughput Ceiling

The RTX 3090 produces approximately **18-20 frames per second** of MuseTalk output (PE + UNet + VAE combined at float16 precision). This has been validated empirically:

```
Single stream:   213 frames / 11.4s = 18.7 fps
Eight streams:  1704 frames / 99.7s = 17.1 fps
Ratio: 0.91x — aggregate throughput is nearly identical
```

Latest follow-up result at the current best `24/12`, `concurrency=8`, `batch_size=2` profile:

```json
{
  "avg_time_to_live_ready_s": 8.035,
  "avg_segment_interval_s": 5.126,
  "max_segment_interval_s": 5.362,
  "wall_time_s": 93.9,
  "gpu": {
    "avg_util_pct": 36.9,
    "peak_util_pct": 100.0,
    "peak_memory_used_mb": 12237.0
  }
}
```

This is a real improvement in wall time and jitter, but it still implies the same basic aggregate ceiling:

- `96 / 5.126 = 18.7 fps` effective throughput
- `1704 / 93.9 = 18.1 fps` wall-time throughput

So the newer work improved overhead and stability, but it did not fundamentally move the hardware throughput limit.

Latest March 15 post-compose / PE refactor result on the same profile:

```json
{
  "avg_time_to_live_ready_s": 3.282,
  "avg_segment_interval_s": 2.301,
  "max_segment_interval_s": 3.521,
  "wall_time_s": 44.4,
  "gpu": {
    "avg_util_pct": 75.18,
    "peak_util_pct": 100.0,
    "peak_memory_used_mb": 10588.0
  }
}
```

Representative scheduler timings from that run:

- `avg_gpu_batch ≈ 0.727s to 0.813s`
- `avg_compose ≈ 0.122s to 0.140s`
- `avg_encode ≈ 0.466s to 0.681s`
- `first_chunk ≈ 2.04s to 3.67s`

Interpretation:

- the compose rewrite and CPU-side PE precompute materially improved startup and steady-state cadence
- the previous `~5.1-5.2s` interval band is no longer the active baseline
- the old aggregate `18-20 fps` ceiling estimate now needs to be recalculated against the new code path before it is treated as a hard constant

This means:
- At `musetalk_fps=12`: 1 stream needs 12 fps (GPU can sustain), 8 streams need 96 fps (5.3x deficit)
- At `musetalk_fps=6`:  1 stream needs 6 fps (easy), 8 streams need 48 fps (2.7x deficit)
- At `musetalk_fps=3`:  1 stream needs 3 fps (trivial), 8 streams need 24 fps (1.3x deficit)

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

Note: The HLS scheduler now batches frames from multiple streams into a single GPU forward pass, which is materially better than the per-thread serialization shown above. However, aggregate throughput is still bounded by the UNet+VAE compute cost per frame.

### Root Causes (Ranked by Impact)

| # | Cause | Component | Severity | Status |
|---|-------|-----------|----------|--------|
| 1 | **Hardware throughput ceiling (~18-20 fps)** | UNet + VAE | 🔴 Critical | Hardware limit |
| 2 | **`_memory_bucket` deadlock at high batch sizes** | `hls_gpu_scheduler.py` | 🔴 Critical | ✅ Fixed (returns 1) |
| 3 | **Round 3 fill guard blocking GPU utilization** | `hls_gpu_scheduler.py` | 🔴 Critical | ✅ Fixed (always fills) |
| 4 | **batch_size=2 under-utilizes GPU SMs** | `api_avatar.py` | 🟡 High | Mitigated by scheduler batching |
| 5 | **NVENC concurrent session limit** | ffmpeg encode | 🟡 High | Fallback works, persistent encoder planned |
| 6 | **Python GIL contention between CUDA calls** | `avatar_manager_parallel.py` | 🟠 Medium | Mitigated by single scheduler thread |
| 7 | **No compute-aware admission control** | `concurrent_gpu_manager.py` | 🟠 Medium | Planned |
| 8 | **`torch.compile` is environment-sensitive** | `avatar_manager_parallel.py` | 🟡 High | Safer fallback implemented; throughput gain still unproven |

### Cause 1: Hardware Throughput Ceiling

The RTX 3090 produces ~18-20 fps of MuseTalk output regardless of batch size or scheduler configuration. Larger batches improve per-tick efficiency, but they do not change the basic ceiling enough to make high-concurrency HLS realtime.

The latest `batch_size=2` vs `batch_size=4` comparison at `concurrency=8` reinforces the same conclusion:

```json
{
  "batch_size_2": {
    "avg_segment_interval_s": 5.126,
    "max_segment_interval_s": 5.362,
    "wall_time_s": 93.9
  },
  "batch_size_4": {
    "avg_segment_interval_s": 5.108,
    "max_segment_interval_s": 5.424,
    "wall_time_s": 94.0
  }
}
```

That result is effectively flat. Under the current scheduler, larger per-stream batches no longer unlock a new throughput regime.

### Cause 2: `_memory_bucket` Deadlock

This was a real scheduler bug, and it is fixed.

Originally:

- `_memory_bucket(batch_size)` could return large lease sizes
- the GPU memory manager only had a small fixed logical pool
- a scheduler batch requesting a lease larger than the pool would block forever before generation started

That is why some earlier runs produced zero segments with no obvious application-level exception.

Current state:

- `_memory_bucket()` returns `1`
- the HLS scheduler only takes one minimal lease per generation turn
- this removed the deadlock and made larger combined scheduler batches viable again

### Cause 3: Fill Guard / Utilization Loss

The older scheduler had a high-concurrency guard that stopped filling the remaining batch budget once too many jobs were active. In practice this left GPU capacity unused exactly when concurrency was highest.

Current state:

- the scheduler now keeps filling remaining capacity
- this improved wall time and reduced worst-case jitter
- however, it still did not move the system outside the same `18-20 fps` aggregate ceiling

### Cause 4: `batch_size` Is No Longer a Primary Lever

At lower concurrency, per-stream `batch_size` mattered more. At current high-concurrency HLS loads, the scheduler is already aggregating enough work that raising individual stream `batch_size` has little effect.

Practical conclusion:

- keep `batch_size=2` as the default
- do not expect `batch_size=4` to materially improve `concurrency=8`
- treat larger `batch_size` mainly as a fairness and latency tradeoff, not as a throughput unlock

### Cause 5: NVENC Concurrent Session Limit

The current live HLS encode path still creates many short-lived ffmpeg jobs using `h264_nvenc`. Under enough overlap, ffmpeg can exit before Python finishes writing raw frames, which appears in logs as:

- `[Errno 32] Broken pipe`

What is happening:

- Python writes frames to ffmpeg stdin
- ffmpeg exits early when the NVENC encoder cannot be acquired or initialized cleanly
- the code retries with `libx264`
- the fallback succeeds, but it is slower and increases tail latency

This does not explain the 18-20 fps model ceiling by itself, but it does explain chunk-encode noise and some of the wall-time and jitter penalties at higher concurrency.

### Cause 6: Python / Pipeline Burstiness

Average GPU utilization can look moderate even when the pipeline is clearly overloaded.

Example older run at `concurrency=5`:

```json
{
  "avg_time_to_live_ready_s": 6.097,
  "avg_segment_interval_s": 3.222,
  "wall_time_s": 60.3,
  "gpu": {
    "avg_util_pct": 40.83,
    "peak_util_pct": 100.0,
    "peak_memory_used_mb": 12520.0
  }
}
```

Interpretation:

- the pipeline is bursty: prep, GPU batch, compose, encode, wait, repeat
- `peak_util_pct=100` shows the GPU still saturates at important moments
- average utilization is diluted by non-GPU stages and idle gaps
- VRAM still is not the first limit being hit

### Cause 7: No Compute-Aware Admission Control

The current HLS stack can accept enough simultaneous work to push every stream into a throttled regime. Memory-based admission is not the same thing as throughput-based admission.

What is still missing:

- a measured per-node generation budget
- admission based on active `musetalk_fps` demand
- a clear policy for queueing or rejecting new starts when the steady-state budget would be exceeded

This matters even more under real user traffic, because staggered arrivals reduce burst pain but do not change the steady-state GPU ceiling once multiple streams are active.

### Cause 8: `torch.compile` Is Still Environment-Sensitive

`torch.compile` remains the most promising software-only way to raise the current throughput ceiling, but it is not yet a guaranteed win on every runtime stack.

What was fixed:

- dtype values are now captured before compilation
- duplicate compile calls were removed
- failed compile warmup restores the eager models instead of leaving broken compiled modules installed
- compiled HLS execution uses `torch.no_grad()`

What is still true:

- some environments still fail during compile warmup with errors such as `Inference tensors do not track version counter`
- compile success remains sensitive to the exact PyTorch + CUDA + diffusers combination
- throughput gains still need to be measured on the target machine

So `torch.compile` is no longer a correctness blocker, but it is still an unstable optimization until validated end-to-end.

### Startup Latency Is Its Own Bottleneck

Recent runs also made it clear that startup delay and steady-state cadence are separate problems.

When stream count rises, time-to-first-video rises because each stream must pass through:

1. session creation / idle-HLS preparation
2. audio prep and feature extraction
3. scheduler queue wait
4. compose and first-chunk encode
5. browser startup after the first live chunk exists

That is why a system can show moderate average GPU usage and still have poor `avg_time_to_live_ready_s`.

This also explains why staggered-arrival tests are important:

- they are more realistic than synchronized bursts
- they often look better at startup
- but they do not change the steady-state aggregate GPU ceiling once the active streams overlap

The concrete mitigation plan now lives in `gpu_allocation_improvement.md`:

- HLS tuning cheat sheet for the current env vars
- a dedicated startup-optimization phase focused on partial prep, first-chunk priority, warm paths, and adaptive prep admission

### March 15 Persistent Encoder Experiment

The next major experiment replaced per-segment ffmpeg spawning with a continuous per-request ffmpeg process.

What changed:

1. HLS generation fed frames into a persistent encoder instead of creating one ffmpeg process per segment.
2. A simple NVENC session pool was added so overflow streams would start on `libx264` instead of crashing.
3. Additional scheduler metrics were added for frame-buffer depth, pending compose/encode work, and post-generation drain time.

What this experiment proved:

1. It did **not** materially improve the steady-state throughput ceiling on the RTX 3090.
2. After the serving-layer fixes, a representative `concurrency=8`, `24/12`, `batch_size=2` run landed back around:
   - `avg_segment_interval_s ≈ 5.223`
   - `wall_time_s ≈ 96.3`
   - `avg_time_to_live_ready_s ≈ 8.733`
3. Therefore the old hypothesis "per-segment ffmpeg spawn is the main cause of the `concurrency=8` ceiling" is not supported by the latest measurements.

Important later update:

- after reverting the player experiments and then refactoring CPU compose plus PE handling in the scheduler, the current best `concurrency=8` run improved to `avg_segment_interval_s ≈ 2.301` and `wall_time_s ≈ 44.4`
- so the persistent encoder experiment remains a valid negative result, but it is no longer the latest performance baseline

Interpretation:

- the encode path still matters
- NVENC session handling still matters
- but the primary throughput ceiling is still elsewhere in the pipeline

### Why the HLS Player Regressed

This experiment also introduced an important HLS correctness lesson: backend throughput work and player work should be separated.

The backend **was** generating media:

1. scheduler logs reported `first chunk ready`
2. jobs completed with `chunks=4` or `chunks=5`
3. ffmpeg produced a valid `live.m3u8`

But the browser could still sit forever at `Preparing live...`.

The failure mode was a serving / player-state mismatch:

1. The new persistent ffmpeg path wrote playlist entries like `chunk_0000.ts`.
2. The actual files lived under nested request-specific directories such as `segments/{request_id}/chunk_0000.ts`.
3. Until the API serving layer rewrote or resolved those paths, the player could not fetch the segment assets.
4. After that was fixed, short streams could still complete and reset session state before the iframe cleanly revealed live.
5. Additional reveal-gating changes then made the wall/player state machine too fragile to trust for backend benchmarking.

Practical conclusion:

- if the player says `Preparing live...` forever, that does **not** mean chunks were never generated
- it can also mean the live playlist, segment path mapping, or player reveal logic regressed

### NVENC Pool And Metric Caveat

The persistent encoder work still produced two useful findings.

First:

1. A persistent per-stream NVENC design can exhaust encoder resources at `concurrency=8`.
2. An NVENC session pool avoids the `OpenEncodeSessionEx failed` / `No capable devices found` failure mode.
3. Overflow `libx264` fallback can increase CPU pressure, so it is a stability fix, not a throughput fix.

Second:

1. `avg_encode` in scheduler logs is no longer directly comparable to the older per-segment path.
2. Under the persistent path it measures frame submission / queue behavior into one long-lived ffmpeg process, not true per-segment encode duration.

### Current Recommendation

At the current state of the codebase:

1. Treat the persistent encoder experiment as informative but not yet production-ready.
2. Freeze or revert player / serving changes to the last known-good UI behavior before continuing throughput work.
3. Keep future optimization passes focused on backend-only bottlenecks first:
   - prep
   - compose
   - model-path speed

### March 15 Post-Revert Optimization Direction

After reverting the player and serving experiments, the HLS browser path is back to generating live chunks correctly. The remaining problem is buffering under concurrent load, which means the backend is still producing live media slower than playback consumes it.

The ranked backend focus areas are now:

1. **CPU compose cost** in `scripts/api_avatar.py` and `musetalk/utils/blending.py`
2. **GPU Whisper prep contention** in `scripts/hls_gpu_scheduler.py` and `musetalk/utils/audio_processor.py`
3. **Repeated PE work** inside the live scheduler
4. **Scheduler batch assembly / copy overhead**
5. **Later architectural work** such as keeping decoded VAE output on GPU longer

Practical rule:

- if `avg_segment_interval_s > segment_duration`, the player will eventually buffer
- therefore the next throughput cycle should remain backend-only until cadence improves measurably

The detailed file-level plan now lives in `gpu_allocation_improvement.md` under:

- `March 15 Post-Revert Throughput Plan`

### March 15 priority status snapshot

That plan has now progressed enough that the current status is worth recording explicitly:

| Priority | Status | Architectural implication |
| --- | --- | --- |
| 1. CPU compose optimization | ✅ Major pass complete | The ROI-only NumPy/OpenCV compose refactor materially reduced CPU-side frame assembly cost. Compose is no longer the dominant hot-path bottleneck. |
| 2. GPU Whisper prep contention | 🟡 Partially complete | HLS jobs now prepare only an initial conditioning window and backfill the rest in the background. This improved startup behavior, but Whisper encode still competes on the main GPU. |
| 3. PE precompute | ✅ Complete for current design | PE is no longer paid inside every live scheduler turn. |
| 4. Scheduler assembly / copy overhead | 🟡 Partial, low payoff so far | Tensorization and pinned/staging buffers are in place, but the latest scheduler logs still show `avg_gpu_batch ≈ 0.788-0.791s`, so this is not currently the dominant remaining lever. |
| 5. Model-path acceleration | 🟡 Partially complete | `torch.compile` is now integrated in a safer per-module form and has produced a real measured throughput gain on the RTX 3090. This is no longer hypothetical, but it is not fully complete because the pipeline still misses the strict realtime target at `concurrency=8`. |
| 6. Keep decoded output on GPU longer | ⏳ Not complete | `musetalk/models/vae.py` still forces an immediate CPU handoff, so this remains a larger future architectural option. |

Current measured state on the RTX 3090, using warm-cache `concurrency=8`, `playback_fps=24`, `musetalk_fps=12`, `batch_size=2`:

- `avg_time_to_live_ready_s ≈ 3.1-3.4`
- `avg_segment_interval_s ≈ 2.35-2.40`
- `wall_time_s ≈ 44.2-44.5`
- `avg GPU util ≈ 70-77%`

Current architectural interpretation:

1. The major March 15 gains came from removing CPU compose cost and removing PE from the live loop.
2. Incremental conditioning prep improved startup pressure but did not materially change the steady-state cadence ceiling.
3. Later scheduler staging-buffer work was safe but did not produce a clear additional throughput win.
4. The current backend is now mainly constrained by the GPU-side turn cadence of the UNet + VAE path, plus whatever per-turn overhead still remains around that path.

Therefore the next backend-only optimization step should be:

1. model-path acceleration (`torch.compile` or runtime-stack upgrade)
2. only then, if still needed, larger architectural work around VAE decode / GPU-side compose retention

### March 17 compile validation update

The Priority 5 work was continued on March 17 with a direct refactor of the compile path in `scripts/avatar_manager_parallel.py` and a supporting VAE tensor-decode helper in `musetalk/models/vae.py`.

Architectural changes:

1. UNet and VAE compile independently, instead of using an all-or-nothing restore path.
2. The compile manager now tries safer modes first (`reduce-overhead`) and emits real tracebacks when warmup fails.
3. Warmup for the VAE now runs through a tensor-returning decode path instead of the older NumPy conversion path.

A concrete compile bug was also identified and fixed during validation:

1. the first UNet warmup used a dummy latent tensor shaped like `[bs, 4, 32, 32]`
2. the MuseTalk UNet actually expects **8** input channels because the runtime input is a concatenation of masked and reference latents
3. this caused the original March 17 compile warmup failure at `conv_in`
4. after switching the warmup tensor to use the model's actual `in_channels`, the UNet warmup no longer failed immediately

Measured March 17 effect on the RTX 3090, using warm-cache `concurrency=8`, `playback_fps=24`, `musetalk_fps=12`, `batch_size=2`:

- pre-compile reference point:
  - `avg_time_to_live_ready_s ≈ 3.1-3.4`
  - `avg_segment_interval_s ≈ 2.35-2.40`
  - `wall_time_s ≈ 44.2-44.5`
  - `avg_gpu_batch ≈ 0.788-0.791s`
- compile-enabled runs:
  - run 1:
    - `avg_time_to_live_ready_s = 3.670`
    - `avg_segment_interval_s = 2.062`
    - `max_segment_interval_s = 3.562`
    - `wall_time_s = 39.1`
  - run 2:
    - `avg_time_to_live_ready_s = 3.455`
    - `avg_segment_interval_s = 2.076`
    - `max_segment_interval_s = 3.106`
    - `wall_time_s = 39.2`
- compile-enabled server-side scheduler metrics:
  - `avg_gpu_batch ≈ 0.691s`
  - `avg_compose ≈ 0.154-0.158s`
  - `avg_encode ≈ 0.770-0.806s`

Architectural interpretation after the March 17 compile tests:

1. `torch.compile` **did** improve the actual model-path throughput.
2. The most direct signal is `avg_gpu_batch`, which improved from about `~0.79s` to about `~0.69s`.
3. Because the pipeline still effectively needs about **3 scheduler turns per segment**, the segment cadence improved but remained slightly above the strict `2.0s` realtime threshold.
4. This means the bottleneck has shifted again: compose is already cheap, compile improved the model path, and the next exposed constraints are scheduler-turn math plus encode/tail jitter.

Updated implication for the roadmap:

1. Priority 5 should now be treated as **partially complete and validated**, not merely speculative.
2. The next iteration should focus on either reducing turns-per-segment or trimming the encode/tail cost now that the compiled UNet path is faster.

### March 17 scheduler policy update

The scheduler was then refactored again so the allocation loop spends spare GPU capacity on the jobs that are closest to emitting their next HLS chunk, instead of only distributing equal warmed-job slices. This is a policy change in `scripts/hls_gpu_scheduler.py`, not a model-path change.

Measured effect on the RTX 3090, again using warm-cache `concurrency=8`, `playback_fps=24`, `musetalk_fps=12`, `batch_size=2`:

- compile-enabled reference point:
  - `avg_time_to_live_ready_s ≈ 3.46-3.67`
  - `avg_segment_interval_s ≈ 2.06-2.08`
  - `wall_time_s ≈ 39.1-39.2`
  - `avg_gpu_batch ≈ 0.691s`
- post scheduler-policy runs:
  - run 1:
    - `avg_time_to_live_ready_s = 3.598`
    - `avg_segment_interval_s = 1.944`
    - `max_segment_interval_s = 3.030`
    - `wall_time_s = 38.5`
  - run 2:
    - `avg_time_to_live_ready_s = 3.458`
    - `avg_segment_interval_s = 1.984`
    - `max_segment_interval_s = 3.039`
    - `wall_time_s = 38.2`
- representative server-side scheduler timings after the policy change:
  - `avg_gpu_batch ≈ 0.675-0.676s`
  - `avg_compose ≈ 0.085-0.097s`
  - `avg_encode ≈ 0.583-0.609s`

Architectural interpretation after the scheduler-policy change:

1. This was another real throughput win, not noise.
2. The average HLS cadence at `concurrency=8` is now essentially at the realtime boundary for 1-second segments.
3. The system is no longer mainly failing on average throughput; it is now mainly failing on **worst-case latency spikes**.
4. Those spikes line up with:
   - late-stream startup skew / prep skew
   - encode and tail jitter

### March 18 shared-batch retuning update

The next architectural experiment was to treat larger combined scheduler batches as first-class compiled shapes instead of leaving everything above `32` in the "legal but not specially warmed" path.

What changed:

1. `scripts/hls_gpu_scheduler.py` now extends its fixed padded batch sizes beyond `[4, 8, 16, 32]` based on `HLS_SCHEDULER_MAX_BATCH`.
2. `scripts/avatar_manager_parallel.py` mirrors that same logic in the default compile warmup shape list, so the runtime and the compile warmup no longer disagree about higher scheduler batch sizes.
3. This makes `48` a real compile-friendly combined batch target instead of an ad-hoc actual-batch shape.

Measured effect on the RTX 3090, using warm-cache `concurrency=8`, `playback_fps=24`, `musetalk_fps=12`:

- `HLS_SCHEDULER_MAX_BATCH=32`, `batch_size=4`:
  - `avg_time_to_live_ready_s = 3.652`
  - `avg_segment_interval_s = 2.034`
  - `max_segment_interval_s = 3.183`
  - `wall_time_s = 38.9`
- `HLS_SCHEDULER_MAX_BATCH=48`, `batch_size=4`:
  - `avg_time_to_live_ready_s = 4.648`
  - `avg_segment_interval_s = 1.965`
  - `max_segment_interval_s = 3.129`
  - `wall_time_s = 39.1`
- best observed near-pass run with the same `HLS_SCHEDULER_MAX_BATCH=48`, `batch_size=4` settings:
  - `avg_time_to_live_ready_s = 4.897`
  - `avg_segment_interval_s = 1.921`
  - `max_segment_interval_s = 2.046`
  - `wall_time_s = 38.5`
- `HLS_SCHEDULER_MAX_BATCH=32`, `batch_size=8`:
  - `avg_time_to_live_ready_s = 3.682`
  - `avg_segment_interval_s = 1.891`
  - `max_segment_interval_s = 4.557`
  - `wall_time_s = 37.6`
- `HLS_SCHEDULER_MAX_BATCH=48`, `batch_size=8`:
  - `avg_time_to_live_ready_s = 4.650`
  - `avg_segment_interval_s = 1.881`
  - `max_segment_interval_s = 4.095`
  - `wall_time_s = 37.5`

Observed resource effect:

- `peak_memory_used_mb ≈ 22.3-22.5 GB`

Architectural interpretation after the March 18 retune:

1. Raising the **total combined scheduler batch** from `32` to `48` is another genuine throughput lever.
2. The most meaningful gain is in the `batch_size=4` comparison, where the 8-stream average improved from `2.034s` to `1.965s`. This puts the average 8-stream cadence effectively at the realtime boundary for 1-second segments.
3. The cost is startup spread: `avg_time_to_live_ready_s` became worse because each scheduler turn is heavier, so later jobs wait longer for their first high-value turn.
4. This result reinforces an important architectural distinction:
   - larger **total combined batch** can raise the shared throughput ceiling
   - larger per-stream **request `batch_size`** is still mainly a fairness tradeoff, not a universal speed win
5. `48` looks viable on the RTX 3090, but it also pushes VRAM close to the wall, so `64` should be treated as risky until explicitly validated.
6. The fastest observed `batch_size=4` run shows that the retuned scheduler can get extremely close to 8-stream realtime while keeping the average cadence under `1.93s`; the remaining failure mode is now mostly worst-case variance, not average throughput.
7. That variance is expected in the current design: with `HLS_SCHEDULER_MAX_BATCH=48` and request `batch_size=4`, the fair warmed round consumes `32` frames and the remaining `16` frames are assigned by the chunk-completion heuristic. Small prep/order differences can therefore change which stream gets the final push to the next chunk boundary.
8. The next clean test is an inference from these results: keep `HLS_SCHEDULER_MAX_BATCH=48`, but return request `batch_size` to `2` and measure whether the larger combined batch can be preserved while recovering better startup and fairness.

### Latest priority order

The current backend optimization order is now:

1. **Re-tune shared batch throughput** around the compiled path in `scripts/hls_gpu_scheduler.py`
2. **Reduce late-stream startup skew** in `scripts/hls_gpu_scheduler.py` and `musetalk/utils/audio_processor.py`
3. **Reduce encode / tail jitter** in `scripts/api_avatar.py`
4. **Continue model-path acceleration** only if the above still leaves a material gap
5. **Only then consider larger architectural work** to keep decoded output on GPU longer in `musetalk/models/vae.py`

### March 19 validation and architectural read

The March 18 near-pass result is intentionally preserved above:

- `HLS_SCHEDULER_MAX_BATCH=48`
- request `batch_size=4`
- `avg_segment_interval_s = 1.921`
- `max_segment_interval_s = 2.046`

That run remains architecturally important because it proves the current design can occasionally land very close to the target. The March 19 work was aimed at learning whether that near-pass could be made stable, and whether the remaining gap was mostly queue fairness, compile-shape waste, or a deeper pipeline limit.

March 19 changes:

1. `scripts/hls_gpu_scheduler.py` was refactored so encodes are staged through a scheduler-managed pending queue instead of being submitted immediately in compose-arrival order.
2. That queue adds:
   - startup `chunk 0` priority
   - lag-aware ordering
   - explicit tracking of encode submit wait vs thread-pool queue wait
3. `scripts/hls_gpu_scheduler.py` and `scripts/avatar_manager_parallel.py` were also briefly expanded so `24` and `40` became default compile-friendly shapes in addition to `4,8,16,32,48`.

Architectural result of the `24/40` shape expansion:

- with:
  - `MUSETALK_COMPILE_WARMUP_BATCHES=4,8,16,24,32,40,48`
  - `HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16,24,32,40,48`
  - `MUSETALK_COMPILE_MODE=reduce-overhead`
- the process often sat in the `~23.4-24.1 GB` VRAM band before streams even started
- peak VRAM reached about `24.2 GB`
- but the throughput band did **not** materially improve

Architectural interpretation:

1. The added intermediary shapes appear to have increased startup-resident compile / CUDA-graph / workspace memory.
2. The expanded shape set did **not** produce a durable new throughput tier.
3. Therefore the default shape expansion was rolled back, and the sparse normal operating set is again:
   - `MUSETALK_COMPILE_WARMUP_BATCHES=4,8,16,32,48`
   - `HLS_SCHEDULER_FIXED_BATCH_SIZES=4,8,16,32,48`

Representative clean March 19 runs after rolling back to the sparse shape set:

- `batch_size=4`:
  - `avg_time_to_live_ready_s = 4.018`
  - `avg_segment_interval_s = 1.986`
  - `max_segment_interval_s = 3.027`
  - `wall_time_s = 38.5`
- `batch_size=2`:
  - `avg_time_to_live_ready_s = 4.018`
  - `avg_segment_interval_s = 1.985`
  - `max_segment_interval_s = 3.096`
  - `wall_time_s = 38.9`
- later `batch_size=4` clean run:
  - `avg_time_to_live_ready_s = 4.659`
  - `avg_segment_interval_s = 1.980`
  - `max_segment_interval_s = 3.064`
  - `wall_time_s = 38.4`

Recurring March 19 startup pattern:

- first stream `live_ready` around `~3s`
- next wave around `~4s`
- final wave around `~5s`

Architectural interpretation after the March 19 clean runs:

1. The system is no longer failing because of a bad compile-shape configuration.
2. The system is no longer primarily constrained by VRAM headroom in the clean sparse-shape mode.
3. The encode fairness / backlog refactor did not materially shift the observed throughput band.
4. Request `batch_size=2` vs `4` also no longer produces a material difference in the final band.
5. Therefore the remaining limit is best understood as an **architectural startup convoy / first-segment delivery ceiling** on a saturated single-GPU pipeline, not a small policy bug.
6. In practical terms, the clean March 19 stack still stabilizes around:
   - `avg_segment_interval_s ≈ 1.98`
   - `max_segment_interval_s ≈ 3.0`
   - `avg_time_to_live_ready_s ≈ 4.0-4.7`

### Updated architectural conclusion

The roadmap should now be read this way:

1. The March 18 near-pass remains a real and preserved milestone.
2. However, repeated clean March 19 runs show that the current architecture does not sustain that near-pass reliably.
3. Additional environment tuning around request `batch_size`, worker counts, or intermediary compile shapes is no longer yielding step-change improvements.
4. If strict `12 fps` on one 24 GB GPU must remain the product target, the next meaningful work is a deeper startup-path redesign:
   - smaller or special-cased first chunk
   - dedicated first-segment fast path / reservation
   - direct measurement of first-segment delay contributors
5. If practical realtime behavior is more important than strict `12 fps`, lowering `musetalk_fps` to `10` or `8` is now the highest-ROI way to create real headroom.

### March 19 scripts-folder architectural audit

A full audit of every file under `scripts/` was completed to identify which modules are actually on the live HLS throughput path.

Architecturally active HLS path:

- `api_server.py`
- `scripts/avatar_manager_parallel.py`
- `scripts/hls_session_manager.py`
- `scripts/hls_gpu_scheduler.py`
- `scripts/api_avatar.py`

Architecturally secondary or irrelevant to the current HLS throttle:

- `scripts/avatar_cache.py` -> cold-start / cache behavior, not steady-state live throughput
- `scripts/concurrent_gpu_manager.py` -> admission control and lease bookkeeping; not the current pacing bottleneck because the HLS scheduler uses one lease per turn
- `scripts/session_manager.py` -> legacy non-HLS session path
- `scripts/webrtc_manager.py` and `scripts/webrtc_tracks.py` -> separate transport and buffering architecture
- `scripts/realtime_inference.py`, `scripts/inference.py`, and `scripts/preprocess.py` -> reference, offline, or preprocessing paths

Architectural bottleneck read from the audit:

1. The HLS stack is now best understood as a three-stage steady-state bottleneck:
   - a single shared GPU generation loop in `scripts/hls_gpu_scheduler.py`
   - CPU-side per-frame compose in `scripts/api_avatar.py`
   - CPU/process-side per-chunk encode / mux in `scripts/api_avatar.py`
2. `HLS_SCHEDULER_MAX_BATCH=48` was a real breakpoint because it gave the scheduler a meaningful chunk-completion budget after the fair warmed round.
3. However, pushing higher than `48` does not automatically create a new throughput tier, because the scheduler still allocates warmed work in per-job `batch_size` slices before spending any remaining capacity.
4. This means larger max batch values only help when there is enough useful backlog for later completion / fill rounds to consume, and they can become worse if the heavier turns increase cadence cost more than they increase useful completed work.

Updated architectural priority order:

1. **Refactor chunk encoding** in `scripts/api_avatar.py`.
   - The current HLS path still pays fresh `ffmpeg` bootstrap, audio seek, AAC encode, and TS mux setup for every chunk.
   - This is now the highest-confidence steady-state throughput target.
2. **Reduce CPU compose cost** in `scripts/api_avatar.py`.
   - `compose_frame()` remains a per-frame copy / resize / blend operation on CPU.
   - This is the next most likely place to free steady-state headroom.
3. **Refactor scheduler burst usage** in `scripts/hls_gpu_scheduler.py`.
   - Larger combined batch ceilings should be paired with logic that can spend spare capacity above request `batch_size` more effectively once every active stream has its fair minimum slice.
4. **Add better batch-stop instrumentation** in `scripts/hls_gpu_scheduler.py`.
   - The scheduler should expose why a turn stopped growing and how often actual batch sizes land at each point in the shape ladder.
5. **Only then re-open worker-count tuning**.
   - Worker pools may still matter, but the audit suggests they are downstream of the larger steady-state costs above.

Architectural takeaway:

- The next meaningful optimization cycle should shift away from env-level batch / worker experimentation.
- The highest-ROI code changes now live first in `scripts/api_avatar.py`, then in `scripts/hls_gpu_scheduler.py`.
- If those refactors still do not break the current `~2.0 avg / ~3.0 max` band, the product choice becomes clearer:
  - accept lower per-stream demand such as `musetalk_fps=10`, or
  - pursue a larger architectural change beyond the current HLS chunking path.

### March 19 encode-path architecture update

Two more backend refactors were completed and measured on the clean sparse-shape `max_batch=48` baseline:

1. reusable per-request AAC sidecars so chunk creation could use `audio=copy`
2. cached compose-state / batched compose work so CPU blend geometry and alpha preparation were not recomputed frame by frame

Architectural result:

1. Both changes were active in the logs.
2. Neither produced a new sustained throughput tier.
3. The system still stayed in the same practical band:
   - `avg_segment_interval_s ≈ 2.00-2.04`
   - `max_segment_interval_s ≈ 3.07-3.12`
   - `avg_time_to_live_ready_s ≈ 3.89-3.91`

Representative measured scheduler metrics:

- `avg_batch=48.0/48.0`
- `avg_gpu_batch≈0.98-1.02s`
- `avg_compose≈0.18-0.23s`
- `avg_encode≈0.94-1.03s`
- `avg_encode_wait≈0.001s`

Architectural read:

1. The shared scheduler is already saturating the chosen `48`-frame ceiling.
2. The encode worker pool is not the main limiter by itself.
3. The cost is now best understood as actual segment encode work plus remaining per-chunk pipeline overhead, with CPU compose still a secondary contributor.

### March 19 Option A architecture outcome

Option A was the proposed refactor to replace subprocess-per-chunk encode/mux with a true in-process PyAV path.

What was verified in the real runtime:

1. PyAV is installed and usable in `/content/py310`.
2. CPU codecs such as `libx264` and `aac` can open in-process.
3. `h264_nvenc` is not viable through this path in the current runtime because actual open / encode attempts fail, surfacing as `NotImplementedError: avcodec_open2(h264_nvenc)`.

Architectural implication:

1. A true in-process encoder would force this backend onto CPU `libx264`.
2. That would move the system away from the current NVENC-assisted design and likely make throughput worse.
3. Therefore Option A was investigated and ruled out for this machine/runtime instead of being fully landed.

### March 19 Option B architecture outcome

Option B was the persistent per-stream `ffmpeg` segmenter using NVENC for steady-state chunks.

What happened during the real `concurrency=8` runs:

1. The persistent path did work for some segments and logged `persistent-h264_nvenc`.
2. But the architecture held one long-lived NVENC-backed encoder process per stream.
3. At `8` concurrent streams, that exhausted encoder resources and caused:
   - `OpenEncodeSessionEx failed: out of memory (10)`
   - `No capable devices found`
   - `Broken pipe`
4. Runs also showed very large encode-submit delays in this mode, with some jobs reaching:
   - `avg_encode_submit_wait≈7-9s`
   - `max_encode_submit_wait≈15-16s`

Architectural implication:

1. Persistent per-stream NVENC is not a viable architecture for the `8`-stream target on this GPU/runtime.
2. The design consumes the scarce resource the system is already closest to exhausting: active NVENC encoder sessions.
3. The experiment reduced uncertainty, but it was rolled back as the default path.
4. The stable launch config now explicitly keeps:
   - `HLS_PERSISTENT_SEGMENTER=0`

### Updated architectural conclusion after Option A / Option B

These experiments narrow the architecture search space substantially:

1. Simply raising scheduler capacity above `48` is not the answer.
2. A true in-process NVENC encode path is blocked in the current runtime.
3. A persistent per-stream NVENC design is not viable at `concurrency=8`.
4. The remaining bottleneck is therefore not best framed as scheduler underfill or AAC re-encode overhead.
5. The remaining ceiling is the cost of steady-state segment production itself under the current HLS architecture.

Practical architectural consequence:

1. Keep `HLS_SCHEDULER_MAX_BATCH=48` as the normal ceiling.
2. Keep `HLS_PERSISTENT_SEGMENTER=0` as the normal operating mode.
3. Treat the remaining work as a harder encode-architecture problem, not a batch-size or worker-count problem.
