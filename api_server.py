import os
import sys
import argparse
from pathlib import Path
from typing import Optional
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import shutil
import uuid
import asyncio
import json
import re
import aiofiles
from datetime import datetime, timedelta

# Persistent cleanup tracking
CLEANUP_DB = Path("chunks/.cleanup_queue.json")

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scripts.avatar_manager_parallel import ParallelAvatarManager
from argparse import Namespace

# ============================================================================
# Pydantic Models (Request/Response schemas)
# ============================================================================

class PrepareAvatarRequest(BaseModel):
    avatar_id: str
    video_url: Optional[str] = None  # If provided, download from URL
    bbox_shift: int = 0
    batch_size: int = 20
    force_recreate: bool = False

class GenerateRequest(BaseModel):
    avatar_id: str
    audio_url: Optional[str] = None  # If provided, download from URL
    batch_size: int = 2
    output_name: Optional[str] = None
    fps: int = 25

class StatusResponse(BaseModel):
    request_id: str
    status: str
    result: Optional[dict] = None

class StatsResponse(BaseModel):
    gpu: dict
    cache: dict
    active_requests: int
    total_requests: int

# ============================================================================
# Initialize FastAPI App
# ============================================================================

app = FastAPI(
    title="MuseTalk Real-Time API",
    description="Multi-user avatar generation with smart caching",
    version="1.0.0"
)

# CORS middleware (allow all origins for testing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Global Manager Instance
# ============================================================================

manager: Optional[ParallelAvatarManager] = None

# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the avatar manager on startup"""
    global manager
    
    print("🚀 Starting MuseTalk API Server...")
    
    # Parse command line args or use defaults
    args = Namespace(
        version="v15",
        gpu_id=0,
        vae_type="sd-vae",
        unet_config="./models/musetalkV15/musetalk.json",
        unet_model_path="./models/musetalkV15/unet.pth",
        whisper_dir="./models/whisper",
        left_cheek_width=90,
        right_cheek_width=90,
        extra_margin=10,
        parsing_mode='jaw',
        audio_padding_length_left=2,
        audio_padding_length_right=2,
        result_dir='./results',
        ffmpeg_path='./ffmpeg-4.4-amd64-static/'
    )
    
    # Initialize manager
    manager = ParallelAvatarManager(args, max_concurrent_inferences=3)
    
    print("✅ MuseTalk API Server ready!")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    global manager
    if manager:
        manager.shutdown()
    print("👋 MuseTalk API Server stopped")

@app.on_event("startup")
async def start_cleanup_worker():
    """Background task to cleanup old chunks"""
    
    async def cleanup_loop():
        while True:
            await asyncio.sleep(60)  # Check every minute
            
            if not CLEANUP_DB.exists():
                continue
            
            async with aiofiles.open(CLEANUP_DB, 'r') as f:
                queue = json.loads(await f.read())
            
            now = datetime.now()
            cleaned = []
            
            for chunk_dir_str, cleanup_time_str in queue.items():
                cleanup_time = datetime.fromisoformat(cleanup_time_str)
                
                if now >= cleanup_time:
                    chunk_dir = Path(chunk_dir_str)
                    if chunk_dir.exists():
                        try:
                            shutil.rmtree(chunk_dir)
                            print(f"🗑️  Cleaned: {chunk_dir.name}")
                            cleaned.append(chunk_dir_str)
                        except Exception as e:
                            print(f"⚠️  Cleanup failed: {chunk_dir.name} - {e}")
            
            # Update queue
            if cleaned:
                for key in cleaned:
                    del queue[key]
                
                async with aiofiles.open(CLEANUP_DB, 'w') as f:
                    await f.write(json.dumps(queue, indent=2))
    
    asyncio.create_task(cleanup_loop())

# ============================================================================
# Health Check
# ============================================================================

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "MuseTalk Real-Time API",
        "version": "1.0.0"
    }

@app.get("/ui", response_class=HTMLResponse)
async def streaming_interface():
    """Streaming interface with instant chunk playback (zero buffering)"""
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>MuseTalk Streaming Interface</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: #fff;
                border-radius: 16px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                overflow: hidden;
            }
            
            .header {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 30px;
                text-align: center;
            }
            
            .header h1 {
                font-size: 2.5rem;
                margin-bottom: 10px;
            }
            
            .header p {
                font-size: 1.1rem;
                opacity: 0.9;
            }
            
            .content {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 30px;
                padding: 30px;
            }
            
            @media (max-width: 968px) {
                .content {
                    grid-template-columns: 1fr;
                }
            }
            
            .panel {
                background: #f8f9fa;
                padding: 25px;
                border-radius: 12px;
                border: 1px solid #e0e0e0;
            }
            
            .panel h2 {
                color: #333;
                margin-bottom: 20px;
                font-size: 1.5rem;
                border-bottom: 3px solid #667eea;
                padding-bottom: 10px;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            label {
                display: block;
                margin-bottom: 8px;
                font-weight: 600;
                color: #555;
            }
            
            input[type="text"],
            input[type="number"],
            select {
                width: 100%;
                padding: 12px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 1rem;
                transition: border-color 0.3s;
            }
            
            input[type="text"]:focus,
            input[type="number"]:focus,
            select:focus {
                outline: none;
                border-color: #667eea;
            }
            
            input[type="file"] {
                width: 100%;
                padding: 12px;
                border: 2px dashed #667eea;
                border-radius: 8px;
                background: white;
                cursor: pointer;
                transition: background 0.3s;
            }
            
            input[type="file"]:hover {
                background: #f0f4ff;
            }
            
            .btn {
                width: 100%;
                padding: 15px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 1.1rem;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
                margin-top: 10px;
            }
            
            .btn:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(102, 126, 234, 0.4);
            }
            
            .btn:disabled {
                background: #ccc;
                cursor: not-allowed;
            }
            
            .status-box {
                background: white;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #667eea;
                margin-bottom: 20px;
            }
            
            .status-box h3 {
                color: #667eea;
                margin-bottom: 10px;
            }
            
            .time-stats {
                display: flex;
                justify-content: space-between;
                margin: 10px 0;
                padding: 10px;
                background: #f8f9fa;
                border-radius: 6px;
                font-size: 0.9rem;
            }
            
            .time-stat {
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            
            .time-stat-label {
                color: #666;
                font-size: 0.8rem;
                margin-bottom: 4px;
            }
            
            .time-stat-value {
                color: #667eea;
                font-weight: 700;
                font-size: 1.2rem;
                font-family: 'Courier New', monospace;
            }
            
            .progress-bar {
                width: 100%;
                height: 30px;
                background: #e0e0e0;
                border-radius: 15px;
                overflow: hidden;
                margin: 15px 0;
                position: relative;
            }
            
            .progress-fill {
                height: 100%;
                background: linear-gradient(90deg, #667eea, #764ba2);
                transition: width 0.3s ease;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-weight: 600;
            }
            
            .chunk-list {
                max-height: 300px;
                overflow-y: auto;
                margin-top: 15px;
            }
            
            .chunk-item {
                background: #f8f9fa;
                padding: 12px;
                margin: 8px 0;
                border-radius: 8px;
                border-left: 4px solid #4CAF50;
                display: flex;
                justify-content: space-between;
                align-items: center;
                animation: slideIn 0.3s ease;
            }
            
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateX(-20px);
                }
                to {
                    opacity: 1;
                    transform: translateX(0);
                }
            }
            
            .chunk-item a {
                color: #667eea;
                text-decoration: none;
                font-weight: 600;
                padding: 5px 10px;
                border-radius: 5px;
                transition: background 0.2s;
            }
            
            .chunk-item a:hover {
                background: #e0e7ff;
            }
            
            #videoPlayer {
                width: 100%;
                border-radius: 8px;
                box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                background: #000;
                display: block;
            }
            
            .hidden {
                display: none;
            }
            
            .spinner {
                border: 3px solid #f3f3f3;
                border-top: 3px solid #667eea;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                animation: spin 1s linear infinite;
                display: inline-block;
                margin-right: 10px;
            }
            
            @keyframes spin {
                0% { transform: rotate(0deg); }
                100% { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🎬 MuseTalk Streaming Interface</h1>
                <p>Upload audio and watch your avatar come to life in real-time</p>
            </div>
            
            <div class="content">
                <!-- Left Panel -->
                <div class="panel">
                    <h2>📤 Upload & Configure</h2>
                    
                    <form id="uploadForm">
                        <div class="form-group">
                            <label for="avatar_id">🎭 Avatar ID</label>
                            <input type="text" id="avatar_id" value="test_avatar" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="audio_file">🎵 Audio File</label>
                            <input type="file" id="audio_file" accept="audio/*,.mpga,.wav,.mp3" required>
                        </div>
                        
                        <div class="form-group">
                            <label for="fps">🎞️ Frame Rate (FPS)</label>
                            <select id="fps">
                                <option value="15">15 fps (Fast)</option>
                                <option value="20">20 fps (Balanced)</option>
                                <option value="25" selected>25 fps (High Quality)</option>
                            </select>
                        </div>
                        
                        <div class="form-group">
                            <label for="chunk_duration">⏱️ Chunk Duration (seconds)</label>
                            <input type="number" id="chunk_duration" value="2" min="1" max="5" step="0.5">
                        </div>
                        
                        <div class="form-group">
                            <label for="batch_size">🔢 Batch Size</label>
                            <select id="batch_size">
                                <option value="2" selected>2 (Recommended)</option>
                                <option value="4">4 (Faster)</option>
                                <option value="8">8 (Fastest)</option>
                            </select>
                        </div>
                        
                        <button type="submit" class="btn" id="submitBtn">
                            🚀 Start Generation
                        </button>
                    </form>
                </div>
                
                <!-- Right Panel -->
                <div class="panel">
                    <h2>📊 Progress & Preview</h2>
                    
                    <div class="status-box">
                        <h3 id="statusText">Ready to start</h3>
                        
                        <div class="time-stats" id="timeStats" style="display: none;">
                            <div class="time-stat">
                                <span class="time-stat-label">⏱️ Elapsed</span>
                                <span class="time-stat-value" id="elapsedTime">00:00</span>
                            </div>
                            <div class="time-stat">
                                <span class="time-stat-label">⚡ Speed</span>
                                <span class="time-stat-value" id="processingSpeed">--</span>
                            </div>
                            <div class="time-stat">
                                <span class="time-stat-label">🎯 ETA</span>
                                <span class="time-stat-value" id="estimatedTime">--</span>
                            </div>
                        </div>
                        
                        <div class="progress-bar">
                            <div class="progress-fill" id="progress" style="width: 0%">0%</div>
                        </div>
                        <p id="statusDetail">Upload an audio file to begin</p>
                    </div>
                    
                    <div id="chunksSection" class="hidden">
                        <h3 style="margin-bottom: 10px;">📦 Generated Chunks</h3>
                        <div class="chunk-list" id="chunkList"></div>
                    </div>
                    
                    <div id="videoSection" class="hidden">
                        <h3 style="margin-bottom: 10px;">🎥 Live Preview</h3>
                        <video id="videoPlayer" autoplay muted></video>
                        <p id="videoInfo" style="margin-top: 10px; text-align: center; color: #666;">Waiting for chunks...</p>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            let state = {
                chunks: [],
                currentChunk: 0,
                requestId: '',
                totalChunks: 0,
                isStreaming: false,
                startTime: null,
                elapsedTimer: null,
                fps: 25
            };
            
            const form = document.getElementById('uploadForm');
            const submitBtn = document.getElementById('submitBtn');
            const statusText = document.getElementById('statusText');
            const statusDetail = document.getElementById('statusDetail');
            const progress = document.getElementById('progress');
            const chunkList = document.getElementById('chunkList');
            const chunksSection = document.getElementById('chunksSection');
            const videoSection = document.getElementById('videoSection');
            const videoPlayer = document.getElementById('videoPlayer');
            const videoInfo = document.getElementById('videoInfo');
            const timeStats = document.getElementById('timeStats');
            const elapsedTime = document.getElementById('elapsedTime');
            const processingSpeed = document.getElementById('processingSpeed');
            const estimatedTime = document.getElementById('estimatedTime');
            
            // ✅ ZERO-BUFFERING CHUNK PLAYER
            class InstantChunkPlayer {
                constructor(videoElement) {
                    this.video = videoElement;
                    this.chunkQueue = [];
                    this.blobCache = new Map(); // Cache blob URLs
                    this.currentIndex = 0;
                    this.isPlaying = false;
                    this.isPreloading = false;
                }
                
                async addChunk(chunkUrl) {
                    console.log(`📥 Adding chunk ${this.chunkQueue.length + 1}: ${chunkUrl}`);
                    
                    // Add to queue
                    this.chunkQueue.push(chunkUrl);
                    
                    // Preload immediately (don't wait for previous chunk)
                    this.preloadChunk(chunkUrl);
                    
                    // Start playing if not already playing
                    if (!this.isPlaying) {
                        await this.playNext();
                    }
                }
                
                async preloadChunk(chunkUrl) {
                    if (this.blobCache.has(chunkUrl)) {
                        return; // Already preloaded
                    }
                    
                    try {
                        console.log(`⬇️  Preloading: ${chunkUrl}`);
                        const response = await fetch(chunkUrl);
                        const blob = await response.blob();
                        const blobUrl = URL.createObjectURL(blob);
                        
                        this.blobCache.set(chunkUrl, blobUrl);
                        console.log(`✅ Preloaded: ${chunkUrl} → ${blobUrl}`);
                    } catch (err) {
                        console.error(`❌ Failed to preload ${chunkUrl}:`, err);
                    }
                }
                
                async playNext() {
                    if (this.currentIndex >= this.chunkQueue.length) {
                        this.isPlaying = false;
                        console.log('✅ All chunks played');
                        videoInfo.textContent = '✅ All chunks played!';
                        return;
                    }
                    
                    this.isPlaying = true;
                    const chunkUrl = this.chunkQueue[this.currentIndex];
                    
                    // Wait for chunk to be preloaded
                    while (!this.blobCache.has(chunkUrl)) {
                        console.log(`⏳ Waiting for preload: ${chunkUrl}`);
                        await new Promise(resolve => setTimeout(resolve, 100));
                    }
                    
                    const blobUrl = this.blobCache.get(chunkUrl);
                    console.log(`▶️  Playing chunk ${this.currentIndex + 1}/${this.chunkQueue.length}: ${blobUrl}`);
                    
                    // ✅ INSTANT SWITCH - Use preloaded blob URL
                    this.video.src = blobUrl;
                    this.video.load(); // Force immediate load
                    
                    try {
                        await this.video.play();
                        videoInfo.textContent = `Playing chunk ${this.currentIndex + 1} of ${this.chunkQueue.length}`;
                    } catch (e) {
                        console.warn('Play prevented:', e);
                    }
                    
                    // Preload next 2 chunks aggressively
                    if (this.currentIndex + 1 < this.chunkQueue.length) {
                        this.preloadChunk(this.chunkQueue[this.currentIndex + 1]);
                    }
                    if (this.currentIndex + 2 < this.chunkQueue.length) {
                        this.preloadChunk(this.chunkQueue[this.currentIndex + 2]);
                    }
                    
                    // When chunk ends, play next immediately
                    this.video.onended = () => {
                        console.log(`✅ Chunk ${this.currentIndex + 1} finished`);
                        this.currentIndex++;
                        this.playNext();
                    };
                }
                
                reset() {
                    // Cleanup blob URLs
                    for (const blobUrl of this.blobCache.values()) {
                        URL.revokeObjectURL(blobUrl);
                    }
                    
                    this.chunkQueue = [];
                    this.blobCache.clear();
                    this.currentIndex = 0;
                    this.isPlaying = false;
                    this.video.src = '';
                }
            }
            
            let chunkPlayer = new InstantChunkPlayer(videoPlayer);
            
            function formatTime(seconds) {
                const mins = Math.floor(seconds / 60);
                const secs = Math.floor(seconds % 60);
                return `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
            }
            
            function updateElapsedTime() {
                if (!state.startTime) return;
                
                const elapsed = (Date.now() - state.startTime) / 1000;
                elapsedTime.textContent = formatTime(elapsed);
                
                if (state.chunks.length > 0) {
                    const speed = state.chunks.length / elapsed;
                    processingSpeed.textContent = speed.toFixed(2) + ' c/s';
                    
                    if (state.totalChunks > 0 && speed > 0) {
                        const remaining = state.totalChunks - state.chunks.length;
                        const eta = remaining / speed;
                        estimatedTime.textContent = formatTime(eta);
                    }
                }
            }
            
            function startTimer() {
                state.startTime = Date.now();
                timeStats.style.display = 'flex';
                state.elapsedTimer = setInterval(updateElapsedTime, 1000);
            }
            
            function stopTimer() {
                if (state.elapsedTimer) {
                    clearInterval(state.elapsedTimer);
                    state.elapsedTimer = null;
                }
            }
            
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                state = { 
                    chunks: [], 
                    currentChunk: 0, 
                    requestId: '', 
                    totalChunks: 0, 
                    isStreaming: true,
                    startTime: null,
                    elapsedTimer: null,
                    fps: parseInt(document.getElementById('fps').value)
                };
                
                chunkList.innerHTML = '';
                chunksSection.classList.add('hidden');
                videoSection.classList.add('hidden');
                timeStats.style.display = 'none';
                
                chunkPlayer.reset();
                
                submitBtn.disabled = true;
                submitBtn.innerHTML = '<span class="spinner"></span>Processing...';
                statusText.textContent = 'Uploading audio...';
                progress.style.width = '0%';
                progress.textContent = '0%';
                
                const formData = new FormData();
                formData.append('audio_file', document.getElementById('audio_file').files[0]);
                
                const params = new URLSearchParams({
                    avatar_id: document.getElementById('avatar_id').value,
                    batch_size: document.getElementById('batch_size').value,
                    fps: state.fps,
                    chunk_duration: document.getElementById('chunk_duration').value
                });
                
                try {
                    const response = await fetch(`/generate/stream?${params}`, {
                        method: 'POST',
                        body: formData
                    });
                    
                    if (!response.ok) {
                        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
                    }
                    
                    startTimer();
                    statusText.textContent = 'Generating video chunks...';
                    chunksSection.classList.remove('hidden');
                    videoSection.classList.remove('hidden');
                    
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = '';
                    
                    while (true) {
                        const { done, value } = await reader.read();
                        if (done) break;
                        
                        buffer += decoder.decode(value, { stream: true });
                        const lines = buffer.split('\\n\\n');
                        buffer = lines.pop();
                        
                        for (const line of lines) {
                            if (line.startsWith('data: ')) {
                                const data = JSON.parse(line.slice(6));
                                await handleSSEEvent(data);
                            }
                        }
                    }
                } catch (error) {
                    stopTimer();
                    statusText.textContent = '❌ Error occurred';
                    statusDetail.textContent = error.message;
                    submitBtn.disabled = false;
                    submitBtn.textContent = '🚀 Start Generation';
                }
            });
            
            async function handleSSEEvent(data) {
                if (data.event === 'chunk') {
                    state.chunks.push(data.url);
                    state.requestId = data.url.split('/')[2];
                    state.totalChunks = data.total_chunks || state.chunks.length;
                    
                    const percent = Math.round((state.chunks.length / state.totalChunks) * 100);
                    progress.style.width = percent + '%';
                    progress.textContent = percent + '%';
                    
                    statusText.textContent = `Chunk ${state.chunks.length} of ${state.totalChunks}`;
                    statusDetail.textContent = `Processing... ${state.chunks.length}/${state.totalChunks} chunks ready`;
                    
                    const chunkDiv = document.createElement('div');
                    chunkDiv.className = 'chunk-item';
                    chunkDiv.innerHTML = `
                        <span>✅ Chunk ${data.index + 1}</span>
                        <a href="${data.url}" target="_blank">Download</a>
                    `;
                    chunkList.appendChild(chunkDiv);
                    
                    // ✅ Add to instant player
                    await chunkPlayer.addChunk(data.url);
                }
                else if (data.event === 'complete') {
                    stopTimer();
                    state.isStreaming = false;
                    
                    const totalTime = (Date.now() - state.startTime) / 1000;
                    statusText.textContent = '✅ Generation Complete!';
                    statusDetail.textContent = `All ${state.chunks.length} chunks generated in ${formatTime(totalTime)}`;
                    progress.style.width = '100%';
                    progress.textContent = '100%';
                    
                    submitBtn.disabled = false;
                    submitBtn.textContent = '🚀 Start Generation';
                }
                else if (data.event === 'error') {
                    stopTimer();
                    statusText.textContent = '❌ Error';
                    statusDetail.textContent = data.message;
                    submitBtn.disabled = false;
                    submitBtn.textContent = '🚀 Start Generation';
                }
            }
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

@app.get("/health")
async def health_check():
    """Detailed health check"""
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not initialized")
    
    stats = manager.get_stats()
    return {
        "status": "healthy",
        "gpu_available": stats['gpu']['free_gb'] > 1,
        "cached_avatars": stats['cache']['cached_avatars'],
        "active_requests": stats['active_requests']
    }

# ============================================================================
# Avatar Management Endpoints
# ============================================================================

@app.post("/avatars/prepare")
async def prepare_avatar(
    avatar_id: str,
    video_file: UploadFile = File(...),
    bbox_shift: int = 0,
    batch_size: int = 20,
    force_recreate: bool = False
):
    """
    Prepare an avatar from uploaded video file.
    This is a one-time operation that saves materials to disk.
    """
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not initialized")
    
    try:
        # Save uploaded video to temp location
        upload_dir = Path("./uploads/videos")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        video_path = upload_dir / f"{avatar_id}_{video_file.filename}"
        
        with video_path.open("wb") as buffer:
            shutil.copyfileobj(video_file.file, buffer)
        
        print(f"📥 Uploaded video for {avatar_id}: {video_path}")
        
        # Prepare avatar (blocks until complete)
        manager.prepare_avatar(
            avatar_id=avatar_id,
            video_path=str(video_path),
            bbox_shift=bbox_shift,
            batch_size=batch_size,
            force_recreate=force_recreate
        )
        
        return {
            "status": "success",
            "avatar_id": avatar_id,
            "message": f"Avatar {avatar_id} prepared successfully"
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/avatars/list")
async def list_avatars():
    """List all prepared avatars"""
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not initialized")
    
    # Get avatars from disk
    if manager.args.version == "v15":
        avatars_dir = Path(f"./results/{manager.args.version}/avatars")
    else:
        avatars_dir = Path("./results/avatars")
    
    if not avatars_dir.exists():
        return {"avatars": []}
    
    avatars = []
    for avatar_dir in avatars_dir.iterdir():
        if avatar_dir.is_dir():
            avatar_info_path = avatar_dir / "avator_info.json"
            if avatar_info_path.exists():
                import json
                with open(avatar_info_path) as f:
                    info = json.load(f)
                avatars.append({
                    "avatar_id": avatar_dir.name,
                    "version": info.get("version"),
                    "bbox_shift": info.get("bbox_shift")
                })
    
    return {"avatars": avatars}

@app.delete("/avatars/{avatar_id}")
async def delete_avatar(avatar_id: str):
    """Delete an avatar from cache and optionally from disk"""
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not initialized")
    
    # Evict from cache
    evicted = manager.evict_avatar(avatar_id)
    
    return {
        "status": "success",
        "evicted_from_cache": evicted,
        "message": f"Avatar {avatar_id} evicted from cache (still on disk)"
    }

# ============================================================================
# Generation Endpoints
# ============================================================================

@app.post("/generate")
async def generate_video(
    avatar_id: str,
    audio_file: UploadFile = File(...),
    batch_size: int = 2,
    fps: int = 25,
    output_name: Optional[str] = None
):
    """
    Generate video for an avatar with uploaded audio file.
    Returns request_id for tracking.
    """
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not initialized")
    
    try:
        # Save uploaded audio to temp location
        upload_dir = Path("./uploads/audio")
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        import uuid
        audio_filename = f"{uuid.uuid4().hex}_{audio_file.filename}"
        audio_path = upload_dir / audio_filename
        
        with audio_path.open("wb") as buffer:
            shutil.copyfileobj(audio_file.file, buffer)
        
        print(f"📥 Uploaded audio: {audio_path}")
        
        # ✅ PRE-CHECK: This will raise ValueError if avatar doesn't exist
        request_id = manager.generate_async(
            avatar_id=avatar_id,
            audio_path=str(audio_path),
            batch_size=batch_size,
            output_name=output_name,
            fps=fps
        )
        
        return {
            "status": "queued",
            "request_id": request_id,
            "avatar_id": avatar_id,
            "message": f"Video generation queued. Check status at /generate/{request_id}/status"
        }
    
    except ValueError as e:
        # ✅ Catch avatar not found errors
        raise HTTPException(status_code=404, detail=str(e))
    
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"❌ Error queuing generation: {error_trace}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.post("/generate/stream")
async def generate_video_streaming(
    avatar_id: str,
    audio_file: UploadFile = File(...),
    batch_size: int = 2,
    fps: int = 15,
    chunk_duration: int = 2
):
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not initialized")
    
    # Generate unique request ID
    request_id = f"{avatar_id}_req_{uuid.uuid4().hex[:8]}"
    
    # Create request-specific chunk directory
    chunk_dir = Path("chunks") / request_id
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    # Save audio
    upload_dir = Path("uploads/audio")
    upload_dir.mkdir(parents=True, exist_ok=True)
    audio_path = upload_dir / f"{request_id}.{audio_file.filename.split('.')[-1]}"
    
    with audio_path.open("wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
    
    async def event_generator():
        try:
            avatar = manager._get_or_load_avatar(avatar_id, batch_size)
            
            # ✅ FIX: Pass custom chunk directory
            for chunk_data in avatar.inference_streaming(
                audio_path=str(audio_path),
                audio_processor=manager.audio_processor,
                whisper=manager.whisper,
                timesteps=manager.timesteps,
                device=manager.device,
                fps=fps,
                chunk_duration_seconds=chunk_duration,
                chunk_output_dir=str(chunk_dir)  # ← Custom directory
            ):
                event_data = {
                    'event': 'chunk',
                    'index': chunk_data['chunk_index'],
                    'url': f"/chunks/{request_id}/chunk_{chunk_data['chunk_index']:04d}.mp4",
                    'duration': chunk_data['duration_seconds']
                }
                yield f"data: {json.dumps(event_data)}\n\n"
                await asyncio.sleep(0.01)
            
            yield f"data: {json.dumps({'event': 'complete'})}\n\n"
        
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
        finally:
            if audio_path.exists():
                audio_path.unlink()
            await schedule_cleanup(chunk_dir, delay_seconds=3600)
    
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/chunks/{request_id}/{chunk_filename}")
async def download_chunk(request_id: str, chunk_filename: str):
    """Download a specific video chunk with security validation"""
    
    # Validate request_id format (prevent path traversal)
    if not re.match(r'^[a-z0-9_]+_req_[a-f0-9]{8}$', request_id):
        raise HTTPException(status_code=400, detail="Invalid request_id format")
    
    # Validate chunk filename
    if not re.match(r'^chunk_\d{4}\.mp4$', chunk_filename):
        raise HTTPException(status_code=400, detail="Invalid chunk filename")
    
    # Construct and validate path
    chunk_path = Path("chunks") / request_id / chunk_filename
    
    # Prevent path traversal attacks
    try:
        chunk_path = chunk_path.resolve()
        chunks_base = Path("chunks").resolve()
        if not str(chunk_path).startswith(str(chunks_base)):
            raise HTTPException(status_code=403, detail="Access denied")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid path")
    
    if not chunk_path.exists():
        raise HTTPException(status_code=404, detail=f"Chunk not found: {chunk_filename}")
    
    # Check file integrity
    file_size = chunk_path.stat().st_size
    if file_size < 1024:  # Less than 1KB is suspicious
        raise HTTPException(status_code=500, detail="Chunk file corrupted")
    
    return FileResponse(
        chunk_path,
        media_type="video/mp4",
        headers={
            "Content-Disposition": f'inline; filename="{chunk_filename}"',
            "Cache-Control": "public, max-age=3600"
        }
    )

@app.get("/generate/{request_id}/status")
async def get_generation_status(request_id: str):
    """Check status of a generation request"""
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not initialized")
    
    status = manager.get_request_status(request_id)
    
    if status['status'] == 'unknown':
        raise HTTPException(status_code=404, detail="Request not found")
    
    return status

@app.get("/generate/{request_id}/download")
async def download_result(request_id: str):
    """Download the generated video"""
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not initialized")
    
    status = manager.get_request_status(request_id)
    
    if status['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Request not completed (status: {status['status']})"
        )
    
    if not status['result']['success']:
        raise HTTPException(status_code=500, detail=status['result'].get('error'))
    
    output_path = status['result']['output_path']
    
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=os.path.basename(output_path)
    )

@app.get("/player/{request_id}", response_class=HTMLResponse)
async def video_player(request_id: str):
    """Serve HTML video player for a streaming session"""
    
    # Validate request_id
    if not re.match(r'^[a-z0-9_]+_req_[a-f0-9]{8}$', request_id):
        raise HTTPException(status_code=400, detail="Invalid request_id")
    
    # Check if chunks exist
    chunk_dir = Path("chunks") / request_id
    if not chunk_dir.exists():
        raise HTTPException(status_code=404, detail="Streaming session not found")
    
    # Get all chunks
    chunks = sorted(chunk_dir.glob("chunk_*.mp4"))
    chunk_urls = [f"/chunks/{request_id}/{chunk.name}" for chunk in chunks]
    
    # Generate HTML with video player
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Video Player - {request_id}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                background: #1a1a1a;
                color: #fff;
            }}
            h1 {{
                text-align: center;
                color: #4CAF50;
            }}
            #video-container {{
                position: relative;
                margin: 20px 0;
            }}
            video {{
                width: 100%;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.3);
            }}
            #controls {{
                display: flex;
                gap: 10px;
                margin-top: 10px;
            }}
            button {{
                padding: 10px 20px;
                background: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 14px;
            }}
            button:hover {{
                background: #45a049;
            }}
            button:disabled {{
                background: #666;
                cursor: not-allowed;
            }}
            #info {{
                background: #2a2a2a;
                padding: 15px;
                border-radius: 4px;
                margin-top: 20px;
            }}
            .chunk-list {{
                max-height: 200px;
                overflow-y: auto;
                background: #333;
                padding: 10px;
                border-radius: 4px;
                margin-top: 10px;
            }}
            .chunk-item {{
                padding: 5px;
                cursor: pointer;
                border-bottom: 1px solid #444;
            }}
            .chunk-item:hover {{
                background: #444;
            }}
            .chunk-item.active {{
                background: #4CAF50;
                color: white;
            }}
        </style>
    </head>
    <body>
        <h1>🎥 Video Player</h1>
        <p style="text-align: center; color: #888;">Session: {request_id}</p>
        
        <div id="video-container">
            <video id="player" controls autoplay>
                <source src="{chunk_urls[0] if chunk_urls else ''}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        
        <div id="controls">
            <button id="prev-btn" onclick="playPrevious()">⏮️ Previous Chunk</button>
            <button id="next-btn" onclick="playNext()">Next Chunk ⏭️</button>
            <button onclick="playAll()">▶️ Play All Sequentially</button>
            <button onclick="downloadAll()">⬇️ Download All</button>
        </div>
        
        <div id="info">
            <strong>Current Chunk:</strong> <span id="current-chunk">1</span> / <span id="total-chunks">{len(chunks)}</span>
            <br>
            <strong>Duration:</strong> <span id="duration">--:--</span>
            <div class="chunk-list" id="chunk-list"></div>
        </div>
        
        <script>
            const chunks = {json.dumps(chunk_urls)};
            let currentChunkIndex = 0;
            const player = document.getElementById('player');
            
            // Update UI
            function updateUI() {{
                document.getElementById('current-chunk').textContent = currentChunkIndex + 1;
                document.getElementById('prev-btn').disabled = currentChunkIndex === 0;
                document.getElementById('next-btn').disabled = currentChunkIndex === chunks.length - 1;
                
                // Update chunk list
                const chunkList = document.getElementById('chunk-list');
                chunkList.innerHTML = chunks.map((url, i) => `
                    <div class="chunk-item ${{i === currentChunkIndex ? 'active' : ''}}" 
                         onclick="playChunk(${{i}})">
                        Chunk ${{i + 1}}: ${{url.split('/').pop()}}
                    </div>
                `).join('');
            }}
            
            // Play specific chunk
            function playChunk(index) {{
                if (index < 0 || index >= chunks.length) return;
                currentChunkIndex = index;
                player.src = chunks[index];
                player.play();
                updateUI();
            }}
            
            // Navigation
            function playNext() {{
                playChunk(currentChunkIndex + 1);
            }}
            
            function playPrevious() {{
                playChunk(currentChunkIndex - 1);
            }}
            
            // Auto-play next chunk when current ends
            player.addEventListener('ended', () => {{
                if (currentChunkIndex < chunks.length - 1) {{
                    playNext();
                }}
            }});
            
            // Play all chunks sequentially
            function playAll() {{
                playChunk(0);
            }}
            
            // Download all chunks
            function downloadAll() {{
                chunks.forEach((url, i) => {{
                    const a = document.createElement('a');
                    a.href = url;
                    a.download = `chunk_${{String(i).padStart(4, '0')}}.mp4`;
                    document.body.appendChild(a);
                    a.click();
                    document.body.removeChild(a);
                }});
            }}
            
            // Update duration
            player.addEventListener('loadedmetadata', () => {{
                const duration = player.duration;
                const mins = Math.floor(duration / 60);
                const secs = Math.floor(duration % 60);
                document.getElementById('duration').textContent = 
                    `${{String(mins).padStart(2, '0')}}:${{String(secs).padStart(2, '0')}}`;
            }});
            
            // Initialize
            updateUI();
        </script>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content)

# ============================================================================
# Statistics Endpoints
# ============================================================================

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not initialized")
    
    return manager.get_stats()

@app.get("/stats/cache")
async def get_cache_stats():
    """Get cache-specific statistics"""
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not initialized")
    
    stats = manager.get_stats()
    return stats['cache']

# ============================================================================
# Utility Functions
# ============================================================================

async def schedule_cleanup(chunk_dir: Path, delay_seconds: int = 3600):
    """Schedule cleanup of chunk directory"""
    cleanup_time = (datetime.now() + timedelta(seconds=delay_seconds)).isoformat()
    
    # Load existing queue
    if CLEANUP_DB.exists():
        async with aiofiles.open(CLEANUP_DB, 'r') as f:
            queue = json.loads(await f.read())
    else:
        queue = {}
    
    # Add to queue
    queue[str(chunk_dir)] = cleanup_time
    
    # Save queue
    CLEANUP_DB.parent.mkdir(exist_ok=True)
    async with aiofiles.open(CLEANUP_DB, 'w') as f:
        await f.write(json.dumps(queue, indent=2))
    
    print(f"🗑️  Scheduled cleanup: {chunk_dir.name} at {cleanup_time}")

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MuseTalk API Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload (dev mode)")
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api_server:app",
        host=args.host,
        port=args.port,
        reload=args.reload
    )