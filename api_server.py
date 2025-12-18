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
from templates.streaming_ui import streaming_ui_endpoint
from templates.mobile_player import mobile_player_endpoint
from templates.session_player import get_session_player_html  # ✅ ADD THIS
from scripts.session_manager import SessionManager

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
session_manager: Optional[SessionManager] = None  # ✅ NEW

# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize the avatar manager on startup"""
    global manager, session_manager
    
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
    manager = ParallelAvatarManager(args, max_concurrent_inferences=5)
    
    # ✅ Initialize session manager
    session_manager = SessionManager(session_ttl_seconds=3600)
    session_manager.start_cleanup()
    
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
    """Streaming interface with instant chunk playback"""
    return await streaming_ui_endpoint()


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
async def delete_avatar(avatar_id: str, from_disk: bool = False):
    """
    Delete an avatar from cache and optionally from disk.
    
    Args:
        avatar_id: Avatar identifier
        from_disk: If True, permanently delete from disk (WARNING: irreversible!)
    """
    if manager is None:
        raise HTTPException(status_code=503, detail="Manager not initialized")
    
    # 1. Evict from cache first
    evicted_from_cache = manager.evict_avatar(avatar_id)
    
    deleted_from_disk = False
    disk_path = None
    
    # 2. Delete from disk if requested
    if from_disk:
        if manager.args.version == "v15":
            avatar_dir = Path(f"./results/{manager.args.version}/avatars/{avatar_id}")
        else:
            avatar_dir = Path(f"./results/avatars/{avatar_id}")
        
        if avatar_dir.exists():
            try:
                import shutil
                shutil.rmtree(avatar_dir)
                deleted_from_disk = True
                disk_path = str(avatar_dir)
                print(f"🗑️  Deleted avatar from disk: {disk_path}")
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to delete from disk: {str(e)}"
                )
        else:
            raise HTTPException(
                status_code=404,
                detail=f"Avatar {avatar_id} not found on disk"
            )
    
    return {
        "status": "success",
        "avatar_id": avatar_id,
        "evicted_from_cache": evicted_from_cache,
        "deleted_from_disk": deleted_from_disk,
        "disk_path": disk_path,
        "message": (
            f"Avatar {avatar_id} permanently deleted"
            if deleted_from_disk
            else f"Avatar {avatar_id} evicted from cache (still on disk)"
        )
    }

@app.get("/avatars/{avatar_id}/video")
async def get_avatar_placeholder_video(avatar_id: str):
    """
    Serve the original avatar video as placeholder.
    This loops continuously until audio is uploaded.
    """
    # Hardcoded for testing - use the specific uploaded video
    video_file = Path("uploads/videos/test_avatar_ai_test_default_moving_vid.mp4")
    
    if not video_file.exists():
        raise HTTPException(
            status_code=404, 
            detail=f"Placeholder video not found at {video_file}"
        )
    
    return FileResponse(
        video_file,
        media_type="video/mp4",
        headers={
            "Accept-Ranges": "bytes",
            "Cache-Control": "public, max-age=3600"
        }
    )

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
    
    # ✅ Get event loop BEFORE creating thread
    main_loop = asyncio.get_event_loop()
    
    # ✅ Create async queue for chunk communication
    chunk_queue = asyncio.Queue()
    
    # ✅ Worker function that runs in ThreadPoolExecutor
    def streaming_worker():
        """
        Runs in thread pool - allows up to 5 concurrent streaming requests.
        """
        try:
            print(f"🎬 [{request_id}] Starting streaming generation")
            
            # ✅ Allocate GPU memory (blocks if not enough)
            with manager.gpu_memory.allocate(batch_size):
                # Get avatar from cache/disk
                avatar = manager._get_or_load_avatar(avatar_id, batch_size)
                
                # ✅ Generate chunks (no model_lock needed - handled by gpu_memory.allocate)
                for chunk_info in avatar.inference_streaming(
                    audio_path=str(audio_path),
                    audio_processor=manager.audio_processor,
                    whisper=manager.whisper,
                    timesteps=manager.timesteps,
                    device=manager.device,
                    fps=fps,
                    chunk_duration_seconds=chunk_duration,
                    chunk_output_dir=str(chunk_dir)
                ):
                    # ✅ Put chunk in async queue (thread-safe with correct loop)
                    asyncio.run_coroutine_threadsafe(
                        chunk_queue.put(chunk_info),
                        main_loop  # ✅ Use the captured loop reference
                    ).result()  # ✅ Wait for it to be queued
            
            # Signal completion
            asyncio.run_coroutine_threadsafe(
                chunk_queue.put({'event': 'complete'}),
                main_loop
            ).result()
            
            print(f"✅ [{request_id}] Streaming generation complete")
        
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"❌ [{request_id}] Error: {error_detail}")
            
            # Put error in queue
            try:
                asyncio.run_coroutine_threadsafe(
                    chunk_queue.put({'event': 'error', 'message': str(e)}),
                    main_loop
                ).result(timeout=1.0)
            except Exception:
                pass  # Queue might be closed
    
    # ✅ Submit to ThreadPoolExecutor (enables parallelism!)
    future = main_loop.run_in_executor(manager.executor, streaming_worker)
    
    # ✅ Track this streaming request
    manager.active_requests[request_id] = {
        'avatar_id': avatar_id,
        'status': 'streaming',
        'future': future,
        'type': 'stream'
    }
    
    print(f"📥 [{request_id}] Queued for streaming (active requests: {len(manager.active_requests)})")
    
    # ✅ Event generator that reads from queue
    async def event_generator():
        try:
            chunk_count = 0
            
            # ✅ Read chunks from queue as they arrive
            while True:
                # Wait for next chunk or completion
                chunk_info = await chunk_queue.get()
                
                # Check for completion
                if chunk_info.get('event') == 'complete':
                    yield f"data: {json.dumps({'event': 'complete', 'total_chunks': chunk_count})}\n\n"
                    break
                
                # Check for error
                if chunk_info.get('event') == 'error':
                    yield f"data: {json.dumps(chunk_info)}\n\n"
                    break
                
                # Normal chunk - construct public URL
                chunk_filename = Path(chunk_info['chunk_path']).name
                chunk_url = f"/chunks/{request_id}/{chunk_filename}"
                
                event_data = {
                    'event': 'chunk',
                    'index': chunk_info['chunk_index'],
                    'url': chunk_url,
                    'total_chunks': chunk_info['total_chunks'],
                    'duration': chunk_info['duration_seconds'],
                    'creation_time': chunk_info['creation_time']
                }
                
                # ✅ Yield immediately
                yield f"data: {json.dumps(event_data)}\n\n"
                
                chunk_count += 1
                print(f"    📤 [{request_id}] Sent chunk {chunk_count} to client")
        
        except asyncio.CancelledError:
            print(f"⚠️ [{request_id}] Client disconnected")
            # Cancel the worker if client disconnects
            future.cancel()
        
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"❌ [{request_id}] Event generator error: {error_detail}")
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
        
        finally:
            # Schedule cleanup
            await schedule_cleanup(chunk_dir, delay_seconds=3600)
            
            # Update request status
            if request_id in manager.active_requests:
                manager.active_requests[request_id]['status'] = 'completed'
            
            print(f"🏁 [{request_id}] Stream ended")
    
    # ✅ Return streaming response
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"  # ✅ Add CORS for mobile
        }
    )

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

@app.get("/player/mobile", response_class=HTMLResponse)
async def mobile_player(session_id: str = None):
    """
    Minimal embeddable player for mobile apps.
    Each session gets isolated streaming via unique session_id.
    """
    return await mobile_player_endpoint(session_id)

# ============================================================================
# Session Management Endpoints (NEW)
# ============================================================================

@app.post("/sessions/create")
async def create_session(
    avatar_id: str,
    user_id: Optional[str] = None,
    batch_size: int = 2,
    fps: int = 15,
    chunk_duration: int = 2
):
    """
    Create a new streaming session for a user.
    Returns session_id and player URL.
    
    **Mobile App Flow:**
    1. Call this endpoint when user opens chat
    2. Get session_id and player_url
    3. Load player_url in WebView
    4. Send audio via /sessions/{session_id}/stream
    """
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    # Verify avatar exists
    if not manager._avatar_exists(avatar_id):
        raise HTTPException(
            status_code=404,
            detail=f"Avatar '{avatar_id}' not found. Please prepare it first."
        )
    
    session = await session_manager.create_session(
        avatar_id=avatar_id,
        user_id=user_id,
        batch_size=batch_size,
        fps=fps,
        chunk_duration=chunk_duration
    )
    
    return {
        'session_id': session.session_id,
        'player_url': f'/player/session/{session.session_id}',
        'avatar_id': avatar_id,
        'user_id': user_id,
        'config': {
            'batch_size': batch_size,
            'fps': fps,
            'chunk_duration': chunk_duration
        },
        'expires_in_seconds': session_manager.session_ttl
    }


@app.post("/sessions/{session_id}/stream")
async def session_stream(
    session_id: str,
    audio_file: UploadFile = File(...)
):
    """
    Start streaming for an existing session.
    This replaces the need for postMessage - just upload audio directly.
    
    **Mobile App Flow:**
    1. User records audio
    2. POST audio to this endpoint
    3. WebView automatically receives chunks via SSE
    """
    if session_manager is None or manager is None:
        raise HTTPException(status_code=503, detail="Services not initialized")
    
    # Get session
    session = await session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    # Check if already streaming
    if session.active_stream is not None:
        raise HTTPException(
            status_code=409,
            detail=f"Session already streaming (request_id: {session.active_stream})"
        )
    
    # Generate request ID
    request_id = f"{session.avatar_id}_req_{uuid.uuid4().hex[:8]}"
    session.active_stream = request_id
    
    # Save audio
    upload_dir = Path("uploads/audio")
    upload_dir.mkdir(parents=True, exist_ok=True)
    audio_path = upload_dir / f"{request_id}.{audio_file.filename.split('.')[-1]}"
    
    with audio_path.open("wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
    
    # Create chunk directory
    chunk_dir = Path("chunks") / request_id
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    # Get event loop
    main_loop = asyncio.get_event_loop()
    
    # Worker function
    def streaming_worker():
        try:
            print(f"🎬 [{request_id}] Starting streaming for session {session_id}")
            
            with manager.gpu_memory.allocate(session.batch_size):
                avatar = manager._get_or_load_avatar(session.avatar_id, session.batch_size)
                
                for chunk_info in avatar.inference_streaming(
                    audio_path=str(audio_path),
                    audio_processor=manager.audio_processor,
                    whisper=manager.whisper,
                    timesteps=manager.timesteps,
                    device=manager.device,
                    fps=session.fps,
                    chunk_duration_seconds=session.chunk_duration,
                    chunk_output_dir=str(chunk_dir)
                ):
                    # Put chunk in session queue
                    asyncio.run_coroutine_threadsafe(
                        session.chunk_queue.put(chunk_info),
                        main_loop
                    ).result()
            
            # Signal completion
            asyncio.run_coroutine_threadsafe(
                session.chunk_queue.put({'event': 'complete'}),
                main_loop
            ).result()
            
            print(f"✅ [{request_id}] Streaming complete")
        
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"❌ [{request_id}] Error: {error_detail}")
            
            try:
                asyncio.run_coroutine_threadsafe(
                    session.chunk_queue.put({'event': 'error', 'message': str(e)}),
                    main_loop
                ).result(timeout=1.0)
            except Exception:
                pass
        
        finally:
            # Clear active stream
            session.active_stream = None
    
    # Submit to executor
    future = main_loop.run_in_executor(manager.executor, streaming_worker)
    
    # Track request
    manager.active_requests[request_id] = {
        'avatar_id': session.avatar_id,
        'status': 'streaming',
        'future': future,
        'type': 'session_stream',
        'session_id': session_id
    }
    
    return {
        'request_id': request_id,
        'session_id': session_id,
        'status': 'streaming',
        'message': 'Stream started. WebView will receive chunks automatically.'
    }


@app.get("/sessions/{session_id}/events")
async def session_events(session_id: str):
    """
    SSE endpoint for WebView to receive chunks.
    This is called automatically by the player - not by your app.
    """
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    session = await session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    async def event_generator():
        try:
            chunk_count = 0
            
            while True:
                # Wait for next chunk
                chunk_info = await session.chunk_queue.get()
                
                # Check for completion
                if chunk_info.get('event') == 'complete':
                    yield f"data: {json.dumps({'event': 'complete', 'total_chunks': chunk_count})}\n\n"
                    break
                
                # Check for error
                if chunk_info.get('event') == 'error':
                    yield f"data: {json.dumps(chunk_info)}\n\n"
                    break
                
                # Normal chunk
                chunk_filename = Path(chunk_info['chunk_path']).name
                request_id = chunk_info['chunk_path'].split('/')[-2]  # Extract from path
                chunk_url = f"/chunks/{request_id}/{chunk_filename}"
                
                event_data = {
                    'event': 'chunk',
                    'index': chunk_info['chunk_index'],
                    'url': chunk_url,
                    'total_chunks': chunk_info['total_chunks'],
                    'duration': chunk_info['duration_seconds'],
                    'creation_time': chunk_info['creation_time']
                }
                
                yield f"data: {json.dumps(event_data)}\n\n"
                
                chunk_count += 1
                print(f"    📤 [{session_id}] Sent chunk {chunk_count} to client")
        
        except asyncio.CancelledError:
            print(f"⚠️ [{session_id}] Client disconnected")
        
        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            print(f"❌ [{session_id}] Event generator error: {error_detail}")
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": "*"
        }
    )


@app.get("/sessions/{session_id}/status")
async def get_session_status(session_id: str):
    """Get session status"""
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    session = await session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        'session_id': session.session_id,
        'avatar_id': session.avatar_id,
        'user_id': session.user_id,
        'active_stream': session.active_stream,
        'created_at': session.created_at,
        'last_activity': session.last_activity,
        'age_seconds': time.time() - session.created_at,
        'idle_seconds': time.time() - session.last_activity
    }


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session (call when user closes chat)"""
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    deleted = await session_manager.delete_session(session_id)
    
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {'status': 'deleted', 'session_id': session_id}


@app.get("/sessions/stats")
async def get_session_stats():
    """Get all session statistics"""
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    return session_manager.get_stats()


@app.get("/player/session/{session_id}", response_class=HTMLResponse)
async def session_player(session_id: str):
    """
    Simplified player that auto-connects to session.
    No postMessage needed - just load in WebView and upload audio to /sessions/{session_id}/stream
    """
    if session_manager is None:
        raise HTTPException(status_code=503, detail="Session manager not initialized")
    
    session = await session_manager.get_session(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    
    return HTMLResponse(content=get_session_player_html(session))

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

def cleanup_old_chunks():
    """Cleanup expired chunks (safe version with error handling)"""
    if not CLEANUP_DB.exists():
        return
    
    try:
        with open(CLEANUP_DB, 'r') as f:
            cleanup_queue = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        cleanup_queue = {}
    
    now = datetime.now()
    to_remove = []
    
    for chunk_path, expire_time_str in cleanup_queue.items():
        expire_time = datetime.fromisoformat(expire_time_str)
        
        if now >= expire_time:
            chunk_dir = Path(chunk_path)
            
            # ✅ Check if exists before deleting
            if chunk_dir.exists():
                try:
                    shutil.rmtree(chunk_dir)
                    print(f"🗑️  Cleaned up: {chunk_path}")
                except Exception as e:
                    print(f"⚠️  Failed to delete {chunk_path}: {e}")
            else:
                print(f"ℹ️  Already deleted: {chunk_path}")
            
            to_remove.append(chunk_path)
    
    # Remove from queue
    for path in to_remove:
        del cleanup_queue[path]
    
    # Save updated queue
    with open(CLEANUP_DB, 'w') as f:
        json.dump(cleanup_queue, f, indent=2)

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