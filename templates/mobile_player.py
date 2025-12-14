from fastapi.responses import HTMLResponse
import secrets

def get_mobile_player_html(session_id: str = None) -> str:
    """
    Returns minimal, embeddable video player for mobile apps.
    Each user gets a unique session ID for isolated streaming.
    """
    if not session_id:
        session_id = secrets.token_urlsafe(16)
    
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <meta name="mobile-web-app-capable" content="yes">
    <title>Avatar Player</title>
    <style>
        * {{
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            -webkit-tap-highlight-color: transparent;
        }}
        
        html, body {{
            width: 100%;
            height: 100%;
            overflow: hidden;
            background: #000;
        }}
        
        #videoPlayer {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            display: block;
        }}
        
        .loading-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.8);
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: white;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            transition: opacity 0.3s;
        }}
        
        .loading-overlay.hidden {{
            opacity: 0;
            pointer-events: none;
        }}
        
        .spinner {{
            width: 50px;
            height: 50px;
            border: 4px solid rgba(255,255,255,0.3);
            border-top-color: #fff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }}
        
        @keyframes spin {{
            to {{ transform: rotate(360deg); }}
        }}
        
        .status-text {{
            margin-top: 20px;
            font-size: 16px;
            text-align: center;
            padding: 0 20px;
        }}
        
        .error-overlay {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: rgba(0,0,0,0.9);
            display: none;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            color: white;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            padding: 20px;
            text-align: center;
        }}
        
        .error-overlay.visible {{
            display: flex;
        }}
        
        .error-title {{
            font-size: 24px;
            margin-bottom: 10px;
        }}
        
        .error-message {{
            font-size: 14px;
            color: #ccc;
        }}
    </style>
</head>
<body>
    <video id="videoPlayer" playsinline webkit-playsinline></video>
    
    <div class="loading-overlay" id="loadingOverlay">
        <div class="spinner"></div>
        <div class="status-text" id="statusText">Initializing...</div>
    </div>
    
    <div class="error-overlay" id="errorOverlay">
        <div class="error-title">⚠️ Error</div>
        <div class="error-message" id="errorMessage"></div>
    </div>
    
    <script>
        const SESSION_ID = '{session_id}';
        const videoPlayer = document.getElementById('videoPlayer');
        const loadingOverlay = document.getElementById('loadingOverlay');
        const errorOverlay = document.getElementById('errorOverlay');
        const statusText = document.getElementById('statusText');
        const errorMessage = document.getElementById('errorMessage');
        
        let mediaSource = null;
        let sourceBuffer = null;
        let pendingChunks = [];
        let isAppending = false;
        let isFirstChunk = true;
        let streamActive = false;
        
        // Mobile-specific optimizations
        videoPlayer.setAttribute('playsinline', '');
        videoPlayer.setAttribute('webkit-playsinline', '');
        videoPlayer.muted = false; // Allow audio
        
        function showError(message) {{
            errorMessage.textContent = message;
            errorOverlay.classList.add('visible');
            loadingOverlay.classList.add('hidden');
        }}
        
        function initMediaSource() {{
            mediaSource = new MediaSource();
            videoPlayer.src = URL.createObjectURL(mediaSource);
            
            mediaSource.addEventListener('sourceopen', function() {{
                console.log('✅ MediaSource opened');
                
                try {{
                    sourceBuffer = mediaSource.addSourceBuffer('video/mp4; codecs="avc1.42E01E, mp4a.40.2"');
                    sourceBuffer.mode = 'sequence';
                    
                    sourceBuffer.addEventListener('updateend', function() {{
                        isAppending = false;
                        processNextChunk();
                    }});
                    
                    sourceBuffer.addEventListener('error', function(e) {{
                        console.error('❌ SourceBuffer error:', e);
                        showError('Video buffer error. Please refresh.');
                    }});
                    
                    console.log('✅ SourceBuffer ready');
                }} catch (e) {{
                    showError('Browser not supported: ' + e.message);
                }}
            }});
        }}
        
        function processNextChunk() {{
            if (isAppending || pendingChunks.length === 0 || !sourceBuffer) {{
                return;
            }}
            
            isAppending = true;
            const chunk = pendingChunks.shift();
            
            try {{
                sourceBuffer.appendBuffer(chunk);
                
                if (isFirstChunk) {{
                    isFirstChunk = false;
                    
                    // Auto-play with audio (mobile-friendly)
                    videoPlayer.play().then(function() {{
                        console.log('▶️ Playing');
                        loadingOverlay.classList.add('hidden');
                    }}).catch(function(e) {{
                        // Fallback: show play button
                        console.warn('Autoplay blocked:', e);
                        statusText.textContent = 'Tap to play';
                        
                        videoPlayer.addEventListener('click', function() {{
                            videoPlayer.play();
                            loadingOverlay.classList.add('hidden');
                        }}, {{ once: true }});
                    }});
                }}
            }} catch (e) {{
                console.error('Append error:', e);
                isAppending = false;
            }}
        }}
        
        async function handleSSEEvent(data) {{
            if (data.event === 'chunk') {{
                statusText.textContent = 'Loading chunk ' + (data.index + 1) + '/' + data.total_chunks;
                
                try {{
                    const response = await fetch(data.url);
                    if (!response.ok) throw new Error('Chunk fetch failed');
                    
                    const arrayBuffer = await response.arrayBuffer();
                    pendingChunks.push(arrayBuffer);
                    processNextChunk();
                }} catch (e) {{
                    showError('Failed to load video chunk: ' + e.message);
                }}
            }}
            else if (data.event === 'complete') {{
                console.log('✅ Stream complete');
                
                if (!sourceBuffer.updating && pendingChunks.length === 0) {{
                    mediaSource.endOfStream();
                }}
                
                streamActive = false;
            }}
            else if (data.event === 'error') {{
                showError(data.message);
            }}
        }}
        
        // Listen for stream start from parent app
        window.addEventListener('message', async function(event) {{
            const data = event.data;
            
            if (data.type === 'START_STREAM') {{
                if (streamActive) {{
                    console.warn('Stream already active');
                    return;
                }}
                
                streamActive = true;
                statusText.textContent = 'Connecting...';
                loadingOverlay.classList.remove('hidden');
                errorOverlay.classList.remove('visible');
                
                pendingChunks = [];
                isAppending = false;
                isFirstChunk = true;
                
                initMediaSource();
                
                const params = new URLSearchParams({{
                    avatar_id: data.avatar_id || 'test_avatar',
                    batch_size: data.batch_size || '2',
                    fps: data.fps || '15',
                    chunk_duration: data.chunk_duration || '2'
                }});
                
                try {{
                    const response = await fetch('/generate/stream?' + params.toString(), {{
                        method: 'POST',
                        body: data.audioBlob
                    }});
                    
                    if (!response.ok) {{
                        throw new Error('HTTP ' + response.status);
                    }}
                    
                    statusText.textContent = 'Generating...';
                    
                    const reader = response.body.getReader();
                    const decoder = new TextDecoder();
                    let buffer = '';
                    
                    while (true) {{
                        const result = await reader.read();
                        if (result.done) break;
                        
                        buffer += decoder.decode(result.value, {{ stream: true }});
                        
                        const lines = buffer.split('\\n\\n');
                        buffer = lines.pop() || '';
                        
                        for (const line of lines) {{
                            if (line.startsWith('data: ')) {{
                                try {{
                                    const eventData = JSON.parse(line.slice(6));
                                    await handleSSEEvent(eventData);
                                }} catch (e) {{
                                    console.warn('Parse error:', e);
                                }}
                            }}
                        }}
                    }}
                }} catch (error) {{
                    showError('Connection failed: ' + error.message);
                    streamActive = false;
                }}
            }}
            else if (data.type === 'STOP_STREAM') {{
                if (mediaSource && mediaSource.readyState === 'open') {{
                    mediaSource.endOfStream();
                }}
                streamActive = false;
            }}
        }});
        
        // Notify parent that player is ready
        window.parent.postMessage({{
            type: 'PLAYER_READY',
            sessionId: SESSION_ID
        }}, '*');
        
        console.log('📱 Mobile player initialized (session: ' + SESSION_ID + ')');
    </script>
</body>
</html>
    """


async def mobile_player_endpoint(session_id: str = None):
    """FastAPI endpoint for mobile player"""
    return HTMLResponse(content=get_mobile_player_html(session_id))