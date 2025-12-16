def get_session_player_html(session) -> str:
    """
    Simplified player for session-based streaming.
    Auto-connects to SSE endpoint with placeholder video loop.
    Matches streaming_ui.py autoplay behavior EXACTLY.
    """
    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <title>Avatar - {session.avatar_id}</title>
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
        
        .video-container {{
            width: 100%;
            height: 100%;
            position: relative;
        }}
        
        video {{
            width: 100%;
            height: 100%;
            object-fit: contain;
            position: absolute;
            top: 0;
            left: 0;
        }}
        
        #placeholderVideo {{
            z-index: 1;
        }}
        
        #streamVideo {{
            z-index: 2;
            display: none;
        }}
        
        .status-overlay {{
            position: absolute;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.7);
            color: white;
            padding: 10px 20px;
            border-radius: 20px;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 14px;
            transition: opacity 0.3s;
            z-index: 10;
        }}
        
        .status-overlay.hidden {{
            opacity: 0;
            pointer-events: none;
        }}
        
        .error {{
            color: #ff5555;
        }}
    </style>
</head>
<body>
    <div class="video-container">
        <!-- Placeholder video (loops continuously) -->
        <video id="placeholderVideo" playsinline webkit-playsinline autoplay muted loop></video>
        
        <!-- Stream video (shows during playback) -->
        <video id="streamVideo" playsinline webkit-playsinline></video>
        
        <div class="status-overlay hidden" id="statusOverlay">Ready</div>
    </div>
    
    <script>
        const SESSION_ID = '{session.session_id}';
        const AVATAR_ID = '{session.avatar_id}';
        
        const placeholderVideo = document.getElementById('placeholderVideo');
        const streamVideo = document.getElementById('streamVideo');
        const statusOverlay = document.getElementById('statusOverlay');
        
        // ✅ STATE: Reset completely between streams
        let currentStream = null;  // Track active stream state
        
        // Load placeholder video
        placeholderVideo.src = `/avatars/${{AVATAR_ID}}/video`;
        placeholderVideo.load();
        
        // Auto-play placeholder
        placeholderVideo.play().then(() => {{
            console.log('✅ Placeholder playing');
        }}).catch(e => {{
            console.warn('⚠️ Autoplay blocked');
            updateStatus('Tap to start', false);
            document.body.addEventListener('click', () => {{
                placeholderVideo.play();
                statusOverlay.classList.add('hidden');
            }}, {{ once: true }});
        }});
        
        function updateStatus(message, isError = false) {{
            console.log('[STATUS]', message, isError ? '(ERROR)' : '');
            statusOverlay.textContent = message;
            statusOverlay.classList.remove('hidden');
            if (isError) {{
                statusOverlay.classList.add('error');
            }} else {{
                statusOverlay.classList.remove('error');
            }}
            
            setTimeout(() => {{
                statusOverlay.classList.add('hidden');
            }}, 3000);
        }}
        
        function showStreamVideo() {{
            console.log('📺 Switching to stream video');
            streamVideo.style.display = 'block';
            placeholderVideo.style.display = 'none';
        }}
        
        function showPlaceholderVideo() {{
            console.log('📺 Switching to placeholder video');
            streamVideo.style.display = 'none';
            placeholderVideo.style.display = 'block';
            
            if (placeholderVideo.paused) {{
                placeholderVideo.play().catch(e => console.warn('Placeholder play failed:', e));
            }}
        }}
        
        function cleanupStream(stream) {{
            console.log('🧹 Cleaning up stream');
            
            if (stream.mediaSource) {{
                if (stream.mediaSource.readyState === 'open') {{
                    try {{
                        stream.mediaSource.endOfStream();
                    }} catch (e) {{
                        console.warn('Error ending stream:', e);
                    }}
                }}
                
                // Revoke object URL
                if (streamVideo.src && streamVideo.src.startsWith('blob:')) {{
                    URL.revokeObjectURL(streamVideo.src);
                }}
            }}
            
            // Reset video element
            streamVideo.src = '';
            streamVideo.load();
            
            showPlaceholderVideo();
        }}
        
        function createNewStream() {{
            console.log('🎬 Creating new stream');
            
            // ✅ CRITICAL: Clean up previous stream
            if (currentStream) {{
                cleanupStream(currentStream);
            }}
            
            // ✅ Create fresh state for this stream
            currentStream = {{
                mediaSource: null,
                sourceBuffer: null,
                pendingChunks: [],
                isAppending: false,
                isFirstChunk: true,
                totalChunks: 0,
                receivedChunks: 0,
                allChunksReceived: false
            }};
            
            // Initialize MediaSource
            currentStream.mediaSource = new MediaSource();
            const objectURL = URL.createObjectURL(currentStream.mediaSource);
            streamVideo.src = objectURL;
            
            currentStream.mediaSource.addEventListener('sourceopen', () => {{
                console.log('✅ MediaSource opened');
                
                try {{
                    currentStream.sourceBuffer = currentStream.mediaSource.addSourceBuffer('video/mp4; codecs="avc1.42E01E, mp4a.40.2"');
                    currentStream.sourceBuffer.mode = 'sequence';
                    
                    currentStream.sourceBuffer.addEventListener('updateend', () => {{
                        console.log('✅ Chunk appended successfully');
                        currentStream.isAppending = false;
                        
                        // Process next chunk
                        processNextChunk();
                        
                        // End stream if all chunks processed
                        if (currentStream.allChunksReceived && 
                            currentStream.pendingChunks.length === 0 && 
                            !currentStream.sourceBuffer.updating) {{
                            console.log('🏁 All chunks processed, ending stream');
                            if (currentStream.mediaSource.readyState === 'open') {{
                                try {{
                                    currentStream.mediaSource.endOfStream();
                                    console.log('✅ Stream ended gracefully');
                                }} catch (e) {{
                                    console.warn('Error ending stream:', e);
                                }}
                            }}
                        }}
                    }});
                    
                    currentStream.sourceBuffer.addEventListener('error', (e) => {{
                        console.error('❌ SourceBuffer error:', e);
                        updateStatus('❌ Playback error', true);
                        setTimeout(() => cleanupStream(currentStream), 2000);
                    }});
                    
                    console.log('✅ SourceBuffer created (mode=sequence)');
                    
                    // Process any pending chunks
                    processNextChunk();
                }} catch (e) {{
                    console.error('❌ Failed to create SourceBuffer:', e);
                    updateStatus('❌ Browser not supported', true);
                }}
            }});
            
            currentStream.mediaSource.addEventListener('sourceended', () => {{
                console.log('📺 MediaSource ended event');
            }});
            
            currentStream.mediaSource.addEventListener('error', (e) => {{
                console.error('❌ MediaSource error:', e);
                updateStatus('❌ Stream error', true);
                setTimeout(() => cleanupStream(currentStream), 2000);
            }});
            
            return currentStream;
        }}
        
        function processNextChunk() {{
            if (!currentStream || 
                currentStream.isAppending || 
                currentStream.pendingChunks.length === 0 || 
                !currentStream.sourceBuffer) {{
                return;
            }}
            
            if (currentStream.sourceBuffer.updating) {{
                console.log('⏳ SourceBuffer still updating, waiting...');
                return;
            }}
            
            currentStream.isAppending = true;
            const chunk = currentStream.pendingChunks.shift();
            
            console.log('⚡ Appending chunk (' + currentStream.pendingChunks.length + ' remaining in queue)');
            
            try {{
                // Append chunk
                currentStream.sourceBuffer.appendBuffer(chunk);
                
                // Auto-play on first chunk
                if (currentStream.isFirstChunk) {{
                    currentStream.isFirstChunk = false;
                    
                    // Switch to stream video
                    showStreamVideo();
                    
                    // Start playback
                    streamVideo.play().then(() => {{
                        console.log('▶️ Playback started (first chunk)');
                        updateStatus('Playing...', false);
                    }}).catch(e => {{
                        console.warn('⚠️ Autoplay blocked:', e);
                        updateStatus('Click to play', false);
                        
                        // Fallback: play on click
                        streamVideo.addEventListener('click', () => {{
                            streamVideo.play().then(() => {{
                                showStreamVideo();
                                updateStatus('Playing...', false);
                            }});
                        }}, {{ once: true }});
                    }});
                }}
            }} catch (e) {{
                console.error('❌ Failed to append chunk:', e);
                currentStream.isAppending = false;
                updateStatus('❌ Chunk error', true);
                
                // Try to recover
                if (currentStream.pendingChunks.length > 0) {{
                    console.log('🔄 Retrying next chunk...');
                    setTimeout(processNextChunk, 100);
                }}
            }}
        }}
        
        // Stream video events
        streamVideo.addEventListener('ended', () => {{
            console.log('⏹️ Stream video ended');
            updateStatus('✅ Complete', false);
            // ✅ DON'T reset immediately - wait for next audio
            setTimeout(() => {{
                if (currentStream && currentStream.allChunksReceived) {{
                    cleanupStream(currentStream);
                    currentStream = null;
                }}
            }}, 2000);
        }});
        
        streamVideo.addEventListener('error', (e) => {{
            if (!currentStream) return;
            
            const error = streamVideo.error;
            if (error) {{
                console.error('❌ Stream video error - Code:', error.code);
                updateStatus('❌ Playback error', true);
                setTimeout(() => {{
                    cleanupStream(currentStream);
                    currentStream = null;
                }}, 2000);
            }}
        }});
        
        async function connectToSession() {{
            updateStatus('Connecting...', false);
            
            try {{
                const eventSource = new EventSource('/sessions/{session.session_id}/events');
                
                eventSource.addEventListener('message', async (e) => {{
                    const data = JSON.parse(e.data);
                    console.log('📨 SSE event:', data.event, data);
                    
                    if (data.event === 'chunk') {{
                        // ✅ CRITICAL: Create new stream on first chunk
                        if (!currentStream || currentStream.allChunksReceived) {{
                            console.log('🎬 New audio stream detected - creating fresh MediaSource');
                            currentStream = createNewStream();
                            // Wait for MediaSource to be ready
                            await new Promise(resolve => setTimeout(resolve, 100));
                        }}
                        
                        currentStream.totalChunks = data.total_chunks;
                        currentStream.receivedChunks++;
                        
                        updateStatus('Loading ' + currentStream.receivedChunks + '/' + currentStream.totalChunks, false);
                        
                        try {{
                            console.log('📥 Fetching chunk ' + (data.index + 1) + ':', data.url);
                            const response = await fetch(data.url);
                            if (!response.ok) throw new Error('HTTP ' + response.status);
                            
                            const arrayBuffer = await response.arrayBuffer();
                            console.log('✅ Chunk ' + (data.index + 1) + ' fetched (' + (arrayBuffer.byteLength / 1024).toFixed(1) + ' KB)');
                            
                            currentStream.pendingChunks.push(arrayBuffer);
                            
                            // Try to process if SourceBuffer is ready
                            if (currentStream.sourceBuffer && !currentStream.sourceBuffer.updating) {{
                                processNextChunk();
                            }}
                        }} catch (error) {{
                            console.error('❌ Failed to fetch chunk:', error);
                            updateStatus('❌ Download error', true);
                        }}
                    }}
                    else if (data.event === 'complete') {{
                        console.log('✅ Stream complete - all chunks sent');
                        if (currentStream) {{
                            currentStream.allChunksReceived = true;
                            
                            // Try to end stream if all chunks are processed
                            if (currentStream.sourceBuffer && 
                                !currentStream.sourceBuffer.updating && 
                                currentStream.pendingChunks.length === 0) {{
                                if (currentStream.mediaSource && currentStream.mediaSource.readyState === 'open') {{
                                    try {{
                                        currentStream.mediaSource.endOfStream();
                                    }} catch (e) {{
                                        console.warn('Error ending stream:', e);
                                    }}
                                }}
                            }}
                        }}
                    }}
                    else if (data.event === 'error') {{
                        console.error('❌ Stream error:', data.message);
                        updateStatus('❌ ' + data.message, true);
                        if (currentStream) {{
                            cleanupStream(currentStream);
                            currentStream = null;
                        }}
                    }}
                }});
                
                eventSource.addEventListener('error', (e) => {{
                    console.warn('⚠️ SSE connection error');
                    // ✅ DON'T close - allow reconnection
                }});
                
                eventSource.addEventListener('open', () => {{
                    console.log('✅ SSE connected');
                    updateStatus('Ready - waiting for audio', false);
                }});
            }}
            catch (error) {{
                console.error('❌ Connection failed:', error);
                updateStatus('❌ Connection failed', true);
            }}
        }}
        
        // Initialize
        console.log('🚀 Session player initializing...');
        console.log('📱 Session ID:', SESSION_ID);
        console.log('🎭 Avatar ID:', AVATAR_ID);
        
        connectToSession();
    </script>
</body>
</html>
    """