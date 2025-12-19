def get_session_player_html(session) -> str:
    """
    Session-based streaming player.

    - Placeholder video is dynamic (from disk) per avatar:
      GET /avatars/{avatar_id}/video

    - Streaming chunks arrive through SSE:
      GET /sessions/{session_id}/events

    This keeps the original working MSE chunk append/autoplay logic,
    but fixes origin/port mismatches by using window.location.origin.
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
        <video id="placeholderVideo" playsinline webkit-playsinline autoplay muted loop></video>
        <video id="streamVideo" playsinline webkit-playsinline></video>
        <div class="status-overlay hidden" id="statusOverlay">Ready</div>
    </div>

    <script>
        const SESSION_ID = '{session.session_id}';
        const AVATAR_ID = '{session.avatar_id}';
        const API_ORIGIN = window.location.origin;

        const placeholderVideo = document.getElementById('placeholderVideo');
        const streamVideo = document.getElementById('streamVideo');
        const statusOverlay = document.getElementById('statusOverlay');

        let currentStream = null;
        let backToPlaceholderTimer = null;

        // ✅ Dynamic placeholder video (per avatar)
        placeholderVideo.src = `${{API_ORIGIN}}/avatars/${{AVATAR_ID}}/video?v=${{Date.now()}}`;
        placeholderVideo.load();

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

            // ✅ clear stream video so the last decoded frame cannot "stick"
            try {{
                streamVideo.pause();
                streamVideo.removeAttribute('src');
                streamVideo.load();
            }} catch (e) {{
                console.warn('Failed to reset streamVideo:', e);
            }}

            streamVideo.style.display = 'none';
            placeholderVideo.style.display = 'block';

            if (placeholderVideo.paused) {{
                placeholderVideo.play().catch(e => console.warn('Placeholder play failed:', e));
            }}
        }}

        function cleanupStream(stream) {{
            console.log('🧹 Cleaning up stream');

            if (backToPlaceholderTimer) {{
                clearTimeout(backToPlaceholderTimer);
                backToPlaceholderTimer = null;
            }}

            if (stream?.mediaSource && stream.mediaSource.readyState === 'open') {{
                try {{
                    stream.mediaSource.endOfStream();
                }} catch (e) {{
                    console.warn('Error ending stream:', e);
                }}
            }}

            // Revoke blob URL if we used one
            if (streamVideo.src && streamVideo.src.startsWith('blob:')) {{
                try {{ URL.revokeObjectURL(streamVideo.src); }} catch (_) {{}}
            }}

            showPlaceholderVideo();
        }}

        function createNewStream() {{
            console.log('🎬 Creating new stream');

            if (currentStream) {{
                cleanupStream(currentStream);
            }}

            currentStream = {{
                mediaSource: null,
                sourceBuffer: null,
                pendingChunks: [],
                isAppending: false,
                isFirstChunk: true,
                totalChunks: 0,
                receivedChunks: 0,
                allChunksReceived: false,
                switchedBack: false
            }};

            // ✅ if playback naturally ends, immediately return to base video
            streamVideo.onended = () => {{
                console.log('🏁 streamVideo ended -> back to placeholder');
                if (currentStream && !currentStream.switchedBack) {{
                    currentStream.switchedBack = true;
                    cleanupStream(currentStream);
                    currentStream = null;
                    updateStatus('Ready - waiting for audio', false);
                }} else {{
                    showPlaceholderVideo();
                }}
            }};

            // (optional but useful)
            streamVideo.onerror = () => {{
                console.warn('⚠️ streamVideo error -> back to placeholder');
                if (currentStream && !currentStream.switchedBack) {{
                    currentStream.switchedBack = true;
                    cleanupStream(currentStream);
                    currentStream = null;
                }} else {{
                    showPlaceholderVideo();
                }}
            }};

            currentStream.mediaSource = new MediaSource();
            streamVideo.src = URL.createObjectURL(currentStream.mediaSource);

            currentStream.mediaSource.addEventListener('sourceopen', () => {{
                console.log('✅ MediaSource opened');

                try {{
                    currentStream.sourceBuffer = currentStream.mediaSource.addSourceBuffer(
                        'video/mp4; codecs="avc1.42E01E, mp4a.40.2"'
                    );
                    currentStream.sourceBuffer.mode = 'sequence';

                    currentStream.sourceBuffer.addEventListener('updateend', () => {{
                        currentStream.isAppending = false;
                        processNextChunk();

                        // ✅ when complete + no pending, end MSE and schedule fallback swap-back
                        if (currentStream.allChunksReceived &&
                            currentStream.pendingChunks.length === 0 &&
                            !currentStream.sourceBuffer.updating &&
                            currentStream.mediaSource.readyState === 'open') {{
                            try {{
                                currentStream.mediaSource.endOfStream();
                            }} catch (e) {{
                                console.warn('Error ending stream:', e);
                            }}

                            // Some browsers never fire 'ended' for MSE; force swap back.
                            if (backToPlaceholderTimer) clearTimeout(backToPlaceholderTimer);
                            backToPlaceholderTimer = setTimeout(() => {{
                                if (currentStream && !currentStream.switchedBack) {{
                                    console.log('⏱️ fallback -> back to placeholder');
                                    currentStream.switchedBack = true;
                                    cleanupStream(currentStream);
                                    currentStream = null;
                                    updateStatus('Ready - waiting for audio', false);
                                }}
                            }}, 400);
                        }}
                    }});

                    currentStream.sourceBuffer.addEventListener('error', (e) => {{
                        console.error('❌ SourceBuffer error:', e);
                        updateStatus('❌ Playback error', true);
                    }});

                    processNextChunk();
                }} catch (e) {{
                    console.error('❌ Error creating SourceBuffer:', e);
                    updateStatus('❌ Playback error', true);
                }}
            }});
        }}

        function processNextChunk() {{
            if (!currentStream ||
                currentStream.isAppending ||
                currentStream.pendingChunks.length === 0 ||
                !currentStream.sourceBuffer) {{
                return;
            }}

            if (currentStream.sourceBuffer.updating) {{
                return;
            }}

            currentStream.isAppending = true;
            const chunk = currentStream.pendingChunks.shift();

            try {{
                currentStream.sourceBuffer.appendBuffer(chunk);

                // ✅ Autoplay + switch ONLY after first successful append attempt
                if (currentStream.isFirstChunk) {{
                    currentStream.isFirstChunk = false;
                    showStreamVideo();

                    streamVideo.play().then(() => {{
                        console.log('▶️ Playback started (first chunk)');
                        updateStatus('Playing...', false);
                    }}).catch(e => {{
                        console.warn('⚠️ Autoplay blocked:', e);
                        updateStatus('Click to play', false);
                        document.body.addEventListener('click', () => {{
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
                updateStatus('❌ Chunk append error', true);
                setTimeout(processNextChunk, 50);
            }}
        }}

        async function connectToSession() {{
            updateStatus('Connecting...', false);

            const sseUrl = `${{API_ORIGIN}}/sessions/${{SESSION_ID}}/events`;
            console.log('SSE URL:', sseUrl);

            const eventSource = new EventSource(sseUrl);

            eventSource.addEventListener('open', () => {{
                console.log('✅ SSE connected');
                updateStatus('Ready - waiting for audio', false);
            }});

            eventSource.addEventListener('message', async (e) => {{
                const data = JSON.parse(e.data);
                console.log('📨 SSE event:', data.event, data);

                if (data.event === 'chunk') {{
                    if (!currentStream || currentStream.allChunksReceived) {{
                        createNewStream();
                        // allow sourceopen to fire if needed
                        await new Promise(r => setTimeout(r, 50));
                    }}

                    const chunkUrl = data.url?.startsWith('http')
                        ? data.url
                        : `${{API_ORIGIN}}${{data.url}}`;

                    try {{
                        const resp = await fetch(chunkUrl);
                        if (!resp.ok) throw new Error(`HTTP ${{resp.status}}`);
                        const buf = await resp.arrayBuffer();

                        // ✅ IMPORTANT: push as Uint8Array (works best with appendBuffer)
                        currentStream.pendingChunks.push(new Uint8Array(buf));
                        currentStream.receivedChunks += 1;

                        processNextChunk();
                    }} catch (err) {{
                        console.error('❌ Error fetching chunk:', err);
                        updateStatus('❌ Download error', true);
                    }}
                }} else if (data.event === 'complete') {{
                    console.log('🏁 All chunks received');
                    if (currentStream) {{
                        currentStream.allChunksReceived = true;
                        processNextChunk();
                    }}
                }} else if (data.event === 'error') {{
                    console.error('❌ Stream error:', data.message);
                    updateStatus('❌ ' + (data.message || 'Stream error'), true);
                }}
            }});

            eventSource.addEventListener('error', (e) => {{
                console.warn('⚠️ SSE connection error', e);
            }});
        }}

        connectToSession();
    </script>
</body>
</html>
    """