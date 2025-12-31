def get_hls_player_html(session) -> str:
    manifest_url = f"/hls/sessions/{session.session_id}/index.m3u8"
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <title>HLS Avatar - {session.avatar_id}</title>
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
            background: #000;
        }}

        .status {{
            position: absolute;
            bottom: 16px;
            left: 50%;
            transform: translateX(-50%);
            background: rgba(0,0,0,0.7);
            color: #fff;
            padding: 8px 14px;
            border-radius: 16px;
            font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            font-size: 13px;
            z-index: 10;
            transition: opacity 0.2s ease;
        }}

        .status.hidden {{
            opacity: 0;
            pointer-events: none;
        }}

        .status.button {{
            cursor: pointer;
        }}
    </style>
</head>
<body>
    <div class="video-container">
        <video id="hlsVideo" playsinline webkit-playsinline autoplay muted></video>
        <div class="status button" id="status">Loading...</div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/hls.js@1/dist/hls.min.js"></script>
    <script>
        const idleManifestUrl = '{manifest_url}';
        const liveManifestUrl = '{manifest_url}'.replace('index.m3u8', 'live.m3u8');
        const statusUrl = '{manifest_url}'.replace('/index.m3u8', '/status');
        const video = document.getElementById('hlsVideo');
        const status = document.getElementById('status');
        let currentMode = 'idle';
        let hls = null;
        let started = false;
        let userActivated = false;

        function showStatus(text, isButton = false) {{
            status.textContent = text;
            status.classList.remove('hidden');
            if (isButton) {{
                status.classList.add('button');
            }} else {{
                status.classList.remove('button');
            }}
        }}

        function hideStatus() {{
            status.classList.add('hidden');
        }}

        function attemptPlay() {{
            const playPromise = video.play();
            if (playPromise && playPromise.catch) {{
                playPromise
                    .then(() => {{
                        hideStatus();
                    }})
                    .catch((err) => {{
                        if (err && err.name === 'NotAllowedError') {{
                            showStatus('Tap to start', true);
                            return;
                        }}
                        if (userActivated) {{
                            showStatus('Buffering...');
                        }} else {{
                            showStatus('Tap to start', true);
                        }}
                    }});
            }}
        }}

        function loadManifest(url, autoPlay = false) {{
            if (hls) {{
                hls.destroy();
                hls = null;
            }}
            if (video.canPlayType('application/vnd.apple.mpegurl')) {{
                video.src = url;
                video.load();
                if (autoPlay) {{
                    attemptPlay();
                }}
                return;
            }}

            if (window.Hls && Hls.isSupported()) {{
                hls = new Hls({{ lowLatencyMode: true }});
                hls.loadSource(url);
                hls.attachMedia(video);
                if (autoPlay) {{
                    attemptPlay();
                }}
                hls.on(Hls.Events.MANIFEST_PARSED, () => {{
                    if (autoPlay) {{
                        attemptPlay();
                    }}
                }});
                hls.on(Hls.Events.ERROR, (_, data) => {{
                    if (data && data.fatal) {{
                        showStatus('HLS error', true);
                    }}
                }});
                return;
            }}

            showStatus('HLS not supported', true);
        }}

        function setMode(mode) {{
            if (mode === currentMode) return;
            currentMode = mode;
            video.loop = mode === 'idle';
            if (!userActivated) {{
                showStatus('Tap to start', true);
            }} else {{
                showStatus(mode === 'live' ? 'Live streaming' : 'Idle');
            }}
            loadManifest(mode === 'live' ? liveManifestUrl : idleManifestUrl, userActivated);
        }}

        function handleTap() {{
            if (!userActivated) {{
                userActivated = true;
                video.muted = false;
                video.volume = 1.0;
            }}
            if (!started) {{
                started = true;
                loadManifest(currentMode === 'live' ? liveManifestUrl : idleManifestUrl, true);
                attemptPlay();
                return;
            }}
            attemptPlay();
        }}

        status.addEventListener('pointerdown', handleTap);
        status.addEventListener('click', handleTap);
        status.addEventListener('touchstart', handleTap);
        video.addEventListener('pointerdown', handleTap);
        video.addEventListener('click', handleTap);

        document.addEventListener('pointerdown', handleTap, {{ once: true }});

        async function pollStatus() {{
            try {{
                const resp = await fetch(statusUrl, {{ cache: 'no-store' }});
                if (!resp.ok) return;
                const data = await resp.json();
                if (data.status === 'streaming' && data.live_ready) {{
                    setMode('live');
                }} else if (data.status === 'streaming' && currentMode === 'idle') {{
                    showStatus('Preparing live...');
                }}
            }} catch (_) {{
                // Ignore polling failures.
            }}
        }}

        video.addEventListener('playing', () => {{
            hideStatus();
        }});

        video.addEventListener('canplay', () => {{
            if (userActivated) {{
                attemptPlay();
            }}
        }});

        video.addEventListener('waiting', () => {{
            if (userActivated) {{
                showStatus('Buffering...');
            }} else {{
                showStatus('Tap to start', true);
            }}
        }});

        video.addEventListener('ended', () => {{
            if (currentMode === 'live') {{
                setMode('idle');
            }}
        }});

        video.loop = true;
        showStatus('Tap to start', true);
        loadManifest(idleManifestUrl, false);
        setInterval(pollStatus, 1500);
    </script>
</body>
</html>
"""
