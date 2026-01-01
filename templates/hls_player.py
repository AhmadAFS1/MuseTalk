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

        .video-layer {{
            transition: opacity 0.25s ease;
        }}

        .video-layer.hidden {{
            opacity: 0;
            pointer-events: none;
        }}

        .video-layer.visible {{
            opacity: 1;
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
    <div class="video-container" id="videoContainer">
        <video id="idleVideo" class="video-layer visible" playsinline webkit-playsinline autoplay muted></video>
        <video id="liveVideo" class="video-layer hidden" playsinline webkit-playsinline autoplay muted></video>
        <div class="status button" id="status">Loading...</div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/hls.js@1/dist/hls.min.js"></script>
    <script>
        const idleManifestUrl = '{manifest_url}';
        const liveManifestBaseUrl = '{manifest_url}'.replace('index.m3u8', 'live.m3u8');
        const statusUrl = '{manifest_url}'.replace('/index.m3u8', '/status');
        const container = document.getElementById('videoContainer');
        const idleVideo = document.getElementById('idleVideo');
        const liveVideo = document.getElementById('liveVideo');
        const status = document.getElementById('status');
        let currentMode = 'idle';
        let idleHls = null;
        let liveHls = null;
        let started = false;
        let userActivated = false;
        let livePrepared = false;
        let liveManifestUrl = liveManifestBaseUrl;
        let currentStreamId = null;
        let idleAnchorTime = 0;
        let idleAnchorWallTime = 0;
        let idleDuration = 0;

        idleVideo.loop = true;
        idleVideo.muted = true;
        idleVideo.volume = 0;
        liveVideo.loop = false;

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

        function attemptPlay(targetVideo) {{
            const playPromise = targetVideo.play();
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

        function setLayer(mode) {{
            if (mode === 'live') {{
                liveVideo.classList.remove('hidden');
                liveVideo.classList.add('visible');
                idleVideo.classList.remove('visible');
                idleVideo.classList.add('hidden');
            }} else {{
                idleVideo.classList.remove('hidden');
                idleVideo.classList.add('visible');
                liveVideo.classList.remove('visible');
                liveVideo.classList.add('hidden');
            }}
        }}

        function destroyLive() {{
            if (liveHls) {{
                liveHls.destroy();
                liveHls = null;
            }}
            liveVideo.removeAttribute('src');
            liveVideo.load();
            livePrepared = false;
        }}

        function markIdleAnchor() {{
            idleAnchorTime = idleVideo.currentTime || 0;
            idleAnchorWallTime = performance.now();
        }}

        function computeIdleResumeTime() {{
            if (!idleDuration || idleDuration <= 0) {{
                return null;
            }}
            const elapsed = (performance.now() - idleAnchorWallTime) / 1000;
            const resumeTime = (idleAnchorTime + elapsed) % idleDuration;
            return Number.isFinite(resumeTime) ? resumeTime : null;
        }}

        function resumeIdlePlayback() {{
            const resumeTime = computeIdleResumeTime();
            const startIdle = () => {{
                setLayer('idle');
                idleVideo.loop = true;
                attemptPlay(idleVideo);
                currentMode = 'idle';
            }};
            if (resumeTime === null) {{
                startIdle();
                return;
            }}
            const onSeeked = () => {{
                idleVideo.removeEventListener('seeked', onSeeked);
                startIdle();
            }};
            idleVideo.addEventListener('seeked', onSeeked, {{ once: true }});
            try {{
                idleVideo.currentTime = resumeTime;
            }} catch (_) {{
                idleVideo.removeEventListener('seeked', onSeeked);
                startIdle();
            }}
        }}

        function setLiveStreamId(streamId) {{
            if (!streamId || streamId === currentStreamId) {{
                return false;
            }}
            currentStreamId = streamId;
            liveManifestUrl = liveManifestBaseUrl + '?stream_id=' + encodeURIComponent(streamId);
            destroyLive();
            return true;
        }}

        function attachHls(videoEl, url, config, onReady) {{
            if (videoEl.canPlayType('application/vnd.apple.mpegurl')) {{
                videoEl.src = url;
                videoEl.load();
                if (onReady) {{
                    onReady();
                }}
                return null;
            }}

            if (window.Hls && Hls.isSupported()) {{
                const instance = new Hls(config);
                instance.loadSource(url);
                instance.attachMedia(videoEl);
                if (onReady) {{
                    instance.on(Hls.Events.MANIFEST_PARSED, () => {{
                        onReady();
                    }});
                }}
                instance.on(Hls.Events.ERROR, (_, data) => {{
                    if (data && data.fatal) {{
                        showStatus('HLS error', true);
                    }}
                }});
                return instance;
            }}

            showStatus('HLS not supported', true);
            return null;
        }}

        function loadIdle(autoPlay = false) {{
            if (idleHls) {{
                idleHls.destroy();
                idleHls = null;
            }}
            idleHls = attachHls(
                idleVideo,
                idleManifestUrl,
                {{ lowLatencyMode: false }},
                () => {{
                    if (autoPlay) {{
                        attemptPlay(idleVideo);
                    }}
                }}
            );
            if (autoPlay && idleVideo.canPlayType('application/vnd.apple.mpegurl')) {{
                attemptPlay(idleVideo);
            }}
        }}

        function prepareLive() {{
            if (livePrepared) return;
            livePrepared = true;

            if (liveHls) {{
                liveHls.destroy();
                liveHls = null;
            }}

            liveHls = attachHls(
                liveVideo,
                liveManifestUrl,
                {{
                    lowLatencyMode: false,
                    liveSyncDurationCount: 1,
                    liveMaxLatencyDurationCount: 3,
                    maxBufferLength: 2,
                    backBufferLength: 0,
                }},
                () => {{
                    if (userActivated) {{
                        attemptPlay(liveVideo);
                    }}
                }}
            );

            if (userActivated && liveVideo.canPlayType('application/vnd.apple.mpegurl')) {{
                attemptPlay(liveVideo);
            }}
        }}

        function setMode(mode) {{
            if (mode === currentMode && !(mode === 'live' && !livePrepared)) return;
            currentMode = mode;
            if (!userActivated) {{
                showStatus('Tap to start', true);
            }} else {{
                showStatus(mode === 'live' ? 'Live streaming' : 'Idle');
            }}
            if (mode === 'live') {{
                prepareLive();
            }} else {{
                setLayer('idle');
                idleVideo.loop = true;
                attemptPlay(idleVideo);
            }}
        }}

        function handleTap() {{
            if (!userActivated) {{
                userActivated = true;
                liveVideo.muted = false;
                liveVideo.volume = 1.0;
            }}
            if (!started) {{
                started = true;
                loadIdle(true);
                if (currentMode === 'live') {{
                    prepareLive();
                }}
                attemptPlay(currentMode === 'live' ? liveVideo : idleVideo);
                return;
            }}
            attemptPlay(currentMode === 'live' ? liveVideo : idleVideo);
        }}

        status.addEventListener('pointerdown', handleTap);
        status.addEventListener('click', handleTap);
        status.addEventListener('touchstart', handleTap);
        container.addEventListener('pointerdown', handleTap);
        container.addEventListener('click', handleTap);
        container.addEventListener('touchstart', handleTap);

        document.addEventListener('pointerdown', handleTap, {{ once: true }});

        async function pollStatus() {{
            try {{
                const resp = await fetch(statusUrl, {{ cache: 'no-store' }});
                if (!resp.ok) return;
                const data = await resp.json();
                if (data.status === 'streaming' && data.live_ready) {{
                    setLiveStreamId(data.active_stream);
                    prepareLive();
                    setMode('live');
                }} else if (data.status === 'streaming' && currentMode === 'idle') {{
                    setLiveStreamId(data.active_stream);
                    showStatus('Preparing live...');
                }} else if (data.status !== 'streaming' && currentMode === 'live') {{
                    showStatus('Finishing...');
                }}
            }} catch (_) {{
                // Ignore polling failures.
            }}
        }}

        idleVideo.addEventListener('playing', () => {{
            if (currentMode === 'idle') {{
                hideStatus();
            }}
        }});

        idleVideo.addEventListener('loadedmetadata', () => {{
            if (idleVideo.duration && Number.isFinite(idleVideo.duration)) {{
                idleDuration = idleVideo.duration;
            }}
        }});

        idleVideo.addEventListener('waiting', () => {{
            if (currentMode === 'idle') {{
                showStatus(userActivated ? 'Buffering...' : 'Tap to start', !userActivated);
            }}
        }});

        liveVideo.addEventListener('playing', () => {{
            if (currentMode !== 'live') {{
                currentMode = 'live';
            }}
            markIdleAnchor();
            setLayer('live');
            idleVideo.pause();
            hideStatus();
        }});

        liveVideo.addEventListener('waiting', () => {{
            if (currentMode === 'live') {{
                showStatus('Buffering...');
            }}
        }});

        liveVideo.addEventListener('ended', () => {{
            if (currentMode === 'live') {{
                resumeIdlePlayback();
                destroyLive();
            }}
        }});

        liveVideo.addEventListener('stalled', () => {{
            if (currentMode === 'live') {{
                showStatus('Buffering...');
            }}
        }});

        idleVideo.addEventListener('canplay', () => {{
            if (userActivated && currentMode === 'idle') {{
                attemptPlay(idleVideo);
            }}
        }});

        liveVideo.addEventListener('canplay', () => {{
            if (userActivated && currentMode === 'live') {{
                attemptPlay(liveVideo);
            }}
        }});

        showStatus('Tap to start', true);
        loadIdle(false);
        setInterval(pollStatus, 800);
    </script>
</body>
</html>
"""
