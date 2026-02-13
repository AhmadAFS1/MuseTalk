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

        /* No crossfade – instant swap */
        .video-layer {{
            transition: none;
        }}

        .video-layer.hidden {{
            opacity: 0;
            pointer-events: none;
        }}

        .video-layer.visible {{
            opacity: 1;
        }}

        .frame-hold {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: #000;
            pointer-events: none;
            opacity: 0;
            z-index: 5;
            transition: none;
        }}

        .frame-hold.visible {{
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
        <video id="idleVideo" class="video-layer visible" style="z-index:1" playsinline webkit-playsinline autoplay muted></video>
        <video id="liveVideo" class="video-layer hidden" style="z-index:2" playsinline webkit-playsinline autoplay muted></video>
        <canvas id="holdCanvas" class="frame-hold"></canvas>
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
        const holdCanvas = document.getElementById('holdCanvas');
        const holdCtx = holdCanvas.getContext('2d');
        const statusEl = document.getElementById('status');
        let currentMode = 'idle';
        let idleHls = null;
        let liveHls = null;
        let started = false;
        let userActivated = false;
        let livePrepared = false;
        let liveRevealPending = false;
        let liveRevealInFlight = false;
        let liveRevealed = false;
        let liveManifestUrl = liveManifestBaseUrl;
        let currentStreamId = null;
        let idleAnchorTime = 0;
        let idleAnchorWallTime = 0;
        let idleDuration = 0;

        const LIVE_PREBUFFER_SECONDS = 1.2;

        function postToHost(payload) {{
            if (window.ReactNativeWebView && window.ReactNativeWebView.postMessage) {{
                try {{
                    window.ReactNativeWebView.postMessage(JSON.stringify(payload));
                }} catch (_) {{}}
            }}
        }}

        function getIdleTimePayload() {{
            const idleTime = Number.isFinite(idleVideo.currentTime) ? idleVideo.currentTime : 0;
            const duration = Number.isFinite(idleDuration) ? idleDuration : (idleVideo.duration || 0);
            return {{
                type: 'idle_time',
                idle_time: idleTime,
                idle_duration: duration,
                mode: currentMode,
            }};
        }}

        function handleHostMessage(event) {{
            let data = event && event.data !== undefined ? event.data : event;
            if (typeof data === 'string') {{
                try {{
                    data = JSON.parse(data);
                }} catch (_) {{
                    if (data !== 'get_idle_time') return;
                    data = {{ type: 'get_idle_time' }};
                }}
            }}
            if (!data || data.type !== 'get_idle_time') return;
            postToHost(getIdleTimePayload());
        }}

        idleVideo.loop = true;
        idleVideo.muted = true;
        idleVideo.volume = 0;
        liveVideo.loop = false;

        function showStatus(text, isButton = false) {{
            statusEl.textContent = text;
            statusEl.classList.remove('hidden');
            if (isButton) statusEl.classList.add('button');
            else statusEl.classList.remove('button');
        }}

        function hideStatus() {{
            statusEl.classList.add('hidden');
        }}

        function attemptPlay(targetVideo) {{
            const p = targetVideo.play();
            if (p && p.catch) {{
                p.then(() => hideStatus())
                 .catch((err) => {{
                    if (err && err.name === 'NotAllowedError') {{
                        showStatus('Tap to start', true);
                        return;
                    }}
                    showStatus(userActivated ? 'Buffering...' : 'Tap to start', !userActivated);
                }});
            }}
        }}

        /* Instant layer swap – no crossfade */
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
            if (liveHls) {{ liveHls.destroy(); liveHls = null; }}
            liveVideo.removeAttribute('src');
            liveVideo.load();
            livePrepared = false;
            liveRevealPending = false;
            liveRevealInFlight = false;
            liveRevealed = false;
        }}

        function markIdleAnchor() {{
            idleAnchorTime = idleVideo.currentTime || 0;
            idleAnchorWallTime = performance.now();
        }}

        function computeIdleResumeTime() {{
            if (!idleDuration || idleDuration <= 0) return null;
            const elapsed = (performance.now() - idleAnchorWallTime) / 1000;
            const resumeTime = (idleAnchorTime + elapsed) % idleDuration;
            return Number.isFinite(resumeTime) ? resumeTime : null;
        }}

        function resizeHoldCanvas() {{
            const rect = container.getBoundingClientRect();
            const dpr = window.devicePixelRatio || 1;
            const w = Math.max(1, Math.floor(rect.width * dpr));
            const h = Math.max(1, Math.floor(rect.height * dpr));
            if (holdCanvas.width !== w || holdCanvas.height !== h) {{
                holdCanvas.width = w;
                holdCanvas.height = h;
            }}
        }}

        function drawFrameToHold(videoEl) {{
            if (!holdCtx) return false;
            const vw = videoEl.videoWidth;
            const vh = videoEl.videoHeight;
            if (!vw || !vh) return false;
            resizeHoldCanvas();
            const cw = holdCanvas.width;
            const ch = holdCanvas.height;
            const scale = Math.min(cw / vw, ch / vh);
            const drawW = vw * scale;
            const drawH = vh * scale;
            const offsetX = (cw - drawW) / 2;
            const offsetY = (ch - drawH) / 2;
            holdCtx.fillStyle = '#000';
            holdCtx.fillRect(0, 0, cw, ch);
            holdCtx.drawImage(videoEl, offsetX, offsetY, drawW, drawH);
            return true;
        }}

        function showHoldFrame(videoEl) {{
            const ok = drawFrameToHold(videoEl);
            if (ok) holdCanvas.classList.add('visible');
            return ok;
        }}

        function hideHoldFrame() {{
            holdCanvas.classList.remove('visible');
        }}

        function captureHoldFrame(videoEl) {{
            if (showHoldFrame(videoEl)) return Promise.resolve(true);
            if (videoEl.requestVideoFrameCallback) {{
                return new Promise((resolve) => {{
                    videoEl.requestVideoFrameCallback(() => resolve(showHoldFrame(videoEl)));
                }});
            }}
            return Promise.resolve(false);
        }}

        /* waitForVideoFrame: use requestVideoFrameCallback for
           accurate "frame is painted" detection */
        function waitForVideoFrame(videoEl, timeoutMs = 1500) {{
            return new Promise((resolve) => {{
                let done = false;
                const finish = (result) => {{
                    if (done) return;
                    done = true;
                    clearTimeout(timer);
                    resolve(result);
                }};
                const timer = setTimeout(() => finish(false), timeoutMs);

                if (videoEl.requestVideoFrameCallback) {{
                    videoEl.requestVideoFrameCallback(() => finish(true));
                }} else {{
                    const onTU = () => {{
                        videoEl.removeEventListener('timeupdate', onTU);
                        requestAnimationFrame(() => finish(true));
                    }};
                    videoEl.addEventListener('timeupdate', onTU);
                }}
            }});
        }}

        function getBufferedAheadSeconds(videoEl) {{
            const buf = videoEl.buffered;
            if (!buf || buf.length === 0) return 0;
            const t = videoEl.currentTime;
            for (let i = 0; i < buf.length; i++) {{
                if (t >= buf.start(i) && t <= buf.end(i)) return buf.end(i) - t;
            }}
            return buf.end(buf.length - 1) - t;
        }}

        async function revealLive() {{
            if (liveRevealed || !liveRevealPending) return;
            await waitForVideoFrame(liveVideo, 1500);
            if (!liveRevealPending) return;
            liveRevealPending = false;
            liveRevealed = true;
            currentMode = 'live';
            markIdleAnchor();
            liveVideo.muted = false;
            liveVideo.volume = 1.0;
            setLayer('live');
            idleVideo.pause();
            hideHoldFrame();
            hideStatus();
        }}

        function maybeRevealLive() {{
            if (!liveRevealPending || liveRevealed || liveRevealInFlight) return;
            const buffered = getBufferedAheadSeconds(liveVideo);
            if (buffered < LIVE_PREBUFFER_SECONDS) return;
            liveRevealInFlight = true;
            revealLive().finally(() => {{ liveRevealInFlight = false; }});
        }}

        async function primeIdlePlayback() {{
            const resumeTime = computeIdleResumeTime();
            if (resumeTime !== null) {{
                try {{ idleVideo.currentTime = resumeTime; }} catch (_) {{}}
            }}
            idleVideo.loop = true;
            attemptPlay(idleVideo);
            await waitForVideoFrame(idleVideo, 1200);
        }}

        /* transitionToIdle: hold canvas bridges the gap.
         * 1) Capture last live frame on canvas (opaque, z-index 8)
         * 2) Swap layers behind canvas (user can't see)
         * 3) Prime idle, wait for painted frame
         * 4) Remove hold canvas instantly
         */
        async function transitionToIdle() {{
            if (currentMode !== 'live') return;
            liveRevealPending = false;
            liveRevealInFlight = false;
            liveRevealed = false;

            /* 1) Capture last live frame */
            holdCanvas.style.zIndex = '8';
            await captureHoldFrame(liveVideo);

            /* 2) Swap layers behind the hold canvas */
            idleVideo.classList.remove('hidden');
            idleVideo.classList.add('visible');
            liveVideo.classList.remove('visible');
            liveVideo.classList.add('hidden');
            currentMode = 'idle';

            /* 3) Prime idle and wait for a real painted frame */
            await primeIdlePlayback();

            /* 4) Idle is now rendering – remove hold canvas */
            hideHoldFrame();
            holdCanvas.style.zIndex = '5';
            destroyLive();
        }}

        function setLiveStreamId(streamId) {{
            if (!streamId || streamId === currentStreamId) return false;
            currentStreamId = streamId;
            liveManifestUrl = liveManifestBaseUrl + '?stream_id=' + encodeURIComponent(streamId);
            destroyLive();
            return true;
        }}

        function attachHls(videoEl, url, config, onReady) {{
            if (videoEl.canPlayType('application/vnd.apple.mpegurl')) {{
                videoEl.src = url;
                videoEl.load();
                if (onReady) onReady();
                return null;
            }}
            if (window.Hls && Hls.isSupported()) {{
                const inst = new Hls(config);
                inst.loadSource(url);
                inst.attachMedia(videoEl);
                if (onReady) inst.on(Hls.Events.MANIFEST_PARSED, () => onReady());
                inst.on(Hls.Events.ERROR, (_, data) => {{
                    if (data && data.fatal) showStatus('HLS error', true);
                }});
                return inst;
            }}
            showStatus('HLS not supported', true);
            return null;
        }}

        function loadIdle(autoPlay = false) {{
            if (idleHls) {{ idleHls.destroy(); idleHls = null; }}
            idleHls = attachHls(idleVideo, idleManifestUrl, {{ lowLatencyMode: false }}, () => {{
                if (autoPlay) attemptPlay(idleVideo);
            }});
            if (autoPlay && idleVideo.canPlayType('application/vnd.apple.mpegurl')) {{
                attemptPlay(idleVideo);
            }}
        }}

        function prepareLive() {{
            if (livePrepared) return;
            livePrepared = true;
            liveRevealPending = true;
            liveRevealed = false;
            if (liveHls) {{ liveHls.destroy(); liveHls = null; }}
            liveHls = attachHls(liveVideo, liveManifestUrl, {{
                lowLatencyMode: false,
                liveSyncDurationCount: 1,
                liveMaxLatencyDurationCount: 3,
                maxBufferLength: 2,
                backBufferLength: 0,
            }}, () => {{
                if (userActivated) attemptPlay(liveVideo);
            }});
            if (userActivated && liveVideo.canPlayType('application/vnd.apple.mpegurl')) {{
                attemptPlay(liveVideo);
            }}
        }}

        function setMode(mode) {{
            if (mode === 'live') {{
                if (currentMode === 'live' && livePrepared) return;
                showStatus(userActivated ? 'Preparing live...' : 'Tap to start', !userActivated);
                prepareLive();
                maybeRevealLive();
                return;
            }}
            if (mode === currentMode) return;
            currentMode = mode;
            showStatus(userActivated ? 'Idle' : 'Tap to start', !userActivated);
            setLayer('idle');
            idleVideo.loop = true;
            attemptPlay(idleVideo);
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
                if (currentMode === 'live') prepareLive();
                attemptPlay(liveRevealPending || currentMode === 'live' ? liveVideo : idleVideo);
                return;
            }}
            attemptPlay(liveRevealPending || currentMode === 'live' ? liveVideo : idleVideo);
        }}

        statusEl.addEventListener('pointerdown', handleTap);
        statusEl.addEventListener('click', handleTap);
        statusEl.addEventListener('touchstart', handleTap);
        container.addEventListener('pointerdown', handleTap);
        container.addEventListener('click', handleTap);
        container.addEventListener('touchstart', handleTap);
        document.addEventListener('pointerdown', handleTap, {{ once: true }});
        if (window.addEventListener) window.addEventListener('message', handleHostMessage);
        if (document && document.addEventListener) document.addEventListener('message', handleHostMessage);

        async function pollStatus() {{
            try {{
                const resp = await fetch(statusUrl, {{ cache: 'no-store' }});
                if (!resp.ok) return;
                const data = await resp.json();

                if (data.idle_duration_seconds && data.idle_duration_seconds > 0) {{
                    idleDuration = data.idle_duration_seconds;
                }}

                if (data.status === 'streaming' && data.live_ready) {{
                    setLiveStreamId(data.active_stream);
                    setMode('live');
                }} else if (data.status === 'streaming' && currentMode === 'idle') {{
                    setLiveStreamId(data.active_stream);
                    showStatus('Preparing live...');
                }} else if (data.status !== 'streaming' && currentMode === 'live') {{
                    showStatus('Finishing...');
                }}
            }} catch (_) {{}}
        }}

        idleVideo.addEventListener('playing', () => {{
            if (currentMode === 'idle' && !liveRevealPending) hideStatus();
        }});

        idleVideo.addEventListener('loadedmetadata', () => {{
            if (idleVideo.duration && Number.isFinite(idleVideo.duration)) {{
                if (!idleDuration || idleDuration <= 0) idleDuration = idleVideo.duration;
            }}
        }});

        idleVideo.addEventListener('waiting', () => {{
            if (currentMode === 'idle') showStatus(userActivated ? 'Buffering...' : 'Tap to start', !userActivated);
        }});

        liveVideo.addEventListener('playing', () => maybeRevealLive());
        liveVideo.addEventListener('waiting', () => {{
            if (currentMode === 'live') showStatus('Buffering...');
        }});
        liveVideo.addEventListener('ended', () => {{
            if (currentMode === 'live') transitionToIdle();
        }});
        liveVideo.addEventListener('stalled', () => {{
            if (currentMode === 'live') showStatus('Buffering...');
        }});
        idleVideo.addEventListener('canplay', () => {{
            if (userActivated && currentMode === 'idle') attemptPlay(idleVideo);
        }});
        liveVideo.addEventListener('canplay', () => {{
            if (userActivated && liveRevealPending) attemptPlay(liveVideo);
            maybeRevealLive();
        }});
        liveVideo.addEventListener('timeupdate', () => maybeRevealLive());
        liveVideo.addEventListener('progress', () => maybeRevealLive());

        showStatus('Tap to start', true);
        loadIdle(false);
        resizeHoldCanvas();
        window.addEventListener('resize', resizeHoldCanvas);
        setInterval(pollStatus, 800);
    </script>
</body>
</html>
"""
