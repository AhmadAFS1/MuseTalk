import json


def get_webrtc_player_html(session) -> str:
    ice_servers = json.dumps(session.ice_servers or [])
    ice_transport_policy = json.dumps(getattr(session, "ice_transport_policy", "all"))
    source_fps = getattr(session, "fps", 10)
    playback_fps = getattr(session, "playback_fps", source_fps)

    return f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <meta name="apple-mobile-web-app-capable" content="yes">
    <title>WebRTC Avatar - {session.avatar_id}</title>
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

        body {{
            display: flex;
            flex-direction: column;
            position: relative;
        }}

        .video-container {{
            width: 100%;
            flex: 1 1 auto;
            min-height: 0;
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

        .status-overlay.button {{
            cursor: pointer;
        }}

        .error {{
            color: #ff5555;
        }}

        .debug-panel {{
            flex: 0 0 auto;
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 2px 12px;
            background: rgba(3,7,18,0.94);
            border-top: 1px solid rgba(255,255,255,0.12);
            color: #fff;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 11px;
            line-height: 1.35;
            padding: 6px 8px;
            z-index: 20;
        }}

        .debug-line {{
            min-width: 0;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        .debug-line:first-child {{
            grid-column: 1 / -1;
        }}

        body.debug-overlay-mode .debug-panel {{
            position: absolute;
            top: 12px;
            left: 12px;
            max-width: calc(100% - 24px);
            border: none;
            border-radius: 8px;
            background: rgba(0,0,0,0.6);
            z-index: 20;
        }}

        body.debug-off .debug-panel {{
            display: none;
        }}
    </style>
</head>
	<body>
	    <div class="video-container">
	        <video id="remoteVideo" playsinline webkit-playsinline autoplay muted></video>
	        <div class="status-overlay button" id="statusOverlay">Tap to start (enable audio)</div>
	    </div>
	    <div class="debug-panel" id="debugOverlay"></div>

    <script>
        const SESSION_ID = '{session.session_id}';
        const API_ORIGIN = window.location.origin;
        const ICE_SERVERS = {ice_servers};
        const ICE_TRANSPORT_POLICY = {ice_transport_policy};
        const SOURCE_FPS = {source_fps};
        const PLAYBACK_FPS = {playback_fps};
        const DEBUG_MODES = ['docked', 'off', 'overlay'];

        const remoteVideo = document.getElementById('remoteVideo');
        const statusOverlay = document.getElementById('statusOverlay');
        const debugOverlay = document.getElementById('debugOverlay');
        const queryParams = new URLSearchParams(window.location.search);
        let debugMode = queryParams.get('debug') || 'docked';
        if (!DEBUG_MODES.includes(debugMode)) {{
            debugMode = 'docked';
        }}

        let pc = null;
        let started = false;
        let audioUnlocked = false;
        let remoteStream = null;
        let lastAudioBytes = null;
        let lastAudioTs = null;
        let lastVideoFrames = null;
        let lastVideoTs = null;

        function updateStatus(message, isError = false, isButton = false) {{
            statusOverlay.textContent = message;
            statusOverlay.classList.remove('hidden');
            if (isError) {{
                statusOverlay.classList.add('error');
            }} else {{
                statusOverlay.classList.remove('error');
            }}
            if (isButton) {{
                statusOverlay.classList.add('button');
            }} else {{
                statusOverlay.classList.remove('button');
            }}
        }}

        function hideStatus() {{
            statusOverlay.classList.add('hidden');
        }}

        function applyDebugMode(mode) {{
            if (!DEBUG_MODES.includes(mode)) {{
                mode = 'docked';
            }}
            debugMode = mode;
            document.body.classList.toggle('debug-off', mode === 'off');
            document.body.classList.toggle('debug-overlay-mode', mode === 'overlay');
        }}

        function setDebug(lines) {{
            debugOverlay.replaceChildren(...lines.map((line) => {{
                const item = document.createElement('div');
                item.className = 'debug-line';
                item.textContent = line;
                return item;
            }}));
        }}

        async function sendIceCandidate(candidate) {{
            if (!candidate) return;
            await fetch(`${{API_ORIGIN}}/webrtc/sessions/${{SESSION_ID}}/ice`, {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    candidate: candidate.candidate,
                    sdpMid: candidate.sdpMid,
                    sdpMLineIndex: candidate.sdpMLineIndex
                }})
            }});
        }}

        async function unlockAudio() {{
            try {{
                const vStream = remoteVideo.srcObject;
                remoteVideo.pause();
                remoteVideo.srcObject = null;
                remoteVideo.srcObject = vStream;

                remoteVideo.muted = false;
                remoteVideo.volume = 1.0;
                await remoteVideo.play();
                audioUnlocked = true;
                hideStatus();
            }} catch (err) {{
                updateStatus('Tap to start (enable audio)', false, true);
            }}
        }}

        async function start() {{
            if (started) return;
            started = true;
            updateStatus('Connecting...');
            setDebug(['debug: connecting']);

            pc = new RTCPeerConnection({{
                iceServers: ICE_SERVERS,
                iceTransportPolicy: ICE_TRANSPORT_POLICY
            }});
            remoteStream = new MediaStream();
            remoteVideo.srcObject = remoteStream;
            // Audio is unlocked via user gesture before calling start().
            remoteVideo.muted = !audioUnlocked;
            remoteVideo.volume = audioUnlocked ? 1.0 : 0.0;
            remoteVideo.play().catch(() => {{
                if (audioUnlocked) {{
                    audioUnlocked = false;
                    remoteVideo.muted = true;
                    remoteVideo.volume = 0.0;
                    remoteVideo.play().catch(() => {{}});
                    updateStatus('Tap to start (enable audio)', false, true);
                }}
            }});

            pc.ontrack = (event) => {{
                if (!event.track) {{
                    return;
                }}

                if (!remoteStream) {{
                    remoteStream = new MediaStream();
                }}
                if (!remoteStream.getTracks().includes(event.track)) {{
                    remoteStream.addTrack(event.track);
                }}
                remoteVideo.srcObject = remoteStream;

                if (event.track.kind === 'audio' && !audioUnlocked) {{
                    remoteVideo.muted = true;
                    remoteVideo.volume = 0.0;
                    remoteVideo.play().catch(() => {{}});
                    updateStatus('Tap to start (enable audio)', false, true);
                    return;
                }}

                if (audioUnlocked) {{
                    remoteVideo.muted = false;
                    remoteVideo.volume = 1.0;
                    remoteVideo.play().catch(() => updateStatus('Tap to start (enable audio)', false, true));
                }}
            }};

            pc.onicecandidate = (event) => {{
                if (event.candidate) {{
                    sendIceCandidate(event.candidate).catch(() => {{}});
                }}
            }};

            pc.oniceconnectionstatechange = () => {{
                const state = pc.iceConnectionState;
                if (state === 'failed' || state === 'disconnected') {{
                    updateStatus('Connection lost', true, true);
                    started = false;
                    audioUnlocked = false;
                }}
            }};

            pc.addTransceiver('video', {{ direction: 'recvonly' }});
            pc.addTransceiver('audio', {{ direction: 'recvonly' }});

            const offer = await pc.createOffer();
            await pc.setLocalDescription(offer);

            const resp = await fetch(`${{API_ORIGIN}}/webrtc/sessions/${{SESSION_ID}}/offer`, {{
                method: 'POST',
                headers: {{ 'Content-Type': 'application/json' }},
                body: JSON.stringify({{
                    sdp: pc.localDescription.sdp,
                    type: pc.localDescription.type
                }})
            }});

            if (!resp.ok) {{
                updateStatus('Failed to connect', true, true);
                started = false;
                return;
            }}

            const answer = await resp.json();
            await pc.setRemoteDescription(answer);
            remoteVideo.muted = !audioUnlocked;
            remoteVideo.volume = audioUnlocked ? 1.0 : 0.0;
            remoteVideo.play().catch(() => {{
                if (audioUnlocked) {{
                    audioUnlocked = false;
                    remoteVideo.muted = true;
                    remoteVideo.volume = 0.0;
                    remoteVideo.play().catch(() => {{}});
                    updateStatus('Tap to start (enable audio)', false, true);
                }}
            }});
            hideStatus();
        }}

        async function startFromWall(muted = false) {{
            audioUnlocked = !muted;
            remoteVideo.muted = muted;
            remoteVideo.volume = muted ? 0.0 : 1.0;
            if (!started) {{
                await start();
                return;
            }}
            if (muted) {{
                remoteVideo.play().catch(() => {{}});
            }} else {{
                await unlockAudio();
            }}
        }}
        window.startWebrtcPlayer = startFromWall;

        async function updateDebugStats() {{
            if (!pc) return;
            const lines = [];
            lines.push('pc: ' + pc.connectionState + ' / ice: ' + pc.iceConnectionState);
            lines.push('ice policy: ' + ICE_TRANSPORT_POLICY);
            const vTracks = remoteVideo.srcObject ? remoteVideo.srcObject.getVideoTracks().length : 0;
            const aTracks = remoteVideo.srcObject ? remoteVideo.srcObject.getAudioTracks().length : 0;
            lines.push('tracks: video=' + vTracks + ' audio=' + aTracks);
            lines.push('fps target: src=' + SOURCE_FPS + ' out=' + PLAYBACK_FPS);
            lines.push('audioRoute: video element');

            try {{
                const stats = await pc.getStats();
                let audioReport = null;
                let videoReport = null;
                stats.forEach(r => {{
                    if (r.type === 'inbound-rtp' && (r.kind === 'audio' || r.mediaType === 'audio')) {{
                        audioReport = r;
                    }}
                    if (r.type === 'inbound-rtp' && (r.kind === 'video' || r.mediaType === 'video')) {{
                        videoReport = r;
                    }}
                }});
                if (videoReport) {{
                    const now = Date.now();
                    let fps = null;
                    if (typeof videoReport.framesPerSecond === 'number') {{
                        fps = videoReport.framesPerSecond;
                    }} else {{
                        const frames = videoReport.framesDecoded ?? videoReport.framesReceived ?? 0;
                        if (lastVideoFrames !== null && lastVideoTs !== null) {{
                            const deltaFrames = frames - lastVideoFrames;
                            const deltaMs = now - lastVideoTs;
                            if (deltaMs > 0) {{
                                fps = (deltaFrames * 1000) / deltaMs;
                            }}
                        }}
                        lastVideoFrames = frames;
                        lastVideoTs = now;
                    }}
                    lines.push('video fps: ' + (fps !== null ? fps.toFixed(1) : 'n/a'));
                }} else {{
                    lines.push('video fps: n/a');
                }}
                if (audioReport) {{
                    const bytes = audioReport.bytesReceived || 0;
                    const packets = audioReport.packetsReceived || 0;
                    let kbps = 'n/a';
                    const now = Date.now();
                    if (lastAudioBytes !== null && lastAudioTs !== null) {{
                        const deltaBytes = bytes - lastAudioBytes;
                        const deltaMs = now - lastAudioTs;
                        if (deltaMs > 0) {{
                            kbps = ((deltaBytes * 8) / deltaMs).toFixed(1);
                        }}
                    }}
                    lastAudioBytes = bytes;
                    lastAudioTs = now;
                    lines.push('audio bytes: ' + bytes + ' pkts: ' + packets + ' kbps: ' + kbps);
                    if (audioReport.jitter !== undefined) {{
                        lines.push('audio jitter: ' + audioReport.jitter);
                    }}
                }} else {{
                    lines.push('audio inbound: none');
                }}
            }} catch (err) {{
                lines.push('stats error: ' + err);
            }}

            setDebug(lines);
        }}

        let lastTapMs = 0;
        function handleTap() {{
            const now = Date.now();
            if (now - lastTapMs < 350) {{
                return;
            }}
            lastTapMs = now;
            if (!started) {{
                // Unlock audio in the user gesture before async negotiation.
                audioUnlocked = true;
                remoteVideo.muted = false;
                remoteVideo.volume = 1.0;
                remoteVideo.play().catch(() => {{}});
                start().catch(() => updateStatus('Failed to connect', true, true));
                return;
            }}
            unlockAudio();
        }}

        statusOverlay.addEventListener('pointerdown', handleTap);
        statusOverlay.addEventListener('click', handleTap);

        window.addEventListener('message', (event) => {{
            if (event.origin !== window.location.origin) {{
                return;
            }}
            const data = event.data || {{}};
            if (data.type === 'webrtc-debug-mode') {{
                applyDebugMode(data.mode);
            }} else if (data.type === 'webrtc-start') {{
                startFromWall(Boolean(data.muted)).catch(() => {{
                    updateStatus('Tap to start (enable audio)', false, true);
                }});
            }}
        }});

        applyDebugMode(debugMode);
        updateStatus('Tap to start (enable audio)', false, true);
        setDebug(['debug: idle']);
        setInterval(updateDebugStats, 2000);

        window.addEventListener('beforeunload', () => {{
            if (pc) {{
                pc.close();
            }}
        }});
    </script>
</body>
</html>
    """
