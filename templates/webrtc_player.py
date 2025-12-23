import json


def get_webrtc_player_html(session) -> str:
    ice_servers = json.dumps(session.ice_servers or [])

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

        .debug-overlay {{
            position: absolute;
            top: 12px;
            left: 12px;
            background: rgba(0,0,0,0.6);
            color: #fff;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 12px;
            line-height: 1.4;
            padding: 8px 10px;
            border-radius: 8px;
            z-index: 20;
            max-width: 90%;
            white-space: pre;
        }}
    </style>
</head>
<body>
    <div class="video-container">
        <video id="remoteVideo" playsinline webkit-playsinline autoplay></video>
        <audio id="remoteAudio" autoplay></audio>
        <div class="status-overlay button" id="statusOverlay">Tap to start (enable audio)</div>
        <div class="debug-overlay" id="debugOverlay">debug: idle</div>
    </div>

    <script>
        const SESSION_ID = '{session.session_id}';
        const API_ORIGIN = window.location.origin;
        const ICE_SERVERS = {ice_servers};

        const remoteVideo = document.getElementById('remoteVideo');
        const remoteAudio = document.getElementById('remoteAudio');
        const statusOverlay = document.getElementById('statusOverlay');
        const debugOverlay = document.getElementById('debugOverlay');

        let pc = null;
        let started = false;
        let audioUnlocked = false;
        let remoteStream = null;
        let lastAudioBytes = null;
        let lastAudioTs = null;

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

        function setDebug(lines) {{
            debugOverlay.textContent = lines.join('\\n');
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

                const aStream = remoteAudio.srcObject;
                remoteAudio.pause();
                remoteAudio.srcObject = null;
                remoteAudio.srcObject = aStream;

                remoteVideo.muted = false;
                remoteVideo.volume = 1.0;
                remoteAudio.muted = false;
                remoteAudio.volume = 1.0;
                await remoteVideo.play();
                await remoteAudio.play();
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

            pc = new RTCPeerConnection({{ iceServers: ICE_SERVERS }});
            remoteStream = new MediaStream();
            remoteVideo.srcObject = remoteStream;
            // Audio is unlocked via user gesture before calling start().
            remoteVideo.muted = !audioUnlocked;
            remoteVideo.volume = 1.0;
            remoteAudio.muted = !audioUnlocked;
            remoteAudio.volume = 1.0;

            pc.ontrack = (event) => {{
                if (event.track && event.track.kind === 'video') {{
                    if (!remoteStream) {{
                        remoteStream = new MediaStream();
                        remoteVideo.srcObject = remoteStream;
                    }}
                    if (!remoteStream.getTracks().includes(event.track)) {{
                        remoteStream.addTrack(event.track);
                    }}
                    return;
                }}

                if (event.track && event.track.kind === 'audio') {{
                    const audioStream = new MediaStream([event.track]);
                    remoteAudio.srcObject = audioStream;
                    if (!audioUnlocked) {{
                        updateStatus('Tap to start (enable audio)', false, true);
                    }} else {{
                        remoteAudio.play().catch(() => {{}});
                    }}
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
            hideStatus();
        }}

        async function updateDebugStats() {{
            if (!pc) return;
            const lines = [];
            lines.push('pc: ' + pc.connectionState + ' / ice: ' + pc.iceConnectionState);
            const vTracks = remoteVideo.srcObject ? remoteVideo.srcObject.getVideoTracks().length : 0;
            const aTracks = remoteAudio.srcObject ? remoteAudio.srcObject.getAudioTracks().length : 0;
            lines.push('tracks: video=' + vTracks + ' audio=' + aTracks);

            try {{
                const stats = await pc.getStats();
                let audioReport = null;
                stats.forEach(r => {{
                    if (r.type === 'inbound-rtp' && (r.kind === 'audio' || r.mediaType === 'audio')) {{
                        audioReport = r;
                    }}
                }});
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
                remoteAudio.muted = false;
                remoteAudio.volume = 1.0;
                remoteVideo.play().catch(() => {{}});
                remoteAudio.play().catch(() => {{}});
                start().catch(() => updateStatus('Failed to connect', true, true));
                return;
            }}
            unlockAudio();
        }}

        statusOverlay.addEventListener('pointerdown', handleTap);
        statusOverlay.addEventListener('click', handleTap);

        updateStatus('Tap to start (enable audio)', false, true);
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
