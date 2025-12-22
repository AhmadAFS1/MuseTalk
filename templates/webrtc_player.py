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
    </style>
</head>
<body>
    <div class="video-container">
        <video id="remoteVideo" playsinline webkit-playsinline autoplay></video>
        <div class="status-overlay button" id="statusOverlay">Tap to start</div>
    </div>

    <script>
        const SESSION_ID = '{session.session_id}';
        const API_ORIGIN = window.location.origin;
        const ICE_SERVERS = {ice_servers};

        const remoteVideo = document.getElementById('remoteVideo');
        const statusOverlay = document.getElementById('statusOverlay');

        let pc = null;
        let started = false;

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

        async function start() {{
            if (started) return;
            started = true;
            updateStatus('Connecting...');

            pc = new RTCPeerConnection({{ iceServers: ICE_SERVERS }});
            remoteVideo.muted = false;
            remoteVideo.volume = 1.0;

            pc.ontrack = (event) => {{
                if (event.streams && event.streams[0]) {{
                    remoteVideo.srcObject = event.streams[0];
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

            try {{
                await remoteVideo.play();
                hideStatus();
            }} catch (err) {{
                updateStatus('Tap to play', false, true);
            }}
        }}

        statusOverlay.addEventListener('click', () => {{
            if (!started) {{
                start().catch(() => updateStatus('Failed to connect', true, true));
            }} else {{
                remoteVideo.play().then(hideStatus).catch(() => {{
                    updateStatus('Tap to play', false, true);
                }});
            }}
        }});

        start().catch(() => updateStatus('Failed to connect', true, true));

        window.addEventListener('beforeunload', () => {{
            if (pc) {{
                pc.close();
            }}
        }});
    </script>
</body>
</html>
    """
