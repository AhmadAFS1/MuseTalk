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
        <video id="hlsVideo" playsinline webkit-playsinline autoplay muted loop></video>
        <div class="status button" id="status">Loading...</div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/hls.js@1/dist/hls.min.js"></script>
    <script>
        const manifestUrl = '{manifest_url}';
        const video = document.getElementById('hlsVideo');
        const status = document.getElementById('status');

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

        function startPlayback() {{
            if (video.canPlayType('application/vnd.apple.mpegurl')) {{
                video.src = manifestUrl;
                video.play()
                    .then(() => hideStatus())
                    .catch(() => showStatus('Tap to play', true));
                return;
            }}

            if (window.Hls && Hls.isSupported()) {{
                const hls = new Hls({{ lowLatencyMode: true }});
                hls.loadSource(manifestUrl);
                hls.attachMedia(video);
                hls.on(Hls.Events.MANIFEST_PARSED, () => {{
                    video.play()
                        .then(() => hideStatus())
                        .catch(() => showStatus('Tap to play', true));
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

        status.addEventListener('click', () => {{
            video.play()
                .then(() => hideStatus())
                .catch(() => showStatus('Tap to play', true));
        }});

        video.addEventListener('playing', () => {{
            hideStatus();
        }});

        video.addEventListener('pause', () => {{
            showStatus('Tap to play', true);
        }});

        startPlayback();
    </script>
</body>
</html>
"""
