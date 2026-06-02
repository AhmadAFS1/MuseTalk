from html import escape

from templates.hls_wall import get_hls_wall_html


def get_webrtc_wall_html(default_avatar_id: str = "test_avatar") -> str:
    html = get_hls_wall_html()
    safe_default_avatar_id = escape(default_avatar_id, quote=True)
    replacements = {
        "<title>HLS Wall</title>": "<title>WebRTC Wall</title>",
        "HLS Session Wall": "WebRTC Session Wall",
        "Create a batch of HLS sessions, view them together, and start them all with one upload.": (
            "Create a batch of WebRTC sessions, view them together, and start them all with one upload."
        ),
        "HLS Jobs": "WebRTC Streams",
        "No live scheduler jobs yet.": "No live WebRTC streams yet.",
        "Status API: GET ${location.origin}/hls/groups/${currentGroup.group_id}": (
            "Status API: GET ${location.origin}/webrtc/groups/${currentGroup.group_id}"
        ),
    }
    for old, new in replacements.items():
        html = html.replace(old, new)

    html = html.replace("/hls/", "/webrtc/")
    html = html.replace('/hls/sessions/stats', '/webrtc/sessions/stats')
    html = html.replace('id="batchSize" type="number" min="1" value="2"', 'id="batchSize" type="number" min="1" value="8"')
    html = html.replace('id="avatarId" value="test_avatar"', f'id="avatarId" value="{safe_default_avatar_id}"')
    html = html.replace('const initialGroupMatch = window.location.pathname.match(/^\\/hls\\/groups\\/([^/]+)\\/wall$/);',
                        'const initialGroupMatch = window.location.pathname.match(/^\\/webrtc\\/groups\\/([^/]+)\\/wall$/);')
    html = html.replace('frame.allow = "autoplay";', 'frame.allow = "autoplay; fullscreen";')
    html = html.replace(
        '    </style>',
        """        .debug-toggle-on {
            border-color: rgba(56,189,248,0.5);
            color: #bae6fd;
        }
    </style>""",
    )
    html = html.replace(
        '<button id="deleteBtn" class="danger">Delete Group</button>',
        '<button id="deleteBtn" class="danger">Delete Group</button>\n'
        '                <button id="webrtcPlayAllBtn" type="button">Reconnect All</button>\n'
        '                <button id="webrtcDebugBtn" type="button">Stats: Off</button>',
    )
    html = html.replace(
        '        const initialGroupId = initialGroupMatch ? initialGroupMatch[1] : null;\n',
        """        const initialGroupId = initialGroupMatch ? initialGroupMatch[1] : null;
        const webrtcDebugModes = ["off", "docked", "overlay"];
        let webrtcDebugMode = localStorage.getItem("webrtcWallDebugMode") || "off";
        if (!webrtcDebugModes.includes(webrtcDebugMode)) {
            webrtcDebugMode = "off";
        }

        function getWebrtcPlayerUrl(playerUrl) {
            const url = new URL(playerUrl, location.origin);
            url.searchParams.set("debug", webrtcDebugMode);
            return url.pathname + url.search + url.hash;
        }

        function updateWebrtcDebugButton() {
            const btn = byId("webrtcDebugBtn");
            if (!btn) return;
            const label = webrtcDebugMode.charAt(0).toUpperCase() + webrtcDebugMode.slice(1);
            btn.textContent = `Stats: ${label}`;
            btn.classList.toggle("debug-toggle-on", webrtcDebugMode !== "off");
        }

        function postToWebrtcFrames(message) {
            document.querySelectorAll("iframe").forEach((frame) => {
                try {
                    frame.contentWindow.postMessage(message, location.origin);
                } catch (err) {}
            });
        }

        function setWebrtcDebugMode(mode) {
            if (!webrtcDebugModes.includes(mode)) {
                mode = "off";
            }
            webrtcDebugMode = mode;
            localStorage.setItem("webrtcWallDebugMode", mode);
            updateWebrtcDebugButton();
            postToWebrtcFrames({ type: "webrtc-debug-mode", mode });
        }

        function cycleWebrtcDebugMode() {
            const index = webrtcDebugModes.indexOf(webrtcDebugMode);
            const nextMode = webrtcDebugModes[(index + 1) % webrtcDebugModes.length];
            setWebrtcDebugMode(nextMode);
        }

        function connectWebrtcPlayers() {
            document.querySelectorAll("iframe").forEach((frame) => {
                try {
                    if (
                        frame.contentWindow &&
                        typeof frame.contentWindow.startWebrtcPlayer === "function"
                    ) {
                        frame.contentWindow.startWebrtcPlayer(false);
                    } else if (frame.contentWindow) {
                        frame.contentWindow.postMessage(
                            { type: "webrtc-start", muted: false },
                            location.origin
                        );
                    }
                } catch (err) {}
            });
        }
""",
    )
    html = html.replace(
        '            frame.src = session.player_url;',
        '            frame.src = getWebrtcPlayerUrl(session.player_url);',
    )
    html = html.replace(
        """                    const frame = card.querySelector("iframe");
                    if (frame && frame.getAttribute("src") !== session.player_url) {
                        frame.setAttribute("src", session.player_url);
                    }""",
        """                    const frame = card.querySelector("iframe");
                    const playerUrl = getWebrtcPlayerUrl(session.player_url);
                    if (frame && frame.getAttribute("src") !== playerUrl) {
                        frame.setAttribute("src", playerUrl);
                    }""",
    )
    html = html.replace(
        '        byId("deleteBtn").addEventListener("click", deleteGroup);',
        """        byId("deleteBtn").addEventListener("click", deleteGroup);
        byId("webrtcPlayAllBtn").addEventListener("click", connectWebrtcPlayers);
        byId("webrtcDebugBtn").addEventListener("click", cycleWebrtcDebugMode);""",
    )
    html = html.replace(
        '        refreshMetrics();',
        '        updateWebrtcDebugButton();\n        refreshMetrics();',
        1,
    )
    return html
