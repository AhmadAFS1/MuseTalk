def get_hls_wall_html() -> str:
    return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HLS Wall</title>
    <style>
        :root {
            --bg: #0f172a;
            --panel: #111827;
            --panel-2: #1f2937;
            --text: #e5e7eb;
            --muted: #9ca3af;
            --accent: #22c55e;
            --accent-2: #38bdf8;
            --danger: #ef4444;
            --border: rgba(255,255,255,0.08);
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            background: radial-gradient(circle at top, #1e293b 0%, var(--bg) 50%);
            color: var(--text);
            font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .page {
            padding: 24px;
            display: grid;
            gap: 20px;
        }
        .panel {
            background: rgba(17,24,39,0.92);
            border: 1px solid var(--border);
            border-radius: 18px;
            padding: 18px;
            backdrop-filter: blur(12px);
            box-shadow: 0 18px 60px rgba(0,0,0,0.28);
        }
        h1 {
            margin: 0 0 8px;
            font-size: 28px;
        }
        p {
            margin: 0;
            color: var(--muted);
        }
        .controls {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 12px;
            align-items: end;
        }
        label {
            display: grid;
            gap: 6px;
            font-size: 13px;
            color: var(--muted);
        }
        input, button {
            border-radius: 12px;
            border: 1px solid var(--border);
            background: var(--panel-2);
            color: var(--text);
            padding: 10px 12px;
            font-size: 14px;
        }
        input[type="file"] {
            padding: 8px;
        }
        button {
            cursor: pointer;
            font-weight: 600;
        }
        button.primary {
            background: linear-gradient(135deg, var(--accent-2), var(--accent));
            color: #08111f;
            border: none;
        }
        button.danger {
            background: rgba(239,68,68,0.15);
            color: #fecaca;
        }
        .actions {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin-top: 14px;
            align-items: center;
        }
        .meta {
            display: grid;
            gap: 8px;
            margin-top: 14px;
            font-size: 14px;
        }
        .code {
            background: rgba(0,0,0,0.35);
            border-radius: 12px;
            padding: 10px 12px;
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 13px;
            overflow-x: auto;
            white-space: pre-wrap;
            word-break: break-all;
        }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 14px;
        }
        .card {
            background: rgba(15,23,42,0.7);
            border: 1px solid var(--border);
            border-radius: 16px;
            overflow: hidden;
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            gap: 8px;
            padding: 10px 12px;
            font-size: 13px;
            border-bottom: 1px solid var(--border);
        }
        .status {
            color: var(--muted);
        }
        .ok { color: #86efac; }
        .warn { color: #fde68a; }
        .err { color: #fca5a5; }
        iframe {
            width: 100%;
            aspect-ratio: 16 / 9;
            border: none;
            background: #000;
            display: block;
        }
        .empty {
            color: var(--muted);
            text-align: center;
            padding: 48px 18px;
            border: 1px dashed var(--border);
            border-radius: 16px;
        }
    </style>
</head>
<body>
    <div class="page">
        <div class="panel">
            <h1>HLS Session Wall</h1>
            <p>Create a batch of HLS sessions, view them together, and start them all with one upload.</p>
            <div class="controls" style="margin-top: 18px;">
                <label>Avatar ID
                    <input id="avatarId" value="test_avatar" />
                </label>
                <label>Count
                    <input id="count" type="number" min="1" max="12" value="8" />
                </label>
                <label>Playback FPS
                    <input id="playbackFps" type="number" min="1" value="24" />
                </label>
                <label>MuseTalk FPS
                    <input id="musetalkFps" type="number" min="1" value="12" />
                </label>
                <label>Batch Size
                    <input id="batchSize" type="number" min="1" value="2" />
                </label>
                <label>Segment Duration
                    <input id="segmentDuration" type="number" min="0.5" step="0.5" value="2" />
                </label>
            </div>
            <div class="actions">
                <button id="createBtn" class="primary">Create Group</button>
                <input id="audioFile" type="file" accept="audio/*" />
                <button id="startBtn">Start All</button>
                <button id="refreshBtn">Refresh Status</button>
                <button id="deleteBtn" class="danger">Delete Group</button>
            </div>
            <div class="meta">
                <div id="statusLine" class="status">No group created yet.</div>
                <div id="endpointBox" class="code" style="display:none;"></div>
            </div>
        </div>
        <div class="panel">
            <div id="sessionGrid" class="empty">Create a group to populate the wall.</div>
        </div>
    </div>
    <script>
        let currentGroup = null;
        const byId = (id) => document.getElementById(id);
        const initialGroupMatch = window.location.pathname.match(/^\\/hls\\/groups\\/([^/]+)\\/wall$/);
        const initialGroupId = initialGroupMatch ? initialGroupMatch[1] : null;

        function setStatus(text, cls = "status") {
            const el = byId("statusLine");
            el.className = cls;
            el.textContent = text;
        }

        function renderEndpoints() {
            const box = byId("endpointBox");
            if (!currentGroup) {
                box.style.display = "none";
                box.textContent = "";
                return;
            }
            box.style.display = "block";
            box.textContent =
`Group ID: ${currentGroup.group_id}
Wall URL: ${location.origin}/hls/groups/${currentGroup.group_id}/wall
Start-all API: POST ${location.origin}/hls/groups/${currentGroup.group_id}/stream
Status API: GET ${location.origin}/hls/groups/${currentGroup.group_id}
Delete API: DELETE ${location.origin}/hls/groups/${currentGroup.group_id}`;
        }

        function renderSessions() {
            const container = byId("sessionGrid");
            if (!currentGroup || !currentGroup.sessions || currentGroup.sessions.length === 0) {
                container.className = "empty";
                container.innerHTML = "Create a group to populate the wall.";
                return;
            }
            container.className = "grid";
            container.innerHTML = currentGroup.sessions.map((session) => `
                <div class="card">
                    <div class="card-header">
                        <span>${session.session_id}</span>
                        <span class="status">${session.status || "created"}</span>
                    </div>
                    <iframe src="${session.player_url}" allow="autoplay"></iframe>
                </div>
            `).join("");
        }

        async function createGroup() {
            setStatus("Creating sessions...", "warn");
            const params = new URLSearchParams({
                avatar_id: byId("avatarId").value,
                count: byId("count").value,
                playback_fps: byId("playbackFps").value,
                musetalk_fps: byId("musetalkFps").value,
                batch_size: byId("batchSize").value,
                segment_duration: byId("segmentDuration").value,
            });
            const response = await fetch(`/hls/groups/create?${params.toString()}`, { method: "POST" });
            const data = await response.json();
            if (!response.ok) {
                setStatus(data.detail ? JSON.stringify(data.detail) : "Failed to create group.", "err");
                return;
            }
            currentGroup = data;
            renderEndpoints();
            renderSessions();
            setStatus(`Created group ${data.group_id} with ${data.sessions.length} sessions.`, "ok");
        }

        async function startAll() {
            if (!currentGroup) {
                setStatus("Create a group first.", "warn");
                return;
            }
            const file = byId("audioFile").files[0];
            if (!file) {
                setStatus("Choose an audio file first.", "warn");
                return;
            }
            setStatus("Starting all sessions...", "warn");
            const form = new FormData();
            form.append("audio_file", file);
            const response = await fetch(`/hls/groups/${currentGroup.group_id}/stream`, {
                method: "POST",
                body: form,
            });
            const data = await response.json();
            if (!response.ok) {
                setStatus(data.detail ? JSON.stringify(data.detail) : "Failed to start group.", "err");
                return;
            }
            setStatus(`Queued ${data.started} session(s).`, "ok");
            await refreshGroup();
        }

        async function refreshGroup() {
            const groupId = currentGroup ? currentGroup.group_id : initialGroupId;
            if (!groupId) return;
            const response = await fetch(`/hls/groups/${groupId}`);
            const data = await response.json();
            if (!response.ok) {
                setStatus(data.detail ? JSON.stringify(data.detail) : "Failed to refresh group.", "err");
                return;
            }
            currentGroup = data;
            renderEndpoints();
            renderSessions();
            setStatus(`Refreshed group ${data.group_id}.`, "ok");
        }

        async function deleteGroup() {
            if (!currentGroup) {
                setStatus("Nothing to delete.", "warn");
                return;
            }
            setStatus("Deleting group...", "warn");
            const response = await fetch(`/hls/groups/${currentGroup.group_id}`, { method: "DELETE" });
            const data = await response.json();
            if (!response.ok) {
                setStatus(data.detail ? JSON.stringify(data.detail) : "Failed to delete group.", "err");
                return;
            }
            currentGroup = null;
            renderEndpoints();
            renderSessions();
            setStatus(`Deleted ${data.deleted_sessions.length} session(s).`, "ok");
        }

        byId("createBtn").addEventListener("click", createGroup);
        byId("startBtn").addEventListener("click", startAll);
        byId("refreshBtn").addEventListener("click", refreshGroup);
        byId("deleteBtn").addEventListener("click", deleteGroup);
        if (initialGroupId) {
            refreshGroup();
        }
    </script>
</body>
</html>"""
