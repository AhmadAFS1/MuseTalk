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
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
        }
        .metric-card {
            background: rgba(15,23,42,0.72);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 12px;
            display: grid;
            gap: 4px;
        }
        .metric-label {
            color: var(--muted);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .metric-value {
            font-size: 24px;
            font-weight: 700;
        }
        .metric-sub {
            color: var(--muted);
            font-size: 12px;
        }
        .section-title {
            margin: 0 0 12px;
            font-size: 16px;
            font-weight: 700;
        }
        .metrics-layout {
            display: grid;
            gap: 14px;
        }
        .job-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 12px;
        }
        .job-table th,
        .job-table td {
            padding: 8px 10px;
            border-bottom: 1px solid var(--border);
            text-align: left;
            vertical-align: top;
        }
        .job-table th {
            color: var(--muted);
            font-weight: 600;
        }
        .muted {
            color: var(--muted);
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
            <div class="metrics-layout">
                <div>
                    <div class="section-title">Live Metrics</div>
                    <div id="metricsGrid" class="metrics-grid"></div>
                </div>
                <div>
                    <div class="section-title">Scheduler Jobs</div>
                    <div id="jobTableWrap" class="code">No live scheduler jobs yet.</div>
                </div>
            </div>
        </div>
        <div class="panel">
            <div id="sessionGrid" class="empty">Create a group to populate the wall.</div>
        </div>
    </div>
    <script>
        let currentGroup = null;
        let autoRefreshTimer = null;
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

        function getSessionStatusClass(status) {
            const value = String(status || "created").toLowerCase();
            if (value === "failed" || value === "error") return "status err";
            if (value === "streaming" || value === "completed" || value === "live") return "status ok";
            if (value === "starting" || value === "queued" || value === "preparing") return "status warn";
            return "status";
        }

        function buildSessionCard(session) {
            const card = document.createElement("div");
            card.className = "card";
            card.dataset.sessionId = session.session_id;

            const header = document.createElement("div");
            header.className = "card-header";

            const idSpan = document.createElement("span");
            idSpan.textContent = session.session_id;

            const statusSpan = document.createElement("span");
            statusSpan.className = getSessionStatusClass(session.status);
            statusSpan.dataset.role = "session-status";
            statusSpan.textContent = session.status || "created";

            header.appendChild(idSpan);
            header.appendChild(statusSpan);

            const frame = document.createElement("iframe");
            frame.src = session.player_url;
            frame.allow = "autoplay";
            frame.loading = "eager";

            card.appendChild(header);
            card.appendChild(frame);
            return card;
        }

        function renderSessions() {
            const container = byId("sessionGrid");
            const sessions = currentGroup && currentGroup.sessions ? currentGroup.sessions : [];
            if (sessions.length === 0) {
                container.className = "empty";
                container.textContent = "Create a group to populate the wall.";
                return;
            }

            container.className = "grid";
            const existingCards = new Map(
                Array.from(container.querySelectorAll(".card[data-session-id]")).map((card) => [card.dataset.sessionId, card])
            );

            const nextCards = [];
            for (const session of sessions) {
                let card = existingCards.get(session.session_id);
                if (!card) {
                    card = buildSessionCard(session);
                } else {
                    const statusEl = card.querySelector('[data-role="session-status"]');
                    if (statusEl) {
                        statusEl.className = getSessionStatusClass(session.status);
                        statusEl.textContent = session.status || "created";
                    }
                    const frame = card.querySelector("iframe");
                    if (frame && frame.getAttribute("src") !== session.player_url) {
                        frame.setAttribute("src", session.player_url);
                    }
                    existingCards.delete(session.session_id);
                }
                nextCards.push(card);
            }

            for (const staleCard of existingCards.values()) {
                staleCard.remove();
            }

            let needsReplace = container.children.length !== nextCards.length;
            if (!needsReplace) {
                for (let i = 0; i < nextCards.length; i += 1) {
                    if (container.children[i] !== nextCards[i]) {
                        needsReplace = true;
                        break;
                    }
                }
            }

            if (needsReplace) {
                container.replaceChildren(...nextCards);
            }
        }

        function formatNum(value, digits = 1, suffix = "") {
            if (value === null || value === undefined || Number.isNaN(Number(value))) {
                return "n/a";
            }
            return `${Number(value).toFixed(digits)}${suffix}`;
        }

        function renderMetricCards(metrics) {
            const el = byId("metricsGrid");
            const cards = [
                {
                    label: "GPU Util",
                    value: formatNum(metrics.gpuUtil, 1, "%"),
                    sub: `Peak unknown on wall; current live sample`,
                },
                {
                    label: "VRAM Used",
                    value: metrics.gpuMemGb,
                    sub: `${metrics.gpuMemMb} / ${metrics.gpuTotalGb}`,
                },
                {
                    label: "GPU Temp",
                    value: formatNum(metrics.gpuTemp, 0, "C"),
                    sub: `Power ${formatNum(metrics.gpuPower, 0, "W")}`,
                },
                {
                    label: "HLS Jobs",
                    value: String(metrics.hlsJobs ?? 0),
                    sub: `Preparing ${metrics.preparingJobs ?? 0}, startup ${metrics.startupJobs ?? 0}`,
                },
                {
                    label: "Sessions",
                    value: String(metrics.totalSessions ?? 0),
                    sub: `Streaming ${metrics.streamingSessions ?? 0}`,
                },
                {
                    label: "GPU Lease",
                    value: metrics.leaseGb,
                    sub: `Slots ${metrics.slotsInUse ?? 0}/${metrics.maxSlots ?? 0}`,
                },
                {
                    label: "Compose / Encode",
                    value: `${metrics.composeWorkers ?? 0} / ${metrics.encodeWorkers ?? 0}`,
                    sub: `worker pools`,
                },
                {
                    label: "Cache",
                    value: String(metrics.cachedAvatars ?? 0),
                    sub: `hit rate ${metrics.cacheHitRate ?? "n/a"}`,
                },
            ];

            el.innerHTML = cards.map((card) => `
                <div class="metric-card">
                    <div class="metric-label">${card.label}</div>
                    <div class="metric-value">${card.value}</div>
                    <div class="metric-sub">${card.sub}</div>
                </div>
            `).join("");
        }

        function renderJobTable(jobs) {
            const wrap = byId("jobTableWrap");
            if (!jobs || jobs.length === 0) {
                wrap.textContent = "No live scheduler jobs yet.";
                return;
            }
            const rows = jobs.slice(0, 8).map((job) => `
                <tr>
                    <td>${job.request_id}</td>
                    <td>${job.current_frame_idx}/${job.total_frames}</td>
                    <td>${formatNum(job.time_to_first_chunk_s, 2, "s")}</td>
                    <td>${formatNum(job.avg_gpu_batch_s, 3, "s")}</td>
                    <td>${formatNum(job.avg_compose_s, 3, "s")}</td>
                    <td>${formatNum(job.avg_encode_s, 3, "s")}</td>
                    <td>${job.frame_buffer_len}/${job.frames_per_chunk} <span class="muted">(max ${job.max_frame_buffer_len})</span></td>
                    <td>${job.pending_composes}/${job.pending_encodes} <span class="muted">(max ${job.max_pending_composes}/${job.max_pending_encodes})</span></td>
                    <td>${formatNum(job.post_generation_drain_s, 2, "s")}</td>
                </tr>
            `).join("");
            wrap.innerHTML = `
                <table class="job-table">
                    <thead>
                        <tr>
                            <th>Request</th>
                            <th>Frames</th>
                            <th>First Chunk</th>
                            <th>GPU Batch</th>
                            <th>Compose</th>
                            <th>Encode</th>
                            <th>Buffer</th>
                            <th>Pending</th>
                            <th>Drain</th>
                        </tr>
                    </thead>
                    <tbody>${rows}</tbody>
                </table>
            `;
        }

        async function refreshMetrics() {
            try {
                const [liveGpuResp, statsResp, hlsStatsResp] = await Promise.all([
                    fetch("/stats/gpu-live"),
                    fetch("/stats"),
                    fetch("/hls/sessions/stats"),
                ]);
                const liveGpu = await liveGpuResp.json();
                const stats = await statsResp.json();
                const hlsStats = await hlsStatsResp.json();

                const scheduler = hlsStats.scheduler || stats.hls_scheduler || {};
                const sessions = hlsStats.sessions || [];
                const streamingSessions = sessions.filter((s) => s.status === "streaming" || s.active_stream).length;
                const gpuLease = stats.gpu || {};
                const compute = gpuLease.compute || {};
                const cache = stats.cache || {};

                renderMetricCards({
                    gpuUtil: liveGpu.gpu_util_pct,
                    gpuMemGb: `${formatNum(liveGpu.memory_used_gb, 2, "GB")}`,
                    gpuMemMb: `${formatNum(liveGpu.memory_used_mb, 0, "MB")}`,
                    gpuTotalGb: formatNum(liveGpu.memory_total_gb, 2, "GB"),
                    gpuTemp: liveGpu.temperature_c,
                    gpuPower: liveGpu.power_draw_w,
                    hlsJobs: scheduler.queued_or_active_jobs,
                    preparingJobs: scheduler.preparing_jobs ?? scheduler.prep_queue_depth,
                    startupJobs: scheduler.startup_pending_jobs,
                    totalSessions: hlsStats.total_sessions,
                    streamingSessions,
                    leaseGb: `${formatNum(gpuLease.current_usage_gb, 2, "GB")}`,
                    slotsInUse: compute.slots_in_use,
                    maxSlots: compute.max_live_generations,
                    composeWorkers: scheduler.compose_workers,
                    encodeWorkers: scheduler.encode_workers,
                    cachedAvatars: cache.cached_avatars,
                    cacheHitRate: cache.hit_rate,
                });
                renderJobTable(scheduler.jobs || []);
            } catch (err) {
                byId("metricsGrid").innerHTML = `<div class="metric-card"><div class="metric-label">Metrics</div><div class="metric-value">n/a</div><div class="metric-sub">${String(err)}</div></div>`;
            }
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

        async function refreshGroup(silent = false) {
            const groupId = currentGroup ? currentGroup.group_id : initialGroupId;
            if (!groupId) return;
            const response = await fetch(`/hls/groups/${groupId}`);
            const data = await response.json();
            if (!response.ok) {
                if (!silent) {
                    setStatus(data.detail ? JSON.stringify(data.detail) : "Failed to refresh group.", "err");
                }
                return;
            }
            currentGroup = data;
            renderEndpoints();
            renderSessions();
            if (!silent) {
                setStatus(`Refreshed group ${data.group_id}.`, "ok");
            }
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
        byId("refreshBtn").addEventListener("click", () => refreshGroup(false));
        byId("deleteBtn").addEventListener("click", deleteGroup);
        refreshMetrics();
        autoRefreshTimer = setInterval(() => {
            refreshMetrics();
            if (currentGroup || initialGroupId) {
                refreshGroup(true);
            }
        }, 2000);
        if (initialGroupId) {
            refreshGroup(true);
        }
    </script>
</body>
</html>"""
