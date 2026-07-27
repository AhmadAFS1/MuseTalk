"""Standalone browser lab for MuseTalk pose-protocol WebRTC sessions.

The page intentionally has no framework or external assets.  It talks directly
to the worker so it can be used on a GPU host without the Lingua broker.
"""

from __future__ import annotations

import json


POSE_IDS = (
    "neutral_resting",
    "active_listening",
    "speaking_direct",
    "nod_agree",
    "empathetic_head_tilt",
    "light_smile",
)

DEFAULT_POSE_SET = {
    "version": 1,
    "pose_set_id": "indian_tutor_essential_six_v1",
    "default_pose_id": "neutral_resting",
    "switch_mode": "next_boundary",
    "poses": {
        "neutral_resting": {
            "avatar_id": "indian_tutor_essential_six_v1_neutral_resting",
            "role": "idle",
            "duration_seconds": 10,
            "cycle_seconds": 10,
            "fps": 30,
            "frame_count": 300,
        },
        "active_listening": {
            "avatar_id": "indian_tutor_essential_six_v1_active_listening",
            "role": "listening",
            "duration_seconds": 8,
            "cycle_seconds": 8,
            "fps": 30,
            "frame_count": 240,
        },
        "speaking_direct": {
            "avatar_id": "indian_tutor_essential_six_v1_speaking_direct",
            "role": "talking",
            "duration_seconds": 10,
            "cycle_seconds": 10,
            "fps": 30,
            "frame_count": 300,
        },
        "nod_agree": {
            "avatar_id": "indian_tutor_essential_six_v1_nod_agree",
            "role": "reaction",
            "duration_seconds": 2.933333,
            "cycle_seconds": 2.933333,
            "fps": 30,
            "frame_count": 88,
        },
        "empathetic_head_tilt": {
            "avatar_id": "indian_tutor_essential_six_v1_empathetic_head_tilt",
            "role": "reaction",
            "duration_seconds": 4.8,
            "cycle_seconds": 4.8,
            "fps": 30,
            "frame_count": 144,
        },
        "light_smile": {
            "avatar_id": "indian_tutor_essential_six_v1_light_smile",
            "role": "reaction",
            "duration_seconds": 4,
            "cycle_seconds": 4,
            "fps": 30,
            "frame_count": 120,
        },
    },
}


def _validated_pose_set(pose_set: dict | None) -> dict:
    value = pose_set or DEFAULT_POSE_SET
    poses = value.get("poses") if isinstance(value, dict) else None
    if not isinstance(poses, dict) or tuple(poses) != POSE_IDS:
        raise ValueError("Pose lab requires the six protocol poses in canonical order.")
    for pose_id in POSE_IDS:
        entry = poses.get(pose_id)
        if not isinstance(entry, dict) or not str(entry.get("avatar_id") or "").strip():
            raise ValueError(f"Pose lab entry '{pose_id}' requires avatar_id.")
    if value.get("default_pose_id") != "neutral_resting":
        raise ValueError("Pose lab default_pose_id must be neutral_resting.")
    return value


def get_webrtc_pose_lab_html(
    pose_set: dict | None = None,
    *,
    sample_audio_url: str = "/webrtc/pose-lab/sample-audio",
) -> str:
    """Return the self-contained worker-side pose lab.

    ``sample_audio_url`` should point to a local WAV served by MuseTalk.  A user
    can always choose a WAV from the browser instead.
    """

    embedded_pose_set = json.dumps(
        _validated_pose_set(pose_set),
        separators=(",", ":"),
    ).replace("</", "<\\/")
    embedded_sample_url = json.dumps(sample_audio_url).replace("</", "<\\/")

    return (
        r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>MuseTalk pose lab</title>
  <style>
    :root {
      color-scheme: dark;
      --bg: #080b10;
      --panel: #111720;
      --panel-2: #171f2a;
      --line: #283341;
      --ink: #f5f7fa;
      --muted: #9ca9b8;
      --accent: #a5f3c7;
      --accent-ink: #092116;
      --warn: #ffd18a;
      --bad: #ff938f;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      background:
        radial-gradient(circle at 12% 8%, rgba(53, 113, 84, .22), transparent 28rem),
        var(--bg);
      color: var(--ink);
      font: 14px/1.45 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main {
      width: min(1440px, 100%);
      margin: 0 auto;
      padding: 24px;
      display: grid;
      grid-template-columns: minmax(320px, 1.15fr) minmax(340px, .85fr);
      gap: 18px;
    }
    .panel {
      background: color-mix(in srgb, var(--panel) 94%, transparent);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 20px 60px rgba(0, 0, 0, .28);
      overflow: hidden;
    }
    .video-panel { min-height: calc(100vh - 48px); display: flex; flex-direction: column; }
    header, section { padding: 18px; border-bottom: 1px solid var(--line); }
    section:last-child { border-bottom: 0; }
    h1, h2, p { margin-top: 0; }
    h1 { margin-bottom: 4px; font-size: clamp(22px, 3vw, 34px); letter-spacing: -.04em; }
    h2 { margin-bottom: 12px; font-size: 13px; color: var(--muted); text-transform: uppercase; letter-spacing: .12em; }
    p { color: var(--muted); }
    .warning {
      margin: 12px 0 0;
      padding: 10px 12px;
      color: var(--warn);
      background: rgba(255, 209, 138, .08);
      border: 1px solid rgba(255, 209, 138, .24);
      border-radius: 10px;
    }
    .stage {
      position: relative;
      flex: 1;
      min-height: 420px;
      background: #020305;
      display: grid;
      place-items: center;
    }
    video { width: 100%; height: 100%; object-fit: contain; position: absolute; inset: 0; }
    .empty { color: #657384; text-align: center; padding: 28px; }
    .badges { display: flex; flex-wrap: wrap; gap: 8px; }
    .badge { padding: 6px 10px; border: 1px solid var(--line); border-radius: 999px; color: var(--muted); }
    .badge strong { color: var(--ink); font-weight: 650; }
    .controls { display: flex; flex-direction: column; min-height: calc(100vh - 48px); }
    .grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 8px; }
    .grid.three { grid-template-columns: repeat(3, minmax(0, 1fr)); }
    button, select, input {
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 10px;
      background: var(--panel-2);
      color: var(--ink);
      padding: 9px 11px;
      font: inherit;
    }
    button { cursor: pointer; font-weight: 650; transition: transform .12s, border-color .12s, background .12s; }
    button:hover:not(:disabled) { transform: translateY(-1px); border-color: #516277; }
    button:disabled { cursor: not-allowed; opacity: .45; }
    button.primary { background: var(--accent); border-color: var(--accent); color: var(--accent-ink); }
    button.danger { color: var(--bad); }
    label { display: grid; gap: 5px; color: var(--muted); font-size: 12px; }
    .row { display: flex; gap: 8px; align-items: end; }
    .row > * { flex: 1; }
    .pose-button.active { border-color: var(--accent); color: var(--accent); background: rgba(165, 243, 199, .08); }
    .log {
      min-height: 120px;
      max-height: 270px;
      overflow: auto;
      margin: 0;
      padding: 12px;
      background: #090d12;
      border: 1px solid var(--line);
      border-radius: 10px;
      color: #bdc8d4;
      font: 11px/1.55 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    @media (max-width: 900px) {
      main { grid-template-columns: 1fr; padding: 10px; }
      .video-panel, .controls { min-height: auto; }
      .stage { min-height: 58vh; }
    }
  </style>
</head>
<body>
<main>
  <article class="panel video-panel">
    <header>
      <h1>Pose protocol lab</h1>
      <p>Direct worker WebRTC, deterministic pose events, and local sample TTS.</p>
      <p class="warning">Test-only motion masters. Their generated clip boundaries are not production-approved or switch-safe yet.</p>
    </header>
    <div class="stage">
      <video id="remoteVideo" autoplay playsinline></video>
      <div class="empty" id="emptyState">Create a session to start the avatar stream.</div>
    </div>
    <section class="badges">
      <span class="badge">session <strong id="sessionState">none</strong></span>
      <span class="badge">peer <strong id="peerState">idle</strong></span>
      <span class="badge">idle pose <strong id="poseState">neutral_resting</strong></span>
      <span class="badge">rendered pose <strong id="renderedPoseState">none</strong></span>
      <span class="badge">queued <strong id="queueState">none</strong></span>
      <span class="badge">stream <strong id="streamState">idle</strong></span>
      <span class="badge">seq <strong id="seqState">0</strong></span>
    </section>
  </article>

  <aside class="panel controls">
    <section>
      <h2>Session</h2>
      <div class="row">
        <label>Render FPS<input id="fps" type="number" min="1" max="60" value="20"></label>
        <label>Batch size<input id="batchSize" type="number" min="1" max="32" value="4"></label>
      </div>
      <div class="grid" style="margin-top:10px">
        <button class="primary" id="connectButton">Create + connect</button>
        <button class="danger" id="disconnectButton" disabled>End session</button>
      </div>
    </section>

    <section>
      <h2>Idle pose queue · next boundary</h2>
      <div class="grid three" id="poseButtons"></div>
      <button id="cycleButton" style="width:100%;margin-top:8px">Queue all six poses</button>
      <p style="margin:8px 0 0">Idle controls pause during sample-TTS rendering; stream pose order comes from the conversation metadata.</p>
    </section>

    <section>
      <h2>Conversation events</h2>
      <div class="grid">
        <button data-event="user_speech_started">User starts speaking</button>
        <button data-event="user_speech_ended">User stops speaking</button>
        <button data-event="assistant_thinking">Assistant thinking</button>
        <button data-event="assistant_turn_aborted">Abort → neutral</button>
      </div>
      <div class="row" style="margin-top:8px">
        <label>Reaction intent
          <select id="reactionIntent">
            <option value="none">none</option>
            <option value="acknowledge">acknowledge · nod</option>
            <option value="warmth">warmth · smile</option>
            <option value="empathy">empathy · head tilt</option>
          </select>
        </label>
        <button id="reactionButton">Reaction ready</button>
      </div>
      <button id="demoButton" style="width:100%;margin-top:8px">Run listening → thinking → reaction demo</button>
    </section>

    <section>
      <h2>Sample TTS · no provider call</h2>
      <label>WAV, MP3, or MPGA<input id="audioFile" type="file" accept="audio/*,.wav,.mp3,.mpga"></label>
      <button class="primary" id="streamButton" disabled style="width:100%;margin-top:8px">Stream selected / bundled sample</button>
      <p style="margin:8px 0 0">If no file is selected, the lab fetches the server’s bundled sample WAV.</p>
    </section>

    <section style="flex:1">
      <h2>Status log</h2>
      <pre class="log" id="log"></pre>
    </section>
  </aside>
</main>

<script>
  "use strict";
  const POSE_SET = __POSE_SET__;
  const SAMPLE_AUDIO_URL = __SAMPLE_AUDIO_URL__;
  const API_ORIGIN = window.location.origin;
  const POSE_IDS = Object.keys(POSE_SET.poses);
  const REACTION_POSES = {
    none: null,
    acknowledge: "nod_agree",
    warmth: "light_smile",
    empathy: "empathetic_head_tilt",
  };

  const remoteVideo = document.getElementById("remoteVideo");
  const emptyState = document.getElementById("emptyState");
  const logElement = document.getElementById("log");
  const connectButton = document.getElementById("connectButton");
  const disconnectButton = document.getElementById("disconnectButton");
  const streamButton = document.getElementById("streamButton");
  let sessionId = null;
  let pc = null;
  let remoteStream = null;
  let pollTimer = null;
  let seq = 0;
  let turnId = null;
  let protocolChain = Promise.resolve();

  function setText(id, value) {
    document.getElementById(id).textContent = String(value);
  }

  function log(label, value) {
    const stamp = new Date().toLocaleTimeString();
    const detail = value === undefined ? "" : " " + JSON.stringify(value);
    logElement.textContent += `[${stamp}] ${label}${detail}\n`;
    logElement.scrollTop = logElement.scrollHeight;
  }

  async function request(path, options = {}) {
    const response = await fetch(API_ORIGIN + path, options);
    const text = await response.text();
    let body = text;
    try { body = text ? JSON.parse(text) : {}; } catch (_) {}
    if (!response.ok) {
      throw new Error(`${response.status} ${typeof body === "string" ? body : JSON.stringify(body)}`);
    }
    return body;
  }

  function nextSequence() {
    seq += 1;
    setText("seqState", seq);
    return seq;
  }

  function ensureTurn() {
    if (!turnId) {
      turnId = `pose_lab_${Date.now()}_1`;
      log("turn started", { turn_id: turnId });
    }
    return turnId;
  }

  function serializeProtocol(action) {
    const next = protocolChain.then(action);
    protocolChain = next.catch(() => {});
    return next;
  }

  function setActivePose(poseId) {
    setText("poseState", poseId);
    document.querySelectorAll(".pose-button").forEach((button) => {
      button.classList.toggle("active", button.dataset.pose === poseId);
    });
  }

  function setIdlePoseControlsDisabled(disabled) {
    document.querySelectorAll(".pose-button").forEach((button) => {
      button.disabled = disabled;
    });
    document.getElementById("cycleButton").disabled = disabled;
  }

  function applyPoseStatus(body) {
    const status = body && (body.pose_protocol || body.pose_status || body);
    if (!status) return;
    const pose = status.current_pose_id || status.active_pose_id || status.idle_pose_id;
    if (pose && POSE_IDS.includes(pose)) setActivePose(pose);
    setText("renderedPoseState", status.rendered_pose_id || "none");
    const queued = status.queued_pose_ids || status.pending_pose_ids || [];
    setText("queueState", queued.length ? queued.join(" → ") : "none");
  }

  async function sendIceCandidate(candidate) {
    if (!sessionId || !candidate) return;
    await request(`/webrtc/sessions/${sessionId}/ice`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        candidate: candidate.candidate,
        sdpMid: candidate.sdpMid,
        sdpMLineIndex: candidate.sdpMLineIndex,
      }),
    });
  }

  async function createAndConnect() {
    if (sessionId) return;
    connectButton.disabled = true;
    try {
      const fps = Math.max(1, Number(document.getElementById("fps").value) || 20);
      const batchSize = Math.max(1, Number(document.getElementById("batchSize").value) || 4);
      const params = new URLSearchParams({
        avatar_id: POSE_SET.poses.neutral_resting.avatar_id,
        user_id: `pose_lab_${Date.now()}`,
        fps: String(fps),
        playback_fps: String(fps),
        batch_size: String(batchSize),
        chunk_duration: "2",
        pose_switch_mode: "next_boundary",
        pose_set: JSON.stringify(POSE_SET),
      });
      log("creating session");
      const created = await request(`/webrtc/sessions/create?${params}`, { method: "POST" });
      sessionId = created.session_id;
      setText("sessionState", sessionId);

      pc = new RTCPeerConnection({
        iceServers: created.ice_servers || [],
        iceTransportPolicy: created.ice_transport_policy || "all",
      });
      remoteStream = new MediaStream();
      remoteVideo.srcObject = remoteStream;
      pc.ontrack = (event) => {
        if (event.track && !remoteStream.getTracks().includes(event.track)) {
          remoteStream.addTrack(event.track);
        }
        emptyState.hidden = true;
        remoteVideo.play().catch((error) => log("autoplay blocked; tap video", String(error)));
      };
      pc.onicecandidate = (event) => {
        if (event.candidate) sendIceCandidate(event.candidate).catch((error) => log("ICE send failed", String(error)));
      };
      pc.onconnectionstatechange = () => {
        setText("peerState", pc.connectionState);
        log("peer state", pc.connectionState);
      };
      pc.oniceconnectionstatechange = () => log("ICE state", pc.iceConnectionState);
      pc.addTransceiver("video", { direction: "recvonly" });
      pc.addTransceiver("audio", { direction: "recvonly" });

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);
      const answer = await request(`/webrtc/sessions/${sessionId}/offer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sdp: pc.localDescription.sdp, type: pc.localDescription.type }),
      });
      await pc.setRemoteDescription(answer);
      disconnectButton.disabled = false;
      streamButton.disabled = false;
      setActivePose("neutral_resting");
      startStatusPolling();
      log("session connected", created);
    } catch (error) {
      log("connect failed", String(error));
      await endSession();
    } finally {
      connectButton.disabled = Boolean(sessionId);
    }
  }

  async function queuePose(poseId, replacePending = true) {
    if (!sessionId) throw new Error("Create a session first.");
    const body = await request(`/webrtc/sessions/${sessionId}/pose`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        pose_id: poseId,
        effective: "next_boundary",
        replace_pending: replacePending,
      }),
    });
    applyPoseStatus(body);
    log("pose queued", body);
  }

  async function sendEvent(event, reactionIntent = null) {
    if (!sessionId) throw new Error("Create a session first.");
    if (event === "user_speech_started") {
      turnId = `pose_lab_${Date.now()}_${seq + 1}`;
    }
    const payload = {
      event,
      turn_id: ensureTurn(),
      seq: nextSequence(),
    };
    if (event === "assistant_reaction_ready") payload.reaction_intent = reactionIntent || "none";
    const body = await request(`/webrtc/sessions/${sessionId}/events`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    applyPoseStatus(body);
    log(`event ${event}`, body);
    if (event === "assistant_turn_aborted") turnId = null;
    return body;
  }

  async function resolveAudioFile() {
    const selected = document.getElementById("audioFile").files[0];
    if (selected) return selected;
    if (!SAMPLE_AUDIO_URL) throw new Error("Choose an audio file.");
    const response = await fetch(SAMPLE_AUDIO_URL);
    if (!response.ok) throw new Error(`Bundled sample unavailable (${response.status}); choose a local WAV.`);
    const blob = await response.blob();
    return new File([blob], "sample_tts.wav", { type: blob.type || "audio/wav" });
  }

  async function streamAudio() {
    if (!sessionId) return;
    streamButton.disabled = true;
    try {
      const audioFile = await resolveAudioFile();
      const intent = document.getElementById("reactionIntent").value;
      const reactionPose = REACTION_POSES[intent];
      const sequence = [...(reactionPose ? [reactionPose] : []), "speaking_direct", "neutral_resting"];
      const form = new FormData();
      form.append("audio_file", audioFile, audioFile.name);
      form.append("reaction_intent", intent);
      form.append("pose_id", "speaking_direct");
      form.append("pose_sequence", JSON.stringify(sequence));
      form.append("turn_id", ensureTurn());
      form.append("seq", String(nextSequence()));
      form.append("effective", "next_boundary");
      form.append("mouth_mode", "lip_sync");
      form.append("audio_start", "immediate");
      setText("streamState", "uploading");
      const body = await request(`/webrtc/sessions/${sessionId}/stream`, { method: "POST", body: form });
      setText("streamState", "processing");
      setIdlePoseControlsDisabled(true);
      applyPoseStatus(body);
      log("sample TTS accepted", body);
    } catch (error) {
      setText("streamState", "error");
      log("stream failed", String(error));
    } finally {
      streamButton.disabled = false;
    }
  }

  async function pollStatus() {
    if (!sessionId) return;
    try {
      const body = await request(`/webrtc/sessions/${sessionId}/status`);
      applyPoseStatus(body);
      setText("streamState", body.active_stream ? "active" : "idle");
      setIdlePoseControlsDisabled(Boolean(body.active_stream));
    } catch (error) {
      log("status failed", String(error));
    }
  }

  function startStatusPolling() {
    clearInterval(pollTimer);
    pollTimer = setInterval(pollStatus, 1000);
    pollStatus();
  }

  async function endSession() {
    clearInterval(pollTimer);
    pollTimer = null;
    const closingId = sessionId;
    sessionId = null;
    if (pc) {
      pc.close();
      pc = null;
    }
    remoteStream = null;
    remoteVideo.srcObject = null;
    emptyState.hidden = false;
    disconnectButton.disabled = true;
    streamButton.disabled = true;
    connectButton.disabled = false;
    setText("sessionState", "none");
    setText("peerState", "idle");
    setText("queueState", "none");
    setText("streamState", "idle");
    setText("renderedPoseState", "none");
    setIdlePoseControlsDisabled(false);
    if (closingId) {
      try {
        await request(`/webrtc/sessions/${closingId}`, { method: "DELETE" });
        log("session ended", closingId);
      } catch (error) {
        log("delete failed", String(error));
      }
    }
  }

  POSE_IDS.forEach((poseId) => {
    const button = document.createElement("button");
    button.className = "pose-button";
    button.dataset.pose = poseId;
    button.textContent = poseId.replaceAll("_", " ");
    button.addEventListener("click", () => queuePose(poseId).catch((error) => log("pose failed", String(error))));
    document.getElementById("poseButtons").appendChild(button);
  });
  document.querySelectorAll("[data-event]").forEach((button) => {
    button.addEventListener("click", () => {
      serializeProtocol(() => sendEvent(button.dataset.event))
        .catch((error) => log("event failed", String(error)));
    });
  });
  connectButton.addEventListener("click", createAndConnect);
  disconnectButton.addEventListener("click", endSession);
  document.getElementById("cycleButton").addEventListener("click", async () => {
    try {
      const cyclePoseIds = [
        ...POSE_IDS.filter((poseId) => poseId !== "neutral_resting"),
        "neutral_resting",
      ];
      for (let index = 0; index < cyclePoseIds.length; index += 1) {
        await queuePose(cyclePoseIds[index], index === 0);
      }
    } catch (error) {
      log("pose cycle failed", String(error));
    }
  });
  streamButton.addEventListener("click", () => {
    serializeProtocol(streamAudio).catch((error) => log("stream failed", String(error)));
  });
  remoteVideo.addEventListener("click", () => remoteVideo.play().catch(() => {}));
  document.getElementById("reactionButton").addEventListener("click", () => {
    serializeProtocol(() => sendEvent(
      "assistant_reaction_ready",
      document.getElementById("reactionIntent").value,
    ))
      .catch((error) => log("reaction failed", String(error)));
  });
  document.getElementById("demoButton").addEventListener("click", () => {
    serializeProtocol(async () => {
      await sendEvent("user_speech_started");
      await new Promise((resolve) => setTimeout(resolve, 700));
      await sendEvent("user_speech_ended");
      await sendEvent("assistant_thinking");
      await new Promise((resolve) => setTimeout(resolve, 700));
      await sendEvent("assistant_reaction_ready", document.getElementById("reactionIntent").value);
    }).catch((error) => log("demo failed", String(error)));
  });
  window.addEventListener("beforeunload", () => {
    if (sessionId) {
      fetch(`/webrtc/sessions/${sessionId}`, { method: "DELETE", keepalive: true }).catch(() => {});
    }
    if (pc) pc.close();
  });
  setActivePose("neutral_resting");
  log("lab ready", { pose_set_id: POSE_SET.pose_set_id, poses: POSE_IDS });
</script>
</body>
</html>"""
        .replace("__POSE_SET__", embedded_pose_set)
        .replace("__SAMPLE_AUDIO_URL__", embedded_sample_url)
    )
