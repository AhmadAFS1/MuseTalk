"""Standalone browser lab for MuseTalk pose-protocol WebRTC sessions.

The page intentionally has no framework or external assets.  It talks directly
to the worker so it can be used on a GPU host without the Lingua broker.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.pose_protocol import normalize_pose_set


POSE_IDS = (
    "neutral_resting",
    "active_listening",
    "speaking_direct",
    "nod_agree",
    "empathetic_head_tilt",
    "light_smile",
)

DEFAULT_MANIFEST = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "pose_test"
    / "sample_ai_human_ltx23_facetime_closeup_production_v1.json"
)
DEFAULT_POSE_SET = normalize_pose_set(
    json.loads(DEFAULT_MANIFEST.read_text())
)


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
    characters: list[dict] | None = None,
    selected_character: str | None = None,
    expected_motion: dict | None = None,
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

    embedded_characters = json.dumps(characters or [], separators=(",", ":")).replace("</", "<\\/")
    embedded_selected = json.dumps(selected_character).replace("</", "<\\/")
    embedded_motion = json.dumps(expected_motion, separators=(",", ":")).replace("</", "<\\/")

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
      <p class="warning" id="bankWarning"></p>
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
      <span class="badge">bank <strong id="bankState">unchecked</strong></span>
    </section>
  </article>

  <aside class="panel controls">
    <section>
      <h2>Session</h2>
      <label>Character<select id="characterSelect"></select></label>
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
      <label style="margin-top:8px">Speech movement
        <select id="speechMovement">
          <option value="talking">Talking</option>
          <option value="talking_smiling">Talking → smiling → talking</option>
        </select>
      </label>
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
  const CHARACTERS = __CHARACTERS__;
  const SELECTED_CHARACTER = __SELECTED_CHARACTER__;
  const EXPECTED_MOTION = __EXPECTED_MOTION__;
  const API_ORIGIN = window.location.origin;
  const POSE_IDS = Object.keys(POSE_SET.poses);
  const REACTION_POSES = {
    none: null, acknowledge: "nod_agree", warmth: "light_smile", empathy: "empathetic_head_tilt",
  };
  const remoteVideo = document.getElementById("remoteVideo");
  const emptyState = document.getElementById("emptyState");
  const logElement = document.getElementById("log");
  const connectButton = document.getElementById("connectButton");
  const disconnectButton = document.getElementById("disconnectButton");
  const streamButton = document.getElementById("streamButton");
  const characterSelect = document.getElementById("characterSelect");
  let sessionId = null;
  let sessionEpoch = 0;
  let pc = null;
  let pollTimer = null;
  let seq = 0;
  let turnId = null;
  let turnCounter = 0;
  let demoEpoch = 0;
  let protocolChain = Promise.resolve();
  let pendingUpload = null;
  let cancellingUpload = null;
  let activeStream = false;
  let motionReady = false;
  let motionSettled = true;
  let ready = false;
  let latestStatusRequest = 0;

  function setText(id, value) { document.getElementById(id).textContent = String(value); }
  function log(label, value) {
    const detail = value === undefined ? "" : " " + JSON.stringify(value);
    logElement.textContent += `[${new Date().toLocaleTimeString()}] ${label}${detail}\n`;
    logElement.scrollTop = logElement.scrollHeight;
  }
  async function request(path, options = {}) {
    const response = await fetch(API_ORIGIN + path, options);
    const text = await response.text();
    let body = text;
    try { body = text ? JSON.parse(text) : {}; } catch (_) {}
    if (!response.ok) throw new Error(`${response.status} ${typeof body === "string" ? body : JSON.stringify(body)}`);
    return body;
  }
  function context() { return { id: sessionId, epoch: sessionEpoch }; }
  function isCurrent(ctx) { return Boolean(ctx.id && ctx.id === sessionId && ctx.epoch === sessionEpoch); }
  function assertCurrent(ctx) {
    if (!isCurrent(ctx)) throw new Error("Session ended or changed; this action was discarded.");
  }
  function nextSequence() { seq += 1; setText("seqState", seq); return seq; }
  function newTurn(kind = "user") { return `pose_lab_${kind}_${Date.now()}_${++turnCounter}`; }
  function ensureTurn() {
    if (!turnId) { turnId = newTurn(); log("turn started", { turn_id: turnId }); }
    return turnId;
  }
  function serializeProtocol(action, ctx = context()) {
    const next = protocolChain.then(() => { assertCurrent(ctx); return action(ctx); });
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
    document.querySelectorAll(".pose-button").forEach((button) => { button.disabled = disabled; });
    document.getElementById("cycleButton").disabled = disabled;
  }
  function updateControls() {
    const busy = Boolean(pendingUpload || cancellingUpload || activeStream || !motionSettled);
    streamButton.disabled = !ready || !motionReady || busy;
    setIdlePoseControlsDisabled(!ready || busy);
    document.querySelectorAll("[data-event]").forEach((button) => { button.disabled = !ready; });
    document.getElementById("reactionButton").disabled = !ready;
    document.getElementById("demoButton").disabled = !ready;
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
  function validateMotionStatus(body) {
    const motion = body.track_stats && body.track_stats.video && body.track_stats.video.motion;
    if (EXPECTED_MOTION) {
      if (!motion || !motion.enabled || !motion.bank) throw new Error("Selected character has no enabled motion bank.");
      if (motion.bank.routing_sha256 !== EXPECTED_MOTION.routing_sha256) throw new Error("Selected character routing hash does not match this session.");
      for (const [pose, hash] of Object.entries(EXPECTED_MOTION.source_hashes)) {
        if (!motion.bank.sources || !motion.bank.sources[pose] || motion.bank.sources[pose].sha256 !== hash) {
          throw new Error(`Selected character source hash does not match: ${pose}.`);
        }
      }
    }
    return motion;
  }
  function applySessionStatus(body) {
    const motion = validateMotionStatus(body);
    motionReady = true;
    motionSettled = !motion || (motion.settled && !motion.building);
    activeStream = Boolean(body.active_stream);
    applyPoseStatus(body);
    setText("bankState", EXPECTED_MOTION ? "verified" : "legacy · unverified");
    setText("streamState", pendingUpload ? (pendingUpload.dispatched ? "preparing" : "loading audio") : activeStream ? "active" : !motionSettled ? "returning" : "idle");
    updateControls();
  }
  async function sendIceCandidate(ctx, candidate) {
    if (!isCurrent(ctx) || !candidate) return;
    await request(`/webrtc/sessions/${ctx.id}/ice`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ candidate: candidate.candidate, sdpMid: candidate.sdpMid, sdpMLineIndex: candidate.sdpMLineIndex }),
    });
  }
  async function createAndConnect() {
    if (sessionId || connectButton.disabled) return;
    const epoch = ++sessionEpoch;
    connectButton.disabled = true;
    disconnectButton.disabled = false;
    characterSelect.disabled = true;
    try {
      const fps = Math.max(1, Number(document.getElementById("fps").value) || 20);
      const batchSize = Math.max(1, Number(document.getElementById("batchSize").value) || 4);
      const params = new URLSearchParams({
        avatar_id: POSE_SET.poses.neutral_resting.avatar_id,
        user_id: `pose_lab_${Date.now()}`, fps: String(fps), playback_fps: String(fps),
        batch_size: String(batchSize), chunk_duration: "2", pose_switch_mode: "next_boundary", pose_set: JSON.stringify(POSE_SET),
      });
      log("creating session");
      const created = await request(`/webrtc/sessions/create?${params}`, { method: "POST" });
      if (epoch !== sessionEpoch) {
        await request(`/webrtc/sessions/${created.session_id}`, { method: "DELETE" });
        return;
      }
      sessionId = created.session_id;
      const ctx = context();
      setText("sessionState", sessionId);
      const status = await request(`/webrtc/sessions/${ctx.id}/status`);
      assertCurrent(ctx);
      applySessionStatus(status);
      const peer = new RTCPeerConnection({ iceServers: created.ice_servers || [], iceTransportPolicy: created.ice_transport_policy || "all" });
      pc = peer;
      const media = new MediaStream();
      remoteVideo.srcObject = media;
      peer.ontrack = (event) => {
        if (!isCurrent(ctx) || pc !== peer) return;
        if (event.track && !media.getTracks().includes(event.track)) media.addTrack(event.track);
        emptyState.hidden = true;
        remoteVideo.play().catch((error) => { if (isCurrent(ctx)) log("autoplay blocked; tap video", String(error)); });
      };
      peer.onicecandidate = (event) => {
        if (event.candidate) sendIceCandidate(ctx, event.candidate).catch((error) => { if (isCurrent(ctx)) log("ICE send failed", String(error)); });
      };
      peer.onconnectionstatechange = () => {
        if (!isCurrent(ctx) || pc !== peer) return;
        setText("peerState", peer.connectionState);
        ready = peer.connectionState === "connected";
        updateControls();
        log("peer state", peer.connectionState);
      };
      peer.oniceconnectionstatechange = () => { if (isCurrent(ctx)) log("ICE state", peer.iceConnectionState); };
      peer.addTransceiver("video", { direction: "recvonly" });
      peer.addTransceiver("audio", { direction: "recvonly" });
      const offer = await peer.createOffer();
      assertCurrent(ctx);
      await peer.setLocalDescription(offer);
      assertCurrent(ctx);
      const answer = await request(`/webrtc/sessions/${ctx.id}/offer`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sdp: peer.localDescription.sdp, type: peer.localDescription.type }),
      });
      assertCurrent(ctx);
      await peer.setRemoteDescription(answer);
      assertCurrent(ctx);
      ready = peer.connectionState === "connected";
      updateControls();
      startStatusPolling();
      log("session negotiated", created);
    } catch (error) {
      if (epoch === sessionEpoch) { log("connect failed", String(error)); await endSession(); }
    } finally {
      if (epoch === sessionEpoch) connectButton.disabled = Boolean(sessionId);
    }
  }
  async function queuePose(poseId, replacePending = true, ctx = context()) {
    assertCurrent(ctx);
    const body = await request(`/webrtc/sessions/${ctx.id}/pose`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pose_id: poseId, effective: "next_boundary", replace_pending: replacePending }),
    });
    if (!isCurrent(ctx)) return;
    applyPoseStatus(body);
    log("pose queued", body);
  }
  function cancelPendingUpload() {
    const upload = pendingUpload || cancellingUpload;
    if (upload) {
      upload.cancelled = true;
      upload.controller.abort();
      cancellingUpload = upload;
      updateControls();
    }
    return upload;
  }
  async function sendEvent(event, reactionIntent = null, ctx = context()) {
    assertCurrent(ctx);
    if (event === "assistant_turn_aborted") ++demoEpoch;
    const abortingUpload = event === "assistant_turn_aborted" ? cancelPendingUpload() : null;
    try {
      if (abortingUpload && !abortingUpload.turnId) {
        log("local audio load cancelled");
        return { accepted: false, reason: "audio_load_cancelled" };
      }
      if (event === "user_speech_started") turnId = newTurn();
      let eventTurnId = ensureTurn();
      if (event === "assistant_turn_aborted") {
        const current = await request(`/webrtc/sessions/${ctx.id}/status`);
        assertCurrent(ctx);
        const protocol = current.pose_protocol || {};
        const activeTurnId = protocol.active_turn_id;
        if (current.active_stream && !activeTurnId) throw new Error("Active reply has no turn ID; refresh status before aborting.");
        if (abortingUpload && abortingUpload.turnId && activeTurnId !== abortingUpload.turnId) {
          // A newer user event can supersede an upload before it is reserved.
          // Its higher sequence already prevents that delayed upload from
          // starting. Preserve the new user's ID and never abort its reply.
          if (!abortingUpload.dispatched || protocol.last_seq >= abortingUpload.seq) {
            log("pending reply already superseded", { turn_id: abortingUpload.turnId });
            return { accepted: false, reason: "pending_reply_already_superseded" };
          }
          throw new Error("Pending reply ownership is unknown; end the session to stop it.");
        }
        eventTurnId = activeTurnId || eventTurnId;
      }
      const payload = { event, turn_id: eventTurnId, seq: nextSequence() };
      if (event === "assistant_reaction_ready") payload.reaction_intent = reactionIntent || "none";
      const body = await request(`/webrtc/sessions/${ctx.id}/events`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload),
      });
      if (!isCurrent(ctx)) return body;
      applyPoseStatus(body);
      log(`event ${event}`, body);
      if (event === "assistant_turn_aborted" && body.accepted && turnId === eventTurnId) turnId = null;
      return body;
    } finally {
      if (isCurrent(ctx) && abortingUpload && cancellingUpload === abortingUpload) {
        cancellingUpload = null;
        updateControls();
        await pollStatus(ctx);
      }
    }
  }
  async function resolveAudioFile(signal) {
    const selected = document.getElementById("audioFile").files[0];
    if (selected) return selected;
    if (!SAMPLE_AUDIO_URL) throw new Error("Choose an audio file.");
    const response = await fetch(SAMPLE_AUDIO_URL, { signal });
    if (!response.ok) throw new Error(`Bundled sample unavailable (${response.status}); choose a local WAV.`);
    const blob = await response.blob();
    return new File([blob], "sample_tts.wav", { type: blob.type || "audio/wav" });
  }
  async function streamAudio() {
    const ctx = context();
    if (!isCurrent(ctx) || !ready || !motionReady || pendingUpload || cancellingUpload || activeStream || !motionSettled) return;
    const upload = { cancelled: false, dispatched: false, completed: false, turnId: null, controller: new AbortController() };
    pendingUpload = upload;
    updateControls();
    let responsePromise = null;
    try {
      const audioFile = await resolveAudioFile(upload.controller.signal);
      assertCurrent(ctx);
      if (upload.cancelled) return;
      const intent = document.getElementById("reactionIntent").value;
      const movement = document.getElementById("speechMovement").value;
      await serializeProtocol(async () => {
        if (upload.cancelled) return;
        upload.turnId = newTurn("reply");
        turnId = upload.turnId;
        // Establish the pending reply identity before uploading any audio.
        // A later abort can target it even when multipart upload is stalled;
        // a later user event fences its lower stream sequence instead.
        const preflight = await sendEvent("assistant_thinking", null, ctx);
        assertCurrent(ctx);
        if (!preflight.accepted) throw new Error("Reply preflight was rejected; refresh session status.");
        if (upload.cancelled) return;
        const form = new FormData();
        form.append("audio_file", audioFile, audioFile.name);
        form.append("reaction_intent", intent);
        form.append("pose_id", "speaking_direct");
        if (movement === "talking_smiling") {
          form.append("pose_plan", JSON.stringify({
            version: 2, clock: "audio_progress", switch_mode: "next_boundary", on_complete: "neutral_resting",
            segments: [{ at_permille: 0, pose_id: "speaking_direct" }, { at_permille: 350, pose_id: "light_smile" }, { at_permille: 700, pose_id: "speaking_direct" }],
          }));
        } else {
          const reactionPose = REACTION_POSES[intent];
          form.append("pose_sequence", JSON.stringify([...(reactionPose ? [reactionPose] : []), "speaking_direct", "neutral_resting"]));
        }
        form.append("turn_id", upload.turnId);
        upload.seq = nextSequence();
        form.append("seq", String(upload.seq));
        form.append("effective", "next_boundary");
        form.append("mouth_mode", "lip_sync");
        form.append("audio_start", "immediate");
        upload.dispatched = true;
        setText("streamState", "uploading / preparing");
        // Release the ordered event queue immediately after dispatch. Await
        // the response outside it so upload / preparation cannot block abort.
        responsePromise = request(`/webrtc/sessions/${ctx.id}/stream`, { method: "POST", body: form, signal: upload.controller.signal }).then(
          (body) => { upload.completed = true; return { body }; },
          (error) => { upload.completed = true; return { error }; },
        );
      }, ctx);
      if (!responsePromise) return;
      const result = await responsePromise;
      assertCurrent(ctx);
      if (result.error) throw result.error;
      applyPoseStatus(result.body);
      log("sample TTS response", result.body);
    } catch (error) {
      if (isCurrent(ctx)) { setText("streamState", "error"); log("stream failed", String(error)); }
    } finally {
      if (isCurrent(ctx) && pendingUpload === upload) {
        pendingUpload = null;
        // Status decides whether speech / recovery still owns the session.
        // Keep the button blocked until that response arrives.
        if (upload.dispatched) activeStream = true;
        updateControls();
        await pollStatus(ctx);
      }
    }
  }
  async function pollStatus(ctx = context()) {
    if (!isCurrent(ctx)) return;
    const statusRequest = ++latestStatusRequest;
    try {
      const body = await request(`/webrtc/sessions/${ctx.id}/status`);
      if (!isCurrent(ctx) || statusRequest !== latestStatusRequest) return;
      try { applySessionStatus(body); }
      catch (error) { log("motion bank verification failed", String(error)); await endSession(); }
    } catch (error) { if (isCurrent(ctx)) log("status failed", String(error)); }
  }
  function startStatusPolling() {
    clearInterval(pollTimer);
    const ctx = context();
    pollTimer = setInterval(() => pollStatus(ctx), 1000);
    pollStatus(ctx);
  }
  async function endSession() {
    const closingId = sessionId;
    ++sessionEpoch;
    ++demoEpoch;
    clearInterval(pollTimer);
    pollTimer = null;
    sessionId = null;
    const peer = pc;
    pc = null;
    if (peer) peer.close();
    if (pendingUpload) pendingUpload.controller.abort();
    if (cancellingUpload) cancellingUpload.controller.abort();
    pendingUpload = null;
    cancellingUpload = null;
    activeStream = false;
    motionReady = false;
    ready = false;
    motionSettled = true;
    turnId = null;
    seq = 0;
    protocolChain = Promise.resolve();
    remoteVideo.srcObject = null;
    emptyState.hidden = false;
    disconnectButton.disabled = true;
    connectButton.disabled = false;
    characterSelect.disabled = !CHARACTERS.length;
    for (const [id, value] of Object.entries({ sessionState: "none", peerState: "idle", queueState: "none", streamState: "idle", renderedPoseState: "none", seqState: 0, bankState: "unchecked" })) setText(id, value);
    updateControls();
    if (closingId) {
      try { await request(`/webrtc/sessions/${closingId}`, { method: "DELETE" }); log("session ended", closingId); }
      catch (error) { log("delete failed", String(error)); }
    }
  }

  for (const character of CHARACTERS) {
    const option = document.createElement("option");
    option.value = character.id;
    option.textContent = character.label;
    characterSelect.appendChild(option);
  }
  if (CHARACTERS.length) characterSelect.value = SELECTED_CHARACTER;
  else {
    const option = document.createElement("option");
    option.textContent = "Legacy six-pose sample";
    characterSelect.appendChild(option);
    characterSelect.disabled = true;
  }
  characterSelect.addEventListener("change", () => {
    if (sessionId || connectButton.disabled) return;
    window.location.assign(`/webrtc/pose-lab?character=${encodeURIComponent(characterSelect.value)}`);
  });
  setText("bankWarning", EXPECTED_MOTION
    ? "This character’s prepared motion bank is verified before streaming. Visual approval is still required."
    : "Legacy sample: no selected motion-bank verification. Configure the character catalog to test prepared three-pose avatars.");
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
      // Cancel bytes immediately; the ordered abort still reaches the server
      // to fence late upload completion or cancel an already-owned stream.
      if (button.dataset.event === "assistant_turn_aborted") { ++demoEpoch; cancelPendingUpload(); }
      serializeProtocol((ctx) => sendEvent(button.dataset.event, null, ctx)).catch((error) => log("event failed", String(error)));
    });
  });
  connectButton.addEventListener("click", createAndConnect);
  disconnectButton.addEventListener("click", endSession);
  document.getElementById("cycleButton").addEventListener("click", async () => {
    const ctx = context();
    try {
      const cyclePoseIds = [...POSE_IDS.filter((poseId) => poseId !== "neutral_resting"), "neutral_resting"];
      for (let index = 0; index < cyclePoseIds.length; index += 1) await queuePose(cyclePoseIds[index], index === 0, ctx);
    } catch (error) { if (isCurrent(ctx)) log("pose cycle failed", String(error)); }
  });
  streamButton.addEventListener("click", streamAudio);
  remoteVideo.addEventListener("click", () => remoteVideo.play().catch(() => {}));
  document.getElementById("reactionButton").addEventListener("click", () => {
    const intent = document.getElementById("reactionIntent").value;
    serializeProtocol((ctx) => sendEvent("assistant_reaction_ready", intent, ctx)).catch((error) => log("reaction failed", String(error)));
  });
  document.getElementById("demoButton").addEventListener("click", async () => {
    const ctx = context();
    const epoch = ++demoEpoch;
    try {
      await serializeProtocol(() => sendEvent("user_speech_started", null, ctx), ctx);
      const demoTurn = turnId;
      const step = (event, intent = null) => serializeProtocol(() => {
        if (epoch !== demoEpoch || turnId !== demoTurn) throw new Error("Demo stopped because its turn ended or changed.");
        return sendEvent(event, intent, ctx);
      }, ctx);
      await new Promise((resolve) => setTimeout(resolve, 700));
      await step("user_speech_ended");
      await step("assistant_thinking");
      await new Promise((resolve) => setTimeout(resolve, 700));
      await step("assistant_reaction_ready", document.getElementById("reactionIntent").value);
    } catch (error) { if (isCurrent(ctx)) log("demo failed", String(error)); }
  });
  window.addEventListener("beforeunload", () => {
    if (sessionId) fetch(`/webrtc/sessions/${sessionId}`, { method: "DELETE", keepalive: true }).catch(() => {});
    if (pc) pc.close();
  });
  setActivePose("neutral_resting");
  updateControls();
  log("lab ready", { character: SELECTED_CHARACTER, pose_set_id: POSE_SET.pose_set_id, poses: POSE_IDS });
</script>
</body>
</html>"""
        .replace("__POSE_SET__", embedded_pose_set)
        .replace("__SAMPLE_AUDIO_URL__", embedded_sample_url)
        .replace("__CHARACTERS__", embedded_characters)
        .replace("__SELECTED_CHARACTER__", embedded_selected)
        .replace("__EXPECTED_MOTION__", embedded_motion)
    )
