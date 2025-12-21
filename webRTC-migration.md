# Plan

Add WebRTC support alongside existing SSE streaming by introducing a parallel set of APIs and server-side WebRTC session management, without changing current SSE endpoints. The approach keeps both stacks operational, allowing gradual client migration and direct comparison of latency, quality, and reliability.

## Requirements
- Keep all existing SSE session APIs working unchanged.
- Introduce new WebRTC-specific APIs (new paths) to avoid breaking current clients.
- Support per-user sessions with isolated media streams (similar to current session model).
- Provide a WebRTC player path for browser/WebView clients.
- Define signaling, ICE, and TURN configuration for real-world connectivity.
- Ensure compatibility across iOS Safari/WebView, Android Chrome/WebView, and desktop browsers.
- Stream live frames as they are generated, then fall back to a default loop when idle (same behavior as session player).

## Scope
- In: WebRTC signaling endpoints, session lifecycle management for WebRTC, server media pipeline, player UI, idle/default video fallback behavior, configuration for STUN/TURN, docs updates.
- Out: Replacing SSE APIs, removing chunk-based storage, large refactors of existing inference pipeline.

## Files and entry points
- api_server.py (new WebRTC endpoints, feature flags/config)
- scripts/session_manager.py (extend or add parallel WebRTC session manager)
- templates/session_player.py or new templates/webrtc_player.py (WebRTC client player)
- New module(s): scripts/webrtc_manager.py, scripts/webrtc_tracks.py (server-side WebRTC integration)
- newcode_readme.md / docs (add WebRTC API reference and migration notes)
- assets/ or templates/ (idle/default video asset and configuration)

## Data model / API changes
- Add new WebRTC-specific session routes, e.g.:
  - POST /webrtc/sessions/create
  - POST /webrtc/sessions/{session_id}/offer (SDP offer -> answer)
  - POST /webrtc/sessions/{session_id}/ice (ICE candidate exchange)
  - GET  /webrtc/player/{session_id}
- WebRTC session routes should mirror existing `/sessions` payloads and defaults to keep client integration consistent.
- Session metadata should include WebRTC state (peer connection, ICE status, codec selection).
- Session configuration should include an idle/default video reference for fallback playback.
- Configuration inputs for STUN/TURN (env or config file) without affecting SSE flows.

## Behavior contract (WebRTC player)
- Idle: when no active stream, player shows a looped default video (or idle frames) continuously.
- Live start: when a stream begins, the player switches to live WebRTC media without a full reload.
- Live playback: frames are rendered as they arrive; audio and video stay in sync.
- Live end: when the stream finishes or errors, the player returns to the idle loop within a short grace period.
- Reconnect: if the peer connection drops, the player attempts a limited reconnect; on failure, it falls back to idle.
- Autoplay policy: same as current MSE player (click-to-play required), use playsinline for iOS.

## Concrete change notes (by class/module)
- api_server.py: add `/webrtc/...` endpoints for create/offer/ice/player; wire to a WebRTC session manager; keep existing SSE endpoints unchanged.
- scripts/session_manager.py (SessionManager, UserSession): add a parallel WebRTC session structure (or a subclass) to store peer connection state, ICE status, and track handles; replace chunk_queue with WebRTC-specific fields for live media delivery.
- scripts/avatar_manager_parallel.py (ParallelAvatarManager): add a streaming hook that yields raw frames/audio buffers instead of writing chunk files; expose an API that WebRTC tracks can pull from or be pushed to.
- scripts/api_avatar.py (APIAvatar): add a WebRTC streaming mode that returns frames/audio in-memory with timestamps; keep current chunk file streaming for SSE.
- templates/session_player.py or new templates/webrtc_player.py: implement WebRTC client logic (RTCPeerConnection, SDP/ICE exchange), attach remote stream, and manage idle fallback behavior.
- scripts/concurrent_gpu_manager.py: likely no functional change, but consider adding per-session telemetry or constraints if real-time encoding increases GPU pressure.
- scripts/avatar_cache.py: likely no functional change, but monitor cache hit/miss impact when streaming continuously.

## Action items
[ ] Use `aiortc` for the WebRTC stack (decision made) and document signaling details and tradeoffs.
[ ] Define the new WebRTC API surface and response shapes (separate from SSE), including auth/headers if needed.
[ ] Implement a WebRTC session manager that mirrors SSE session lifecycle but stores peer connections and media tracks.
[ ] Build server media tracks that push generated frames and audio buffers into WebRTC (with timestamps for A/V sync).
[ ] Add a WebRTC player template/page that negotiates the connection and plays the stream.
[ ] Implement idle/default playback behavior when no live stream is active (looped video or idle frames).
[ ] Set codec preferences for cross-device support (H264 baseline for iOS; Opus for audio where supported).
[ ] Handle mobile autoplay restrictions (playsinline, user gesture gating, muted-start fallback).
[ ] Wire STUN/TURN config and document required deployment steps for NAT traversal.
[ ] Document WebRTC dependencies here (no `requirements.txt` changes yet): `aiortc`, `av`, `aioice`.
[ ] Update docs with a parallel WebRTC workflow and a migration guide that compares SSE vs WebRTC.
[ ] Add monitoring/logging for per-session latency, ICE failures, and resource usage.

## Testing and validation
- Manual browser test: create WebRTC session, connect, stream audio, verify A/V sync.
- Regression check: existing SSE session flow still works end-to-end.
- ICE/NAT test: verify connectivity with and without TURN.
- Load test: concurrent WebRTC sessions to observe CPU/GPU encode and memory usage.
- Device matrix: iPhone/iPad Safari, Android Chrome/WebView, macOS Safari/Chrome.

## Risks and edge cases
- WebRTC connectivity failures without TURN; mobile carrier NATs may require TURN.
- Increased CPU/GPU usage for real-time encoding; may reduce concurrency.
- A/V sync drift if timestamps are incorrect or audio buffering differs from SSE.
- Client compatibility across iOS WebView vs desktop browsers.
- Autoplay and audio policy constraints on iOS can block playback without user interaction.

## Open questions
- Do we need authentication/authorization for signaling endpoints?
- What should be the default idle video source (static mp4, looped avatar, or configurable per avatar)?

## Stack decision
- Use `aiortc` for the initial WebRTC implementation to keep integration seamless with FastAPI and the existing Python inference pipeline.
- Keep SSE as a parallel path for migration and fallback.
- Default to H264 baseline video and Opus audio for widest device compatibility, especially iOS.
- Plan to revisit a dedicated media server (Janus/mediasoup) only if concurrency or routing needs exceed aiortc capacity.

## aiortc dependencies and server config (plain language)
- aiortc: the Python library that provides WebRTC server support inside FastAPI.
- TLS/HTTPS: WebRTC requires secure origins in browsers, so the API must be served over HTTPS in production.
- STUN server: helps clients discover their public network address; required for most networks.
- TURN server: relays traffic when direct peer-to-peer fails (common on mobile networks); needed for reliable iOS/Android connectivity.
- Codec support: enable H264 (video) and Opus (audio); these are the most compatible across devices.

## Implementation spec (MVP decisions)
- WebRTC session APIs use the same payloads and defaults as current `/sessions` endpoints.
- Client behavior matches the MSE player, including click-to-play for autoplay compliance.
- Encoding settings: fps = 10, H264 software encoding, bitrate set to a stable default (tune after initial tests).
- Audio sample rate: use source/default settings (no resampling unless required).

## Infrastructure config (proposal)
- Development: use a public STUN server (example: `stun:stun.l.google.com:19302`) to validate connectivity.
- Production: deploy a TURN server (coturn) and require it as a fallback for mobile networks.
- TLS: terminate HTTPS at the API or a reverse proxy so WebRTC negotiation works on iOS Safari and WebViews.
- Config surface: `WEBRTC_STUN_URLS`, `WEBRTC_TURN_URLS`, `WEBRTC_TURN_USER`, `WEBRTC_TURN_PASS`.

## Hosting / port mapping notes (Vast.ai example)
- WebRTC media uses random UDP ports, not the HTTP port.
- Keep TCP open for API/SSH (ex: TCP 8000 mapped to a public port like `36961`).
- Open a UDP port range (ex: `40000-40100`) and restrict the OS ephemeral range to match:
  - `sudo sysctl -w net.ipv4.ip_local_port_range="40000 40100"`
- Use the mapped TCP port for all HTTP requests and player URLs:
  - `POST http://<public-ip>:<tcp-port>/webrtc/sessions/create?...`
  - `http://<public-ip>:<tcp-port>/webrtc/player/{session_id}`
- UDP ports are not part of the URL; they are used by ICE behind the scenes.

## Local vs production 
- Local dev: running `api_server.py` is usually enough. WebRTC can work without TURN on the same network or when NAT is permissive.
- Production/mobile users: TURN is often required for reliable connections on iOS/Android cellular networks.
- HTTPS is required by browsers for WebRTC in production, so plan for TLS even if local dev works over HTTP.

## Media pipeline definition (proposal)
- Single PeerConnection with one MediaStream containing two tracks (video + audio). This keeps audio and video together for the client while preserving WebRTC's track model.
- Idle mode: a looped default video is served continuously to the WebRTC video track.
- Live mode: when audio is uploaded to `/webrtc/sessions/{session_id}/stream`, inference emits raw frames (and audio if available) directly into WebRTC tracks.
- Track design: implement `VideoStreamTrack` backed by an async queue (pull frames), and optionally an `AudioStreamTrack` backed by audio buffers.
- Timing: timestamp frames using `fps` pacing; drop or skip frames if the queue backs up to keep latency low.
- Transition: on live start, switch the video track source from idle to live; on live end, return to idle after a short grace period.
