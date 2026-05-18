# Current WebRTC Implementation And Retest Plan

Date: 2026-05-17

This file is the fresh chat-context note for restarting WebRTC work. It is not
the old migration doc and should be treated as the current retest plan after
the HLS/TRT-stagewise changes.

## Docs Reviewed

Relevant Markdown found from a repo-wide `.md` scan:

- `webRTC-migration.md`
- `WEBRTC_REACT_NATIVE_README.md`
- `TURN_API_README.md`
- `turn-setup.md`
- `newcode_readme.md`
- `docs/hls_migration.md`
- `HLS_REACT_NATIVE_README.md`
- `runbook_hls_load_test.md`
- `archive_hls_throughput_experiment_history.md`
- `current_webrtc_playback_smoothing_findings.md`
- Vast/startup docs that explain the current TRT-stagewise server path

The old WebRTC docs were not just speculative. The repo still has a real
WebRTC path: aiortc endpoints, a session manager, switchable media tracks, a
browser/WebView player, TURN setup notes, and an HTTP scratch file.

## Current Local Findings

The current server is already running on the TRT-stagewise venv:

- Process: `/workspace/.venvs/musetalk_trt_stagewise/bin/python api_server.py --host 0.0.0.0 --port 8000`
- Health: `http://127.0.0.1:8000/health` returns healthy.
- Public worker base URL from stats: `http://84.50.156.125:16359`
- Cached avatar: `avatar_22ed8902-f691-4042-a4e8-69affacc6415_1778448036`
- Sample audio files exist under `data/audio/`.

Installed WebRTC runtime in the active venv:

- `aiortc 1.14.0`
- `aioice 0.10.1`
- `av 16.1.0`
- `fastapi 0.135.1`
- `uvicorn 0.42.0`

Important shell detail: plain `python` is not available in this shell. Use the
venv Python or activate the venv:

```bash
source /workspace/.venvs/musetalk_trt_stagewise/bin/activate
```

I also verified that `/webrtc/sessions/create` currently works locally. The
probe response returned only public STUN:

```json
"ice_servers": [{"urls": ["stun:stun.l.google.com:19302"]}]
```

That means TURN is not active in the currently running API process.

## Current WebRTC Code Path

The active WebRTC files are:

- `api_server.py`
  - `POST /webrtc/sessions/create`
  - `POST /webrtc/sessions/{session_id}/offer`
  - `POST /webrtc/sessions/{session_id}/ice`
  - `POST /webrtc/sessions/{session_id}/stream`
  - `GET /webrtc/player/{session_id}`
  - `DELETE /webrtc/sessions/{session_id}`
- `scripts/webrtc_manager.py`
  - owns `RTCPeerConnection`, session TTL, ICE config, idle video track, silence audio track
- `scripts/webrtc_tracks.py`
  - owns idle video, switchable live video, silence audio, synced live audio, A/V sync clock
- `templates/webrtc_player.py`
  - browser/WebView player, SDP exchange, ICE candidate POSTs, tap-to-enable-audio, debug overlay
- `http_scripts_webrtc.http`
  - already points at the current public base URL and cached avatar

Current dirty changes are relevant and should not be reverted casually:

- `api_server.py`: WebRTC default `batch_size` changed from `2` to `8`; active request cleanup was added.
- `scripts/api_avatar.py`: tail batches are padded for the active VAE/backend path.
- `http_scripts_webrtc.http`: updated public URL/avatar and batch size.

## What Changed Because HLS Became Main Path

The current boot and performance docs are centered on HLS and the TRT-stagewise
server path:

- `scripts/vast_onstart.sh` and `scripts/vast_server_ctl.sh` are now the normal
  Vast startup path.
- `scripts/run_trt_stagewise_server.sh` is the foreground/debug launcher.
- HLS has a shared GPU scheduler (`scripts/hls_gpu_scheduler.py`), but WebRTC
  does not currently use that scheduler.
- WebRTC streaming still goes through `manager.executor` plus
  `manager.gpu_memory.allocate(session.batch_size)`.
- Current HLS throughput tuning does not automatically make WebRTC concurrent
  streaming safe.
- The old `scripts/run_api_server.sh` has WebRTC/TURN env defaults, but it is
  stale compared with the current TRT-stagewise launch path.

The current setup scripts do install WebRTC server deps (`aiortc`, `aioice`,
`av`), so dependency installation is not the main blocker. Network exposure and
retesting the old media path against the newer model backend are the blockers.

## Current Network/Port State

Current host state:

- API TCP `8000` is listening.
- No `turnserver`/`coturn` process is listening.
- `turnserver` was not found in PATH from this shell.
- Current Linux ephemeral UDP range:

```text
net.ipv4.ip_local_port_range = 32768 60999
```

That means the old direct-WebRTC assumption of `40000-40100` is not active.

aiortc/aioice binds host ICE UDP sockets with `local_addr=(address, 0)`, so it
uses the OS ephemeral port range. There is no app-level port range knob in the
installed `aioice` path. If we want direct host ICE to use `40000-40100`, we
need to set the OS range before starting the API:

```bash
sudo sysctl -w net.ipv4.ip_local_port_range="40000 40100"
```

The cloud/Vast firewall or port mapping also has to expose UDP `40000-40100`.
The HTTP URL still uses the mapped API TCP port, e.g.
`http://84.50.156.125:16359`; the UDP ports are used behind the scenes by ICE.

## TURN State

The old TURN docs and config are stale for the current host:

- `turnserver.conf` still references `96.28.173.248`.
- `scripts/run_api_server.sh` defaults to old TURN URLs at `195.142.145.66`.
- `turnserver.conf` still has `CHANGE_THIS_PASSWORD`.
- The running API session response exposes STUN only, so the process was not
  started with `WEBRTC_TURN_URLS`.

For same-machine/local testing, TURN is not required. For public browser,
iPhone, Android, or carrier-network tests, TURN should be treated as required
unless direct UDP mapping is definitely open and verified.

This chat added a new TCP relay setup:

- `scripts/run_turnserver_tcp_relay.sh`
- `scripts/run_webrtc_relay_api_server.sh`
- local ignored env file: `.env.webrtc-turn.local`

The intended no-public-UDP-range setup is:

- expose one TURN listener port publicly
- set `WEBRTC_ICE_TRANSPORT_POLICY=relay`
- let the server connect to coturn locally through `WEBRTC_SERVER_TURN_URLS`
- let browsers/mobile clients connect through public `WEBRTC_TURN_URLS`

Important: coturn/aiortc still allocate internal relay endpoints, and aiortc
reports relay candidates as UDP relay candidates. The difference is that both
peers are forced through TURN, so the old public UDP relay range does not need
to be opened for this test mode. On the current Vast instance, internal
`3478/udp` maps to public `15979/udp`, so the browser/client TURN URL is:

```text
turn:84.50.156.125:15979?transport=udp
```

## Immediate Retest Path

### 1) Fast local sanity test

This confirms the API and WebRTC session manager are alive:

```bash
curl -X POST \
  "http://127.0.0.1:8000/webrtc/sessions/create?avatar_id=avatar_22ed8902-f691-4042-a4e8-69affacc6415_1778448036&user_id=manual_test&fps=10&playback_fps=10&batch_size=8&chunk_duration=2"
```

Then open:

```text
http://127.0.0.1:8000/webrtc/player/<session_id>
```

Tap the overlay to start negotiation/audio.

### 2) Public browser test without TURN

Only do this after restoring direct UDP exposure:

```bash
sudo sysctl -w net.ipv4.ip_local_port_range="40000 40100"
```

Vast/cloud mapping must include:

- TCP public port to internal `8000` for the API/player
- UDP `40000-40100` to the container/host

Start the current server path with WebRTC-oriented env:

```bash
cd /workspace/MuseTalk
bash scripts/vast_server_ctl.sh stop

WEBRTC_STUN_URLS="stun:stun.l.google.com:19302" \
WEBRTC_TURN_URLS="" \
WEBRTC_VIDEO_PREBUFFER_SECONDS=0 \
WEBRTC_AUDIO_PREBUFFER_SECONDS=0.15 \
PROFILE=baseline \
PORT=8000 \
bash scripts/vast_server_ctl.sh start
```

Create a session and open:

```text
http://84.50.156.125:16359/webrtc/player/<session_id>
```

### 3) Public/mobile test with TURN

This is the realistic mobile path. Either use managed TURN or start coturn on a
stable host. Coturn is installed on the current node.

```bash
cd /workspace/MuseTalk
setsid nohup scripts/run_turnserver_tcp_relay.sh \
  > /workspace/logs/musetalk/turnserver_tcp_relay.log 2>&1 < /dev/null &
```

The current Vast mapping for internal `3478/udp` is public `15979/udp`, so
`.env.webrtc-turn.local` should contain:

```text
TURN_PUBLIC_IP=<PUBLIC_IP>
TURN_PUBLIC_PORT=15979
TURN_PUBLIC_TRANSPORT=udp
```

Restart the API through the relay launcher:

```bash
cd /workspace/MuseTalk
bash scripts/vast_server_ctl.sh stop

setsid nohup scripts/run_webrtc_relay_api_server.sh \
  --profile baseline \
  --host 0.0.0.0 \
  --port 8000 \
  --venv-path /workspace/.venvs/musetalk_trt_stagewise \
  --repo-root /workspace/MuseTalk \
  > /workspace/logs/musetalk/api_server_8000.log 2>&1 < /dev/null &
```

For iOS/React Native WebView, use HTTPS as soon as possible. HTTP can work for
some desktop/local cases, but HTTPS is the safer requirement for real devices.

## Full Manual Test Flow

1. Start API with the env above.
2. Create WebRTC session.
3. Open `/webrtc/player/{session_id}` and tap to start.
4. Confirm the debug overlay reaches `connected` and shows video stats.
5. Upload audio:

```bash
curl -X POST \
  -F "audio_file=@data/audio/ai-test-avatar.mpga;type=audio/mpeg" \
  "http://127.0.0.1:8000/webrtc/sessions/<session_id>/stream"
```

6. Watch logs:

```bash
tail -f /workspace/logs/musetalk/api_server_8000.log
```

7. Watch state:

```bash
curl http://127.0.0.1:8000/stats
curl http://127.0.0.1:8000/live/sessions
```

## Risks To Check Before Trusting Results

1. Video prebuffer/cleanup race.
   `SwitchableVideoStreamTrack` defaults to a 2 second prebuffer, but
   `api_server.py` calls cleanup in the stream worker `finally` block. Cleanup
   calls `end_live()`, which drains queued frames. Short clips or fast
   generation may return to idle before the queued live frames are actually
   watched. For immediate testing set `WEBRTC_VIDEO_PREBUFFER_SECONDS=0`; the
   durable fix is to let the track drain naturally after
   `signal_generation_complete()`.

2. Offer can wait forever.
   `_wait_for_ice_gathering()` has no timeout. If STUN/TURN gathering stalls,
   `/webrtc/sessions/{id}/offer` can hang. Add a 3-5 second timeout if this
   shows up during browser tests.

3. No WebRTC status endpoint.
   We currently rely on `/stats`, `/live/sessions`, browser getStats, and logs.
   A small `GET /webrtc/sessions/{id}/status` endpoint would make debugging much
   easier by exposing peer connection state, ICE state, active stream, and track
   queue stats.

4. HLS scheduler improvements do not cover WebRTC.
   WebRTC still uses the older direct executor path. Concurrency/load testing
   must be redone separately; do not assume the HLS 8-stream findings apply.

5. TURN config is stale.
   Do not use existing TURN docs by copy/paste without replacing IPs, public
   ports, and passwords.

6. Codec/encoder path needs validation.
   The server patches aiortc H.264 to prefer `h264_nvenc` and fallback to
   `libx264` unless strict mode is enabled. Watch logs for encoder selection and
   test Safari/iOS early.

## Recommendation

Start with local browser WebRTC while the existing server is healthy, because
session creation already works. Then restart with `WEBRTC_VIDEO_PREBUFFER_SECONDS=0`
and do one real audio upload. If local playback works, restore either direct UDP
`40000-40100` or a real TURN server before testing phones or React Native. The
first code hardening pass should be the prebuffer/cleanup race plus an ICE
gathering timeout and a WebRTC status endpoint.

## 2026-05-18 Playback Smoothing Update

See `current_webrtc_playback_smoothing_findings.md` for the current jitter/audio
analysis. The new finding is that the WebRTC symptoms are largely caused by the
server-side media clocking policy:

- video queue depth currently changes effective playout FPS
- audio drift correction can sleep or skip audio frames to follow the variable
  video clock
- custom video tracks call aiortc's default `next_timestamp()`, which advances
  RTP timestamps at 30 FPS even for non-30 FPS sessions
- `playback_fps > source_fps` can consume source frames too quickly instead of
  duplicating frames

The next WebRTC hardening pass should make audio a fixed-speed master clock,
timestamp video at the configured playback FPS, disable adaptive FPS by default,
add a real prebuffer before live reveal, and adapt video by holding/duplicating
frames rather than speeding up or slowing down audio.
