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

Update on 2026-05-26: an INT8 calibration restart exposed a WebRTC launch/network
issue, not a model/decoder regression. The wall generated frames and calibration
tensors, but the browser stayed black because ICE never reached `connected`.
The running process had only STUN/direct ICE (`WEBRTC_ICE_TRANSPORT_POLICY=all`,
empty TURN URLs), aiortc bound random UDP sockets on the private Docker address,
and `output_frames_sent` stayed at zero. The normal Vast on-start path does not
secretly start TURN: `/workspace/onstart.sh` is only a stub, `scripts/vast_onstart.sh`
calls `scripts/vast_server_ctl.sh start`, and without relay env the control
script launches `scripts/run_trt_stagewise_server.sh` directly. The saved WebRTC
load-test reports also used `ice_transport_policy: all`; previous success was
direct/local ICE working in that environment, not coturn being auto-started.

The control script now supports `WEBRTC_RELAY_ENABLED=1` to launch through
`scripts/run_webrtc_relay_api_server.sh` and `WEBRTC_TURN_AUTOSTART=1` to start
local coturn through `scripts/run_turnserver_tcp_relay.sh` before the API. Use
`.env.webrtc-turn.local.example` as the local ignored config template.

The current Vast bootstrap script clones `origin/main` into a fresh
`/workspace/MuseTalk` checkout and then runs:

```bash
SETUP_CLEAN=1 SETUP_FULL_STACK=1 STARTUP_TIMEOUT_SECONDS=1800 \
PROFILE=throughput_record PORT=8000 bash scripts/vast_onstart.sh
```

That bootstrap installs WebRTC Python dependencies through setup and starts the
API, but it does not pass WebRTC relay/TURN env. It also discards local
uncommitted edits by recloning the repo, so relay-control changes must be
committed and pushed before a new Vast boot can use them.

Update on 2026-05-21: WebRTC wall generation now defaults to the shared HLS GPU
scheduler with a WebRTC frame sink. The previous wall turn-taking was backend
serialization from independent per-session GPU leases, not a one-browser WebRTC
connection limit. See `current_webrtc_playback_smoothing_findings.md` for the
shared-generation implementation notes and the `20/20` HLS wall reference data.

Update on 2026-05-18: WebRTC strict FIFO now includes an audio-ready A/V start
barrier. Audio is prepared before live video leaves the FIFO gate, and the
shared `VideoSyncClock` reports audio/video readiness, playout release, first
packet/frame timing, and initial A/V start delta. See
`current_webrtc_playback_smoothing_findings.md` for the detailed implementation
notes and smoke-test results.

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
- HLS has a shared GPU scheduler (`scripts/hls_gpu_scheduler.py`), and WebRTC
  now uses that scheduler by default through `submit_webrtc_stream(...)`.
- The older WebRTC path through `manager.executor` plus
  `manager.gpu_memory.allocate(session.batch_size)` remains as a fallback when
  `WEBRTC_SHARED_GPU_SCHEDULER=0`.
- Current HLS throughput tuning now applies to WebRTC generation concurrency,
  but WebRTC still needs separate playback/sync validation because it sends
  separate RTP audio and video tracks to the browser.
- `scripts/run_api_server.sh` is now only a compatibility shim that delegates to
  the WebRTC relay launcher; current TURN values belong in
  `.env.webrtc-turn.local` or explicit environment variables.

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
- `scripts/run_api_server.sh` no longer carries hard-coded TURN URLs; use
  `.env.webrtc-turn.local` for the current host.
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
peers are forced through TURN, so the old broad public UDP relay range does not
need to be opened for this test mode. The single browser/client TURN URL must
be filled from the current host's actual public mapping, for example:

```text
turn:<PUBLIC_IP>:<PUBLIC_TURN_PORT>?transport=<udp-or-tcp>
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
WEBRTC_SYNC_MODE=strict_fifo \
WEBRTC_VIDEO_PREBUFFER_SECONDS=2.0 \
WEBRTC_AUDIO_PREBUFFER_SECONDS=0.0 \
WEBRTC_ADAPTIVE_FPS=0 \
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
stable host. If `turnserver` is not installed on the current node, install
coturn first or use managed TURN with `WEBRTC_TURN_AUTOSTART=0`.

```bash
cd /workspace/MuseTalk
cp .env.webrtc-turn.local.example .env.webrtc-turn.local
```

Fill the ignored `.env.webrtc-turn.local` with the current host mapping and a
real password. For local coturn, it should include:

```text
WEBRTC_RELAY_ENABLED=1
WEBRTC_TURN_AUTOSTART=1
TURN_PUBLIC_IP=<PUBLIC_IP>
TURN_PUBLIC_PORT=<PUBLIC_TURN_PORT>
TURN_PUBLIC_TRANSPORT=udp
TURN_LISTEN_PORT=3478
TURN_PASS=<LONG_RANDOM_PASSWORD>
WEBRTC_USE_LOCAL_TURN=1
```

Then restart through the normal control helper; it will source the ignored env,
start coturn if requested, and launch the API through the WebRTC relay wrapper:

```bash
cd /workspace/MuseTalk
PROFILE=throughput_record bash scripts/vast_server_ctl.sh restart
```

For managed/external TURN, set `WEBRTC_TURN_AUTOSTART=0`,
`WEBRTC_USE_LOCAL_TURN=0`, `WEBRTC_TURN_URLS`, `WEBRTC_SERVER_TURN_URLS`,
`WEBRTC_TURN_USER`, and `WEBRTC_TURN_PASS` instead of local coturn settings.

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

1. Video prebuffer/playback drain.
   The 2026-05-18 smoothing/sync pass changed WebRTC to default to strict FIFO
   sync with a 2 second prebuffer, start live before queueing frames, release
   short clips when generation completes, and let `signal_generation_complete()`
   drain naturally before cleanup. Retest with `WEBRTC_SYNC_MODE=strict_fifo`,
   `WEBRTC_VIDEO_PREBUFFER_SECONDS=2.0`, `WEBRTC_AUDIO_PREBUFFER_SECONDS=0.0`,
   and `WEBRTC_ADAPTIVE_FPS=0`.

2. Offer can wait forever.
   `_wait_for_ice_gathering()` has no timeout. If STUN/TURN gathering stalls,
   `/webrtc/sessions/{id}/offer` can hang. Add a 3-5 second timeout if this
   shows up during browser tests.

3. WebRTC status endpoint still needs browser validation.
   `GET /webrtc/sessions/{id}/status` now exposes video/audio/sync stats. Use it
   alongside browser getStats during real playback to verify stalls, queue depth,
   and sync mode.

4. HLS scheduler improvements cover WebRTC generation, not WebRTC transport.
   The shared scheduler can generate frames and calibration tensors while ICE is
   still broken. Always verify browser stats reach `connected` and
   `output_frames_sent` increases before calling playback healthy.

5. TURN config is stale.
   Do not use existing TURN docs by copy/paste without replacing IPs, public
   ports, and passwords. Prefer `.env.webrtc-turn.local.example` plus
   `WEBRTC_RELAY_ENABLED=1`.

6. Codec/encoder path needs validation.
   The server patches aiortc H.264 to prefer `h264_nvenc` and fallback to
   `libx264` unless strict mode is enabled. Watch logs for encoder selection and
   test Safari/iOS early.

## Recommendation

Start with the relay-preserving control path:
`.env.webrtc-turn.local` plus `WEBRTC_RELAY_ENABLED=1`, then
`bash scripts/vast_server_ctl.sh restart`. After restart, the API log should
show `WEBRTC_ICE_TRANSPORT_POLICY=relay` and non-empty `WEBRTC_TURN_URLS`.
Then create one wall/session, connect it, upload real audio, and confirm
browser stats reach `connected` with `output_frames_sent > 0`. The next code
hardening pass should be an ICE gathering timeout.

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

Implemented on 2026-05-18:

- audio is now fixed-speed and no longer skips/sleeps for drift correction
- video RTP timestamps now use the configured playback FPS
- adaptive FPS defaults to disabled
- live video uses a real prebuffer before audio is signaled
- `playback_fps > source_fps` now holds/duplicates frames instead of consuming
  source frames too quickly

See `current_webrtc_playback_smoothing_findings.md` for the smoke tests and the
next browser validation checklist.

## 2026-05-18 A/V Sync Follow-Up

After the Phase 1 smoothing pass, video playout could look good while A/V sync
could still be fragile. The follow-up implementation now supports a strict
FIFO/HLS-like sync mode instead of treating audio and video as independent tracks
that are corrected after drift appears.

Preferred behavior for this product:

- keep generated video frames in FIFO order
- keep audio packets in FIFO order
- do not speed up or slow down audio
- do not skip generated video frames
- start playout only after a shared A/V buffer exists
- if video falls behind, stall both audio and video rather than letting audio run
  ahead

This mode trades latency and occasional buffering for stable lip sync and
complete playback. It should be separate from a future low-latency mode, where
holding/dropping video might be acceptable to keep audio continuous.

Implemented default target:

```bash
WEBRTC_SYNC_MODE=strict_fifo
WEBRTC_VIDEO_PREBUFFER_SECONDS=2.0
WEBRTC_AUDIO_PREBUFFER_SECONDS=0.0
WEBRTC_ADAPTIVE_FPS=0
```

Implemented code direction:

- add a shared A/V playout gate in `scripts/webrtc_tracks.py`
- timestamp generated video frames with source media time
- packetize audio against the same media timeline
- allow both `SyncedAudioStreamTrack.recv()` and
  `SwitchableVideoStreamTrack.recv()` to wait on the same playout state
- expose stall count/duration and shared buffer depth through WebRTC status
  telemetry
- attach remote audio and video to one browser `MediaStream` on the video
  element for native browser A/V sync
