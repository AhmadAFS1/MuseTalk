# HLS / LL-HLS Migration Plan (from MSE)

This document describes how to add HLS / LL-HLS streaming as a new, parallel API surface without changing the existing MSE session APIs or the WebRTC APIs.

Goals
- Add new HLS endpoints that mirror the existing /sessions APIs.
- Keep /player/session and /webrtc unchanged.
- Support iOS playback (native HLS) with optional LL-HLS for lower latency.
- Accept 1-2 seconds of added latency in exchange for smooth playback.

Non-goals
- Replace or modify MSE or WebRTC.
- Require MSE on iOS.

Compatibility overview (iOS)
- iOS Safari and WKWebView support HLS natively.
- LL-HLS is supported on iOS 14+ (more reliable on 15+). Older iOS will fall back to regular HLS.
- Use H.264 + AAC for best compatibility.
- Use CMAF fMP4 segments for LL-HLS and good cross-browser support.

Current implementation snapshot (as in repo)
- HLS is implemented as dual playlists: a static idle VOD playlist (`index.m3u8`) plus a live event playlist (`live.m3u8`).
- Idle HLS is generated once per session via ffmpeg as fMP4 segments in `results/hls/{session_id}/segments/seg_*.m4s`.
- Live HLS is assembled by the server: each inference chunk is encoded as a `.ts` segment under `results/hls/{session_id}/segments/{request_id}/chunk_*.ts`, and `live.m3u8` is appended with `EXTINF` entries.
- The player uses two stacked `<video>` elements plus a hold-frame canvas overlay; it reveals live only after prebuffering and a decoded frame, and returns to idle by holding the last live frame until idle has a frame ready.
- `part_duration` is accepted in the session config but LL-HLS parts are not generated yet.

New API surface (parallel to /sessions)
All endpoints below are new and independent from existing session and WebRTC routes.

1) Create HLS Session
POST /hls/sessions/create?avatar_id=...&user_id=...&playback_fps=30&musetalk_fps=10&batch_size=2&segment_duration=1&part_duration=0.2
Response:
{
  "session_id": "...",
  "player_url": "/hls/player/{session_id}",
  "manifest_url": "/hls/sessions/{session_id}/index.m3u8",
  "expires_in_seconds": 3600,
  "config": {"playback_fps": 30, "musetalk_fps": 10, "batch_size": 2, "segment_duration": 1, "part_duration": 0.2}
}

2) Start HLS Stream (upload audio)
POST /hls/sessions/{session_id}/stream?start_offset_seconds=5.12
- multipart/form-data: audio_file
Response mirrors /sessions/stream semantics.
Note: `start_offset_seconds` is expressed in idle playback seconds; the server scales it by `playback_fps / musetalk_fps` before generation.

3) HLS Manifest
GET /hls/sessions/{session_id}/index.m3u8

4) HLS Segments / Assets
GET /hls/sessions/{session_id}/segments/seg_000001.m4s
GET /hls/sessions/{session_id}/segments/{active_stream}/chunk_0000.ts
GET /hls/sessions/{session_id}/init.mp4

5) Session status, delete, stats
GET /hls/sessions/{session_id}/status
DELETE /hls/sessions/{session_id}
GET /hls/sessions/stats

6) HLS Player
GET /hls/player/{session_id}
- If the browser can play HLS natively, set `video.src` to the manifest.
- Otherwise, fall back to hls.js.

Data flow (high-level)
1) Client creates session and opens /hls/player/{id}.
2) Client uploads audio to /hls/sessions/{id}/stream.
3) Server runs inference and writes per-chunk `.ts` segments (one ffmpeg call per chunk).
4) `live.m3u8` is appended with `EXTINF` lines and `segments/{active_stream}/chunk_*.ts` entries; `live_ready` flips true on the first segment.
5) The player polls `/hls/sessions/{id}/status`, attaches the live manifest when `status=streaming`, and reveals live only after prebuffer + decoded frame.
6) When generation ends, the server appends `#EXT-X-ENDLIST`, sets `status=idle`, and clears `active_stream`.

How HLS queuing and idle/live switching works (current)

Dual-playlist model (implemented)
- Idle playlist: static VOD playlist generated at session creation (`index.m3u8` + `segments/seg_*.m4s`). The HTML video element loops it.
- Live playlist: event playlist built by the server (`live.m3u8`). Each inference chunk becomes one `.ts` segment and is appended with `EXTINF`.
- There is no long-running segmenter or in-memory frame queue for HLS. The browser's buffer and hls.js settings provide smoothing.

Switching idle -> live (player behavior)
- The player polls `/hls/sessions/{id}/status` and preloads the live manifest when `status=streaming`.
- Live reveal is gated: it waits for a decoded frame and a minimum buffered lead (`LIVE_PREBUFFER_SECONDS`, currently 0.75s).
- Until that gate opens, idle remains visible. This avoids black flicker and first-frame stalls.

Switching live -> idle (player behavior)
- On `liveVideo.ended`, the player captures the last live frame into a canvas overlay.
- It primes idle playback (seek to a continuity point + wait for a decoded frame), then swaps layers and removes the overlay.
- This yields a seamless handoff without a visible fade (only any encoder-level crossfade remains).

Optional future model (not implemented yet)
- A single continuous HLS pipeline (idle frames when idle, live frames when streaming) could eliminate manifest switches entirely.
- LL-HLS parts would require a persistent segmenter and playlist writer; the current per-chunk TS approach does not generate parts.

Suggested latency model
- Current: segment duration ~1.0s with per-chunk TS segments; player reveal gate is ~0.75s of buffered live video.
- Future (LL-HLS): part duration ~0.2s with blocking reload + preload hints.
- Playlist window: event playlist today; sliding window can be added if needed for memory control.

Components (current)
1) HLS Session Manager (implemented in `scripts/hls_session_manager.py`)
- Tracks session metadata and playlist paths; generates the idle HLS VOD playlist at session creation.
- Builds the live event playlist by appending `EXTINF` lines as segments arrive.
- TTL cleanup removes the session directory.

2) HLS segmenter (implemented in `scripts/api_avatar.py`)
- Per-chunk encoding to MPEG-TS (`chunk_*.ts`) with audio slices aligned to the chunk window.
- Chunks are written under `results/hls/{session_id}/segments/{request_id}/`.

3) Playlist builder (implemented in `scripts/hls_session_manager.py`)
- Writes `live.m3u8` with `EXT-X-PLAYLIST-TYPE:EVENT`, `EXTINF`, and `#EXT-X-ENDLIST`.
- Does not emit LL-HLS tags or parts yet.

4) Storage layout (current)
- Base: `results/hls/{session_id}/`
  - `index.m3u8`
  - `live.m3u8`
  - `segments/seg_*.m4s` (idle fMP4)
  - `segments/{request_id}/chunk_*.ts` (live)

5) Player implementation (current)
- `templates/hls_player.py` uses native HLS when available, otherwise hls.js.
- Two video layers + a hold-frame canvas overlay; live reveal gated by prebuffer and decoded frame.

Future components (LL-HLS)
- Persistent segmenter emitting CMAF fMP4 + parts.
- Playlist tags for LL-HLS (EXT-X-PART, EXT-X-SERVER-CONTROL, EXT-X-PRELOAD-HINT).

HLS player behavior (detailed)
- Two stacked video elements plus a canvas hold-frame overlay. The player does not use CSS fades for layer switching.
- Idle path:
  - Load `/hls/sessions/{id}/index.m3u8` into the idle video on page load.
  - Idle video is muted and looped; playback starts on user gesture where required.
- Live path:
  - When the server reports `status=streaming` and `live_ready=true`, the player prepares `/hls/sessions/{id}/live.m3u8?stream_id={active_stream}` (cache-busting per stream).
  - Live reveal is gated: the player waits for a decoded frame and a minimum buffered lead (`LIVE_PREBUFFER_SECONDS`, currently 0.75s) before showing live.
  - On reveal, the player sets `currentMode=live`, pauses idle, and captures an idle anchor time for continuity.
- Live end:
  - On `liveVideo.ended`, the player captures the last live frame into the hold canvas, primes idle playback until a frame is decoded, then swaps layers and removes the overlay.
  - The "Finishing..." status is UI-only; the actual switch happens on `ended`.
- Player polling:
  - Poll `/hls/sessions/{id}/status` every ~800ms to detect stream start/end.
  - Show "Preparing live..." while streaming but not yet ready; show "Finishing..." when streaming ends.
- Autoplay / user activation:
  - A user gesture unlocks playback; live is unmuted, idle remains muted.
- Buffering UI:
  - "Buffering..." is shown only for the active layer; the hold overlay prevents black flicker during transitions.

Session logs and observability (current)
- Session lifecycle: `Created HLS session: {session_id} (avatar: ..., user: ...)` and `Deleted HLS session: {session_id}` are emitted by `HlsSessionManager` (log lines include emoji prefixes in actual output).
- Stream lifecycle: `[request_id] Starting HLS streaming for session {session_id}` and `[request_id] HLS streaming complete` come from the stream worker (`api_server.py`).
- Inference timeline (`api_avatar.py`):
  - `STARTING STREAMING GENERATION` marks request start and output directories.
  - `PHASE 1: Audio Processing` reports whisper features, total frames, expected chunks.
  - `PHASE 2: Frame Generation & Streaming` reports time to first frame, then per-chunk logs (`CREATING CHUNK`, buffer size, progress, creation time, duration).
  - `STREAMING GENERATION COMPLETE` summarizes total frames/chunks and elapsed time.
- Playlist correlation: each `Chunk created` log corresponds to one new `.ts` segment and one new `EXTINF` entry appended to `live.m3u8`; the first chunk flips `live_ready=true`.

Idle continuity (resume head)
- Goal: when returning from live to idle, resume idle at the logical head position instead of restarting at 0.
- Capture idle head on live start:
  - `idleAnchorTime = idleVideo.currentTime`
  - `idleAnchorWallTime = performance.now()`
- On live end, compute the resume point:
  - `elapsed = (performance.now() - idleAnchorWallTime) / 1000`
  - `resumeTime = (idleAnchorTime + elapsed) % idleDuration`
- Seek idle video to `resumeTime` before showing it (wait for a decoded frame via `requestVideoFrameCallback`, `playing`, or `timeupdate`).
- Optional alternative: keep the idle video playing muted in the background while live is active, then simply reveal it on return. This avoids seeking but costs extra bandwidth.
- Track `idleDuration` from `idleVideo.duration` (fall back to 0 if not known yet). For short idle loops, modulo wrap will be obvious; consider longer idle loops for smoother continuity.

Potential enhancement: align live start to idle head (proposal)
Goal
- Start live chunk generation at an offset that matches the current idle playback head, so the base frame continuity is preserved when talking begins.

Why this needs client input
- The server does not know the client's true idle playback time (autoplay delays, buffering, user gesture timing).
- A client-provided timestamp is required for accurate alignment.

Client side (React Native WebView bridge)
- Add a WebView -> RN message that reports `idleVideo.currentTime`.
- Either request on-demand (right before /stream) or push periodically.
- Use the most recent `idle_time` when calling `/hls/sessions/{id}/stream`.

Server side (API surface)
- Add an optional query param or form field, e.g. `start_offset_seconds` (default 0). (Implemented as a query param on `/hls/sessions/{id}/stream`.)
- Interpret the value as idle playback time; scale to generation time using `playback_fps / musetalk_fps`.
- Keep it backward-compatible: if missing, behave as today.

Generation side (api_avatar.py)
- Apply the same offset to both:
  - latent selection (`datagen(..., delay_frame=offset_frames)`), and
  - background frame/mask selection (start `frame_idx` at `offset_frames` or rotate the cycle lists).
- `offset_frames = round(adjusted_offset_seconds * generation_fps)` after scaling in the API layer.

Crossfade and idle frames
- If using tail crossfade, consider offsetting the idle frame cache to the same head so the blend matches.

Pitfalls
- Idle HLS is forward-loop, while the avatar cycle is ping-pong; perfect long-run alignment is not possible.
- If frame lists and latent lists are different lengths (skipped bboxes), offsets must be applied carefully to keep mouth and base frame aligned.
- If playback_fps != musetalk_fps, the offset is approximate; prefer matching fps for best continuity.

Server-authoritative timing plan (proposed)
Goal
- Make `/hls/sessions/{id}/stream` compute the start offset on the server (no WebView bridge or client-sent idle time).
- Keep idle -> live transitions consistent across devices, with optional client-side alignment for tighter sync.

Design overview
- Track the idle loop timeline on the server using a monotonic clock.
- Compute the current idle position at stream start and map it to the avatar cycle.
- Pass the computed offset to generation as frames (or seconds at generation fps).

Data model additions (HlsSession)
- `idle_start_monotonic`: monotonic time when idle loop is considered to start (time.monotonic()).
- `idle_start_wall_time`: wall-clock time for debugging/optional client alignment (time.time()).
- `idle_duration_seconds`: duration of the idle loop in seconds (from idle video or HLS manifest).
- `idle_cycle_frames`: frame count of the avatar cycle used for generation (from avatar preprocessed data).
- `timing_source`: `"server"` or `"client"` (optional, for logging/debug).

Implementation steps
1) Session creation (after idle HLS generation)
   - Compute `idle_duration_seconds`:
     - Prefer ffprobe on the idle video, or sum `#EXTINF` durations from `index.m3u8`.
   - Cache `idle_cycle_frames` from the avatar data (length of the cycle used in `datagen`).
   - Set `idle_start_monotonic = time.monotonic()` and `idle_start_wall_time = time.time()`.

2) Stream start (`/hls/sessions/{id}/stream`)
   - Compute the idle head time:
     - `idle_elapsed = (time.monotonic() - idle_start_monotonic) % idle_duration_seconds`
     - If `idle_duration_seconds` is unknown/0, fall back to `0.0`.
   - Map to a generation offset:
     - Preferred (frame-accurate):  
       `offset_frames = round((idle_elapsed / idle_duration_seconds) * idle_cycle_frames) % idle_cycle_frames`
     - Fallback (fps-based):  
       `offset_seconds_gen = idle_elapsed * (generation_fps / playback_fps)`  
       `offset_frames = round(offset_seconds_gen * generation_fps)`
   - Pass offset to generation:
     - Either add `start_offset_frames` to the generation API, or convert to seconds:
       `start_offset_seconds = offset_frames / generation_fps`.

3) API response metadata (debugging)
   - Include in `/stream` response:
     - `server_offset_seconds`, `server_offset_frames`
     - `idle_elapsed_seconds`, `idle_duration_seconds`
     - `timing_source: "server"`

4) Status endpoint (debugging + optional client alignment)
   - Add to `/hls/sessions/{id}/status`:
     - `idle_start_wall_time`, `idle_elapsed_seconds`, `idle_duration_seconds`, `timing_source`

5) Optional client-side alignment (for tighter sync)
   - On client load, seek idle playback to:
     - `idle_position = (Date.now()/1000 - idle_start_wall_time) % idle_duration_seconds`
   - Or emit `EXT-X-PROGRAM-DATE-TIME` in the idle playlist and derive current time from playlist dates.

Compatibility and rollout
- Keep `start_offset_seconds` accepted for now, but ignore it when `HLS_SERVER_TIMING=true`.
- Log any client-sent offset for comparison with server offsets.
- Provide a rollback toggle to use client timing if server duration/frames are missing.

Edge cases
- Unknown `idle_duration_seconds`: use `0.0` offset and log a warning.
- If the avatar cycle is ping-pong, prefer `idle_cycle_frames` mapping to avoid fps mismatch drift.
- If playback_fps != musetalk_fps, prefer cycle-based mapping or keep fps matched for best continuity.

Testing and validation
- Unit test the offset mapping (idle_elapsed -> offset_frames) across multiple durations.
- Integration test: start stream at known idle times (1s, 3s, 5s) and verify first chunk matches expected cycle position.
- Long-run test: idle loop for multiple minutes, then start stream and confirm no significant drift.
- Observability: log `idle_elapsed`, `offset_frames`, and `timing_source` per stream request.

Live queue + cleanup behavior
- Each live stream writes to its own segment namespace: `segments/{active_stream}/chunk_0000.ts`, etc. This avoids stale playback state when multiple `/hls/sessions/{id}/stream` calls happen over time.
- The live playlist (`live.m3u8`) points to the per-stream segment path. The player receives the current `active_stream` from `/status` and appends it as `?stream_id=` to force a fresh playlist fetch.
- `live_ready` flips true on the first appended live segment; `finish_live_playlist` writes `#EXT-X-ENDLIST` and sets `status=idle` / `live_ready=false`, and the API clears `active_stream`.
- Live segments are **not** reused across streams. The server sets `Cache-Control: no-store` on `chunk_*.ts` to prevent clients from caching old live chunks (idle segments are `no-cache`).
- Cleanup:
  - `start_live_playlist` only removes `segments/chunk_*` at the root; per-stream segment directories remain until the session is deleted.
  - Deleting the HLS session removes the entire `results/hls/{session_id}` directory.
  - Optional optimization (not required): remove old `segments/{old_stream_id}` folders after a stream finishes to cap disk usage.

Encoding details
Recommended base settings for compatibility:
- Video: H.264 (AVC), yuv420p, baseline/main profile
- Audio: AAC-LC, 48kHz
- CMAF fMP4 segments

Current encoding paths (implemented)
- Idle: ffmpeg generates a VOD HLS playlist with fMP4 segments (`-hls_segment_type fmp4`) during session creation.
- Live: each inference chunk is encoded as an independent MPEG-TS segment (`-f mpegts`) with GOP = frames per chunk, then appended to `live.m3u8`.
- Tail crossfade: the final chunk may blend into cached idle frames if available (encoder-level, not a UI fade).

Future option: persistent ffmpeg process (LL-HLS)
- Feed raw video frames to ffmpeg via stdin.
- Provide audio input file.
- Let ffmpeg output HLS segments/parts continuously.

Example ffmpeg command (LL-HLS, CMAF)
ffmpeg -y \
  -f rawvideo -pix_fmt bgr24 -s {W}x{H} -r {FPS} -i pipe:0 \
  -i {audio_path} \
  -c:v libx264 -profile:v main -g {GOP} -keyint_min {GOP} -sc_threshold 0 -pix_fmt yuv420p \
  -c:a aac -b:a 128k -ar 48000 \
  -f hls -hls_time 1 -hls_part_duration 0.2 \
  -hls_playlist_type event -hls_list_size 0 \
  -hls_flags independent_segments+program_date_time+split_by_time \
  -hls_segment_type fmp4 \
  -hls_fmp4_init_filename init.mp4 \
  -hls_segment_filename {out_dir}/segments/live_seg_%06d.m4s \
  -hls_part_filename {out_dir}/parts/live_part_%06d.m4s \
  {out_dir}/live.m3u8

Notes:
- Keep GOP = fps to align keyframes with segment boundaries.
- If you cannot keep a persistent ffmpeg process, use per-segment creation and skip LL-HLS at first.

Config knobs (proposed)
- HLS_SEGMENT_SECONDS (default 1.0)
- HLS_PART_SECONDS (default 0.2, only for LL-HLS; not used in current HLS)
- HLS_WINDOW_SEGMENTS (default 6)
- HLS_OUTPUT_DIR (default results/hls)
- HLS_USE_LL (true/false)
- HLS_PLAYBACK_FPS (idle playback rate)
- HLS_MUSETALK_FPS (lip-sync generation rate)
- HLS_LIVE_PREBUFFER_SECONDS (player reveal gate; currently hardcoded to ~0.75s)

LL-HLS conversion (required changes)
- Ensure FFmpeg supports LL-HLS (`hls_part_duration` / `hls_part_filename`). Check with `ffmpeg -h muxer=hls`. Ubuntu 22.04's default FFmpeg 4.4 does not expose these options.
- Use a persistent HLS segmenter per stream (one ffmpeg process) fed by raw frames; avoid per-chunk encoding because it resets timestamps and stalls LL playback.
- Output CMAF fMP4 with an init segment (`-hls_segment_type fmp4` + `-hls_fmp4_init_filename init.mp4`) so the playlist includes `EXT-X-MAP`. Parts must be pure moof/mdat fragments, not self-initializing MP4s.
- Emit real segments (~1s) plus parts (200-400ms) via `-hls_time` and `-hls_part_duration`, with GOP aligned to segment duration to keep keyframes on segment boundaries.
- Keep live outputs separate from idle assets (distinct filenames/dirs) to prevent clobbering idle playback.
- Let the segmenter write the playlist; don't hand-build `live.m3u8` when using LL-HLS. This ensures `EXT-X-PART`, `EXT-X-PRELOAD-HINT`, and `EXT-X-SERVER-CONTROL` tags are consistent.
- Delivery (`api_server.py`): serve `init.mp4`, segments, and parts with correct MIME types; no-cache playlists; allow normal caching for media; blocking reload is optional (Safari uses it, hls.js does not).
- Player (`templates/hls_player.py`): enable hls.js low-latency mode and keep the buffer small; switch to live only after the live playlist has at least one part/segment; switch back to idle after live ends.
- Encoding parity: keep audio rate (48k) and codec settings consistent between idle and live to avoid decoder resets or silent stalls.
- Validation: confirm `live.m3u8` contains `EXT-X-MAP`, `EXT-X-PART`, `EXT-X-SERVER-CONTROL`, and `EXT-X-PRELOAD-HINT`; verify parts arrive before full segments; measure end-to-end latency.

Backward compatibility
- Existing /player/session and /webrtc endpoints remain unchanged.
- New HLS endpoints live under /hls/*.
- Shared avatar cache and inference code can be reused.

Suggested rollout
1) Implement Phase A (regular HLS) with 1s segments and event playlist.
2) Validate iOS playback and latency target.
3) Add LL-HLS parts and blocking reload for lower latency.
4) Add metrics (segment generation time, playlist latency, playback delay).

Testing checklist
- iOS Safari: confirm playback with native HLS.
- Desktop Chrome: confirm hls.js playback.
- Playlist updates during live generation.
- Endlist appended after stream completes.
- TTL cleanup removes session data.

Open questions
- Target latency (1s vs 2s) to set segment/part durations.
- Whether to keep audio and video in the same segments (recommended) or split audio-only track.
- Whether to use a persistent ffmpeg process or per-segment encoding.
