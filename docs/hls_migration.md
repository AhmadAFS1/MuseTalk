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
POST /hls/sessions/{session_id}/stream
- multipart/form-data: audio_file
Response mirrors /sessions/stream semantics.

3) HLS Manifest
GET /hls/sessions/{session_id}/index.m3u8

4) HLS Segments / Parts
GET /hls/sessions/{session_id}/segments/seg_000001.m4s
GET /hls/sessions/{session_id}/parts/part_000001_000003.m4s

5) Session status, delete, stats
GET /hls/sessions/{session_id}/status
DELETE /hls/sessions/{session_id}
GET /hls/sessions/stats

6) HLS Player
GET /hls/player/{session_id}
- iOS: use native HLS (video src points to index.m3u8).
- non-iOS: use hls.js fallback.

Data flow (high-level)
1) Client creates session and opens /hls/player/{id}.
2) Client uploads audio to /hls/sessions/{id}/stream.
3) Server runs inference, generates frames, and pushes them into an HLS segmenter.
4) Segmenter writes CMAF fMP4 segments (and parts for LL-HLS) + updates playlist.
5) Player polls playlist and plays segments/parts as they appear.
6) When generation ends, server writes EXT-X-ENDLIST.

How HLS queuing and idle/live switching works

HLS queue model (playlist = queue)
- The HLS playlist is the queue. Each new segment (and each part for LL-HLS) is appended to index.m3u8.
- The player reads the playlist head-to-tail and plays in order.
- The server controls the effective buffer by setting the playlist window size (EXT-X-MEDIA-SEQUENCE + removing old segments).

Frame queue feeding the segmenter
- Maintain an in-memory frame queue (ring buffer) per session.
- A segmenter process reads from that queue and emits fMP4 segments.
- Queue depth (in seconds) is your smoothing buffer, e.g. 1-2s.
- If the queue runs low, the output will still be smooth; latency grows slightly as the player stays behind the live edge.

Idle video playback
Two viable options (choose one):

Option A: Single pipeline (recommended)
- Keep one segmenter running all the time.
- When there is no live audio, feed idle loop frames into the queue.
- When live frames arrive, switch the frame source to live frames.
- This produces a continuous HLS stream with no player reload.

Option B: Dual playlists
- Maintain an idle playlist (loop) and a live playlist.
- The player starts on idle, then swaps src to live when a stream begins.
- This is simpler for generation logic but causes a player source swap.

Switching idle -> live (Option A)
- When the first live frame is available, flip the frame source to live.
- If you want a clean decoder boundary, insert EXT-X-DISCONTINUITY in the playlist at the switch.
- Use EXT-X-PROGRAM-DATE-TIME to mark the transition in wall time.
- If idle/live resolution, fps, and codec are identical, you can skip discontinuity.

Switching live -> idle (Option A)
- If you want the stream to continue after generation, switch back to idle frames.
- If you want a finite playback, append EXT-X-ENDLIST and stop the segmenter.

LL-HLS parts and smoothness
- With LL-HLS, the playlist also exposes parts (EXT-X-PART).
- The player can start rendering on a part before the full segment exists.
- This reduces latency while keeping smoothing from the queue and window.

Suggested latency model
- Segment duration: 1.0s, part duration: 0.2s (LL-HLS).
- Queue depth: 1-2 seconds (keep 1-2s worth of frames in memory).
- Playlist window: 6-10 segments (depends on desired buffer size).

Components to implement
1) HLS Session Manager
- Similar to scripts/session_manager.py, but track HLS playlist state.
- Fields per session: output_dir, playlist path, sequence numbers, segment list, part list, timestamps, status.
- TTL cleanup: remove segment directory and session metadata.

2) HLS Segmenter
Two phases are recommended:

Phase A: Regular HLS (simpler, stable)
- Create 1-2 second segments per chunk (similar to MSE chunk creation).
- Generate index.m3u8 with EXTINF entries and a sliding window.
- End with EXT-X-ENDLIST after generation finishes.

Phase B: LL-HLS (lower latency)
- Use CMAF fMP4 segments with parts (EXT-X-PART).
- Keep part duration 0.2-0.5s, segment duration 1s.
- Use blocking reload and preload hints.

3) Playlist builder
- Maintains sequence numbers and window size.
- Adds EXT-X-PROGRAM-DATE-TIME for sync.
- Adds LL-HLS tags when enabled:
  - EXT-X-PART-INF:PART-TARGET=0.2
  - EXT-X-SERVER-CONTROL:CAN-BLOCK-RELOAD=YES,HOLD-BACK=1.5,PART-HOLD-BACK=0.6
  - EXT-X-PRELOAD-HINT for next part

4) Storage layout
- Base: results/hls/{session_id}/
  - index.m3u8
  - segments/seg_000001.m4s
  - parts/part_000001_000003.m4s
 - Keep a fixed window size (eg 6-12 segments) and delete older segments.

5) Player implementation
- templates/hls_player.py
- Detect iOS user agent:
  - iOS: video.src = manifest
  - others: use hls.js (MSE) as fallback
- Keep a small buffer for smoothness (1-2 seconds).

HLS player behavior (detailed)
- Two stacked video elements: one for idle playback, one for live playback. Idle stays visible until live is actually playing to avoid black flashes and first-frame freezes.
- Idle path:
  - Load `/hls/sessions/{id}/index.m3u8` into the idle video on page load.
  - Idle video is muted, looped, and autoplays (subject to user-gesture policy).
- Live path:
  - When the server reports `status=streaming`, the player preloads `/hls/sessions/{id}/live.m3u8?stream_id={active_stream}` into the live video in the background (cache-busting per stream).
  - Once the live video emits `playing`, the player cross-fades to the live layer and pauses the idle video.
  - If streaming ends, the player fades back to idle after the live video `ended` event.
- Player polling:
  - Poll `/hls/sessions/{id}/status` ~800–1500ms to detect stream start/end.
  - Switch to live only when `live_ready` is true (first live segment written).
  - Show "Preparing live..." while live is generating but not yet ready.
- Autoplay / user activation:
  - Initial tap enables audio and playback (iOS and mobile browsers require a user gesture).
  - After activation, live playback is unmuted; idle remains muted.
- Buffering UI:
  - "Buffering..." shown only for the active layer; idle remains visible.
  - Avoid destroying the idle player during live transitions; keep it ready for smooth return.

Idle continuity (resume head)
- Goal: when returning from live to idle, resume idle at the logical head position instead of restarting at 0.
- Capture idle head on live start:
  - `idleAnchorTime = idleVideo.currentTime`
  - `idleAnchorWallTime = performance.now()`
- On live end, compute the resume point:
  - `elapsed = (performance.now() - idleAnchorWallTime) / 1000`
  - `resumeTime = (idleAnchorTime + elapsed) % idleDuration`
  - Seek idle video to `resumeTime` before showing it (wait for `loadedmetadata` and `canplay`).
- Optional alternative: keep the idle video playing muted in the background while live is active, then simply reveal it on return. This avoids seeking but costs extra bandwidth.
- Track `idleDuration` from `idleVideo.duration` (fall back to 0 if not known yet). For short idle loops, modulo wrap will be obvious; consider longer idle loops for smoother continuity.

Live queue + cleanup behavior
- Each live stream writes to its own segment namespace: `segments/{active_stream}/chunk_0000.ts`, etc. This avoids stale playback state when multiple `/hls/sessions/{id}/stream` calls happen over time.
- The live playlist (`live.m3u8`) points to the per-stream segment path. The player receives the current `active_stream` from `/status` and appends it as `?stream_id=` to force a fresh playlist fetch.
- Live segments are **not** reused across streams. The server sets `Cache-Control: no-store` on `chunk_*.ts` to prevent clients from caching old live chunks.
- Cleanup:
  - Per-stream segment directories remain until the session is deleted.
  - Deleting the HLS session removes the entire `results/hls/{session_id}` directory.
  - Optional optimization (not required): remove old `segments/{old_stream_id}` folders after a stream finishes to cap disk usage.

Encoding details
Recommended base settings for compatibility:
- Video: H.264 (AVC), yuv420p, baseline/main profile
- Audio: AAC-LC, 48kHz
- CMAF fMP4 segments

Option A: per-segment encoding (simple)
- For each chunk of frames + audio slice, run ffmpeg to produce one fMP4 segment.
- Update index.m3u8 after each segment.
- This mirrors existing MSE chunk code paths.

Option B: persistent ffmpeg process (LL-HLS)
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
- HLS_PART_SECONDS (default 0.2, only for LL-HLS)
- HLS_WINDOW_SEGMENTS (default 6)
- HLS_OUTPUT_DIR (default results/hls)
- HLS_USE_LL (true/false)
- HLS_PLAYBACK_FPS (idle playback rate)
- HLS_MUSETALK_FPS (lip-sync generation rate)

LL-HLS conversion (required changes)
LL-HLS conversion (required changes)
- Ensure FFmpeg supports LL-HLS (`hls_part_duration` / `hls_part_filename`). Check with `ffmpeg -h muxer=hls`. Ubuntu 22.04's default FFmpeg 4.4 does not expose these options.
- Use a persistent HLS segmenter per stream (one ffmpeg process) fed by raw frames; avoid per-chunk encoding because it resets timestamps and stalls LL playback.
- Output CMAF fMP4 with an init segment (`-hls_segment_type fmp4` + `-hls_fmp4_init_filename init.mp4`) so the playlist includes `EXT-X-MAP`. Parts must be pure moof/mdat fragments, not self-initializing MP4s.
- Emit real segments (~1s) plus parts (200–400ms) via `-hls_time` and `-hls_part_duration`, with GOP aligned to segment duration to keep keyframes on segment boundaries.
- Keep live outputs separate from idle assets (distinct filenames/dirs) to prevent clobbering idle playback.
- Let the segmenter write the playlist; don’t hand-build `live.m3u8` when using LL-HLS. This ensures `EXT-X-PART`, `EXT-X-PRELOAD-HINT`, and `EXT-X-SERVER-CONTROL` tags are consistent.
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
