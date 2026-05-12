# React Native HLS Integration (MuseTalk)

This guide explains how to use the HLS session API from a React Native app. It covers the recommended iOS path and the Android fallback.

---

## High-level flow

1) Create an HLS session (`POST /hls/sessions/create`).
2) Load the idle manifest (`/hls/sessions/{id}/index.m3u8`).
3) Upload audio to start live generation (`POST /hls/sessions/{id}/stream`).
4) Poll `/hls/sessions/{id}/status` and switch to the live manifest when `live_ready` becomes true.
5) When streaming ends, return to the idle manifest (optionally resume idle head).

---

## Platform guidance

- iOS (Safari/WKWebView): native HLS playback is reliable. Use HLS directly.
- Android (Chrome/WebView): HLS can be unreliable; use `/player/session/{id}` (SSE player) if you see buffering/black screen. If you still want HLS on Android, use a native player (ExoPlayer via `react-native-video`) and test thoroughly.

---

## 0) Prepare the avatar (one-time)

Before creating HLS sessions, prepare an avatar from a source video. This is a one-time operation per `avatar_id`.

```http
POST /avatars/prepare?avatar_id=test_avatar&batch_size=20&bbox_shift=0
Content-Type: multipart/form-data
```

Form field name must be `video_file`.

Notes:
- `batch_size` here is for preparation only; HLS streaming uses its own `batch_size`.
- `bbox_shift` is optional; adjust if the face crop feels too tight/loose.
- To re-run preparation, pass `force_recreate=true`.

---

## 1) Create an HLS session

```http
POST /hls/sessions/create?avatar_id=test_avatar&playback_fps=30&musetalk_fps=10&batch_size=2&segment_duration=1
```

Notes:
- `segment_duration=1` reduces latency (higher CPU).
- Omit `part_duration` unless LL-HLS is supported in your FFmpeg build.

Response:
```json
{
  "session_id": "...",
  "player_url": "/hls/player/{session_id}",
  "manifest_url": "/hls/sessions/{session_id}/index.m3u8",
  "config": { "...": "..." }
}
```

---

## 2) Upload audio to start live generation

```http
POST /hls/sessions/{session_id}/stream
Content-Type: multipart/form-data
```

Form field name must be `audio_file`.

---

## Recommended: WebView (iOS)

The simplest path is to embed the server-hosted HLS player:

```tsx
import { WebView } from "react-native-webview";

<WebView
  source={{ uri: `${BASE_URL}/hls/player/${sessionId}` }}
  mediaPlaybackRequiresUserAction={false}
/>
```

This player already handles idle/live switching, cache-busting, and UI.

---

## Latest playback stability update (2026-05-12)

The hosted `/hls/player/{session_id}` now includes two recovery paths for the
FaceTime/WebView flow:

- Live-to-idle handoff no longer waits forever for
  `requestVideoFrameCallback`. The player first tries to capture the current
  frame for the hold canvas, then waits only briefly for a fresh frame before
  continuing the transition. The hold canvas is cosmetic, so it should preserve
  the smooth visual swap when frames are available without trapping the UI on
  `Finishing...` when the live media element has ended or stalled.
- Idle playback now tracks idle video progress and recovers from native HLS
  loop stalls. If the idle layer fires `waiting`, `stalled`, or `ended` while no
  live stream is pending, the player seeks back to the computed idle resume
  time, attempts playback again, waits for a painted frame, and reloads the idle
  HLS manifest if the element still does not recover.

Why this changed:

- Backend checks during the failure showed `active_stream: null`, no queued HLS
  scheduler jobs, and no live generation in progress. That made the long
  `Buffering...` state an idle-player problem rather than a batch-size or cold
  generation problem.
- The earlier `Finishing...` hang was caused by a player-side transition wait:
  once live media had ended or stalled, `requestVideoFrameCallback` could stop
  firing, so the transition back to idle could remain blocked.

Impact:

- Smooth live-to-idle transitions are preserved in the normal case because the
  player still captures and holds the last visible frame while the idle layer is
  primed behind it.
- The fallback only runs when the player is idle and no live reveal/transition
  is in flight, so it should not interrupt active live playback or the
  live-to-idle handoff.
- If native HLS gets stuck while looping the idle VOD, the player now has a
  bounded recovery path instead of remaining frozen behind `Buffering...`.

Result:

- The server was restarted with the updated player, the served HTML was verified
  to contain the new recovery code, and the avatar cache was warmed again.
- Recent FaceTime/HLS runs did not reproduce the black screen or the stuck
  `Finishing...` state.

---

## Native Player (react-native-video)

If you want to use HLS directly, use two video layers (idle + live) to avoid black flashes.

```tsx
import Video from "react-native-video";

const idleUrl = `${BASE_URL}/hls/sessions/${sessionId}/index.m3u8`;
const [liveUrl, setLiveUrl] = useState<string | null>(null);
const [mode, setMode] = useState<"idle" | "preparingLive" | "live">("idle");
const [liveVisible, setLiveVisible] = useState(false);
const idleCurrentTimeRef = useRef(0);
const liveProgressRef = useRef({ currentTime: 0, lastMovedAt: Date.now() });

function revealLive() {
  markIdleAnchor(idleCurrentTimeRef.current);
  setLiveVisible(true);
  setMode("live");
}

function returnToIdle() {
  setLiveVisible(false);
  setMode("idle");
}

<Video
  source={{ uri: idleUrl }}
  paused={mode === "live"}
  muted
  repeat
  style={[styles.video, liveVisible && styles.hiddenVideo]}
  onProgress={(event) => {
    idleCurrentTimeRef.current = event.currentTime;
  }}
/>

{liveUrl && (
  <Video
    source={{ uri: liveUrl }}
    paused={mode === "idle"}
    muted={mode !== "live"}
    style={[styles.video, !liveVisible && styles.hiddenVideo]}
    onReadyForDisplay={revealLive}
    onProgress={(event) => {
      const currentTime = event.currentTime;
      if (Math.abs(currentTime - liveProgressRef.current.currentTime) > 0.02) {
        liveProgressRef.current = { currentTime, lastMovedAt: Date.now() };
      }
    }}
    onEnd={returnToIdle}
  />
)}
```

### Live switching logic (polling)

```ts
async function pollStatus() {
  const res = await fetch(`${BASE_URL}/hls/sessions/${sessionId}/status`, {
    cache: "no-store",
  });
  if (!res.ok) return;
  const data = await res.json();

  if (data.status === "streaming" && data.live_ready) {
    // Cache-bust per stream to avoid stale live.m3u8
    setLiveUrl(`${BASE_URL}/hls/sessions/${sessionId}/live.m3u8?stream_id=${data.active_stream}`);
    setLiveVisible(false);
    setMode("preparingLive");
  } else if (data.status !== "streaming" && mode === "live") {
    // Backend generation is done, but playback may still have buffered HLS
    // segments. Let onEnd return to idle; use this only as a stuck-player
    // fallback for mobile HLS players that never fire onEnd.
    const liveHasNotMovedMs = Date.now() - liveProgressRef.current.lastMovedAt;
    if (liveHasNotMovedMs > Math.max(3000, (data.segment_duration ?? 2) * 1500)) {
      returnToIdle();
    }
  }
}
```

Poll every 800–1500ms while the session is active.

Keep the idle layer visible until the live `Video` fires `onReadyForDisplay`.
Switching the visible layer as soon as `live_ready` becomes true can expose a
blank/native-player frame while the live HLS decoder is still attaching.
`styles.hiddenVideo` should keep the layer mounted and decoding, usually with
`opacity: 0`, not `display: none`.

Do not switch back to idle only because `/status` reports `status: "idle"` or
`active_stream: null`. That means the server finished generating the HLS
playlist; the user may still be watching buffered live segments. Prefer
`onEnd`, with a progress-based fallback like the snippet above for mobile
players that can stall at the end of an HLS event playlist.

---

## Optional: idle continuity (resume head)

To make the transition back to idle seamless, track idle head time when live starts and seek back when live ends.

```ts
let idleAnchorTime = 0;
let idleAnchorWallTime = 0;
let idleDuration = 0;

function markIdleAnchor(currentTime: number) {
  idleAnchorTime = currentTime;
  idleAnchorWallTime = Date.now();
}

function computeResumeTime() {
  if (!idleDuration) return 0;
  const elapsed = (Date.now() - idleAnchorWallTime) / 1000;
  return (idleAnchorTime + elapsed) % idleDuration;
}
```

Call `markIdleAnchor` when live begins, then seek the idle player to `computeResumeTime()` before returning to idle.

When `hls_server_timing=true`, the server computes the talking-video start
offset for an expected reveal delay. The stream response includes `timing`, and
`/hls/sessions/{id}/status` includes `live_timing`; if you reveal live earlier
than that expected delay, the first talking frame can be ahead of the visible
idle head. For the smoothest head-position continuity, either use the hosted
`/hls/player/{session_id}` WebView or let the hidden live layer play muted until
it is display-ready and your chosen reveal delay has elapsed.

---

## React Native fetch examples

```ts
const BASE_URL = "http://YOUR_HOST:PORT";

async function createHlsSession() {
  const url = `${BASE_URL}/hls/sessions/create?avatar_id=test_avatar&playback_fps=30&musetalk_fps=10&batch_size=2&segment_duration=1`;
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) throw new Error("Create HLS session failed");
  return res.json();
}

async function sendAudio(sessionId: string, fileUri: string) {
  const form = new FormData();
  form.append("audio_file", {
    uri: fileUri,
    type: "audio/mpeg",
    name: "tts.mpga",
  } as any);

  const res = await fetch(`${BASE_URL}/hls/sessions/${sessionId}/stream`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error("Audio upload failed");
}
```

---

## Common errors

- 404 on `/hls/sessions/{id}/stream`: session expired or wrong ID.
- 409 on `/hls/sessions/{id}/stream`: a stream is already active; wait until it finishes.
- Black screen on Android WebView: use `/player/session/{id}` (SSE) instead.

---

## Production notes

- Use HTTPS for real devices (ATS on iOS blocks plain HTTP).
- Keep segment duration low (1–2s) for faster live start.
- For Android, prefer the SSE session player or ExoPlayer if HLS is required.
