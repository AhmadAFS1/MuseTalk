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

## Native Player (react-native-video)

If you want to use HLS directly, use two video layers (idle + live) to avoid black flashes.

```tsx
import Video from "react-native-video";

const idleUrl = `${BASE_URL}/hls/sessions/${sessionId}/index.m3u8`;
const [liveUrl, setLiveUrl] = useState<string | null>(null);
const [mode, setMode] = useState<"idle" | "live">("idle");

<Video
  source={{ uri: idleUrl }}
  paused={mode !== "idle"}
  muted
  repeat
  style={styles.video}
/>

{liveUrl && (
  <Video
    source={{ uri: liveUrl }}
    paused={mode !== "live"}
    muted={false}
    style={styles.video}
    onEnd={() => setMode("idle")}
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
    setMode("live");
  } else if (data.status !== "streaming" && mode === "live") {
    setMode("idle");
  }
}
```

Poll every 800–1500ms while the session is active.

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
