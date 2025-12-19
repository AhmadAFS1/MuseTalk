# 📱 Mobile Integration (React Native) — Session-Based Streaming Player

This guide explains how to integrate the **MuseTalk Session API** into a **React Native** app so you can:

- Create an isolated **session per user**
- Open a **session player** that auto-connects via **SSE**
- Upload audio to the session (`multipart/form-data`)
- Watch the avatar video stream as **MSE fMP4 chunks**
- Automatically return to the **base/placeholder avatar video** when generation completes

> The session system is designed for multi-user concurrency: each session has its own queue/state and does not interfere with other sessions.

---

## ✅ Architecture Overview

### Components

1. **Session API (Backend)**
   - Creates and tracks sessions
   - Accepts audio uploads per session
   - Generates and serves chunked fMP4 video segments
   - Emits SSE events (`chunk`, `complete`, `error`) per session

2. **Session Player (Web UI)**
   - URL: `GET /player/session/{session_id}`
   - Uses:
     - **placeholder video**: `GET /avatars/{avatar_id}/video`
     - **SSE**: `GET /sessions/{session_id}/events`
     - **MSE**: appends `chunk_XXXX.mp4` into a `MediaSource`

3. **React Native App**
   - Creates sessions in backend
   - Embeds the player in a `WebView`
   - Uploads audio to `/sessions/{session_id}/stream`
   - (Optional) deletes the session on chat close

---

## 🔌 Session Lifecycle

### 1) Create session
`POST /sessions/create?avatar_id=...&user_id=...&batch_size=...&fps=...&chunk_duration=...`

Response typically includes:
- `session_id` (string)
- (recommended) a `player_url` / `player_full_url` (if you provide it)

### 2) Load player UI
`GET /player/session/{session_id}`

The player:
- Starts showing the avatar’s **base input video** (placeholder loop).
- Connects to the session’s **SSE events** endpoint and waits.

### 3) Stream audio to session
`POST /sessions/{session_id}/stream` (multipart form)
- `audio_file=@something.mp3`
- Optional: `trailing_silence=...` query param

The server:
- Kicks off streaming generation
- Enqueues chunk paths into the session queue
- Player receives SSE events and appends chunks

### 4) Complete
When server emits `complete`, player:
- Ends MSE stream
- Switches back to placeholder/base video automatically

### 5) Cleanup (recommended)
`DELETE /sessions/{session_id}` when user closes chat to free resources.

---

## 🌐 Key Endpoints (Session Flow)

### Create Session
```bash
curl -X POST "http://localhost:8000/sessions/create?avatar_id=test_avatar&user_id=user1&batch_size=2&fps=15&chunk_duration=2"
```

### Open Player
Open in browser (or WebView):
```
http://localhost:8000/player/session/{session_id}
```

### Upload Audio to Session
```bash
curl -X POST "http://localhost:8000/sessions/{session_id}/stream?trailing_silence=1.5" \
  -F "audio_file=@data/audio/response-2.mpga"
```

### Session Stats
```bash
curl "http://localhost:8000/sessions/stats"
```

### Delete Session
```bash
curl -X DELETE "http://localhost:8000/sessions/{session_id}"
```

---

## 📦 How Streaming Video Works (Chunks + SSE + MSE)

### Server
- Generates video frames and packages them into **fragmented MP4 (fMP4)** chunks suitable for **MediaSource Extensions**.
- Each chunk is typically ~`chunk_duration` seconds (e.g. 2s).
- Chunks are served from URLs like:
  - `/chunks/{request_id}/chunk_0000.mp4` *(example; actual path depends on your implementation)*

### SSE Event Contract (Player expects)
The session player listens to:
`GET /sessions/{session_id}/events`

It expects JSON `message` events like:

- `chunk`
  ```json
  { "event": "chunk", "url": "/chunks/REQ_ID/chunk_0000.mp4", "index": 0 }
  ```

- `complete`
  ```json
  { "event": "complete" }
  ```

- `error`
  ```json
  { "event": "error", "message": "..." }
  ```

### Client (Player)
- Receives `chunk` events
- Fetches the chunk URL
- Appends `Uint8Array` to a `SourceBuffer`
- Starts playback after first append (`streamVideo.play()`)

When complete:
- Calls `endOfStream()` (MSE)
- Switches back to placeholder video

---

## 📱 React Native Integration (Recommended)

### Approach A (Recommended): Embed the Session Player in a WebView

This is the simplest and most robust approach because MSE + SSE are already implemented in the player.

#### 1) Install WebView
```bash
npm install react-native-webview
# if iOS:
cd ios && pod install && cd ..
```

#### 2) Create session from RN app
Use your preferred HTTP library (`fetch`, axios). Example using `fetch`:

```ts
// CreateSession.ts
export async function createSession({
  apiBaseUrl,
  avatarId,
  userId,
  batchSize = 2,
  fps = 15,
  chunkDuration = 2,
}: {
  apiBaseUrl: string;
  avatarId: string;
  userId: string;
  batchSize?: number;
  fps?: number;
  chunkDuration?: number;
}) {
  const url =
    `${apiBaseUrl}/sessions/create` +
    `?avatar_id=${encodeURIComponent(avatarId)}` +
    `&user_id=${encodeURIComponent(userId)}` +
    `&batch_size=${batchSize}` +
    `&fps=${fps}` +
    `&chunk_duration=${chunkDuration}`;

  const res = await fetch(url, { method: "POST" });
  if (!res.ok) throw new Error(`Create session failed: HTTP ${res.status}`);
  return res.json() as Promise<{ session_id: string; player_url?: string; }>;
}
```

#### 3) Load the player in WebView
```tsx
// AvatarPlayerScreen.tsx
import React, { useEffect, useMemo, useState } from "react";
import { View, ActivityIndicator } from "react-native";
import { WebView } from "react-native-webview";
import { createSession } from "./CreateSession";

export function AvatarPlayerScreen() {
  const apiBaseUrl = "http://YOUR_SERVER_HOST:8000";
  const avatarId = "test_avatar";
  const userId = "user_123";

  const [sessionId, setSessionId] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      const session = await createSession({
        apiBaseUrl,
        avatarId,
        userId,
        batchSize: 2,
        fps: 15,
        chunkDuration: 2,
      });
      setSessionId(session.session_id);
    })();
  }, []);

  const playerUrl = useMemo(() => {
    if (!sessionId) return null;
    return `${apiBaseUrl}/player/session/${sessionId}`;
  }, [apiBaseUrl, sessionId]);

  if (!playerUrl) {
    return (
      <View style={{ flex: 1, alignItems: "center", justifyContent: "center" }}>
        <ActivityIndicator />
      </View>
    );
  }

  return (
    <WebView
      source={{ uri: playerUrl }}
      javaScriptEnabled
      domStorageEnabled
      allowsInlineMediaPlayback
      mediaPlaybackRequiresUserAction={false}
      originWhitelist={["*"]}
      style={{ flex: 1, backgroundColor: "black" }}
    />
  );
}
```

> Note: iOS needs `allowsInlineMediaPlayback` and `mediaPlaybackRequiresUserAction={false}` for autoplay.

---

## 🎙️ Upload Audio from React Native

### Audio Upload Requirements
- Endpoint: `POST /sessions/{session_id}/stream`
- Content-Type: `multipart/form-data`
- Field name **must be**: `audio_file`

You need a file URI from:
- `react-native-audio-recorder-player`, `expo-av`, `react-native-voice`, etc.
- Then use `FormData`.

Example:

```ts
export async function sendAudioToSession({
  apiBaseUrl,
  sessionId,
  fileUri,
  mimeType = "audio/mpeg",
  filename = "audio.mp3",
  trailingSilence,
}: {
  apiBaseUrl: string;
  sessionId: string;
  fileUri: string; // e.g. file:///.../recording.mp3
  mimeType?: string;
  filename?: string;
  trailingSilence?: number;
}) {
  const qs = trailingSilence != null ? `?trailing_silence=${trailingSilence}` : "";
  const url = `${apiBaseUrl}/sessions/${sessionId}/stream${qs}`;

  const form = new FormData();
  form.append("audio_file", {
    uri: fileUri,
    type: mimeType,
    name: filename,
  } as any);

  const res = await fetch(url, {
    method: "POST",
    body: form,
    // DO NOT set Content-Type manually; fetch will set proper boundary.
  });

  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`Audio upload failed: HTTP ${res.status} ${text}`);
  }
}
```

---

## 🧹 Cleanup (Important in Mobile Apps)

When user closes the chat/player:
- Delete the session to free server resources

```ts
export async function deleteSession(apiBaseUrl: string, sessionId: string) {
  await fetch(`${apiBaseUrl}/sessions/${sessionId}`, { method: "DELETE" });
}
```

---

## 🔒 Multi-User Isolation Rules

Each session has:
- dedicated `session_id`
- independent stream state (`active_stream`)
- its own `chunk_queue` (SSE pushes only your session’s chunks)

This is what allows many users to use the system concurrently.

---

## ✅ Recommended Production Notes

### Networking
- Use HTTPS in production (especially required for many WebView features).
- Ensure the server is reachable from the device (not `localhost`).

### CORS / WebView
- If the WebView is loading from your API domain, SSE and chunk fetches should be same-origin.
- If you embed from a different domain, confirm CORS is configured appropriately server-side.

### Autoplay
- iOS can still require a user gesture in some scenarios.
- Your player already displays “Tap to start” fallback.

---

## 🧪 Debugging Checklist (Mobile)

If you see a black screen / no stream:
1. Verify player URL loads in mobile browser
2. Check SSE connectivity:
   - device can reach `GET /sessions/{id}/events`
3. Check chunk URLs return `200` and are MP4
4. Check backend logs for chunk creation errors
5. Ensure audio upload succeeds (200/202)

---

## Appendix: Minimal Flow Summary

1. `POST /sessions/create` → get `session_id`
2. RN WebView loads `/player/session/{session_id}`
3. RN uploads audio via `/sessions/{session_id}/stream`
4. Player receives SSE `chunk` events → plays video
5. `complete` event → player switches back to base video

---