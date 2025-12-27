# React Native WebRTC Integration (MuseTalk)

This guide shows how to integrate MuseTalk WebRTC into a React Native app to build a callAnnie-like experience. It covers the recommended WebView approach and an advanced native WebRTC option.

---

## High-level Flow

1. Create a WebRTC session (`POST /webrtc/sessions/create`).
2. Open the WebRTC player (WebView or native WebRTC).
3. Generate TTS audio on device/server.
4. Upload audio to the active WebRTC session (`POST /webrtc/sessions/{id}/stream`).
5. Repeat step 4 for each new utterance.

The server keeps an idle video loop until audio arrives, then switches to live video and audio.

---

## Recommended: WebView Player + HTTP Audio Upload

This is the most stable path (especially on iOS/Safari). The server hosts the player HTML and handles SDP/ICE internally.

### 1) Create WebRTC Session

```http
POST /webrtc/sessions/create?avatar_id=test_avatar&user_id=user1&fps=10&playback_fps=15&batch_size=2&chunk_duration=2
```

Response includes `session_id` and `player_url`.

### 2) Open Player in WebView

Set the WebView URL to:

```
${BASE_URL}${player_url}
```

The player auto-negotiates WebRTC. Users must tap once to enable audio (iOS/Safari policy).

### 3) Upload Audio (TTS output)

```http
POST /webrtc/sessions/{session_id}/stream
Content-Type: multipart/form-data
```

Form field name must be `audio_file`.

### React Native Example (WebView + fetch)

```tsx
const BASE_URL = "http://YOUR_HOST:PORT";

async function createSession() {
  const url = `${BASE_URL}/webrtc/sessions/create?avatar_id=test_avatar&user_id=user1&fps=10&playback_fps=15&batch_size=2&chunk_duration=2`;
  const res = await fetch(url, { method: "POST" });
  if (!res.ok) throw new Error("Create session failed");
  return res.json(); // { session_id, player_url, ice_servers, ... }
}

async function sendAudio(sessionId: string, fileUri: string) {
  const form = new FormData();
  form.append("audio_file", {
    uri: fileUri,
    type: "audio/mpeg",
    name: "tts.mpga",
  } as any);

  const res = await fetch(`${BASE_URL}/webrtc/sessions/${sessionId}/stream`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error("Audio upload failed");
}
```

Notes:
- Use a real device file URI (e.g., from `react-native-fs` or `expo-file-system`).
- The server converts audio to 48k mono PCM internally (FFmpeg + soxr).

---

## Advanced: Native WebRTC (react-native-webrtc)

If you want full native control (no WebView), you must implement SDP and ICE signaling yourself.

### Required steps

1. `POST /webrtc/sessions/create` → get `ice_servers`.
2. Create an `RTCPeerConnection` with those ICE servers.
3. Add recvonly transceivers for audio/video.
4. Create SDP offer → POST to `/webrtc/sessions/{id}/offer`.
5. Set remote description from the answer.
6. Send ICE candidates to `/webrtc/sessions/{id}/ice`.

Pseudo-flow:

```ts
const pc = new RTCPeerConnection({ iceServers });
pc.addTransceiver("video", { direction: "recvonly" });
pc.addTransceiver("audio", { direction: "recvonly" });

pc.onicecandidate = ({ candidate }) => {
  if (candidate) {
    fetch(`${BASE_URL}/webrtc/sessions/${id}/ice`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(candidate),
    });
  }
};

const offer = await pc.createOffer();
await pc.setLocalDescription(offer);

const answer = await fetch(`${BASE_URL}/webrtc/sessions/${id}/offer`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
}).then(r => r.json());

await pc.setRemoteDescription(answer);
```

After the connection is established, upload audio via `/webrtc/sessions/{id}/stream` as shown above.

---

## Sync / Quality Tuning

Recommended defaults:
- `fps=10`
- `playback_fps=15`
- `batch_size=2`
- `chunk_duration=2`

If audio drifts:
- Keep `fps` at a value your GPU can sustain.
- Increase only `playback_fps` if you want smoother motion.
- Avoid large `batch_size` unless you can keep real-time FPS.

Server-side sync knobs (env vars):
- `WEBRTC_AUDIO_PREBUFFER_SECONDS` (default `0.2`)
- `WEBRTC_AUDIO_MAX_LEAD_SECONDS` (default `0.08`)
- `WEBRTC_AUDIO_MAX_LAG_SECONDS` (default `0.12`)

Reference setup: 20 FPS / 20 FPS playback (batch_size=8 on 8GB GPU)
```
POST /webrtc/sessions/create?avatar_id=test_avatar&user_id=user1&fps=20&playback_fps=20&batch_size=8&chunk_duration=2

export WEBRTC_TURN_URLS="turn:195.142.145.66:12885?transport=udp,turn:195.142.145.66:12964?transport=tcp"
export WEBRTC_TURN_USER="webrtc"
export WEBRTC_TURN_PASS="CHANGE_THIS_PASSWORD"
export WEBRTC_AUDIO_PREBUFFER_SECONDS=0.15
# optional: force relay-only
export WEBRTC_STUN_URLS=""
```

---

## Common Errors

- **404 on `/webrtc/sessions/{id}/stream`**: session expired or wrong ID.
- **409 on `/webrtc/sessions/{id}/stream`**: stream already active; wait until it finishes.
- **Silent audio on iOS**: user must tap once to unlock audio.

---

## Production Notes

- Use HTTPS for real devices and Safari.
- Keep the WebView or RTCPeerConnection alive while the call is active.
- Use TURN for reliable mobile connections (set `WEBRTC_TURN_URLS`, user, pass).
