# MuseTalk Avatar Cache Warmup Handoff

## Short Answer

This is not 100% React Native / EC2-side if you want the first real talking-head response to feel smooth. The UI can show a nice "Calling..." screen, but the MuseTalk server currently has no explicit "warm this avatar into the runtime cache" API. So the clean fix is:

1. MuseTalk server: add an avatar warm/cache-status endpoint.
2. EC2 backend: call that endpoint as soon as the call starts, then keep routing that call to the same MuseTalk worker.
3. React Native: show a polished calling/preparing state until EC2 says the avatar is warm.

## What I Found

The server already restores prepared avatar assets from S3 lazily. `_avatar_exists()` checks local disk, then downloads the prepared avatar tarball from S3 if needed: `scripts/avatar_manager_parallel.py`.

But that only restores the avatar to disk. The actual in-memory MuseTalk cache is created later in `_get_or_load_avatar()`. That loads latents, frames, masks, coords, and compose plans from disk, then stores the avatar in `AvatarCache`.

The session creation endpoints only verify the avatar exists. They do not warm the in-memory cache:

- SSE session create: `POST /sessions/create`
- HLS session create: `POST /hls/sessions/create`
- WebRTC session create: `POST /webrtc/sessions/create`

So the first stream/inference still pays the cache-load cost. For HLS, the scheduler even records `avatar_load_s`, which confirms avatar loading is part of first-stream prep.

## Recommended Flow

```text
React Native user taps Call
  -> EC2 creates call state
  -> EC2 chooses a MuseTalk worker
  -> EC2 calls: POST /avatars/{avatar_id}/cache/warm
  -> RN shows Calling... / Preparing avatar...
  -> EC2 polls: GET /avatars/{avatar_id}/cache/status
  -> once warm, EC2 creates HLS/WebRTC session
  -> RN connects player and enables speaking/audio upload
```

The important part: the same MuseTalk worker must handle warmup and the later session, because this cache is process-local memory. If EC2 warms worker A and sends the stream to worker B, you lose the benefit.

## Server Changes I Would Make

Add something like:

```http
POST /avatars/{avatar_id}/cache/warm?batch_size=2&wait=false
GET  /avatars/{avatar_id}/cache/status
```

The warm endpoint should:

- restore from S3 if needed via existing `_avatar_exists()`
- call `_get_or_load_avatar(avatar_id, batch_size)`
- reuse the existing per-avatar load lock, so duplicate warm requests do not double-load
- return `warming`, `ready`, or `failed`
- expose timings like `s3_restore_seconds`, `avatar_load_seconds`, and `cached`

The status endpoint should tell EC2/RN whether the avatar is:

```text
missing
restoring_from_s3
loading_into_memory
ready
failed
```

The server already has `/stats/cache`, but it is not enough as a clean call-state API.

## EC2 Changes Needed

EC2 should become the call orchestrator:

- pick the MuseTalk worker
- start avatar warmup immediately
- store `call_id -> worker_base_url`
- poll warm status
- tell React Native when the call can move from "Calling..." to connected
- route all later audio/session calls to that same worker

This also gives you a nice place to enforce timeouts, retries, and fallback workers.

## React Native Changes Needed

React Native should mostly be UI/state:

- show avatar image/video placeholder
- show "Calling..." for the first 20-40 seconds
- update copy based on backend status, e.g. "Connecting", "Preparing avatar", "Almost ready"
- do not let the user send the first audio until EC2 says the avatar cache is warm
- handle timeout/retry cleanly around 60-90 seconds

## Can You Do It Without MuseTalk Server Changes?

Only as a workaround. RN/EC2 could show a calling screen while the first real `/stream` request blocks, but the user would still wait after speaking. Or EC2 could trigger a fake/silent inference to warm the cache, but that wastes GPU and complicates behavior.

So the recommendation is: small MuseTalk server change, modest EC2 orchestration change, polished RN calling UI. That gives the smooth experience without abusing the inference path.
