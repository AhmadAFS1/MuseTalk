# Lingua Avatar Recreation Handoff

## Goal

When Lingua tries to create a MuseTalk session and MuseTalk returns a structured
`404` saying the avatar is missing, Lingua should recreate the avatar from its
canonical source video, then retry session creation.

MuseTalk should remain the runtime/artifact service. Lingua should remain the
source-of-truth app that knows which original video files belong to a user/avatar.

## MuseTalk Contract

Session creation endpoints now try to recover the avatar before failing:

- local prepared avatar directory
- S3 restore through `AVATAR_S3_ENABLED`
- retained upload rebuild if MuseTalk still has source files in `uploads/videos`

If all recovery paths fail, MuseTalk returns:

```json
{
  "detail": {
    "code": "avatar_not_found",
    "avatar_id": "avatar_123",
    "recreate_required": true,
    "message": "Avatar 'avatar_123' was not found locally or in S3. Recreate it with /avatars/prepare before creating a session."
  }
}
```

This can happen on:

- `POST /sessions/create`
- `POST /hls/sessions/create`
- `POST /webrtc/sessions/create`

Lingua should key off `status === 404` and
`response.detail.code === "avatar_not_found"`.

## Lingua Flow

1. Try normal MuseTalk session creation.
2. If MuseTalk returns structured `avatar_not_found`, look up the avatar's
   canonical source video in Lingua storage.
3. Call MuseTalk avatar preparation:

```text
POST /avatars/prepare?avatar_id=<avatar_id>&batch_size=<batch_size>&bbox_shift=0&force_recreate=true
Content-Type: multipart/form-data

video_file=<canonical talking/source video>
idle_video_file=<optional idle video, if Lingua stores one>
```

4. After prepare succeeds, retry the original session-create request once.
5. If the retry still returns `avatar_not_found`, surface a real failure and do
   not loop indefinitely.

## Important Details

- Use the same `avatar_id` Lingua originally sent to session creation.
- Pass `force_recreate=true` so MuseTalk replaces broken or partial local state.
- Preserve the original session request shape when retrying, especially:
  `batch_size`, `fps`, `playback_fps`, `musetalk_fps`, `segment_duration`, and
  WebRTC/HLS mode.
- If Lingua has separate talking and idle videos, send talking as `video_file`
  and idle as `idle_video_file`.
- If Lingua only has one video, send it as `video_file`; MuseTalk will use the
  single-video avatar layout.
- Treat non-`avatar_not_found` 404s differently. For example, a missing session
  id or missing output file should not trigger avatar recreation.

## Suggested Lingua Pseudocode

```ts
async function createMuseTalkSessionWithAvatarRecovery(request) {
  const first = await createMuseTalkSession(request);
  if (first.ok) return first.data;

  const detail = first.error?.detail;
  const shouldRecreate =
    first.status === 404 &&
    detail?.code === "avatar_not_found" &&
    detail?.recreate_required === true;

  if (!shouldRecreate) {
    throw first.error;
  }

  const source = await loadCanonicalAvatarSource(request.avatarId);
  await prepareMuseTalkAvatar({
    avatarId: request.avatarId,
    talkingVideo: source.talkingVideo,
    idleVideo: source.idleVideo,
    batchSize: request.batchSize,
    forceRecreate: true,
  });

  const retry = await createMuseTalkSession(request);
  if (retry.ok) return retry.data;
  throw retry.error;
}
```

## Validation Checklist

- Delete or hide a prepared avatar locally and from S3.
- Ask Lingua to create a WebRTC/HLS session for that avatar.
- Confirm Lingua receives `avatar_not_found`.
- Confirm Lingua calls `/avatars/prepare` with the canonical source video.
- Confirm the original session creation succeeds on exactly one retry.
- Confirm repeated failure does not create an infinite recreate loop.

