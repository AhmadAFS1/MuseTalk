# Two-Video Avatar Prep Handoff

## Summary

MuseTalk avatar preparation now supports two base videos for one avatar:

- `video_file`: the talking/bobbing source used by MuseTalk to build frames, masks, coordinates, and latents.
- `idle_video_file`: optional idle loop used by HLS/WebRTC while no audio is active.

If `idle_video_file` is not sent, the server keeps the old behavior and uses
`video_file` for both MuseTalk prep and idle playback.

## Prepare API

Endpoint:

```http
POST /avatars/prepare?avatar_id=<avatar_id>&batch_size=20&bbox_shift=0&force_recreate=true
Content-Type: multipart/form-data
```

Multipart fields:

```text
video_file       required MP4; talking/bobbing MuseTalk source
idle_video_file  optional MP4; idle loop shown before/after speech
```

Example:

```bash
curl -X POST "http://localhost:8000/avatars/prepare?avatar_id=test_avatar&batch_size=20&bbox_shift=0&force_recreate=true" \
  -F "video_file=@data/video/chatgpt_moving_vid.mp4" \
  -F "idle_video_file=@data/video/ai_test_default_moving_vid.mp4"
```

Response includes:

```json
{
  "status": "success",
  "avatar_id": "test_avatar",
  "video_layout": "separate_idle_talking",
  "message": "Avatar test_avatar prepared successfully"
}
```

For old single-video prep, `video_layout` is `single_video`.

## Stored Avatar Artifacts

After preparation with a separate idle source, each avatar directory contains:

```text
input_video.mp4   talking/bobbing source used for MuseTalk prep
idle_video.mp4    idle playback loop
latents.pt
coords.pkl
mask_coords.pkl
full_imgs/
mask/
avator_info.json
```

`avator_info.json` records:

```json
{
  "input_video_path": ".../input_video.mp4",
  "talking_video_path": ".../input_video.mp4",
  "idle_video_path": ".../idle_video.mp4",
  "video_layout": "separate_idle_talking"
}
```

For single-video prep, the server does not create a duplicate `idle_video.mp4`.
`idle_video_path` points at `input_video.mp4`, and `video_layout` is
`single_video`. S3 persistence stores the whole avatar directory and skips any
redundant single-video idle copy from older local artifacts, so the avatar
tarball does not bloat with two copies of the same MP4. Older S3 avatars without
`idle_video.mp4` still work because the server falls back to `input_video.mp4`.

## Playback Behavior

HLS and WebRTC session creation now resolve the idle video like this:

1. Prefer `idle_video.mp4`.
2. Fall back to `input_video.mp4` for older prepared avatars.

Live talking frames are still generated from the prepared MuseTalk frame cycle
that came from `video_file`.

For the current single-video flow, WebRTC now anchors live generation to the
idle frame that is already on screen. When a stream starts, the server captures
the displayed idle source-frame index, starts MuseTalk from the matching prepared
cycle frame, and holds that idle pose while live frames prebuffer. This avoids
the old jump where WebRTC live generation always started at frame zero.

Useful debug endpoint:

```http
GET /avatars/<avatar_id>/video
```

This returns the idle loop by default. To inspect the talking source:

```http
GET /avatars/<avatar_id>/video?role=talking
```

## Mobile Lingua Integration

When creating or replacing an avatar from Mobile Lingua, send:

- idle/base-resting video as `idle_video_file`
- bobbing/talking-base video as `video_file`

Then create HLS/WebRTC sessions exactly as before:

```http
POST /hls/sessions/create?avatar_id=<avatar_id>&...
POST /webrtc/sessions/create?avatar_id=<avatar_id>&...
```

No session API change is required. The server chooses the right idle asset from
the prepared avatar directory.

## Important Caveat

HLS server timing and WebRTC idle-frame anchoring can map the idle loop position
onto the MuseTalk frame cycle. For single-video avatars this is the intended
smooth path because idle and talking share the same source. If the idle and
talking videos have different motion, length, or timing, body continuity is
still approximate. The main guarantee is:

- idle mode visually uses the idle video
- live mode visually uses MuseTalk output based on the talking/bobbing source
- old one-video avatars remain compatible

## Changed Files

Core implementation:

- `api_server.py`
- `scripts/api_avatar.py`
- `scripts/avatar_manager_parallel.py`
- `scripts/avatar_s3_store.py`
- `scripts/hls_gpu_scheduler.py`
- `scripts/webrtc_manager.py`
- `scripts/webrtc_tracks.py`

Docs/examples:

- `http_scripts_streaming.http`
- `api_calls.md`
- `HLS_REACT_NATIVE_README.md`
- `WEBRTC_REACT_NATIVE_README.md`
- `docs/hls_migration.md`
- `docs/musetalk_model_pipeline_breakdown.md`
- `docs/avatar_s3_persistence.md`

Verification run:

```bash
python3 -m py_compile api_server.py scripts/api_avatar.py scripts/avatar_manager_parallel.py scripts/avatar_s3_store.py scripts/hls_gpu_scheduler.py scripts/webrtc_manager.py scripts/webrtc_tracks.py test_avatar_cache_warmup.py
python3 -m unittest test_avatar_s3_store.py test_avatar_cache_warmup.py
git diff --check
```
