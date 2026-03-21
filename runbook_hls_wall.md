# HLS Wall Runbook

This runbook shows the fastest way to open many HLS avatar sessions in the browser at once and start them all with one audio upload.

The wall flow is useful when you want to:

- watch multiple sessions live in the browser
- visually inspect smoothness, stalls, lip sync, and startup timing
- compare real user experience against load-test numbers

## What This Uses

Relevant routes:

- `GET /hls/lab`
- `POST /hls/groups/create`
- `GET /hls/groups/{group_id}`
- `GET /hls/groups/{group_id}/wall`
- `POST /hls/groups/{group_id}/stream`
- `DELETE /hls/groups/{group_id}`

Related files:

- [api_server.py](/content/MuseTalk/api_server.py)
- [templates/hls_wall.py](/content/MuseTalk/templates/hls_wall.py)
- [http_scripts_hls.http](/content/MuseTalk/http_scripts_hls.http)

## Quick Start

### 1. Start the API server

Example:

```bash
python api_server.py --host 0.0.0.0 --port 8000
```

If you normally run with custom env vars, start the server the same way you usually do.

### 2. Open the lab page

In your browser:

```text
http://localhost:8000/hls/lab
```

### 3. Create a group

In the lab UI:

- set `avatar_id`
- set the session count, for example `6` or `8`
- set `playback_fps`, `musetalk_fps`, `batch_size`, and `segment_duration`
- click `Create Group`

Notes:

- the current API only allows `count` from `1` to `12`
- after creation, the page shows a grid of HLS players, one per session

### 4. Start all sessions with one audio file

In the same wall page:

- choose one audio file
- click `Start All`

That sends the same uploaded audio to every session in the group.

### 5. Watch the sessions live

Each tile is a normal HLS player for one session.

This is the easiest way to visually check:

- startup timing
- live transition from idle to talking
- segment stutter
- whether some sessions lag behind others

## API-Only Flow

If you prefer to drive the wall without using the lab page:

### 1. Create a group

Example request:

```http
POST http://localhost:8000/hls/groups/create?avatar_id=test_avatar&count=6&playback_fps=24&musetalk_fps=12&batch_size=2&segment_duration=2
```

The response includes:

- `group_id`
- `wall_url`
- `stream_all_url`

### 2. Open the wall in the browser

```text
http://localhost:8000/hls/groups/{group_id}/wall
```

Replace `{group_id}` with the real group id.

### 3. Start all sessions

Example request:

```http
POST http://localhost:8000/hls/groups/{group_id}/stream
Content-Type: multipart/form-data
```

Attach one `audio_file`.

You can copy the ready-made example from [http_scripts_hls.http](/content/MuseTalk/http_scripts_hls.http).

### 4. Check group status

```http
GET http://localhost:8000/hls/groups/{group_id}
```

This returns current session ids and session statuses.

### 5. Delete the group when done

```http
DELETE http://localhost:8000/hls/groups/{group_id}
```

This is the cleanup step you should use after a wall test.

## Example Browser Workflow

Use this when you want the simplest repeatable flow:

1. Start server
2. Open `http://localhost:8000/hls/lab`
3. Create group with `count=8`
4. Upload `./data/audio/ai-assistant.mpga`
5. Click `Start All`
6. Watch all 8 sessions
7. When done, delete the group from the UI or with the `DELETE` API

## Example API Workflow

Use this when you want more control or want to save requests in your HTTP client:

1. Run the `Create HLS Group` request from [http_scripts_hls.http](/content/MuseTalk/http_scripts_hls.http)
2. Copy the returned `group_id`
3. Open `/hls/groups/{group_id}/wall` in the browser
4. Run the `Start All Sessions In Group With One Audio Upload` request
5. Watch the wall in the browser
6. Run `Delete HLS Group`

## Troubleshooting

### The wall page opens, but nothing is talking

Check:

- the group was created successfully
- you uploaded a real audio file to `POST /hls/groups/{group_id}/stream`
- the session statuses from `GET /hls/groups/{group_id}`

### Some players stay idle or lag

That usually means the backend is overloaded or one stream is behind in live chunk production.

Useful checks:

- [load_test.py](/content/MuseTalk/load_test.py) for benchmark comparison
- `GET /hls/sessions/stats`
- `GET /stats`

### I want to test realistic user arrivals instead of all-at-once starts

Use the wall for visual inspection, and use [load_test.py](/content/MuseTalk/load_test.py) with `--stagger-seconds` for the measured benchmark.

### I only want to see one session

Open the normal single-session player:

```text
http://localhost:8000/hls/player/{session_id}
```

## Recommended Defaults

If you want a known-good wall test starting point, use:

- `playback_fps=24`
- `musetalk_fps=12`
- `batch_size=2`
- `segment_duration=2`
- `count=6` or `count=8`

For stricter latency testing, you can reduce `segment_duration` to `1`, but that is a harder workload.
