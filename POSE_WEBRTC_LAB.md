# MuseTalk six-pose WebRTC lab

This lab runs directly on a MuseTalk worker. It does not call LTX, Segmind,
Kling, Gemini TTS, or any other generation provider. Its default is the local
close-up LTX 2.3 production bank and an existing WAV/MP3 file.

The default bank is `switch_safe: true`: all six unique physical clips share
the same six-frame decoded boundary at both ends. It is the local migration
default, while final visual approval remains a separate human review step.

## Files

- `templates/webrtc_pose_lab.py` — self-contained browser UI.
- `scripts/test_pose_webrtc.py` — six-cache preparation and headless WebRTC
  smoke test.
- `configs/pose_test/sample_ai_human_ltx23_facetime_closeup_production_v1.json` —
  approved prepared avatar IDs, local filenames, direct-speaking variants, and
  media timing.
- `assets/ltx23_pose_banks/sample_ai_human_facetime_closeup_production_v1/` —
  immutable source image, certified MP4s, hashes, and validation report.

The exact pose IDs are:

1. `neutral_resting`
2. `active_listening`
3. `speaking_direct`
4. `nod_agree`
5. `empathetic_head_tilt`
6. `light_smile`

## One-time API integration

Add the template import beside the existing WebRTC template imports in
`api_server.py`:

```python
from templates.webrtc_pose_lab import get_webrtc_pose_lab_html
```

Add these two routes after the WebRTC routes. `POSE_LAB_SAMPLE_AUDIO` lets the
server use any existing sample TTS file without generating new audio:

```python
@app.get("/webrtc/pose-lab", response_class=HTMLResponse)
async def webrtc_pose_lab():
    return HTMLResponse(content=get_webrtc_pose_lab_html())


@app.get("/webrtc/pose-lab/sample-audio")
async def webrtc_pose_lab_sample_audio():
    sample_path = Path(
        os.getenv("POSE_LAB_SAMPLE_AUDIO", "./data/audio/eng.wav")
    ).expanduser()
    if not sample_path.is_file():
        raise HTTPException(
            status_code=404,
            detail=(
                "Set POSE_LAB_SAMPLE_AUDIO to an existing WAV/MP3, "
                "or choose a local file in the lab."
            ),
        )
    return FileResponse(sample_path)
```

The page expects the pose-protocol worker implementation to expose:

- `GET /capabilities` with `features.pose_sets_v1=true`;
- pose-aware `POST /webrtc/sessions/create`;
- `POST /webrtc/sessions/{id}/events`;
- `POST /webrtc/sessions/{id}/pose`;
- pose metadata on `POST /webrtc/sessions/{id}/stream`;
- pose state on `GET /webrtc/sessions/{id}/status`.

## Local assets

The six unique physical MP4s are already in the repository-local asset bank.
Neutral and active listening share one cache. Direct speech has V14 and V15
physical variants under the single semantic `speaking_direct` pose, so the
preparation command creates six physical caches from six files.

Use an existing sample audio file in `data/audio`, or pass any local WAV/MP3
with `--audio-file`.

## Prepare and warm all six stable avatar IDs

Run this inside the MuseTalk virtual environment on the worker:

```bash
cd /workspace/MuseTalk
python scripts/test_pose_webrtc.py \
  --base-url http://127.0.0.1:8000 \
  --prepare-missing \
  --prepare-only
```

The manifest and asset-directory arguments default to the close-up production
bank. `--prepare-missing` uploads only absent IDs; existing byte-verified
prepared caches are warmed and reused. Do not use `--force-recreate` during
normal rollout because the prepared directories are part of the rollback path.

## Run the headless end-to-end session

This path performs an SDP offer/answer, consumes WebRTC video and audio, sends
ordered conversation events, exercises the explicit pose queue, uploads the
sample audio with deterministic pose metadata, polls completion, and deletes
the session:

```bash
python scripts/test_pose_webrtc.py \
  --base-url http://127.0.0.1:8000 \
  --audio-file /workspace/MuseTalk/data/audio/eng.wav \
  --reaction-intent warmth
```

A passing result reports nonzero `video_frames_received` and
`audio_frames_received`, a completed stream, and the final pose.

## Watch and control the test in a browser

Set the bundled sample path before server startup if desired:

```bash
export POSE_LAB_SAMPLE_AUDIO=/workspace/MuseTalk/data/audio/eng.wav
```

Open:

```text
http://50.40.184.100:60525/webrtc/pose-lab
```

Use **Create + connect**, then:

- click any of the six pose buttons to exercise `/pose`;
- click **Queue all six poses** to watch the complete boundary-ordered library;
- use the conversation buttons to exercise `/events`;
- choose a local audio file, or leave it empty to use the bundled sample;
- click **Stream selected / bundled sample** to exercise `/stream`;
- watch current server state update through `/status`.

The sample stream sequence is fixed:

```text
optional reaction → speaking_direct → neutral_resting
```

Audio uses lip sync immediately; body-pose changes are requested at
`next_boundary`.
