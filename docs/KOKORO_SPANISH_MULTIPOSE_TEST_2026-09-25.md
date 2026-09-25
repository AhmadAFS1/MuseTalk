# Kokoro Spanish idle → talking → idle pilot — 2026-09-25

This checkpoint answers two separate questions: whether every physical motion
clip has its own MuseTalk preparation, and whether one short Kokoro sentence can
be received over WebRTC while the body enters and leaves the talking source.
Both live recordings passed the transport and pose-trace verifier. Visual review
is still a candidate review, not general approval of arbitrary transitions.

## Separate `/avatars/prepare` caches

Yes. The Latina package contains three physical source videos and three distinct
MuseTalk cache IDs. They were originally prepared through the real avatar API by
`ensure_six_avatars`/`POST /avatars/prepare`; public reaction aliases resolve to
these three physical caches instead of sharing one idle preparation.

| Physical role | Prepared avatar ID | Input MP4 SHA-256 |
| --- | --- | --- |
| idle | `latina_guided_20260925_idle_ee675cb4fd` | `ee675cb4fdc230f1a67f995d3a22975c1954b0238b8c34a57827eec23379a14c` |
| talking | `latina_guided_20260925_talking_84c5bc80b8` | `84c5bc80b858c864e5a4578f19eb37c66da4041de0ffe07d8b1e22f6d808bae0` |
| smiling | `latina_guided_20260925_smiling_a28498e053` | `a28498e05356de4eba8e770ca85483db33bd5e4266491ae3c9586dc9624a12c9` |

Each directory has its own `input_video.mp4`, `latents.pt`, `coords.pkl`, decoded
frames and masks. The three source hashes in the cache directories match the
packaged source hashes exactly. Both new live runs called the preparation helper
with `prepare_missing=true`; all three responses were `ready`, `cached=true` and
`disk_prepared=true`, so no cache was silently substituted or rebuilt.

The same three-cache preparation records exist for the Japanese and Indian
packages in their v6 evidence directories. This pilot uses Latina because it is
the latest fresh-character package.

## TTS input and the three-second rule

Exact text: `Hola, como estas? Me alegra que estas aqui. `

The repository-pinned Kokoro 0.9.4 package had been removed from the active
virtual environment even though the endpoint and old documentation remained.
For this pilot it was restored only under `/dev/shm`, reusing the already cached
`hexgrad/Kokoro-82M` model and existing SoulX English phonemizer dependencies.
No package was installed into the nearly full root filesystem. Synthesis ran on
CPU with network access disabled.

The installed application wrapper supports only language codes `a` and `b`.
These files therefore use the US `af_heart` voice and English phonemization for
Spanish text; they test MuseTalk motion/lipsync, not native Spanish TTS quality.

| Input | Kokoro speed | Duration | SHA-256 | Runtime policy |
| --- | ---: | ---: | --- | --- |
| `kokoro-default-speed.wav` | 1.0 | 1.550 s | `cee25fbf450ddd29aaa606a967135382e9ca3f160c1fb48e204b753f57a09aee` | Production regards it as short and retains idle. |
| `kokoro-talking-threshold.wav` | 0.5 | 3.075 s | `727263baf379f5ac9010937f784b3762bdfa6e83a50ac8ac71628e5ed6a3e9f9` | Crosses the unchanged three-second threshold. |

Both are mono PCM16 WAV at 24 kHz. The slower file exists because the user asked
to see idle → talking → idle with this exact sentence, while the existing product
requirement says sub-three-second replies should remain on idle.

## Actual received recordings

### Natural-speed forced-transition diagnostic

[Received MP4](/workspace/experiments/multipose_kokoro_spanish_20260925/natural-live/talking-only.mp4) ·
[review page](/workspace/experiments/multipose_kokoro_spanish_20260925/natural-live/review.html) ·
[transition sheet](/workspace/experiments/multipose_kokoro_spanish_20260925/natural-transition-review.jpg) ·
[strict verification](/workspace/experiments/multipose_kokoro_spanish_20260925/natural-live/verification.json)

This is the most useful lipsync-quality sample because Kokoro stays at speed 1.0.
A separate diagnostic atlas changes only `short_reply_seconds` from 3 to 0. It
does not alter sources, prepared caches, transition edges, blend code or the
production atlas. Its compressed exact copy is retained as
`threshold-zero-atlas.json.gz`.

- Audio: 1.550 seconds, 31 generated frames.
- Body switch: idle frame 68 → talking frame 134 at generation frame 6.
- Terminal switch: talking frame 150 → idle frame 128 at generation frame 21.
- Received file: 145 video frames, exact 50 ms cadence, no missing/anomalous RTP timestamps.
- Entry preparation/bridge: 0.2953 seconds.
- Final recovery: 0.4000 seconds.
- Video SHA-256: `02783a0a73df50fa1f769c442b7454041906081b072b96e83b245caccb30e2a4`.

### Production-threshold transition

[Received MP4](/workspace/experiments/multipose_kokoro_spanish_20260925/live/talking-only.mp4) ·
[review page](/workspace/experiments/multipose_kokoro_spanish_20260925/live/review.html) ·
[transition sheet](/workspace/experiments/multipose_kokoro_spanish_20260925/transition-review.jpg) ·
[strict verification](/workspace/experiments/multipose_kokoro_spanish_20260925/live/verification.json)

This run uses the unchanged three-second policy and the 3.075-second 0.5× audio.

- Audio: 3.075 seconds, 61 generated frames.
- Body switch: idle frame 66 → talking frame 12 at generation frame 6.
- Terminal switch: talking frame 64 → idle frame 128 at generation frame 51.
- Received file: 173 video frames, exact 50 ms cadence, no missing/anomalous RTP timestamps.
- Entry preparation/bridge: 0.2959 seconds.
- Final recovery: 0.3999 seconds.
- Video SHA-256: `a3c3077ed30792ac2f7d05f3227f44d0a68aa25bc51935692553711a2964c431`.

## Quality finding and limits

The 24 inspected received frames around each switch show stable framing and
shoulders, one coherent mouth, and no obvious camera-distance or head-position
jump. No doubled-nostril failure is visible in these selected source pairs. Both
files contain the actual received audio and video; no post-recording smoothing or
dubbed audio was applied.

This does not establish that every arbitrary source phase is seamless. The
current-phoneme atlas remains unreviewed, previous difficult nasal transitions
remain retained, and a still sheet does not substitute for the user's normal-speed
viewing decision. The natural-speed run intentionally overrides only the duration
threshold and must not be cited as production short-reply policy.

## Reproduction

`scripts/test_webrtc_motion_sentence.py` is the reusable single-audio recorder.
