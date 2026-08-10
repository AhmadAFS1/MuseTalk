# WebRTC long-form speech quality validation — 2026-08-08

## Outcome

MuseTalk does not inherently stop lip-sync at sentence boundaries. Direct and
browser WebRTC runs completed short, multi-sentence, and long-paragraph audio
through the final packet. The reported partial-sentence and mid-call freeze
symptoms came from three orchestration and player-lifecycle defects outside the
core 15-fps lip-sync generator.

## Defects found and fixed

### 1. An audio `422` was mistaken for a pose-contract rejection

Lingua treated every MuseTalk `400`/`422` response as evidence that multipose
metadata was unsupported. A real audio-timeline decode failure was therefore
retried as v1 and then legacy. The retry reused the same turn sequence, so
MuseTalk returned HTTP `200` with `status: ignored`. The mobile client discarded
that JSON body and logged the upload as successful even though no speech stream
had started.

Lingua now falls back only when headers or error content explicitly identify a
pose metadata/plan rejection. Audio decode failures remain failures, and the
mobile client rejects a successful response whose body says `status: ignored`
or omits `request_id`.

### 2. The iOS microphone restarted before assistant playback ended

The upload endpoint accepts a turn after it has validated and queued the audio;
that HTTP response is not playback EOF. `FaceTime.tsx` previously restarted
native speech recognition 150 ms after upload acceptance. On iOS, that can
reconfigure the shared audio session while WebRTC is still playing a long
answer.

The client now reads the returned `request_id` and media duration, polls the
session status, and restarts the microphone only after that exact
`active_stream` clears. It also detects session expiry or another request
replacing the active turn.

### 3. The browser load-test wall detached live iframes every two seconds

The MuseTalk test wall rebuilt its card container with `replaceChildren` on
every metrics refresh. Even when the same iframe objects were reinserted,
detaching them closed their `RTCPeerConnection`. MuseTalk then deleted the
closed sessions, leaving frozen/missing tiles.

The wall now performs a keyed in-place update: stale cards are removed, existing
cards remain attached, and only new or reordered cards are inserted. A focused
test prevents `replaceChildren(...next)` from returning.

## Kokoro installation and performance

Kokoro is installed in `/workspace/.venvs/musetalk_trt_stagewise` and is exposed
by MuseTalk at `/webrtc/tts/kokoro`. The loaded model is
`hexgrad/Kokoro-82M`, running at 24 kHz on CPU.

| Input | Synthesized WAV | Synthesis | RTF |
| --- | ---: | ---: | ---: |
| Short | 3.575 s | 3.247 s | 0.91 |
| Multi-sentence | 17.475 s | 7.022 s | 0.40 |
| Long paragraph | 62.675 s | 24.214 s | 0.39 |

Kokoro currently renders the complete WAV before upload. Its synthesis time is
therefore part of perceived response latency. Once the avatar caches are warm,
MuseTalk itself released the first live block in 0.24–0.34 seconds; the
long-form browser run spent 22.812 seconds in Kokoro before the WebRTC upload.

## Direct MuseTalk validation

All runs used 15-fps MuseTalk generation, 30-fps WebRTC transport, batch size 4,
strict FIFO synchronization, and the Indian tutor multipose manifest.

| Test | Normalized media | Generated frames | Transport result | Waveform tail |
| --- | ---: | ---: | --- | --- |
| Short | 3.140 s | 47 | 0 drops, underruns, or stalls | present; correlation 0.974 |
| Multi-sentence | 16.980 s | 254 | 0 drops, underruns, or stalls | present; correlation 0.965 |
| Long paragraph | 61.920 s | 928 | 0 drops, underruns, or stalls | present; correlation 0.968 |

The short, medium, and long recordings contained 157, 849, and 3,096 20-ms
audio packets respectively—exactly the normalized media duration. Comparing
five evenly spaced source/recording waveform windows, including the final
window, produced correlations above 0.91 in every case. This rules out a
sentence-boundary cutoff or missing final clause in these runs.

The 30-fps transport intentionally duplicates each 15-fps generated lip-sync
frame once. Those duplicates are cadence conversion, not skipped generation.

## Live browser validation

After the iframe lifecycle fix, two peer sessions remained connected across the
two-second wall refresh and both completed an 8.810-second multi-sentence media
timeline. Each emitted 441 20-ms audio packets. The worst first-frame RTP
mismatch was 6.7 ms, worst sender start skew was 8.9 ms, and server video stalls
were zero.

A one-peer 66.250-second browser media timeline then completed all 3,313 audio
packets and 993 generated frames through the final sentence. Results:

- first MuseTalk block ready in 0.34 seconds with warm avatar caches;
- audio/video RTP timestamps exactly aligned at start;
- 17.276 ms wall-clock start skew;
- two four-frame pose crossfades;
- 29-fps browser decode samples during playback;
- zero server frame drops, queue underruns, audio stalls, or video stalls; and
- cleanup only after audio EOF and playback drain.

Chrome's cumulative `droppedVideoFrames` increased rapidly while the iframe was
below the visible viewport, which is browser rendering throttling rather than a
transport failure. With a fresh tile kept visible, the counter increased by 35
during the complete setup and 8.100-second short playback window; decode stayed
at 29 fps and MuseTalk reported zero transport stalls.

## Artifacts

Local copies of the three source WAVs, response headers, and recorded WebRTC
MP4s are under:

`backend/tmp/webrtc_tests/20260808_kokoro_quality_matrix/`

The corresponding server artifacts are under:

`/workspace/MuseTalk/generated/webrtc_quality/2026-08-08/kokoro_duration_matrix/`

The MP4s include the harness's neutral connection pre-roll and post-roll; their
container duration is therefore longer than the normalized TTS media. The
short recording has an unusually long pre-roll because the harness completed
its initial pose demonstration before the short turn. Audio completeness was
measured against the located TTS window, not the full MP4 duration.

## Acceptance criteria for future releases

1. A stream response must contain a non-empty `request_id`; `status: ignored`
   is never success.
2. The microphone must not restart until that request leaves `active_stream`.
3. A non-pose `400`/`422` must not trigger protocol fallback.
4. Test pages must not detach a connected player iframe during polling.
5. Direct long-form validation must show audio EOF at the normalized duration,
   no strict FIFO stalls, and a final waveform window matching the source.
6. Browser validation must keep the peer connected through playback and verify
   the final sentence, not only stream acceptance.
