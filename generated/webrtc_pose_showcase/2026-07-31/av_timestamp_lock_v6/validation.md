# Indian tutor WebRTC A/V timestamp-lock v6 validation

## Outcome: superseded / failed end-of-speech timing

This run fixed the 23-second trailing-silence failure and produced a valid mux,
but an independent RTP/content-timeline review found that it still returned to
`neutral_resting` too early. Do not use v6 as final approval evidence.

- Live speaking begins at approximately 3.467 seconds.
- The first neutral-resting frame appears at 14.300 seconds.
- The final audible phrase runs from 14.330–14.663 seconds.
- Neutral therefore begins roughly 363 ms before audible speech finishes.
- Correlation to the normalized source places its 11.140-second media endpoint
  at approximately 14.782 seconds, about 482 ms after the neutral switch.

Root cause: producer-side audio progress can catch up after event-loop lateness,
while the receiver still plays its contiguous RTP timestamps at normal speed.
Video was ending on producer EOF instead of occupying the matching audio media
horizon in its own RTP/output timeline. This is corrected in the next run.

## Rejected capture diagnosis

Source:
`semantic_sync_fix_v1/indian_tutor_semantic_sync_fix_v1_webrtc_capture.mp4`

- MP4 duration: 39.098 seconds.
- Detectable speech: 4.336–15.989 seconds.
- Silent tail after speech: 23.109 seconds.
- Video: 1,132 frames at 30 fps, 37.733 seconds.
- Audio: 39.097 seconds.
- AAC timing damage: 19 inflated packets, adding 3.877486 seconds; largest
  single excess duration was 1.922666 seconds.
- The uploaded 26.35-second WAV contained 15.21 seconds of trailing silence.
  That entire file was treated as live media, so MuseTalk remained active long
  after audible TTS ended.
- Replacing/restarting the audio sender for a turn also broke the continuous
  RTP timestamp history and made the capture gap worse.

## Runtime changes exercised by v6

- Detect sustained speech and normalize long leading/trailing edge silence
  once. The exact same normalized WAV is used by MuseTalk generation and
  WebRTC audio playout.
- Keep one persistent audio sender for the peer connection:
  idle silence → TTS → idle silence, with contiguous 20 ms timestamps.
- Arm audio and video behind one release gate.
- Use emitted audio media time as the master clock. Video selects the generated
  source frame for that audio timestamp instead of maintaining an independent
  playback clock.
- Treat the packet after the final TTS packet as the completion point, discard
  any queued live frames, and activate a predecoded neutral frame immediately.
- Retain explicit per-turn generation ownership so callbacks from an older turn
  cannot enter or complete a newer turn.

## v6 server telemetry

- Normalized media duration: 11.140 seconds.
- Speech bounds in normalized media: 0.020–11.020 seconds.
- Removed trailing silence: 15.210 seconds.
- First video frame after shared release: 83.985 ms.
- First audio packet after shared release: 98.178 ms.
- Initial A/V start delta: 14.193 ms (less than one 30 fps frame).
- Audio media/playout time: 11.140 / 11.140 seconds.
- Generated coverage: 167 frames at 15 fps = 11.133 seconds.
- Generated frames played: 167 of 167.
- Audio stalls: 0; video stalls: 0.
- Final state: stream inactive, pose `neutral_resting`.
- Pose-plan semantic drift: 0 frames for all three requested segments.

## v6 recorded-file validation

Raw source:
`indian_tutor_av_timestamp_lock_v6_webrtc_capture.mp4`

- Video: 461 frames, exactly 30 fps, 15.367 seconds.
- Audio: 742 AAC packets, 15.821 seconds.
- Recorded track start delta: 34 ms (approximately one video frame).
- Inflated AAC packets: 0; excess packet duration: 0 seconds.
- Detectable speech: 3.560–14.562 seconds.
- Recorded post-speech silence: 1.301 seconds.
- Packet timing is structurally valid, but content inspection shows the
  neutral-resting asset begins before the final audible phrase completes.

The labeled copy uses the recorded speech bounds only for the viewer overlay.
The raw capture remains the forensic source of truth.

## Automated verification

- Full repository unit suite: 70/70 passed.
- Focused WebRTC sync/pose suite: 51/51 passed.
- Python compile checks: passed.
- `git diff --check`: passed.
