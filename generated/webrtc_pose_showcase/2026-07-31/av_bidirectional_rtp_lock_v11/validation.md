# WebRTC A/V sync validation — v11

## Verdict

PASS. The receiver-visible first TTS packet and first MuseTalk live frame differ by 13.333 ms, below one 30 fps frame (33.333 ms). Neutral idle begins 9.550 ms after the final -45 dB audible speech sample. There is no delayed lip-sync start and no lingering lip motion after audible speech.

## Fixes exercised by this recording

- The WebRTC session keeps one persistent audio RTP track across idle, TTS, and idle instead of replacing the sender track per turn.
- Audio is normalized once. The same trimmed PCM is used by MuseTalk inference and WebRTC audio transport.
- The 26.35 s input contained speech only through 11.02 s. The shared timeline is now 11.06 s, including a deliberate 40 ms tail, so 15.29 s of trailing silence no longer extends lip sync.
- Audio and live video wait on one release gate.
- The first live video RTP timestamp is matched to the persistent audio sample clock. The actual first TTS transport PTS is recorded and validated, rather than inferring sync from callback wall time.
- If audio is ahead, video advances to it. If video is ahead by more than one frame, only the still-silent audio prefix may advance. Neither RTP clock ever moves backward.
- The first neutral frame is decoded before playout and activated only after the complete receiver-visible audio media horizon.
- Multipose cues remain on the audio-progress timeline; this run rendered every requested cue with zero semantic drift.

## In-run RTP proof

| Measurement | Result |
|---|---:|
| Actual first TTS RTP PTS | 4.620000 s |
| First live video RTP PTS | 4.633333 s |
| Absolute first-frame mismatch | 13.333 ms |
| Allowed limit | 33.333 ms |
| Normalized audio media | 11.060000 s |
| Required live output frames | 332 |
| Actual live output frames | 332 |
| Generated / duplicated frames | 165 / 167 |
| Audio RTP anomalies | 0 |
| Undeclared video RTP anomalies | 0 |
| Queue underruns / A/V stalls | 0 / 0 |
| Completion state | neutral_resting; stream inactive |

There is exactly one declared video timestamp step at the first live frame: four correction frames (133.333 ms) plus the normal 33.333 ms interval. The recorder found that exact step at the exact declared RTP PTS and rejected all other anomaly patterns.

## Independent MP4 measurement

The raw capture contains 505 decoded video frames and 811 decoded AAC frames.

| Receiver-visible event | Absolute MP4 PTS |
|---|---:|
| First live/lip-sync frame | 4.766341 s |
| Last live/lip-sync frame | 15.799674 s |
| First neutral-idle frame | 15.833008 s |
| Decoded source-media sample zero | 4.785833 s |
| Audible speech onset (-45 dB) | 4.797188 s |
| Audible speech end (-45 dB) | 15.823458 s |
| Padded media endpoint | 15.845833 s |

- Live video begins 19.492 ms before decoded source-media sample zero and 30.847 ms before -45 dB audible onset.
- Neutral begins 9.550 ms after -45 dB audible speech ends.
- Neutral begins 12.825 ms before the end of the deliberate silent tail.
- The live interval is exactly 332 frames: frames 135–466 inclusive. Frame 467 is neutral.
- All audio packet intervals are 1,024 / 48,000 s. All video intervals are 1 / 30 s except the single declared start correction.
- Six separate source/capture waveform windows inferred the same media origin to within one 48 kHz sample, so there is no accumulated drift.

## Regression verification

- 69 WebRTC audio, video, endpoint, recorder, pose-runtime, and semantic-timing tests passed.
- Compilation checks passed.
- `git diff --check` passed.
- The restarted server is healthy with all six avatar caches loaded and no active requests.

## Artifacts

- `indian_tutor_av_bidirectional_rtp_lock_v11_labeled.mp4` — review copy with state and timing labels.
- `indian_tutor_av_bidirectional_rtp_lock_v11_webrtc_capture.mp4` — unmodified receiver recording.
- `run.log` — complete harness output, server telemetry, and RTP proof.

This report validates A/V timing. Any remaining visual size/geometry seam between a speaking asset and `neutral_resting` is an asset-boundary issue, not an audio/lip-sync timing delay.
