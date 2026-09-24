# Two-avatar LTX 2.3 + MuseTalk WebRTC evidence

Date: 2026-09-23

## Scope

- Source portraits: the close-up Japanese and Latina portraits in the parent asset set.
- Motion source: local LTX 2.3 Q4 workflow through `generate_three_pose_videos.py`.
- Physical poses per avatar: idle, talking, smiling.
- MuseTalk preparation: `POST /avatars/prepare`, invoked by `scripts/test_pose_webrtc.py`.
- Live test: one recorded WebRTC warmth session per avatar at 20 fps with the existing 60-second `data/audio/eng.wav` fixture.
- Segmind: not used.

## LTX source validation

Every source is 480x832 H.264/yuv420p at 24 fps, has no audio stream, and has an exact decoded first/last frame match.

The three poses for an avatar do not share one decoded boundary hash. They are exact self-loops, but they have not passed the character factory's stronger shared-anchor certification. The tested runtime used its configured two-frame pose crossfade. Keep both manifests as test candidates until normal-speed review and shared-anchor certification are complete.

| Avatar | Pose | Frames | Duration | SHA-256 |
| --- | --- | ---: | ---: | --- |
| Japanese | idle | 241 | 10.041667 s | `0a747550738964e33eb0e1f6a82563179c289b251cf6310da9e0c5a748abc207` |
| Japanese | talking | 289 | 12.041667 s | `fc510715abe40a8ac6f0f1a6dd3683961a11593739b2d41e655952f8342244dd` |
| Japanese | smiling | 145 | 6.041667 s | `054fdb7c45dd4e774f4958a4834d0ace85aaa48ea1d3a2115ab9b3d7735502a4` |
| Latina | idle | 241 | 10.041667 s | `62fb275517e65ca7e71fa5e9e14d82041c46aa30ffb0b877295839c6a3f90882` |
| Latina | talking | 289 | 12.041667 s | `282c8e28d63ed9242b18dbc648c6b95e1378f5977aa4927162aa930e88459ff7` |
| Latina | smiling | 145 | 6.041667 s | `509be58b1f33198d41af911622cc19a522ebbbbacd920878947124d773b57e54` |

## Create-avatar API result

All six physical avatars finished with `status=ready`, `cached=true`, and `disk_prepared=true` after preparation, warming, both WebRTC sessions, and the final status recheck.

| Avatar | Idle ID | Talking ID | Smiling ID |
| --- | --- | --- | --- |
| Japanese | `japanese_baddie_ltx23_idle_v1` | `japanese_baddie_ltx23_talking_v1` | `japanese_baddie_ltx23_smiling_v1` |
| Latina | `latina_baddie_ltx23_idle_v1` | `latina_baddie_ltx23_talking_v1` | `latina_baddie_ltx23_smiling_v1` |

The required six logical protocol poses are mapped to these three physical caches in the two test manifests. Idle is reused for neutral, listening, nod, and empathy; talking is used for direct speech; smiling is used for warmth.

## Passing WebRTC sessions

Both sessions completed the rendered pose trace `light_smile -> speaking_direct -> neutral_resting` and recovered to neutral after speech.

| Metric | Japanese | Latina |
| --- | ---: | ---: |
| Harness result | pass | pass |
| Elapsed | 65.456 s | 65.468 s |
| Server video frames played | 1,200 | 1,200 |
| Server audio frames sent | 3,000 | 3,000 |
| Receiver video frames | 1,286 | 1,286 |
| Receiver audio frames | 3,264 | 3,263 |
| Dropped frames | 0 | 0 |
| Duplicated frames | 0 | 0 |
| Queue underruns | 0 | 0 |
| Audio stalls | 0 | 0 |
| Video stalls | 0 | 0 |
| First-live A/V RTP delta | 10 ms | 10 ms |
| Allowed first-live delta | 50 ms | 50 ms |
| Final pose | neutral_resting | neutral_resting |
| Proof duration | 65.396 s | 65.378 s |
| Proof SHA-256 | `d2c76d796ab7c6c4bb35c38fb3c7bc8fcd37657f489ed8d15f071bd85dfdb812` | `ae92ac5cfebaec06be9d5385b61dfd1f04fd10fdd321696d597e7e94db2be593` |

Each proof MP4 contains 480x832 H.264 video and 48 kHz stereo AAC audio. The receiver timestamp validator passed. It observed one declared one-frame/50 ms video phase correction at the idle-to-live release in each session, with no audio timestamp anomaly.

## Visual review

Five-frame contact sheets for every LTX source show stable identity, crop, clothing, background, and lighting. The talking sources keep a closed, relaxed mouth so MuseTalk owns speech articulation. The smile sources stay closed-lip and return to neutral.

Six-frame samples from each received WebRTC proof show normal-size MuseTalk mouth shapes across centered and tilted LTX head positions. No tiny-mouth failure is apparent in the sampled frames. Full-motion approval still requires watching the two proof MP4s at normal speed.

## Short-audio edge case

An initial Japanese run with the existing 8-second `data/audio/yongen.wav` fixture established a WebRTC connection and streamed, but the harness correctly failed it because the audio clock ended after smile and talking, before the required neutral segment. The partial proof and log are retained as `japanese_webrtc_short_audio_failed.*`. The subsequent 60-second session passed.
