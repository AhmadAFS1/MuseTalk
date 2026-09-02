# Multipose AI Video Call Architecture

## Executive answer

Lingua does call the configured OpenAI chat model to analyze the conversation,
but it does not make a separate motion-analysis request. The same structured
model response contains:

1. the exact text the avatar will speak;
2. one optional pre-speech reaction intent; and
3. one to three semantic delivery segments for the spoken response.

The model provides semantics such as `direct`, `warm`, and `empathetic`.
Application code maps those labels to an allowlisted pose. MuseTalk then decides
the exact frame timing after decoding the real TTS audio.

This separation prevents the language model from inventing video names or
timestamps and avoids the latency, cost, and failure mode of a second model
request.

The current runtime exposes exactly six logical pose IDs. The active 2026-09-01
close-up LTX 2.3 production bank resolves them to six unique physical MP4s:

- neutral and active listening intentionally alias one restrained idle/listener
  file;
- `speaking_direct` owns two renderer-level variants, V14 and V15;
- nod, empathy, and smile each own one physical file.

The semantic contract remains six poses even though direct speech has two
physical render choices. The model never sees or selects V14/V15 filenames.

During v2 lip-synced speech, only three logical speaking poses are eligible:
`speaking_direct`, `light_smile`, and `empathetic_head_tilt`. Across the entire
call lifecycle, neutral, listening, reactions, and speaking use the same six-ID
protocol regardless of which framing bank is active.

## Three different layers that must not be confused

### Semantic delivery labels

These labels are produced by the chat model:

- `direct`
- `warm`
- `empathetic`

They describe how a clause should be delivered. They are not filenames, pose
IDs, or counts of available assets.

### Logical pose IDs

These are the six identifiers allowed by the v1 runtime contract:

- `neutral_resting`
- `active_listening`
- `speaking_direct`
- `nod_agree`
- `empathetic_head_tilt`
- `light_smile`

Logical IDs allow the application protocol to remain stable even when two IDs
temporarily share one physical video or an older compatibility asset remains
installed.

### Physical motion videos

The active close-up production bank consists of:

1. one shared neutral/listening loop;
2. V14 subtle direct speech;
3. V15 reference-paced direct speech;
4. one compact yes nod;
5. one slow empathetic head tilt; and
6. one moderate closed-mouth smile.

### Current best-of asset set

The best videos approved so far are the close-up V6 active-listening loop for both idle and listening, close-up V14 and V15 together for direct speech, the close-up V6 nod, the close-up V6 slow empathetic tilt, and the close-up V8 moderate smile. V14 provides the calmer speaking baseline; V15 provides occasional reference-paced accents on alternating turns. The shoulder-width rerenders remain installed as an inactive alternative and are not the current production choice.

| Communication function | Semantic label | Logical pose ID | Physical MVP motion |
|---|---|---|---|
| Default idle | N/A | `neutral_resting` | Shared restrained idle/listener |
| User is speaking / assistant is thinking | N/A | `active_listening` | Shared restrained idle/listener |
| Ordinary spoken explanation | `direct` | `speaking_direct` | Deterministic V14/V15 direct variant |
| Positive or encouraging spoken clause | `warm` | `light_smile` | Light smile |
| Reassuring or understanding spoken clause | `empathetic` | `empathetic_head_tilt` | Slow independent empathy tilt |
| Agreement before speech | `acknowledge` reaction | `nod_agree` | Compact yes nod |

## Why the speaking labels are direct, warm, and empathetic

The labels divide spoken communication into three useful, visually distinct
functions:

### Direct

`direct` is the stable default. It is used for explanations, instructions,
questions, corrections, and ordinary conversation. Most of every response
should remain direct so the avatar does not constantly gesture.

It maps to `speaking_direct`.

### Warm

`warm` is reserved for genuinely positive clauses: praise, encouragement,
greetings, gratitude, or shared happiness.

It maps to `light_smile`.

### Empathetic

`empathetic` is reserved for reassurance or understanding when the user
expresses difficulty, sadness, frustration, or uncertainty.

It maps to the independent `empathetic_head_tilt` physical video.

Neutral and active listening are not delivery labels because they represent
conversation state rather than how spoken words are delivered. A nod is also
not a general delivery style; it is a short acknowledgment reaction before
speech.

Keeping the semantic vocabulary small makes the model more consistent and
reduces unnecessary pose switching.

## One model request, not two

The configured model receives the user message, conversation history, character
persona, memory context, language constraints, and the video-delivery
instructions.

It returns strict JSON:

```json
{
  "reaction_intent": "empathy",
  "response_segments": [
    {
      "text": "I understand why that feels difficult.",
      "delivery": "empathetic"
    },
    {
      "text": "We can work through it together.",
      "delivery": "warm"
    },
    {
      "text": "Start by repeating the first sentence.",
      "delivery": "direct"
    }
  ]
}
```

The schema permits:

- `reaction_intent`: `none`, `acknowledge`, `warmth`, or `empathy`;
- one to three response segments; and
- `delivery`: `direct`, `warm`, or `empathetic`.

Every spoken word must appear exactly once in `response_segments`. Stage
directions are prohibited. The segment text is joined back into one response
and synthesized as one continuous TTS WAV.

Using the same request for the reply and its annotations has four advantages:

1. the model labels the exact text it wrote;
2. there is no second model charge;
3. there is no second network/model latency; and
4. the reply and delivery analysis cannot disagree because of separate model
   contexts.

## Deterministic call-state order

The call uses deterministic events around the model-authored speech plan:

1. The session starts in `neutral_resting`.
2. `user_speech_started` replaces pending poses with `active_listening`.
3. `user_speech_ended` keeps the avatar attentive.
4. `assistant_thinking` keeps `active_listening` active.
5. The model returns the response, reaction intent, and delivery segments.
6. An optional pre-speech reaction is queued while TTS is prepared:
   - `acknowledge` -> `nod_agree`;
   - `warmth` -> `light_smile`;
   - `empathy` -> `empathetic_head_tilt`;
   - `none` -> no additional reaction.
7. One WAV and its pose plan are uploaded to MuseTalk.
8. MuseTalk plays the compiled speaking-pose order while generating lip-sync.
9. Audio completion replaces pending work with `neutral_resting`.
10. An aborted turn also returns immediately to `neutral_resting`.

Voice turns emit explicit user-speech start/end events. Typed messages begin at
the assistant-thinking state.

Every turn has a unique `turn_id`, and every event has an increasing `seq`.
MuseTalk rejects stale sequence numbers. Major state changes use
`replace_pending=true` so a reaction from an older turn cannot play during a
newer turn.

## How response segments become an audio-progress plan

The language model does not estimate timestamps. Lingua weights each response
segment by its visible, non-whitespace character count. This is more
language-neutral than word counting.

For segment weights of 20%, 30%, and 50%, Lingua produces:

```json
{
  "version": 2,
  "clock": "audio_progress",
  "segments": [
    {
      "at_permille": 0,
      "pose_id": "empathetic_head_tilt"
    },
    {
      "at_permille": 200,
      "pose_id": "light_smile"
    },
    {
      "at_permille": 500,
      "pose_id": "speaking_direct"
    }
  ],
  "on_complete": "neutral_resting",
  "switch_mode": "next_boundary"
}
```

`at_permille=200` means approximately 20% through the decoded audio, not 200
milliseconds.

The Lingua compiler:

- accepts at most three spoken segments;
- maps semantic delivery to the fixed pose allowlist;
- coalesces adjacent duplicate poses;
- forces the last spoken segment to `speaking_direct`; and
- always sets completion to `neutral_resting`.

Forcing a direct tail creates the safest handoff from lip-synced speech back to
neutral.

## WebRTC audio/video synchronization fix (2026-07-31)

The multi-pose scheduler and the WebRTC media tracks now share the decoded
audio timeline. This fixes two separate problems that were previously easy to
confuse:

1. A pose request could be held until the next complete source-video loop,
   making a requested switch several seconds late.
2. Audio could begin while the newly selected video was still being prepared,
   producing an apparent audio/video or lip-sync offset.

The current implementation addresses both problems as follows:

- Spoken audio is normalized once (including long leading/trailing silence
  trimming) and the same timeline is used for MuseTalk generation and the
  WebRTC audio sender.
- The audio sender is persistent and timestamp-locked. It is not replaced for
  every turn, so RTP timestamps remain contiguous across a response.
- Video waits behind a shared release gate until audio and the first live video
  frame are ready. The first live video RTP timestamp is forward-aligned to the
  persistent audio timeline.
- Pose plans switch at their requested audio-progress frame using a bounded
  crossfade. They no longer wait for an arbitrary full source-loop boundary.
- Generation ownership tokens reject stale frames or completion callbacks from
  an earlier turn.
- A neutral frame is staged before generation completes and is activated at the
  shared audio media endpoint, preventing a long silent tail or early neutral
  handoff.

The v11 end-to-end validation run used a three-segment plan
(`speaking_direct -> light_smile -> speaking_direct`) and recorded these
results:

| Measurement | Result |
|---|---:|
| Pose semantic drift | 0 frames / 0 seconds |
| Initial audio/video start delta | 5.2 ms |
| First live RTP delta | 13.3 ms |
| Maximum first-RTP mismatch | 33.3 ms |
| Audio stalls / video stalls | 0 / 0 |
| Audio media and playout duration | 11.06 s / 11.06 s |
| Pose crossfade | 4 frames at requested-time switches |

The focused WebRTC, pose-plan, and timestamp tests pass (69 tests in the
validation run). These checks establish transport and pose-timeline alignment;
they do not replace a dedicated phoneme-to-mouth metric such as SyncNet or a
human review of the recorded capture.

Validation artifacts are stored under
`generated/webrtc_pose_showcase/2026-07-31/av_bidirectional_rtp_lock_v11/`.

## Lingua backend validation and negotiation

Lingua does not trust pose metadata from the client. It validates that:

- the plan version is `2`;
- the clock is `audio_progress`;
- there are one to three segments;
- the first anchor is zero;
- anchors strictly increase and stay between 0 and 999;
- only `speaking_direct`, `light_smile`, and `empathetic_head_tilt` appear
  during speech;
- adjacent poses differ;
- the final speech pose is `speaking_direct`; and
- completion is `neutral_resting`.

At session creation, Lingua enables v2 only when:

- the character has a complete approved pose-set manifest;
- every required cache is ready on one worker;
- the worker advertises `pose_sets_v1`; and
- the worker advertises `pose_plans_v2`.

The selected worker and its negotiated capability are pinned to the session.
The app defaults are 30-fps WebRTC playback and 15-fps MuseTalk lip-sync
generation.

## MuseTalk timing and video ordering

MuseTalk decodes the actual TTS audio before finalizing pose timing.

For duration `D` at generation rate `F`, it calculates approximately:

```text
total_generation_frames = D * F
requested_frame = total_generation_frames * at_permille / 1000
```

The requested frame is the semantic audio deadline for the pose cue.

MuseTalk then:

1. phase-aligns the first speaking pose to the currently visible idle phase;
2. selects the nearest complete source-loop boundary only when it is within
   `WEBRTC_POSE_MAX_SEMANTIC_DRIFT_SECONDS` (0.75 seconds by default);
3. otherwise switches on the requested audio frame and starts the incoming pose
   at its canonical first frame;
4. applies `WEBRTC_POSE_FORCED_CROSSFADE_FRAMES` (four generated frames by
   default) only to requested-time switches;
5. preserves at least 0.75 seconds of terminal direct speaking;
6. records signed semantic drift and switch strategy in session telemetry;
7. rejects smoke tests when absolute cue drift exceeds 750 milliseconds; and
8. returns to neutral when audio finishes.

Ordinary boundary-safe changes retain the two-generated-frame cosine blend.
MuseTalk generates lip-sync frames at 15 fps and sends video on a 30-fps WebRTC
cadence. Crossfades replace incoming frames and never add media-clock duration.

## Measured behavior from the validated run

The backend/media end-to-end test submitted:

```text
speaking_direct at 0%
light_smile at 20%
speaking_direct at 60%
neutral_resting on completion
```

The former next-boundary-only scheduler compiled:

```text
speaking_direct: 0.00s-9.00s
light_smile: 9.00s-21.00s
speaking_direct: 21.00s-26.33s
neutral_resting: after audio completion
```

That output matched its compiled pose trace and had no initial A/V skew, but the
compiler itself was semantically wrong:

- the 20% smile cue was applied 3.73 seconds after its nominal target;
- the 60% direct cue was applied 5.20 seconds after its nominal target.

The bounded-semantic scheduler now compiles the same cues at generation frames
79 and 237 (approximately 5.27 and 15.80 seconds at 15 fps), using
requested-time crossfades because neither complete source boundary is within
750 milliseconds. Automated validation fails if this regression returns.

The 2026-07-31 WebRTC regression recording confirmed the complete media path:

- both switches rendered exactly at frames 79 and 237 with zero semantic drift;
- live audio and video began 0.335 milliseconds apart;
- video covered 26.333 seconds of the 26.350-second WAV;
- no video frames were dropped and there were no queue underruns or video
  stalls; and
- the received 30-fps MP4 is at
  `generated/webrtc_pose_showcase/2026-07-31/semantic_sync_fix_v1/indian_tutor_semantic_sync_fix_v1_webrtc_capture.mp4`.

The test used a representative compiled pose plan to isolate the real
Lingua-to-MuseTalk media path. It did not include live STT, model generation, or
TTS network latency.

## Fallback behavior

The contract falls back without abandoning the entire call:

1. If v2 was not negotiated, Lingua sends the validated v1 order:
   optional reaction -> `speaking_direct` -> `neutral_resting`.
2. If a v2 contract is rejected, Lingua rewinds the same WAV and retries without
   the v2 plan while preserving v1 metadata.
3. If v1 is also rejected, Lingua retries once without pose metadata.
4. Non-contract server failures are returned without replaying the WAV, which
   prevents duplicate spoken responses.

## Direct-speaking physical variants

`speaking_direct` remains one semantic pose with two physical plates. MuseTalk
selects a plate after semantic-plan compilation:

- a new assistant turn alternates to the next configured variant;
- retries of the same `turn_id` reuse the same assignment;
- V14 is the top-level fallback for workers or callers that do not negotiate
  `pose_variants_v1`; and
- telemetry reports the semantic `pose_id` separately from the physical
  `render_key`.

Variant selection is renderer-owned. Lingua continues to request only
`speaking_direct`, so physical motion variety cannot alter response semantics.

## Implementation locations

### Lingua

- `app/components/APIutils.tsx`: model prompt, strict JSON schema, parsing, and
  TTS response text.
- `app/components/videoCallPosePolicy.ts`: semantic-to-pose mapping and
  character-weighted plan compiler.
- `app/components/FaceTime.tsx`: session events, TTS, pose metadata upload, and
  capability handling.
- `backend/services/avatar_pose_protocol.py`: server-side contract validation.
- `backend/services/musetalk_router.py`: per-session v1/v2 capability pinning.
- `backend/routes/live_sessions.py`: cache/session negotiation, forwarding, and
  v2-to-v1-to-legacy fallback.

### MuseTalk

- `scripts/pose_protocol.py`: MuseTalk-side contract validation.
- `scripts/webrtc_manager.py`: call-state events, stale-event protection,
  reaction deduplication, plan staging, neutral recovery, and stable per-turn
  variant assignment.
- `scripts/webrtc_pose_router.py`: audio-progress compilation and safe-boundary
  scheduling with semantic-pose/physical-render separation.
- `scripts/hls_gpu_scheduler.py`: pose-aware frame generation, phase alignment,
  telemetry, and two-frame crossfades.
- `api_server.py`: WebRTC capabilities, session creation, stream upload, and
  pose-plan endpoints.
