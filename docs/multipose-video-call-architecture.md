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

The current runtime is **not strictly four-pose-only**:

- The pose protocol exposes exactly six logical pose IDs, not ten.
- The latest MVP introduced four new physical motions.
- `active_listening` and `empathetic_head_tilt` are logical aliases backed by
  the same combined listener/empathy MP4.
- The older `nod_agree` compatibility MP4 is still installed and can still be
  selected for the `acknowledge` reaction.
- Therefore, the current six logical IDs resolve to five distinct physical
  MP4s.

During v2 lip-synced speech, only three logical speaking poses are eligible:
`speaking_direct`, `light_smile`, and `empathetic_head_tilt`. Across the entire
call lifecycle, neutral, listening, reactions, and speaking can use the wider
six-ID protocol.

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

The latest four-motion MVP consists of:

1. neutral resting;
2. direct speaking;
3. light smile;
4. combined active listening and empathy.

The combined motion is installed under both `active_listening` and
`empathetic_head_tilt`. Those two runtime files are byte-identical. The older
`nod_agree` file is the additional fifth distinct runtime MP4.

| Communication function | Semantic label | Logical pose ID | Physical MVP motion |
|---|---|---|---|
| Default idle | N/A | `neutral_resting` | Neutral resting |
| User is speaking / assistant is thinking | N/A | `active_listening` | Combined listener/empathy |
| Ordinary spoken explanation | `direct` | `speaking_direct` | Direct speaking |
| Positive or encouraging spoken clause | `warm` | `light_smile` | Light smile |
| Reassuring or understanding spoken clause | `empathetic` | `empathetic_head_tilt` | Combined listener/empathy |
| Agreement before speech | `acknowledge` reaction | `nod_agree` | Older compatibility nod |

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

It maps to `empathetic_head_tilt`, which currently uses the same physical
combined listener/empathy video as `active_listening`.

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

The requested frame is a semantic cue, not permission to cut a physical motion
clip.

MuseTalk then:

1. phase-aligns the first speaking pose to the currently visible idle phase;
2. finds the next complete source boundary at or after each requested cue;
3. lets an expressive insert play its complete source cycle;
4. preserves at least 0.75 seconds of terminal direct speaking;
5. skips a cue when the complete motion cannot fit safely;
6. records requested versus effective timing in session telemetry; and
7. returns to neutral when audio finishes.

Skipping a gesture is preferable to cutting a clip, delaying the audio, or
creating a visible body discontinuity.

At each speaking-pose change, the scheduler uses a two-generated-frame cosine
blend. MuseTalk generates lip-sync frames at 15 fps and sends video on a 30-fps
WebRTC cadence.

## Measured behavior from the validated run

The backend/media end-to-end test submitted:

```text
speaking_direct at 0%
light_smile at 20%
speaking_direct at 60%
neutral_resting on completion
```

For the 26.35-second WAV, MuseTalk compiled:

```text
speaking_direct: 0.00s-9.00s
light_smile: 9.00s-21.00s
speaking_direct: 21.00s-26.33s
neutral_resting: after audio completion
```

The run rendered all 395 generated frames, received 1,329 30-fps video frames,
reported no dropped frames or video stalls, matched the compiled pose trace
exactly, and ended in neutral.

Media submission to A/V release was 2.36 seconds. Most of that was the
deliberate two-second video prebuffer. Initial A/V skew was 0.26 milliseconds.

The smoothness policy introduced semantic timing drift:

- the 20% smile cue was applied 3.73 seconds after its nominal target;
- the 60% direct cue was applied 5.20 seconds after its nominal target.

The current system therefore prioritizes seamless complete motions over exact
semantic cue timing. Shorter certified motion cycles would improve semantic
responsiveness without adding another model call.

The test used a representative compiled pose plan to isolate the real
Lingua-to-MuseTalk media path. It did not include live STT, model generation, or
TTS latency in the measured 2.36 seconds.

## Fallback behavior

The contract falls back without abandoning the entire call:

1. If v2 was not negotiated, Lingua sends the validated v1 order:
   optional reaction -> `speaking_direct` -> `neutral_resting`.
2. If a v2 contract is rejected, Lingua rewinds the same WAV and retries without
   the v2 plan while preserving v1 metadata.
3. If v1 is also rejected, Lingua retries once without pose metadata.
4. Non-contract server failures are returned without replaying the WAV, which
   prevents duplicate spoken responses.

## Four-pose-only product decision

If the product requirement is to use only the latest four physical motions,
the current implementation needs one small policy change. The
`acknowledge -> nod_agree` mapping must no longer route to the older nod file.

A strict four-motion mapping can be:

| Intent or state | Physical motion |
|---|---|
| Idle and completion | Neutral resting |
| User speaking, thinking, empathy, or acknowledgment | Combined listener/empathy |
| Ordinary speech | Direct speaking |
| Warmth and positive encouragement | Light smile |

This can retain the six logical IDs for wire compatibility while ensuring that
only four physical MP4s are ever selected. Alternatively, the protocol can be
reduced to four logical IDs in a breaking v3 cleanup.

As currently deployed, the runtime does **not** enforce this strict four-only
policy because `acknowledge` can still select the legacy `nod_agree` asset.

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
  reaction deduplication, plan staging, and neutral recovery.
- `scripts/webrtc_pose_router.py`: audio-progress compilation and safe-boundary
  scheduling.
- `scripts/hls_gpu_scheduler.py`: pose-aware frame generation, phase alignment,
  telemetry, and two-frame crossfades.
- `api_server.py`: WebRTC capabilities, session creation, stream upload, and
  pose-plan endpoints.

