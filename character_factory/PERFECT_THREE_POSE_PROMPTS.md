# User-approved perfect three-pose prompt pack

This is the canonical prompt record for the Japanese FaceTime avatar trio that
the user approved as **perfect** on 2026-09-25 after reviewing idle, talking,
and smiling at normal playback speed. Use the versioned machine-readable pack
[japanese_fixed_distance_shared_anchor_v1.json](config/prompt_packs/japanese_fixed_distance_shared_anchor_v1.json).
Do not paraphrase or shorten these prompts when reproducing the accepted trio.

The approval applies to the complete recipe, not prompt text alone:

- source portrait SHA-256: 14ea206c301eccdf9461ab2b706d9c3489927156267ca74e8ba478d9124fdf7b
- native LTX 2.3 22B distilled 1.1 Q4_K_M model
- 512 x 832, 241 frames, 24 fps, silent output
- eight Euler steps with saved sigmas:
  1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0
- CFG 1.0; source portrait guidance at frames 0 and -1, strength 1.0
- center_crop guide preparation
- temporal decode 64/16 and spatial decode 256/64
- --shared-anchor, which installs the same prepared portrait at the first and
  last delivery frame of every clip and verifies decoded cross-clip equality
- no fade, crossfade, scale stabilization, motion warp, SoulX, Segmind, NAG,
  Prompt Relay, MuseTalk, or WebRTC rendering in the source clips

## Idle - seed 193

Positive prompt:

~~~text
Fixed-camera photorealistic FaceTime close-up of the same female language tutor in the reference image, with the same room, lighting, clothing, and framing. She stays seated at exactly the reference distance from the camera for the entire shot. Her head keeps the same size in the frame, with her eyes at the reference height; her torso stays in its starting position. She listens quietly to the caller, alert and engaged, looking directly into the lens. Her eyes stay open and attentive. Her lips stay comfortably closed. Her head remains upright and level, without leaning or bobbing. Over this ten-second shot, two ordinary shallow breaths make the fabric at her upper chest move gently outward and back while her shoulders remain relaxed. She continues her natural listening pose and finishes at the starting pose. One continuous shot, stable background.
~~~

Negative prompt:

~~~text
leaning toward camera, leaning away from camera, rocking torso, changing head size, zoom, dolly, reframing, sleeping, closed eyes, prolonged blink, drowsy face, sideways head tilt, head bobbing, head wobble, sighing, gasping, mouth opening, speaking, lip articulation, shoulder distortion, distorted face, identity change, duplicate facial features, off-camera gaze, camera movement, scene cut, text, subtitles, watermark
~~~

## Talking - seed 191

Positive prompt:

~~~text
Fixed-camera photorealistic close-up of the same female language tutor in the reference image, with the same room, light, clothing, and framing. She stays seated at exactly the reference distance from the camera for the entire shot. Her head keeps the same size in the frame, with her eyes at the reference height; her torso stays in its starting position. She looks straight into the lens and speaks one lively, uninterrupted ten-second sentence. Her mouth and jaw keep forming new conversational syllables throughout the shot, including the final seconds; brief lip closures are part of the words, never a long silent pause. She stays engaged in the sentence until the last moment. Her head remains upright and level, her neck relaxed. Small, regular breaths happen naturally between words, with barely visible motion in her upper chest and shoulders. She gives two quick natural blinks. Continuous shot, stable background.
~~~

Negative prompt:

~~~text
leaning toward camera, leaning away from camera, rocking torso, changing head size, zoom, dolly, reframing, silent person, frozen mouth, upward chin lift, neck extension, shoulder distortion, stretched frame borders, off-camera gaze, distorted face, identity change, duplicate facial features, camera movement, scene cut, text, subtitles, watermark
~~~

## Smiling - seed 191

Positive prompt:

~~~text
Fixed-camera photorealistic close-up of the same female language tutor in the reference image, with the same room, light, clothing, and framing. She stays seated at exactly the reference distance from the camera for the entire shot. Her head keeps the same size in the frame, with her eyes at the reference height; her torso stays in its starting position. A restrained closed-lip smile slowly appears in the corners of her mouth and softly in her cheeks, holds, then fades back to neutral. Her upper and lower lips stay gently in contact through the entire smile. She looks into the lens and gives one quick natural blink. Her eye line and chin remain level at the reference height, her neck keeps its relaxed starting length, and gentle breathing moves her upper chest and both shoulders together. She ends in the exact starting expression and pose. Continuous shot, stable background.
~~~

Negative prompt:

~~~text
leaning toward camera, leaning away from camera, rocking torso, changing head size, zoom, dolly, reframing, speaking, lip articulation, parted lips, visible teeth, upward chin lift, neck extension, shoulder distortion, stretched frame borders, distorted face, identity change, duplicate facial features, off-camera gaze, camera movement, scene cut, text, subtitles, watermark
~~~

## Reproduce the approved recipe

~~~bash
cd /workspace/MuseTalk
/workspace/experiments/soulx_ltx_motion_pilot_20260922/A1/.venv/bin/python character_factory/scripts/generate_three_pose_videos.py \
  --image character_factory/generated/portrait_pose_set_20260923/japanese_woman.png \
  --output-dir /workspace/experiments/japanese_ltx_fixed_distance_20260925_rerun \
  --prompt-pack character_factory/config/prompt_packs/japanese_fixed_distance_shared_anchor_v1.json \
  --guide-fit center_crop \
  --shared-anchor
~~~

The accepted evidence is in
[/workspace/experiments/japanese_ltx_fixed_distance_20260925/README.md](/workspace/experiments/japanese_ltx_fixed_distance_20260925/README.md).
All three decoded endpoint pairs share RGB SHA-256
a84f1f37dfb5c381d50891e2dd62e985b57bacf00d2c2cfd523e2a80152f2ddd.
The delivery-video SHA-256 values are:

- idle: 41f91fd0816ed8002b677112d501522e41dc56b37832c26730205c0d6ffc6a9c
- talking: 7d94520b7fe776313b2ea80815876116305b198280e5a73672ba20bc47a0e278
- smiling: f682b41267d3f79c1d70ed9e44c220f8dd0d6f32529c43e0a1f67275ee881d91

New identities can reuse this behavioral recipe, but LTX remains stochastic.
Keep the prompts, graph, settings, guide policy, shared-anchor packaging, and
review gate fixed. A new portrait still requires normal-speed human review
before its trio inherits the approved label.
