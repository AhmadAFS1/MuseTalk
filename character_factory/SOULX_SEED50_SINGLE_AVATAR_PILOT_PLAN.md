# Single-avatar SoulX seed-50 → LTX 2.3 → MuseTalk pilot

**Status: execution plan only. Written 2026-09-22. No pilot inference, tests, installation, or runtime changes have been performed to create this plan.**

**Executor:** GPT Luna, or another coding agent. Read this entire file before executing. Commands for new pilot scripts below describe interfaces you must implement first; those scripts do not exist yet. Commands targeting existing repository tools have been checked against their source, but must be rechecked if that source changes.

**Start here when executing:** use the checkable [TASKS list in section 18](#18-tasks--the-executable-work-queue). It provides task IDs, prerequisites, outputs, verification, failure handling and resume instructions. Sections 1–17 are the detailed specifications for those tasks.

**Hardware provenance:** the reused SoulX PRO video was generated on an NVIDIA GeForce RTX 4070 SUPER, physical 12 GB class / 12,282 MiB visible, driver 595.84, Torch 2.7.1+cu128, CUDA runtime 12.8. Evidence is its results.json specified below. Historical LTX production assets came from a different RTX 5060 Ti environment; do not assign the donor GPU to those assets. Current execution hardware and co-resident load must be measured again. This document contains source analysis and proposed engineering thresholds, not fresh GPU results or a capacity claim.

## 1. Objective and completion criteria

Produce a reviewable experiment answering this specific question:

> Can the head movement in one saved SoulX PRO seed-50 performance guide LTX 2.3 to generate a useful, mostly closed-mouth talking-motion base for the existing close-up avatar, which MuseTalk can lip-sync to different speech?

There is exactly **one target avatar**. The saved SoulX performer is the motion donor. Using a donor does not authorize creating a second target character. Do not expand into a language roster, regenerate portraits, train a model, or rewrite the production motion system.

Deliver either:

1. A passing experimental three-physical-clip bank, measurements, labeled comparisons, and recorded MuseTalk playback; or
2. A useful negative/partial result showing exactly which gate failed, with retained evidence and a concrete next experiment.

A negative finding is a valid experiment result. Never mark an arm successful just because ComfyUI returned a video or because its endpoint hashes match.

The pilot tests one donor, target, and motion take. It does not establish cross-identity generalization, a reusable seed independent of audio, or production readiness across languages.

### 1.1 Meaning of “match SoulX seed 50”

> **Measured evidence added 2026-09-22:** the claim in this section is now
> quantified in `SoulX-FlashHead/docs/research/MOTION_REUSE_FEASIBILITY_2026-09-22.md`.
> A 4-step -> 2-step schedule change moves head motion as much as changing the seed
> (1.04x), audio drives 52-70% of head-motion variance, and exact repeats are
> bit-identical. That document also records that the A1 arm has already executed
> (eye-x r=0.995, eye-y r=0.995, roll r=0.969) and that IC-LoRA LipDub is not
> installable on this box.


The target is the timestamped head trajectory in the **saved PRO video**, not the numeric seed by itself. Seed 50 does not mean the same movement across PRO, LITE, quantization profiles, portraits, or audio. Do not change donor audio to silence and assume the head trajectory survives. Do not copy SoulX latents into LTX. Their latent spaces and conditioning contracts differ.

The control video should guide rotation, position, scale changes, and broad shoulder motion. The original spoken mouth shapes should not become the new base video's speech. MuseTalk supplies new articulation later. Blinks and expressions are additional visual qualities, not guaranteed outputs of a head-motion signal.

### 1.2 Scope and preservation rules

- Execute only after the user asks to execute this plan. Creating this file is not execution authorization.
- When execution is authorized, ordinary reversible local setup, model-adapter downloads, and isolated pilot runs are part of the pilot; do not repeatedly ask for routine implementation choices.
- Preserve all existing dirty work. In particular, SoulX's pipeline, PRO decoder, quantization harness, and iteration log had unrelated edits when this plan was written.
- Do not change SoulX PRO/LITE weights, runtime settings, optimization files, or services.
- Do not run the character factory roster/orchestrator CLI. It uses shared roster and ledger state, which this pilot does not need.
- Do not change production banks, production manifests, default pose-set selection, six-pose protocol, or existing avatar IDs.
- No paid external generation API, remote publication, bulk run, or deployment is included.
- Start only pilot-owned processes. Never kill an unrelated process to free GPU memory. A genuinely conflicting GPU owner is an execution blocker until it finishes or the user authorizes coordination.
- Do not delete old output directories to retry. Allocate a new attempt directory and retain failures.

## 2. Frozen inputs and references

These paths were found during planning. Recheck existence, media facts, and hashes during preflight. A path mentioned by an old README is not proof that its media is present.

### 2.1 Motion donor: use this exact existing take

| Purpose | Absolute path |
|---|---|
| PRO seed-50 video | /workspace/SoulX-FlashHead/benchmarks/pro_lite_150x_20260917/recreated_20260921/pro/video.mp4 |
| PRO provenance | /workspace/SoulX-FlashHead/benchmarks/pro_lite_150x_20260917/recreated_20260921/pro/results.json |
| Recreation explanation | /workspace/SoulX-FlashHead/benchmarks/pro_lite_150x_20260917/recreated_20260921/README.md |
| Donor reference portrait | /workspace/SoulX-FlashHead/benchmarks/pro_lite_150x_20260917/reference-150x.png |
| Original donor audio | /workspace/SoulX-FlashHead/benchmarks/pro_lite_150x_20260917/audio.wav |
| Optional existing comparison, for inspection only | /workspace/SoulX-FlashHead/benchmarks/pro_lite_150x_20260917/recreated_20260921/soulx-LITE-vs-PRO-indian-man-1.50x-320x576-seed50-recreated.mp4 |

Expected donor media: 320×576, 25 FPS, 250 frames, ten seconds, H.264 with audio. Recorded model profile: PRO, BF16, four sampling steps, seed 50, 1.50× framing, eager execution. It is a September 21 recreation, not a guaranteed byte-identical replay of September 17.

Known reference portrait SHA-256:

~~~text
a0a2317609afe70ed6c3fa3c620654dc56b123981429b2316dc6f7cc8222d676
~~~

Known donor audio SHA-256:

~~~text
0bbc0e4d1f1e4ecaad1f425e311e8f3ced2d1011d65a30b46487389331b456b2
~~~

Compute the video hash; this plan does not invent one. If donor media is missing, search retained artifacts read-only. Do not silently substitute the LITE take or rerun the currently changing PRO code. Report the missing fixture and the available alternatives.

### 2.2 Target avatar and existing reference bank

Bank root:

~~~text
/workspace/MuseTalk/assets/ltx23_pose_banks/sample_ai_human_facetime_closeup_production_v1
~~~

| Use | File relative to bank root |
|---|---|
| Identity provenance | source/sample_ai_human_facetime_v1.png |
| Idle/listening to retain | certified/idle_active_listening.mp4 |
| Smile to retain | certified/light_smile.mp4 |
| Current talking benchmark | certified/speaking_direct_v14_subtle.mp4 |
| Secondary talking reference, inspection only | certified/speaking_direct_v15_reference_paced.mp4 |
| Source metadata | manifest.json, validation_report.json, README.md |

Production runtime manifest, read-only:

~~~text
/workspace/MuseTalk/configs/pose_test/sample_ai_human_ltx23_facetime_closeup_production_v1.json
~~~

Known portrait hash: 0e893bd133c22de532ced133f91e136a2896663612c506b9801c64ddd3d1e919.

Expected assets: idle 241 frames; smile 145 frames; V14 talking 289 frames; all 480×832 at 24 FPS, silent. Verify these facts rather than trusting filename conventions.

**Canonical target image:** decode the first frame of the existing idle video to lossless RGB PNG. Use it as the actual pilot identity guide and eventual bank anchor. Retain the larger source portrait as provenance. This keeps the experiment in the current bank's camera composition and color space.

Do not assume the donor portrait and target portrait are identical, or that their crops agree. This plan explicitly separates donor appearance from target appearance. Only the target portrait/anchor determines the target character.

### 2.3 Existing code to reuse selectively

| Code | How to use it |
|---|---|
| MuseTalk/character_factory/scripts/certify_pose_bank.py | Reuse decode_frames, apply_handles, encode_frames, probe_contract; inspect source before importing |
| MuseTalk/scripts/test_pose_webrtc.py | Actual manifest loader, cache preparation, runtime recording and timestamp validation |
| MuseTalk/scripts/pose_protocol.py | Actual six-logical-pose validation and routing contract |
| SoulX-FlashHead/benchmarks/distance_lipsync/analyze_head_motion.py | Metric formulas only; its CLI is hardcoded to older fixtures |
| SoulX-FlashHead/soulx_rtc/face_landmarker.py | Example of CPU Tasks FaceLandmarker setup; its wrapper returns only lip points, so extend in pilot-local code |
| SoulX-FlashHead/models/ojin-components/face_landmarker.task | Existing model for pilot-local facial landmark extraction |
| experiments/ltx23-soulx-transfer/q4_sdpa_tiled_10s_api.json | Known local GGUF loader names and decode settings; not the complete new control graph |
| experiments/ltx23-soulx-transfer/ablation/save_latent_graphs.py | Pattern for generating first and decoding in a separate process |
| experiments/ltx23-soulx-transfer/ablation/prepare_decode.py | Pattern only; do not run its hardcoded-path/glob-based CLI against pilot output |
| SoulX-FlashHead/benchmarks/ltx_transfer_20260917/ROOT_COMPONENTS.md | Documents NAG/Prompt Relay patch conflicts; avoid carrying these into the pilot |

### 2.4 External primary sources

Resolve and record immutable revisions during execution. Do not download a different LTX version because a current documentation homepage defaults to it.

- [LTX 2.3 Union Control model card](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/blob/main/README.md).
- [LTX 2.3 official Union Control workflow](https://github.com/Lightricks/ComfyUI-LTXVideo/blob/master/example_workflows/2.3/LTX-2.3_ICLoRA_Union_Control_Distilled.json).
- [LTX 2.3 Motion Track Control](https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Motion-Track-Control/blob/main/README.md), fallback research only; not a required pilot arm.
- [ComfyUI control preprocessors](https://github.com/Fannovel16/comfyui_controlnet_aux), for DWPose.
- [Video Depth Anything ComfyUI nodes](https://github.com/kijai/ComfyUI-VideoDepthAnything), depth candidate used in the official workflow family.

Union Control's file is named ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors. Its expected reference dimensions are half the generated width/height. Match the adapter revision to the installed distilled-1.1 base. Merely finding matching filenames is insufficient compatibility evidence.

## 3. Experiment design: one target, three main LTX arms

Keep the donor fixed throughout. Do not regenerate it for each arm.

| Arm ID | Description | Purpose |
|---|---|---|
| REF_DONOR | Existing PRO video, resampled and spatially aligned for measurement | Defines requested movement; never counted as a generated target avatar |
| REF_V14 | Existing target V14 talking video | Historical visual benchmark, not a matched causal control |
| A0_PORTRAIT | New LTX portrait-only render | Matched baseline without the Union adapter or motion guide |
| A1_POSE | Same generation recipe, Union adapter plus donor pose guidance | Main transfer candidate |
| A2_DEPTH | Same generation recipe, same adapter plus donor depth guidance | Alternative transfer candidate |
| A3_REPAIR | At most one predeclared repair of the better guided arm | Optional, selected by the decision tree in section 11 |

A0 vs A1/A2 measures the **adapter-plus-guidance package**, not guidance alone. Do not claim it isolates the guide from the adapter weights. A1 vs A2 isolates the control representation more closely because the adapter remains the same.

No direct donor-video MuseTalk bank is required: its identity/framing may differ from the target, which would confound this one-target experiment. Keep its original video visible in motion comparisons. A direct-donor baseline can be a later explicitly scoped experiment.

### 3.1 Fixed recipe

- Generation canvas: **512×832**.
- Delivery crop: **480×832**, remove 16 columns from the left and right; no vertical crop or resize after generation.
- Why: the half-size Union reference is 256×416, both divisible by the LTX VAE spatial stride 32. A native width of 480 would have a half-size width of 240. Using a padded generation canvas avoids relying on undocumented rounding.
- Identity guide: canonical 480×832 target anchor, edge-padded 16 pixels on each side to 512×832. Record padding policy and hash. Do not stretch the face.
- Frame rate: 24 FPS.
- Smoke length: 49 frames = 2.0416667 seconds.
- Full length: 241 frames = 10.0416667 seconds.
- Both lengths satisfy LTX's 8n+1 frame arithmetic.
- Batch size: one.
- LTX base: existing LTX-2.3-22B-distilled-1.1-Q4_K_M.gguf.
- Text encoder and video/audio VAEs: exact existing locked files from the local research stack.
- Initial LTX noise seed: **189**, identical across A0/A1/A2. This is an experiment seed, not a transfer of SoulX seed 50.
- Sampler: Euler; one continuous eight-step schedule, CFG 1.0.
- Proposed sigmas: [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]. Verify compatibility with the pinned base/official control recipe before GPU execution.
- Initial image-guide strength: 1.0.
- Initial Union LoRA model strength: 1.0 for guided arms; absent from A0.
- Initial control-guide strength: 1.0 for guided arms.
- PyTorch SDPA attention, matching the existing local low-VRAM adaptation.
- NAG off; Prompt Relay off; custom attention tuning off; no 6+2 denoised restart; no spatial upscaler; no mouth enhancer or sharpening.
- All arms use the same identity preprocessing, prompt, schedule, output decode, and encoding settings.

The continuous schedule is deliberately a new controlled experimental recipe. It does not reproduce the old V14 production recipe. If the official compatible adapter revision requires a different schedule/model-sampling setting, record that resolved recipe **before any full renders**, apply it to all three arms, and keep the original proposed recipe in the metadata. Do not silently tune only an attractive-looking arm.

### 3.2 Prompt: store literally in configuration

Positive:

~~~text
Fixed-camera photorealistic close-up of the person in the reference image, in the same room, lighting, clothing and framing. The person maintains relaxed eye contact and natural blinking, with conversational head and shoulder movement. The lips remain gently closed throughout; the jaw is relaxed. Preserve the person's facial proportions and appearance. Continuous shot, stable background.
~~~

Negative:

~~~text
speaking, lip articulation, open mouth, visible teeth, identity change, distorted face, duplicate facial features, off-camera gaze, camera movement, scene cut, text, subtitles, watermark
~~~

Keep the prompt identical in all main arms. In CFG=1 without NAG, the negative conditioning may not materially contribute; record this and do not attribute mouth closure to the negative prompt. The positive prompt and control signal must carry the task. Do not add motion-amplitude negatives such as “movement” or “head motion.”

## 4. Directory layout and implementation interfaces

Use this fresh experiment root if it does not already exist:

~~~text
/workspace/experiments/soulx_ltx_motion_pilot_20260922
~~~

If it exists and is not this pilot's resumable directory, create a UTC-suffixed sibling. Keep one resolved absolute root in config.json. Never use “latest” globs to select a source, latent, video, or result.

~~~text
pilot-root/
  config.json
  state.json
  README.md
  scripts/
    common.py
    preflight.py
    setup_environment.py
    prepare_inputs.py
    extract_controls.py
    build_graphs.py
    run_comfy.py
    analyze_motion.py
    build_review.py
    certify_package.py
    run_webrtc.py
    report.py
  tests/test_pilot_contracts.py
  environment/
    source-lock.json
    requirements-base.lock
    requirements-resolved.lock
    dependency-diff.json
    hardware.json
    object-info.json
    model-lock.json
  inputs/
    source-manifest.json
    donor-original.mp4
    donor-original-audio.wav
    donor-reference.png
    target-portrait-original.png
    target-anchor.png
    target-guide-512x832.png
    donor-timeline-24fps/000000.png ... 000240.png
    donor-timeline-map.json
    donor-alignment.json
    controls/pose-full/
    controls/depth-full/
    control-validity.json
    speech-a.wav
    speech-b.wav
  graphs/<arm>/<smoke-or-full>/generation.json
  graphs/<arm>/<smoke-or-full>/decode.json
  renders/<arm>/<attempt>/
    submission.json
    history.json
    telemetry.jsonl
    result.json
    video.latent
    native.mp4
    delivery.mp4
  analysis/<arm>/frames.jsonl
  analysis/metrics.json
  analysis/decision.json
  review/raw-motion.mp4
  review/raw-mouths.mp4
  review/control-overlays.mp4
  review/certification-seams.mp4
  review/index.html
  banks/<arm>/certified/*.mp4
  banks/<arm>/runtime-manifest.json
  banks/<arm>/bank-manifest.json
  banks/<arm>/validation-report.json
  runtime/<arm>/<speech-id>/capture.mp4
  runtime/<arm>/<speech-id>/stdout.log
  runtime/<arm>/<speech-id>/result.json
  report/RESULTS.md
  report/artifacts.json
  logs/
  ComfyUI/
  .venv/
~~~

Large model weights may be reused through read-only paths or symlinks; do not duplicate all base checkpoints unnecessarily. New control adapter/preprocessor downloads belong to this pilot or a clearly recorded shared model cache, not SoulX's model directories.

### 4.1 Rules for every new helper

- Use argparse with a required --config absolute path. Print resolved inputs/outputs before costly work.
- Support --dry-run wherever a command would download, submit inference, prepare caches, or open runtime sessions. Dry-run must have no such side effects.
- Exit zero only on that stage's success. Use nonzero with a named gate and actionable reason otherwise.
- Store execution_kind: static_check, cpu_processing, gpu_inference, reused_historical, or runtime_validation.
- Use explicit subprocess argument arrays, check=True, and bounded network timeouts. No shell interpolation of arbitrary text or JSON.
- Refuse to overwrite a completed attempt. Resume only if input/config/source hashes match; otherwise allocate a new attempt.
- Record real timestamps, complete logs, source hashes, graph hashes, raw outputs, and failures.
- Make state.json changes atomically. It tracks stage, arm, attempt, input hash, outputs, status, and reason. Do not use the factory's shared state/ledger.json.
- State must distinguish planned, running, passed, failed, blocked, and pending_user_review.
- Human visual approval cannot be inferred from a numeric threshold, a contact sheet, or no reply.
- Keep scripts small and direct. This is an experiment harness, not a new generalized framework.

### 4.2 Standard shell context for future commands

~~~bash
export PILOT_ROOT=/workspace/experiments/soulx_ltx_motion_pilot_20260922
export PILOT_PY="$PILOT_ROOT/.venv/bin/python"
export PILOT_CONFIG="$PILOT_ROOT/config.json"
~~~

Until setup creates PILOT_PY, use the existing research interpreter for standard-library-only bootstrap scripts. Do not run a GPU model just to discover CLI options.

### 4.3 Required config.json fields

Implement a concrete configuration with these groups; reject unknown arm names and missing required fields:

~~~json
{
  "schema_version": 1,
  "experiment_id": "soulx_ltx_motion_pilot_20260922",
  "root": "/workspace/experiments/soulx_ltx_motion_pilot_20260922",
  "target_id": "sample_ai_human_facetime_closeup",
  "source_paths": {
    "donor_video": "/workspace/SoulX-FlashHead/benchmarks/pro_lite_150x_20260917/recreated_20260921/pro/video.mp4",
    "donor_metadata": "/workspace/SoulX-FlashHead/benchmarks/pro_lite_150x_20260917/recreated_20260921/pro/results.json",
    "donor_reference": "/workspace/SoulX-FlashHead/benchmarks/pro_lite_150x_20260917/reference-150x.png",
    "donor_audio": "/workspace/SoulX-FlashHead/benchmarks/pro_lite_150x_20260917/audio.wav",
    "target_bank": "/workspace/MuseTalk/assets/ltx23_pose_banks/sample_ai_human_facetime_closeup_production_v1"
  },
  "geometry": {
    "generation_width": 512,
    "generation_height": 832,
    "delivery_width": 480,
    "delivery_height": 832,
    "delivery_crop_x": 16,
    "delivery_crop_y": 0,
    "fps": 24,
    "smoke_frames": 49,
    "full_frames": 241
  },
  "generation": {
    "seed": 189,
    "sampler": "euler",
    "cfg": 1.0,
    "sigmas": [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0],
    "image_strength": 1.0,
    "control_strength": 1.0,
    "lora_strength": 1.0
  },
  "comfy": {"host": "127.0.0.1", "port": 18190},
  "limits": {
    "main_full_renders": 3,
    "optional_repair_full_renders": 1,
    "per_submission_timeout_seconds": 3600,
    "total_pilot_gpu_wall_seconds": 21600,
    "poll_seconds": 5
  },
  "certification": {"handle_frames": 6, "blend_frames": 6, "qp": 18},
  "musetalk": {
    "base_url": "http://127.0.0.1:8000",
    "playback_fps": 24,
    "musetalk_fps": 24,
    "batch_size": 4
  }
}
~~~

Verify the source paths before any script runs. Add literal prompts, resolved package/model revisions, preprocessor settings, and evaluation thresholds. Validators must reject placeholder strings. The six-hour ceiling is an experiment stop limit, not a runtime prediction. Record time spent on model setup separately from inference.

## 5. Phase 0 — preflight and input freeze

**Implement:** preflight.py and common.py. CPU/read-only checks first.

1. Read applicable AGENTS.md files. Snapshot git HEAD, branch, and status for MuseTalk, SoulX, and the existing LTX research ComfyUI/custom-node repositories.
2. Save current GPU model, UUID, physical/visible memory evidence, driver, running compute processes, CPU RAM, cgroup memory limit, and free disk. Read nvidia-smi; do not initialize Torch CUDA just for this step.
3. Record which existing ports/processes belong to whom. Check 18190 is free. If occupied, select a different unused loopback port and update the one configuration field.
4. Inventory donor/target assets; compute SHA-256; save ffprobe JSON and frame counts. Fully decode source videos to a null sink to reject damaged assets before generation.
5. Snapshot target bank and production manifest hashes for the final preservation check.
6. Verify donor portrait/audio hashes against section 2. If a known hash differs, stop and identify the difference.
7. Copy small immutable source fixtures into inputs with shutil.copy2, or read them by explicit immutable path while storing their hashes. Use lossless PNG intermediates. Do not edit source fixtures.
8. Inspect the donor and target images and donor video. The goal is to catch an accidental wrong file, hard face occlusion, or missing head/shoulder content; do not invent identity equivalence from filenames.
9. Record all package/environment paths and whether the required face-landmarker task exists.

Useful existing read-only commands:

~~~bash
nvidia-smi --query-gpu=name,uuid,memory.total,memory.used,driver_version --format=csv
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
git -C /workspace/SoulX-FlashHead status --short
git -C /workspace/MuseTalk status --short
ffprobe -v error -count_frames -show_streams -show_format -of json /workspace/SoulX-FlashHead/benchmarks/pro_lite_150x_20260917/recreated_20260921/pro/video.mp4
~~~

**Gate P0:** all source assets identifiable and readable, hardware provenance saved, no conflicting pilot root, no accidental source mutation. A busy GPU does not prevent CPU preparation; postpone only GPU-dependent phases.

## 6. Phase 1 — timing, anchor, and spatial alignment

**Implement:** prepare_inputs.py. Do not run an image generator.

### 6.1 Anchor and padding

Decode target idle frame 0 into target-anchor.png. Assert RGB shape (832, 480, 3). Compare the first six and last six decoded frames of the existing idle/smile clips before copying them. Save the actual decoded anchor hash.

Make target-guide-512x832.png by adding 16 edge-replicated columns to each side. Save the inverse crop specification. This padding is temporary context; those columns will never reach the final avatar frame.

### 6.2 Donor 25 → 24 FPS conversion

Do not reinterpret a 25-FPS stream as 24 FPS: that slows the gestures. Resample by timestamps and preserve approximately ten seconds.

Implement an explicit deterministic nearest-source-frame mapping:

~~~text
For output frame i = 0..240:
    target_time_seconds = i / 24
    source_index = min(floor(target_time_seconds * 25 + 0.5), 249)
    output_frame[i] = decoded_source[source_index]
~~~

Frame 240 uses the final available source frame. The 241-frame container is 10.0416667 seconds long; the last sample's timestamp is 10.0 seconds. Record the small held tail explicitly. Save every mapping row and source timestamp. No optical-flow interpolation, frame blending, stabilization, or temporal smoothing in the initial conversion.

The smoke fixture uses output frames 0..48 of this same mapping. It is a shape/memory smoke test, not proof of full-take quality.

### 6.3 Static donor-to-target alignment

Use a CPU Tasks FaceLandmarker to obtain full landmark arrays on donor frame 0 and the target anchor. Use eye outer corners (indices 33, 263) to compute eye midpoint, eye span, and roll. Derive one similarity transform that maps the donor's initial eye midpoint/span/roll onto the target's. Add the 16-pixel generation-canvas x offset last.

Store the transform as a 2×3 or 3×3 matrix plus its inverse, keypoint coordinates, chosen frame indices, and mapping error. Apply this **same transform to every donor frame/control frame**. Per-frame face centering or stabilization would erase the motion being tested.

If frame 0 has a failed detection or a blink that prevents stable alignment, choose the earliest usable frame among 0..12 and document it. Use its pose as the alignment reference without trimming or re-zeroing the video timeline.

For pose controls, transform coordinates before rasterization. For depth controls, warp the depth map with the fixed transform. Store the valid-pixel mask; fill uncovered background consistently using the corresponding static target depth map. Feather only the coverage boundary; do not overwrite valid moving face depth with target depth.

Do not distort donor anatomy with frame-dependent nonrigid warping in the first pilot. Normalizing initial pose/framing is not full cross-identity retargeting. Record this limitation.

**Gate P1:** annotated overlays show the donor's starting eye position/scale aligns with the target, and the subsequent head movement remains visible. All source and control timelines contain exactly 241 samples. Save the alignment overlay before expensive rendering.

## 7. Phase 2 — isolated environment and adapter setup

**Implement:** setup_environment.py. It may create pilot-local files, environments and dependencies after execution is authorized.

### 7.1 Reuse knowledge, isolate mutable components

Base research installation:

~~~text
/workspace/experiments/ltx23-soulx-transfer
~~~

Create a separate ComfyUI checkout under the pilot root at the **same resolved commit** as that installation, and a new venv using the same Python version. Clone required existing custom-node repositories at their recorded commits. Shared existing checkpoints can be read through model paths/symlinks.

Do not pip-install into the existing research venv or SoulX/MuseTalk environments. Do not upgrade their Torch/CUDA packages. Use the existing research package freeze as the initial exact constraints, then resolve only pilot-local additions. Save before/after package locks and the dependency diff. A new venv does not automatically inherit another venv's packages.

Use uv if available; otherwise normal venv/pip is acceptable. In the bootstrap helper, capture the existing interpreter's package freeze via subprocess, write it as a file, and install into the new environment. Check local path references/editable requirements before replaying them. If dependency resolution would replace the pinned Torch build, resolve that conflict explicitly; never accept an incidental major upgrade.

If the original ComfyUI/custom-node source is dirty, save its diff and explain whether the new checkout needs those local changes. Do not assume a commit hash reproduces an uncommitted runtime patch. Copy necessary changes only into the pilot checkout, with provenance.

### 7.2 Resolve only the required additional models

Download the compatible Union Control adapter from the official Lightricks repository. Resolve the repository commit, expected file size, and content SHA/LFS metadata first. Download to a temporary filename, validate, then atomically rename. Save a model lock including repository, revision, filename, size, and SHA-256.

Install/pin DWPose and Video Depth Anything preprocessors in the pilot checkout. Discover their exact node schemas and model downloads from the pinned source. Do not guess checkpoint filenames, download unrelated bundles, or silently substitute a different preprocessor. Record all downloaded model files and licenses/source URLs.

If the selected depth model cannot fit, use its documented smaller supported variant at the same method, record that choice before comparing arms, and label it precisely. A different depth method is a changed experimental arm, not an invisible fallback.

### 7.3 GPU ownership and server lifecycle

SoulX uses an advisory flock at /workspace/SoulX-FlashHead/.gpu-owner.lock. An empty file does **not** prove the lock is free. Inspect current processes as well; not every external job honors that lock.

Have a pilot runner hold the compatible nonblocking flock while its ComfyUI process or GPU preprocessor is alive. Record the child PID/start time. If the lock is owned, continue CPU work and wait; do not remove the lock file or stop the owner. Do not let an independent child outlive the lease-holding supervisor.

Start the pilot server on loopback only using the existing launcher pattern, adapted to the new checkout:

~~~bash
"$PILOT_PY" "$PILOT_ROOT/ComfyUI/main.py" \
  --listen 127.0.0.1 --port 18190 --disable-auto-launch \
  --lowvram --reserve-vram 3 --cache-none \
  --disable-pinned-memory --disable-async-offload \
  --preview-method none --use-pytorch-cross-attention
~~~

Launch from the pilot ComfyUI directory, under the runner, with stdout/stderr captured. Change the port only through config. The 3-GiB reserve comes from the historical local adaptation; do not treat it as guaranteed fit for control inference.

Save GET /system_stats, GET /object_info, server commit, and model filenames. These read endpoints are not generation. GPU preprocessors and diffusion must run sequentially on this host unless independently established safe.

**Gate P2:** isolated environment imports successfully; schemas and model revisions are saved; the adapter file and compatible loading path are identified; no unrelated environment or service changed. Actual weight application is confirmed by the graph audit and guided smoke runs, not by file presence alone.

## 8. Phase 3 — control extraction

**Implement:** extract_controls.py. First generate preprocessing-only graphs, then save their exact outputs as lossless image sequences. Diffusion should consume these saved signals so that preprocessing is not rerun differently per arm.

### 8.1 Pose arm

- Use DWPose, body enabled, face enabled, hands disabled.
- Save numeric keypoints/confidences as well as the rendered standard control image.
- Preserve the pinned preprocessor's expected background, point colors, and connectivity. Do not draw an arbitrary MediaPipe mesh and call it a DWPose control.
- Estimate on the donor RGB sequence, transform coordinates using section 6, then rasterize at 512×832.
- Do not crop away the face to favor body pose; head movement is the target.
- Keep facial landmarks including lips in A1_POSE. This is the unmodified standard-control baseline. If original articulation transfers, label it as such and apply the bounded repair rule later.
- If numeric keypoints are unavailable from the node, use the pinned package's documented estimator API; save that API/version. Do not reverse-engineer coordinates from lossy rendered video.

### 8.2 Depth arm

- Use the pinned Video Depth Anything estimator with its documented input normalization, RGB channel order, and near/far polarity.
- Estimate depth on the source donor frames and a target anchor image.
- Normalize the donor sequence consistently over the clip using fixed robust near/far percentiles, recording the numeric bounds. Avoid independent per-frame min/max scaling, which can make apparent depth pulse.
- Apply the static donor transform; fill uncovered regions with consistently scaled target static background depth. Record the scale/shift used to put target and donor relative depth in compatible display ranges.
- Review that face turn direction and foreground/background ordering are preserved. Relative depth is not metric 3D motion capture.
- Save 512×832 lossless control images and any original higher-precision depth arrays. Let the guide encode path perform the adapter's required half-size processing; do not accidentally downscale twice.

### 8.3 Control validity and visual check

Produce control-validity.json with per-frame detection status, confidence summaries, coverage mask area, frame count, FPS, and hashes. Review a labeled donor/pose/depth side-by-side at normal speed and at the same timestamps.

If detection fails in more than 5% of frames or for a continuous gap over 0.25 seconds, fail that control arm before diffusion. For gaps of at most two frames, optional coordinate interpolation is allowed only if marked per frame and if total gaps remain below the limit. Longer failures must remain visible in diagnostics, not silently time-compressed.

Use a CPU task tracker separately for evaluation even if DWPose runs on GPU. Control-estimator success does not prove evaluation success or vice versa.

**Gate P3:** valid, time-aligned control sequences with visible donor movement and no missing-frame compression. A bad control signal is not evidence that LTX cannot transfer motion.

## 9. Phase 4 — build and inspect the actual ComfyUI API graphs

**Implement:** build_graphs.py. All generated API graphs must be plain node-ID dictionaries, not GUI graph exports.

### 9.1 Loader and schema policy

Use the existing local Q4 graph to obtain actual loader names and model filenames. Verify every new node and input against saved object-info.json. The newer local ComfyUI core has GetICLoRAParameters and an LTXVAddGuide input named iclora_parameters; older official examples use LTXICLoRALoaderModelOnly and LTXAddVideoICLoRAGuide.

Choose **one** compatible path and record it in source-lock.json. Prefer the installed core path if its loader preserves adapter metadata with the GGUF model. Do not mix node names/fields from different APIs or assume generic LoRA loading retained reference_downscale_factor.

Validate the adapter produces a reference downscale factor of 2 and that the actual guiding latents have the intended dimensions. Inspect loader output/metadata and one executed smoke history. No missing-key or unmatched-weight warning may be ignored without accounting for every affected key.

If the existing GGUF loader cannot correctly apply the adapter, stop at a compatibility failure with the exact error. Do not silently download a 22B full-precision base or upgrade the active stack to make the graph run.

### 9.2 Required graph topology

Construct the graph by semantic node roles, using a saved role-to-node-ID map. The required flow is:

~~~text
GGUF model loader ── [Union LoRA loader only for A1/A2] ── sampler model
GGUF text encoder ── positive/negative text conditioning
Video VAE ────────── reference/guide encoding
Target anchor PNG ── image guide at frame 0
Empty video latent: 512×832, N=49 or 241
Control PNG sequence ── video guide with IC-LoRA parameters [A1/A2 only]
Empty audio latent matched to N / 24 FPS
Video+audio latent concat ── Euler sampling, eight steps, seed 189
Separate sampled video latent ── crop appended guide tokens/frames
Save video latent ── separate decode-only process ── tiled VAE decode
Video output: native 512×832, exactly N frames, 24 FPS, no saved audio
~~~

Retain the ordinary portrait guide AND the structural video guide in guided arms. Keep their parameter paths separate: only the structural control uses the adapter's reference downscale parameters. A portrait guide incorrectly downscaled as control may weaken identity.

The current joint audio/video base can keep empty audio latents for architecture compatibility. Do not attach donor audio, do not provide a spoken transcript, and do not decode or mux generated audio into the base video. Discarding audio output does not guarantee closed lips; that is evaluated visually.

Use a standard image-sequence loader supported by the pinned nodes, with explicit sorted zero-padded filenames and exact count. If none exists, add a minimal pilot-local ComfyUI input node that reads a configured directory in sorted order, enforces RGB/count/shape, and returns an IMAGE batch. It must not resample time, resize, or silently skip corrupt images.

Guide cropping must consume the **conditioned** positive/negative metadata from the guide chain. Do not wire the original unconditioned text into crop-guides: that can leave appended reference frames in the decoded output. Assert exact output frame count afterward.

### 9.3 Memory-safe output strategy

Generate and save final cropped video latents first. Fully unload/stop the **pilot-owned** diffusion server before starting a decode-only process. Then decode with only the video VAE resident.

Initial tiled decoder settings copied from the local adaptation: tile_size 256, overlap 64, temporal_size 16, temporal_overlap 8. Check actual node schema. Tiling can introduce artifacts; keep identical settings in all arms and include seam review.

Do not use old helper scripts that select the lexicographically last latent from a shared directory. Select the exact latent returned in this submission's history. Persist its content hash and the corresponding graph hash.

Only if memory headroom and a successful smoke explicitly support it may generation/decode share a process. Do not repeat the historical full-length host-RAM OOM by assuming a short smoke proves a ten-second graph is safe.

### 9.4 Static graph validation

Before POST /prompt:

1. Every class exists in object-info.json.
2. Every required input exists and enum value is valid.
3. Every edge points to an existing node and valid output index/type.
4. Output dependency traversal reaches the intended target image, sampler, and, for guided arms, control sequence and adapter.
5. A0 has neither adapter nor control input; A1/A2 share all unrelated fields.
6. Resolution and frame counts are explicit constants; no stale calculator overrides them.
7. No historical “woman speaking a product advertisement” prompt survives from the template.
8. No NAG/Relay/old stage-2 branch remains reachable.
9. No output points into the production bank or previous experiment directories.
10. Complete node dictionary and semantic role map are saved and hashed.

**Gate P4:** an auditable graph with no unresolved schema or connectivity assumptions. HTTP graph acceptance alone is not inference success.

## 10. Phase 5 — smoke runs and main renders

**Implement:** run_comfy.py. Its responsibilities are ownership, submission, telemetry, polling, exact output collection, media facts, state updates, and failure preservation.

### 10.1 Submission contract

- POST /prompt with JSON containing prompt and a unique client_id.
- Save the whole response, including prompt_id and node_errors. Any node_errors is a failed submission.
- Poll the exact /history/<prompt_id> every five seconds with bounded request timeouts.
- Read actual status and exception messages. A returned history entry is not necessarily success.
- Collect outputs only from this prompt's history and resolve them under the pilot-owned output directory. Reject path escapes and stale mtime/hash reuse unless intentionally declared.
- Save GPU device/process memory and utilization, CPU RSS, and cgroup memory use at roughly one-second intervals, with run timestamps and process identity.
- Record setup/load time, generation time, decode time, media-processing time, and total wall time separately. Sampling may miss peaks; do not call a sampled device peak an allocator peak.
- On timeout, save diagnostics. Interrupt only this pilot-owned server's task, and confirm it is stopped before another submission. Do not POST a global interrupt to a shared server.

### 10.2 Run order

1. A0_PORTRAIT smoke, 49 frames.
2. A1_POSE smoke, 49 frames.
3. A2_DEPTH smoke, 49 frames.
4. Review smoke outputs for valid identity, nonblack/noncontrol-map output, sane motion, and exact geometry/count.
5. A0_PORTRAIT full, 241 frames.
6. A1_POSE full, 241 frames.
7. A2_DEPTH full, 241 frames.
8. Analyze before deciding on optional A3_REPAIR.

Do not require strong movement in the first two seconds to pass a smoke; that interval may be naturally quiet. Smokes establish execution/shape/memory sanity. The full take establishes movement fidelity.

For each decoded native output, preserve an unaltered version and produce delivery.mp4 with crop=480:832:16:0, 24-FPS timestamps and no audio. Prefer lossless intermediates for metrics; save H.264 copies for review. Do not let VideoHelperSuite pad to generated audio duration. Require exact requested frame count and full ffmpeg decode.

### 10.3 Resource/error rules

- CUDA OOM: release all pilot-owned generation models, confirm process exit, and retry the same arm once using sequential generation/decode and existing offload settings. Do not stop the quantized-PRO worker.
- Host-RAM OOM: retain failure evidence, use saved-latent/separate-decode execution and avoid simultaneous full-frame float32 copies. If already doing this, report resource blockage rather than reducing quality silently.
- Shape failure around guides: inspect half-resolution factor, image vs structural parameters, length multiple, and crop-guides metadata. Do not “fix” the final MP4 by trimming unexplained extra generated frames.
- Control-map-looking output: check adapter/base version, control polarity, strengths and graph wiring before interpreting appearance.
- Any geometry, schedule, precision, or tiling change creates a new profile; rerender matched arms if the change is necessary to compare them.
- Maximum three main full renders, one optional repaired full render. One operational retry per failed submission. Stop at the configured total GPU wall limit and report remaining work.

### 10.4 Intended command interfaces, after implementation

~~~bash
"$PILOT_PY" "$PILOT_ROOT/scripts/build_graphs.py" --config "$PILOT_CONFIG" --all --dry-run
"$PILOT_PY" "$PILOT_ROOT/scripts/build_graphs.py" --config "$PILOT_CONFIG" --all
"$PILOT_PY" "$PILOT_ROOT/scripts/run_comfy.py" --config "$PILOT_CONFIG" --arm A0_PORTRAIT --length smoke
"$PILOT_PY" "$PILOT_ROOT/scripts/run_comfy.py" --config "$PILOT_CONFIG" --arm A1_POSE --length smoke
"$PILOT_PY" "$PILOT_ROOT/scripts/run_comfy.py" --config "$PILOT_CONFIG" --arm A2_DEPTH --length smoke
~~~

Use --length full only after the smoke gate. Implement --resume so it checks hashes rather than merely skipping a filename that happens to exist.

## 11. Phase 6 — motion, mouth and visual evaluation

**Implement:** analyze_motion.py and build_review.py. Do this on raw delivery clips **before** boundary certification can disguise errors.

### 11.1 Landmarks and coordinate definitions

Use CPU Tasks FaceLandmarker with the existing task model. Request full landmarks and, if supported, face transformation matrices. Save one row per frame with explicit PTS/time, detection status, and normalized landmark coordinates. Never remove missed detections from the timeline.

Use the same tracker/version for donor, target anchor, A0/A1/A2, and V14 reference. The donor is measured on the exact time-resampled sequence and transformed into target coordinates. If rendering transformed donor RGB for analysis introduces empty borders, mark them; do not measure a different crop without updating the transform.

Primary trajectories:

- Eye-midpoint x/y displacement, centered at its alignment reference and divided by that sequence's fixed reference eye span.
- Roll from eye-corner line, unwrapped and centered at the alignment frame.
- Log eye-span ratio relative to the alignment reference, as a scale-change proxy.
- Nose displacement in an eye-aligned coordinate system for yaw/pitch **proxies**.
- Inner lip separation (13, 14) divided by eye span; optional mouth width and jaw/blink blendshape outputs.

Name 2D yaw/pitch proxies accurately. If using the optional transformation matrix for 3D estimates, document matrix axes, rotation extraction, and calibration; do not claim ground-truth degrees from uncalibrated 2D proxies.

### 11.2 Metrics and masks

Compare at matching timestamps, initially without any time shift. Report:

1. Valid detection fraction and longest missing interval.
2. Per-axis donor/candidate Pearson correlation at zero lag.
3. RMSE divided by donor robust motion range (p95−p5), with degenerate-axis handling.
4. Candidate/donor robust amplitude ratio.
5. Estimated best lag within ±0.5 seconds, reported as a diagnostic alongside zero-lag results; never use dynamic time warping to hide delayed gestures.
6. Velocity and acceleration distributions, plus largest discontinuities.
7. Mouth opening distribution, open-mouth intervals, jaw changes, and visible teeth observations.
8. Head-framing excursions: forehead/chin clipping, gaze drift, shoulders leaving expected bounds.

Use pairwise-valid samples; no bridging derivatives across detection gaps. Do not score near-constant donor axes: initial floors are 0.02 eye-span units for displacement, 1 degree robust roll range, and 0.02 log-scale range. Save both valid and excluded axes. If fewer than two axes contain measurable movement, mark numeric fidelity inconclusive and rely on a labeled trajectory/visual inspection; do not reward a frozen output.

For mouth closure, compare with the target anchor and existing idle using the same tracker. Flag any frame with normalized opening above max(0.03, idle p95 + 0.015), and any run lasting more than three frames. These are **heuristic review flags**, not universal physiological thresholds. Visible teeth, repeated phonetic articulation, or a moving chin outside MuseTalk's blend area can fail an arm even if the lip metric passes.

### 11.3 Proposed pilot gates

These are predeclared engineering targets, not validated population thresholds:

| Gate | Initial criterion |
|---|---|
| Detection validity | ≥95% valid frames; no gap >0.25 s |
| Measurable donor movement | At least two informative trajectory axes, otherwise report inconclusive metrics |
| Motion resemblance | Median zero-lag correlation ≥0.70 over informative primary axes |
| Relative motion amplitude | Median amplitude ratio 0.6–1.4; no obvious sign reversal |
| Timing | Median best-lag magnitude ≤0.15 s; zero-lag scores remain primary |
| Evidence that guidance helps | Guided median correlation exceeds A0 by ≥0.15, or show a clear lower-error advantage and explain the mixed result |
| Mouth suitability | No sustained speech-like articulation or visible teeth in normal-speed review; flags above inspected individually |
| Identity/framing | Target appearance remains recognizable and stable, no cropping or background distortion that distracts at call size |

Do not average identity failure away with a good motion score. Do not call lip-opening correlation a lip-sync accuracy score. Same seed across guided and unguided LTX arms does not make their generated facial pixels directly comparable as a decoder-only test.

### 11.4 Review artifacts

Create labeled, normal-speed comparisons containing donor, A0, A1, A2 and REF_V14. Put REF_V14 in a clearly marked historical-reference section because it has different movement, length and generation recipe. Do not invent temporal correspondence between V14 and the donor.

Required views:

- Full-frame movement comparison with synchronized timecodes for donor/A0/A1/A2.
- Fixed relative mouth crops with labels; preserve original aspect ratios and disclose nearest-neighbor magnification.
- Pose/depth overlays beside their donor input.
- Curves of eye translation, roll, scale and mouth opening at matching timestamps.
- A contact sheet as a supplement, never as the sole quality gate.

Do not add labels to metric inputs. Do not sharpen, smooth, stabilize, interpolate, or silently trim bad intervals in comparison videos. Show failed arms too. A local HTML index should link every video, metric file, source and profile.

### 11.5 Optional single repair decision tree

Choose only one of these, save the choice in decision.json, and call it A3_REPAIR:

1. **Pose transfers head motion but also obvious speech shapes:** create a pose representation that keeps rigid head/body points but omits mouth/jaw articulation points using the pinned estimator's documented landmark topology. Save exactly which points/edges were removed. Do not assume MediaPipe indices are DWPose indices. Leave missing features absent rather than drawing a black rectangle over the RGB face. Use the same seed/prompt/strength. This is an experimental out-of-training-distribution guide variant, not a guaranteed fix.
2. **The better guided arm follows too weakly but identity/mouth are good:** change only control-guide strength from 1.0 to 1.25; keep LoRA strength fixed.
3. **The better guided arm overconstrains shape or leaks artifacts but has useful motion:** change only control-guide strength from 1.0 to 0.65; keep LoRA strength fixed.
4. **Both guided arms fail for unrelated severe reasons:** do not use a speculative reroll. Report the failure and recommend a separately scoped next step such as Motion Track Control or explicit pose retargeting.

One repaired full render is the limit. Do not keep changing prompts or seeds until something looks good. If A3 improves only one criterion while damaging another, retain the tradeoff rather than declaring victory.

**Gate P6:** decision.json names the candidate to carry forward, rejected arms and reasons, or explains why no candidate qualifies. Agent visual assessment and user approval are separate fields. A plausible candidate may proceed to a reversible test-only runtime demonstration while final subjective approval remains pending; no production activation follows automatically.

## 12. Phase 7 — experimental three-clip bank and strict certification

**Implement:** certify_package.py. Do not invoke the stock factory's roster-dependent CLI. Import its pure helpers through a controlled sys.path or copy small functions into the pilot with attribution/hash.

Create a candidate bank and a comparable V14-reference bank under the pilot root. Each contains only three physical files:

~~~text
idle_active_listening.mp4
speaking_direct.mp4
light_smile.mp4
~~~

For the candidate, speaking_direct is the selected 241-frame guided render. For the V14 reference bank, it is the existing 289-frame V14 clip. Preserve each actual duration; do not truncate V14 and call the cut an accepted production loop. Idle and smile come from the existing target bank.

### 12.1 Re-encode the whole pilot bank consistently

Use target-anchor.png as the canonical anchor for all three files. Apply six repeated anchor frames at the beginning and end, plus six interior blend frames at each end. Apply the same procedure/QP to reference-bank copies as to candidate-bank copies; do not modify the production originals.

Start at fixed x264 QP 18, H.264/yuv420p, 24 FPS, no audio, forced keyframes at frame 0 and frame N−6. Reuse encode_frames, but never assume these settings alone prove decoded equality.

For N=241 and handle=blend=6:

- Frames 0..5: canonical handle.
- Frames 6..11: fade from anchor into motion.
- Frames 12..228: unmodified interior before encoding.
- Frames 229..234: fade back toward anchor.
- Frames 235..240: canonical handle.

This modifies approximately the first and last half-second. Report motion scores both before certification and on the unchanged temporal interior afterward. Do not claim exact donor fidelity in the treated boundary region.

### 12.2 Required decoded-pixel assertions

After final encode, fully decode all physical files and explicitly assert:

1. Every first-handle frame is identical to every other first-handle frame in that file.
2. Every final-handle frame is identical to that file's first-handle frame.
3. That canonical decoded RGB frame is identical across all three files.
4. Every ordered nonself transition's tail handle equals its destination's head handle: three files mean six directed pairs.
5. Each file's tail/head equality also covers its self-loop: three additional cases.
6. Resolution, frame count, FPS, codec, pixel format, audio-stream absence and full decode pass.

**Important:** the factory helper boundary_hash concatenates head and tail. Matching concatenated hashes across clips is not by itself a proof that a head equals a tail. Perform the explicit equalities above and then hash the shared decoded handle as an artifact identifier.

If QP 18 fails equality, retry **all clips in the affected pilot bank** consistently at QP 14, then QP 0 if needed. Save failed encoding reports. If equality still fails, stop and diagnose prediction/color-conversion behavior; do not disable the gate.

Create a CPU negative-control test that changes one decoded tail pixel/frame in memory and verifies the validator names the offending file/transition. Do not corrupt a real source or packaged asset.

### 12.3 Perceptual boundary gate

Make a normal-speed montage of all six ordered transitions and all three self-loops, with labels and at least two seconds on each side where available. Inspect fades for double faces, abrupt position changes, frozen pauses, eye flashes, and chin/shoulder discontinuities.

Exact handles can pass while blends look bad. If a candidate fails perceptual continuity, do not enlarge fades repeatedly to hide large pose differences. Permit one documented blend-length trial of 12 frames, applied consistently in a separate bank revision, only if the problem is a short fade and the interior motion is otherwise acceptable. Keep the six-frame exact handles. Otherwise report the candidate as unsuitable for this loop bank.

### 12.4 Manifest mapping: retain six logical IDs

| Logical pose | Physical file | Role |
|---|---|---|
| neutral_resting | idle_active_listening.mp4 | idle |
| active_listening | idle_active_listening.mp4 | listening |
| speaking_direct | speaking_direct.mp4 | talking |
| nod_agree | idle_active_listening.mp4 | reaction |
| empathetic_head_tilt | idle_active_listening.mp4 | reaction |
| light_smile | light_smile.mp4 | reaction |

Nod and empathy are compatibility aliases; the pilot does not claim they perform those reactions. Document them in the bank manifest and review labels.

Set version=1, default_pose_id=neutral_resting, switch_mode=next_boundary, test_only=true, activation_status=experimental_pending_review. Set switch_safe=true only after the strict boundary gate passes; explain this is a mechanical endpoint guarantee, not production approval.

Use a distinct pose_set_id, such as sx50_pilot_target01_a1_r01. Use unique physical avatar IDs containing arm/revision and the first 12 characters of the **final certified MP4 hash**. Aliases to the same physical file share the same avatar ID. No variants array is needed; remove inherited V14/V15 variant rotation.

Every pose entry must include the actual file's frame_count, fps, duration_seconds=N/24, cycle_seconds=N/24, role, avatar_id, and plain basename asset_file. The asset-dir passed to the harness must be the certified directory.

Validate using BOTH the pilot's stricter media checks and the real load_pose_set in MuseTalk/scripts/test_pose_webrtc.py, plus the actual pose-protocol validator as applicable. Do not rely only on the factory's duplicated validator.

**Gate P7:** experimental manifests load; three unique physical caches resolve; all boundary and media assertions pass; perceptual seam review is recorded.

## 13. Phase 8 — MuseTalk cache preparation and runtime proof

**Implement:** run_webrtc.py. Complete the reversible runtime experiment; do not activate the bank as a default.

### 13.1 Keep runtime comparisons controlled

Use the same already-validated MuseTalk backend for the reference and candidate. Do not switch to TAESD, change TensorRT engines, or start a decoder optimization experiment during this pilot. Record actual backend, FPS, batch size, model versions, GPU and co-resident processes.

Run one session at a time. Release pilot LTX/preprocessor GPU allocations first. Read GET /health and GET /capabilities or the actual capability endpoint used by the harness. Do not assume a running server's profile from the name of a launch script.

If no suitable MuseTalk server is running, inspect existing startup/configuration and start an isolated test server only with an understood, documented command and unused port. **Do not blindly run /workspace/run-musetalk-local-trt.sh**: it delegates to a broad startup script that may do more than launch one API process. If an isolated service cannot be started without conflicting with active work, report that specific runtime blocker; preserve the completed offline experiment.

### 13.2 Audio fixtures

Use two existing, distinct waveforms:

- Speech A: /workspace/MuseTalk/data/audio/eng.wav.
- Speech B: /workspace/SoulX-FlashHead/benchmarks/pro_lite_150x_20260917/audio.wav.

Decode/canonicalize copies to 16-kHz mono PCM WAV in inputs. Save source/canonical hashes and duration. Verify the two normalized waveforms differ; reject accidental duplicate content. Speech B is the donor's original speech; Speech A tests different dialogue. Do not regenerate TTS unnecessarily.

For each bank use exactly the same audio files and start conditions. Do not compare candidate with one sentence and reference with another. If neither fixture lasts beyond the longer of the reference and candidate talking loops, construct a clearly labeled two-repeat fixture with a 0.5-second silent gap and use it for both banks; verify its resulting duration exceeds both loops and keep the unmodified originals too.

### 13.3 Existing harness invocation

Choose the known working MuseTalk interpreter after checking imports; the historical path is /workspace/.venvs/musetalk_trt_stagewise/bin/python. Verify this at execution time. The pilot interpreter may lack aiortc or other MuseTalk runtime dependencies.

The following uses existing CLI flags. Substitute the chosen bank/audio/output paths explicitly:

~~~bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python \
  /workspace/MuseTalk/scripts/test_pose_webrtc.py \
  --base-url http://127.0.0.1:8000 \
  --manifest "$PILOT_ROOT/banks/A1_POSE/runtime-manifest.json" \
  --asset-dir "$PILOT_ROOT/banks/A1_POSE/certified" \
  --audio-file "$PILOT_ROOT/inputs/speech-a.wav" \
  --prepare-missing \
  --reaction-intent none \
  --speaking-case-count 1 \
  --musetalk-fps 24 --playback-fps 24 --batch-size 4 \
  --record-output "$PILOT_ROOT/runtime/A1_POSE/speech-a/capture.mp4" \
  --record-postroll-seconds 2 \
  --completion-timeout 300
~~~

The wrapper must create output directories first, capture stdout/stderr to files, record the return code, and parse/preserve the harness's final JSON object separately from progress lines. It must not claim success from MP4 existence alone.

The existing harness prepares missing caches and warms them. **Never pass --force-recreate for an existing ID.** Final clip hashes define new pilot IDs. Runtime cache creation is the only intended write into MuseTalk's ordinary cache storage.

Run the V14-reference bank and candidate bank with both Speech A and Speech B. Use --reaction-intent none for the controlled lip-sync runs. Then make one additional candidate showcase with --reaction-intent warmth and --showcase-six-poses to exercise idle, talking, smile and the documented aliases. Use --showcase-timeout 180 and --showcase-initial-neutral-seconds 2 for this showcase. It is not six distinct behaviors.

If the server cannot maintain this rate, preserve the failed result. A lower-FPS exploratory comparison must rerun both banks with identical rates and be labeled a different profile; do not present it as a 24-FPS pass.

### 13.4 What to inspect in runtime output

- Correct pose_set_id and avatar IDs used; no accidental V14/V15 rotation in candidate.
- Completion and neutral recovery; no missing/corrupt recording frames.
- Receiver timestamp validation from the harness and full decoded MP4 validation.
- Speech A actually drives the new mouth despite the donor having spoken Speech B.
- Visible donor phonetic shapes do not leak into nonblended jaw/cheek areas.
- No doubled lip edges, mask flicker, teeth ghosts, or large head motion that defeats the face crop.
- Talking self-loop and return to idle/smile do not visibly jump.
- Playback looks acceptable at normal speed in full frame and mouth crop.

RTP/timestamp correctness proves media timing consistency, not phoneme accuracy. Do not call landmark correlation with the original donor a new-dialogue lip-sync score. Record perceptual judgments as such.

**Gate P8:** recorded reference/candidate runtime comparisons exist, actual harness checks pass, and limitations are documented. If subjective quality needs the user's review, leave that status pending rather than changing production settings.

## 14. Focused CPU checks to implement before expensive runs

Write only tests that prevent meaningful experiment mistakes:

1. 250@25 → 241@24 timeline mapping is monotonic/in-bounds and keeps the declared time span; last sample is explicitly held.
2. Padding then cropping returns the original target anchor bytes exactly.
3. Static similarity transform correctly maps synthetic landmarks and is unchanged across frames; a moving synthetic trajectory remains moving.
4. Graph validator detects missing control edges, stale output paths, wrong guide downscale, and unreachable adapter nodes.
5. Masked metric calculation keeps missed-frame timestamps and refuses correlations on degenerate axes.
6. Bank validator catches a changed tail frame even if other clips still share their concatenated hash.
7. Manifest contains six logical poses but exactly three physical identities; duration fields come from real frame counts.
8. Resume rejects stale config/source hashes and cannot overwrite a completed attempt.

These tests do not replace the GPU smoke, real media decode, or visual review. Do not import or initialize diffusion weights in CPU tests. Run once after implementing the relevant helpers; rerun only affected tests after meaningful changes.

## 15. Reporting and handoff

**Implement:** report.py. Produce one report/RESULTS.md, one machine-readable artifact index, and the local review index. Include failures and rejected arms.

Report order:

1. Verdict: promising / unsuccessful / inconclusive / runtime-blocked; one sentence explaining why.
2. Exact target avatar, donor file/hash, source-model profile, and whether user selected this specific take.
3. Hardware/runtime provenance near the top; distinguish reused historical evidence, CPU work, fresh GPU inference, and WebRTC validation.
4. Method: control representation, geometry, timing conversion, model revisions, generation settings, and any deviations from this plan.
5. Per-arm table: status, generation/decode wall time, sampled memory, motion metrics, mouth flags, identity assessment, boundary result, runtime result.
6. Links to full-frame and mouth comparisons, donor video, controls, curves, seam montage and receiver recordings.
7. What was actually established, with explicit nonclaims about cross-identity reuse, speech prosody, exact motion cloning, and concurrency.
8. Recommended next action based on the result. Do not automatically add more characters or controls.
9. Source-preservation check and remaining pilot-owned processes/caches.

Verify production bank/manifest hashes still match preflight. Review git status for both repositories and attribute only known pilot changes. Stop/release pilot-owned GPU processes and lease, leaving artifacts intact. Record experimental avatar IDs so they can later be removed deliberately; do not delete arbitrary cache directories.

### 15.1 Definition of done checklist

- [ ] One target avatar, exact donor/video provenance, no hidden fixture substitution.
- [ ] Explicit timing map and static spatial alignment saved.
- [ ] Base/adapter/preprocessor versions and compatible schemas locked.
- [ ] Three main arms completed or each failure explained with evidence.
- [ ] Optional repair stayed within its stated scope and budget.
- [ ] Raw motion and mouth metrics evaluated before certification.
- [ ] Labeled full-frame/mouth/control comparisons available at normal speed.
- [ ] Candidate selected for a stated reason, or a defensible negative result reported.
- [ ] Three physical clips with six logical IDs; no production protocol edit.
- [ ] Explicit decoded head/tail/cross-clip/self-loop equality verified.
- [ ] Perceptual seams checked separately from pixel equality.
- [ ] Reference/candidate MuseTalk recordings on two distinct audio fixtures, or exact runtime blocker documented.
- [ ] No production activation; subjective approval remains honestly represented.
- [ ] Existing work and source assets preserved; only pilot-owned processes cleaned up.

## 16. Luna execution checklist, in order

This is a sequence, not permission to skip the detailed gates above.

1. Read this file and repository instructions completely; record current dirty work.
2. Create the fresh pilot directory, config, common helper and state file.
3. Implement preflight; freeze input hashes and source facts.
4. Build the isolated environment and resolve the compatible adapter/preprocessors; install the CPU landmark dependency before using it.
5. Prepare target anchor, timing map and donor alignment; inspect overlays.
6. Extract and inspect standard pose/depth controls; save lossless artifacts.
7. Build graphs from actual node schemas; run the focused CPU contract tests.
8. Acquire GPU ownership and perform the three short smoke runs sequentially.
9. If smoke gates pass, perform the three full renders sequentially with separate decode.
10. Compute raw motion/mouth metrics and make review videos.
11. Apply at most one documented repair if the decision tree warrants it.
12. Package candidate and V14-reference banks locally; validate exact boundaries and visual seams.
13. Release LTX GPU resources; prepare only new MuseTalk IDs and record controlled runtime comparisons.
14. Write the final report, validate artifact links, and leave the experiment reviewable.
15. Tell the user the result and link the most useful comparison video and report. Do not end with only benchmark numbers or a claim that “the pipeline ran.”

## 17. Why this pilot is bounded this way

The existing MuseTalk architecture already moves expensive base-video creation offline. This pilot changes where the base motion comes from. It does not by itself accelerate MuseTalk's live inference, transfer PRO's dental renderer into MuseTalk, or make a repeated gesture track respond to new prosody.

If it works, the next step is to preserve the selected control sequence as an immutable reusable motion asset and try one additional target identity. If it fails because fine head orientation is lost, the next step is a separate Motion Track Control or explicit 3D-pose-retargeting experiment. If it fails because speech articulation leaks through, solve that separation before expanding the roster. These later steps are recommendations, not automatic extensions of this pilot.

## 18. TASKS — the executable work queue

### 18.1 How Luna must use this list

This list is the execution checklist. Do not treat it as background reading or replace it with a vague new plan. Complete tasks in dependency order. Each checkbox means the task's stated output and verification are complete, not merely that its code was written.

Mirror task IDs and state in the pilot's state.json. Once execution begins, maintain checkboxes in a working copy at the pilot root named EXECUTION_PLAN.md; retain this source plan as the original specification. Add links to evidence under checked tasks in the working copy. A failed or blocked task stays unchecked and has an explicit status/reason in state.json. A conditional task not needed is marked not_applicable there, with the condition that made it unnecessary; do not invent a successful run for it.

At the start of each resumed session:

1. Read config.json, state.json, the most recent task log, and this working checklist.
2. Confirm saved source/config hashes still match the inputs used by completed tasks.
3. Reconcile any task marked running with its exact PID/start time, Comfy prompt_id and history. Do not submit a duplicate while its original may still be executing.
4. Find the earliest incomplete task whose prerequisites passed.
5. Continue that task; do not regenerate completed media to recover conversational context.

After each task, record a short entry with task ID, timestamp, commands or actions, outputs, verification, status, and next task. During long GPU work, update the user at least once per minute with the running arm/stage and any new finding. Save process status before yielding; do not abandon a running job untracked.

Do not start another agent merely because the executor is called Luna. This is a single-agent execution plan. CPU preparation may overlap unrelated waiting, but pilot GPU jobs are sequential.

### 18.2 Bootstrap, inputs and environment

- [ ] **T01 — Read instructions and establish scope.**
  Prerequisites: explicit user instruction to execute. Read this entire plan and applicable AGENTS.md files. Inspect repository status; record pre-existing changes. Output: initial logs/T01.md explaining target, donor, scope and protected files. Verify: no source or service mutation occurred. Next: T02.

- [ ] **T02 — Create the pilot root, actual config and resumable state.**
  Prerequisites: T01. Create the directory tree in section 4 and fill config.json with the exact paths and prompts in this document. Implement common.py for safe paths, hashes, subprocess capture, atomic state and attempt allocation. Output: config.json, state.json, working EXECUTION_PLAN.md. Verify: no placeholders, root is new or verified resumable, no path resolves to an existing bank output. Next: T03.

- [ ] **T03 — Implement and run read-only preflight.**
  Prerequisites: T02. Implement preflight.py using standard library, ffprobe/ffmpeg and nvidia-smi; do not require the not-yet-created pilot venv. Save hardware/process/disk/source snapshots and asset hashes. Verify: P0, including donor media existence and known portrait/audio hash checks. On missing donor: report fixture failure and search retained files; do not regenerate automatically. Next: T04.

- [ ] **T04 — Create isolated interpreter and code checkouts.**
  Prerequisites: T03. Implement setup_environment.py; resolve commits and package constraints, create the pilot ComfyUI checkout/venv, preserve any necessary local-source patches with hashes. Output: environment source/package locks and dependency diff. Verify: no package upgrades in other environments; no undocumented Torch/CUDA replacement. Next: T05.

- [ ] **T05 — Resolve adapter and preprocessing models.**
  Prerequisites: T04. Download only the compatible official Union adapter and pinned preprocessor models. Record immutable revisions, sizes, hashes, actual filenames and local paths. Install pilot-local DWPose, depth and CPU Tasks FaceLandmarker dependencies. Verify: package consistency and model file checks; existing large LTX base checkpoints are reused without modification. Next: T06; CPU T07 may proceed while waiting for the GPU.

- [ ] **T06 — Implement GPU ownership and capture actual node schemas.**
  Prerequisites: T04–T05. Implement the process/lease portion of run_comfy.py. Start only the pilot server, save object-info/system stats, then stop it if GPU ownership is needed by another task. Verify: lock is actually held, PID/start time is recorded, expected nodes exist, port is loopback/private to the pilot. On contention: do CPU work and wait; do not kill another owner. Next: T09 after input preparation.

- [ ] **T07 — Freeze media and build canonical timeline/anchor.**
  Prerequisites: T03–T05. Implement prepare_inputs.py with --stage media. Copy/snapshot originals; decode target anchor; pad to 512×832; create the exact 241-frame donor time map and lossless frames. Output: inputs/source-manifest.json, anchor/guide PNGs, donor frames/map. Verify: frame mapping in bounds, expected RGB shapes, padding/crop byte parity. Next: T08.

- [ ] **T08 — Estimate one static alignment and inspect overlays.**
  Prerequisites: T07. Implement prepare_inputs.py --stage align. Extract donor/target eye geometry with the CPU task model; save fixed transform and visual overlay. Verify: target-space eye position/scale match at the reference frame, donor movement is retained later, no per-frame stabilization. On failed face detection: use only the bounded alternate reference-frame rule. Next: T09.

### 18.3 Control inputs and graph construction

- [ ] **T09 — Implement pose extraction and retain numeric keypoints.**
  Prerequisites: T06 and T08. Implement extract_controls.py --kind pose. Generate donor DWPose points, apply static transform, rasterize the standard control representation. Output: pose-full PNG sequence, keypoints/confidences and model provenance. Verify: 241 matching frames, face enabled/body enabled/hands disabled, no guesswork about landmark indices. Next: T10.

- [ ] **T10 — Implement depth extraction and coverage handling.**
  Prerequisites: T06 and T08; run sequentially after any GPU portion of T09. Implement extract_controls.py --kind depth. Generate temporally consistent relative depth, fixed normalization, fixed transform and target-background coverage fill. Output: depth-full PNG sequence, depth arrays, coverage masks, normalization/scale metadata. Verify: no per-frame contrast pulsing, correct polarity, no artificial moving boundary from filling. Next: T11.

- [ ] **T11 — Validate and review control sequences.**
  Prerequisites: T09–T10, or an explicit failed-arm record. Implement build_review.py --stage controls and control-validity.json. Verify: P3; inspect donor/pose/depth at normal speed and frame count/PTS equality. Failed controls do not proceed to diffusion. Next: T12 for valid arms; if neither is valid, proceed to T35 with a control-extraction failure report.

- [ ] **T12 — Implement schema-driven graph construction.**
  Prerequisites: T06 and T11. Implement build_graphs.py with --all, --arm, --length and --dry-run. Build A0/A1/A2 smoke/full graphs and decode-only graphs from section 9. Verify: role-to-ID maps, actual node fields, adapter metadata downscale=2, portrait path separate from structural guide, no stale legacy branch. Output: graphs and recipe-lock metadata. Next: T13.

- [ ] **T13 — Implement the focused contract tests.**
  Prerequisites: T07–T12. Write tests/test_pilot_contracts.py for the meaningful checks in section 14. Implement metric/bank validation pure functions now if needed for these tests. Verify: tests execute on CPU without loading diffusion weights; negative cases fail for the intended reason. Next: T14.

- [ ] **T14 — Implement robust submission/output collection.**
  Prerequisites: T12–T13. Finish run_comfy.py with submission JSON, exact prompt history polling, telemetry, per-attempt directories, timeout cleanup, exact latent selection, separate decoding and media verification. Verify: --dry-run prints the exact graph and output target without POST /prompt; stale attempt/config inputs are rejected. Next: T15.

### 18.4 GPU smokes and full renders

- [ ] **T15 — Run A0 portrait-only smoke.**
  Prerequisites: T14 and an available GPU lease. Run A0_PORTRAIT --length smoke. Output: submission/history/telemetry/result/latent/native/delivery. Verify: exactly 49 frames; valid target face; no stale text or audio; successful full decode. On failure: use the bounded diagnostic retry, then report the blocker. Next: T16.

- [ ] **T16 — Run A1 pose-guided smoke.**
  Prerequisites: T15 and valid pose control. Run A1_POSE --length smoke. Verify: adapter application and guide geometry are actually exercised, exact media contract, no control map leakage or unresolved weight warning. Output: same evidence as T15. Next: T17.

- [ ] **T17 — Run A2 depth-guided smoke.**
  Prerequisites: T15 and valid depth control; no other pilot GPU job running. Run A2_DEPTH --length smoke. Verify the same contract, with depth polarity/appearance checks. Next: T18. If one guided arm failed, retain its status and continue only valid arms; final coverage must disclose the missing comparison.

- [ ] **T18 — Record the smoke gate decision.**
  Prerequisites: T15–T17 completed or explicitly failed. Review actual short outputs, guide-removal behavior, temporal shapes, memory and warnings. Output: logs/T18.md with go/no-go per arm and actual recipe. Verify: no claim that smoke proves full-duration memory/quality. At least one guided arm and A0 must pass to proceed to full matched comparison. Next: T19 or T35 on a hard blocker.

- [ ] **T19 — Render full A0 matched baseline.**
  Prerequisites: T18. Run A0_PORTRAIT --length full, 241 frames, seed 189. Save complete provenance and unaltered result. Verify exact count/size/FPS, final crop and absence of audio. Next: T20.

- [ ] **T20 — Render full A1 pose candidate.**
  Prerequisites: T19 and passing A1 smoke. Run A1_POSE --length full with unchanged shared recipe. Verify graph/config differences from A0 are limited to declared adapter/control additions. Next: T21.

- [ ] **T21 — Render full A2 depth candidate.**
  Prerequisites: T19 and passing A2 smoke. Run A2_DEPTH --length full. Verify unrelated fields match A1 and output history belongs to this submission. Next: T22. Mark not_applicable if A2 was blocked at an earlier gate; retain that reason prominently.

### 18.5 Analysis and the optional repair

- [ ] **T22 — Implement and run raw-motion evaluation.**
  Prerequisites: T19 and at least one full guided output. Finish analyze_motion.py --stage raw. Save frame-wise landmarks/PTS, per-axis metrics, missing-data masks, mouth flags and before-certification summaries. Verify: no time compression, no dynamic time warping, no constant-axis correlation inflation. Next: T23.

- [ ] **T23 — Assemble normal-speed visual comparisons.**
  Prerequisites: T22. Run build_review.py --stage raw. Produce full-frame and mouth views, curves and artifact index; label historical V14 separately. Verify videos decode, labels are legible, compared timestamps align, and all rejected arms remain visible. Inspect videos rather than just stills. Next: T24.

- [ ] **T24 — Select candidate or one bounded repair.**
  Prerequisites: T22–T23. Write decision.json with numeric/visual reasons and quality uncertainties. If no repair is needed, mark T25 not_applicable and select a candidate. If the section 11.5 repair conditions apply, identify exactly one changed variable and proceed to T25. If nothing is usable, do not manufacture a bank; proceed to T35 with a negative result.

- [ ] **T25 — Optional A3 repair and reassessment.**
  Prerequisites: T24 explicitly selects a repair. Implement the declared changed guide/strength, save its inputs/config, run at most one full repaired render, then repeat the affected T22/T23 analysis. Verify no hidden prompt/seed/model changes. Output: updated decision.json naming selected arm or no candidate. Next: T26 or T35. If unnecessary, record not_applicable and no GPU run.

### 18.6 Bank, certification and runtime

- [ ] **T26 — Implement experimental packaging and strict certification.**
  Prerequisites: T24/T25 selects a plausible candidate. Finish certify_package.py. Build only pilot-local candidate and V14-reference three-file banks with copied idle/smile assets, consistent handle/blend/QP treatment and final-content cache IDs. Verify no production writes and no factory-ledger dependency. Next: T27.

- [ ] **T27 — Run exact decoded boundary validation.**
  Prerequisites: T26. Decode every final file and check all equalities/media facts in section 12.2. Verify the changed-tail negative control fails and names its target. Output: validation-report.json per bank with six directed cross-file transitions plus three self-loops. On failure: use bounded all-clips QP fallback, never weaken equality. Next: T28.

- [ ] **T28 — Review seams and load runtime manifests.**
  Prerequisites: T27. Produce seam montage; inspect all transitions/self-loops; validate six logical IDs, three physical files, roles, counts and durations using the actual MuseTalk loader. Output: seam review plus manifest-validation result. On bad perceptual seam: at most one 12-blend-frame bank revision as specified; otherwise fail candidate suitability. Next: T29 or T35.

- [ ] **T29 — Prepare distinct audio fixtures and runtime environment record.**
  Prerequisites: T28. Canonicalize Speech A/B and verify they differ. Release all pilot diffusion/preprocessor GPU allocations. Inspect the actual MuseTalk service/backend and available test interpreter; record GPU/runtime profile. If unavailable, diagnose an isolated launch path within scope; otherwise report the exact blocker. Next: T30.

- [ ] **T30 — Implement runtime wrapper and prepare only new pilot caches.**
  Prerequisites: T29. Implement run_webrtc.py around the verified existing harness. Resolve candidate-arm paths from decision.json, not a hardcoded assumption that A1 won. Implement --bank, --speech, --prepare-only, --showcase, --dry-run. Verify argv/output paths and unique new IDs. Prepare/warm the two banks without --force-recreate. Save responses/cache IDs. Next: T31.

- [ ] **T31 — Record V14-reference bank with both speech fixtures.**
  Prerequisites: T30. Run one session at a time, same backend/FPS settings, reaction none. Capture reference Speech A and Speech B MP4s and final harness JSON. Verify completed speaking and neutral recovery, timestamp proof and full decode. Next: T32.

- [ ] **T32 — Record candidate bank with both speech fixtures.**
  Prerequisites: T31. Repeat exactly the same two fixtures/settings on the candidate. Verify actual candidate IDs in pose traces, no variant rotation, no stale cache and all runtime checks. Inspect mouths/jaws and talking loops at normal speed. Next: T33.

- [ ] **T33 — Record candidate idle/talking/smile showcase.**
  Prerequisites: T32. Use warmth and the existing six-logical-pose showcase flags, labeling nod/empathy as idle aliases. Record full transitions, postroll and proof metadata. Verify the three actual behaviors are visible and no six-behavior claim is made. Next: T34.

- [ ] **T34 — Assemble runtime A/B review and final assessment.**
  Prerequisites: T31–T33, or explicit runtime failure evidence. Run build_review.py --stage runtime; align comparisons by actual speech-start/pose trace timestamps rather than recording file start. Show full frame and mouth views. Do not retime audio/video to hide errors. Output: updated review index, runtime-comparison videos and assessment separating mechanical checks from subjective judgment. Next: T35.

### 18.7 Reporting, preservation and handoff

- [ ] **T35 — Produce the evidence-backed final report.**
  Prerequisites: completed successful path, or a recorded blocker/negative result from any gate. Implement/run report.py. Include all section 15 fields, executed/skipped arms, exact errors, and the narrow conclusion supported. Verify every linked artifact exists; use relative paths within the artifact bundle and absolute links in the chat handoff. A blocked run still needs this report. Next: T36.

- [ ] **T36 — Verify preservation, release owned resources and hand off.**
  Prerequisites: T35. Compare production hashes and pre-existing repository changes; stop only recorded pilot-owned processes, release lease and list created test caches. Preserve outputs/failures. Verify the report states whether user visual approval is pending. Final user message: outcome, best comparison video, report, and any exact remaining blocker. Do not claim capacity gains or activate production. Experiment execution ends here.

### 18.8 Failure routing and autonomous continuation

The detailed subsections above intentionally allow Luna to make ordinary implementation choices and continue through safe, reversible gates without asking permission repeatedly. Use these rules when a task fails:

| Failure | Immediate next action | Continue independent work? |
|---|---|---|
| GPU is occupied by ongoing PRO work | Keep CPU preparation/reporting moving; wait for ownership | Yes; no GPU submission until available |
| One control arm fails detection/schema/smoke | Save failure; run other valid guided arm plus A0 | Yes; disclose incomplete comparison |
| All guided arms fail before full rendering | Finish diagnostics/control review; T35 | No speculative replacement model |
| Generation works but none is visually suitable | T24/T25 bounded repair, then negative report if still unsuitable | Yes, within repair budget |
| Numerical motion pass but identity/mouth fail | Mark arm failed quality; do not package as passing | Evaluate another already-rendered arm |
| Exact boundaries fail | Bounded consistent QP fallback, then diagnose/report | No runtime claim from an invalid bank |
| Mechanical boundary pass but visual seams fail | One documented blend trial or reject | Do not let the hash override appearance |
| Offline bank passes but server unavailable | Preserve bank and report runtime-blocked | Yes: finish all offline review artifacts |
| User approval pending after test recordings | Deliver reviewable result as pending approval | No production activation |
| Task requires new scope, paid service or destructive change | Explain exact blocker and request that specific authority | Finish all independent authorized work first |

**A task is complete when its evidence is complete. The experiment is complete when it has an honest, reviewable answer—not when every arm has been forced to pass.**
