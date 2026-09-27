# H3 avatar creation with the accepted TAESD chin refinement

For a recursive folder of existing portraits, all three H3 poses, production
MuseTalk cache preparation, and verified S3 publication, use the
[batch three-pose runner](BATCH_THREE_POSE.md).

Recipe version: `h3-taesd-chin100-seam-v1`, 27 September 2026.

This is the current best visually reviewed talking-avatar recipe. The user
selected the more expressive revised-portrait H3 sources and called the latest
chin seam refinement “much better.” The Japanese and Latina examples establish
the baseline; new identities still need individual visual review.

## Recipe

1. **Create one portrait per identity with the built-in imagegen tool.** Use the
   prompts in the batch JSON. Keep a centered head and upper torso, direct gaze,
   closed resting lips, a subtly lowered face with the chin gently forward,
   upright neck, clear jaw outline, diffuse frontal light and natural texture.
   Use a navy crew-neck top and softly blurred home office for this controlled
   batch. Vary faces and facial hair deliberately. A prompted four-degree head
   pitch is a composition target, not a measured physical angle.
2. **Inspect and save the portrait.** Save the original PNG in the workspace;
   record its prompt, source, hash and review. The reusable command accepts this
   image as input. Built-in image generation is an interactive tool step, not an
   unattended API dependency. The script does not purchase images or switch to
   an image API. Center-fit a separate 512 × 896 anchor without stretching.
3. **Generate the expressive H3 base.** Adapt only identity, pronouns, hair and
   facial-hair descriptors in the accepted prompt. Preserve natural speaking
   articulation, nearly fixed head pose and camera. Use the same portrait as
   both first/last conditioning images. Settings: 243 frames at 24 FPS, seed 42,
   eight Euler/simple steps, CFG 1, video/audio sigma shifts 5/3, pruned INT8 H3
   DiT, INT8 video VAE, NVFP4 Qwen and FP32 audio VAE. On this 12 GB GPU, use
   `--fast-disk --disable-pinned-memory --cache-none` and a fresh owned ComfyUI
   process. Run GPU jobs sequentially.
4. **Package the source.** Use the proven 243 → 240 frame conversion, record the
   three removed low-change frames, then apply the shared identity anchor and
   eight neighboring transition frames. Decode and require exact first/last
   RGB equality. Keep native H3 audio as evidence; H3 has previously improvised
   speech despite an exact-line prompt, so do not present it as verified TTS.
5. **Prepare this source independently.** Detect every source face using the
   native DWPose/S3FD path, add the established 10-pixel lower crop margin, use
   jaw masks with cheek widths 90, and encode with MuseTalk's native FP16 VAE.
   Use preparation seed 123. Never reuse another identity's image latents or
   masks. Continuous replacement speech avoids the earlier long-pause test
   problem. Cache the audio features with the exact audio file hash.
6. **Render the override.** TensorRT FP16 batch-8 UNet plus compiled TAESD
   **decoder**. Native avatar encoder stays unchanged. Track each generated face
   and apply the accepted three-frame chin-offset filter, full-strength chin
   alignment and refined target-relative mask fade. Protect generated lip
   pixels. This is the accepted seam-refinement math, not the older INT8 decoder
   or rejected narrow-mouth/elongated-chin experiment.
7. **Validate and package.** Keep standard MuseTalk and refined outputs from the
   same predictions for comparison. Check lips, map Jacobian, exact endpoints,
   full-video decoding, source/generated/final landmarks and geometry. Review
   side-by-side videos for cheek drift, double chin, skin seams, mouth flicker,
   teeth and beard/moustache loss. Landmark and texture statistics are diagnostic
   proxies; they cannot certify realism or beard identity by themselves.

## Reusable command

The entry point is `create_avatar.py`. The JSON specifies identities and portrait
paths. Optional `audio` accepts a ten-second mono WAV; otherwise the runner
creates continuous Kokoro speech, using `voice` or the batch's female/male
defaults. Use a new output directory for changed inputs.

```bash
WORKFLOW=/workspace/MuseTalk/character_factory/h3_avatar_workflow
python3 "$WORKFLOW/create_avatar.py" \
  --config "$WORKFLOW/config/diversity_batch_20260927.json" \
  --output /workspace/experiments/avatar_diversity_20260927 \
  --stage plan

python3 "$WORKFLOW/create_avatar.py" \
  --config "$WORKFLOW/config/diversity_batch_20260927.json" \
  --output /workspace/experiments/avatar_diversity_20260927 \
  --stage all
```

Completed stages are resumable and verify input/output hashes before accepting existing
results. Select one identity with `--only ID`; individual stages support
incremental inspection. Each GPU stage runs in its established environment:
H3 in `.venvs/comfy-h3`, preparation/rendering in `.venvs/musetalk_trt_stagewise`,
and FaceMesh/media analysis in `SoulX-FlashHead/.venv`. Pin rendering to the TRT
environment: cross-environment NumPy/OpenCV arithmetic previously changed raw
pixels. The controller uses only the standard library and launches the workers.

If a stage is interrupted after creating a delivery but before writing its
completion record, inspect its log and use a fresh output directory or preserve
and move that unrecorded partial delivery before retrying. The runner refuses to
silently overwrite such a file.

For a new identity, copy a batch entry and set a unique lowercase `id`, `label`,
`age`, `gender`, `hair`, `tone`, `facial_hair`, `image_prompt` and `portrait`.
After visually inspecting the portrait, set `portrait_review` to `passed`.
This is a local review record, not a claim of demographic classification.

Before reusing this extracted implementation with changed code, run:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python \
  "$WORKFLOW/verify_baseline.py"
```

It requires exact raw-pixel parity with all 480 accepted Japanese/Latina frames.
The current check is saved in [baseline_parity.json](baseline_parity.json).
Installed H3 model filenames, lock checksums, sizes and ComfyUI revision are in
[installed_profile.json](installed_profile.json); preparation and rendering also
record their model/code hashes. Existing H3 weights are not downloaded again.

## Scope and acceptance

- The `create_avatar.py` runner above produces ten-second talking clips, with
  shared endpoints **within each identity**. Use the separate batch runner
  linked at the top for idle/talking/smiling sources and S3 caches.
- Exact endpoints do not guarantee invisible arbitrary mid-clip transitions.
- The first diversity batch is a stress test across designed appearances, not a
  statistically representative fairness study or evidence about an entire group.
- Beard preservation is an explicit open risk: MuseTalk generates a lower face,
  including facial-hair regions. The chin refinement cannot guarantee hairs
  inside the protected speech area remain identical to the source.
- Keep generated sources, original portraits, complete prompts, model settings,
  stage logs and test results. Temporary decoded frames may be cleaned after
  preparation; do not remove earlier experiment media or checkpoints.
- Historical measured refined-render speed was about 169/163 FPS on Japanese/
  Latina, excluding preparation, TTS, encoding and transport. A new batch's
  generation time or playback FPS is not the same measurement.

## Baseline evidence

- [Accepted expressive prompts](../../../minimax-h3/prompts/successful_portrait_jaw_20260926.json)
- [Accepted seam refinement](../../../experiments/chin_seam_refinement_20260927/README.md)
- [User review](../../../experiments/chin_seam_refinement_20260927/user_review.json)

## Completed six-identity validation

[Videos, metrics and test records](../../../experiments/avatar_diversity_20260927/README.md)
and [visual findings](../../../experiments/avatar_diversity_20260927/VISUAL_REVIEW.md).
All six full pipelines passed: 1,440 new frames, zero protected-lip pixel changes,
zero optimized/reference differences, valid warp Jacobians, 36 complete media
decodes, and exact shared source/override endpoints. New warm-render measurements
were 148.0–170.6 FPS; this excludes setup, preparation, encoding and delivery.

The three facial-hair overrides still blur or remove moustache and near-mouth
hairs, although the refined lower beard edge retains more detail. The other
three have visible generated-skin smoothing. Jaw-step measurements improved,
but absolute generated-target chin error increased in all six. Keep these
limitations with the recipe; this batch does not establish universal visual
acceptance or perfect beard preservation. All original portraits and H3 bases
are retained. The accepted Japanese/Latina examples remain the user-approved
baseline.
