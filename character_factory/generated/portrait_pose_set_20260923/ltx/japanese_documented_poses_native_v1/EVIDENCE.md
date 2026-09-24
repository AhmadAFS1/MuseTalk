# Japanese portrait: documented-pose prompt rerun

Source image: `../../japanese_woman.png`.

The accepted close-up selection is V6 listening for idle, V14/V15 direct speaking, and V8 moderate closed-lip smile (`docs/ltx23-closeup-production-migration.md`). The original historical versioned prompt-pack JSON files named by the production manifest are absent from this checkout. This rerun uses the **available derived wording** in `character_factory/config/prompt_packs/pose_prompt_pack_v1.json`, flattened into the compact native Q4 LTX 2.3 graph that worked in the Indian-avatar test. It is not an exact historical workflow or prompt-pack reproduction.

| Clip | Derived prompt | Frames / 24 fps | SHA-256 | Visible contact-sheet finding |
|---|---|---:|---|---|
| `idle.mp4` | V6 active listening | 241 / 10.042 s | `3ea5242fda706b79a3168cbbbedffe4e255928d0c08ca3d83c9d6f6022b1a83d` | Head tilt becomes much larger than the restrained approved reference. |
| `talking.mp4` | V14 subtle baseline | 289 / 12.042 s | `cc36b6581a39a151cab2ed4b462b54f396e8f5a568fe3716269827f30727ddbd` | Gaze stays mostly forward; several long-looking eyelid closures need normal-speed review. |
| `smiling.mp4` | V8 moderate smile | 145 / 6.042 s | `a48989ef56d98c3e5329ace1bcd52c749eea66700002fa86789ea76f06336a65` | Closed-lip smile develops and releases in sampled frames. |
| `../japanese_documented_v15_native_variation_v1/talking.mp4` | V15 reference-paced alternate | 289 / 12.042 s | `45a6b7277c0951dd0b6d0099cbf4ad4439d431290b95b468917b754fc6a5c888` | Visible open-mouth shapes and stronger head tilt despite the closed-mouth negative prompt. Poor candidate for a MuseTalk lip override. |

Each clip is silent H.264/yuv420p at 512×832. Independent decoding confirmed its first and last RGB frames are identical, its submitted positive and negative prompts match the selected pack, and its frame count matches the documented pose. The JSON reports are `validation.json` here and `../japanese_documented_v15_native_variation_v1/validation.json` for V15. The submitted graphs, histories, latents, and per-pose prompt text are beside each `manifest.json`.

The four endpoint hashes **differ between poses**. These clips are individually loopable at the decoded endpoint but are not certified as a switch-safe multipose bank. No MuseTalk cache or runtime manifest was changed.

Review at normal playback speed: `japanese-three-pose-review.mp4` (V6, V14, V8), and `japanese-v14-v15-talking-review.mp4` (both talking candidates). The contact-sheet findings above are preliminary; still frames cannot establish motion quality or lack of flicker.
