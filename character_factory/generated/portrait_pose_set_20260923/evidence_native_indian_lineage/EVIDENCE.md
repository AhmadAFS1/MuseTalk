# Corrected native LTX 2.3 three-pose evidence

Date: 2026-09-23

## Correction

The rejected run used the character-factory Prompt Relay V6/V14/V8 workflow. That was not the
native LTX workflow used by the successful September 22 Indian avatar test. The rejected assets
remain under `../rejected/20260923_prompt_relay_wrong_lineage/` and are not used here.

These replacement clips use the compact native Q4 graph preserved at:

`/workspace/experiments/ltx23_native_flf_talking_20260922/graphs/generation.json`

The invariant settings are:

- LTX 2.3 distilled 1.1 Q4 model and the same Q3 Gemma text encoder;
- 512x832, 241 frames, 24 fps;
- Euler, CFG 1, the accepted eight-step sigma schedule;
- the same prepared portrait guide at frame indices `0` and `-1`, both at strength `1.0`;
- the accepted 64/16 tiled decoder;
- no SoulX, Segmind, Prompt Relay, or NAG;
- no fade or cross-fade;
- the final decoded frame is replaced by the first before all-intra H.264 packaging, giving exact
  decoded first/last pixels while preserving the preceding 240 native frames.

## Prompt provenance

- **Idle:** exact closed-mouth natural-motion prompt preserved in the Indian A1 graph, with its
  control nodes removed in the same way as the later direct-LTX tests. Seed `189`.
- **Talking:** exact accepted native first/last talking prompt and negative prompt from the Indian
  seed-191 graph. Only `male/he/his` identity nouns were generalized to `person/the person` so the
  prompt applies to both portraits. Seed `191`.
- **Smiling:** the exact approved V8 moderate closed-lip smile behavior, placed in the same proven
  fixed-camera native wrapper. Seed `193`.

The full submitted graphs, ComfyUI histories, prompts, seeds, retained latents, native decodes, and
delivery hashes are stored beside each avatar's `manifest.json`.

## Independent validation

All six files passed a separate post-generation validator that performs a complete ffmpeg decode,
checks the stream contract, compares decoded first and last RGB frames, and verifies that the
submitted graph's prompt and seed match the manifest.

| Avatar | Pose | Size | Frames | Seconds | SHA-256 |
|---|---|---:|---:|---:|---|
| Japanese | idle | 512x832 | 241 | 10.042 | `ce25f36b4b40486e26b8255a360d083031b3a2ee8a7037e22105a13e5b88d283` |
| Japanese | talking | 512x832 | 241 | 10.042 | `6127c299e58202c79308ca434df694ee1da70a7b779edeabc4d294db920d6246` |
| Japanese | smiling | 512x832 | 241 | 10.042 | `e832b9a75c48bd2b84ad74c3e6d4052cc6a499999eb9cf25f20cd70efbb2677a` |
| Latina | idle | 512x832 | 241 | 10.042 | `8c6837162da64e34a5c9e0cc66b0a9643d537bb04f7286d213753130c399ad3e` |
| Latina | talking | 512x832 | 241 | 10.042 | `72f4ad3ab5dca57be6fa33b42f1491da50c4283f19e5811ce46ff2b8e522fb7c` |
| Latina | smiling | 512x832 | 241 | 10.042 | `a2336e53df4a14d7c0a72f8033704909202f12381c0188998ec65705081fd29b` |

Every file is silent H.264/yuv420p at 24 fps. Every file has exact decoded first/last pixels. The
machine-readable report is `validation.json`. Its temporal diagnostic records adjacent-frame luma
change for the whole frame and the mostly static top-quarter background; this is only an abrupt
change check and does not replace normal-speed human review.

## Review assets

- `japanese-three-pose-review.mp4`: idle, talking, and smiling side by side at normal speed.
- `latina-three-pose-review.mp4`: idle, talking, and smiling side by side at normal speed.
- `all-six-contact-sheets.jpg`: one frame per second from all six clips.
