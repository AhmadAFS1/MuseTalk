# Japanese LTX 2.3 flicker validation — 2026-09-25

The alternate Q4 Japanese talking test's visible double-face smear has a confirmed decoder contribution. Re-decoding its **identical saved latent** from temporal 16/8 to 64/16 substantially removes that ghosting in inspected frames, without changing prompt, seed or generated motion. The 64/16 decode took 14.130 seconds. The small-window alternate decode was a regression from the already-correct native runner and prior A1 pilot.

- [Corrected ten-second review clip](/workspace/experiments/ltx23_japanese_flicker_validation_20260925/corrected-talking-10s.mp4)
- [Synchronized decoder before/after](/workspace/experiments/ltx23_japanese_flicker_validation_20260925/decoder-comparison-10s.mp4)
- [All test videos and review page](/workspace/experiments/ltx23_japanese_flicker_validation_20260925/review.html)
- [Detailed methods, exact graphs, limitations and results](/workspace/experiments/ltx23_japanese_flicker_validation_20260925/README.md)

Three additional Japanese generations completed: NAG on/off crossed with the 6+2 restart versus uninterrupted eight steps, using the original alternate generation as the fourth arm. Portrait, text, base model and seed were preserved; every arm used 64/16 decode. These texts are the alternate test's texts, not the accepted native talking prompt. No new prompt engineering or LoRA test was performed.

| Clip | Generation execution (s) | Eye vertical range (px) | Range / eye span | Last central lip gap >3 px (s) | Exact endpoints |
| --- | ---: | ---: | ---: | ---: | --- |
| NAG + 6+2, corrected decode | 231.318 (original run) | 33.3 | 22.9% | 8.38 | No |
| No NAG + 6+2 | 225.831 | 31.5 | 21.7% | 8.38 | No |
| NAG + uninterrupted 8 | 232.239 | 33.4 | 23.0% | 8.38 | No |
| No NAG + uninterrupted 8 | 224.649 | 30.2 | 20.8% | 8.38 | No |
| Accepted native reference (different setup) | existing asset | 18.5 | 12.1% | 8.71 | Yes |

Ranges use the common 0.5–8.5 s interval. Lip gap is a geometric diagnostic, not audio silence. The accepted native clip has a different prompt, seed, portrait fit, resolution and endpoint conditioning; it is a product reference, not an isolated sampler comparison. Each alternate has more measured vertical head movement, and all finish their central lip opening above 3 px at 8.38 s. The new sampler variants do not justify replacing the accepted native clip.

The three new generations took 232.239, 224.649 and 225.831 seconds; their combined decode took 42.458 seconds. A same-latent 128/32 decode also succeeded (13.696 seconds), with only a modest additional diagnostic improvement over 64/16 and no material motion change. Timings exclude process startup/restarts and review. A combined multi-branch attempt exited 143 before producing outputs; its cause was not established, and the successful isolated reruns are the reported results.

## Decision

Keep `generate_three_pose_videos.py`'s existing spatial 256/64 and temporal 64/16 decode, `center_crop`, first/last guides and native eight-step setup. Keep the accepted seed-191 talking and smiling prompts/clips. Do not relabel the alternate tests as the new best prompts.

Further prompt iteration belongs on the still-unapproved **idle**, after reviewing the existing [v17 candidate](generated/portrait_pose_set_20260923/ltx/japanese_idle_breath_review_v4/README.md). If it still fails, make an idle-only version, initially hold the seed/settings fixed, change one instruction, and compare actual shirt breathing plus head/eye stability. The old idle's wobble was generated with the proper decoder already, so the new decode finding does not explain that failure.

A new motion-control/LoRA route is unnecessary to repair this demonstrated decode artifact. Historical Union Control A1 still has unresolved mouth-motion leakage; no lip-keypoint ablation was run here. The separate character-specific talking-head LoRA was not installed or tested. These runs do not change MuseTalk/WebRTC or production caches.

The corrected alternate preview is a ten-second cut, **not an endpoint-matched loop**. Full raw alternate clips remain 241 or 249 frames at 24 FPS. Contact sheets and all-frame diagnostics support the findings; no normal-speed human acceptance or zero-flicker certification is claimed. Accepted assets and prompt packs remain unchanged.
