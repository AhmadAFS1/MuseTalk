# Multi-agent workflows used for the MuseTalk throughput work (2026-09-27 to 09-29)

## Reusable (run by name)

These live in `.claude/workflows/`. From a Claude Code session in this repo, ask for them by name, e.g. "run the
verify-video-evidence workflow".

| Workflow | Use it when | Args (all optional) |
|---|---|---|
| `verify-video-evidence` | A new round of comparison videos exists (`scripts/video_lineage.py` / `scripts/video_signoff.py`), before you sign off | `{sets: [{dir, what}], ids, readmes, quality_labels, throughput_jsons}`; defaults to the r2/r5 sign-off sets |
| `review-repro-scripts` | After changing `scripts/repro_400fps/`, or before rebuilding on a new machine | none |

Both are read-only, CPU-only reviews. They spawn about five agents and never touch the GPU.

## Archive (the scripts as they were run)

These are in `archive/`, kept for provenance and as templates. They reference the paths and state of their day, so
re-running one verbatim repeats that day's task. Adapt a copy instead.

| Script | When (UTC) | What it did |
|---|---|---|
| `musetalk-300fps-understand.js` | 09-27 19:19 | Mapped every per-frame cost of the serving path on the RTX 4070 SUPER and probed the key levers (300 fps feasibility). |
| `musetalk-300fps-plan.js` | 09-27 21:48 | Competing 300 fps plans from measured evidence, scored, synthesized, adversarially verified, then revised into `docs/musetalk_4070s_300fps_plan_2026-09-27.md`. |
| `musetalk-300fps-engines.js` | 09-28 00:25 | box_guard + UNet validation corpus, TensorRT TAESD backend, stagewise FP16 UNet backend, then independent verification of gates, speed and quality. |
| `musetalk-300fps-serving.js` | 09-28 02:15 | Serving-path and memory optimizations (flag-gated), golden replay exactness harness, pre-change vs candidate A/B video tool, WebRTC load-test harness, adversarial review. |
| `musetalk-startup-understand.js` | 09-28 02:08 | Read-only map of the docs, start/install scripts and env knobs for the startup rework (other session). |
| `musetalk-startup-implement.js` | 09-28 05:22 | First implementation of the startup rework, 4 parallel owners (other session). |
| `musetalk-300fps-code-complete.js` | 09-28 05:21 | CPU-only completion of scheduler pipeline, WebRTC/api_server wiring, avatar memory layout and the A/B tool; GPU steps queued as scripts. |
| `musetalk-startup-rework.js` | 09-28 05:33 | Start/install rework in the perf worktree: resolver, engine store, pinned installers, launch chain, 300 fps levers. |
| `musetalk-300fps-chin-multistream.js` | 09-28 05:38 | The multi-stream TAESD + 100% chin harness (bit-exact vs accepted renders), ≥ 300 fps with the new engines, quality metrics and videos. |
| `verify-lineage-videos.js` | 09-28 22:49 | First verification of the BEFORE/r2/r3/r4 lineage videos: 6 per-identity agents plus a skeptic. Results: `experiments/video_validation/lineage_all_rounds/verification/pass1_first_4col_set.json`. |
| `verify-final-lineage.js` | 09-28 23:54 | Second pass on the final sets: 3 video agents, a README auditor and a skeptic. Results: `.../verification/pass2_final_sets.json`. |

The SoulX-FlashHead sessions on this box ran their own workflows (30 fps levers, quantization targets, tiny-VAE
feasibility, motion reuse). Those scripts are in `~/.claude/projects/-workspace-SoulX-FlashHead/*/workflows/scripts/`
and belong with that repo.
