# Git audit — 2026-09-15

> **GPU provenance:** Audit host: NVIDIA GeForce RTX 4070, 12 GB (12,282 MiB visible), driver 570.181. Git synchronization and disk inspection are read-only checks, not GPU inference tests.

Result: the main application checkouts matched freshly fetched GitHub main branches before this deployment change. **Not everything on this machine is in Git.** No existing experiment edits were committed or overwritten by this audit.

| Repository | Local HEAD at audit | Ahead / behind origin/main | Tracked/untracked source changes |
|---|---|---|---|
| /workspace/MuseTalk | `e8e5de5` | 0	0 | Clean before new deployment files |
| /workspace/SoulX-FlashHead | `5033786` | 0	0 | Clean before new deployment files |
| /workspace/omnivoice-triton | `3227f19` | 0	0 | Clean before new deployment files |
| /workspace/experiments/ref2va-evaluation/ComfyUI-Ref2VA-VSA | `92e3f83` | 0	0 | Clean before new deployment files |

The `optimize/flashhead-exact-performance` branch at `e754dc6` is an ancestor of pushed SoulX main. Its apparent untracked `.venv`, `models`, and `.torchinductor` entries are local links/runtime state.

## Old experiment worktrees

These detached worktrees retain local changes. Some are earlier versions of code subsequently merged into main; do not treat every difference as missing implementation. The exact local bytes are not all backed up.

### flashhead-optimization

```text
M flash_head/inference.py
 M flash_head/src/pipeline/flash_head_pipeline.py
 M soulx_rtc/avatars.py
 M soulx_rtc/calls.py
 M soulx_rtc/engine.py
 M soulx_rtc/experiment.py
 M soulx_rtc/server.py
 M soulx_rtc/wall.html
 M soulx_rtc/wall.js
 M soulx_rtc/worker.py
 M tests/test_avatars.py
 M tests/test_calls.py
 M tests/test_wall_browser.py
?? .trt-experiment
?? .venv
?? models
?? soulx_rtc/._compact_weights.py
?? soulx_rtc/._engine.py
?? soulx_rtc/._experiment.py
?? soulx_rtc/._server.py
?? soulx_rtc/._worker.py
?? soulx_rtc/compact_weights.py
?? tests/._test_compact_weights.py
?? tests/test_compact_weights.py
```

### flashhead-pipeline-ARsFTh

```text
M flash_head/inference.py
 M flash_head/src/pipeline/flash_head_pipeline.py
 M soulx_rtc/avatars.py
 M soulx_rtc/benchmark_calls.py
 M soulx_rtc/calls.py
 M soulx_rtc/codec.py
 M soulx_rtc/engine.py
 M soulx_rtc/experiment.py
 M soulx_rtc/server.py
 M soulx_rtc/wall.html
 M soulx_rtc/wall.js
 M soulx_rtc/worker.py
 M tests/test_avatars.py
 M tests/test_calls.py
 M tests/test_codec.py
 M tests/test_wall_browser.py
?? .trt-experiment
?? .venv
?? docs/research/PIPELINE_PROFILE_2026-09-11.md
?? models
?? soulx_rtc/compact_weights.py
?? tests/test_compact_weights.py
```

In `flashhead-optimization`, these exact file blobs were absent from the main SoulX repository object database:

- `flash_head/src/pipeline/flash_head_pipeline.py`
- `soulx_rtc/calls.py`
- `soulx_rtc/engine.py`
- `soulx_rtc/server.py`
- `soulx_rtc/worker.py`
- `tests/test_calls.py`

## Outside Git

Models, virtual environments, wheel/download caches, TensorRT engines, generated experimental media, runtime deployments and secrets are not covered by Git synchronization. `._*` entries are Apple metadata sidecars, not source. `/workspace/experiments` contains about 6.3 GiB, including large `.npy`/`.npz` arrays and videos. `/workspace/lingua-soulx` contains deployed release/config/runtime state outside a Git checkout.

**Do not retire this machine on the assumption that cloning Git reproduces all its data.** Back up any required experimental files, prepared avatars, engine artifacts and secrets to appropriate private storage first. This task does not copy secrets or bulk runtime data into Git.

The new installer and this report are delivered on the `deploy/vast-talkingheads-20260915` branch of the MuseTalk fork. Main application revisions remain the pinned revisions listed above.
