# Video A/B validation (standing tool)

Every optimization round ships video evidence against the **pre-change** output, and the evidence can be re-created at any time. The tool is `scripts/video_ab.py`. The chin renderer is `scripts/video_ab_chin_render.py`, and the live-path clips come from `scripts/replay_scheduler_exactness.py`. The GPU steps are in [`gpu_sequence.sh`](../../docs/fps_comparisons/4070s_300fps_impl_20260928/video_ab/gpu_sequence.sh).

## What is compared

An **arm** is a code tree plus a set of env flags.

- **A, pre-change:** `/workspace/MuseTalk` at main's git HEAD, with no new flags. This is what ships today.
  - If main's working copy has uncommitted code changes, the tool exports exactly HEAD into `/workspace/.cache/video_ab/trees/<head12>/` and uses that copy. It uses `git archive` (read-only) and symlinks the data directories.
- **B, candidate:** `/workspace/MuseTalk-perf300` plus flags.
- **Isolation:** each arm runs in a fresh process. Inherited `MUSETALK_*`, `HLS_*`, `WEBRTC_*` and `AVATAR_*` variables and `PYTHONPATH` are removed first. Bytecode goes to `/workspace/.cache/video_ab/pycache`, so nothing is written into either tree.
- **Same instrument for both arms:** the renderer and harness code is shared, and each arm imports the product code from its own tree. The chin renderer records a module-origin check. The harness runs with its `ROOT` set to the arm's tree.

| Clip | Source | Frames |
|---|---|---|
| `chin_japanese`, `chin_latina` | Chin recipe: native-encoder latents → TRT FP16 UNet bs8 → TAESD → per-frame FaceMesh → 100% chin alignment, refined seam, `.25/.5/.25` lookahead. Adapted from `chin_fps_validation_20260927/run.py` and `h3_avatar_workflow/{backend,render_stage}.py`; the originals are unchanged. | 240 @ 24 fps (10 s) |
| `replay_bob_mid` | Live scheduler path (golden replay, all 10 harness jobs per arm). 3-pose `chinese_bob` motion avatar with a mid-turn pose switch. | ≤ 20 s @ 20 fps |
| `replay_jp_d10` | Same replay. Standard avatar `japanese_realtime_talking`, dense 10 s audio. | ≤ 20 s @ 20 fps |

- **Chin clips:** measured fps is the median of 3 warm 240-frame runs. It covers GPU work, D2H, tracking and composition, and excludes load, preparation and encode. The runs must be SHA-identical to one another.
- **Replay clips:** measured fps is the golden run's aggregate unpaced rate (scheduler plus compose, no transport).
- **Storage:** all frames are stored losslessly (`libx264rgb -qp 0`) together with a SHA-256 for every frame. For replay, the frame hashes equal the harness's own pre-encoder hashes.

## Reading a `<clip>__<arm>_ab.mp4`

- **Playback:** the clip's native fps at 1× speed (stated in the video), with the clip's audio. Maximum 20 s. libx264 crf 10, yuv420p.
- **Columns:** **A pre-change**, **B candidate**, and **|A−B|×8** (per channel, clipped). The diff panel outlines the mouth ROI.
- **Rows:** the full frame is on top, and a 3× nearest-neighbour mouth+chin zoom is below it.
- **Headers:** arm, tree (`@head`, clean / `+dirty:<digest>` / HEAD snapshot), flags, active backends and measured fps. The diff column shows the SHA-identical count, min PSNR, max/mean LSB and the gate result.
- **Footer:** per-frame numbers. The timeline strip is green when a frame is SHA-identical, amber when it differs only outside the mouth ROI, and red when the mouth ROI differs.
- **Exact numbers are in the JSON.** The mp4 is lossy (yuv420p), so a faint diff panel in the video is not proof of a difference; `<clip>__<arm>_ab.json` has the exact values:
  - Per frame: SHA equality (both SHAs), PSNR, max/mean absolute LSB and differing pixels, for the full frame and the mouth ROI.
  - Summary and gate.
  - Replay clips: the harness golden compare of all jobs.
  - Chin clips: FaceMesh jaw+lip deviation between the arms.
  - Layout rectangles and the exact label strings.
- **Contact sheet:** `_contact.jpg` shows 6 evenly spaced frames plus the worst frames.

## Gates and verdicts

- **exact** (E0: serving, scheduling and memory levers). Passes when:
  - every compared frame is SHA-identical;
  - frame counts are equal;
  - for replay clips, the golden compare of all 10 jobs is identical (faces, composed BGR, yuv420p, order and status).
- **fp16** (E1: engine rebuilds). This is a proposed numeric gate, and **the video decides** (plan D14). Thresholds:
  - full-frame min PSNR ≥ 40 dB;
  - mouth min PSNR ≥ 36 dB;
  - full-frame mean ≤ 0.5 LSB.
  - The FaceMesh jaw+lip deviation (G-TRACK proposal: mean ≤ 0.05 px, p99 ≤ 0.15 px) is reported but does not gate.
- **Render checks (every arm):**
  - Chin repeats must be identical.
  - The recipe checks must hold: protected lip pixels unchanged, ROI path equals the full-frame reference, Jacobian > 0.25.
  - A requested backend that did not activate (silent fallback) is a **FAIL**.

## Running

```bash
# CPU only: synthetic A/B layout, panels, labels, JSON, lossless round trip, cache keys, harness tree shim,
# replay adapter, orchestration, a mocked GPU render loop, and a real dry-run chin A/B on both trees
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/video_ab.py selftest --out /tmp/selftest.json

# GPU (every step leased through scripts/box_guard.sh): baselines, r1_engines, r1_serving, rollup
docs/fps_comparisons/4070s_300fps_impl_20260928/video_ab/gpu_sequence.sh            # all steps
docs/fps_comparisons/4070s_300fps_impl_20260928/video_ab/gpu_sequence.sh B1 B2 E1   # selected steps
docs/fps_comparisons/4070s_300fps_impl_20260928/video_ab/gpu_sequence.sh --print    # show commands only
```

A new round or arm takes two commands. Render under the lease, then compose on the CPU:

```bash
scripts/box_guard.sh run --min-avail-gb 11 --wait-min 60 --label video_ab_<round>_<arm> -- \
  /workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/video_ab.py render-arm \
  --round <round> --arm <arm> --flags K=V,K=V --clips chin_japanese,chin_latina,replay_bob_mid,replay_jp_d10
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/video_ab.py compose-arm --round <round> --arm <arm> --expect exact|fp16
```

- **Missing baselines:** `render-arm` does not render them. Run `video_ab.py baseline` first (the sequence does this in steps B1/B2).
- **Other identities:** `--clips replay_<job>` accepts any harness golden job id, for example `replay_bob_t1` or `replay_lat_m10`.
- **Ad hoc comparisons:** `video_ab.py compose --a A/clip.json --b B/clip.json --out-prefix ...` compares any two dumps.

## Where things live

- `baselines/<clip>/<main-head12>/`:
  - `clip.json`, the frames and `baseline.json`, which holds the signature: HEAD, instrument SHA-256 and clip parameters.
  - Rendering happens only when main's HEAD, the instrument or the parameters change.
  - Replay frames live in `baselines/replay_golden/<head12>/<jobs>/`.
- `<round>/<clip>__<arm>_ab.{mp4,json}` and `_contact.jpg` are the deliverables.
- `<round>/_dumps/<arm>/` holds the candidate dumps and `render_arm.json`.
  - A candidate's lossless frames are deleted after a passing compose. Their SHA-256 values stay in `clip.json`. Use `--keep-dumps` to keep the frames.
- Step reports are written to `docs/fps_comparisons/4070s_300fps_impl_20260928/video_ab/*.json`, and `rollup.json` summarizes them.

## Limits

- **Replay covers only the pre-encoder frames.** The harness replaces the transport, so encoder and WebRTC effects are not in the replay clips. Those need a receiver recording.
- **Replay audio starts at t = 0.** The harness does not record A/V offsets, so lip sync in the replay videos is approximate. The chin clips' 240 frames cover exactly the 10 s WAV.
- **Chin fps depends on CPU load.** It includes CPU FaceMesh tracking in the SoulX venv. The load average is recorded for each repeat.

## Index

<!-- VIDEO_AB_INDEX:BEGIN (generated by scripts/video_ab.py index; do not edit by hand) -->

_Index regenerated 2026-09-28T06:05:27Z._

### r1_unet_stagewise

| Clip | Arm | Result | One-line verdict | Files |
|---|---|---|---|---|
| middle_eastern_man_full_beard | - | n/a | not produced by video_ab.py (no per-frame JSON); see the producing area's notes | [video](r1_unet_stagewise/middle_eastern_man_full_beard_ab.mp4) |
| south_asian_woman | - | n/a | not produced by video_ab.py (no per-frame JSON); see the producing area's notes | [video](r1_unet_stagewise/south_asian_woman_ab.mp4) |

### Pre-change baselines (cache)

None rendered yet.

<!-- VIDEO_AB_INDEX:END -->

## Rounds (full chin recipe, six accepted identities; A = accepted pre-change render)

| Round | Candidate | Measured fps | Quality verdict (tool) | Folder |
|---|---|---|---|---|
| r1 | stagewise FP16 bs16 UNet + compiled TAESD (2 identities, single-stream render loop) | GPU path 309.6 | all perceptual gates pass; beard landmark p99 0.24 px | `r1_unet_stagewise/` |
| r2 | stagewise bs16 + source-prefix cache + INT8 down3/mid + TensorRT TAESD, 6 streams | **350.2** aggregate | perceptual gates 6/6 PASS; strict landmark gate 2/6 (FaceMesh floor), calibrated 5/6 | `r2_srcmix_taesdtrt_chin/` |
| r3 | as r2 + INT8 down1/down2/up2/up3 (PTQ, no recovery), 6 streams | **415.6** aggregate | measurable change: landmarks 0.20-0.36 / 0.65-0.98 px, aperture delta 0.28-0.33 px, lip corr 0.997-0.998 | `r3_srcv1_int8_taesdtrt_chin/` |

Round videos (`*_ab.mp4`, `mosaic_candidate.mp4`) are kept on disk and ignored by git (100+ MB per round).
Regenerate them with `scripts/video_ab_round.py` from a `chin_multistream_render.py --encode --save-arrays` capture.
