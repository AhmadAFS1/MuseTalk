# Codex playbook — generating MuseTalk characters

Operational runbook. Read `README.md` first for what the pipeline is and why certification
matters. This file is how to run it.

For the current one-avatar, three-pose Lingua pilot, use
[CHARACTER_CREATION_WORKFLOW.md](CHARACTER_CREATION_WORKFLOW.md) and
[BEST_TESTED_CHARACTER_PROMPTS.md](BEST_TESTED_CHARACTER_PROMPTS.md). The roster
and six-pose stages below are a different path.

All commands are run from `MuseTalk/character_factory/scripts/`.

---

## 0. Preconditions

| Needs | Check | If missing |
|---|---|---|
| `ffmpeg` + `ffprobe` | `which ffmpeg ffprobe` | required for certification, no way around it |
| Python `numpy`, `PIL` | `python3 -c "import numpy, PIL"` | required |
| OpenCV (optional) | `python3 -c "import cv2"` | without it the portrait face gate reports itself skipped |
| LTX 2.3 Q4 ComfyUI | `curl -s localhost:18188/system_stats` | `/workspace/LTX-2.3/scripts/start-q4-comfyui.sh` |
| MuseTalk API | `curl -s localhost:8000/health` | only needed for the final `prepare` stage |

The Q4 stack is restored by `/workspace/LTX-2.3/scripts/bootstrap-q4.sh`. It peaks near
13.7 GiB, so treat it as a 16 GB configuration despite its "8 GB" publication.

---

## 1. Settle the language list before anything else

`config/languages.json` is a **seed**, marked `"status": "seed_requires_confirmation"`. Every
character ID, pose-set ID, bank directory, and prepared avatar ID derives from `code`. Changing
a code after generation orphans that language's work.

Reconcile it against Lingua's authoritative supported-language list, then regenerate:

```bash
python3 _seed_languages.py          # edits go in the table inside this file
python3 build_character_roster.py
```

The seed carries 104 entries. Chinese, Portuguese, and Spanish are single castable languages
with their locale splits recorded under `locale_variants`, because casting is per-language while
the locale only changes caption script, TTS voice, and wording.

---

## 2. Build the roster

```bash
python3 build_character_roster.py --dry-run     # preview counts
python3 build_character_roster.py               # writes state/roster.json
```

Useful flags:

```bash
--languages ja ko zh          # restrict to specific codes
--characters-per-language 3   # default
--agnostic-count 20           # language-agnostic companions
--tier core --flagship-tier full
```

The builder warns if any language casts two slots to the same look or the same display name.
Both are deterministic de-collided, so a warning means that region's pool in
`config/archetypes.json` is genuinely too thin — widen it rather than ignoring the warning.

**Review `state/roster.json` before generating images.** Casting is the one thing that is cheap
to change now and expensive to change after 1,784 renders. Check specifically:

- does each language's heritage read right? `casting.heritage_source` says whether it came from
  the per-language override or the wider region pool;
- are display names plausible to a native speaker?
- do the three slots of a language read as three different people?

---

## 3. Generate portraits — the interactive step

This is the only stage Codex performs by hand. There is no image API in this factory.

```bash
python3 render_portrait_prompts.py --languages ja       # or --characters ja_01 ja_02
python3 render_portrait_prompts.py --only-missing       # what is still outstanding
```

This writes `state/portrait_prompts/<id>.md` and a machine-readable
`state/portrait_prompts/queue.json`.

For each queue item:

1. generate the `prompt` with your own ChatGPT image tool, at the largest 9:16 size available;
2. check it against the rejection list in the prompt file — **reject aggressively**, a bad
   portrait becomes five or six bad renders;
3. save to `state/portrait_inbox/<character_id>.png`.

Then:

```bash
python3 ingest_portraits.py
```

Accepted portraits are canonicalised to 480x832 and recorded in the ledger with both hashes.
Rejections name the reason and exit non-zero; regenerate those and re-run. Batch this stage —
generating 20 portraits then ingesting once is far cheaper than interleaving.

---

## 4. Render poses

### Standalone three-pose runner

For the current Lingua three-pose design, one portrait can be rendered without creating a
roster entry:

```bash
/workspace/experiments/soulx_ltx_motion_pilot_20260922/A1/.venv/bin/python \
  generate_three_pose_videos.py \
  --image /absolute/path/to/portrait.png \
  --output-dir /absolute/path/to/avatar-videos \
  --prompt-pack /workspace/MuseTalk/character_factory/config/prompt_packs/japanese_selected_native_three_pose_v2.json \
  --guide-fit center_crop
```

This produces `idle.mp4`, `talking.mp4`, `smiling.mp4`, `manifest.json`, and the exact submitted
generation and decode graph for each pose. It starts and stops its own low-VRAM ComfyUI workers
under the shared GPU lock. The runner uses the compact native graph from the September 22
Indian-avatar talking test. The explicit pack above selects the best-observed Japanese idle,
talking, and smile review candidates; it does not claim the missing original V6/V8 text.
The centered crop avoids the 22-pixel replicated side borders that distorted shoulders
in the previous Japanese guide. The selected smile still has a brief upward head lift.
Every render receives the same portrait at native LTX guide indices `0` and `-1`; packaging then replaces
only the final decoded frame with the first and verifies literal decoded endpoint equality. It does
not freeze a handle or add a fade/cross-fade. Delivery clips are silent because MuseTalk supplies
the speech audio.

Useful controls:

```bash
python3 generate_three_pose_videos.py --image portrait.png --output-dir out --prompt-pack ../config/prompt_packs/japanese_selected_native_three_pose_v2.json --guide-fit center_crop --dry-run
python3 generate_three_pose_videos.py --image portrait.png --output-dir out --prompt-pack ../config/prompt_packs/japanese_selected_native_three_pose_v2.json --guide-fit center_crop --poses smiling
python3 generate_three_pose_videos.py --image portrait.png --output-dir out --prompt-pack ../config/prompt_packs/japanese_selected_native_three_pose_v2.json --guide-fit center_crop --force
```

All three clips are 241 frames at 24 fps (10.041667 seconds). The selected Japanese pack uses
seeds 197, 197, and 191 for idle, talking, and smiling respectively; its identity words are
female-specific. The script's historical default is a different pack, so pass `--prompt-pack`
explicitly. Existing outputs resume by default; `--force` replaces only the requested pose files.

```bash
python3 generate_pose_videos.py --characters ja_01 --dry-run   # inspect prompts and seeds
python3 generate_pose_videos.py --characters ja_01
```

Seeds are deterministic per character, render, and attempt, so any render is reproducible and a
regression is bisectable. Rerolling shifts the seed pair by +1000:

```bash
python3 generate_pose_videos.py --characters ja_01 --renders empathetic_head_tilt --attempt 1 --force
```

The reroll budget is 2 per render (`config/generation_defaults.json`). Past that, edit the pose
prompt rather than rerolling — the accepted bank reached its empathy clip through a prompt
change, not luck.

Already-rendered poses are skipped, so a run that dies on character 40 of 332 restarts where it
stopped. Every render's wall clock is appended to `state/timings.jsonl`.

### Review before certifying

Nothing in this pipeline judges whether a render looks right. The accepted bank came from human
review **at normal playback speed** — a contact sheet hides the defects that matter. At minimum,
watch the speaking and listening clips before continuing. Known failure modes from the LTX work:

- motion amplitude exceeding the request on stochastic rerolls, especially listening;
- the empathy tilt drifting into an off-camera yaw;
- visible speech shapes, which fight MuseTalk's overlay — the mouth must stay closed.

---

## 5. Certify

```bash
python3 certify_pose_bank.py --characters ja_01
```

Passing output names the shared boundary hash and the number of ordered transitions checked
(30 for six clips). Failing output names the offending clip and exits non-zero.

**Do not bypass this gate.** If it fails only on the boundary hash, try a lower `--qp` first —
fixed-QP encoding is what makes identical input handles decode identically across separate
files. If a clip's frame count is wrong, LTX produced the wrong length and the render must be
redone, not patched.

If the freeze reads as a visible stutter on real motion, lengthen `blend_frames_each_end` in
`config/pose_spec.json` rather than shortening `canonical_handle_frames_each_end` — the freeze
is what the cross-cut depends on.

---

## 6. Package

```bash
python3 package_pose_bank.py --characters ja_01
```

Writes, outside this directory because MuseTalk is what loads them:

```
assets/ltx23_pose_banks/lingua_ja_01_facetime_closeup_v1/
  certified/*.mp4  source/*.png  manifest.json  validation_report.json  README.md
configs/pose_test/lingua_ja_01_ltx23_facetime_closeup_v1.json
```

The runtime manifest is validated against the same rules as
`scripts/test_pose_webrtc.py::load_pose_set`, so a manifest that would be rejected at session
creation is rejected here instead.

Banks are immutable by convention. `--force` replaces one; only do that for a bank nothing has
been prepared from.

Packaging defaults to `--activation-status generated_pending_review`. Only a human-reviewed,
WebRTC-verified bank should claim a production status.

---

## 7. Prepare MuseTalk caches

```bash
python3 prepare_musetalk_avatars.py --characters ja_01 --dry-run
python3 prepare_musetalk_avatars.py --characters ja_01
```

One cache per physical clip. Avatar IDs embed the first eight characters of each clip's content
hash, so a regenerated clip can never be served from a stale cache — keep that convention.

Then verify in the real runtime:

```bash
cd /workspace/MuseTalk
python3 scripts/test_pose_webrtc.py --manifest configs/pose_test/lingua_ja_01_ltx23_facetime_closeup_v1.json
```

A bank is only "done" once a WebRTC session has cycled all six poses and the recording decodes.

---

## 8. Driving it

One character, as far as it can go:

```bash
python3 run_character_pipeline.py --characters ja_01
```

It pauses at the ChatGPT boundary, tells you what to generate, and picks the character back up
on the next invocation.

Progress across the roster:

```bash
python3 run_character_pipeline.py --status
```

Suggested batch rhythm, which keeps the interactive step batched and the GPU busy:

```bash
# 1. emit 20 prompts
python3 render_portrait_prompts.py --only-missing --limit 20
# 2. generate all 20 in ChatGPT, save to state/portrait_inbox/
python3 ingest_portraits.py
# 3. hand the GPU the whole batch
for id in $(python3 -c "import json;print(' '.join(json.load(open('../state/portrait_prompts/queue.json'))['items'][i]['character_id'] for i in range(20)))"); do
  python3 run_character_pipeline.py --characters "$id" || echo "STOPPED at $id"
done
```

---

## 9. State and recovery

`state/ledger.json` is the source of truth for what has been done. Each character records its
portrait, renders, certification, bank, runtime manifest, and prepared avatars, plus a `stage`.

| Stage | Meaning |
|---|---|
| `new` | in the roster, nothing done |
| `portrait_ingested` | canonical 480x832 portrait exists |
| `rendered` | LTX clips exist |
| `certification_failed` | **blocked** — read `state/certified/<id>/validation_report.json` |
| `certified` | clips share a boundary hash |
| `packaged` | bank and runtime manifest written |
| `prepared` | MuseTalk caches exist |

To redo a character from scratch, delete its `state/renders/<id>/` and `state/certified/<id>/`
directories and its ledger entry. To redo one pose, use `--renders <key> --attempt N --force`.

`state/` is disposable except for `roster.json` and `ledger.json`. Packaged banks under
`assets/ltx23_pose_banks/` are not — treat them as immutable.

---

## What this playbook does not cover

- **Deciding a bank is good.** No automated check substitutes for watching the clips.
- **Switching production defaults.** Pointing the pose lab, wall, or WebRTC harness at a new
  bank is a separate, deliberate change; see `docs/ltx23-closeup-production-migration.md`.
- **Lingua-side wiring.** The app must be told which `pose_set_id` a character uses. The Lingua
  repository is not on this host.
