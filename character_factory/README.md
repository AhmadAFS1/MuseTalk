# Character factory

**Current Lingua pilot:** one character and three poses (idle, talking,
smiling). Start with [CHARACTER_CREATION_WORKFLOW.md](CHARACTER_CREATION_WORKFLOW.md)
and [BEST_TESTED_CHARACTER_PROMPTS.md](BEST_TESTED_CHARACTER_PROMPTS.md). The
standalone renderer and Japanese review clips have been run on a GPU; they
have not been certified as a shared-anchor MuseTalk pose bank. The larger
roster pipeline described below remains a separate six-pose design.

Baseline scripts for producing MuseTalk multipose avatars at roster scale: **3 characters for
each of 104 languages, plus 20 language-agnostic companions**, each with a certified six-pose
LTX 2.3 bank that MuseTalk can switch between mid-call.

The roster-scale six-pose pipeline is scaffolding for Codex to reference,
refine, and run. Its GPU stages have not been validated end to end. What
*has* been verified for that path is in [Validation status](#validation-status).

## What one character costs

| | |
|---|---|
| Source portrait | 1 image, generated interactively in ChatGPT |
| Pose renders | 6 (`full` tier) or 5 (`core` tier) LTX 2.3 clips, 6–12 s each |
| Prepared MuseTalk caches | one per physical clip |
| Output | an immutable pose bank + a runtime manifest exposing exactly 6 logical pose IDs |

At the default roster — flagship slot on `full`, the rest on `core` — that is **332 characters
and 1,784 LTX renders**. On the single RTX 5060 Ti that produced the accepted bank, that is a
multi-month queue. Read [Scale reality](#scale-reality) before scheduling a bulk run.

## The six-pose contract

Fixed by [`scripts/pose_protocol.py`](../scripts/pose_protocol.py). Do not add, rename, or drop one.

| Logical pose ID | Role | Physical clip (`full` tier) |
|---|---|---|
| `neutral_resting` | idle | shared idle/listening loop |
| `active_listening` | listening | shared idle/listening loop |
| `speaking_direct` | talking | V14 subtle, with V15 reference-paced as a rotating variant |
| `nod_agree` | reaction | compact yes nod |
| `empathetic_head_tilt` | reaction | slow empathetic tilt |
| `light_smile` | reaction | moderate closed-lip smile |

Neutral and listening deliberately alias one file. `speaking_direct` is the only pose allowed
to carry physical variants. The chat model never sees a variant ID — it emits semantic delivery
labels (`direct`, `warm`, `empathetic`) and the router does the rest.

## Pipeline

```
build_character_roster.py    104 languages x archetypes  ->  state/roster.json
render_portrait_prompts.py   roster                      ->  state/portrait_prompts/*.md
        ↓  ChatGPT, interactively, through Codex's own image tool  ↓
ingest_portraits.py          inbox PNGs                  ->  state/portraits/*.png  (480x832)
generate_pose_videos.py      portrait + prompt pack      ->  state/renders/<id>/*.mp4
certify_pose_bank.py         renders                     ->  state/certified/<id>/*.mp4 + report
package_pose_bank.py         certified                   ->  assets/ltx23_pose_banks/<bank>/
                                                             configs/pose_test/<pose_set>.json
prepare_musetalk_avatars.py  bank                        ->  one MuseTalk cache per clip
```

`run_character_pipeline.py` drives all of it and pauses at the ChatGPT boundary.

For the reduced Lingua set, `scripts/generate_three_pose_videos.py` is the roster-free entry
point. It accepts one portrait and emits `idle.mp4`, `talking.mp4`, and `smiling.mp4` with the
compact native LTX 2.3 Q4 graph used in the September 22 Indian-avatar talking test: 512x832,
24 fps, the same portrait at guide indices `0` and `-1`, one eight-step Euler
schedule, and the accepted 64/16 tiled decode. The current Japanese
`config/prompt_packs/japanese_selected_native_three_pose_v3.json` pack generates an 81-frame
idle cycle and repeats it three times to deliver 241 frames; talking and smiling are native
241-frame renders. It does not use SoulX, Segmind, Prompt Relay, or NAG. The script's historical
default pack is not the selected Japanese result. See the pilot workflow for the command and
endpoint behavior.

## Why certification is the load-bearing step

MuseTalk switches poses at a clip boundary. If two clips do not end and begin on the exact same
decoded pixels, every switch shows a visible jump. The accepted production banks share one
`decoded_boundary_rgb_sha256` across every physical file — that single hash is what proves all
30 ordered transitions between six clips are clean.

LTX will not hand that over. `certify_pose_bank.py` enforces it: freeze both ends of every clip
onto one shared anchor frame, cross-fade the organic motion in and out so the freeze does not
read as a stutter, re-encode at fixed QP with keyframes forced on both handle boundaries, then
decode again and require the handles to match across every clip. **That check is a hard gate.**
A bank that fails it must not be packaged.

## Configuration

| File | What it controls | Status |
|---|---|---|
| `config/languages.json` | the 104-language roster | **seed — replace with Lingua's authoritative list** |
| `config/archetypes.json` | casting pools per region, per language, and language-agnostic | seed — wants art-direction review |
| `config/pose_spec.json` | the six-pose contract, frame arithmetic, tiers | locked to the runtime contract |
| `config/generation_defaults.json` | ComfyUI node IDs, sampler, seeds | transcribed from the winning Q4 graph |
| `config/prompt_packs/pose_prompt_pack_v1.json` | per-pose LTX prompts | derived from the accepted V6/V8/V14/V15 profiles |
| `config/prompt_packs/portrait_prompt_template.md` | the ChatGPT portrait prompt | the proven LumaTalk companion prompt |

Character IDs derive from `languages.json[].code`, so changing a code after generation orphans
that language's banks. Settle the language list first.

## Tiers

| Tier | Renders | When |
|---|---:|---|
| `full` | 6 | language-agnostic characters and each language's flagship slot |
| `core` | 5 | the bulk run — drops only the V15 speaking variant |
| `minimal` | 2 | smoke tests only; reactions alias the idle loop and stop reading as distinct |

All three still emit all six logical pose IDs. `minimal` is a visible quality compromise, and
`package_pose_bank.py` prints every alias it had to invent.

## Scale reality

1,784 renders at the accepted durations is the headline number, and the accepted bank needed
rerolls on listening and empathy, so budget above it. Three things follow:

- **Measure before scheduling.** `generate_pose_videos.py` appends every observed render
  duration to `state/timings.jsonl`. After a handful of characters the estimate stops being a
  guess.
- **Stage the roster.** Generate launch languages first. `--languages` and `--characters` take
  explicit lists everywhere, and the ledger makes the run resumable.
- **Consider reuse.** The earlier LumaTalk work shared 12 characters across 15 languages. If a
  language's three characters do not have to be unique people, the roster collapses by an order
  of magnitude. The current design assumes they *are* unique, per the brief.

## Validation status

Verified here, without a GPU:

- the roster builds to 332 characters with no duplicate face or display name inside a language,
  and a 169/163 gender split;
- every render's `segment_lengths` sums to its certified frame count, and every frame count
  satisfies LTX's `1 + 8*round((seconds*fps - 1)/8)`;
- every node ID in `generation_defaults.json` exists in the live Q4 graph;
- prompts fill with correct pronouns and sentence casing;
- portrait ingest accepts a 941x1672 portrait and rejects a square one;
- **certification was run end-to-end on synthetic renders**: six clips, 30 ordered transitions,
  one shared boundary hash — and corrupting a single frame in one clip's tail handle makes the
  gate fail and name the offending clip;
- the emitted runtime manifest is accepted by MuseTalk's real
  `test_pose_webrtc.py::load_pose_set` and has field parity with the live production manifest.

Not verified: anything requiring a GPU — LTX render quality, whether real LTX output survives
the handle freeze without visible stutter, MuseTalk avatar preparation, and WebRTC playback.

## Refinement notes for Codex

The seams most likely to need work, in the order they will bite:

1. **`config/languages.json` is a seed.** Reconcile it against Lingua's supported languages
   before generating anything.
2. **The handle freeze is untested on real LTX motion.** Six frozen frames at each end may read
   as a stutter at 24 fps if LTX did not leave a quiet interval. If it does, lengthen the blend
   rather than shortening the freeze — the freeze is what the cross-cut depends on.
3. **Fixed QP is doing real work in certification.** If the boundary gate fails on real renders,
   lower `--qp` before questioning the approach; rate-control state is what makes identical
   input frames decode differently across files.
4. **Casting pools are seed quality.** Names especially want a native-speaker pass.
5. **Nothing here reviews a render.** The accepted bank came from human review at normal
   playback speed. Build a contact-sheet or side-by-side review step before trusting a bulk run.
