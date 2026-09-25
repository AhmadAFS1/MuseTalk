# Resuming and validating a realtime character

The current three-pose entry point is
`scripts/build_realtime_character.py`. The accepted motion recipe remains
[PERFECT_THREE_POSE_PROMPTS.md](PERFECT_THREE_POSE_PROMPTS.md); the safeguards
below do not change prompts, seeds, or generation settings.

Use one output directory for one immutable character version. A new portrait,
prompt pack, graph, guide-fit policy, or shared-anchor policy requires a new
directory. Rendering records those input hashes and options as
`generation_fingerprint` before starting the GPU worker. Repeating an identical
request verifies completed video bytes and the prepared guide before resuming.
Completed pose records survive a partial resume; only missing poses render.
Corrupt, missing, or untracked completed videos fail with a specific error.
`--force` rerenders the same inputs; it cannot bypass an identity mismatch.

An interrupted job can be resumed with its original command. If every requested
clip is complete and verified, the command returns without changing the guide,
graphs, videos, or manifest. A dry run cannot replace completed provenance.
Historical videos without the immutable fingerprint remain supported through
`--source-dir`; their old render jobs cannot safely resume through `--image`.

Packaging verifies cached measurements and atlas data on every invocation:

- Actual input-file hashes must match both artifacts.
- Streaming video decode verifies frame count, dimensions, fps, and shared
  decoded first/last frames against the saved measurements.
- Measurement rows must have consecutive frame numbers and finite geometry.
- The atlas must pass the runtime's structural validation.
- Every phase of idle, talking, and smiling must have an admissible idle return.
  Talking and smiling must each have an admissible entry from idle. Missing
  coverage is an error, since the runtime would otherwise suppress that pose.
- After initial packaging, altered measurement content or altered route data
  requires a new version directory. Updating atlas `status` and `review`
  metadata is allowed; it does not change motion routes.

The package keeps existing preparation and review evidence when the source
hashes are unchanged. Its status follows the atlas status. A normal resume
therefore preserves prepared-cache records without requiring another server
preparation request. JSON publication is atomic. Packaging also refreshes the
small `motion-registration.json` discovery record if it is absent or stale,
without rewriting an unchanged atlas.

These checks establish integrity and route availability. They do not establish
unnoticeable transitions, identity quality, natural motion, or lip-sync quality.
Review the received WebRTC recordings for each new character before marking its
atlas `reviewed`.




## Reproduction note: CFG 1 and new portraits

With its default CFG-1 optimization, the
[installed sampler](/workspace/experiments/soulx_ltx_motion_pilot_20260922/A1/ComfyUI/comfy/samplers.py:608)
omits the negative-conditioning branch at `cfg=1`. The accepted Japanese
[render graph](/workspace/experiments/japanese_ltx_fixed_distance_20260925/graphs/idle-generation.json)
uses that same setting. Saved negative prompt text remains part of the exact
recipe/provenance; its presence does not mean those constraints are sampled.
Do not silently change CFG or remove the recorded negative text.

Exact prompt reuse does not guarantee every portrait/seed passes. The Latina
[idle seed193](/workspace/experiments/latina_fixed_distance_multipose_20260925/sources/idle.mp4)
failed on 58/241 frames above the 3.5 px mouth-gap gate
([rejection evidence](/workspace/experiments/latina_fixed_distance_multipose_20260925/initial-quality-rejection.json)).
The otherwise unchanged
[idle seed194](/workspace/experiments/latina_fixed_distance_multipose_20260925/idle-reroll-seed194/sources/idle.mp4)
failed on 86/241 frames, with a maximum gap of 15.226 px
([rejection evidence](/workspace/experiments/latina_fixed_distance_multipose_20260925/seed194-quality-rejection.json)).
The final unchanged-text
[idle seed195](/workspace/experiments/latina_fixed_distance_multipose_20260925/idle-reroll-seed195/sources/idle.mp4)
failed on **71/241 frames**, with a maximum gap of **8.556367874145508 px**
([hash-bound rejection](/workspace/experiments/latina_fixed_distance_multipose_20260925/seed195-quality-rejection.json)).
Generation took 330.54 seconds and decoding 30.04 seconds. All three idle attempts
are rejected under the unchanged 3.5 px gate. No Latina character package or
MuseTalk caches were published, and no further seed run is scheduled. A successful
render or assembly does not make a character automatically ready for use.
The sampler finding does not prove why these particular samples opened their
mouths or that every future seed will fail.

## Rerolling one failed pose without replacing good poses

A reusable prompt does not guarantee every seed passes on every portrait. For
example, the first Latina render using the approved pack opened its mouth in the
idle clip. Talking and smiling can remain intact while idle is retried.

The commands below document the reusable selective-reroll procedure. They do
not authorize another run for the rejected Latina portrait or publication of its
failed selections. A future retry needs a separate deliberate decision.

Copy the approved pack to a new seed-specific JSON file and change only
`poses.idle.seed`. Retain the positive and negative prompts, frame count, repeat
policy, and all other generation options. Use the same portrait and guide policy
and a new generation directory; render only the failed pose:

```bash
/workspace/experiments/soulx_ltx_motion_pilot_20260922/A1/.venv/bin/python \
  character_factory/scripts/generate_three_pose_videos.py \
  --image /absolute/original-portrait.png \
  --output-dir /absolute/idle-reroll-seed194/sources \
  --prompt-pack /absolute/idle-seed194-pack.json \
  --poses idle --guide-fit center_crop --shared-anchor
```

Keep the original completed trio and its generation manifest. Assemble a new
source selection using the rerolled idle and the original talking/smiling:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python \
  character_factory/scripts/assemble_three_pose_sources.py \
  --idle-manifest /absolute/idle-reroll-seed194/sources/manifest.json \
  --talking-manifest /absolute/original-trio/manifest.json \
  --smiling-manifest /absolute/original-trio/manifest.json \
  --output-dir /absolute/selected-seed194
```

The default comparison pack is the approved fixed-distance/shared-anchor pack.
Use `--approved-prompt-pack /absolute/approved-pack.json` when the character uses
another explicitly chosen baseline, such as an approved identity adaptation.
For each selected pose, the exact positive/negative text and generation profile
must match that baseline, apart from its seed and descriptive prompt citation.

Assembly checks the hashes of the original manifests, portrait, prepared guide,
prompt files, graph template, and selected MP4s. It verifies the saved generation
graph against the template and checks actual decoded dimensions, fps, frame count,
and shared loop endpoints. It refuses mixed portraits, guides, workflow geometry,
changed motion text, or unrelated graph changes. Only byte-identical copies of the
selected delivery videos and guide are written; source masters remain untouched.

The output contains `idle.mp4`, `talking.mp4`, `smiling.mp4`, the copied guide, and
`assembly-provenance.json` with every selection's manifest hash, file bindings,
seed, prompt text, and decoded metadata. It deliberately contains **no fabricated
single-job generation manifest or generation fingerprint**. An identical assembly
resumes safely after a copy interruption; changing any selected input requires a
new output directory. A corrupted existing output is rejected.

Finally, measure and package this selection through the existing ingest route,
using a new package version so old measurements cannot be reused:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python \
  character_factory/scripts/build_realtime_character.py \
  --source-dir /absolute/selected-seed194 \
  --output-dir /absolute/realtime-characters/character-seed194 \
  --character-id character_seed194
```

Assembly establishes provenance and source compatibility, **not pose quality**.
The per-frame measurement and transition-coverage gates can still reject the new
seed. Received WebRTC recordings and explicit visual review remain required.

## Recording an operator's visual decision

First create the source-bound review page and `verification.json` from the actual
received recordings. This command runs diagnostics and leaves visual acceptance
pending:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/review_motion_evidence.py \
  --directory /absolute/character-recordings \
  --atlas /absolute/character-package/motion-atlas.json \
  --label "Character transition review"
```

Watch `review.html` at normal speed, including the short reply, the full
talking/smiling reply, and the interrupted reply followed by another turn.
Only after an operator accepts those recordings, record that explicit decision:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python \
  character_factory/scripts/review_realtime_character.py \
  --character-dir /absolute/character-package \
  --evidence-dir /absolute/character-recordings \
  --decision accepted --reviewer "Operator name" \
  --notes "Describe the visual review and any remaining limitations." \
  --watched-normal-speed
```

The command has no default decision. Acceptance requires both `accepted` and
the normal-speed viewing attestation; passing diagnostics alone cannot approve a
bank. To record an unsuccessful visual review, use `--decision rejected` and
describe the visible issue in `--notes`. Rejection does not require the viewing
attestation and leaves the bank disabled unless the server deliberately enables
the unreviewed pilot override. Both decisions require complete, unchanged,
source-bound evidence with passing automated checks. Failed or incomplete
evidence cannot be used to publish a recorded-review decision.

The review command rechecks source hashes, routing identity, each report and MP4
hash, actual media streams, timing diagnostics, and the required cases
`short-idle`, `long-talking-smiling`, and `interrupted-and-next-turn`. A case label
alone is insufficient: the evidence must show its expected body sources, completed
entries/returns, and a live talking interruption followed by another reply.
An unexplained change to the full atlas file requires regenerating verification.
A prior valid review receipt can explain a status-only atlas change.

Each decision writes an immutable `reviews/<review-id>.json` receipt containing
the supplied reviewer name, notes, decision, input hashes, and viewing attestation.
This is a local record signed by name, **not a cryptographic signature or proof
of who operated the command**. The atlas and package reference the receipt and
retain preparation records and unchanged routing hashes. Atlas and registration
publication use atomic JSON replacement; a publication error restores the prior
bank/package state. An unreferenced receipt from a failed publication does not
activate a bank. Earlier receipts remain available in `review_history`.

CPU regression checks (no GPU or server is started):

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python -m unittest \
  test_character_factory_realtime test_character_recorded_review \
  test_three_pose_source_assembly test_motion_registry -q
```

The focused tests exercise wrong-portrait refusal before writes, prompt/graph/
option changes, completed and partial resumes, corrupt outputs, real decoded
test-video metadata, route coverage, stale measurements, and preparation/review
state preservation.

Recorded-review tests use synthetic local audio/video fixtures. They verify
explicit viewing attestation, evidence tamper rejection, required case behavior,
acceptance/rejection state changes, receipt history, and publication rollback.
They do not accept any real character bank.
