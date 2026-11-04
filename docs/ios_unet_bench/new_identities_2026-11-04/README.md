# Thirty new characters: full model versus trained student

**Status: all 30 portraits generated; GPU quality evaluation pending.** These are new fictional identities
created after the student weights were frozen, using the built-in image generator without reference images.
They will never be used for training or checkpoint selection. The previously reported student quality failure
remains unchanged.

![All 30 new identities](contact_sheet.jpg)

[Character profiles and complete prompts](characters.json), [generation provenance](generation_provenance.json)
and [input face checks](portrait_checks.json) record the accepted images. All 30 have distinct source hashes
and exactly one detectable 468-landmark face after the documented normalization. This checks the inputs;
it does not establish lipsync quality. Personality descriptions are authored fictional concepts.

Each character will receive the same two held-out speech clips, producing 60 full-model/student comparisons.
The [fixed evaluation plan](plan.md) defines the models, hashes, thresholds, controls and required coverage.
This uses static source portraits: lips move during inference, while head/body motion remains frozen.
It tests a broader set of new faces, and cannot establish support for every possible character or pose.

Small JPEG previews are committed in [portraits/](portraits/). Native 1254-square source PNGs remain locally
under `generated/new_identities_2026-11-04/portraits/`, with hashes here and full source copies in the eventual
result archive. Previews are resized copies; evaluation preparation uses the native sources.

The user explicitly approved one additional validation-only RTX 3090, at most $0.30/hour for four hours,
with expiry installed before creation. No additional training is authorized. Previous rental spend is
$1.72268 estimated; the new rental receipt and combined estimate will be recorded after cleanup.
