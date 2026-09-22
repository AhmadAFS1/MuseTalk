# Source-portrait prompt template

This is the ChatGPT image-generation prompt that produces the single source frame every
pose render is conditioned on. It is the proven LumaTalk companion-portrait prompt
(`lumatalk_completed_videos_.../remake/source_frames/imagegen-prompts.md`), tightened for
MuseTalk pose banks.

Image generation runs **interactively through Codex's own ChatGPT image tool**, not through
an API. `render_portrait_prompts.py` writes one prompt file per character; Codex pastes each
one, saves the result into the inbox, and `ingest_portraits.py` validates it.

## Template

Placeholders in `{braces}` are filled from `state/roster.json`.

```
Use case: ads-marketing. Asset type: source frame for an AI language-companion video call.
Create a photorealistic vertical smartphone selfie portrait of a friendly {subject}, looking
directly into the camera with a calm attentive closed-mouth half-smile. Place them in
{setting}, softly blurred and without landmarks. They have {hair} and wear {wardrobe}. Use
highly realistic candid UGC phone-camera photography, natural skin texture and pores, no
beauty filter. Frame at 9:16 in a tight medium close-up from the chest up, centered at arm's
length, with both shoulders visible and no phone visible. Use {lighting}. Preserve stable
realistic facial anatomy and an unobstructed face. No text, logo, watermark, extra people,
distorted hands, artificial bokeh, or glamour-editorial styling.
```

## Why each clause is there

| Clause | Purpose | What breaks without it |
|---|---|---|
| `closed-mouth half-smile` | MuseTalk drives the mouth; the source must not pre-commit a mouth shape | Teeth and lip artefacts fight the overlay — this was the single hardest defect in the LTX work |
| `tight medium close-up from the chest up` | The user rejected the full-shoulder framing as too small in a vertical call frame | Subject occupies too little of a 480x832 frame |
| `centered at arm's length` | Anchors the subject scale LTX must hold for 12 s | LTX drifts the framing between poses, so the six renders will not cut together |
| `no phone visible` | A visible phone becomes an artefact once the head moves | Hands and phone warp during motion |
| `natural skin texture and pores, no beauty filter` | Keeps the identity photographic | Plastic skin, which survives into every pose render |
| `unobstructed face` | Hair or hands over the face destroy MuseTalk's face detection | Avatar preparation fails or picks a bad bbox |
| `no text, logo, watermark` | Any baked text warps under motion | Unusable renders |

## Delivery requirements

- Vertical 9:16. Generate at the largest 9:16 size available; `ingest_portraits.py` downscales
  to the canonical 480x832 with bicubic and records both hashes.
- PNG, RGB, no alpha.
- One face only, facing the camera, eyes open, mouth closed.
- Save as `state/portrait_inbox/{character_id}.png`.

## Rejection criteria — regenerate rather than proceed

A bad portrait multiplies: it is re-encoded into five or six 10-second renders that then need
certification and avatar preparation. Reject and regenerate if any of these are true.

1. The mouth is open or teeth are visible.
2. The gaze is off-camera or the head is turned more than slightly off-axis.
3. Hair, a hand, or a microphone crosses the face.
4. Any text, logo, or watermark is present.
5. The crop is tighter than mid-chest or looser than mid-torso.
6. Skin reads as retouched or illustrated rather than photographic.
7. The face sits far off the horizontal centre; a decentred face drifts further under motion.
8. The character is visibly the same person as another character in the same language.
