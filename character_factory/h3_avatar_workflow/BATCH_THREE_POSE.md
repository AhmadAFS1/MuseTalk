# Batch H3 three-pose avatars with MuseTalk S3 caches

`batch_three_pose.py` consumes a recursive folder of already approved PNG, JPEG,
or WebP portraits. It makes **idle, talking, and smiling** H3 videos for each
portrait, prepares each physical clip through the production MuseTalk
`POST /avatars/prepare` path, and requires a confirmed upload of each prepared
avatar directory to S3. No image generation is part of this command.

This is a batch extension of the [accepted talking workflow](WORKFLOW.md).
For a portrait with the six identity fields below, the talking prompt is
exactly the accepted `create_avatar.prompt()` output. Idle and smiling motion
comes from the saved H3 first/last-frame prompt runs, with identity and pronouns
adapted to each portrait. The exact prompt submitted to ComfyUI is recorded in
`<output>/<identity>/<pose>/submitted_api.json`. A generic prompt derived from
the photo is used when metadata is absent; supplying metadata gives the model
more precise identity, hair, skin, clothing, and facial-hair cues.

## Input and command

Keep the output outside the input photo tree. Paths below are examples; point
`--images` at the actual large portrait folder. The metadata file is optional:

```json
{
  "images": {
    "people/jane.png": {
      "id": "jane",
      "label": "Japanese woman",
      "age": 26,
      "gender": "woman",
      "hair": "straight black hair tucked behind her shoulders",
      "tone": "warm-light skin",
      "facial_hair": "none"
    },
    "people/alex.png": {
      "id": "alex",
      "label": "Latina woman",
      "age": 27,
      "gender": "woman",
      "hair": "dark-brown loose waves tucked behind her shoulders",
      "tone": "olive-golden skin",
      "facial_hair": "none"
    }
  }
}
```

The earlier `diversity_batch_20260927.json` format with an `avatars` array and
`portrait` paths also works. Generic metadata can instead set `subject`,
`wardrobe`, and `pronouns` (`she`, `he`, or `they`). The exact accepted talking
prompt requires all of `label`, `age`, `gender`, `hair`, `tone`, and
`facial_hair`.

```bash
cd /workspace/MuseTalk
SCRIPT=character_factory/h3_avatar_workflow/batch_three_pose.py
PHOTOS=/path/to/approved_portraits
OUT=/workspace/experiments/h3_three_pose_batch
META=/path/to/portrait_metadata.json

python3 "$SCRIPT" --images "$PHOTOS" --metadata "$META" \
  --output "$OUT" --stage plan

# Run one identity first. --only accepts a photo stem or generated identity ID.
python3 "$SCRIPT" --images "$PHOTOS" --metadata "$META" \
  --output "$OUT" --stage generate --only jane
```

To generate and persist the whole folder while releasing each identity's local
MP4s and owned API cache after all three S3 uploads succeed:

```bash
export AVATAR_S3_ENABLED=1
export AVATAR_S3_BUCKET=lingua-musetalk-s3-storage
export AVATAR_S3_PREFIX=avatars
# Configure AWS credentials through the host's normal profile or workload role.

python3 "$SCRIPT" --images "$PHOTOS" --metadata "$META" \
  --output "$OUT" --stage all --local-api --release-local \
  --continue-on-error
```

The owned local API uses the installed TensorRT UNet and TAESD **decoder**;
MuseTalk's native avatar encoder remains in place. H3 and MuseTalk run one
after the other on the single GPU. For an already running production API,
run `--stage generate` first, stop H3 generation, then run `--stage prepare
--prepare-url http://127.0.0.1:8000`. The connected API must expose the
same production `/avatars/prepare` and `/stats` endpoints with S3 enabled.
`--limit N`, `--only`, `--poses idle talking smiling` (generation only), and
`--continue-on-error` allow controlled rollout. Do not use `--release-local`
until you want to rely on the verified S3 copies.

## Output and checks

Each identity gets one 512×896 portrait anchor used as **both H3 conditioning
frames for every pose**. H3 runs at 24 FPS, seed 42, eight steps, CFG 1, using
the installed low-VRAM INT8/NVFP4 graph. Idle and talking start at 243 native
frames and are packaged to exactly 240 frames (10 seconds). Smiling is 158
frames (6.58 seconds), following the saved smile run. Idle and smiling delivery
MP4s have no audio; talking retains H3's generated audio. The eight-frame
transition blend and intra-only H.264 make all three clips share the same
decoded first and last RGB frame. The controller checks full media decode and
the cross-pose endpoint hash before S3 preparation.

MuseTalk preparation is **per physical clip**. Each cache uses its own pose
clip for both its input and idle playback path, because the WebRTC pose router
reads that idle path for each physical pose. The three caches are then mapped
to the idle/talking/smiling pose set. Cache IDs include the photo identity and each source MP4's
content hash. The API archives the production avatar directory (including
`latents.pt`, frame PNGs, masks and coordinates), rather than the earlier
research `cache.pt` test file. The controller requires the API S3 destination,
upload counter, exact object key, and upload acknowledgement to agree before
writing `s3.json`; it writes `published.json` only after all three uploads.
Typical objects are `s3://<bucket>/avatars/v15/<avatar_id>.tar.gz`.

`plan.json`, each `spec.json`, `h3.json`, `s3.json`, and `published.json`
contain prompt/input hashes, media hashes, endpoint proofs, and S3 object keys.
`batch_status.json` lists failures. Runs are resumable with the same output
directory and unchanged inputs; a changed portrait, prompt implementation, or
graph requires a new output directory. `--release-local` deletes generated MP4s
and, for an owned local API only, the three newly created local avatar caches
after confirmed publication. It leaves the original photos and S3 objects.

Technical checks do not approve expression quality. Review each new person's
idle breathing, closed lips, smile return, speaking realism, identity and
facial hair, jaw and cheek seams, and pose transitions before serving it.
Earlier six-identity testing found moustache and near-mouth hair loss in
MuseTalk overrides; creating a production cache does not resolve that issue.
Native H3 audio on the talking clip is not verified exact speech, and muted
idle/smile clips can still have unwanted generated mouth motion.

## Verification performed on this machine

The planning pass discovered the six existing diversity portraits and emitted
18 pose specifications. All six talking prompts matched the accepted prompt
builder exactly. A [Black woman pilot](../../../experiments/three_pose_factory_pilot_20260927/)
generated [idle](../../../experiments/three_pose_factory_pilot_20260927/black_woman_435da43f/idle/source.mp4)
and [smiling](../../../experiments/three_pose_factory_pilot_20260927/black_woman_435da43f/smiling/source.mp4)
clips with full decode, no audio, and exact first/last decoded RGB equality.
The idle has 240 frames and the smile 158; both endpoint hashes equal the
previously accepted talking clip's endpoint hash
`fe9ca3d589391c9946cf8a6417eea789f76ffbb8fc4962c49b369567da2b1863`.
Sampled [idle](../../../experiments/three_pose_factory_pilot_20260927/idle_contact.jpg)
and [smile](../../../experiments/three_pose_factory_pilot_20260927/smiling_contact.jpg)
frames show the intended motions; normal-speed review is still required for
production approval. The first back-to-back run exposed a socket port reuse
error, which was fixed before resuming the smiling pose. The controlled S3
tests exercise a real multipart request against a local HTTP test server,
confirmed and failed upload responses, and source/idle file delivery; the
existing S3-store tests cover archive round trip. Live S3 publication requires
credentials and a bucket configured in the API environment; this shell had
neither, so no S3 object was uploaded in this pilot.
