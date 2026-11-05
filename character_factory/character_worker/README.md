# Character-creation worker for Vast.ai

One reusable boot script turns a fresh Vast.ai instance into a character factory, and one EC2-side
orchestrator rents those instances from a template hash, queues `createCharacter` requests, collects
the results and tears the instances down. The recipe is the accepted three-pose pipeline already in
this repository: one portrait -> three MiniMax H3 FL2VA loops (idle, talking, smiling) with shared
exact anchor endpoints -> three production MuseTalk v15 caches published to S3 through
`POST /avatars/prepare` ([h3_avatar_workflow/BATCH_THREE_POSE.md](../h3_avatar_workflow/BATCH_THREE_POSE.md)).
What is new here is the automation around it and the QC gate that rejects faulty or glitchy clips
before they are prepared or served.

```
Lingua backend ──POST /createCharacter──▶ charfactory.py serve (EC2)
                                          │  SQLite queue; rents Vast instances by template hash
                                          │  polls /health, dispatches, polls, destroys idle workers
                                          ▼
                              Vast instance (template "charfactory-h3-worker")
                                on-start ──▶ vast_onstart_character.sh
                                  1 preflight (GPU/RAM/disk/driver)   2 character_api.py up on :8010 (/health = phase)
                                  3 install_character_worker.sh       4 ready
                                character_api.py ──▶ pipeline.py per job:
                                  ingest+portrait check ─▶ H3 three poses ─▶ qc/video_qc.py (reroll once with a new seed)
                                  ─▶ owned MuseTalk API: /avatars/prepare x3 (S3 upload receipts) ─▶ HEAD every cache
                                  ─▶ smoke lip-sync render + QC ─▶ records to S3 ─▶ release local media ─▶ callback
```

**Integrating with the EC2 autoscaler:** [EC2_AUTOSCALER_HANDOFF.md](EC2_AUTOSCALER_HANDOFF.md) has the exact Vast calls,
the worker contract, the pool settings, a test ladder and troubleshooting for temporary instances.

## Files

| Path | Role |
| --- | --- |
| `vast_onstart_character.sh` | Vast on-start entry (run from the MuseTalk checkout). Preflight, start API, install, verify. Prints one `CHARACTER_WORKER_ONSTART COMPLETE`/`FAILED` marker; keeps `status.json` current. `--plan` prints what it would do. |
| `install_character_worker.sh` | Idempotent installer: apt prerequisites, `scripts/install_musetalk.sh --with-avatar-prep --with-chin-tools`, runtime secret from AWS Secrets Manager -> `runtime.env`, pinned `minimax-h3-workflows` checkout + `./install.sh` at the absolute paths the batch code expects (`/workspace/minimax-h3`, `/workspace/ComfyUI`, `/workspace/.venvs/comfy-h3`), `lib/h3` helper copies, `versions.json`. `--check` is read-only, `--plan` prints. |
| `locks/character_worker.lock.json` | The pins: H3 workflows commit `168f143c`, model-lock hash, recipe name, install paths, host requirements (existing Ada8.9–9.x at 12 GB / 29 GB, plus separately qualified exact RTX 3090 at 24,500 MiB actual VRAM/128,000 MiB effective RAM; driver>=580.65). Change pins here only. |
| `character_api.py` | Standard-library HTTP API + one-at-a-time job queue. Starts on the system python before any venv exists. Persists every job under `/workspace/character_worker/jobs/<job_id>/`. |
| `pipeline.py` | The per-job stages (below). Subprocesses only; the H3 and MuseTalk venvs are never imported into the API process. |
| `qc/video_qc.py`, `qc/profiles.json` | The glitch/technical gate; thresholds per pose and profile. `qc/calibrate.py` re-measures margins on known-good clips; `qc/calibration/` holds the 2026-10-25 record. |
| `tools/portrait_check.py`, `tools/s3_tool.py` | Portrait validation (decodable, size, one frontal face, position) and S3 head/get/put in the MuseTalk venv. |
| `orchestrator/charfactory.py` | EC2 CLI + `serve` mode. `engine.py` is the loop, `vast_client.py` the REST client, `job_store.py` the SQLite state, `config.example.env` the knobs. |
| `orchestrator/template/` | The Vast template: `onstart_cmd.sh` (paste into the template), `create_template.sh` (creates it with the CLI and prints the hash), `README.md` (field by field). |
| `tests/` | `test_video_qc.py` (glitch injection), `test_character_api.py`, `test_orchestrator.py` (fake Vast + real worker API in fake mode). |

## Worker API

All endpoints except `/health` require `Authorization: Bearer <CHARACTER_WORKER_TOKEN>` once a token is configured
(template env or a key in the AWS runtime secret).

```http
POST /createCharacter
{
  "character_id": "en_05_amara",
  "portrait": {"base64": "..."} | {"url": "https://..."} | {"s3_uri": "s3://bucket/key"},
  "metadata": {"label": "Black British woman", "age": 28, "gender": "woman",
               "hair": "short natural black coils", "tone": "deep neutral-brown skin", "facial_hair": "none",
               "wardrobe": "muted teal crew-neck knit top", "background": "softly blurred sunlit flat", "pronouns": "she"},
  "poses": ["idle", "talking", "smiling"],
  "options": {"seed": 42, "max_attempts": 2, "qc_profile": "strict", "smoke_render": true,
              "smoke_render_required": true, "release_local": true, "publish_records": true, "batch_size": 8},
  "callback_url": "https://backend/hooks/character", "callback_token": "..."
}
→ 202 {"job_id": "...", "character_id": "...", "status": "queued", "position": 1, "warnings": [], "poll": "/characters/<job_id>"}

GET  /characters/<job_id>            status: queued | running | done | failed | qc_failed | cancelled; stage; progress; result; error
GET  /characters/<job_id>/artifacts  qc json, contact sheets, logs, smoke.mp4 (…/artifacts/<path> downloads one)
POST /characters/<job_id>/cancel     cancels a queued job or interrupts a running one
GET  /health                         ready/booting/failed, install phase and message, queue depth, idle seconds, versions, GPU
POST /worker/drain | /worker/undrain
```

The six metadata fields `label, age, gender, hair, tone, facial_hair` select the exact accepted talking prompt
builder; without them the generic identity prompt is used and the response says so in `warnings`.
Recipe `h3-three-pose-20260927-v2-reference-defaults` preserves the exact reference clothing and background
when those fields are omitted. Explicit wardrobe/background text retains the accepted v1 prompt bytes.
The new recipe and plan signature reject reuse of older checkpoints that may contain invented scenery.

The controller may provide `external_id` and `options.submission_id`. Repeating that pair returns the
original remote job, including after a lost response or worker API restart. Different normalized
character inputs with the same pair return HTTP 409. Callback tokens and renewable S3 signing parameters
do not change input identity. A real retry after a lost worker uses a new submission token.

### Required mobile release

Workers advertise `capabilities: ["device_release/1"]`. A device-first caller supplies
`options.device_release` with `character_id` (the Lingua character ID), `job_id` and `release_id`
(the Lingua creation job ID). These jobs require all three poses and the smoke-render quality check.

After H3 quality checks, the worker uploads accepted sources, portrait, anchor, complete QC and generation
provenance under `avatars/releases/<release_id>/<worker_job_id>/`, then commits
`source-checkpoint.json`. Its descriptor is persisted on the worker job as `source_checkpoint` before
MuseTalk starts. A retry can pass this descriptor in `options.source_checkpoint`. The worker checks the
bucket and release identity, every downloaded hash, portrait, pinned model revision and recreated
batch-plan signature, restores the original sampler seed and remaps source records to the new host.
It fully decodes the restored clips, checks exact shared endpoints and reruns QC before skipping H3.
Changing the portrait, prompts, graph or model recipe rejects reuse. No unvalidated checkpoint falls
back silently to generation.

After preparation and the smoke render, the worker runs `prepare_avatar_bundle.py --phone-only` and
`pack_phone_bundle.py` against each local cache. It checks the entire archive (paths, counts, finite
float16 latents, crop geometry and all image decoding), then uploads mobile tar/sidecars and required
records. Every immutable upload uses SHA-256, conditional creation and a verified HEAD. `release.json`
is uploaded last; a failed mandatory artifact cannot produce a successful device release.

The result includes `device_release: {bucket, key, sha256, bytes, release_id}` pointing to a
`lingua-character-release/1` manifest with `renderer_abi: "musetalk-v15-phone-v1"`, explicit caller/job
identity, three pose bundle/source descriptors, and QC/provenance descriptors. The controller must
verify and activate this manifest before exposing ready status. Legacy jobs retain the existing output
contract. The fake worker exposes an explicitly fake local manifest for protocol tests; it never claims
verified S3 readiness.

### Result shape (`result.json`, also delivered to `callback_url`)

```json
{"job_id": "...", "character_id": "en_05_amara", "status": "done", "recipe": "h3-three-pose-20260927-v1",
 "identity": "en_05_amara_3f9a1c2b", "attempts": [{"attempt": 1, "seed": 42, "qc": {"pass": true, "...": "..."}, "h3_seconds": {"idle": 327.3, "...": 0}}],
 "published": {"endpoint_rgb_sha256": "...", "poses": {
     "idle":    {"avatar_id": "en_05_amara_3f9a1c2b_idle_93d291932f", "s3_uri": "s3://lingua-musetalk-s3-storage/avatars/v15/en_05_amara_3f9a1c2b_idle_93d291932f.tar.gz", "source_sha256": "..."},
     "talking": {"...": "..."}, "smiling": {"...": "..."}}},
 "s3_verification": {"pass": true}, "smoke_render": {"pass": true, "avatar_id": "...", "mouth": {"...": "..."}},
 "records": {"uri": "s3://.../character-factory/jobs/en_05_amara/<job_id>/", "ok": true}, "seconds": 1210.4}
```

The `avatar_id`s are exactly what `/sessions/create` and the WebRTC pose router expect
(`<character>_<pose>_<sha10>`, see [docs/avatar_s3_persistence.md](../../docs/avatar_s3_persistence.md)).

## Pipeline stages and failure handling

| Stage | What runs | Fails the job when |
| --- | --- | --- |
| ingest | portrait decoded, `tools/portrait_check.py` | not an image, < 512 px, no face / several faces, face off-centre or wrong size |
| generate | `batch_three_pose.py --stage generate --seed N` (owned ComfyUI on :8197, GPU lock) | H3 error, missing record |
| qc | `qc/video_qc.py` per pose + cross-pose endpoint set | any hard check fails on every attempt (default 2 attempts, seed 42 then 43; a reroll regenerates all poses because the three clips must share one anchor and one output directory) |
| prepare | owned MuseTalk API on :8198 (`run_musetalk_server.sh`, control-plane env stripped so it never registers with Lingua), `batch_three_pose.py --stage prepare --prepare-url` | API not ready in 20 min, prepare error, missing upload receipt |
| verify | `s3_tool.py head` on all three objects | any object missing/empty or `avatar-id` metadata mismatch |
| smoke | `POST /generate` with 5 s of `data/audio/eng.wav` on the talking cache, QC with the `smoke` pose | smoke render fails QC (`qc_failed`; the caches stay in S3 for inspection) |
| records | job JSONs, contact sheets and logs -> `s3://<bucket>/character-factory/jobs/<character_id>/<job_id>/` | never (best effort; the runtime IAM key may only cover `avatars/*`) |
| release | attempt MP4s and the owned local caches are deleted after verified publication | never |

Progress is visible while it runs: `stage`, `progress.pose`, `progress.attempt`, per-stage logs in the job directory.

## QC: what "not faulty or glitchy" means here

`qc/video_qc.py` streams every decoded frame once (6-7 s per 10 s clip on CPU) and reports values plus pass/fail
per check so thresholds can be tuned in `qc/profiles.json` without code changes. Hard checks fail the clip:

- contract: 512x896, 24 fps, exact frame count (240/240/158), duration, full decode with `ffmpeg -xerror`
- endpoints: first decoded frame == last decoded frame; all poses share one endpoint hash (the loop/cross-cut contract)
- frozen: runs of near-identical consecutive interior frames; jump: single-frame changes far above the clip's own median motion;
  luma steps (flashes); blank frames; flicker (soft)
- face: detected in >= 98.5% of frames, no gap > 6 frames, centre drift <= 12-14% of height, size drift <= 35%, no teleport
  between adjacent frames
- anchor: SSIM(frame 0, anchor) >= 0.97 (packaging guarantees the anchor is frame 0), mid-clip SSIM >= 0.35 (scene/identity drift proxy)
- audio: present only on talking; level >= -42 dBFS; clipping <= 0.1%
- mouth (soft): idle/smiling must stay closed, talking must articulate. With the MediaPipe venv the lip gap drives this;
  without it a mouth/face motion ratio does (closed references 0.86-1.01, talking 1.07-1.56)
- eyes (soft, landmarks only): no near-closed run longer than a second

Evidence: all 19 accepted clips pass every hard check ([qc/calibration/README.md](qc/calibration/README.md)), and
`tests/test_video_qc.py` injects a frozen run, a black frame, a flash, an identity jump, a face loss, missing audio,
unexpected audio, a wrong frame count, a broken loop, a wrong anchor and an endpoint-set mismatch into an accepted
clip and asserts the named check catches each one (12 tests, all passing on 2026-10-25). These are technical gates;
they do not replace the visual review the recipe docs still call for before a character is served.

## Vast template and orchestrator

1. Create the template once: `orchestrator/template/README.md` lists the fields; `create_template.sh` does it with the
   Vast CLI from your shell environment and prints the hash. The template's on-start is `template/onstart_cmd.sh`.
2. On the EC2 box: copy `orchestrator/config.example.env`, set `CHARFACTORY_TEMPLATE_HASH`, `CHARFACTORY_WORKER_TOKEN`
   (same value as the worker's `CHARACTER_WORKER_TOKEN`), `VAST_API_KEY`, optionally `CHARFACTORY_RESULT_WEBHOOK`.
3. `python3 orchestrator/charfactory.py offers` shows what the offer query matches right now (read-only). The default now selects exact RTX 3090/24GB/128 GB host after the 2026-11-05 Reina qualification; `engine.ADA_QUERY` preserves the old alternative.
   Historically, on 2026-10-25 the old query matched25Ada offers from $0.19/h (RTX 4060 Ti 16 GB) to $0.58/h (RTX 4090).
4. `python3 orchestrator/charfactory.py serve --port 8020` runs the HTTP API for the Lingua backend
   (`POST /createCharacter`, `GET /characters/<id>`, `GET /workers`) and the orchestration loop in one process;
   `queue`, `list`, `workers`, `events`, `tick`, `run`, `cancel`, `destroy-worker`, `destroy-all --yes` are the CLI equivalents.

Engine behaviour: one instance per queued job up to `max_workers`; the cheapest offer that passes the client-side filters
(price cap, driver >= 580.65, disk, RAM) is rented with the template hash and a `charfactory-<id>` label; readiness is
`/health.ready` on the mapped port; a worker whose `/health` reports `boot.failed`, or that is not ready within
`ready_timeout_s`, or whose instance disappears, is destroyed and its job requeued (up to `provision_retries`); after
`max_boot_failures` consecutive boot failures renting pauses for `provision_backoff_s` so a broken template cannot burn
money; a create call that times out is recovered by label on the next tick; idle workers are drained and destroyed after
`idle_ttl_s`; everything is in SQLite so the process can restart at any point.

## Operating notes

- Fresh-instance budget: the 61 GB H3 weights plus two venvs take 20-40 minutes depending on the host's link; the
  default `ready_timeout_s` is 3600. Restarts of the same instance skip the downloads (`h3_install.stamp`).
- Per character on an RTX 4070 SUPER class GPU: about 14 minutes of H3 (3 poses), 1-2 minutes QC, 2-3 minutes prepare,
  1 minute smoke render; a QC reroll adds another 14 minutes.
- The worker runs H3 and MuseTalk strictly in sequence (12 GB GPUs cannot hold both); the MuseTalk API lives only for
  the prepare and smoke stages and never registers with the Lingua control plane.
- Secrets: the template carries only the secret-reader key; S3 credentials, `AVATAR_S3_BUCKET` and optionally
  `CHARACTER_WORKER_TOKEN` / `H3_WORKFLOWS_GIT_TOKEN` come from `lingua/musetalk-worker-runtime`
  ([docs/musetalk_worker_secrets.md](../../docs/musetalk_worker_secrets.md)). Record uploads under `character-factory/`
  need that prefix added to the runtime IAM policy; without it the job still succeeds and `records.ok` is false.
- Local run on this box (no Vast): `CHARACTER_WORKER_TOKEN=... python3 character_api.py --port 8010` after the installer
  has been run once, or `python3 pipeline.py --run-job-dir <dir>` for a single prepared job directory.

## Verified and not verified (2026-10-25)

Verified on the RTX 4070 SUPER box: QC calibration on 19 accepted clips; 12 glitch-injection tests; 9 worker API tests
(fake pipeline); 7 orchestrator tests (fake Vast + real worker API); `vast_onstart_character.sh --plan`,
`install_character_worker.sh --plan` and `--check` (MuseTalk check OK, H3 doctor OK against the existing install);
a live read-only Vast offer query. Not yet run: a real Vast instance booting from the template, and therefore the
end-to-end GPU path (H3 -> QC -> prepare -> S3 -> smoke) under `pipeline.py`. Every GPU stage calls the same scripts
the 16-character batch used, but the first fresh-instance boot should be watched (`charfactory.py events`,
`/health` on the mapped port, `/workspace/character_worker/onstart.log` over SSH).


## Device release and hardware qualification

Lingua requests `options.device_release={character_id:<actual Lingua ID>,job_id:<logical job>,release_id:<same logical release>}`. The worker uploads a verified accepted-H3 source checkpoint before MuseTalk preparation, then all three phone-only pose archives, their sidecars, source clips, QC and provenance. `result.device_release={bucket,key,sha256,bytes,release_id}` points to the final `lingua-character-release/1` manifest under `avatars/releases/<release_id>/<remote_job_id>/`; that manifest is uploaded last. Legacy v15 cache uploads remain available. `options.source_checkpoint` accepts the returned checkpoint descriptor for the same identity/recipe/portrait; it verifies all hashes, decodes videos and reruns QC before skipping H3. `external_id` plus `options.submission_id` makes accepted-but-response-lost submission replays idempotent; a new attempt uses a new submission ID.

The 2026-11-05 RTX 3090 run created requested fictional adult Reina (`char_8ef13e81-d84e-5cce-bb16-d1ce38f12373`, release `charfactory_b3a60391c6914758886f9642e5de0266`). It passed three H3 sources and endpoints, MuseTalk cache preparation, teacher CUDA asset smoke, S3 publication, backend ownership/model-profile checks and 638-frame shipping-native validation. The observed 23,944 MiB peak fits actual 24,576 MiB VRAM. Initial deployment policy requires exact `NVIDIA GeForce RTX 3090`, compute8.6, actual total VRAM>=24,500 MiB and effective RAM>=128,000 MiB. Effective RAM is min(host MemTotal, finite cgroupv2/v1 limit). Other Ampere cards are not promoted; lower-RAM hosts need separate qualification. Existing8.9–9.x requirements are retained. Keep one active job and run H3 then MuseTalk sequentially.

The actual3090 worker profile is resolver recipe `r5`, eager teacher UNet and compiled TAESD, with TRT_FALLBACK=0. The engine group was normally dropped because the 4070 bundle is GPU-specific and AMPERE_PLUS was not restored. This does not qualify AMPERE_PLUS execution. CUDA smoke is explicitly `scope=asset-cache-qc`, `model_role=teacher`, `phone_student_acceptance=false`; it is separate from phone student validation. The phone uses one shared approved student pack (`student-r2-a05d2d98-v1`), not per-character training. Source bundles stay at 24 fps; shipping timing maps to 25 render ticks. Actual Reina student phone/RN short-turn checks reproduce 25 fps; uninterrupted 60 s phone tests are approximately 24.25 fps and must not be described as sustained 25 fps qualification.

`host_requirements.py` writes `hardware_profile.json`; onstart enforces it and versions/job provenance retain it. `CHARACTER_WORKER_ALLOW_UNTESTED_GPU=1` remains an explicit experimental capability override, not a bypass for a qualified3090's resource limits. The default provider offer filter uses 24000 for rounded provider VRAM, while the post-rental nvidia-smi guard requires 24500 MiB. Orchestration rechecks GPU name/capability/RAM locally in case a provider returns an out-of-query row.

Set `MUSETALK_REF` to the final immutable reviewed commit. The repository template bootstrap supports raw SHAs with atomic fetch/detached checkout; fresh/branch/existing/failed-fetch cases have offline git integration coverage. The existing private factory template c4de4d1ca7e553a5129d9e3027a4ccef was independently read and already uses a SHA-compatible gitinit/fetch bootstrap, so it can reach the final code pin without a new template. It does not fetch an old-main/raw bootstrap.
