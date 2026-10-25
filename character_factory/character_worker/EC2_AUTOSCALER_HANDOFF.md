# Handoff: temporary character-creation instances from the EC2 autoscaler

Prepared 2026-10-25 from the MuseTalk repository at commit `e5c7615` (local `main`, not yet pushed).

**Goal.** When the Lingua backend receives a "create character" request, the EC2 autoscaler rents a temporary Vast.ai
GPU instance from a dedicated template, waits for it to report ready, sends it one `POST /createCharacter`, collects
the result (three MuseTalk avatar IDs already in S3), and destroys the instance. The instance installs everything
itself from the template's on-start script. No SSH, image build or manual step is needed.

**Who this is for.** Whoever changes the EC2 backend, which holds the existing autoscaler and the Vast provisioning
code. That code is not in this repository. This document gives the worker side's exact contract and a reference
implementation to copy from. The EC2 side still has to be read before changing it.

## 1. Current status

| Item | State |
| --- | --- |
| Worker code (`character_factory/character_worker/`) | Committed locally as `e5c7615`. **Not pushed.** The template clones `main` from GitHub, so a fresh instance cannot run it until it is pushed. |
| QC gate | Calibrated on 19 accepted clips. 12 glitch-injection tests pass. |
| Worker API | 9 tests pass with the fake pipeline. |
| Reference orchestrator (`orchestrator/charfactory.py`) | 7 tests pass against a fake Vast API and the real worker API. |
| Vast offer search | A live read-only query returned 25 matching Ada GPU offers on 2026-10-25. |
| Fresh Vast boot from the template | **Never run.** |
| Full GPU job through `pipeline.py` | **Not completed yet.** Three local runs on the shared RTX 4070 SUPER box stopped at the H3 start-up guard. The guard refuses a GPU that already has more than 1,000 MiB in use, and the production MuseTalk server was holding it (records in `validation/japanese_20261024/`). A dedicated instance does not have this conflict. |
| EC2 integration | Not started. |

An uncommitted change from another session removes `--fast-disk --disable-pinned-memory` from the H3 launch in
`h3_avatar_workflow/three_pose_h3_stage.py`, to speed up H3 on slow volumes. The accepted September runs used those
flags on a 30 GB RAM host. Decide whether that change is committed before the first real run. Without the flags, H3
holds more model weight in host RAM, so give the first instance at least 32 GB of RAM.

## 2. How it fits together

```
Lingua backend (EC2)
  POST /avatar/characters  ──▶  autoscaler: "character" pool, min 0, max N, one job per instance
                                   │ 1 search offers, rent with the character template hash, label charfactory-<id>
                                   │ 2 poll GET /health on the mapped port until ready (or boot failed -> destroy)
                                   │ 3 POST /createCharacter (bearer token)
                                   │ 4 poll GET /characters/<job_id>, or receive the callback
                                   │ 5 store the three avatar IDs; POST /worker/drain; destroy the instance
                                   ▼
Vast instance (template "charfactory-h3-worker", port 8010)
  on-start: clone MuseTalk main -> vast_onstart_character.sh
    preflight (GPU, RAM, disk, driver) -> worker API up (reports the phase on /health)
    -> install: MuseTalk + avatar prep, AWS runtime secret, pinned MiniMax H3 (61 GB weights, SHA-256 checked) -> ready
  per job: portrait check -> H3 idle/talking/smiling -> QC (one reroll) -> MuseTalk /avatars/prepare x3 -> S3 upload
           -> S3 HEAD check -> 5 s lip-sync smoke render + QC -> records -> callback
```

The serving MuseTalk workers need nothing new. They restore the new avatars from S3 the first time a session asks
for them ([docs/avatar_s3_persistence.md](../../docs/avatar_s3_persistence.md)).

## 3. Blocking prerequisites

1. **Push MuseTalk `main`.** The on-start script clones `https://github.com/AhmadAFS1/MuseTalk.git` (public; it
   answered HTTP 200 without credentials) at `main`. To pin a revision instead, set `MUSETALK_REF=<commit>` in the
   template env.
2. **A read-only GitHub token for `AhmadAFS1/minimax-h3-workflows`.** That repository is private (HTTP 404 without
   credentials). Pass it as `H3_WORKFLOWS_GIT_TOKEN`. The installer hands it to git through `GIT_ASKPASS`, so it never
   appears on a command line or in a log.
3. **The AWS runtime secret** `lingua/musetalk-worker-runtime` must provide `AVATAR_S3_BUCKET`, `AVATAR_S3_PREFIX`, the
   region and the S3 runtime key pair ([docs/musetalk_worker_secrets.md](../../docs/musetalk_worker_secrets.md)).
   The instance boot fails with "AVATAR_S3_BUCKET is not set" without them. `CHARACTER_WORKER_TOKEN` and
   `H3_WORKFLOWS_GIT_TOKEN` can live in this secret instead of the template, because the installer exports every key
   of the secret. The secret's `LINGUA_*` keys are harmless here: the worker strips them before starting its private
   MuseTalk API, so it never registers as a serving worker.
4. **A worker token.** Generate one (`openssl rand -hex 32`). The worker reads it as `CHARACTER_WORKER_TOKEN` and the
   autoscaler sends it as `Authorization: Bearer <token>`. Without a token the worker accepts unauthenticated requests
   and logs a warning.
5. **Optional IAM grant for job records.** Each job uploads its JSON records, QC contact sheets and logs to
   `s3://<bucket>/character-factory/jobs/<character_id>/<job_id>/`. The documented runtime policy covers only
   `avatars/*`, so this upload will be denied and the result will show `records.ok: false`. The job still succeeds.
   Add `s3:PutObject` on `arn:aws:s3:::lingua-musetalk-s3-storage/character-factory/jobs/*` to keep the records.

## 4. Create the Vast template once

Use a **separate template**. Do not reuse the MuseTalk serving template (this box runs hash `4dc0868e…`). That template
deletes and re-clones the checkout on every start and runs the serving install.

From a shell that has the values from section 3:

```bash
cd MuseTalk/character_factory/character_worker/orchestrator/template
export CHARACTER_WORKER_TOKEN=... MUSETALK_AWS_SECRET_ID=... AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=...   # secret-reader key only
export H3_WORKFLOWS_GIT_TOKEN=...      # or put it in the runtime secret
bash create_template.sh --print        # review; secrets are redacted
bash create_template.sh                # creates it with the Vast CLI and prints hash_id
```

The template's fields, in case you create it in the console instead ([template/README.md](orchestrator/template/README.md)):

| Field | Value |
| --- | --- |
| Image | `vastai/pytorch:cuda-12.1.1-auto` |
| Launch mode | SSH, direct |
| On-start | contents of [`orchestrator/template/onstart_cmd.sh`](orchestrator/template/onstart_cmd.sh) |
| Docker options | `-p 8010:8010 -p 22:22 -e CHARACTER_WORKER_PORT=8010 -e CHARACTER_WORKER_TOKEN=… -e MUSETALK_AWS_SECRET_ID=… -e MUSETALK_AWS_SECRET_REGION=us-east-1 -e MUSETALK_SECRETS_STRICT=true -e MUSETALK_SECRETS_VERIFY_S3=1 -e AWS_DEFAULT_REGION=us-east-1 -e AWS_ACCESS_KEY_ID=<secret-reader> -e AWS_SECRET_ACCESS_KEY=<secret-reader> -e H3_WORKFLOWS_GIT_TOKEN=…` |
| Disk | 200 GB |
| Search filters | `num_gpus=1 gpu_ram>=12 cpu_ram>=30 disk_space>=200 driver_version>=580.65.6 cuda_vers>=13.0 inet_down>=500 reliability>0.98 compute_cap>=890 compute_cap<1000 rented=False` |

Why the filters: the pinned H3 stack is torch 2.14 with CUDA 13.0, so it needs host driver 580.65 or newer. The
reference H3 run peaked at 11.5 GB VRAM and 27.6 GiB host RAM. The weights are 61 GB. The accepted recipe was made on
an Ada GPU (compute capability 8.9). Ampere is untested for H3's NVFP4 text encoder, and Blackwell has no MuseTalk
avatar-prep build. The on-start preflight enforces the same limits and fails fast with a readable message.

For the **first real run**, narrow the GPU to models with known H3 timings, for example
`gpu_name in [RTX_4070S,RTX_4070S_Ti,RTX_4090]`. The cheapest matching offers on 2026-10-25 were RTX 4060 Ti 16 GB
cards, whose H3 speed has never been measured.

## 5. The contract the autoscaler implements

### 5.1 Vast calls

These are the calls the official CLI makes (verified in its source, `vast.py`). The API key goes in the `api_key`
query parameter.

| Step | Request | Notes |
| --- | --- | --- |
| Search | `POST https://console.vast.ai/api/v0/bundles/` with the query as JSON, e.g. `{"gpu_ram":{"gte":12000},"cpu_ram":{"gte":30000},"driver_version":{"gte":580065006},"compute_cap":{"gte":890,"lt":1000},"verified":{"eq":true},"rentable":{"eq":true},"rented":{"eq":false},"order":[["dph_total","asc"]],"type":"on-demand","limit":50,"allocated_storage":200}` | RAM fields are in MB. The driver version is encoded as three zero-padded parts: `580.65.6` becomes `580065006`. `vast_client.parse_query()` converts the CLI syntax. |
| Rent | `PUT /api/v0/asks/{offer_id}/` with `{"client_id":"me","template_hash_id":"<hash>","disk":200,"label":"charfactory-<unique>"}` | Returns `{"success":true,"new_contract":<instance_id>}`. Send no `image` or `onstart`: either one overrides the template. |
| Read | `GET /api/v0/instances/{id}/?owner=me` returns `{"instances":{...}}`. `GET /api/v0/instances/?owner=me` lists all of them. | The endpoint is `public_ipaddr` plus `ports["8010/tcp"][0]["HostPort"]`. The mapped port is never 8010. |
| Destroy | `DELETE /api/v0/instances/{id}/` with body `{}` | Destroying stops both compute and storage billing. A stopped instance keeps billing storage. |

**A rent call that times out may still have rented.** Give every rent a unique label and, on the next pass, adopt any
instance whose label you issued but never recorded. `engine.py` does this in `reconcile_workers()`.

### 5.2 Readiness: poll `/health`; there is no registration

The character worker does **not** register or heartbeat with `/api/runtime/workers/register`. The autoscaler polls it
instead. `GET /health` needs no token and answers from the first minute of the boot:

```json
{"service": "character-worker", "status": "booting", "ready": false, "accepting": false, "busy": false,
 "boot": {"phase": "install", "ready": false, "failed": false, "message": "installing the pinned H3 runtime ...", "elapsed_seconds": 812},
 "queue_depth": 0, "current_job": null, "idle_seconds": 0, "gpu": {"name": "NVIDIA GeForce RTX 4090", "memory_total_mib": 24564},
 "versions": {"musetalk_commit": "...", "h3_workflows_commit": "168f143c...", "recipe": "h3-three-pose-20260927-v1"}}
```

| What you see | Meaning | Action |
| --- | --- | --- |
| Connection refused or timeout | The on-start script has not started the API yet (clone, preflight), or the instance is still loading | Keep waiting until the ready deadline |
| `ready: false`, `boot.phase` is `preflight`, `api`, `install` or `verify` | Installing | Keep waiting. `boot.message` says which step is running |
| HTTP 503 with `boot.failed: true` | Boot failed, and `boot.message` says why | Destroy the instance, then retry on a different offer |
| `ready: true`, `accepting: true` | Ready for a job | Submit |
| `draining: true` and `busy: false` | Drained and idle | Destroy |

Use a ready deadline of 60 minutes for a fresh instance (`CHARFACTORY_READY_TIMEOUT_S=3600`). Stop renting after three
consecutive instances fail to boot, because that pattern means the template or secret is broken. The reference engine
pauses renting for 30 minutes in that case.

### 5.3 Submit a job

`POST http://<ip>:<port>/createCharacter` with `Authorization: Bearer <CHARACTER_WORKER_TOKEN>`:

```json
{
  "character_id": "ja_01_haruka",
  "portrait": {"url": "https://<presigned S3 GET URL>"},
  "metadata": {"label": "Japanese woman", "age": 27, "gender": "woman",
               "hair": "straight shoulder-length dark hair tucked behind her ears", "tone": "light warm neutral skin",
               "facial_hair": "none", "wardrobe": "plain muted teal crew-neck top",
               "background": "softly blurred neutral home interior", "pronouns": "she"},
  "poses": ["idle", "talking", "smiling"],
  "options": {"seed": 42, "max_attempts": 2, "qc_profile": "strict"},
  "callback_url": "https://<ec2-backend>/internal/character-jobs/callback",
  "callback_token": "<per-job secret>",
  "external_id": "<your job id>"
}
```

- **Portrait.** Use `base64` (up to 30 MB) or `url`, which must be HTTP(S). A **presigned S3 URL** is the best choice
  for portraits stored in S3. `s3_uri` also works, but only for objects the worker's runtime key can read, which is
  `avatars/*` under the documented policy.
- **Metadata.** Supplying all six fields `label, age, gender, hair, tone, facial_hair` selects the exact accepted
  talking prompt. If any is missing, the generic prompt is used and the response lists a warning.
- **`character_id`** must match `[a-z][a-z0-9_]{1,39}`. It becomes the prefix of the avatar IDs.
- **Responses.** `202 {"job_id", "status": "queued", "position", "warnings", "poll"}` means accepted. `400` means the
  request is invalid; do not retry it. `401` means a wrong token. `503` means the worker is not ready or is draining.
  `429` means the worker's queue is full.

### 5.4 Follow the job

Poll `GET /characters/<job_id>` every 20-30 seconds, or rely on the callback, which delivers the same result body
with `Authorization: Bearer <callback_token>` and an `X-Character-Job` header. Polling is still needed as the
fallback, because the callback is tried only four times.

| `status` | Meaning |
| --- | --- |
| `queued`, `running` | In progress. `stage` is one of `ingest`, `generate`, `qc`, `prepare`, `verify`, `smoke`, `records`, `release`. `progress.pose` and `progress.attempt` show more detail |
| `done` | Three caches are in S3 and verified, and the smoke render passed QC |
| `qc_failed` | Every attempt failed QC, or the smoke render failed QC. `result.qc.hard_failures` names the checks. Do not serve it |
| `failed` | A stage errored. `failed_stage` and `error` say where |
| `cancelled` | Cancelled through `POST /characters/<job_id>/cancel` |

What to store from a `done` result:

```json
"published": {"endpoint_rgb_sha256": "...", "poses": {
  "idle":    {"avatar_id": "ja_01_haruka_d1383190_idle_93d291932f",    "s3_uri": "s3://lingua-musetalk-s3-storage/avatars/v15/ja_01_haruka_d1383190_idle_93d291932f.tar.gz"},
  "talking": {"avatar_id": "ja_01_haruka_d1383190_talking_f6cf8be7fb", "s3_uri": "..."},
  "smiling": {"avatar_id": "ja_01_haruka_d1383190_smiling_d73311869d", "s3_uri": "..."}}}
```

These IDs are what `/sessions/create` and the WebRTC pose router take. Also keep `result.qc`, `result.smoke_render`,
`result.attempts[*].seed` and `result.records.uri` for audits.

Artifacts stay downloadable until the instance is destroyed, through `GET /characters/<job_id>/artifacts` and
`.../artifacts/<path>`. These include `attempts/01/qc/<pose>.jpg` (contact sheets), `smoke/smoke.mp4` and every
stage log. Download any you want to keep before you destroy the instance, unless the records upload succeeded.

### 5.5 Tear down

When the job is terminal, send `POST /worker/drain`, then the Vast `DELETE`. For a batch, keep the instance for up to
15 minutes after its last job and send the next character to it. The install then happens once and each further
character costs only its processing time. Never destroy an instance while `busy` is true unless the job has timed out.

## 6. Mapping onto the existing autoscaler

The serving pool's rules in [docs/musetalk_autoscaling_plan.md](../../docs/musetalk_autoscaling_plan.md) do not carry
over unchanged. Add a separate pool:

| Setting | Serving MuseTalk pool | Character pool |
| --- | --- | --- |
| Template | the MuseTalk serving template | the character template from section 4 |
| Internal port | 8000 | 8010 |
| Readiness | worker registers, then `/health` 200 | autoscaler polls `/health` until `ready: true` |
| Capacity per instance | `LINGUA_WORKER_DEFAULT_CAPACITY` calls | 1 job at a time; the worker queues up to 20 |
| Minimum instances | 1 | 0 |
| Scale up | live capacity or queue signals | a queued character job and no ready, idle character instance |
| Scale down | drain, then idle for 15 minutes | drain right after the last job, or after 15 idle minutes for a batch |
| Session routing | sticky per session | none; the result goes into the Lingua avatar record |
| Job timeout | n/a | 3 hours (two attempts of three H3 poses plus prepare and smoke) |
| Retry on instance loss | re-route sessions | requeue the job on a new instance, at most twice |

Hard rule: **one character worker per GPU and nothing else on that GPU.** H3 refuses to start when more than 1,000 MiB
of GPU memory is already in use. Never schedule a character job onto a serving MuseTalk instance.

Per job, the EC2 store needs: the job ID, `character_id`, the request (the portrait stored by reference, not as
base64), the Vast instance ID and label, the worker endpoint, the remote `job_id`, status, stage, attempts, the result
JSON, the error, timestamps and the callback status. `orchestrator/job_store.py` is a working SQLite schema for this.

## 7. Reference implementation you can run as-is

`orchestrator/charfactory.py` is a complete, tested implementation of sections 5 and 6, using only the standard
library. You can use it two ways:

- **Read it as the specification.** `engine.py` covers reconcile, dispatch, poll and retire. `vast_client.py` has the
  exact Vast request bodies.
- **Run it on the EC2 box as the character pool,** and have the backend call it instead of extending the autoscaler:

```bash
cd MuseTalk/character_factory/character_worker
cp orchestrator/config.example.env /etc/charfactory.env    # set TEMPLATE_HASH, WORKER_TOKEN, API_TOKEN, VAST_API_KEY
set -a; . /etc/charfactory.env; set +a
python3 orchestrator/charfactory.py offers                 # read-only: what would be rented right now
python3 orchestrator/charfactory.py --dry-run tick         # full pass that never rents or destroys
python3 orchestrator/charfactory.py serve --port 8020      # HTTP API + loop: POST /createCharacter, GET /characters/<id>, GET /workers
```

`serve` accepts the same request body as the worker (section 5.3), persists the portrait, and returns its own job ID.
It also posts `{"event":"character.job","job":{...}}` to `CHARFACTORY_RESULT_WEBHOOK` when the job finishes.

## 8. Test ladder, cheapest first

1. **No GPU, no Vast.** On the EC2 box, run a fake worker:
   `CHARACTER_WORKER_TOKEN=test python3 character_api.py --fake --port 8010`. It moves through every stage in a few
   seconds and returns a realistic `done` result. `options.fake_outcome` set to `failed` or `qc_failed` exercises the
   error paths. Point the new autoscaler code at `http://127.0.0.1:8010` with a stub Vast client to test submit, poll,
   callback and storage.
2. **Repo tests:** `python3 -m unittest tests.test_character_api tests.test_orchestrator` (seconds; standard library
   only, passes on Python 3.10 and 3.12, so the EC2 box needs Python 3.10 or newer). The QC suite needs
   OpenCV: run `tests.test_video_qc` in the MuseTalk venv on a worker box.
3. **Read-only Vast:** `charfactory.py offers` and `charfactory.py --dry-run tick`.
4. **One real temporary instance,** narrowed to a GPU with known timings (section 4). Submit one character. Watch these:
   - `/health` on the mapped port, for the boot phase and message;
   - `charfactory.py events` or your own autoscaler log;
   - over SSH if needed: `/workspace/character_bootstrap.log`, `/workspace/character_worker/onstart.log` and
     `/workspace/character_worker/logs/`.
5. **Acceptance for that run:**
   - `status: done`, with three avatar IDs;
   - an S3 HEAD on each `s3_uri` returns a non-empty object whose `avatar-id` metadata matches;
   - a serving worker can open a session on the talking ID;
   - a person has looked at the three QC contact sheets and `smoke/smoke.mp4`;
   - the instance is destroyed and absent from `GET /instances/`.
   Record the measured boot and job times, which replace the estimates below.

## 9. Time and cost

The H3 and prepare times below were measured on the RTX 4070 SUPER. The rest are estimates until the first fresh run.

| Phase | Time |
| --- | --- |
| Fresh install: MuseTalk venv with avatar prep, then 61 GB of H3 weights downloaded and hashed | 20-40 min (estimate; mostly link speed) |
| H3 idle, talking, smiling | 327, 323 and 181 s measured, about 14 min |
| QC of the three clips | about 20 s measured |
| MuseTalk prepare and upload of three caches | 96 s measured, excluding S3 transfer of about 0.9 GB |
| Smoke render and QC | about 1 min (estimate) |
| One QC reroll | another 14 min |

A one-character temporary instance therefore runs about 40-60 minutes. At the 2026-10-25 offer prices of $0.19-0.58
per hour, that is roughly $0.20-0.60 of compute per character, plus disk storage while the instance exists. Batching
characters on one instance spreads the install cost across them.

## 10. Troubleshooting

| Symptom (`boot.message`, `error`, or a log) | Cause | Fix |
| --- | --- | --- |
| `compute capability X is outside the tested range` | Rented a GPU outside 8.9-9.x | Tighten the offer query. Set `CHARACTER_WORKER_ALLOW_UNTESTED_GPU=1` only for a deliberate test |
| `driver X < 580.65` | Old host driver | The offer query needs `driver_version>=580.65.6` |
| `only N GiB free` | Disk below 150 GB on a fresh boot | Rent with `disk: 200` |
| `AVATAR_S3_BUCKET is not set` | Runtime secret not loaded or incomplete | Check `MUSETALK_AWS_SECRET_ID` and the secret-reader key in the template, and the secret contents |
| `AWS Secrets Manager bootstrap failed (strict)` | Wrong ARN, or a reader key without `GetSecretValue` | [docs/musetalk_worker_secrets.md](../../docs/musetalk_worker_secrets.md), troubleshooting section |
| `could not fetch .../minimax-h3-workflows.git` | No or invalid GitHub token | Set `H3_WORKFLOWS_GIT_TOKEN` |
| `models.lock.json differs from the pinned lock` | The H3 repo moved or the lock was edited | Update `locks/character_worker.lock.json` on purpose, or pin the old commit |
| `H3 doctor failed after install` | Wheel or CUDA mismatch on that host | Read `logs/h3_install.log` and `logs/h3_doctor.log`, then try another host |
| `GPU is occupied; stop a local MuseTalk API before H3 generation` | Something else holds more than 1 GB of VRAM | Use a dedicated instance. See the hard rule in section 6 |
| `MuseTalk API not ready after 1200s` | The private prepare server failed | `musetalk_api.log` in the job directory |
| `status: qc_failed` | Both H3 attempts, or the smoke render, failed a hard check | Inspect `attempts/*/qc/<pose>.jpg`. Retry with a new `seed`, a better portrait, or richer metadata |
| `records.ok: false` | Runtime IAM key cannot write `character-factory/jobs/*` | Section 3, item 5. The job itself is fine |

## 11. Open decisions and known gaps

- **Visual approval.** QC catches technical faults and glitches: frozen frames, flashes, cuts, a lost face, broken
  loops, wrong audio. It does not judge realism. The recipe docs still call for a person to review each new
  character. Decide whether `done` characters go live automatically or wait for review in the backend.
- **Registration.** If the autoscaler can only track workers that register, add a register and heartbeat call to
  `character_api.py` with `worker_type: "character"`. It is not implemented, and polling `/health` is sufficient.
- **H3 launch flags.** Commit or drop the other session's flag change (section 1) before the first real run.
- **GPU range.** Widening to Ampere (an RTX 3090 at about $0.20/h) needs one validated end-to-end run first.
- **Fresh boot.** A fresh boot is untested. Expect to fix something on the first instance, and keep it alive (do not
  auto-destroy) until the logs have been read.

## 12. Files

| Path (under `character_factory/character_worker/`) | Purpose |
| --- | --- |
| `orchestrator/template/onstart_cmd.sh`, `create_template.sh`, `README.md` | The Vast template |
| `vast_onstart_character.sh`, `install_character_worker.sh` | What runs on the instance |
| `locks/character_worker.lock.json` | Pins: H3 commit, model lock, host requirements, ports |
| `character_api.py`, `pipeline.py` | Worker API and per-job stages |
| `qc/video_qc.py`, `qc/profiles.json`, `qc/calibration/` | QC gate, thresholds and calibration record |
| `orchestrator/charfactory.py`, `engine.py`, `vast_client.py`, `job_store.py`, `config.example.env` | Reference EC2 orchestrator |
| `tests/` | API, orchestrator and QC tests |
| `README.md` | Full worker documentation |
| `validation/japanese_20261024/` | Another session's local validation records, including the GPU-guard failures (not committed by this handoff) |
