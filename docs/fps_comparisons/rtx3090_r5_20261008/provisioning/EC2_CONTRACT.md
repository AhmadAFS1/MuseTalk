# EC2 discovery and startup observer contract

Observed October 8, 2026. This discovery issued no create/destroy request and changed no service or template. The accompanying JSON is the sanitized inventory. Root execution owns subsequent provisioning state; do not mistake a prepared offer for a rented instance.

## Live deployment and authorization

SSH alias `my-ec2` resolves to `ec2-user@18.205.211.142:22`, existing identity `~/.ssh/lingua-key.pem`, identity-only authentication and no agent forwarding. Repository `/home/ec2-user/lingua` is clean at `171e169682b08054fa7505df72431914d6653c8e`. No applicable AGENTS files were found. The relevant code hashes agree with the operator's Lingua checkout.

`lingua-api.service` serves `127.0.0.1:8000`; public control plane is `https://18-205-211-142.sslip.io`. Health and authenticated autoscaler/worker inventory reads returned HTTP 200. Standalone autoscaler units are disabled, no autoscaler process was found, MuseTalk autoscaling is false and min/max dynamic workers are zero. No dynamic MuseTalk workers were registered. Do not start the production autoscaler to run an isolated experiment.

Existing hourly caps are preferred $0.21, fallback $0.30, with 150 GB disk. Existing transfer caps are $0.015/GB: **too permissive for this experiment**. The human subsequently approved a $30 total experiment budget and $1.50/TB each for download/upload. Use the conservative $1.50/1024 = $0.00146484375/GB cap. Root chose a 12-hour initial development lifetime, with 100 GB reserved each direction and $2 additional reserve. Install the owned-resource expiry timer before the create request; a JSON budget is not itself an enforcement mechanism or new authorization.

The production DynamoDB paid-work control is disabled with monthly budget zero and no `allow_worker_provisioning`. Effective `PAID_WORK_CONTROLS_REQUIRED=false` bypasses it. This discrepancy was reported, not changed, and is not treated as permission to spend. Account balance is likewise not an approved budget.

## Exact provisioner API

All runtime paths below use `X-Admin-Token: <RUNTIME_CONFIG_TOKEN>` except worker callbacks. Fetch tokens in memory using the existing backend Secrets Manager bootstrap; never paste values into a command, report, or image.

| Operation | EC2 contract | Provider operation |
| --- | --- | --- |
| Create one worker | `POST /api/runtime/workers/musetalk/create` | `PUT /api/v0/asks/{offer_id}/` |
| Worker registry/pool | `GET /api/runtime/workers/musetalk` | No fresh provider query implied |
| Autoscaler snapshot | `GET /api/runtime/autoscaler` | Cached state; do not call `/autoscaler/run` as a read |
| Drain | `POST /api/runtime/workers/musetalk/{worker_id}/drain`, `destroy_when_idle` | No immediate destroy required |
| Destroy | `POST /api/runtime/workers/musetalk/destroy`, `instance_id`, `force:false` | `DELETE /api/v0/instances/{id}/`, or drain if busy |
| Offer search | Existing `VastClient.search_offers(filters)` on EC2 | `POST /api/v0/bundles/` |
| Provider inventory | Existing `VastClient.list_instances()` | `GET /api/v0/instances/` |
| Provider detail | Existing `VastClient.show_instance(id)` | `GET /api/v0/instances/{id}/` |

Create JSON is `{"count":1,"offer_id":<fresh inspected id>,"create_request":{...}}`; optional `offer_filters` is a filter object. Success is `{"success":true,"result":{"actions":[{"action":"create","worker_type":"musetalk","instance_id":"...","offer_id":...,"label":"..."}],"requested_count":1,"pending_before":...,"pending_after":...},"autoscaler":{...}}`.

An explicit `offer_id` bypasses EC2's price selector. The observer therefore verifies fresh allocated-storage pricing and all budget terms before POST. Vast can return offers whose actual allocated-storage hourly price exceeds the search filter: discovery rejected $0.321111/hr offers despite a $0.30 query cap. Offer IDs can change between searches; resolve a fresh ID for the chosen physical machine rather than using an old screenshot.

There is no idempotency key on the EC2 endpoint. Write the unique-label ledger before the single POST. An ambiguous response is **not** evidence that no instance was created. Reconcile fresh provider inventory by exact label/instance ID; never retry automatically. The current `create_workers()` call also reconciles its existing pending inventory, so it is a mutating operation even before it submits its new request. The cached snapshot contained pending ID 44143517 and provider ID 43032075, both absent from the initial live account inventory; this stale-state caveat was reported before creation.

## Template selection and isolation

Current EC2 defaults contain only `template_hash_id=4dc0868ee56886c667c217d6c3634a58`, template ID **408714**, owner 356317, named `PyTorch (Vast) - Musetalk w/ NVENC enabled & 12.1 CUDA`. There is no hardcoded image/onstart override in that default JSON.

The browser-requested template is a **different object**: ID **756005**, hash `210968d1e76b34e7cf34e505246e5d47`, owner 356317, exact name `-(NEEDS UPDATE )PyTorch (Vast) - Musetalk w/ NVENC enabled & 12.1 CUDA`. Both currently use `vastai/pytorch:cuda-12.1-auto`. Updating template 756005 alone will not change EC2's reference to 408714.

Top-level `create_request` overrides, including `onstart`, pass through. Environment objects merge, but six injected `LINGUA_*` control-plane fields are protected and cannot be cleared with an env override. The registry has only healthy/draining/dead, no experiment pool, and MuseTalk heartbeats can overwrite a draining state (only SoulX currently preserves it). A label or one-time drain is not safe isolation.

The pinned development revision `dae1e88ad3587fddbebe6d41f85faa569001f6c1` adds explicit `LINGUA_CONTROL_PLANE_ENABLED=0` support. The prepared initial-development onstart checks out that exact SHA on `codex/rtx3090-r5-delivery`, sets this flag and `LINGUA_WORKER_CALLBACK_REQUIRED=0`, and retains the original clean/full-stack source installation to measure its real baseline. These flags are exported before canonical secret bootstrap; the live runtime secret was verified to contain **no LINGUA keys**, so it cannot override them. This is a development boot, not the final image startup contract.

Protected raw rollback and unexecuted payload files live outside Git at `/home/ec2-user/.local/state/musetalk-r5-20261008-preflight-v4`, directory 0700 and files 0600. They contain existing bootstrap credentials. Do not print or commit them. Files: `template-408714.before.json`, `template-756005.before.json`, `development-create.json`, `development-onstart.sh`, `offer.json`, `preparation-summary.json`. Baseline onstart SHA-256 is `14ccb8011f7b5b2cb10e77551ea32826fd729f24eef2570cea9493a086eb8c39`; prepared onstart is `a8abbc3d6a6c14d8d74c9ad37b6ef3b2462fce339410b3c10ee2acc609b5a246`.

## Credentials, artifact access and builder limits

The EC2 instance role can load backend Secrets Manager values, read Vast inventory, and HEAD `lingua-musetalk-s3-storage`. It cannot list Secrets Manager secrets, read the worker runtime secret directly, or describe ECR repositories. No Docker/Podman/BuildKit/nerdctl binary, daemon socket, or root/ec2-user registry config was found. EC2 has only 2 vCPUs, 3.8 GiB RAM and approximately 4.2 GiB free disk; do not use it for a heavy Docker build beside production without a separately safe builder plan.

The existing template's intended bootstrap credential successfully reads `arn:aws:secretsmanager:us-east-1:211125449207:secret:lingua/musetalk-worker-runtime-Dof4b8`. Its runtime credential successfully HEADs the avatar bucket. Runtime secret keys are only `AVATAR_S3_BUCKET`, `AVATAR_S3_ENABLED`, `AVATAR_S3_PREFIX`, `AVATAR_S3_REGION`, `AWS_ACCESS_KEY_ID`, `AWS_DEFAULT_REGION`, `AWS_SECRET_ACCESS_KEY`. Control-plane configuration comes from EC2's injected fields, not that shared secret. No AWS worker credentials are currently injected by the EC2 default create path; the source template supplies its separate bootstrap path.

## Observer usage and evidence

`scripts/repro_3090/50_startup.py` is stdlib-only and supports `init`, explicit paid `create`, `reconcile`, `observe`, and `report`. It writes an atomic 0600 startup ledger and a separate shared budget reservation book. A submitting/ambiguous ledger cannot be submitted again. Reservations are not automatically released after failed/ambiguous creates; reconcile actual provider billing/resources before adjusting the shared budget book.

Run `python3 -m unittest -v test_startup_observer.py` for CPU-only safety tests. No test makes a real cloud request. On EC2, `ec2_startup_runner.py observer <subcommand arguments>` loads credentials internally and delegates to the observer beside it. Example protected paths (not an instruction to rent without current approval/enforcement):

```bash
sudo /home/ec2-user/lingua/venv/bin/python /path/to/private-tools/ec2_startup_runner.py observer create \
  --run-dir /path/to/private-run \
  --request-json /path/to/protected/development-create.json \
  --offer-json /path/to/protected/offer.json \
  --budget-json /path/to/protected/budget.json \
  --budget-ledger /path/to/protected/shared-budget.json
```

The budget JSON requires `approval_reference`, `lifetime_enforcer_reference`, `total_usd`, `max_hourly_usd`, `max_lifetime_hours`, `planned_lifetime_hours`, `max_inet_down_usd_per_gb`, `max_inet_up_usd_per_gb`, `reserved_inet_down_gb`, `reserved_inet_up_gb`, and `additional_reserve_usd`. The offer document requires `observed_at_utc`, `allocated_storage_gb`, and the actual `offer`; stale (>5 minute), wrong-GPU, missing, nonfinite or over-cap prices fail closed. Storage is conservatively reserved again even when already included in hourly price.

The dedicated expiry invocation is `ec2_startup_runner.py expire --run-dir <owned run> --not-before-utc <deadline>`. It refuses protected instance 51074906, wrong labels/IDs/GPUs, duplicate matches and relabelled instances. It uses the existing EC2 destroy path with `force:false`, verifies provider absence, and returns 3 while draining/deletion remains pending. A timer must retry those incomplete outcomes. It never stops or restarts an existing production service and installs no timer itself.

`observe` polls the exact instance's registry and assignable pool row. Worker stages are imported from an explicit JSON document `{instance_id,events:[{name,utc,evidence_ref,clock_uncertainty_seconds}]}`. Only allowed worker-stage names are accepted. Foreign monotonic clocks are never imported or subtracted. Missing image-pull/GPU-warm timestamps remain unavailable; parallel intervals are not added together.

The readiness media document must use schema `musetalk_startup_media_v1`, exact `instance_id`, `network_path=ec2_routed_webrtc`, `client_kind=real_browser`, actual audio SHA-256, first-frame and validation UTC timestamps, at least 10 seconds of measurement, per-stream decoded/fresh/content/cadence P1–P3 measurements, codec, and verified raw artifact paths/hashes. Audio response, avatar, backend, and absence of deferred build/download must be explicitly verified by the live test producer. Missing media evidence cannot pass. First usable frame and completed validation are reported separately. These CPU-tested adapters still require real browser/worker producers and deployment validation; they are not themselves a claim of fast startup.
