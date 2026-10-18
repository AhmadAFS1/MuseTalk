# Non-Docker Vast startup: deployment plan and replacement bootstrap

Prepared 2026-10-16. **Local candidate, not deployed.** No live template,
autoscaler, IAM policy, GPU rental or GitHub branch was changed by this work.

## The answer: change both layers, with a small EC2 launch integration

| Layer | Responsibility | Change |
|---|---|---|
| Vast On-start box | Get the reviewed code; set launch options; call the repo | Replace the pasted script with [vast_bootstrap_locked.sh](../scripts/vast_bootstrap_locked.sh) after publication/integration gates below |
| MuseTalk repo | Install/verify/reuse dependencies; secrets; TURN; r5 engines; API readiness | Publish the tested locked installer/state-checker/helper files; keep canonical `vast_onstart.sh`; add ticket-pipe closure after setup |
| Existing EC2 launcher | Choose revision/offer and supply private launch inputs | Generate short-lived dependency GET tickets with the existing role and inject them at instance creation; preserve the working runtime/control-plane configuration flow |

Changing only `SETUP_CLEAN` does not implement the faster dependency installer.
Changing only the repo leaves the template's checkout deletion and clean-install
behavior in place. EC2's instance role is **not automatically available on Vast**.

## Selected approach and evidence

Use the stock `vastai/pytorch:cuda-12.1-auto` image with the tested Linux x86_64,
Python3.10, cu121 contract: exact verified wheels, four downloads, offline pinned
uv, compiled bytecode, copy mode, full Kokoro/avatar-prep capabilities retained.
Large packages still come from their recorded original CDNs; only the trusted
manifest and four small built wheels need private S3 tickets. This is **not** an
S3 venv restore, model-weight migration, Docker revival or an engine rebuild.

Matched first-host dependency validation median improved 294.2→132.2s. The
different-host package phase was 511.1s and total fresh boot 24m17s. Therefore no
five-minute total promise is justified. Keep [the full report](DEPENDENCY_INSTALL_OPTIMIZATION_REPORT_20261016.md)
and its failed/conditional results as the governing evidence.

Do not adopt no-compile uv, hardlinks or boot-download overlap from the current
screens. Preserve r5 inference/model/latent settings and prebuilt-engine checks.

## Changes to your pasted script

1. Remove both hardcoded AWS credential exports. The key pasted in chat must be
   rotated/deactivated through the existing credential-management process; this
   patch neither rotates it nor copies its value. One secret reference, if used,
   is sufficient. A Secrets Manager ARN is not itself an AWS credential.
2. Remove `rm -rf "$REPO_DIR"`. Keep code, models, engines and runtime files on
   the same worker rather than discarding them on every restart.
3. Set **SETUP_CLEAN=0**, not1. A brand-new dependency path is installed as needed;
   a valid accepted environment is checked/reused; foreign or incompatible envs
   are refused, not deleted or silently adopted.
4. Select the explicit locked-uv mode, trusted contract/cache and four workers.
   Use a new artifact-specific venv outside the checkout. Leave the old source-pip
   venv recoverable for rollback; it cannot be retroactively labeled locked.
5. Deploy a **full Git commit SHA**, not an uncontrolled moving main checkout.
   New code on main is used after EC2's deployment revision is deliberately
   advanced. Source-only changes can retain the same dependency artifact/env;
   changed dependency fingerprints require a newly reviewed/published contract.
6. Keep `SETUP_FULL_STACK=1`, API8000, throughput profile, timeout1800, TURN and
   canonical r5 startup. The replacement does not remove or remap any ports.
   Existing port mappings stay in the template/EC2 create request, not this file.

The replacement checks origin, ownership and tracked edits before updating an
existing checkout, refuses a code update while an API PID file exists, and never
overwrites ignored files deliberately. Untracked collisions are left for Git
to reject. On unchanged pinned code it skips clone/fetch. On first boot it still
checks out the full repository: **sparse checkout is not yet runtime-validated**
and must be a separately measured candidate, not silently enabled here.

## Authentication: two distinct flows

### Dependency delivery, before boto3 is installed

The read-only EC2 helper [issue_dependency_read_tickets.py](../scripts/issue_dependency_read_tickets.py)
uses the existing `linguaEc2Role`, verifies the manifest against the independently
pinned digest, checks the four small wheel objects and issues only presigned
GETs for those exact objects. It does not upload, modify IAM, issue EC2 role
credentials to workers or create instances.

EC2 must capture the returned JSON **in memory** and add it to that instance's
launch environment as `MUSETALK_DEPENDENCY_TICKETS_JSON`. Default ticket TTL is
3600s; supported range300–3600s. Issue it just before creation, not while saving
the reusable template. Do not log the JSON, print the create request, include it
in shell argv, commit it, or store a static/expired ticket in the template.

The bootstrap moves it into a pipe on FD8 and removes the environment variable.
The canonical repo onstart closes that owned descriptor immediately after setup,
including on reuse, before API/TURN launch. No dependency-ticket temp file is
created. A valid cached env may restart without a ticket; a fresh install may not.
Expired/invalid tickets fail closed, not through silent source-pip fallback.

An EC2 Python integration can call `issue(contract, s3_client, sts_identity)` and
JSON-serialize the result directly into the private launch request. The CLI
stdout is a **secret-bearing transport**, not a command to run visibly in chat
or a terminal. The helper is prepared locally; the production caller has not
been edited, inspected for compatibility, or deployed in this step.

### Runtime secrets, after dependencies are available

The replacement supports either existing contract, selected at launch:

- `MUSETALK_RUNTIME_CONFIG_SOURCE=secrets-manager`: EC2 supplies a narrowly scoped
  **worker** AWS read identity and `MUSETALK_AWS_SECRET_ID`/region through launch
  inputs. Existing `bootstrap_aws_secrets.py` fetches runtime config after setup,
  with strict/S3 verification retained. That identity must actually have the
  required GetSecretValue and applicable S3/KMS permissions; the script cannot
  invent them from an ARN. Keep EC2's broader publisher role on EC2.
- `MUSETALK_RUNTIME_CONFIG_SOURCE=injected`: EC2 reads the runtime secret using
  its role and supplies the approved worker runtime payload, including worker
  S3 read identity, bucket/region settings and the existing control-plane fields.
  The worker skips a second Secrets Manager fetch. The bootstrap requires avatar
  and engine bucket settings, but those presence checks alone do not prove S3
  authorization, worker registration or call readiness.

Choose the mode that matches the actual existing EC2 launcher during integration;
do not change IAM or ship EC2 role credentials as a shortcut. Runtime credential
lifetime/renewal must cover worker lifetime and later avatar requests. Dependency
ticket expiry does not provide runtime AWS refresh. Provider/host administrators
can access a worker's launch environment; short-lived, minimal permissions reduce
exposure but do not turn an untrusted GPU host into a secret vault.

## What belongs in Vast versus EC2

Keep the template image, existing valid port mappings and startup script private.
After the new code is reviewed/published, paste the **file contents** of
`scripts/vast_bootstrap_locked.sh` into the On-start Script box. It must be the
bootstrap contents, not `bash scripts/vast_bootstrap_locked.sh`: the repo does
not exist when a fresh instance first starts.

EC2 supplies these per-launch fields; the reusable template cannot populate
them dynamically from AWS on its own:

| Field | Value/source |
|---|---|
| `MUSETALK_SOURCE_REVISION` | Full40-character SHA of the reviewed commit containing these files, after publication |
| `MUSETALK_DEPENDENCY_TICKETS_JSON` | Fresh helper output captured privately in memory |
| `MUSETALK_RUNTIME_CONFIG_SOURCE` | `secrets-manager` or `injected`, matching the existing runtime flow |
| Runtime AWS/config/callback values | Existing approved EC2-to-worker flow, without hardcoded credentials in the script |

No new SHA can be supplied yet: these local changes are uncommitted/unpushed.
The old baseline SHA lacks the locked bootstrap/installer and is not an eligible
replacement merely because it is known. A manual UI launch also needs an
approved one-off private injection path; this is not a self-contained static
template until that integration is present.

## Ordered rollout gates

1. Review this patch and exact production file set. Do not bulk-commit private
   raw logs, ticket JSON, credentials, experiment ledgers or unrelated user edits.
2. Validate the replacement with source/mock tests. No macOS pip/CUDA tests.
3. After user authorization, publish the reviewed repo change and record its SHA.
   Do not change the default install mode repo-wide: template opts in explicitly.
4. Inspect/update the existing EC2 create request to capture read tickets and
   pass that SHA/runtime inputs. No new AWS compute, registry, always-on cache or
   IAM changes are inherently required; missing access is a separate approval.
5. After test authorization, launch one bounded stock-template3090 canary.
   Measure create/container/checkout/dependency/engine/API/first-call boundaries,
   verify exact inventory and no engine rebuild, exercise avatar prep/TTS as
   enabled, direct WebRTC/TURN and production registration on an isolated route.
6. Restart that same worker with unchanged code/contracts; verify no repo wipe,
   no dependency redownload/reinstall, retained model/engine caches and correct
   first usable output. Update source-only code and confirm env reuse. Test a
   changed contract with a **new** venv path, not destructive conversion.
7. Verify expired-ticket refusal/rollback with the real launcher. Keep the old
   template and source-pip env available; route no production work until ready.
8. Only then update the production template/autoscaler selection. Fleet rollout,
   skinny checkout, package mirrors and any capability removal remain separate.

## Progress for this change

- Prepared replacement bootstrap, read-only EC2 ticket issuer and owned-ticket
  FD closure in canonical onstart. Existing source-pip default is preserved.
- No secret value copied into source/docs. No cloud mutation/rental or Git push.
- New onstart hook closes the owned dependency pipe after setup; the outer
  bootstrap also prevents API/TURN daemons from inheriting its process lock.
- **11 new source/mock tests pass** for owner-bound GET-only issuance, manifest
  tampering, wrong role/account/expiry, missing wheel, private pipe handling,
  auth prerequisites, warm checkout reuse, active-PID/dirty-edit protection and
  descriptor closure. Entire combined suite: **146 passing tests**. Shell syntax
  and `git diff --check` pass. No pip/CUDA install test ran on macOS.
- Linux/Vast execution of this **new combined replacement** and production EC2
  wiring remain unperformed; prior installer canaries do not validate these later
  bootstrap/auth/descriptor edits. Follow the rollout gates before deployment.
