# EC2-routed browser-call prerequisites and isolation

Read-only discovery on 2026-10-08. No accounts, tokens, routes, registrations,
services, cloud resources, IAM policies, or production configuration were changed.
No browser call was initiated. This is an implementation proposal, not acceptance
evidence.

## Verified source and runtime

- EC2 `/home/ec2-user/lingua` is clean at
  `171e169682b08054fa7505df72431914d6653c8e`.
- Local `/Users/ahmadsmacair/code/lingua/lingua` is at
  `734db0b7b4fe7d7650a93a23a00af31f8b2d97ff`; its live-session authorization is
  newer. Do not treat local HEAD as the deployed contract.
- Read-only service-environment + approved `bootstrap_secrets()` inspection:
  `PUBLIC_API_BASE_URL=https://ddt5k5hjpomxw.cloudfront.net`,
  `RUNTIME_STATE_BACKEND=dynamodb`, table `LinguaRuntimeState`,
  `LIVE_SESSION_MAX_SECONDS=900`, `ENVIRONMENT=development`,
  `ACCOUNT_CONTROLS_ENABLED=false`.
- The protected existing worker and production `lingua-api` service remain
  untouched. Experimental worker `54798270` is not to be registered globally.

Source paths below refer to the EC2 revision unless explicitly called local.
Use `git show 171e169682b08054fa7505df72431914d6653c8e:<path>` from the local Lingua
checkout to reproduce the deployed source inspection.

## Actual authentication and call contract

1. Cognito authentication is real JWT verification, not the admin token.
   `backend/services/auth_service.py` extracts `Authorization: Bearer ...`;
   `backend/services/cognito_service.py` verifies signature, expiration, issuer,
   token use, and app client. Both Cognito ID and access tokens are supported.
   Runtime pool `us-east-1_4Z2F6wLCt`, client `49jmu2h373qifmklmb2qot5t27`,
   region `us-east-1` were confirmed. These identifiers are not credentials.
2. Existing login is `POST /api/auth/login` with email/password; refresh is
   `POST /api/auth/refresh` with refresh token. No login or refresh was performed.
3. `POST /api/live/sessions` requires the bearer token and JSON
   `avatar_id`, `character_id`, and optional `config` containing `playback_fps`,
   `musetalk_fps`, `batch_size`, `segment_duration`.
4. **Deployed ownership rule is strict:** `_get_owned_character` requires an
   existing character whose `user_id` equals the verified JWT subject, and whose
   `avatar_id` exactly equals the request. Local HEAD's public-catalog interaction
   support is not deployed. A published S3 cache alone is not a callable app
   character and must not be used to bypass this check.
5. Session admission reserves at most two simultaneous calls per user; create
   quota is 10/minute and 100/day. Successful creation records worker route,
   owner, session capability, and avatar affinity. Response includes `session_id`,
   `player_url`, `player_token`, `upstream_base_url`, `stream_upload_url`, and
   `status_url`. These token-bearing responses must not enter public artifacts.
6. Player `GET /api/live/sessions/<id>/player` needs `X-Live-Token`. Lingua fetches
   `/webrtc/player/<id>` from the selected worker, rewrites signaling to Lingua,
   and injects the scoped capability into only that session's fetch calls.
   Normal browser navigation cannot attach this initial header by itself;
   the real-browser harness must supply it without logging the value.
7. Signaling `POST /api/live/upstream/webrtc/sessions/<id>/{offer,ice}` uses the
   same `X-Live-Token`. Only offer and ICE are proxied. Negotiated audio/video
   travels browser↔worker/TURN, **not through EC2**.
8. Audio `POST /api/live/sessions/<id>/stream` requires the original bearer token,
   session ownership, and multipart `audio_file`; optional pose metadata is
   validated before forwarding. Status GET, pose/events POST, and DELETE are
   also owner-authorized. DELETE revokes capability and cleans the route.

Evidence: `backend/routes/live_sessions.py`, `backend/services/live_authorization.py`,
`backend/services/auth_service.py`, `backend/services/cognito_service.py`.

## Identity availability: unresolved, no credentials extracted

No `TEST_*`/Cognito test-username/password configuration keys were present in the
approved loaded service environment. Repository documentation
`docs/ec2-autoscaling-validation-results-2026-06-09.md` explicitly recorded that
safe confirmed test-user credentials were unavailable at that historical run;
its synthetic token tests are not real-login or browser-call evidence.

The existing app stores identity through `app/auth/AuthContext.tsx` and
`app/utils/authStorage.ts`: key names `auth_token`, `auth_access_token`,
`auth_refresh_token`, plus `user_id`. Native tokens use Expo SecureStore;
web fallback uses AsyncStorage. Only source code was inspected; no app storage,
browser credential store, personal account list, or token value was read.

Before an actual routed call, the operator must identify an authorized existing
signed-in test account and its owned ready character, or explicitly authorize an
appropriate test-account/character setup. Admin access does not supply a Cognito
identity, and no fake user or synthetic JWT should be used for acceptance.

## Why existing production routing is not test isolation

`backend/services/musetalk_router.py` merges all dynamic MuseTalk workers with
the global static pool. Healthy dynamic workers are prioritized. There is no
experiment-pool predicate or request-level forced worker parameter.

`prefer_base()` can add a healthy preferred base even if it is not registered.
The preference comes from the global avatar warm map, not authenticated test
context. Changing an avatar's affinity can therefore redirect other calls using
that avatar. `create_session()` also writes the warm map after success.

The create loop checks healthy/status, not only `assignable`, so capacity alone
does not guarantee exclusion. MuseTalk drain is not sticky through ready
heartbeats. Registration followed by draining, capacity zero, global pool
replacement, or temporary warm-map edits are not safe isolation mechanisms.

## Smallest additive deployed-code proposal (requires review/deployment)

Prefer a separate administrator-authorized experiment entry point or explicit
request context, not a healthy globally registered test worker:

- Require **both** a valid app bearer identity and admin authorization.
- Bind one expiring experiment record to an allowed user subject, character,
  avatar, exact owned instance ID, and exact validated worker base. Do not accept
  arbitrary client URLs; prevent SSRF and identity mismatch.
- Derive exactly one candidate from that record. Do not enter normal ranking,
  prefer-base injection, or fallback-to-production on failure.
- Keep worker callback registration disabled and do not insert the experimental
  worker in the normal registry/static pool.
- Suppress global avatar-warm-map writes and production scale-pressure changes
  for the experiment. Retain owner/capability enforcement and bounded cleanup.
- Use a test-scoped route/capability namespace if deployed, or explicitly track
  unique experimental sessions and ensure normal requests cannot select them.
- Test absent/wrong admin, wrong user/character/avatar, expired run, disallowed
  URL, unknown instance, unhealthy worker, no fallback, no global warm-map write,
  unchanged normal ranking, and session capability/cleanup behavior.

Making drain sticky is desirable independently but is not sufficient test-pool
authorization. A permanent pool-tag design must also close the static-pool and
warm-affinity bypasses; an immutable experiment tag alone does not do that.

## Lower-impact staged EC2 option (not deployed-production acceptance)

A separate, loopback-only, single-process WSGI application on EC2 can reuse the
deployed Flask live-session blueprint and genuine Cognito/worker/TURN requests.
Expose only that staged application through a secure SSH port forward. It can
be configured with only the exact owned 3090 in its **private** static pool and
an allowlisted real user/character. This does not require modifying the existing
service or globally registering the worker.

Important isolation requirements:

- Do not call production `app.create_app()` blindly: it starts autoscaler and
  metrics background loops. A small staged entry point should register only the
  needed live-session routes (and optionally existing profile/refresh routes),
  with no background sweeper/autoscaler/queue worker. Do not expose account writes.
- Load required approved secrets in memory, then apply staged overrides **before
  Config and route imports**. `bootstrap_secrets()` can otherwise overwrite
  pre-set environment values from the configured secret.
- Set `RUNTIME_STATE_BACKEND` to a non-DynamoDB value and force Redis to a known
  unavailable loopback endpoint with `REDIS_REQUIRED=false`, so existing
  development-mode memory fallback is isolated. Alternatively implement a
  separate namespaced durable store. A new `LINGUA_REDIS_HASH_TAG` alone is
  insufficient: `live-capabilities:v1`, quota, lease, and paid-work namespaces are
  partly hard-coded and could still mutate production DynamoDB.
- Use one process because memory state is not cross-process. Real app character
  and Cognito reads can remain read-only. Preserve the actual authorization checks;
  no patched auth decorator, fabricated character lookup, mocked HTTP response,
  or fake token is acceptable for this staged integration result.
- Explicitly disable autoscaler, metrics, paid-work controls/shared state,
  account-control background work, and queue processing **only in this staged
  process**; record every difference from the deployed service.
- `_secure_player_url()` requires an HTTPS `PUBLIC_API_BASE_URL`; plain
  `http://localhost` fails the existing route even though browsers treat localhost
  as secure context. A separate TLS terminator/local test certificate behind the
  SSH forward is possible, but its trust override and omission of the real
  CloudFront edge must be disclosed. Do not alter the production proxy/CDN.
- Browser offer/ICE and audio uploads must still traverse this EC2 staged
  application; use actual worker ICE/TURN endpoints. A direct-worker-only call
  does not prove EC2 proxy integration.

Label resulting evidence **staged EC2 same-code browser integration**, not
deployed-production autoscaling/startup acceptance. Production acceptance remains
pending explicit deployment plus the same authenticated real-call measurement.

Existing CPU test patterns: `backend/tests/test_live_session_pose_protocol.py`,
`test_production_hardening.py`, `test_security_regressions.py`; they use Flask test
clients and mocked workers/auth. They are useful regression scaffolding, never a
replacement for the real browser/worker call.

## TURN and browser evidence requirements

MuseTalk `scripts/vast_onstart.sh` generates mode-0600 runtime TURN configuration.
`scripts/run_webrtc_relay_api_server.sh` uses UDP 3478/public mapped port with TCP
1455 fallback, generates public browser URLs, and can use loopback TURN for the
server. `api_server.py` constructs the browser ICE list including transient TURN
credentials and the worker `templates/webrtc_player.py` embeds it.

Instance A observed public TURN: `92.49.17.100:25609/udp` and
`92.49.17.100:25884/tcp`; API `92.49.17.100:25659`. These are this instance's
mapping, not reusable image constants. Never bake its TURN password/IP/ports into
the image. Browser negotiation must prove the selected candidate pair, received
audio/video, decoded frame progression, cadence/freshness, and stable streaming.
Redact TURN passwords/capabilities from saved HTML, SDP diagnostic payloads, and
network traces. Health, session allocation, or ICE connected alone is not success.

CloudFront origin/header forwarding for Authorization and X-Live-Token was not
live-verified in this read-only code pass and remains an edge prerequisite.

## Secret-free image: existing per-create injection works

`backend/services/autoscaler.py::_inject_worker_aws_env` first reads explicit
`VAST_MUSETALK_AWS_*` / `VAST_WORKER_AWS_*`. These key families, including the
passthrough flag, were absent in the approved loaded environment. Broad backend
AWS passthrough should remain off.

`_merge_create_request()` accepts caller-supplied `create_request.env` AWS fields;
only six LINGUA callback fields are protected. Thus the already supported
administrator create endpoint can supply scoped bootstrap AWS credentials for
one new instance without a production restart, backend env update, or IAM change.
Extract the already intended worker-bootstrap credential from the protected
template rollback in memory, place it only in the private request, and never
print/commit it. No create was performed in this discovery task.

The runtime secret reference is
`arn:aws:secretsmanager:us-east-1:211125449207:secret:lingua/musetalk-worker-runtime-Dof4b8`.
The existing template bootstrap credential was previously verified to retrieve
that secret; the EC2 instance role could not retrieve it. Do not substitute broad
backend credentials. The runtime secret supplies avatar S3 AWS/config fields,
not LINGUA control-plane configuration.

There is no generic runtime-config route to hot-set VAST AWS/create policy.
`secrets_loader.bootstrap_secrets()` is a one-time boot loader. Updating a secret
alone is not a verified hot reload, and a one-off create override does not change
future autonomous autoscaler launches. Long-term auto-launch bootstrap must be
kept in the selected Vast template outside the image, or explicitly deployed as
scoped backend launch configuration through a separately reviewed change.

Image requirements: no credential values, saved environment, TURN state,
capability tokens, AWS config/cache, or protected rollback files in build context
or image history. The image must implement a runtime secret fetch/export before
canonical startup, or the template's short onstart bootstrap must do that before
invoking the image entry point. Preserve isolation flags after secret import and
fail closed if the runtime secret unexpectedly attempts to override them.
