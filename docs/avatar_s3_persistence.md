# Avatar S3 Persistence

Prepared avatar materials can be persisted to S3 so new workers do not need to
re-run `/avatars/prepare` for avatars that already exist.

## Enable it

Set these environment variables before starting `api_server.py` or the Vast
startup wrapper:

```bash
export AVATAR_S3_ENABLED=1
export AVATAR_S3_BUCKET=lingua-musetalk-s3-storage
export AVATAR_S3_PREFIX=avatars
export AVATAR_S3_REGION=us-east-1
```

`AVATAR_S3_REGION` is optional when `AWS_REGION` is already set. Credentials can
come from the instance role, the default AWS credential chain, or
`AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` / `AWS_SESSION_TOKEN`.
On Vast workers, prefer loading those credentials from the separate MuseTalk
runtime secret described in `docs/musetalk_worker_secrets.md`.

The app stores objects under:

```text
s3://lingua-musetalk-s3-storage/avatars/<musetalk-version>/<avatar_id>.tar.gz
```

For the current server defaults, that is usually:

```text
s3://lingua-musetalk-s3-storage/avatars/v15/<avatar_id>.tar.gz
```

## Required IAM permissions

The runtime path only needs object reads and writes for the configured prefix:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject"],
      "Resource": "arn:aws:s3:::lingua-musetalk-s3-storage/avatars/*"
    }
  ]
}
```

`s3:ListBucket` is useful for manual debugging or console workflows, but the
application does not require it for prepare or restore.
If `MUSETALK_SECRETS_VERIFY_S3=1` is enabled during Vast bootstrap, the runtime
credential also needs bucket-level `s3:ListBucket` or `s3:GetBucketLocation` so
the bootstrap `head_bucket` verification can pass.

For the production avatar bucket, the runtime S3 policy used by the MuseTalk
worker is:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "BucketLevel",
      "Effect": "Allow",
      "Action": [
        "s3:GetBucketLocation",
        "s3:ListBucket",
        "s3:ListBucketMultipartUploads"
      ],
      "Resource": [
        "arn:aws:s3:::lingua-musetalk-s3-storage"
      ]
    },
    {
      "Sid": "AvatarObjectLevel",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:AbortMultipartUpload",
        "s3:ListMultipartUploadParts"
      ],
      "Resource": [
        "arn:aws:s3:::lingua-musetalk-s3-storage/avatars/*"
      ]
    }
  ]
}
```

## AWS Secrets Manager setup for Vast

Vast workers are outside AWS, so this setup intentionally uses two different
IAM users and two different access keys:

- `musetalk-secret-reader`: the bootstrap IAM user. Its key lives in the Vast
  environment and can only read the MuseTalk runtime secret from Secrets
  Manager.
- `musetalk-s3-runtime`: the runtime S3 IAM user. Its key is stored inside the
  Secrets Manager JSON payload and can read/write avatar tarballs in S3.

Do not put the `musetalk-s3-runtime` key directly in the Vast template. Do not
attach S3 permissions to `musetalk-secret-reader`.

### 1. Create or verify the S3 runtime user

In IAM, create a user named:

```text
musetalk-s3-runtime
```

Do not enable AWS Console access. Attach the avatar S3 policy above, then create
an access key for "Application running outside AWS". Save the access key ID and
secret access key temporarily; they go into Secrets Manager in the next step.

### 2. Create the runtime secret

In AWS Secrets Manager, use the `us-east-1` region and create a secret named:

```text
lingua/musetalk-worker-runtime
```

Use "Other type of secret" and store this JSON. Replace only the two S3 runtime
credential placeholders:

```json
{
  "AWS_ACCESS_KEY_ID": "<musetalk-s3-runtime-access-key-id>",
  "AWS_SECRET_ACCESS_KEY": "<musetalk-s3-runtime-secret-access-key>",
  "AWS_DEFAULT_REGION": "us-east-1",

  "AVATAR_S3_ENABLED": "1",
  "AVATAR_S3_BUCKET": "lingua-musetalk-s3-storage",
  "AVATAR_S3_PREFIX": "avatars",
  "AVATAR_S3_REGION": "us-east-1"
}
```

After the secret is created, copy the exact Secret ARN from the secret details
page. The ARN has a real random suffix from AWS. Example suffixes such as
`xxxxxx` or `AbCdEf` are placeholders and will cause `ResourceNotFoundException`.

You may also use the secret name as `MUSETALK_AWS_SECRET_ID`:

```bash
export MUSETALK_AWS_SECRET_ID="lingua/musetalk-worker-runtime"
```

### 3. Create the bootstrap secret-reader user

Create a second IAM user named:

```text
musetalk-secret-reader
```

Do not enable AWS Console access. Attach a Secrets Manager read policy like:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ReadMusetalkRuntimeSecret",
      "Effect": "Allow",
      "Action": [
        "secretsmanager:GetSecretValue",
        "secretsmanager:DescribeSecret"
      ],
      "Resource": "arn:aws:secretsmanager:us-east-1:211125449207:secret:lingua/musetalk-worker-runtime*"
    }
  ]
}
```

Create an access key for this user. This key goes in the Vast template or shell
environment. This user does not need `secretsmanager:ListSecrets`; an
`AccessDeniedException` for `ListSecrets` during debugging is expected and does
not mean bootstrap is broken.

### 4. Vast/on-start environment

Set these before `scripts/vast_onstart.sh` starts the server:

```bash
export MUSETALK_AWS_SECRET_ID="arn:aws:secretsmanager:us-east-1:211125449207:secret:lingua/musetalk-worker-runtime-REALSUFFIX"
export MUSETALK_AWS_SECRET_REGION="us-east-1"
export MUSETALK_SECRETS_STRICT="true"
export MUSETALK_SECRETS_VERIFY_S3="1"

export AWS_ACCESS_KEY_ID="<musetalk-secret-reader-access-key-id>"
export AWS_SECRET_ACCESS_KEY="<musetalk-secret-reader-secret-access-key>"
export AWS_DEFAULT_REGION="us-east-1"
```

The `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` above must be from
`musetalk-secret-reader`, not from `musetalk-s3-runtime`.

In a full Vast bootstrap script, place those exports after the repo is published
and before this call:

```bash
SETUP_CLEAN=1 \
SETUP_FULL_STACK=1 \
STARTUP_TIMEOUT_SECONDS=1800 \
PROFILE=throughput_record \
PORT=8000 \
bash scripts/vast_onstart.sh
```

For first validation on smaller GPUs, prefer the conservative server start shown
below instead of `PROFILE=throughput_record`, because the throughput profile can
warm TensorRT batch 16 and fail on 16 GB cards.

## Manual validation without reinstalling

When the venv already exists, test Secrets Manager and S3 without rebuilding the
server environment.

First verify the bootstrap key identity:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python - <<'PY'
import boto3
print(boto3.client("sts", region_name="us-east-1").get_caller_identity())
PY
```

Expected `Arn`:

```text
arn:aws:iam::211125449207:user/musetalk-secret-reader
```

Then verify `GetSecretValue` works:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python - <<'PY'
import os
import boto3

secret_id = os.environ["MUSETALK_AWS_SECRET_ID"]
region = os.environ.get("MUSETALK_AWS_SECRET_REGION", "us-east-1")

print("Trying secret:", secret_id)
print("Region:", region)

client = boto3.client("secretsmanager", region_name=region)
response = client.get_secret_value(SecretId=secret_id)

print("Secret found and readable")
print("ARN:", response["ARN"])
print("Name:", response["Name"])
PY
```

Then run the MuseTalk bootstrap script and source the returned runtime S3 env:

```bash
cd /workspace/MuseTalk

TMP_ENV="$(mktemp /tmp/musetalk-runtime.XXXXXX.env)"

if /workspace/.venvs/musetalk_trt_stagewise/bin/python \
  scripts/bootstrap_aws_secrets.py \
  --output "$TMP_ENV" \
  --verify-s3; then
  source "$TMP_ENV"
  rm -f "$TMP_ENV"
  echo "Secret loaded successfully"
else
  rm -f "$TMP_ENV"
  echo "Secret bootstrap failed"
fi
```

Success output includes:

```text
Verified S3 bucket access: s3://lingua-musetalk-s3-storage
Secret loaded successfully
```

Start the existing server without reinstalling:

```bash
cd /workspace/MuseTalk

REPO_ROOT=/workspace/MuseTalk \
VENV_PATH=/workspace/.venvs/musetalk_trt_stagewise \
PROFILE=baseline \
PORT=8000 \
STARTUP_TIMEOUT_SECONDS=1800 \
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=4 \
HLS_SCHEDULER_MAX_BATCH=4 \
HLS_SCHEDULER_FIXED_BATCH_SIZES=4 \
bash scripts/vast_server_ctl.sh start
```

Check readiness:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/stats
```

`/stats` should show:

```json
{
  "avatar_s3": {
    "enabled": true,
    "bucket": "lingua-musetalk-s3-storage"
  }
}
```

## Runtime behavior

- `/avatars/prepare` writes the avatar locally first, validates the required
  prepared materials, then uploads one tarball to S3.
- Inference paths call `_avatar_exists()`. If the avatar is missing locally and
  S3 is enabled, the worker downloads and restores it lazily by `avatar_id`.
- Restores are staged in a temporary directory. The existing local avatar is
  kept until the archive has downloaded, extracted, and passed validation.
- Archives may only contain regular files and directories under the expected
  avatar root. Absolute paths, parent traversal, symlinks, hardlinks, and device
  files are rejected.
- Failed restore attempts are retried after `AVATAR_S3_RESTORE_RETRY_SECONDS`
  to avoid hammering S3 for missing avatars while still allowing recovery from
  transient network issues.
- Calling `/avatars/prepare` without `force_recreate=true` also runs the normal
  existence check first. If the local avatar folder was deleted but the S3
  archive exists, the worker restores from S3 and treats the avatar as already
  prepared.
- Calling `/avatars/prepare?force_recreate=true` intentionally rebuilds the
  avatar after the existence check. Use it when you want to replace the S3
  archive, not when testing restore.

## Retry and timeout controls

Defaults are conservative for startup and autoscaled workers:

```bash
export AVATAR_S3_RETRY_ATTEMPTS=3
export AVATAR_S3_RETRY_BASE_DELAY_SECONDS=0.5
export AVATAR_S3_RETRY_MAX_DELAY_SECONDS=4.0
export AVATAR_S3_RETRY_MODE=standard
export AVATAR_S3_CONNECT_TIMEOUT_SECONDS=5
export AVATAR_S3_READ_TIMEOUT_SECONDS=120
export AVATAR_S3_RESTORE_RETRY_SECONDS=60
```

`boto3` also receives the retry mode and attempt count, and the avatar store
wraps whole-object upload/download calls with an additional retry loop so a
failed transfer does not leave a partial avatar visible on disk.

## Observability

`GET /stats` includes an `avatar_s3` section with counters for upload/download
attempts, successes, failures, retries, restore replacements, byte totals, last
object keys, elapsed seconds, and the latest error.

Server logs use the `[avatar_s3]` prefix. Useful lines to look for:

```text
[avatar_s3] upload success avatar_id=...
[avatar_s3] restore success avatar_id=...
[avatar_s3] restore miss avatar_id=...
[avatar_s3] upload retry avatar_id=...
```

## Smoke test

1. Start the server with S3 enabled.
2. Prepare an avatar with `POST /avatars/prepare`.
3. Confirm the tarball exists in
   `s3://lingua-musetalk-s3-storage/avatars/v15/<avatar_id>.tar.gz`.
4. Remove the local avatar directory or start a fresh worker.
5. Call a generation, session, HLS, or WebRTC endpoint with the same
   `avatar_id`; the worker should restore from S3 before loading the avatar.
6. Check `/stats` and confirm `avatar_s3.download_successes` incremented.

Example prepare request:

```bash
curl -X POST "http://127.0.0.1:8000/avatars/prepare?avatar_id=test_s3_avatar&batch_size=4&bbox_shift=5&force_recreate=true" \
  -F "video_file=@/workspace/MuseTalk/data/video/sun.mp4"
```

After prepare, confirm this object exists in S3:

```text
s3://lingua-musetalk-s3-storage/avatars/v15/test_s3_avatar.tar.gz
```

Restore test through prepare:

```bash
rm -rf /workspace/MuseTalk/results/v15/avatars/test_s3_avatar

curl -X POST "http://127.0.0.1:8000/avatars/prepare?avatar_id=test_s3_avatar&batch_size=4&bbox_shift=5" \
  -F "video_file=@/workspace/MuseTalk/data/video/sun.mp4"
```

Do not include `force_recreate=true` in the restore test.

Quick generation test:

```bash
curl -N -X POST "http://127.0.0.1:8000/generate/stream?avatar_id=test_s3_avatar&batch_size=2&fps=15&chunk_duration=2" \
  -F "audio_file=@/workspace/MuseTalk/data/audio/sun.wav"
```

The SSE response should include a chunk URL like:

```text
/chunks/test_s3_avatar_req_<id>/chunk_0000.mp4
```

Download that chunk to verify generation output:

```bash
curl -o /tmp/chunk_0000.mp4 \
  "http://127.0.0.1:8000/chunks/test_s3_avatar_req_<id>/chunk_0000.mp4"
```

Useful log checks:

```bash
tail -f /workspace/logs/musetalk/api_server_8000.log
```

Look for:

```text
[avatar_s3] upload success avatar_id=test_s3_avatar
[avatar_s3] restore success avatar_id=test_s3_avatar
```

## Troubleshooting notes from the Vast setup

`ResourceNotFoundException` from `GetSecretValue` means
`MUSETALK_AWS_SECRET_ID` does not identify an existing secret in the configured
region and AWS account. Check for placeholder suffixes such as `xxxxxx` or
`AbCdEf`, copy the exact Secret ARN from Secrets Manager, or use the secret name
`lingua/musetalk-worker-runtime`.

`AccessDeniedException` for `secretsmanager:ListSecrets` is expected for the
minimal `musetalk-secret-reader` user. Bootstrap does not need `ListSecrets`; it
needs only `GetSecretValue` and `DescribeSecret`.

`AccessDeniedException` for `GetSecretValue` means the secret-reader policy
does not match the real secret ARN. Use a resource pattern like:

```json
"arn:aws:secretsmanager:us-east-1:211125449207:secret:lingua/musetalk-worker-runtime*"
```

`S3 verification requires AWS_ACCESS_KEY_ID...` means the Secrets Manager JSON
does not contain the runtime S3 access key fields. The bootstrap script does not
fall back to the secret-reader key for S3 verification.

`AccessDenied`, `InvalidAccessKeyId`, or `SignatureDoesNotMatch` during S3
verification means the runtime S3 key stored inside the secret is wrong,
inactive, or missing the avatar bucket policy.

`curl: (7) Failed to connect to 127.0.0.1 port 8000` means nothing is listening
on port 8000 yet. In the validated Vast setup, this happened because server
startup was still warming TensorRT or had crashed before binding the socket.
Check:

```bash
bash scripts/vast_server_ctl.sh status
tail -n 160 /workspace/logs/musetalk/api_server_8000.log
```

On 16 GB GPUs, avoid `PROFILE=throughput_record` for first S3 validation. It can
warm TensorRT batches `[8, 16]` and fail during startup with GPU out-of-memory.
Use `PROFILE=baseline`, `MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=4`, and
`HLS_SCHEDULER_MAX_BATCH=4` first.
