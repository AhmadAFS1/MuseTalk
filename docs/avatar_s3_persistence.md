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
