# MuseTalk Worker Secrets Bootstrap

MuseTalk Vast workers can load runtime environment variables from AWS Secrets
Manager before `api_server.py` starts. This mirrors the Lingua backend pattern,
but uses a separate MuseTalk runtime secret instead of the provider/API-key
secret.

## Vast Bootstrap Env

Vast is outside AWS, so it needs a minimal bootstrap AWS credential in the Vast
template. That credential should only be allowed to read the MuseTalk runtime
secret.

Set these in the Vast template or on-start environment:

```bash
export MUSETALK_AWS_SECRET_ID=arn:aws:secretsmanager:us-east-1:211125449207:secret:lingua/musetalk-worker-runtime-xxxxxx
export MUSETALK_AWS_SECRET_REGION=us-east-1
export MUSETALK_SECRETS_STRICT=true
export MUSETALK_SECRETS_VERIFY_S3=1

export AWS_ACCESS_KEY_ID=<bootstrap-secret-reader-access-key-id>
export AWS_SECRET_ACCESS_KEY=<bootstrap-secret-reader-secret-access-key>
export AWS_DEFAULT_REGION=us-east-1
```

Do not set `MUSETALK_AWS_SECRET_ID` to the backend provider-key secret
`lingua/api-keys-*`. The worker runtime secret should be separate.

## Runtime Secret Shape

Recommended JSON for `lingua/musetalk-worker-runtime`:

```json
{
  "AWS_ACCESS_KEY_ID": "<s3-runtime-access-key-id>",
  "AWS_SECRET_ACCESS_KEY": "<s3-runtime-secret-access-key>",
  "AWS_DEFAULT_REGION": "us-east-1",

  "AVATAR_S3_ENABLED": "1",
  "AVATAR_S3_BUCKET": "lingua-musetalk-s3-storage",
  "AVATAR_S3_PREFIX": "avatars",
  "AVATAR_S3_REGION": "us-east-1",

  "TTS_AUDIO_BUCKET": "lingua-audio-files",
  "IDLE_VIDEO_BUCKET": "lingua-ai-idle-vids",
  "UPLOAD_BUCKETS": "lingua-blob-storage,linguaprofilepics",

  "LINGUA_CONTROL_PLANE_BASE_URL": "http://18.205.211.142:8000",
  "LINGUA_WORKER_REGISTER_URL": "http://18.205.211.142:8000/api/runtime/workers/register",
  "LINGUA_WORKER_HEARTBEAT_URL": "http://18.205.211.142:8000/api/runtime/workers/heartbeat",
  "LINGUA_WORKER_TOKEN": "<same-value-as-backend-WORKER_REGISTRATION_TOKEN>",
  "LINGUA_WORKER_TYPE": "musetalk",
  "LINGUA_WORKER_DEFAULT_CAPACITY": "1"
}
```

The bootstrap script also fills a few safe aliases:

- `AWS_REGION` from `AWS_DEFAULT_REGION` when missing
- `AWS_DEFAULT_REGION` from `AWS_REGION` when missing
- `AVATAR_S3_REGION` from the AWS region when missing
- `AVATAR_S3_ENABLED=1` when `AVATAR_S3_BUCKET` is present

## Startup Flow

`scripts/vast_onstart.sh` now runs this sequence:

1. Build or validate the MuseTalk venv.
2. Run `scripts/bootstrap_aws_secrets.py` when `MUSETALK_AWS_SECRET_ID` is set.
3. The script fetches the secret JSON from AWS Secrets Manager.
4. It writes shell `export` statements to a temporary file with mode `0600`.
5. `vast_onstart.sh` sources that file and deletes it immediately.
6. The worker starts through `scripts/vast_server_ctl.sh start`.
7. `api_server.py` registers with Lingua and starts heartbeats when the
   `LINGUA_*` env vars are present.

The bootstrap logs env var names only. It never prints raw secret values.

## Bootstrap IAM Policy

Attach this to the minimal bootstrap IAM user/key used by Vast:

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
      "Resource": "arn:aws:secretsmanager:us-east-1:211125449207:secret:lingua/musetalk-worker-runtime-*"
    }
  ]
}
```

## Runtime S3 IAM Policy

Attach this to the S3 runtime key stored inside the MuseTalk runtime secret.
This includes the avatar asset bucket required for prepared avatar persistence.

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
        "arn:aws:s3:::lingua-musetalk-s3-storage",
        "arn:aws:s3:::lingua-audio-files",
        "arn:aws:s3:::lingua-ai-idle-vids",
        "arn:aws:s3:::lingua-blob-storage",
        "arn:aws:s3:::linguaprofilepics"
      ]
    },
    {
      "Sid": "ObjectLevel",
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:AbortMultipartUpload",
        "s3:ListMultipartUploadParts"
      ],
      "Resource": [
        "arn:aws:s3:::lingua-musetalk-s3-storage/avatars/*",
        "arn:aws:s3:::lingua-audio-files/*",
        "arn:aws:s3:::lingua-ai-idle-vids/*",
        "arn:aws:s3:::lingua-blob-storage/*",
        "arn:aws:s3:::linguaprofilepics/*"
      ]
    }
  ]
}
```

## Verification

With `MUSETALK_SECRETS_VERIFY_S3=1`, bootstrap calls `head_bucket` for bucket
env vars found in the secret:

- `AVATAR_S3_BUCKET`
- `TTS_AUDIO_BUCKET`
- `IDLE_VIDEO_BUCKET`
- each bucket in `UPLOAD_BUCKETS`

S3 verification requires `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY` from the
MuseTalk runtime secret; it will not fall back to the bootstrap secret-reader key.

After the worker starts:

```bash
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/worker/state
```

Expected control-plane signs:

- `/health` remains healthy only after worker registration succeeds when
  control-plane env vars are configured.
- `/worker/state` shows `registered: true`.
- `last_register_error` and `last_heartbeat_error` are `null`.

If S3 verification passes and `/avatars/prepare` succeeds, `/stats` should show
`avatar_s3.upload_successes` incrementing after prepared avatar upload.
