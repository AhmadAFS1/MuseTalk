# Python venv on S3: bounded RTX 3090 experiment

Date: 2026-10-11. Authorized by the user: one RTX 3090, for at most one hour,
created through the existing EC2 controller; private S3 prefix `python-venv/`.
Docker is not part of this experiment. No pip/CUDA installation runs on macOS.

Outcome update 2026-10-12: A3 published the private archive and validated a fresh
same-host S3 restore. Packages took 22m42s, healthy API 23m36s; single GET was the
bottleneck. The rental was destroyed before the two-hour cap. Video download
verification failed (404 naming mismatch); no fresh-host/production integration
or parallel comparison was completed. See [final progress](Python-s3-venv-progress.md)
and [A3 receipts](python_s3_venv_20261011/a3-receipts.json).

### Approved retry update (Oct 11 22:23 CDT / Oct 12 03:23 UTC)

After two failed publication attempts, user approved **one new retry, at most two
hours**, with the same hourly ($0.30) and bandwidth ($0.0015/GB each direction)
caps. Use a unique A3 ledger/prefix and an independent expiry before one create
request. Original ledgers are immutable; production RTX 4070S is protected.
Before renting, prove detached execution and signed multipart receipts on EC2.
Long GPU jobs receive tickets over stdin only and survive transport closure.
Save multipart identity before transfers, reconcile missing parts using S3
ListParts, and abort an unfinished upload only after the owned worker is absent.
Success requires publication plus a fresh GET/checksum, same-path restore,
identical package manifest, CUDA/native checks and API health. Same-host restore
is not a measured fresh-host boot or a WebRTC acceptance test.

## What is already on S3?

The current `main` source installer still downloads model weights from Hugging Face
and a face-parsing file from Google Drive. TensorRT engine bundles and prepared avatar
caches are on S3. An archive uploaded for earlier Docker work, if present, does not
automatically change the source installer's download path. This experiment packages
Python dependencies, not model weights, avatars, or TensorRT engines.

## Limits and safety

- Use one verified US RTX 3090, stock `vastai/pytorch:cuda-12.1-auto`, Linux amd64,
  Python 3.10/cu121, at least 32 GB RAM and sufficient disk for both environments.
- Offer quote must remain at or below $0.30/hour, including allocated disk; each
  transfer direction must cost at most $0.0015/GB ($1.50/TB). S3 internet egress is
  separate; the user accepts approximately $0.40 for a roughly 4.5 GB restore.
  **Approved exception at 2026-10-11 21:00 UTC:** Kentucky offer `35170923` may use
  ≤ $0.264/hr total / $0.410/hr adjusted, with download ≤ $0.00390625/GB and
  upload ≤ $0.005208333333333333/GB. This is specific to that offer/machine/host;
  it does not authorize arbitrary more expensive rentals.
- Install a durable EC2 one-hour expiry timer before submitting the create request.
  Persist the unique label, selected machine, pre-existing IDs and new instance ID.
  Never retry an uncertain create; reconcile the exact label instead.
- Do not modify EC2 production services, unrelated IAM policy, the saved Vast template, the
  production RTX 4070, or other users' instances. Never print credentials or signed
  URLs. The existing runtime identity has engine/avatar read permissions, but
  `python-venv/` uploads require separate publisher permission. Use EC2-generated
  object-specific signed upload/download requests; do not send EC2 role keys to Vast.
- Stop the test API before moving its venv. Rename it to an explicit rollback path
  rather than irreversibly deleting it. The working venv path must be absent during
  restoration. No package-install fallback is allowed to count as a restore pass.
- Always destroy only the ledger-owned test instance and verify provider absence.
  Keep the private versioned S3 artifact for subsequent work; report its size/cost.

## Experiment sequence

1. Record read-only EC2 access, offer, template, source revision, and existing
   instances. Freeze `main` at `5cc706e90e50e93da1310628c025a84199cd8042`, with
   the local core/Kokoro timing patch applied and its SHA-256 recorded. Keep the
   optional groups unchanged: main server, Kokoro, avatar preparation.
2. Start the one-hour lease. Measure request -> container -> checkout -> installation
   -> healthy API. Collect the canonical phase timings and main/Kokoro timers.
3. On that Linux worker only, inspect the final environment and base interpreter,
   pins, optional group membership, disk and ABI. Validate imports/CUDA before taking
   an archive. Preserve the final path `/workspace/.venvs/musetalk_trt_stagewise`.
4. Build `tar.zst` containing only that environment, preserving normal venv symlinks.
   Exclude runtime secrets, external caches, models and GPU engine/build caches.
   Record archive/payload sizes, SHA-256, base Python path/version, distro/glibc,
   source/pins/installer fingerprints, and package metadata in a manifest. Audit
   suspicious secret filenames and reject unexpected links before upload.
5. Upload to a unique private key under
   `python-venv/20261011/<experiment-label>/`. Record upload duration and HEAD size.
   Upload a small JSON manifest alongside it; S3 folders are key prefixes.
6. Stop the test API. Rename its working venv to a rollback path. Use system Python
   plus a standalone downloader, not the removed venv, to perform a fresh S3 GET.
   Restore the downloaded archive to the identical absolute path. Separately time
   download, checksum, extraction/publication, package/import/CUDA validation, and
   server restart -> health. Keep the original local archive out of the restore path.
7. Verify package versions/groups match the source-installed environment. Run the
   canonical GPU self-test without building TensorRT weights; restart the actual API.
   If lease time permits, run a bounded REST audio/video smoke using the known-good
   existing avatar. This is not WebRTC/TURN/production registration acceptance.
8. Record results and limitations, restore rollback if needed, stop the worker and
   destroy the one test rental before lease expiry. Verify unrelated instances remain.

## Success criteria and interpretation

The archive must round-trip through S3, restore with **no pip installation**, preserve
the pinned stack including optional groups, pass native/CUDA validation, and allow
the API to become healthy. Report actual seconds and MB/s, never advertised network
speed as measured S3 speed. Compare to dependency install time, not to the entire boot.

This is a same-instance dependency replacement test. System packages, repository,
weights, engine bundles and filesystem caches remain warm. It does **not** demonstrate
a fresh-host five-minute scale-out. Next work, after a successful test, would integrate
an opt-in validated restore into the installer so `SETUP_CLEAN=1` cannot erase it and
missing model weights cannot silently cause pip to rerun. That integration and a
second fresh-instance A/B test are not included in this one-hour rental.

## Reporting

Update [Python-s3-venv-progress.md](Python-s3-venv-progress.md) after meaningful
changes/results. Keep evidence under a task-specific directory. Distinguish measured
results, estimates, failures and untested claims. No automatic Git push is authorized.

## Preflight permission finding (2026-10-11 19:44 UTC)

Both `musetalk-s3-runtime` and the EC2 instance role `linguaEc2role` were denied
`s3:PutObject` on the requested prefix. No test object was created. The EC2 role
already has `AmazonS3ReadOnlyAccess`, so an additional read grant is not required
for signing downloads. Do not give write access to the runtime worker identity.

Before renting, request approval for a dedicated inline publisher policy on the
**exact** EC2 role `linguaEc2role`, limited to the following statement:

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Sid": "PublishMuseTalkPythonVenvOnly",
    "Effect": "Allow",
    "Action": ["s3:PutObject", "s3:AbortMultipartUpload"],
    "Resource": "arn:aws:s3:::lingua-musetalk-s3-storage/python-venv/*"
  }]
}
```

The user approved this exact grant and created it through the console. At
2026-10-11 20:28 UTC, an EC2 prefix upload and HEAD verification succeeded. Reading
the policy through the EC2 role is still denied. At that checkpoint no currently
returned North American host met the original bandwidth and price caps.

**Execution update, 21:04 UTC:** the Kentucky exception was not used. A verified
replacement on US machine `136634`, host `399360`, became available within the
original caps. Accepted offer `48396264` created test instance `55446768` at
`21:04:25.644958 UTC`, with independent expiry `22:04:25 UTC`. With 150 GB disk,
the quote was $0.243333/hr total / $0.292161/hr adjusted; each transfer direction
was $0.0013020833333333333/GB. Advertised network was 923 Mbps down / 876 Mbps up.
See the progress log for measurements and final cleanup, not these network ratings.

This does not allow object deletion, bucket configuration changes, ACL changes or
writes outside `python-venv/`. Multipart part creation/completion use PutObject;
AbortMultipartUpload allows failed uploads to be cleaned up. Existing read access
is sufficient to inspect/verify objects. This is the intended narrow policy; only
the prefix upload/HEAD behavior has been independently verified after creation.

## Approved retry after first failure

The user authorized one second bounded attempt on 2026-10-11 local date. Use a
separate `-a2` namespace and unchanged price/bandwidth limits. Current planned
allocation is 128 GB, not 150 GB; first-attempt receipts are retained unchanged.
Validate SSH/helper execution early. Capture sanitized diagnostic evidence before
cleanup; on failure, the same worker can remain available for inspection only
until its original one-hour expiry. This never authorizes extending its lease or
automatically creating a third worker. See the progress log for the accepted quote,
actual ID, cutoff and measured artifact/restore results.
