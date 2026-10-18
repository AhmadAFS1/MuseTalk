#!/usr/bin/env python3
"""Preserve/publish only small reviewed build outputs under the approved prefix.

Never grants permissions. Run on the existing EC2 with its role; worker receives
read-only expiring GET tickets, never publisher credentials. Large CDN wheels
are NOT uploaded by this helper.
"""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.parse

import dependency_optimization_ec2 as c
from dependency_artifacts import validate_manifest, offline_lock, require
from fetch_dependency_wheels import BUCKET, OWNER, PRIVATE_PREFIX


def paths_for_manifest(manifest):
    rows = []
    for a in manifest["artifacts"]:
        if not a["source"].startswith("build:"):
            continue
        rows.append({"name": a["filename"], "sha256": a["sha256"], "bytes": a["size_bytes"], "kind": "built-wheel"})
        p = a["build_provenance"]
        rows.append({"name": "source-"+p["input_filename"], "worker_name": p["input_filename"],
                     "sha256": p["input_sha256"], "bytes": None, "kind": "source-archive"})
    require(len(rows) == 8 and all(re.fullmatch(r"[A-Za-z0-9._+!-]+", r["name"]) for r in rows), "Unexpected small build inventory")
    return rows


def preserve(b):
    manifest_path = c.ROOT/"evidence/wheelhouse-build-a2/manifest.json"
    raw = manifest_path.read_bytes()
    manifest = validate_manifest(json.loads(raw))
    rows = paths_for_manifest(manifest)
    source = '''import base64,hashlib,json,pathlib
root=pathlib.Path(ROOT);rows=ROWS;files=[];total=0
for row in rows:
 path=(root/'wheelhouse-build-a2/wheels'/row['name']) if row['kind']=='built-wheel' else (root/'source-artifacts-4'/row['worker_name'])
 assert path.is_file() and not path.is_symlink() and 0<path.stat().st_size<2*1024**2
 data=path.read_bytes();total+=len(data);assert total<2*1024**2 and hashlib.sha256(data).hexdigest()==row['sha256']
 files.append({'name':row['name'],'sha256':row['sha256'],'bytes':len(data),'data':base64.b64encode(data).decode()})
print(json.dumps({'files':files}))
'''.replace("ROOT", repr(c.WORKER_ROOT)).replace("ROWS", repr(rows))
    result = json.loads(c.worker(b, source))
    dest = c.ROOT/"small-wheel-publication"
    dest.mkdir(mode=0o700, exist_ok=True)
    receipt = []
    for row in result["files"]:
        require(row["name"] in {r["name"] for r in rows}, "Unreviewed preservation file")
        data = base64.b64decode(row["data"])
        require(len(data) == row["bytes"] and hashlib.sha256(data).hexdigest() == row["sha256"], "Preserved output differs")
        target = dest/row["name"]
        if target.exists():require(not target.is_symlink() and target.read_bytes() == data, "Existing preserved bytes differ")
        else:target.write_bytes(data)
        receipt.append({k: row[k] for k in ("name", "sha256", "bytes")})
    if (dest/"manifest.json").exists():require((dest/"manifest.json").read_bytes() == raw, "Preserved manifest differs")
    else:(dest/"manifest.json").write_bytes(raw)
    for group in ("bootstrap", "server", "server+kokoro", "full"):
        data = offline_lock(manifest, group).encode()
        name = group+".lock"
        if (dest/name).exists():require((dest/name).read_bytes() == data, "Preserved lock differs")
        else:(dest/name).write_bytes(data)
        receipt.append({"name": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    b.save("small-wheel-preservation.json", {"at_utc": b.utc(), "files": receipt,
            "artifact_id": manifest["artifact_id"], "manifest_sha256": hashlib.sha256(raw).hexdigest()})
    print(json.dumps({"state": "small-inputs-preserved", "files": len(receipt), "bytes": sum(r["bytes"] for r in receipt),
                      "artifact_id": manifest["artifact_id"]}), flush=True)


def checked_s3():
    import boto3
    from botocore.config import Config
    account = boto3.client("sts").get_caller_identity()
    require(account["Account"] == OWNER and "/linguaec2role/" in account["Arn"].lower(), "Unexpected publisher role/account")
    # In us-east-1 botocore can otherwise choose legacy SigV2 for presigning,
    # even when meta.config displays s3v4. Owner binding must stay a signed header.
    s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version="s3v4"))
    block = s3.get_public_access_block(Bucket=BUCKET, ExpectedBucketOwner=OWNER)["PublicAccessBlockConfiguration"]
    require(all(block.get(k) is True for k in ("BlockPublicAcls", "IgnorePublicAcls", "BlockPublicPolicy", "RestrictPublicBuckets")), "Bucket privacy gate not verified")
    require(s3.get_bucket_location(Bucket=BUCKET, ExpectedBucketOwner=OWNER).get("LocationConstraint") in (None, "us-east-1"), "Wrong artifact region")
    return s3


def presign_expiry_fixture(s3, key):
    """One-second test ticket for the already reviewed manifest only."""
    require(key==PRIVATE_PREFIX+'725c6fe7564f9641fce16845a1346da99a75bf3cb54d993a273f34ceb6de7194/manifest.json',
            'Expiry fixture limited to reviewed manifest')
    url=s3.generate_presigned_url('get_object',Params={'Bucket':BUCKET,'Key':key,'ExpectedBucketOwner':OWNER},ExpiresIn=1)
    query=urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    require(query.get('X-Amz-Algorithm')==['AWS4-HMAC-SHA256'],'Expiry fixture legacy auth rejected')
    return url


def presign_get(s3, key, expires):
    require(key.startswith(PRIVATE_PREFIX) and 300 <= expires <= 3600, "Bounded private read only")
    url=s3.generate_presigned_url("get_object", Params={"Bucket":BUCKET,"Key":key,"ExpectedBucketOwner":OWNER},ExpiresIn=expires)
    query=urllib.parse.parse_qs(urllib.parse.urlsplit(url).query)
    require(query.get("X-Amz-Algorithm")==["AWS4-HMAC-SHA256"] and "AWSAccessKeyId" not in query,
            "Presigner returned legacy auth; do not send a malformed owner-bound request")
    return url


def publish(b):
    from botocore.exceptions import ClientError
    s3 = checked_s3()
    root = c.ROOT/"small-wheel-publication"
    preserved = json.loads((c.ROOT/"small-wheel-preservation.json").read_text())
    manifest = validate_manifest(json.loads((root/"manifest.json").read_text()))
    require(preserved["artifact_id"] == manifest["artifact_id"], "Preserved identity mismatch")
    records = [*preserved["files"], {"name": "manifest.json", "sha256": preserved["manifest_sha256"], "bytes": (root/"manifest.json").stat().st_size}]
    prefix = PRIVATE_PREFIX+manifest["artifact_id"]+"/"
    outcomes = []
    for item in records:
        path = root/item["name"]
        require(path.is_file() and not path.is_symlink(), "Missing preserved file")
        data = path.read_bytes()
        require(len(data) == item["bytes"] and hashlib.sha256(data).hexdigest() == item["sha256"], "Publication input hash failed")
        key = prefix+item["name"]
        try:
            head = s3.head_object(Bucket=BUCKET, Key=key, ExpectedBucketOwner=OWNER)
            existed = True
        except ClientError as e:
            require(e.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}, "Cannot verify private object existence/permission")
            existed = False
        if not existed:
            s3.put_object(Bucket=BUCKET, Key=key, ExpectedBucketOwner=OWNER, Body=data, ServerSideEncryption="AES256",
                          ContentType="application/json" if item["name"].endswith(".json") else "application/octet-stream",
                          Metadata={"sha256": item["sha256"]}, IfNoneMatch="*")
        result = s3.get_object(Bucket=BUCKET, Key=key, ExpectedBucketOwner=OWNER)
        require(result.get("ServerSideEncryption") == "AES256" and result["ContentLength"] == len(data), "Remote size/encryption differs")
        with result["Body"] as body:
            remote = body.read(2*1024**2)
        require(hashlib.sha256(remote).hexdigest() == item["sha256"], "Published bytes fail trusted SHA256")
        outcomes.append({"key": key, "sha256": item["sha256"], "bytes": len(data), "existing_verified": existed})
        b.save("small-wheel-publication-progress.json", {"at_utc": b.utc(), "artifact_id": manifest["artifact_id"], "objects": outcomes})
    receipt = {"at_utc": b.utc(), "state": "published-verified-private", "artifact_id": manifest["artifact_id"],
               "bucket": BUCKET, "prefix": prefix, "objects": outcomes,
               "role_permissions_modified": False, "large_CDN_wheels_uploaded": False}
    b.save("small-wheel-publication.json", receipt)
    print(json.dumps(receipt), flush=True)


def tickets(manifest, expires=1800):
    """Return ephemeral signed reads in memory; caller must never log/save them."""
    require(300 <= expires <= 3600, "Bounded read tickets required")
    validate_manifest(manifest)
    s3 = checked_s3()
    result = {}
    for a in manifest["artifacts"]:
        if a["source"].startswith("build:"):
            key = PRIVATE_PREFIX+manifest["artifact_id"]+"/"+a["filename"]
            result[a["sha256"]] = {"sha256": a["sha256"], "owner": OWNER, "key": key,
                "url": presign_get(s3,key,expires)}
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("preserve", "publish"))
    args = p.parse_args()
    try:
        require(os.getuid() == 0, "Existing EC2 control plane only")
        os.umask(0o077)
        b = c.library()
        (preserve if args.mode == "preserve" else publish)(b)
    except BaseException as error:
        print(json.dumps({"state": "failed", "error_type": type(error).__name__,
                          "detail": str(error) if type(error) is ValueError else "Suppressed; inspect private receipts, do not change IAM automatically"}), flush=True)
        raise SystemExit(2)
