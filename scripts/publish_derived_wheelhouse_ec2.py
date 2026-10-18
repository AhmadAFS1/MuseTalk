#!/usr/bin/env python3
"""Derive (on a test worker) and publish (from EC2) a runtime-TRT + seed-index wheelhouse.

Run as root on my-ec2. `derive RUN --revision SHA` executes, on the owned test
worker of locked-boot run RUN, scripts/derive_runtime_trt_manifest.py and
scripts/build_dependency_seed_index.py from the pinned revision against that
worker's locked installer cache (which holds every pinned wheel), then copies
the outputs back to EC2 with sha256 verification. `publish DIR` uploads the
derived manifest and runtime TensorRT wheel to a new private prefix and copies
the unchanged private built wheels server-side from the current prefix, then
re-reads every object. No credentials or URLs are printed.
"""
import argparse
import base64
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import vast_locked_boot_ec2 as m  # noqa: E402
from fetch_dependency_wheels import BUCKET, OWNER, PRIVATE_PREFIX  # noqa: E402

FILES = ("scripts/derive_runtime_trt_manifest.py", "scripts/build_dependency_seed_index.py", "scripts/dependency_seed.py",
         "scripts/dependency_artifacts.py", "scripts/fetch_dependency_wheels.py", "scripts/install_locked_dependencies.py",
         "scripts/benchmark_dependency_install.py", "scripts/fetch_dependency_tool.py", m.CONTRACT_PATH)

DERIVE = r'''
import base64,gzip,hashlib,json,os,pathlib,subprocess,sys
SOURCES=__SOURCES__
root=pathlib.Path('/workspace/derive');out=root/'out'
assert not out.exists(),'derive output already claimed'
for rel,(sha,data) in SOURCES.items():
  p=root/'repo'/rel;p.parent.mkdir(parents=True,exist_ok=True)
  raw=gzip.decompress(base64.b64decode(data));assert hashlib.sha256(raw).hexdigest()==sha;p.write_bytes(raw)
contract=json.loads((root/'repo'/__CONTRACT__).read_text())
cache=pathlib.Path('/workspace/.cache/musetalk-dependencies')/contract['artifact_id']
manifest=cache/'manifest.json';wheels=cache/'wheels'
trt=next(wheels.glob('tensorrt_cu12_libs-10.3.0-*.whl'))
env={k:v for k,v in os.environ.items() if not k.startswith(('AWS_','PIP_','UV_'))}
r=subprocess.run(['/usr/bin/python3.10','-B',str(root/'repo/scripts/derive_runtime_trt_manifest.py'),'--manifest',str(manifest),
  '--manifest-sha256',contract['manifest_sha256'],'--wheel',str(trt),'--output',str(out)],capture_output=True,text=True,env=env,timeout=900)
assert r.returncode==0,('derive failed',r.stdout[-2000:],r.stderr[-2000:])
receipt=json.loads(r.stdout.strip().splitlines()[-1])
seed_out=out/('seed-index-'+receipt['artifact_id']+'.json.gz')
r=subprocess.run(['/usr/bin/python3.10','-B',str(root/'repo/scripts/build_dependency_seed_index.py'),'--manifest',str(out/'manifest.json'),
  '--manifest-sha256',receipt['manifest_sha256'],'--wheels',str(wheels),'--repo-root',str(root/'repo'),'--output',str(seed_out)],
  capture_output=True,text=True,env=env,timeout=900)
assert r.returncode==0,('seed index failed',r.stdout[-2000:],r.stderr[-2000:])
seed=json.loads(r.stdout)
files={}
for p in [out/'manifest.json',out/'receipt.json',seed_out,out/trt.name]+sorted(out.glob('*.lock')):
  data=p.read_bytes();files[p.name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'data':base64.b64encode(gzip.compress(data,1)).decode()}
print(json.dumps({'receipt':receipt,'seed':{k:seed[k] for k in ('seed_index',)},'artifacts':seed['artifacts'],'files':files}))
'''


def derive(run_name, revision):
    run = m.Run(run_name)
    sources = {}
    for rel in FILES:
        raw = m.fetch_revision_file(revision, rel)
        sources[rel] = (hashlib.sha256(raw).hexdigest(), base64.b64encode(gzip.compress(raw)).decode())
    payload = DERIVE.replace("__SOURCES__", repr(sources)).replace("__CONTRACT__", repr(m.CONTRACT_PATH))
    result = json.loads(m.worker(run, payload, timeout=1800))
    dest = m.STATE / ("mtboot-derived-" + result["receipt"]["artifact_id"][:16])
    dest.mkdir(mode=0o700)
    for name, f in result["files"].items():
        data = gzip.decompress(base64.b64decode(f["data"]))
        m.require(len(data) == f["bytes"] and hashlib.sha256(data).hexdigest() == f["sha256"], "Transfer checksum failed")
        (dest / name).write_bytes(data)
    (dest / "derive.json").write_text(json.dumps({k: result[k] for k in ("receipt", "seed", "artifacts")}, indent=2) + "\n")
    print(json.dumps({"state": "derived", "dir": str(dest), "artifact_id": result["receipt"]["artifact_id"],
                      "trt_wheel_bytes": result["receipt"]["trt_wheel_bytes"], "seed_index": result["seed"]["seed_index"],
                      "files": sorted(result["files"])}), flush=True)


def publish(directory, source_contract_revision):
    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError
    from dependency_artifacts import validate_manifest
    directory = Path(directory)
    manifest_raw = (directory / "manifest.json").read_bytes()
    manifest = validate_manifest(json.loads(manifest_raw))
    old = json.loads(m.fetch_revision_file(source_contract_revision, m.CONTRACT_PATH))
    s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version="s3v4"))
    prefix = PRIVATE_PREFIX + manifest["artifact_id"] + "/"
    m.require(old["prefix"] != prefix, "Derived artifact must use a new prefix")
    old_manifest_raw = s3.get_object(Bucket=BUCKET, Key=old["prefix"] + "manifest.json", ExpectedBucketOwner=OWNER)["Body"].read()
    m.require(hashlib.sha256(old_manifest_raw).hexdigest() == old["manifest_sha256"], "Current manifest differs from its contract")
    old_builds = {a["filename"]: a for a in json.loads(old_manifest_raw)["artifacts"] if a["source"].startswith("build:")}
    plan = []
    for a in manifest["artifacts"]:
        if not a["source"].startswith("build:"):
            continue
        local = directory / a["filename"]
        if local.is_file():
            plan.append(("put", a, local))
        else:
            m.require(a["filename"] in old_builds and old_builds[a["filename"]]["sha256"] == a["sha256"], "Unknown private wheel")
            plan.append(("copy", a, None))
    outcomes = []

    def exists(key):
        try:
            s3.head_object(Bucket=BUCKET, Key=key, ExpectedBucketOwner=OWNER)
            return True
        except ClientError as e:
            m.require(e.response.get("Error", {}).get("Code") in {"404", "NoSuchKey", "NotFound"}, "Cannot verify object")
            return False
    for action, a, local in plan:
        key = prefix + a["filename"]
        if not exists(key):
            if action == "put":
                data = local.read_bytes()
                m.require(hashlib.sha256(data).hexdigest() == a["sha256"] and len(data) == a["size_bytes"], "Local wheel differs")
                s3.put_object(Bucket=BUCKET, Key=key, ExpectedBucketOwner=OWNER, Body=data, ServerSideEncryption="AES256",
                              ContentType="application/octet-stream", Metadata={"sha256": a["sha256"]}, IfNoneMatch="*")
            else:
                s3.copy_object(Bucket=BUCKET, Key=key, CopySource={"Bucket": BUCKET, "Key": old["prefix"] + a["filename"]},
                               ExpectedBucketOwner=OWNER, ExpectedSourceBucketOwner=OWNER, ServerSideEncryption="AES256",
                               MetadataDirective="REPLACE", Metadata={"sha256": a["sha256"]}, ContentType="application/octet-stream")
        body = s3.get_object(Bucket=BUCKET, Key=key, ExpectedBucketOwner=OWNER)
        digest = hashlib.sha256()
        for chunk in iter(lambda: body["Body"].read(1 << 20), b""):
            digest.update(chunk)
        m.require(digest.hexdigest() == a["sha256"] and body.get("ServerSideEncryption") == "AES256", "Published wheel differs")
        outcomes.append({"key": key, "sha256": a["sha256"], "bytes": a["size_bytes"], "action": action})
    key = prefix + "manifest.json"
    if not exists(key):
        s3.put_object(Bucket=BUCKET, Key=key, ExpectedBucketOwner=OWNER, Body=manifest_raw, ServerSideEncryption="AES256",
                      ContentType="application/json", Metadata={"sha256": hashlib.sha256(manifest_raw).hexdigest()}, IfNoneMatch="*")
    remote = s3.get_object(Bucket=BUCKET, Key=key, ExpectedBucketOwner=OWNER)["Body"].read()
    m.require(remote == manifest_raw, "Published manifest differs")
    receipt = {"state": "published", "artifact_id": manifest["artifact_id"], "prefix": prefix,
               "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(), "manifest_bytes": len(manifest_raw),
               "private_wheels": len(plan), "objects": outcomes}
    (directory / "publish.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    d = sub.add_parser("derive")
    d.add_argument("run")
    d.add_argument("--revision", required=True)
    u = sub.add_parser("publish")
    u.add_argument("directory")
    u.add_argument("--current-contract-revision", required=True)
    a = p.parse_args()
    m.require(os.getuid() == 0, "Root-owned EC2 control plane only")
    os.umask(0o077)
    if a.mode == "derive":
        derive(a.run, a.revision)
    else:
        publish(a.directory, a.current_contract_revision)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if isinstance(error, SystemExit):
            raise
        print(json.dumps({"state": "failed", "error_type": type(error).__name__,
                          "detail": str(error)[:400] if type(error) in (ValueError, AssertionError) else "Details suppressed"}), flush=True)
        raise SystemExit(2)
