#!/usr/bin/env python3
"""Continuous remote-acquisition-to-validation trial, not a full Vast boot timer.

Ephemeral private GET tickets enter through stdin and are never written to logs,
argv, manifests or receipts. Exact contract digest is independently pinned.
"""
import argparse
import json
import os
from pathlib import Path
import platform
import time
from types import SimpleNamespace

import benchmark_dependency_install as bench
from dependency_artifacts import require, selected
import fetch_dependency_tool as tool
import fetch_dependency_wheels as fetch


def run(args, tickets):
    require(platform.system() == "Linux" and platform.machine() == "x86_64", "Linux only: no macOS package tests")
    start = time.monotonic()
    root = bench.trial_path(args.experiment_root, args.trial)
    root.mkdir(mode=0o700)
    (root/"owned-delivery.json").write_text(json.dumps({"trial":args.trial,"uid":os.getuid(),"created_epoch":time.time()}))
    result = {"state":"running","trial":args.trial,"method":args.method,"workers":args.workers,"group":args.group,
              "remote_acquisition_included":True,"full_Vast_boot_included":False,
              "cache_state":{"downloaded_artifacts":"fresh","venv":"fresh","installer_cache":"fresh","OS_page_cache":"unknown; not dropped","CDN":"unknown"}}
    try:
        contract = json.loads(args.contract.read_text())
        require(contract["owner"] == fetch.OWNER and contract["bucket"] == fetch.BUCKET
                and contract["prefix"] == fetch.PRIVATE_PREFIX+contract["artifact_id"]+"/", "Unreviewed delivery contract")
        require(set(tickets) == {"manifest", "wheels"}, "Unexpected private ticket envelope")
        metadata = {"name":"dependency-manifest","filename":"manifest.json","sha256":contract["manifest_sha256"],
                    "size_bytes":contract["manifest_bytes"],"source":"build:manifest"}
        result["manifest_acquisition"] = fetch.acquire_one(metadata, root, ticket=tickets["manifest"])
        manifest = fetch.read_trusted(root/"manifest.json", contract["manifest_sha256"])
        require(manifest["artifact_id"] == contract["artifact_id"], "Downloaded manifest identity differs")
        facts = {"python":platform.python_version(),"glibc":platform.libc_ver()[1]}
        require(facts["python"].startswith("3.10.") and tuple(map(int,facts["glibc"].split('.'))) >= (2,35), "Wrong Python/glibc delivery ABI")
        group = "bootstrap" if args.method == "source-pip" else args.group
        needed = {a["sha256"] for a in selected(manifest, group) if a["source"].startswith("build:")}
        result["acquisition"] = fetch.acquire(manifest, root/"wheels", group, args.workers, args.repo_root,
                                               tickets={k:v for k,v in tickets["wheels"].items() if k in needed})
        uv = None
        if args.method.endswith("uv"):
            result["tool_bootstrap"] = tool.fetch(root/"uv-tool")
            uv = root/"uv-tool/uv"
            require(fetch.sha256_file(uv) == contract["uv_binary_sha256"], "Pinned installer binary mismatch")
        nested = bench.run(SimpleNamespace(experiment_root=args.experiment_root,trial="inner-"+args.trial,
            manifest=root/"manifest.json",manifest_sha256=contract["manifest_sha256"],wheels=root/"wheels",group=args.group,
            installer="uv" if uv else "pip",compile_bytecode=True,link_mode="copy",python="python3.10",
            uv=uv,uv_sha256=contract["uv_binary_sha256"] if uv else None,source_pip=args.method == "source-pip",
            repo_root=args.repo_root,reference_wheels=args.reference_wheels))
        result["installation"] = nested
        require(nested["state"] == "microbenchmark-passed", "Acquisition completed but installer/native validation failed")
        result["packages_ready_seconds"] = nested["timer_start_monotonic"]+nested["packages_ready_seconds"]-start
        result["validated_ready_seconds"] = time.monotonic()-start
        result["state"] = "delivery-trial-passed"
        result["limitations"] = ["not provider/container/git/models/engine startup time","tickets pre-issued on EC2; issuance outside worker timer",
                                  "source control's two hashless PyTorch-index reports use trusted local wheel RECORD validation",
                                  "application first-output acceptance is a separate gate"]
    except BaseException as error:
        result.update(state="failed",error_type=type(error).__name__,detail=str(error) if type(error) is ValueError else "Inspect protected per-step evidence")
    result["outer_seconds"] = time.monotonic()-start
    (root/"receipt.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--experiment-root",type=Path,required=True)
    p.add_argument("--trial",required=True)
    p.add_argument("--contract",type=Path,required=True)
    p.add_argument("--method",choices=("source-pip","wheelhouse-pip","wheelhouse-uv"),required=True)
    p.add_argument("--workers",type=int,choices=(1,2,4,8),required=True)
    p.add_argument("--group",choices=("server","server+kokoro","full"),default="full")
    p.add_argument("--repo-root",type=Path,default=Path("/workspace/MuseTalk"))
    p.add_argument("--reference-wheels",type=Path)
    a=p.parse_args()
    envelope=json.loads(__import__('sys').stdin.read(256000))
    r=run(a,envelope)
    print(json.dumps({'result':r}),flush=True)
    raise SystemExit(0 if r['state']=='delivery-trial-passed' else 2)
