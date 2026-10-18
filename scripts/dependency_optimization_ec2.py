#!/usr/bin/env python3
"""Owned, bounded Linux experiment control plane. Run only as root on my-ec2.

Each label has exactly one create intent. Observation failures never authorize a
second create. Expiry is independent of this chat or an SSH observation process.
"""
import argparse
import base64
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid

HERE = Path(__file__).resolve().parent
LABEL = "musetalk-dependency-optimization-20261015-a1"
ROOT = Path("/home/ec2-user/.local/state") / LABEL
WORKER_ROOT = "/workspace/.dependency-optimization-20261015-a1"
LIFETIME = 7200
DISK = 100
MACHINE = 92165
HOST = 443275
LOCATION = "Pennsylvania, US"
PURPOSE = "dependency optimization on isolated Linux; existing 8-core/unverified US fallback"


def library():
    path = HERE / "python_s3_venv_source_library.py"
    spec = importlib.util.spec_from_file_location("optimization_source_safety", path)
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    b.LABEL, b.ROOT, b.DISK, b.LIFETIME_SECONDS = LABEL, ROOT, DISK, LIFETIME
    return b


def expiry_timer(deadline):
    when = dt.datetime.fromisoformat(deadline).strftime("%Y-%m-%d %H:%M:%S UTC")
    subprocess.run(["systemd-run", "--quiet", "--unit=" + LABEL + "-expiry", "--on-calendar=" + when,
                    "--timer-property=AccuracySec=1s", "--property=Restart=on-failure", "--property=RestartSec=30s",
                    "/home/ec2-user/lingua/venv/bin/python", "-B", str(Path(__file__).resolve()), "expire"],
                   check=True, capture_output=True, timeout=15)
    if subprocess.run(["systemctl", "is-active", LABEL + "-expiry.timer"], capture_output=True, timeout=10).returncode:
        raise ValueError("Independent expiry not active")


def approved(row, b):
    return (row.get("machine_id") == MACHINE and row.get("host_id") == HOST
            and row.get("geolocation") == LOCATION and b.acceptable(row, allow_unverified=True)
            and all(type(row.get(k)) in (int, float) and 0 < row[k] <= .30 for k in ("dph_total", "dph_total_adj")))


def freeze(b, template):
    env, script, receipt = b.frozen_template(template)
    patch = (HERE / "python_s3_venv_timing.patch").read_bytes()
    payload = base64.b64encode(gzip.compress(patch)).decode()
    code = "import base64,gzip,subprocess;data=gzip.decompress(base64.b64decode(" + repr(payload) + "));subprocess.run(['git','apply','--check','-'],input=data,check=True);subprocess.run(['git','apply','-'],input=data,check=True)"
    marker = 'echo "[source-baseline] checkout_ready $(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"'
    b.require(script.count(marker) == 1, "Unique verified checkout marker required")
    script = script.replace(marker, marker + "\npython3 -B -c " + shlex.quote(code), 1)
    env.update({"PIP_CACHE_DIR": WORKER_ROOT + "/baseline-pip-cache", "PYTHONUNBUFFERED": "1"})
    subprocess.run(["bash", "-n"], input=script.encode(), capture_output=True, check=True)
    b.require(len(script.encode()) < 12000, "Launch command too large; do not submit")
    receipt.update(run_onstart_sha256=hashlib.sha256(script.encode()).hexdigest(),
                   timing_patch_sha256=hashlib.sha256(patch).hexdigest(),
                   package_cache="fresh experiment-owned path; no inherited pip cache", lifetime_seconds=LIFETIME,
                   test_scope="source baseline then isolated wheelhouse benchmarks; no production callbacks")
    receipt["changes"].append("separate package-group timers; no pip hooks, unchanged pins/indexes")
    return env, script, receipt


def create(b):
    b.require(os.getuid() == 0 and not ROOT.exists(), "Fresh root-owned run only; never repeat create")
    os.umask(0o077)
    client = b.bootstrap()
    import boto3
    import requests
    existing = client.list_instances()
    b.require(not any(r.get("label") == LABEL for r in existing), "Existing label; reconcile, do not create")
    rows = client.search_offers({"limit": 200, "type": "ondemand", "rentable": {"eq": True},
        "num_gpus": {"eq": 1}, "gpu_name": {"eq": "RTX 3090"}, "allocated_storage": DISK,
        "geolocation": {"in": ["US"]}, "dph_total": {"lte": .30},
        "inet_down_cost": {"lte": .0015}, "inet_up_cost": {"lte": .0015}, "order": [["dph_total", "asc"]]})
    offers = [r for r in rows if approved(r, b)]
    b.require(bool(offers), "Reviewed host absent/outside caps; no alternative create")
    offer = min(offers, key=lambda r: (r["dph_total_adj"], r["id"]))
    response = requests.get(client.base_url + "/api/v0/template/", headers=client._headers(),
        params={"select_filters": json.dumps({"hash_id": {"eq": b.TEMPLATE}})}, timeout=20, allow_redirects=False)
    b.require(response.status_code == 200, "Template lookup failed")
    templates = [r for r in response.json()["templates"] if r.get("hash_id") == b.TEMPLATE]
    b.require(len(templates) == 1, "Ambiguous stock template")
    env, script, receipt = freeze(b, templates[0])
    secret = json.loads(boto3.client("secretsmanager", region_name="us-east-1").get_secret_value(SecretId="lingua/musetalk-worker-runtime")["SecretString"])
    session = boto3.Session(aws_access_key_id=secret["AWS_ACCESS_KEY_ID"], aws_secret_access_key=secret["AWS_SECRET_ACCESS_KEY"], region_name="us-east-1")
    b.require(session.client("sts").get_caller_identity()["Arn"].endswith(":user/musetalk-s3-runtime"), "Unexpected worker read identity")
    b.require(session.client("s3").head_object(Bucket=secret["AVATAR_S3_BUCKET"], Key=b.BUNDLE_KEY)["ContentLength"] == 1000524577, "Prebuilt engine unavailable")
    temporary = session.client("sts").get_session_token(DurationSeconds=LIFETIME)["Credentials"]
    env.update({"AWS_ACCESS_KEY_ID": temporary["AccessKeyId"], "AWS_SECRET_ACCESS_KEY": temporary["SecretAccessKey"],
                "AWS_SESSION_TOKEN": temporary["SessionToken"], "AWS_DEFAULT_REGION": "us-east-1", "AWS_REGION": "us-east-1",
                "TRT_ARTIFACT_S3_BUCKET": secret["AVATAR_S3_BUCKET"], "TRT_ARTIFACT_S3_REGION": "us-east-1"})
    env.update({k: str(secret[k]) for k in ("AVATAR_S3_BUCKET", "AVATAR_S3_REGION", "AVATAR_S3_PREFIX", "AVATAR_S3_ENABLED") if secret.get(k)})
    ROOT.mkdir(mode=0o700)
    b.save("template.json", receipt)
    keys = ("id", "machine_id", "host_id", "geolocation", "verification", "gpu_name", "num_gpus", "cpu_cores_effective",
            "cpu_ram", "disk_bw", "disk_space", "storage_cost", "inet_down", "inet_up", "inet_down_cost", "inet_up_cost", "dph_total", "dph_total_adj", "reliability")
    b.save("offer.json", {k: offer.get(k) for k in keys})
    deadline = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=LIFETIME)).replace(microsecond=0).isoformat()
    ledger = {"label": LABEL, "state": "initialized", "instance_id": None, "machine_id": MACHINE,
              "protected_instance_ids": [str(r["id"]) for r in existing], "deadline_utc": deadline,
              "lifetime_seconds": LIFETIME, "budget_usd": 2, "source_revision": b.SOURCE,
              "purpose": PURPOSE,
              "s3_session_expires_utc": temporary["Expiration"].isoformat(), "create_requests": 0}
    b.save("ledger.json", ledger)
    expiry_timer(deadline)
    # Persist intent BEFORE network submission. An exception leaves the exact
    # same label/deadline for inventory reconciliation, never an automatic retry.
    ledger.update(state="submitting", create_requests=1, request_started_utc=b.utc())
    b.save("ledger.json", ledger)
    started = time.monotonic()
    request = {"client_id": "me", "template_hash_id": b.TEMPLATE, "image": b.IMAGE, "disk": DISK,
               "label": LABEL, "env": env, "onstart": script, "cancel_unavail": True}
    result = client.create_instance(offer_id=int(offer["id"]), create_request=request)
    identifier = str(result.get("new_contract") or result.get("contract") or result.get("instance_id") or "")
    b.require(result.get("success") is True and identifier.isdigit() and identifier not in ledger["protected_instance_ids"], "Create uncertain: reconcile only")
    ledger.update(state="created", instance_id=identifier, provider_accepted_utc=b.utc(), create_seconds=time.monotonic()-started)
    b.save("ledger.json", ledger)
    b.event("CREATED", instance_id=identifier, deadline_utc=deadline, rent_per_hour=offer["dph_total_adj"],
            protected_instance_ids=ledger["protected_instance_ids"])


def status(b):
    ledger = json.loads((ROOT / "ledger.json").read_text())
    client = b.bootstrap()
    row = b.owned(client, ledger)
    if row is None:
        print(json.dumps({"owned_state": "absent", "ledger_state": ledger["state"]}), flush=True)
        return
    if ledger.get("instance_id") is None:
        ledger.update(instance_id=str(row["id"]), state="created", reconciled_without_create_retry=True)
        b.save("ledger.json", ledger)
    if row.get("actual_status") == "running" and not ledger.get("ssh_key_attached"):
        import requests
        key = Path("/home/ec2-user/.ssh/id_ed25519.pub").read_text().strip()
        b.require(key.startswith("ssh-ed25519 ") and "\n" not in key, "Unexpected public key")
        response = requests.post(client.base_url + "/api/v0/instances/" + str(row["id"]) + "/ssh/",
                                 headers=client._headers(), json={"ssh_key": key}, timeout=15, allow_redirects=False)
        b.require(response.status_code == 200 and response.json().get("success") is True, "Owned SSH attach failed")
        ledger["ssh_key_attached"] = True
        b.save("ledger.json", ledger)
    safe = {k: row.get(k) for k in ("id", "label", "machine_id", "gpu_name", "num_gpus", "actual_status", "cur_state",
                                  "image_runtype", "public_ipaddr", "ports", "ssh_host", "ssh_port")}
    b.save("provider-latest.json", safe)
    print(json.dumps({"owned_state": safe, "deadline_utc": ledger["deadline_utc"], "create_requests": ledger["create_requests"]}), flush=True)


def ssh_arguments(row):
    require = library().require
    host = row.get("ssh_host")
    port = int(row.get("ssh_port", 0)) + (1 if "jupyter" in str(row.get("image_runtype", "")) else 0)
    require(host and re.fullmatch(r"[A-Za-z0-9.-]+", host) and 1 <= port <= 65535, "Invalid owned SSH endpoint")
    return ["ssh", "-i", "/home/ec2-user/.ssh/id_ed25519", "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=2", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile=" + str(ROOT / "known_hosts"), "-p", str(port), "root@" + host, "python3 -B -"]


def worker(b, payload):
    ledger = json.loads((ROOT / "ledger.json").read_text())
    b.require(ledger["state"] == "created" and dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(ledger["deadline_utc"]) - dt.timedelta(seconds=180), "No live experiment lease")
    row = b.owned(b.bootstrap(), ledger)
    b.require(row and row.get("actual_status") == "running" and ledger.get("ssh_key_attached"), "Owned worker not SSH-ready")
    r = subprocess.run(ssh_arguments(row), input=payload.encode(), capture_output=True, timeout=25)
    b.require(r.returncode == 0, "Owned SSH observation/dispatch failed; re-poll same handle")
    return r.stdout


def probe(b):
    payload = '''import datetime,json,pathlib,subprocess,urllib.request,shutil
health=None
try:
 with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=1) as r:
  body=json.loads(r.read(256000));health={'status':r.status,'ok':body.get('ok'),'service':body.get('service')}
except Exception:pass
lines=[]
p=pathlib.Path('/workspace/bootstrap.log')
if p.is_file():
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-80000));s=f.read().decode(errors='replace')
 lines=[v for v in s.splitlines() if any(k in v for k in ('DONE:','phase finished','[source-baseline]','VAST_ONSTART COMPLETE','TOTAL ELAPSED:'))][-30:]
print(json.dumps({'worker_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'health':health,'timing_lines':lines,'free_disk_bytes':shutil.disk_usage('/workspace').free}))
'''
    value = json.loads(worker(b, payload))
    b.save("worker-latest.json", value)
    print(json.dumps(value), flush=True)


def stage(b, version="tools"):
    # Small reviewed sources/manifests only; no credentialed logs or config.
    files = ["dependency_artifacts.py", "fetch_dependency_wheels.py", "build_dependency_wheelhouse.py",
             "benchmark_dependency_install.py", "fetch_dependency_tool.py", "dependency_screening_worker.py",
             "dependency_startup_profile.py", "dependency_screening_recovery.py", "dependency_local_matrix.py",
             "benchmark_dependency_delivery.py", "dependency_delivery_matrix.py", "dependency_s3_range_probe.py"]
    if version in {"tools-v6","tools-v8","tools-v9","tools-v10"}:
        files += ["install_locked_dependencies.py","dependency_followup_worker.py", "install_musetalk.sh",
                  "musetalk_install_state.py","lib/step_logging.sh"]
    if version=="tools-v8":files += ["dependency_canonical_integration_worker.py"]
    if version=="tools-v9":files += ["dependency_delivery_confirmation_worker.py"]
    if version=="tools-v10":files += ["dependency_warm_start_worker.py","python_s3_venv_worker.py"]
    if version=="tools-v7":
        files += ["dependency_s3_full_worker.py","python_s3_venv_worker.py"]
    payloads = {name: (HERE/name).read_bytes() for name in files}
    payloads["acquisition-manifest.json"] = (HERE/"acquisition-manifest.json").read_bytes()
    if version in {"tools-v4","tools-v5","tools-v6","tools-v7","tools-v8","tools-v9","tools-v10"}:
        payloads["dependency-contract.json"] = (HERE/"dependency-contract.json").read_bytes()
    encoded = {name: base64.b64encode(gzip.compress(data)).decode() for name, data in payloads.items()}
    checks = {name: hashlib.sha256(data).hexdigest() for name, data in payloads.items()}
    source = '''import base64,gzip,hashlib,json,pathlib,os
root=pathlib.Path(ROOT);root.mkdir(mode=0o700,parents=True,exist_ok=True)
assert not root.is_symlink()
files=FILES;hashes=HASHES
for name,data in files.items():
 p=root/name;decoded=gzip.decompress(base64.b64decode(data));assert hashlib.sha256(decoded).hexdigest()==hashes[name]
 p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
 if p.exists():assert not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest()==hashes[name]
 else:p.write_bytes(decoded);os.chmod(p,0o600)
print(json.dumps({'state':'tools-staged','files':hashes}))
'''.replace("ROOT", repr(WORKER_ROOT + "/" + version)).replace("FILES", repr(encoded)).replace("HASHES", repr(checks))
    result = json.loads(worker(b, source))
    b.save("staged-tools.json", result)
    print(json.dumps(result), flush=True)


def screen(b, launch=False, recovery=False, phase="build", matrix=False, delivery=False, delivery_attempt="a1", pilot=False, followup=False):
    from python_s3_venv_detached import launcher, job_path
    prefix = "followup" if followup else ("range-pilot" if pilot else (("delivery2" if delivery_attempt=="a2" else "delivery") if delivery else ("matrix" if matrix else (("imports-recovery" if phase == "imports" else "recovery") if recovery else "screen"))))
    receipt = ROOT / (prefix + "-job.json")
    if launch:
        ledger = json.loads((ROOT / "ledger.json").read_text())
        name=("dependency_followup_worker.py" if followup else ("dependency_s3_range_probe.py" if pilot else ("dependency_delivery_matrix.py" if delivery else ("dependency_local_matrix.py" if matrix else ("dependency_screening_recovery.py" if recovery else "dependency_screening_worker.py")))))
        script=(HERE/name).read_bytes()
        if receipt.exists():
            job = json.loads(receipt.read_text())
            b.require(job["source_sha256"] == hashlib.sha256(script).hexdigest(), "Existing screening source differs; inspect, do not restart")
        else:
            job = {"job_id": uuid.uuid4().hex, "source_sha256": hashlib.sha256(script).hexdigest(), "deadline_utc": ledger["deadline_utc"]}
            b.save(prefix+"-job.json", job)
        if pilot:
            from dependency_wheel_publish_ec2 import checked_s3, BUCKET, OWNER
            from dependency_s3_range_probe import KEY, SIZE, SHA
            s3=checked_s3()
            head=s3.head_object(Bucket=BUCKET,Key=KEY,ExpectedBucketOwner=OWNER)
            b.require(head["ContentLength"]==SIZE and head.get("VersionId"),"Versioned env archive not available")
            config={"key":KEY,"bytes":SIZE,"expected_sha256":SHA,"etag":head["ETag"],"version_id":head["VersionId"],
                "url":s3.generate_presigned_url("get_object",Params={"Bucket":BUCKET,"Key":KEY,"VersionId":head["VersionId"],"ExpectedBucketOwner":OWNER,"IfMatch":head["ETag"]},ExpiresIn=3600)}
            payload=("import sys,json;sys.path.insert(0,"+repr(WORKER_ROOT+"/tools-v5")
                     +");import dependency_s3_range_probe as s;print(json.dumps({'result':s.after_delivery("+repr(config)+","+repr(job["deadline_utc"])+")}))")
        elif delivery or followup:
            from dependency_wheel_publish_ec2 import checked_s3, tickets, presign_get
            publication = json.loads((ROOT/"small-wheel-publication.json").read_text())
            manifest = json.loads((ROOT/"evidence/wheelhouse-build-a2/manifest.json").read_text())
            b.require(publication["state"] == "published-verified-private" and publication["artifact_id"] == manifest["artifact_id"], "Private wheel publication not verified")
            s3 = checked_s3()
            contract = json.loads((HERE/"dependency-contract.json").read_text())
            key = publication["prefix"]+"manifest.json"
            envelope = {"manifest":{"sha256":contract["manifest_sha256"],"owner":contract["owner"],"key":key,
                         "url":presign_get(s3,key,3600)},
                        "wheels":tickets(manifest, expires=3600)}
            if followup:
                payload=("import sys,json;sys.path.insert(0,"+repr(WORKER_ROOT+"/tools-v6")
                         +");import dependency_followup_worker as s;print(json.dumps({'result':s.execute("+repr(envelope)+","+repr(job["deadline_utc"])+")}))")
            else:
                payload = ("import sys,json;sys.path.insert(0,"+repr(WORKER_ROOT+("/tools-v5" if delivery_attempt=="a2" else "/tools-v4"))
                           +");import dependency_delivery_matrix as s;print(json.dumps({'result':s.execute("+repr(job["deadline_utc"])+","+repr(envelope)+","+repr(delivery_attempt)+")}))")
        elif matrix:
            acceptance = json.loads((ROOT/"acceptance-a1/receipt.json").read_text())
            b.require(acceptance["state"] == "passed", "Application output acceptance incomplete")
            payload = ("import sys,json;sys.path.insert(0,"+repr(WORKER_ROOT+"/tools-v3")
                       +");import dependency_local_matrix as s;print(json.dumps({'result':s.execute("+repr(job["deadline_utc"])+")}))")
        elif recovery:
            payload = ("import sys,json;sys.path.insert(0,"+repr(WORKER_ROOT+("/tools-v3" if phase == "imports" else "/tools-v2"))
                       +");import dependency_screening_recovery as s;print(json.dumps({'result':s.execute("+repr(job["deadline_utc"])+","+repr(phase)+")}))")
        else:
            payload = ("import sys,json;from pathlib import Path;sys.path.insert(0," + repr(WORKER_ROOT+"/tools")
                       + ");import dependency_screening_worker as s;print(json.dumps({'result':s.execute(Path("
                       + repr(WORKER_ROOT) + ")," + repr(job["deadline_utc"]) + ")}))")
        result = json.loads(worker(b, launcher(WORKER_ROOT+"/jobs", job["job_id"], payload)))
        b.save(prefix+"-dispatch.json", result)
        print(json.dumps(result), flush=True)
        return
    job = json.loads(receipt.read_text())
    path = job_path(WORKER_ROOT+"/jobs", job["job_id"])
    payload = '''import json,pathlib,os
p=pathlib.Path(JOB)
receipt=json.loads((p/'receipt.json').read_text()) if (p/'receipt.json').is_file() else None
pid=json.loads((p/'pid.json').read_text()).get('pid') if (p/'pid.json').is_file() else None
process=None
if pid:
 try:
  stat=pathlib.Path('/proc')/str(pid)/'stat';data=stat.read_text();process={'pid':pid,'state':data[data.rfind(')')+2:].split()[0],'start_ticks':data[data.rfind(')')+2:].split()[19]}
 except (OSError,IndexError):pass
events=[]
ep=pathlib.Path(ROOT)/EVENTS
if ep.is_file():events=[json.loads(line) for line in ep.read_text().splitlines()[-10:]]
print(json.dumps({'job_id':ID,'receipt':receipt,'process':process,'latest_events':events}))
'''.replace("JOB", repr(path)).replace("ROOT", repr(WORKER_ROOT)).replace("EVENTS", repr("followup-a1/events.jsonl" if followup else ("s3-range-pilot-a1/events.jsonl" if pilot else (("delivery-matrix-"+delivery_attempt+"/events.jsonl") if delivery else ("local-matrix-a1/events.jsonl" if matrix else (("screening-a3/events.jsonl" if phase == "imports" else "screening-a2/events.jsonl") if recovery else "screening-a1/events.jsonl")))))).replace("ID", repr(job["job_id"]))
    result = json.loads(worker(b, payload))
    b.save(prefix+"-latest.json", result)
    final=result.get("receipt")
    outcome=final.get("result",{}).get("result",{}) if isinstance(final,dict) else {}
    print(json.dumps({"job_id":result["job_id"],"process":result["process"],
                      "transport_state":final.get("state") if final else None,
                      "test_outcome":outcome.get("state"),"latest_events":result["latest_events"][-4:]}), flush=True)


def resources(b):
    from python_s3_venv_detached import launcher
    path = ROOT / "resource-job.json"
    ledger = json.loads((ROOT / "ledger.json").read_text())
    if path.exists():
        job = json.loads(path.read_text())
    else:
        job = {"job_id": uuid.uuid4().hex, "deadline_epoch": dt.datetime.fromisoformat(ledger["deadline_utc"]).timestamp()-180}
        b.save("resource-job.json", job)
    payload = ("import sys,os,json;os.environ['MUSETALK_DEPENDENCY_PROFILE_DIR']=" + repr(WORKER_ROOT)
               + ";sys.path.insert(0," + repr(WORKER_ROOT+"/tools") + ");import dependency_startup_profile as p;p.observe("
               + repr(job["deadline_epoch"]) + ");print(json.dumps({'result':{'observer':'finished'}}))")
    result = json.loads(worker(b, launcher(WORKER_ROOT+"/jobs", job["job_id"], payload)))
    b.save("resource-dispatch.json", result)
    print(json.dumps(result), flush=True)


def collect(b):
    source = '''import base64,gzip,hashlib,json,pathlib
root=pathlib.Path(ROOT);files=[];budget=40*1024**2
paths=[pathlib.Path('/workspace/bootstrap.log'),pathlib.Path('/workspace/onstart.log'),pathlib.Path('/workspace/logs/musetalk/api_server_8000.log')]
for d in root.iterdir():
 if d.is_dir() and not d.is_symlink() and d.name not in ('tools','source-artifacts-4','baseline-pip-cache','jobs'):
  paths += [p for p in d.iterdir() if p.name in ('manifest.json','receipt.json','retirement.json','events.jsonl','steps.jsonl','fetch-4-receipt.json','first-imports.log','second-process-imports.log','pip-check.log') or p.suffix=='.log' or p.name.endswith('.report.json')]
paths += [pathlib.Path('/workspace/MuseTalk/.runtime/gpu_selftest.json'),pathlib.Path('/workspace/MuseTalk/.runtime/install_state.json')]
for p in paths:
 if not p.is_file() or p.is_symlink() or p.stat().st_size>4*1024**2 or p.stat().st_size>budget:continue
 data=p.read_bytes();budget-=len(data)
 relative=str(p.relative_to(root)) if p.is_relative_to(root) else 'baseline/'+p.name
 files.append({'path':relative,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'data':base64.b64encode(gzip.compress(data)).decode()})
print(json.dumps({'files':files}))
'''.replace("ROOT", repr(WORKER_ROOT))
    result = json.loads(worker(b, source))
    destination = ROOT / "evidence"
    destination.mkdir(mode=0o700, exist_ok=True)
    inventory = []
    for f in result["files"]:
        relative = Path(f["path"])
        b.require(not relative.is_absolute() and len(relative.parts) == 2 and all(re.fullmatch(r"[A-Za-z0-9+._-]+", p) and p not in {".", ".."} for p in relative.parts), "Unsafe evidence path")
        data = gzip.decompress(base64.b64decode(f["data"]))
        b.require(len(data) == f["bytes"] and hashlib.sha256(data).hexdigest() == f["sha256"], "Evidence checksum failed")
        parent = destination / relative.parent
        parent.mkdir(mode=0o700, exist_ok=True)
        (destination / relative).write_bytes(data)
        inventory.append({k: f[k] for k in ("path", "bytes", "sha256")})
    b.save("evidence-inventory.json", {"at_utc": b.utc(), "files": inventory, "resources_excluded_during_transfer_trials": True})
    print(json.dumps({"state": "evidence-collected", "files": len(inventory), "bytes": sum(f["bytes"] for f in inventory)}), flush=True)


def export(b):
    evidence = ROOT / "evidence"
    def read(path):
        return json.loads(path.read_text()) if path.is_file() else None
    baseline = evidence/"baseline/bootstrap.log"
    timings = [line for line in baseline.read_text(errors="replace").splitlines()
               if any(k in line for k in ("DONE:", "phase finished", "Server start-to-health", "Overall on-start completed", "VAST_ONSTART COMPLETE", "TOTAL ELAPSED:"))] if baseline.is_file() else []
    trials = {p.parent.name: read(p) for p in evidence.glob("*/receipt.json")
              if p.parent.name.startswith(("pip-", "uv-", "screening-", "wheelhouse-", "local-matrix-", "delivery-", "inner-delivery-", "s3-range-", "s3-full-", "followup-", "canonical-integration-", "confirmation-", "warm-start-"))}
    result = {"prepared_utc": b.utc(), "goal_complete": False,
              "scope": "interim dependency experiment evidence; local installs exclude remote acquisition",
              "offer": read(ROOT/"offer.json"), "ledger": read(ROOT/"ledger.json"),
              "baseline_timing_lines": timings,
              "acquisition": read(evidence/"screening-a1/fetch-4-receipt.json"),
              "uv_tool": read(evidence/"uv-0.13.1/receipt.json"), "trials_and_builds": trials,
              "application_acceptance": read(ROOT/"acceptance-a1/receipt.json"),
              "application_matrix": read(ROOT/"application-a2/receipt.json"),
              "cold_jit_application_matrix": read(ROOT/"application-a3/receipt.json"),
              "resources_preservation": read(ROOT/"resources-preservation.json"),
              "second_host_offers": read(ROOT/"second-host-offers-latest.json"),
              "cleanup": read(ROOT/"cleanup.json"),
              "evidence_inventory": read(ROOT/"evidence-inventory.json"),
              "staged_tool_hashes": read(ROOT/"staged-tools.json")}
    raw = json.dumps(result, indent=2, allow_nan=False)+"\n"
    b.require(not re.search(r"(?:AKIA|ASIA)[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,}|X-Amz-(?:Credential|Signature|Security-Token)", raw, re.I), "Credential pattern in public results; keep private")
    output = HERE/"optimization-public-results.json"
    output.write_text(raw)
    os.chmod(output, 0o600)
    os.chown(output, HERE.stat().st_uid, HERE.stat().st_gid)
    manifest = evidence/"wheelhouse-build-a2/manifest.json"
    if manifest.is_file():
        target = HERE/"wheelhouse-manifest-a2.json"
        target.write_bytes(manifest.read_bytes())
        os.chmod(target, 0o600)
        os.chown(target, HERE.stat().st_uid, HERE.stat().st_gid)
    print(json.dumps({"state": "sanitized-export-ready", "bytes": len(raw.encode()), "trials_and_builds": len(trials),
                      "sha256": hashlib.sha256(raw.encode()).hexdigest()}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("create", "status", "probe", "stage", "screen", "screen-status", "recover", "recovery-status", "retry-imports", "imports-status", "matrix", "matrix-status", "delivery", "delivery-status", "retry-delivery", "delivery2-status", "s3-pilot", "s3-pilot-status", "followup", "followup-status", "resources", "collect", "export", "cleanup", "expire"))
    p.add_argument("--tools-dir", choices=("tools", "tools-v2", "tools-v3", "tools-v4", "tools-v5", "tools-v6", "tools-v7", "tools-v8", "tools-v9", "tools-v10"), default="tools")
    args = p.parse_args()
    b = library()
    b.require(os.getuid() == 0, "Existing root-owned EC2 control plane required")
    os.umask(0o077)
    if args.mode == "create":
        create(b)
    elif args.mode == "status":
        status(b)
    elif args.mode == "probe":
        probe(b)
    elif args.mode == "stage":
        stage(b, args.tools_dir)
    elif args.mode in {"screen", "screen-status"}:
        screen(b, launch=args.mode == "screen")
    elif args.mode in {"recover", "recovery-status"}:
        screen(b, launch=args.mode == "recover", recovery=True)
    elif args.mode in {"retry-imports", "imports-status"}:
        screen(b, launch=args.mode == "retry-imports", recovery=True, phase="imports")
    elif args.mode in {"matrix", "matrix-status"}:
        screen(b, launch=args.mode == "matrix", matrix=True)
    elif args.mode in {"delivery", "delivery-status"}:
        screen(b, launch=args.mode == "delivery", delivery=True)
    elif args.mode in {"retry-delivery", "delivery2-status"}:
        screen(b, launch=args.mode == "retry-delivery", delivery=True, delivery_attempt="a2")
    elif args.mode in {"s3-pilot","s3-pilot-status"}:
        screen(b,launch=args.mode=="s3-pilot",pilot=True)
    elif args.mode in {"followup","followup-status"}:
        screen(b,launch=args.mode=="followup",followup=True)
    elif args.mode == "resources":
        resources(b)
    elif args.mode == "collect":
        collect(b)
    elif args.mode == "export":
        export(b)
    else:
        b.require(args.mode in {"cleanup", "expire"}, "Unhandled mode; refusing mutation")
        ledger = json.loads((ROOT / "ledger.json").read_text())
        if args.mode == "expire":
            b.require(dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(ledger["deadline_utc"]), "Expiry too early")
        if ledger["state"] == "initialized":
            b.event("NO_SUBMISSION_TO_CLEAN")
        else:
            b.cleanup(b.bootstrap(), ledger)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(json.dumps({"state": "failed", "error_type": type(error).__name__,
                          "detail": str(error) if type(error) is ValueError else "Details suppressed; reconcile owned ledger before any action"}), flush=True)
        raise SystemExit(2)
