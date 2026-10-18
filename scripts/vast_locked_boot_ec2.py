#!/usr/bin/env python3
"""EC2-side controller for SHA-pinned locked-boot RTX 3090 Vast test workers.

Launches the stock Vast PyTorch template with scripts/vast_bootstrap_locked.sh
from a pushed revision as On-start, injecting what the autoscaler will inject:
MUSETALK_SOURCE_REVISION, EC2-issued dependency read tickets, a short-lived STS
worker identity and the runtime S3 configuration (injected mode). Supports the
mission's cold/warm cycle: provider reboot, SSH re-run with fresh tickets (and
optional wipe), timeline extraction, REST acceptance, evidence and cleanup.

Run only as root on my-ec2 with /home/ec2-user/lingua/venv/bin/python. Every run
has exactly one create intent, an independent systemd expiry timer installed
before the create request, and a label that no autoscaler prefix matches.
Credentials stay in memory and in the provider launch environment.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import time
import urllib.request

HERE = Path(__file__).resolve().parent
STATE = Path("/home/ec2-user/.local/state")
LIBRARY = HERE / "python_s3_venv_source_library.py"
REPO_RAW = "https://raw.githubusercontent.com/AhmadAFS1/MuseTalk/"
BOOTSTRAP_PATH = "scripts/vast_bootstrap_locked.sh"
CONTRACT_PATH = "configs/dependencies/rtx3090-cp310-cu121-screening.json"
RUNTIME_SECRET = "lingua/musetalk-worker-runtime"
RUNTIME_USER_SUFFIX = ":user/musetalk-s3-runtime"
LABEL_PREFIX = "mtboot-"
# Autoscaler classification matches these label prefixes; a test label must not.
AUTOSCALER_PREFIXES = ("lingua-musetalk", "lingua-tts", "lingua-soulxflashhead", "lingua-omnivoice", "lingua-")
PROTECTED = {"51074906"}
MAX_ACTIVE_RUNS = 3
TICKET_SECONDS = 3600
DEFAULT_LIFETIME = 7200
DEFAULT_DISK = 100
WIPE_PATHS = ("/workspace/MuseTalk", "/workspace/.venvs", "/workspace/.cache/musetalk-dependencies",
              "/workspace/logs", "/workspace/bootstrap.log", "/workspace/onstart.log")


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


class Run:
    """Per-run state directory and ledger; label mtboot-<name>."""

    def __init__(self, name):
        require(re.fullmatch(r"[a-z0-9][a-z0-9-]{2,39}", name or ""), "Run name: 3-40 chars [a-z0-9-]")
        self.name = name
        self.label = LABEL_PREFIX + name
        require(not self.label.startswith(AUTOSCALER_PREFIXES), "Label would match an autoscaler prefix")
        self.root = STATE / self.label

    def save(self, name, value):
        path = self.root / name
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        os.chmod(tmp, 0o600)
        tmp.replace(path)

    def load(self, name="ledger.json"):
        return json.loads((self.root / name).read_text())

    def event(self, stage, **fields):
        row = {"stage": stage, "at_utc": utc(), **fields}
        with (self.root / "events.jsonl").open("a") as f:
            f.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)
        return row


def library(run):
    """Shared stock-template helpers, rebound to this run's label/state dir."""
    spec = importlib.util.spec_from_file_location("locked_boot_library", LIBRARY)
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    b.LABEL, b.ROOT = run.label, run.root
    return b


def fetch_revision_file(revision, path, limit=1024 * 1024):
    require(re.fullmatch(r"[a-f0-9]{40}", revision or ""), "Full 40-hex revision required")
    with urllib.request.urlopen(REPO_RAW + revision + "/" + path, timeout=30) as response:
        data = response.read(limit + 1)
    require(len(data) <= limit, "Revision file too large")
    return data


def onstart_stub(revision, bootstrap_sha256):
    """Small constant On-start: fetch the pinned bootstrap, verify, exec.

    Keeps the provider's On-start far below any size limit; a reboot reuses the
    verified local copy without network."""
    require(re.fullmatch(r"[a-f0-9]{40}", revision) and re.fullmatch(r"[a-f0-9]{64}", bootstrap_sha256), "Bad pin")
    return f"""#!/bin/bash
set -Eeuo pipefail
f=/workspace/.mtboot/vast_bootstrap_locked.{bootstrap_sha256[:16]}.sh
mkdir -p /workspace/.mtboot
if ! echo "{bootstrap_sha256}  $f" | sha256sum -c --quiet 2>/dev/null; then
  for i in 1 2 3 4 5; do
    curl -fsSL --max-time 60 {REPO_RAW}{revision}/{BOOTSTRAP_PATH} -o "$f.tmp" && break
    sleep 3
  done
  echo "{bootstrap_sha256}  $f.tmp" | sha256sum -c --quiet
  mv -f "$f.tmp" "$f"
fi
exec bash "$f"
"""


def autoscaler_quiet():
    """Test rentals use the production template hash; no autoscaler pass may run."""
    active = subprocess.run(["systemctl", "is-active", "lingua-vast-autoscaler"], capture_output=True, text=True).stdout.strip()
    pid = subprocess.check_output(["systemctl", "show", "lingua-api", "-p", "MainPID", "--value"], text=True).strip()
    env = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0") if pid.isdigit() and pid != "0" else []
    enabled = [e for e in env if e.startswith(b"AUTOSCALER_ENABLED=")]
    return active != "active" and all(e.split(b"=", 1)[1].strip().lower() in (b"0", b"false", b"no", b"off", b"") for e in enabled)


def compose_env(template_env, revision, tickets, session, secret):
    """Launch environment exactly as the locked-boot autoscaler path composes it."""
    env = dict(template_env)
    env.update({
        "MUSETALK_SOURCE_REVISION": revision,
        "MUSETALK_DEPENDENCY_TICKETS_JSON": json.dumps(tickets, separators=(",", ":")),
        "AWS_ACCESS_KEY_ID": session["AccessKeyId"], "AWS_SECRET_ACCESS_KEY": session["SecretAccessKey"],
        "AWS_SESSION_TOKEN": session["SessionToken"], "AWS_DEFAULT_REGION": "us-east-1", "AWS_REGION": "us-east-1",
        "MUSETALK_RUNTIME_CONFIG_SOURCE": "injected",
        "TRT_ARTIFACT_S3_BUCKET": secret["AVATAR_S3_BUCKET"], "TRT_ARTIFACT_S3_REGION": "us-east-1",
        "PYTHONUNBUFFERED": "1",
    })
    env.update({k: str(secret[k]) for k in ("AVATAR_S3_BUCKET", "AVATAR_S3_REGION", "AVATAR_S3_PREFIX", "AVATAR_S3_ENABLED") if secret.get(k)})
    for key in ("MUSETALK_AWS_SECRET_ID", "LINGUA_CONTROL_PLANE_BASE_URL", "LINGUA_WORKER_TOKEN", "VENV_PATH", "SETUP_CLEAN"):
        require(key not in env, "Template environment must not carry " + key)
    return env


def choose_offer(rows, b, machine=None):
    offers = [r for r in rows if b.acceptable(r, allow_unverified=False)
              and (machine is None or r.get("machine_id") == machine)
              and str(r.get("id")) not in PROTECTED]
    require(offers, "No offer inside caps (verified US RTX 3090, <=$0.30/h, transfer <=$0.0015/GB)")
    return min(offers, key=lambda r: (r["dph_total_adj"], r["id"]))


def active_runs():
    count = 0
    for d in STATE.glob(LABEL_PREFIX + "*"):
        try:
            if json.loads((d / "ledger.json").read_text()).get("state") in {"submitting", "created"}:
                count += 1
        except (OSError, ValueError):
            continue
    return count


def expiry_timer(run, deadline):
    when = dt.datetime.fromisoformat(deadline).strftime("%Y-%m-%d %H:%M:%S UTC")
    subprocess.run(["systemd-run", "--quiet", "--unit=" + run.label + "-expiry", "--on-calendar=" + when,
                    "--timer-property=AccuracySec=1s", "--property=Restart=on-failure", "--property=RestartSec=30s",
                    "/home/ec2-user/lingua/venv/bin/python", "-B", str(Path(__file__).resolve()), "expire", run.name],
                   check=True, capture_output=True, timeout=15)
    require(subprocess.run(["systemctl", "is-active", run.label + "-expiry.timer"], capture_output=True,
                           timeout=10).returncode == 0, "Expiry timer not active")


def issue_tickets(contract):
    import boto3
    from botocore.config import Config
    sys.path.insert(0, str(HERE))
    import issue_dependency_read_tickets as issuer
    s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version="s3v4"))
    return issuer.issue(contract, s3, boto3.client("sts").get_caller_identity(), TICKET_SECONDS)


def runtime_identity(seconds):
    import boto3
    secret = json.loads(boto3.client("secretsmanager", region_name="us-east-1").get_secret_value(SecretId=RUNTIME_SECRET)["SecretString"])
    session = boto3.Session(aws_access_key_id=secret["AWS_ACCESS_KEY_ID"], aws_secret_access_key=secret["AWS_SECRET_ACCESS_KEY"], region_name="us-east-1")
    require(session.client("sts").get_caller_identity()["Arn"].endswith(RUNTIME_USER_SUFFIX), "Unexpected worker read identity")
    credentials = session.client("sts").get_session_token(DurationSeconds=min(max(seconds, 900), 129600))["Credentials"]
    return credentials, {k: v for k, v in secret.items() if not k.startswith("AWS_")}


def create(run, revision, lifetime, disk, machine):
    require(os.getuid() == 0 and not run.root.exists(), "Fresh root-owned run only; never repeat create")
    require(3600 <= lifetime <= 4 * 3600, "Lifetime must be 1-4 hours")
    require(autoscaler_quiet(), "Autoscaler is not provably off; refuse test rental with the production template hash")
    require(active_runs() < MAX_ACTIVE_RUNS, "Too many active test runs")
    os.umask(0o077)
    b = library(run)
    bootstrap = fetch_revision_file(revision, BOOTSTRAP_PATH)
    subprocess.run(["bash", "-n"], input=bootstrap, check=True, capture_output=True)
    require(not re.search(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b", bootstrap), "Credential literal in bootstrap")
    bootstrap_sha256 = hashlib.sha256(bootstrap).hexdigest()
    onstart = onstart_stub(revision, bootstrap_sha256)
    subprocess.run(["bash", "-n"], input=onstart.encode(), check=True, capture_output=True)
    contract = json.loads(fetch_revision_file(revision, CONTRACT_PATH))
    client = b.bootstrap()
    import requests
    existing = client.list_instances()
    require(not any(r.get("label") == run.label for r in existing), "Existing label; reconcile, do not create")
    rows = client.search_offers({"limit": 200, "type": "ondemand", "rentable": {"eq": True}, "num_gpus": {"eq": 1},
        "gpu_name": {"eq": "RTX 3090"}, "allocated_storage": disk, "geolocation": {"in": ["US"]},
        "dph_total": {"lte": .30}, "inet_down_cost": {"lte": .0015}, "inet_up_cost": {"lte": .0015},
        "verified": {"eq": True}, "order": [["dph_total", "asc"]]})
    b.DISK = disk
    offer = choose_offer(rows, b, machine)
    response = requests.get(client.base_url + "/api/v0/template/", headers=client._headers(),
        params={"select_filters": json.dumps({"hash_id": {"eq": b.TEMPLATE}})}, timeout=20, allow_redirects=False)
    require(response.status_code == 200, "Template lookup failed")
    templates = [r for r in response.json()["templates"] if r.get("hash_id") == b.TEMPLATE]
    require(len(templates) == 1, "Ambiguous stock template")
    template_env, _, _ = b.frozen_template(templates[0])
    tickets = issue_tickets(contract)
    credentials, secret = runtime_identity(lifetime + 1800)
    env = compose_env(template_env, revision, tickets, credentials, secret)
    run.root.mkdir(mode=0o700, parents=False)
    keys = ("id", "machine_id", "host_id", "geolocation", "verification", "gpu_name", "num_gpus", "cpu_cores_effective",
            "cpu_ram", "disk_bw", "disk_space", "storage_cost", "inet_down", "inet_up", "inet_down_cost", "inet_up_cost",
            "dph_total", "dph_total_adj", "reliability", "cuda_max_good", "driver_version")
    run.save("offer.json", {k: offer.get(k) for k in keys})
    deadline = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=lifetime)).replace(microsecond=0).isoformat()
    ledger = {"label": run.label, "state": "initialized", "instance_id": None, "machine_id": offer["machine_id"],
              "protected_instance_ids": sorted(PROTECTED | {str(r["id"]) for r in existing}),
              "deadline_utc": deadline, "lifetime_seconds": lifetime, "source_revision": revision,
              "onstart_sha256": hashlib.sha256(onstart.encode()).hexdigest(), "bootstrap_sha256": bootstrap_sha256,
              "contract_artifact_id": contract["artifact_id"], "template_hash_id": b.TEMPLATE, "disk_gb": disk,
              "tickets_issued_utc": utc(), "ticket_seconds": TICKET_SECONDS,
              "s3_session_expires_utc": credentials["Expiration"].isoformat(), "create_requests": 0,
              "rent_per_hour": offer["dph_total_adj"], "env_keys": sorted(env)}
    run.save("ledger.json", ledger)
    expiry_timer(run, deadline)
    ledger.update(state="submitting", create_requests=1, request_started_utc=utc())
    run.save("ledger.json", ledger)
    run.event("T0_CREATE_REQUESTED", offer_id=offer["id"], machine_id=offer["machine_id"], geolocation=offer.get("geolocation"),
              inet_down=offer.get("inet_down"), rent_per_hour=offer["dph_total_adj"], revision=revision)
    started = time.monotonic()
    request = {"client_id": "me", "template_hash_id": b.TEMPLATE, "image": b.IMAGE, "disk": disk,
               "label": run.label, "env": env, "onstart": onstart, "cancel_unavail": True}
    result = client.create_instance(offer_id=int(offer["id"]), create_request=request)
    identifier = str(result.get("new_contract") or result.get("contract") or result.get("instance_id") or "")
    require(result.get("success") is True and identifier.isdigit() and identifier not in ledger["protected_instance_ids"],
            "Create uncertain: reconcile only")
    ledger.update(state="created", instance_id=identifier, provider_accepted_utc=utc(), create_seconds=time.monotonic() - started)
    run.save("ledger.json", ledger)
    run.event("CREATED", instance_id=identifier, deadline_utc=deadline)


def owned(run, b, client, ledger):
    """The run's own rented RTX 3090 only. Production (51074906, RTX 4070 SUPER)
    and every instance that existed before this run's create are refused."""
    b.PROTECTED = next(iter(PROTECTED))
    row = b.owned(client, ledger)
    if row is not None:
        require(str(row.get("id")) not in PROTECTED and str(row.get("id")) not in ledger.get("protected_instance_ids", [])
                and row.get("label") == run.label and row.get("gpu_name") == "RTX 3090",
                "Refusing to act on a non-owned or production instance")
    return row


def status(run):
    ledger = run.load()
    b = library(run)
    client = b.bootstrap()
    row = owned(run, b, client, ledger)
    if row is None:
        print(json.dumps({"owned_state": "absent", "ledger_state": ledger["state"]}), flush=True)
        return None
    if ledger.get("instance_id") is None:
        ledger.update(instance_id=str(row["id"]), state="created", reconciled_without_create_retry=True)
        run.save("ledger.json", ledger)
    if row.get("actual_status") == "running" and not ledger.get("first_running_utc"):
        ledger["first_running_utc"] = utc()
        run.save("ledger.json", ledger)
        run.event("T1_PROVIDER_RUNNING", start_date=row.get("start_date"))
    if row.get("actual_status") == "running" and not ledger.get("ssh_key_attached"):
        import requests
        key = Path("/home/ec2-user/.ssh/id_ed25519.pub").read_text().strip()
        require(key.startswith("ssh-ed25519 ") and "\n" not in key, "Unexpected public key")
        response = requests.post(client.base_url + "/api/v0/instances/" + str(row["id"]) + "/ssh/",
                                 headers=client._headers(), json={"ssh_key": key}, timeout=15, allow_redirects=False)
        require(response.status_code == 200 and response.json().get("success") is True, "Owned SSH attach failed")
        ledger["ssh_key_attached"] = True
        run.save("ledger.json", ledger)
    safe = {k: row.get(k) for k in ("id", "label", "machine_id", "gpu_name", "actual_status", "cur_state", "intended_status",
                                  "status_msg", "image_runtype", "public_ipaddr", "ports", "ssh_host", "ssh_port", "start_date",
                                  "inet_down", "inet_up", "cpu_cores_effective", "gpu_totalram", "driver_version")}
    run.save("provider-latest.json", safe)
    print(json.dumps({"owned_state": {k: safe[k] for k in ("id", "actual_status", "cur_state", "status_msg")},
                      "deadline_utc": ledger["deadline_utc"]}), flush=True)
    return row


def ssh_arguments(run, row, command="python3 -B -"):
    host = row.get("ssh_host")
    port = int(row.get("ssh_port", 0)) + (1 if "jupyter" in str(row.get("image_runtype", "")) else 0)
    require(host and re.fullmatch(r"[A-Za-z0-9.-]+", host) and 1 <= port <= 65535, "Invalid owned SSH endpoint")
    return ["ssh", "-i", "/home/ec2-user/.ssh/id_ed25519", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=3", "-o", "StrictHostKeyChecking=accept-new",
            "-o", "UserKnownHostsFile=" + str(run.root / "known_hosts"), "-p", str(port), "root@" + host, command]


def worker(run, payload, timeout=60):
    ledger = run.load()
    require(ledger["state"] == "created" and dt.datetime.now(dt.timezone.utc)
            < dt.datetime.fromisoformat(ledger["deadline_utc"]) - dt.timedelta(seconds=120), "No live lease")
    b = library(run)
    row = owned(run, b, b.bootstrap(), ledger)
    require(row and row.get("actual_status") == "running" and ledger.get("ssh_key_attached"), "Owned worker not SSH-ready")
    r = subprocess.run(ssh_arguments(run, row), input=payload.encode(), capture_output=True, timeout=timeout)
    require(r.returncode == 0, "Owned SSH dispatch failed (rc %d)" % r.returncode)
    return r.stdout


LOG_TAIL = r'''
import json,pathlib,shutil,datetime,urllib.request
def tail(p,n=400000):
 p=pathlib.Path(p)
 if not p.is_file():return ''
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-n));return f.read().decode(errors='replace')
health=None
try:
 with urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=2) as r:
  b=json.loads(r.read(256000));health={'status':r.status,'ok':b.get('ok')}
except Exception:pass
up=float(pathlib.Path('/proc/uptime').read_text().split()[0])
print(json.dumps({'worker_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'uptime_s':up,'health':health,
 'bootstrap':tail('/workspace/bootstrap.log'),'onstart':tail('/workspace/onstart.log'),
 'free_disk_bytes':shutil.disk_usage('/workspace').free}))
'''

_BOOT = re.compile(r"^\[bootstrap\] \[(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ)\] (.*)$")
_HMS = re.compile(r"^\[([a-z_]+\.sh)\] \[(\d\d:\d\d:\d\d)\] (.*)$")
_STEP = re.compile(r"^\[([a-z_]+\.sh)\]\[(PHASE|STEP)\]\[(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)([+-]\d{4})\] (START|DONE): (.*)$")
_MARK = re.compile(r"^VAST_ONSTART (BEGIN|COMPLETE|FAILED): (\d{4}-\d\d-\d\d \d\d:\d\d:\d\d) UTC")


def timeline(bootstrap_text, onstart_text):
    """Absolute stage times for the LAST boot in append-only worker logs."""
    lines = bootstrap_text.splitlines()
    starts = [i for i, v in enumerate(lines) if _BOOT.match(v) and _BOOT.match(v).group(2).startswith("start")]
    require(starts, "No bootstrap start marker")
    boot = lines[starts[-1]:]
    t0 = dt.datetime.strptime(_BOOT.match(boot[0]).group(1), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    stages = [{"stage": "bootstrap_start", "at": t0}]
    for v in boot[1:]:
        m = _BOOT.match(v)
        if m:
            when = dt.datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
            text = m.group(2)
            name = ("code_ready" if text.startswith("Code ready") else "bootstrap_complete" if text.startswith("Complete")
                    else "bootstrap_failed" if text.startswith(("Failed", "ERROR")) else None)
            if name:
                stages.append({"stage": name, "at": when, "detail": text[:200]})
    onstart = onstart_text.splitlines()
    begins = [i for i, v in enumerate(onstart) if _MARK.match(v) and _MARK.match(v).group(1) == "BEGIN"
              and dt.datetime.strptime(_MARK.match(v).group(2), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc) >= t0]
    if begins:
        section = onstart[begins[0]:]
        last = None
        for v in section:
            m = _MARK.match(v)
            if m:
                last = dt.datetime.strptime(m.group(2), "%Y-%m-%d %H:%M:%S").replace(tzinfo=dt.timezone.utc)
                stages.append({"stage": "onstart_" + m.group(1).lower(), "at": last})
                continue
            m = _STEP.match(v)
            if m and m.group(2) == "PHASE" and m.group(5) == "DONE":
                when = dt.datetime.strptime(m.group(3) + m.group(4), "%Y-%m-%dT%H:%M:%S%z").astimezone(dt.timezone.utc)
                stages.append({"stage": "install:" + m.group(6)[:80], "at": when})
                last = when
                continue
            m = _HMS.match(v)
            if m and "phase finished in" in m.group(3) and last is not None:
                h, mi, s = map(int, m.group(2).split(":"))
                when = last.replace(hour=h, minute=mi, second=s)
                if when < last - dt.timedelta(hours=12):
                    when += dt.timedelta(days=1)
                stages.append({"stage": "onstart:" + m.group(3).split(" phase finished in")[0][:80], "at": when,
                               "detail": m.group(3)[-40:]})
                last = when
    out = []
    for s in sorted(stages, key=lambda s: s["at"]):
        out.append({**s, "at": s["at"].isoformat(), "since_bootstrap_s": round((s["at"] - t0).total_seconds(), 1)})
    return out


def probe(run):
    value = json.loads(worker(run, LOG_TAIL))
    stages = []
    try:
        stages = timeline(value["bootstrap"], value["onstart"])
    except ValueError as error:
        stages = [{"stage": "unparsed", "detail": str(error)}]
    summary = {"worker_utc": value["worker_utc"], "uptime_s": value["uptime_s"], "health": value["health"],
               "free_disk_gb": round(value["free_disk_bytes"] / 1e9, 1), "stages": stages}
    run.save("probe-latest.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


def reboot(run):
    """Provider-level container restart: On-start runs again with the launch env."""
    ledger = run.load()
    b = library(run)
    client = b.bootstrap()
    row = owned(run, b, client, ledger)
    require(row and row.get("actual_status") == "running", "Owned worker not running")
    import requests
    run.event("REBOOT_REQUESTED", instance_id=str(row["id"]))
    response = requests.put(client.base_url + "/api/v0/instances/reboot/" + str(row["id"]) + "/",
                            headers=client._headers(), json={}, timeout=30, allow_redirects=False)
    body = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
    require(response.status_code == 200 and body.get("success") is True, "Reboot not acknowledged (%d)" % response.status_code)
    run.event("REBOOT_ACKNOWLEDGED")


RERUN = r'''
import json,os,pathlib,shutil,subprocess,time
REVISION=__REVISION__;TICKETS=__TICKETS__;WIPE=__WIPE__;ONSTART=__ONSTART__;PATHS=__PATHS__
env={}
for item in pathlib.Path('/proc/1/environ').read_bytes().split(b'\0'):
 if b'=' in item:
  k,v=item.split(b'=',1);env[k.decode()]=v.decode(errors='replace')
env.update({'MUSETALK_SOURCE_REVISION':REVISION,'MUSETALK_DEPENDENCY_TICKETS_JSON':TICKETS})
ctl=pathlib.Path('/workspace/MuseTalk/scripts/vast_server_ctl.sh')
if ctl.is_file():
 subprocess.run(['bash',str(ctl),'stop'],cwd='/workspace/MuseTalk',capture_output=True,timeout=120)
subprocess.run(['pkill','-f','turnserver'],capture_output=True)
subprocess.run(['pkill','-f','api_server.py'],capture_output=True)
time.sleep(2)
removed=[]
if WIPE:
 for p in PATHS:
  q=pathlib.Path(p)
  if q.is_symlink() or q.is_file():q.unlink();removed.append(p)
  elif q.is_dir():shutil.rmtree(q);removed.append(p)
script=pathlib.Path('/root/mtboot-onstart.sh');script.write_text(ONSTART);script.chmod(0o700)
log=open('/workspace/mtboot-rerun.out','ab')
p=subprocess.Popen(['bash',str(script)],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True,cwd='/')
print(json.dumps({'pid':p.pid,'removed':removed}))
'''


def rerun(run, revision=None, wipe=False):
    """Re-execute the pinned bootstrap over SSH with fresh tickets (optional wipe).

    Mirrors a fresh/updated On-start without waiting for provider reboot and
    without the one-hour ticket limit of the original launch environment."""
    ledger = run.load()
    revision = revision or ledger["source_revision"]
    onstart = fetch_revision_file(revision, BOOTSTRAP_PATH).decode()
    subprocess.run(["bash", "-n"], input=onstart.encode(), check=True, capture_output=True)
    contract = json.loads(fetch_revision_file(revision, CONTRACT_PATH))
    tickets = json.dumps(issue_tickets(contract), separators=(",", ":"))
    payload = (RERUN.replace("__REVISION__", repr(revision)).replace("__TICKETS__", repr(tickets))
               .replace("__WIPE__", repr(bool(wipe))).replace("__ONSTART__", repr(onstart)).replace("__PATHS__", repr(WIPE_PATHS)))
    run.event("RERUN_REQUESTED", revision=revision, wipe=bool(wipe), onstart_sha256=hashlib.sha256(onstart.encode()).hexdigest())
    result = json.loads(worker(run, payload, timeout=180))
    ledger.setdefault("reruns", []).append({"at_utc": utc(), "revision": revision, "wipe": bool(wipe), "removed": result["removed"]})
    run.save("ledger.json", ledger)
    run.event("RERUN_STARTED", **result)


def accept(run, cases=2):
    """Health + install check + decoded REST videos through an SSH tunnel (not WebRTC/TURN)."""
    ledger = run.load()
    b = library(run)
    client = b.bootstrap()
    row = owned(run, b, client, ledger)
    require(row and row["actual_status"] == "running", "Not running")
    dest = run.root / ("acceptance-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S"))
    dest.mkdir(mode=0o700)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    ssh = ssh_arguments(run, row)
    command = ssh[:-1] + ["-N", "-o", "ExitOnForwardFailure=yes", "-L", f"127.0.0.1:{port}:127.0.0.1:8000"]
    command = [c for c in command if c != "python3 -B -"]
    tunnel, result = None, {"state": "running", "cases": []}
    try:
        with (dest / "tunnel.log").open("xb") as log:
            tunnel = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        base = f"http://127.0.0.1:{port}"
        until = time.monotonic() + 900
        while time.monotonic() < until:
            require(tunnel.poll() is None, "Forwarding process ended")
            code, body = b.probe(base, "/health")
            if code == 200 and body and body.get("ok") is True:
                break
            time.sleep(3)
        else:
            raise ValueError("API health deadline")
        result["health_ok_utc"] = utc()
        check = r'''import json,os,pathlib,subprocess
repo=pathlib.Path('/workspace/MuseTalk')
env=dict(os.environ);env.update({'MUSETALK_DEPENDENCY_INSTALL_MODE':'locked-wheelhouse-uv','MUSETALK_DEPENDENCY_CONTRACT':str(repo/'configs/dependencies/rtx3090-cp310-cu121-screening.json'),'MUSETALK_DEPENDENCY_CACHE':'/workspace/.cache/musetalk-dependencies'})
venv=sorted(pathlib.Path('/workspace/.venvs').glob('musetalk_locked_*'))
p=subprocess.run(['bash','scripts/install_musetalk.sh','--check','--with-avatar-prep','--json','--venv',str(venv[-1])],cwd=repo,env=env,capture_output=True,text=True,timeout=60)
print(json.dumps({'rc':p.returncode,'check':json.loads(p.stdout) if p.stdout.strip().startswith('{') else p.stdout[-2000:],'head':subprocess.run(['git','rev-parse','HEAD'],cwd=repo,capture_output=True,text=True).stdout.strip()}))
'''
        result["install_check"] = json.loads(worker(run, check, timeout=120))
        require(result["install_check"]["head"] == ledger["source_revision"] or ledger.get("reruns"), "Worker revision differs")
        audio = b.BOOTSTRAP.parent / "startup-speech.wav"
        require(hashlib.sha256(audio.read_bytes()).hexdigest() == "8628701b894babe55b8e04769363925528e50c1352d3edcebf1d31a44a65247e",
                "Different acceptance audio input")
        for n in range(cases):
            case = dest / ("output-" + str(n))
            case.mkdir(mode=0o700)
            old, start = b.ROOT, time.monotonic()
            try:
                b.ROOT = case
                b.SMOKE_PHASE_SECONDS = 300
                b.smoke(base, audio)
            finally:
                b.ROOT = old
            value = json.loads((case / "video-smoke.json").read_text())
            result["cases"].append({"case": n, "warm_generate_download_decode_seconds": round(time.monotonic() - start, 3),
                                    "first_request": n == 0, **value})
        result.update(state="passed", scope="REST generation through SSH tunnel; not WebRTC/TURN or registration")
    except BaseException as error:
        result.update(state="failed", error_type=type(error).__name__,
                      detail=str(error) if type(error) is ValueError else "Diagnostics retained")
    finally:
        if tunnel:
            tunnel.terminate()
            try:
                tunnel.wait(timeout=10)
            except subprocess.TimeoutExpired:
                tunnel.kill()
        (dest / "receipt.json").write_text(json.dumps(result, indent=2) + "\n")
    run.event("ACCEPTANCE_" + result["state"].upper(), receipt=str(dest / "receipt.json"),
              cases=[{k: c.get(k) for k in ("case", "warm_generate_download_decode_seconds", "video_sha256")} for c in result["cases"]])
    require(result["state"] == "passed", "Acceptance failed: " + str(result.get("detail")))


COLLECT = r'''import base64,gzip,hashlib,json,pathlib
files=[];budget=40*1024**2
paths=[pathlib.Path(p) for p in ('/workspace/bootstrap.log','/workspace/onstart.log','/workspace/mtboot-rerun.out','/workspace/logs/musetalk/api_server_8000.log','/workspace/MuseTalk/.runtime/gpu_selftest.json','/workspace/MuseTalk/.runtime/install_state.json')]
cache=pathlib.Path('/workspace/.cache/musetalk-dependencies')
if cache.is_dir():
 for d in cache.glob('*/install-*'):
  paths += [p for p in d.iterdir() if p.name in ('receipt.json','events.jsonl') or p.suffix=='.log']
setup=pathlib.Path('/workspace/MuseTalk/logs/setup')
if setup.is_dir():paths += sorted(setup.glob('*.log'))[-3:]
for p in paths:
 if not p.is_file() or p.is_symlink() or p.stat().st_size>8*1024**2 or p.stat().st_size>budget:continue
 data=p.read_bytes();budget-=len(data)
 name=str(p).strip('/').replace('/','__')
 files.append({'name':name,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'data':base64.b64encode(gzip.compress(data)).decode()})
print(json.dumps({'files':files}))
'''


def collect(run):
    import base64
    import gzip
    result = json.loads(worker(run, COLLECT, timeout=180))
    dest = run.root / ("evidence-" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S"))
    dest.mkdir(mode=0o700)
    inventory = []
    for f in result["files"]:
        require(re.fullmatch(r"[A-Za-z0-9+._-]+", f["name"]) and f["name"] not in {".", ".."}, "Unsafe evidence name")
        data = gzip.decompress(base64.b64decode(f["data"]))
        require(len(data) == f["bytes"] and hashlib.sha256(data).hexdigest() == f["sha256"], "Evidence checksum failed")
        (dest / f["name"]).write_bytes(data)
        inventory.append({k: f[k] for k in ("name", "bytes", "sha256")})
    run.save(dest.name + ".json", {"at_utc": utc(), "files": inventory})
    print(json.dumps({"state": "evidence-collected", "dir": str(dest), "files": len(inventory)}), flush=True)


def cleanup(run, expire=False):
    ledger = run.load()
    if expire:
        require(dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(ledger["deadline_utc"]), "Expiry too early")
    if ledger.get("state") == "initialized":
        subprocess.run(["systemctl", "stop", run.label + "-expiry.timer"], capture_output=True, timeout=10)
        print(json.dumps({"state": "never-submitted"}), flush=True)
        return
    b = library(run)
    client = b.bootstrap()
    owned(run, b, client, ledger)  # Refuses production/non-owned before any destroy.
    b.cleanup(client, ledger)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "status", "probe", "reboot", "rerun", "accept", "collect", "cleanup", "expire"))
    parser.add_argument("run")
    parser.add_argument("--revision")
    parser.add_argument("--lifetime", type=int, default=DEFAULT_LIFETIME)
    parser.add_argument("--disk", type=int, default=DEFAULT_DISK)
    parser.add_argument("--machine", type=int)
    parser.add_argument("--wipe", action="store_true")
    parser.add_argument("--cases", type=int, default=2)
    args = parser.parse_args()
    require(os.getuid() == 0, "Root-owned EC2 control plane only")
    os.umask(0o077)
    run = Run(args.run)
    if args.mode == "create":
        create(run, args.revision, args.lifetime, args.disk, args.machine)
    elif args.mode == "status":
        status(run)
    elif args.mode == "probe":
        probe(run)
    elif args.mode == "reboot":
        reboot(run)
    elif args.mode == "rerun":
        rerun(run, args.revision, args.wipe)
    elif args.mode == "accept":
        accept(run, args.cases)
    elif args.mode == "collect":
        collect(run)
    else:
        cleanup(run, expire=args.mode == "expire")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if isinstance(error, SystemExit):
            raise
        print(json.dumps({"state": "failed", "error_type": type(error).__name__,
                          "detail": str(error) if type(error) is ValueError else "Private diagnostics suppressed"}), flush=True)
        raise SystemExit(2)
