#!/usr/bin/env python3
"""One stock-template baseline; no installer changes or production registration.

Run only on EC2. Secrets stay in memory and in the provider's launch environment.
A separate systemd timer reconciles and destroys only this run's owned instance.
This measures API health and a decoded REST video, NOT EC2/TURN call acceptance.
"""
import argparse
import datetime as dt
import hashlib
import importlib.util
import ipaddress
import json
import logging
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import time
import uuid

LABEL = "musetalk-source-baseline-20261011-a1"
ROOT = Path("/home/ec2-user/.local/state") / LABEL
BOOTSTRAP = Path("/home/ec2-user/.local/state/musetalk-us-template-009788c8-tools-20261011/ec2_startup_runner.py")
TEMPLATE = "4dc0868ee56886c667c217d6c3634a58"
SOURCE = "5cc706e90e50e93da1310628c025a84199cd8042"
IMAGE = "vastai/pytorch:cuda-12.1-auto"
DISK = 150  # Actual EC2 autoscaler default, not the UI's 50-GB suggestion.
PROTECTED = "51074906"
TRANSFER_CAP = .0015
# Post-run fixture correction: A1/A2 receipts retain the original missing ID.
# Existing ledgers forbid replay; this does not authorize another paid attempt.
AVATAR = "en_01_maya_abe451d3_talking_f6cf8be7fb"
BUNDLE_KEY = "trt-artifacts/ampere-plus/r5-srcg50-int8/sha256-07644ff16a170ecbfb4eb3d39ff1150731d5508baff31e0216a5dc1bb371df2e/musetalk-trt-r5-ampere-plus.tar.gz"
SECRET_VALUES = []
START = None
ATTEMPT = "a1"
LIFETIME_SECONDS = 3600
SMOKE_PHASE_SECONDS = 300


def configure_attempt(name):
    global LABEL, ROOT, ATTEMPT
    require(name in {"a1", "a2"}, "Only the original and one bounded baseline retry are supported")
    ATTEMPT = name
    LABEL = "musetalk-source-baseline-20261011-" + name
    ROOT = Path("/home/ec2-user/.local/state") / LABEL


def require(ok, message):
    if not ok:
        raise ValueError(message)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(name, value):
    path = ROOT / name
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.chmod(tmp, 0o600)
    tmp.replace(path)


def event(stage, **fields):
    row = {"stage": stage, "at_utc": utc(), **fields}
    if START is not None:
        row["elapsed_seconds"] = round(time.monotonic() - START, 3)
    with (ROOT / "events.jsonl").open("a") as f:
        f.write(json.dumps(row, allow_nan=False) + "\n")
    print(json.dumps(row), flush=True)
    return row


def bootstrap():
    logging.disable(logging.CRITICAL)
    spec = importlib.util.spec_from_file_location("baseline_credentials", BOOTSTRAP)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.load_credentials()
    from services.vast_client import VastClient
    return VastClient()


def acceptable(o, allow_unverified=False):
    def finite(name, low, high):
        v = o.get(name)
        return type(v) in (int, float) and math.isfinite(v) and low <= v <= high
    prices = [o[k] for k in ("dph_total", "dph_total_adj", "discounted_total_per_hour") if o.get(k) is not None]
    return (o.get("gpu_name") == "RTX 3090" and o.get("num_gpus") == 1
            and o.get("cpu_arch") == "amd64"
            and o.get("verification") in ({"verified", "unverified"} if allow_unverified else {"verified"})
            and str(o.get("geolocation", "")).endswith(", US")
            and finite("cpu_cores_effective", 8 if allow_unverified else 10, 1000) and finite("cpu_ram", 32000, 10**9)
            and finite("disk_space", DISK, 10**9) and finite("inet_down", 500, 10**9)
            and finite("cuda_max_good", 12.1, 100) and finite("reliability", .90, 1)
            and finite("inet_down_cost", 0, TRANSFER_CAP) and finite("inet_up_cost", 0, TRANSFER_CAP)
            and finite("storage_cost", 0, .40) and bool(prices)
            and all(type(v) in (int, float) and math.isfinite(v) and 0 < v <= .30 for v in prices))


def select_offer(rows, offer_id, allow_unverified=False):
    matches = [o for o in rows if str(o.get("id")) == str(offer_id)]
    require(len(matches) == 1 and acceptable(matches[0], allow_unverified), "Offer unavailable or outside caps")
    return matches[0]


def frozen_template(t):
    require(t.get("hash_id") == TEMPLATE and t.get("image") == "vastai/pytorch"
            and t.get("tag") == "cuda-12.1-auto" and t.get("runtype") == "jupyter"
            and t.get("use_ssh") is True and t.get("jup_direct") is True
            and t.get("ssh_direct") is True, "Stock launch contract changed")
    raw_env = t["env"]
    require(isinstance(raw_env, str), "Unexpected template environment")
    # Saved UI options have twenty trailing whitespace/backslash/quote bytes.
    cut = re.search(r"-e\s+NVIDIA_VISIBLE_DEVICES=all", raw_env)
    require(cut and not re.sub(r"[\s\\\x22\x27]", "", raw_env[cut.end():]), "Unexpected trailing options")
    flags = shlex.split(raw_env[:cut.end()])
    require(len(flags) % 2 == 0, "Malformed option pairs")
    env = {}
    for flag, value in zip(flags[::2], flags[1::2]):
        if flag == "-p":
            if value == "72299:72299":
                continue
            require(re.fullmatch(r"\d{1,5}:\d{1,5}(?:/udp)?", value), "Unexpected port option")
            require(all(1 <= int(p) <= 65535 for p in value.split('/')[0].split(':')), "Invalid port")
            env["-p " + value] = "1"
        else:
            require(flag == "-e" and "=" in value, "Unexpected template flag")
            k, v = value.split("=", 1)
            require(k in {"OPEN_BUTTON_PORT", "OPEN_BUTTON_TOKEN", "JUPYTER_DIR", "DATA_DIRECTORY",
                          "PORTAL_CONFIG", "NVIDIA_DRIVER_CAPABILITIES", "NVIDIA_VISIBLE_DEVICES"}, "Unexpected environment key")
            env[k] = v
    original = t["onstart"]
    require("SETUP_CLEAN=1" in original and "SETUP_FULL_STACK=1" in original
            and 'REPO_BRANCH="main"' in original, "Baseline installer contract changed")
    stripped = []
    for line in original.splitlines():
        if re.match(r"\s*export\s+(?:AWS_(?:ACCESS_KEY_ID|SECRET_ACCESS_KEY|SESSION_TOKEN)|MUSETALK_AWS_SECRET_(?:ID|REGION))=", line):
            continue
        stripped.append(line)
    script = "\n".join(stripped) + "\n"
    script = script.replace('echo "[bootstrap] cloning git repo checkout"',
        'echo "[source-baseline] checkout_start $(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"\n'
        'echo "[bootstrap] cloning git repo checkout"', 1)
    marker = 'cd "$REPO_DIR"'
    require(script.count(marker) == 1, "Unique checkout marker required")
    script = script.replace(marker, marker + '\n'
        + 'test "$(git rev-parse HEAD)" = "' + SOURCE + '"\n'
        + 'echo "[source-baseline] checkout_ready $(date -u +%Y-%m-%dT%H:%M:%S.%NZ)"', 1)
    script = script.replace("set -Eeuo pipefail", "set -Eeuo pipefail\n"
        "unset MUSETALK_AWS_SECRET_ID MUSETALK_AWS_SECRET_REGION\n"
        "unset LINGUA_CONTROL_PLANE_BASE_URL LINGUA_WORKER_TOKEN LINGUA_WORKER_REGISTER_URL LINGUA_WORKER_HEARTBEAT_URL", 1)
    require(not re.search(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b", script), "Credential literal remains")
    subprocess.run(["bash", "-n"], input=script.encode(), check=True, capture_output=True)
    return env, script, {
        "hash_id": TEMPLATE, "image": IMAGE, "runtype": "jupyter_direc ssh_direc ssh_proxy",
        "disk_gb": DISK, "template_recommended_disk_gb": t.get("recommended_disk_space"),
        "original_onstart_sha256": hashlib.sha256(original.encode()).hexdigest(),
        "run_onstart_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "source_revision": SOURCE, "setup_clean": 1, "setup_full_stack": 1,
        "changes": ["remove invalid port 72299", "remove malformed trailing Docker-option bytes",
                    "replace plaintext AWS bootstrap with expiring EC2-mediated S3 session",
                    "pin/check source main SHA", "two checkout timestamps", "no production callbacks"],
        "registration_and_ec2_turn_call": "NOT_TESTED_NO_ISOLATED_PRODUCTION_ROUTE",
        "custom_docker_used": False, "production_template_modified": False,
        "environment_keys": sorted(env), "ports": sorted(k[3:] for k in env if k.startswith("-p ")),
    }


def owned(client, ledger):
    rows = client.list_instances()
    matches = [r for r in rows if r.get("label") == LABEL]
    if not matches:
        require(not ledger.get("instance_id") or not any(str(r.get("id")) == ledger["instance_id"] for r in rows), "Owned worker relabelled")
        return None
    require(len(matches) == 1, "Ambiguous owned label")
    r = matches[0]
    i = str(r.get("id", ""))
    require(i.isdigit() and i != PROTECTED and i not in ledger["protected_instance_ids"], "Protected worker")
    require(r.get("gpu_name") == "RTX 3090" and r.get("num_gpus") == 1, "Wrong GPU ownership")
    require(ledger.get("instance_id") in (None, i), "ID differs from ledger")
    require(r.get("machine_id") == ledger["machine_id"], "Machine differs from ledger")
    return r


def cleanup(client, ledger):
    require(ledger.get("label") == LABEL and ledger.get("state") in {"submitting", "created", "cleaned"}, "No create intent")
    row = owned(client, ledger)
    if row:
        i = str(row["id"])
        ledger["instance_id"] = i
        save("ledger.json", ledger)
        response = client.destroy_instance(i)
        require(response.get("success") is True, "Destroy not acknowledged")
    for _ in range(6):
        after = client.list_instances()
        absent = not any(r.get("label") == LABEL or str(r.get("id")) == ledger.get("instance_id") for r in after)
        if absent:
            break
        time.sleep(2)
    receipt = {"at_utc": utc(), "instance_id": ledger.get("instance_id"), "provider_absence_verified": absent,
               "protected_production_running": any(str(r.get("id")) == PROTECTED and r.get("actual_status") == "running" for r in after)}
    save("cleanup.json", receipt)
    require(absent, "Provider absence not verified")
    ledger["state"] = "cleaned"
    save("ledger.json", ledger)
    event("CLEANUP_VERIFIED", **receipt)
    subprocess.run(["systemctl", "stop", LABEL + "-expiry.timer"], capture_output=True, timeout=10)


def timer(deadline):
    when = dt.datetime.fromisoformat(deadline).strftime("%Y-%m-%d %H:%M:%S UTC")
    command = ["systemd-run", "--quiet", "--unit=" + LABEL + "-expiry", "--on-calendar=" + when,
               "--timer-property=AccuracySec=1s", "--property=Restart=on-failure", "--property=RestartSec=30s",
               "/home/ec2-user/lingua/venv/bin/python", "-B", str(Path(__file__).resolve()), "expire", "--attempt", ATTEMPT]
    subprocess.run(command, check=True, capture_output=True, timeout=15)
    require(subprocess.run(["systemctl", "is-active", LABEL + "-expiry.timer"], capture_output=True).returncode == 0, "Expiry timer not active")


def endpoint(row):
    ip = ipaddress.ip_address(row["public_ipaddr"])
    require(ip.version == 4 and ip.is_global, "Public worker IP required")
    ports = {p["HostPort"] for p in row.get("ports", {}).get("8000/tcp", [])}
    if not ports:
        return None
    require(len(ports) == 1, "Ambiguous API mapping")
    port = ports.pop()
    require(str(port).isdigit() and 1 <= int(port) <= 65535, "Bad API port")
    return f"http://{ip}:{port}"


def scrub(text):
    for value in SECRET_VALUES:
        text = text.replace(value, "[REDACTED]")
    return re.sub(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b|\bgh[pousr]_[A-Za-z0-9]+\b", "[REDACTED]", text)


def check_avatar_fixture(s3, bucket, prefix):
    key = "/".join(p for p in (prefix.strip("/"), "v15", AVATAR + ".tar.gz") if p)
    head = s3.head_object(Bucket=bucket, Key=key)
    size = head.get("ContentLength")
    require(head.get("ResponseMetadata", {}).get("HTTPStatusCode") == 200
            and type(size) is int and size > 0, "Avatar fixture missing/empty; do not create")
    return {"avatar_id": AVATAR, "object_key": key, "bytes": size, "http_status": 200}


def save_probe_receipt(name, code, body):
    # Preserve an actionable failure before cleanup; redact injected credentials.
    save(name, {"http_status": code, "body": json.loads(scrub(json.dumps(body)))})


WORKER_SNAPSHOT = r'''
import datetime,json,pathlib,subprocess,shutil,time
def cmd(a):
 try:return subprocess.check_output(a,stderr=subprocess.DEVNULL,timeout=8,text=True).strip()
 except Exception:return None
files=['/workspace/bootstrap.log','/workspace/onstart.log','/workspace/logs/musetalk/api_server_8000.log']
logs={}
for f in files:
 p=pathlib.Path(f)
 if p.is_file():
  with p.open('rb') as s:
   s.seek(max(0,p.stat().st_size-1600000));logs[f]=s.read().decode('utf-8','replace')
git_files=[]
for stage in pathlib.Path('/workspace').glob('MuseTalk.stage.*'):
 for p in (stage/'.git/objects/pack').glob('*'):
  try:git_files.append({'name':p.name,'bytes':p.stat().st_size})
  except FileNotFoundError:pass
print(json.dumps({'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'worker_monotonic':time.monotonic(),'source_revision':cmd(['git','-C','/workspace/MuseTalk','rev-parse','HEAD']),
 'gpu':cmd(['nvidia-smi','--query-gpu=name,driver_version,power.limit,memory.total','--format=csv,noheader']),
 'disk_free_bytes':shutil.disk_usage('/workspace').free,'git_pack_files':git_files,'logs':logs}))
'''


def snapshot(row):
    host, port = row.get("ssh_host"), row.get("ssh_port")
    direct_ports = {p.get("HostPort") for p in (row.get("ports") or {}).get("22/tcp", [])}
    if row.get("public_ipaddr") and len(direct_ports) == 1:
        address = ipaddress.ip_address(row["public_ipaddr"])
        require(address.version == 4 and address.is_global, "Bad direct SSH IP")
        host, port = str(address), direct_ports.pop()
    if not host or not port:
        return False
    require(re.fullmatch(r"[A-Za-z0-9.-]+", host) and str(port).isdigit(), "Bad SSH endpoint")
    argv = ["ssh", "-i", "/home/ec2-user/.ssh/id_ed25519", "-o", "BatchMode=yes", "-o", "ConnectTimeout=3",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "UserKnownHostsFile=" + str(ROOT / "known_hosts"),
            "-p", str(port), "root@" + host, "python3 -B -"]
    try:
        r = subprocess.run(argv, input=WORKER_SNAPSHOT.encode(), capture_output=True, timeout=18)
    except subprocess.TimeoutExpired:
        return False
    if r.returncode:
        return False
    value = json.loads(scrub(r.stdout.decode()))
    save("worker-latest.json", value)
    sample = {k: value.get(k) for k in ("at_utc", "worker_monotonic", "source_revision", "git_pack_files", "disk_free_bytes")}
    with (ROOT / "worker-samples.jsonl").open("a") as out:
        out.write(json.dumps(sample) + "\n")
    for name, contents in value["logs"].items():
        target = ROOT / ("worker-" + Path(name).name)
        target.write_text(contents)
    return value


def probe(base, path, **kwargs):
    import requests
    try:
        r = requests.request(kwargs.pop("method", "GET"), base + path, timeout=kwargs.pop("timeout", 3), allow_redirects=False, **kwargs)
        if r.status_code not in {200, 202}:
            return r.status_code, None
        return r.status_code, r.json()
    except (requests.RequestException, ValueError):
        return 0, None


def smoke(base, audio):
    before = time.monotonic()
    code, body = probe(base, f"/avatars/{AVATAR}/cache/warm?wait=false&batch_size=16", method="POST", timeout=10)
    save_probe_receipt("avatar-warm-request.json", code, body)
    require(code in {200, 202} and body and body.get("status") != "failed", "Avatar warm rejected")
    until = time.monotonic() + SMOKE_PHASE_SECONDS
    while time.monotonic() < until:
        code, body = probe(base, f"/avatars/{AVATAR}/cache/status")
        save_probe_receipt("avatar-warm-status.json", code, body)
        if code == 200 and body and body.get("status") == "ready":
            break
        require(not body or body.get("status") not in {"failed", "error"}, "Avatar warm failed")
        time.sleep(3)
    else:
        raise ValueError("Avatar warm deadline")
    event("AVATAR_READY", avatar_id=AVATAR, restore_warm_seconds=round(time.monotonic()-before,3), scope="one_existing_avatar_not_full_roster")
    # The pinned manager reports None.mp4 when output_name is omitted, despite
    # passing a random fallback to inference. Name this test output explicitly.
    output_name = 'startup_' + uuid.uuid4().hex[:12]
    code, body = probe(base, f"/generate?avatar_id={AVATAR}&batch_size=16&fps=25&output_name={output_name}", method="POST",
                       files={"audio_file": ("speech.wav", audio.read_bytes(), "audio/wav")}, timeout=20)
    save_probe_receipt("generation-request.json", code, body)
    require(code == 200 and body and re.fullmatch(r"[A-Za-z0-9_-]{1,128}", str(body.get("request_id", ""))), "Generation rejected")
    i = body["request_id"]
    event("GENERATION_ACCEPTED", output_name=output_name)
    until = time.monotonic() + SMOKE_PHASE_SECONDS
    while time.monotonic() < until:
        code, body = probe(base, f"/generate/{i}/status")
        save_probe_receipt("generation-status.json", code, body)
        if body and body.get("status") == "completed":
            require(body.get("result", {}).get("success") is True, "Generation unsuccessful")
            break
        require(not body or body.get("status") not in {"failed", "error"}, "Generation failed")
        time.sleep(3)
    else:
        raise ValueError("Generation deadline")
    import requests
    r = requests.get(base + f"/generate/{i}/download", timeout=30, allow_redirects=False)
    require(r.status_code == 200 and 0 < len(r.content) <= 64*1024**2, "Bad video download")
    video = ROOT / "startup-video-smoke.mp4"
    video.write_bytes(r.content)
    subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-i", str(video), "-map", "0:v:0", "-map", "0:a:0", "-f", "null", "-"], capture_output=True, check=True, timeout=45)
    result = subprocess.check_output(["ffprobe", "-v", "error", "-count_frames", "-show_entries",
        "stream=codec_type,codec_name,width,height,nb_read_frames,avg_frame_rate,duration", "-of", "json", str(video)], timeout=20)
    info = json.loads(result)
    require(any(s.get("codec_type") == "video" and int(s.get("nb_read_frames",0)) > 0 for s in info["streams"])
            and any(s.get("codec_type") == "audio" for s in info["streams"]), "Both decoded tracks required")
    receipt = {"avatar_id": AVATAR, "streams": info["streams"], "video_sha256": hashlib.sha256(r.content).hexdigest(),
               "input_audio_sha256": hashlib.sha256(audio.read_bytes()).hexdigest(), "video_bytes": len(r.content),
               "scope": "EC2_to_owned_worker_REST_video_not_WebRTC_TURN_or_registration"}
    save("video-smoke.json", receipt)
    event("VIDEO_SMOKE_PASS", **receipt)


def run(offer_id, allow_unverified=False):
    global START, SECRET_VALUES
    require(os.getuid() == 0 and not ROOT.exists(), "Fresh root-owned run only; never repeat create")
    os.umask(0o077)
    audio = BOOTSTRAP.parent / "startup-speech.wav"
    require(audio.is_file() and audio.stat().st_size < 4*1024**2 and shutil.which("ffmpeg") and shutil.which("ffprobe"), "Probe fixtures/tools missing")
    client = bootstrap()
    import boto3, requests
    rows = client.list_instances()
    require(not any(r.get("label") == LABEL for r in rows), "Existing run label")
    if ATTEMPT == "a2":
        first = json.loads((ROOT.with_name("musetalk-source-baseline-20261011-a1") / "cleanup.json").read_text())
        require(first.get("provider_absence_verified") is True
                and not any(str(r.get("id")) == str(first.get("instance_id")) for r in rows), "A1 absence must be verified before A2")
    # The provider returned an empty id-filtered response while the exact same
    # offer was present in its full search. Bind by exact ID in reviewed rows,
    # never infer absence from that unreliable server-side filter.
    offers = client.search_offers({"limit": 100, "type": "ondemand", "rentable": {"eq": True},
        "num_gpus": {"eq": 1}, "gpu_name": {"eq": "RTX 3090"}, "allocated_storage": DISK,
        "dph_total": {"lte": .30}, "inet_down_cost": {"lte": TRANSFER_CAP},
        "inet_up_cost": {"lte": TRANSFER_CAP}, "order": [["dph_total", "asc"]]})
    offer = select_offer(offers, offer_id, allow_unverified)
    offer_id = int(offer["id"])  # Resolve current equivalent slot ID before the one create request.
    r = requests.get(client.base_url+"/api/v0/template/", headers=client._headers(),
                     params={"select_filters": json.dumps({"hash_id": {"eq": TEMPLATE}})}, timeout=20, allow_redirects=False)
    require(r.status_code == 200, "Template unavailable")
    templates = [t for t in r.json()["templates"] if t.get("hash_id") == TEMPLATE]
    require(len(templates) == 1, "Ambiguous template")
    env, script, frozen = frozen_template(templates[0])
    secret = json.loads(boto3.client("secretsmanager", region_name="us-east-1").get_secret_value(SecretId="lingua/musetalk-worker-runtime")["SecretString"])
    session = boto3.Session(aws_access_key_id=secret["AWS_ACCESS_KEY_ID"], aws_secret_access_key=secret["AWS_SECRET_ACCESS_KEY"], region_name="us-east-1")
    require(session.client("sts").get_caller_identity()["Arn"].endswith(":user/musetalk-s3-runtime"), "Unexpected worker identity")
    require(session.client("s3").head_object(Bucket=secret["AVATAR_S3_BUCKET"], Key=BUNDLE_KEY)["ContentLength"] == 1000524577, "Bundle unavailable")
    fixture = check_avatar_fixture(session.client("s3"), secret["AVATAR_S3_BUCKET"], secret.get("AVATAR_S3_PREFIX", "avatars"))
    require(LIFETIME_SECONDS in {3600,7200}, "Only explicitly bounded experiment leases supported")
    temporary = session.client("sts").get_session_token(DurationSeconds=LIFETIME_SECONDS)["Credentials"]
    env.update({"AWS_ACCESS_KEY_ID": temporary["AccessKeyId"], "AWS_SECRET_ACCESS_KEY": temporary["SecretAccessKey"],
                "AWS_SESSION_TOKEN": temporary["SessionToken"], "AWS_DEFAULT_REGION": "us-east-1", "AWS_REGION": "us-east-1",
                "TRT_ARTIFACT_S3_BUCKET": secret["AVATAR_S3_BUCKET"], "TRT_ARTIFACT_S3_REGION": "us-east-1"})
    env.update({k: str(secret[k]) for k in ("AVATAR_S3_BUCKET", "AVATAR_S3_REGION", "AVATAR_S3_PREFIX", "AVATAR_S3_ENABLED") if secret.get(k)})
    SECRET_VALUES = [str(v) for k,v in {**secret, **env}.items() if any(m in k for m in ("ACCESS_KEY", "SESSION_TOKEN")) and len(str(v)) >= 8]
    ROOT.mkdir(mode=0o700)
    save("avatar-fixture-head.json", fixture)
    save("template.json", frozen)
    (ROOT / "onstart.sanitized.sh").write_text(script)
    fields = ("id", "machine_id", "host_id", "geolocation", "gpu_name", "num_gpus", "cpu_cores_effective", "cpu_ram", "disk_bw", "disk_space", "inet_down", "inet_up", "inet_down_cost", "inet_up_cost", "dph_total", "dph_total_adj", "storage_cost", "verification", "reliability")
    save("offer.json", {k: offer.get(k) for k in fields})
    deadline = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=LIFETIME_SECONDS+2)).replace(microsecond=0).isoformat()
    ledger = {"label": LABEL, "state": "initialized", "instance_id": None, "machine_id": offer["machine_id"],
              "source_revision": SOURCE, "template_hash": TEMPLATE, "image": IMAGE,
              "deadline_utc": deadline, "protected_instance_ids": [str(r["id"]) for r in rows],
              "budget_usd": 1.95 if ATTEMPT == "a2" else 2, "lifetime_seconds": LIFETIME_SECONDS, "poll_interval_seconds": 5,
              "clock_domain": "EC2_collector_monotonic_single_process", "stock_image_cache_state": "unknown",
              "s3_session_expires_utc": temporary["Expiration"].isoformat(), "ready_avatar_set": [AVATAR],
              "call_acceptance": "NOT_MEASURED_NO_ISOLATED_CONTROL_PLANE_ROUTE",
              "allow_unverified": allow_unverified,
              "host_limit": "8-core/unverified US fallback; not a premium-host prediction" if allow_unverified else None}
    save("ledger.json", ledger)
    timer(deadline)
    event("PREFLIGHT_PASS", offer_id=offer_id, disk_gb=DISK, lifetime_seconds=LIFETIME_SECONDS, budget_usd=ledger["budget_usd"],
          note="No production config/IAM/template changes; installation unchanged")
    require(dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(deadline) - dt.timedelta(seconds=LIFETIME_SECONDS-300), "Preflight consumed lease")
    request = {"client_id": "me", "template_hash_id": TEMPLATE, "image": IMAGE, "disk": DISK,
               "label": LABEL, "env": env, "onstart": script, "cancel_unavail": True}
    ledger["state"] = "submitting"
    START = time.monotonic()
    ledger["request_started_utc"] = utc()
    save("ledger.json", ledger)
    event("REQUEST_STARTED")
    try:
        response = client.create_instance(offer_id=offer_id, create_request=request)  # Exactly once.
        i = str(response.get("new_contract") or response.get("contract") or response.get("instance_id") or "")
        require(response.get("success") is True and i.isdigit() and i != PROTECTED, "Create outcome uncertain; reconcile only")
        ledger.update(state="created", instance_id=i)
        save("ledger.json", ledger)
        event("PROVIDER_ACCEPTED", instance_id=i)
        state, ssh_seen, previous_snapshot, read_failures = None, False, -100, 0
        from services.vast_client import VastClientError
        while time.monotonic()-START < 2400:
            try:
                row = owned(client, ledger)
                read_failures = 0
            except VastClientError as read_error:
                read_failures += 1
                status = re.search(r"status (\d{3})", str(read_error))
                event("PROVIDER_READ_RETRY", consecutive_failures=read_failures,
                      http_status=int(status[1]) if status else None,
                      timeout_mentioned="timed out" in str(read_error).lower(),
                      create_retried=False, raw_error_body_saved=False)
                require(read_failures <= 5, "Repeated provider inventory reads failed")
                time.sleep(15)
                continue
            require(row, "Owned instance absent during boot")
            safe = {k: row.get(k) for k in ("id", "label", "machine_id", "gpu_name", "num_gpus", "actual_status", "cur_state", "intended_status", "public_ipaddr", "ports", "ssh_host", "ssh_port", "image_uuid", "image_runtype", "start_date", "driver_version")}
            save("provider-latest.json", safe)
            current = (row.get("actual_status"), row.get("cur_state"))
            if current != state:
                event("PROVIDER_STATE", actual_status=current[0], cur_state=current[1])
                state = current
            require(row.get("actual_status") not in {"error", "failed", "invalid"}, "Provider boot failed")
            now = time.monotonic()
            if now-previous_snapshot >= 20:
                sample = snapshot(row)
                previous_snapshot = time.monotonic()
                if sample and not ssh_seen:
                    event("SSH_READY", worker_utc=sample["at_utc"], cache_state="unknown")
                    ssh_seen = True
                if sample and sample.get("source_revision"):
                    require(sample["source_revision"] == SOURCE, "Worker checkout differs from pinned main")
            base = endpoint(row) if row.get("public_ipaddr") else None
            if base:
                code, body = probe(base, "/health")
                if code == 200 and body and body.get("ok") is True and body.get("service") == "musetalk":
                    save("health.json", json.loads(scrub(json.dumps(body))))
                    event("HEALTH_READY", health_not_full_call_acceptance=True)
                    _, worker = probe(base, "/worker/state")
                    save("worker-state.json", json.loads(scrub(json.dumps(worker))))
                    smoke(base, audio)
                    snapshot(row)
                    event("BASELINE_DIAGNOSTIC_COMPLETE", full_scaleout_call_acceptance=False)
                    break
            time.sleep(5)
        else:
            raise ValueError("Health observation deadline")
    except Exception as error:
        event("BASELINE_FAILED", error_type=type(error).__name__, detail=scrub(str(error)) if isinstance(error, ValueError) else "Provider/credential error bodies suppressed")
        try:
            row = owned(client, ledger)
            if row:
                snapshot(row)
        except Exception:
            pass
        raise
    finally:
        cleanup(client, ledger)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["run", "expire"])
    p.add_argument("--offer-id", type=int)
    p.add_argument("--allow-unverified", action="store_true", help="Explicitly recorded US fallback; does not relax price/transfer caps")
    p.add_argument("--attempt", choices=["a1", "a2"], default="a1")
    a = p.parse_args()
    configure_attempt(a.attempt)
    try:
        if a.mode == "expire":
            require(os.getuid() == 0 and ROOT.is_dir() and not ROOT.is_symlink(), "Wrong expiry root")
            ledger = json.loads((ROOT / "ledger.json").read_text())
            require(dt.datetime.now(dt.timezone.utc) >= dt.datetime.fromisoformat(ledger["deadline_utc"]), "Expiry too early")
            if ledger["state"] == "initialized":
                event("EXPIRY_NO_SUBMISSION")
            else:
                cleanup(bootstrap(), ledger)
        else:
            require(a.offer_id is not None, "Offer ID required")
            run(a.offer_id, a.allow_unverified)
        return 0
    except Exception as error:
        print(json.dumps({"operation_failed": True, "error_type": type(error).__name__,
                          "safe_detail": str(error) if type(error) is ValueError else "Details suppressed; inspect protected evidence"}), flush=True)
        return 2


if __name__ == "__main__":
    raise SystemExit("Library only; use python_s3_venv_ec2.py for this experiment")
