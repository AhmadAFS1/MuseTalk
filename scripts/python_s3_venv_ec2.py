#!/usr/bin/env python3
"""Up-to-two-hour EC2-controlled venv retry, with detached jobs and durable S3 receipts.

Deploy alongside the reviewed source_baseline_ec2.py as source_library.py, the
timing patch, and python_s3_venv_worker.py. Only run as root on the existing EC2.
No provider create retry, no production callbacks, no IAM mutation in this tool.
"""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import datetime as dt
import hashlib
import importlib.util
import ipaddress
import json
import os
from pathlib import Path
import re
import subprocess
import time
import uuid
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from python_s3_venv_detached import launcher, reader

LABEL = "musetalk-python-s3-venv-20261011-a3"
HERE = Path(__file__).resolve().parent
ROOT = Path("/home/ec2-user/.local/state") / LABEL
BUCKET = "lingua-musetalk-s3-storage"
PREFIX = "python-venv/20261011/" + LABEL + "/"
APPROVED_OFFER = 48396264
APPROVED_MACHINE = 136634
APPROVED_HOST = 399360
APPROVED_LOCATION = "United States, US"
DISK_GB = 128
DOWNLOAD_CAP = .0015
UPLOAD_CAP = .0015


def approved_offer(o, base_acceptable):
    # A replacement US offer became available within the original, stricter caps.
    def finite_cap(k, cap):
        value = o.get(k)
        return type(value) in (int, float) and 0 <= value <= cap
    normalized = dict(o)
    normalized.pop("dph_total_adj", None)
    normalized.pop("discounted_total_per_hour", None)
    return (type(o.get("id")) is int and o["id"] > 0 and o.get("machine_id") == APPROVED_MACHINE
            and o.get("host_id") == APPROVED_HOST and o.get("geolocation") == APPROVED_LOCATION
            and finite_cap("dph_total", .30) and finite_cap("dph_total_adj", .30)
            and finite_cap("inet_down_cost", DOWNLOAD_CAP) and finite_cap("inet_up_cost", UPLOAD_CAP)
            and base_acceptable(normalized))


def load_library():
    spec = importlib.util.spec_from_file_location("source_library", HERE / "source_library.py")
    b = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(b)
    b.LABEL, b.ROOT, b.ATTEMPT = LABEL, ROOT, "venv3"
    b.LIFETIME_SECONDS = 7200
    b.DISK = DISK_GB
    b.AVATAR = "en_01_maya_abe451d3_talking_f6cf8be7fb"
    b.TRANSFER_CAP = UPLOAD_CAP
    original_acceptable = b.acceptable
    b.acceptable = lambda o, allow_unverified=False: approved_offer(o, original_acceptable)
    def select(rows, offer_id, allow_unverified=False):
        matches = [o for o in rows if b.acceptable(o)]
        b.require(len(matches) == 1, "Equivalent approved machine slot unavailable or ambiguous")
        return matches[0]
    b.select_offer = select
    return b


def timer(b, deadline):
    when = dt.datetime.fromisoformat(deadline).strftime("%Y-%m-%d %H:%M:%S UTC")
    command = ["systemd-run", "--quiet", "--unit=" + LABEL + "-expiry", "--on-calendar=" + when,
               "--timer-property=AccuracySec=1s", "--property=Restart=on-failure", "--property=RestartSec=30s",
               "/home/ec2-user/lingua/venv/bin/python", "-B", str(Path(__file__).resolve()), "expire"]
    subprocess.run(command, check=True, capture_output=True, timeout=15)
    b.require(subprocess.run(["systemctl", "is-active", LABEL + "-expiry.timer"], capture_output=True).returncode == 0,
              "Expiry timer not active")


def ssh_arguments(row, b):
    direct = {str(p.get("HostPort")) for p in (row.get("ports") or {}).get("22/tcp", [])}
    if len(direct) == 1 and row.get("public_ipaddr"):
        host, port = row["public_ipaddr"], direct.pop()
        b.require(ipaddress.ip_address(host).is_global, "Global worker address required")
    else:
        host, port = row.get("ssh_host"), str(row.get("ssh_port", ""))
    b.require(host and re.fullmatch(r"[A-Za-z0-9.-]+", host) and port.isdigit()
              and 1 <= int(port) <= 65535, "Invalid owned SSH endpoint")
    return ["ssh", "-i", "/home/ec2-user/.ssh/id_ed25519", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
            "-o", "ServerAliveInterval=5", "-o", "ServerAliveCountMax=6",
            "-o", "StrictHostKeyChecking=accept-new", "-o", "UserKnownHostsFile=" + str(ROOT / "known_hosts"),
            "-p", port, "root@" + host, "python3 -B -"]


def worker_script(config, ledger):
    code = (HERE / "python_s3_venv_worker.py").read_text()
    # Acknowledgement distinguishes SSH/interpreter failures from worker failures.
    # Keep source execution inside the exception handler too (e.g. SyntaxError).
    wrapper = "import json\nprint(json.dumps(" + repr({"stage": "WORKER_STARTED", "mode": config.get("mode")}) + "),flush=True)\n"
    wrapper += "try:\n ns={'__name__':'venv_worker'}\n exec(" + repr(code) + ",ns)\n"
    wrapper += " ns['main'](" + repr({"label": LABEL, "deadline_utc": ledger["deadline_utc"], **config}) + ")\n"
    wrapper += "except Exception as e:\n print(__import__('json').dumps({'worker_operation_failed':True,'error_type':type(e).__name__,'safe_detail':str(e) if type(e) is ValueError else 'Details suppressed'}),flush=True)\n raise SystemExit(2)\n"
    return wrapper


def invocation_diagnostics(result):
    # Do not store stderr: Python tracebacks can include a line containing signed
    # URLs/config. Static categories, counts and a hash cannot disclose that text.
    stderr = result.stderr.decode(errors="replace")
    categories = {
        "publickey_authentication": "Permission denied (publickey)",
        "host_key_failure": "Host key verification failed",
        "connection_refused": "Connection refused",
        "connection_timeout": "Connection timed out",
        "peer_closed": "closed by",
        "remote_syntax_error": "SyntaxError",
        "missing_import": "ModuleNotFoundError",
        "killed_process": "Killed",
    }
    return {"returncode": result.returncode, "stdout_bytes": len(result.stdout),
            "stderr_bytes": len(result.stderr), "stderr_sha256": hashlib.sha256(result.stderr).hexdigest(),
            "error_categories": [name for name, marker in categories.items() if marker in stderr],
            "raw_stderr_saved": False}


def worker_sync(b, row, config, timeout, ledger):
    b.require(remaining(ledger) > 40, "Worker lease nearly expired")
    timeout = min(timeout, int(remaining(ledger))-20)
    wrapper = worker_script(config, ledger)
    for attempt in range(3):
        try:
            r = subprocess.run(ssh_arguments(row, b), input=wrapper.encode(), capture_output=True,
                               timeout=min(timeout, int(remaining(ledger))-20))
        except subprocess.TimeoutExpired:
            b.event("WORKER_INVOCATION_TIMEOUT", mode=config.get("mode"), part=config.get("part"), timeout_seconds=timeout)
            raise
        # Empty stdout means no acknowledgement; retries remain limited to
        # read-only, lock-protected finalization, or idempotent part uploads.
        retryable = config.get('mode') in {'probe','diagnose','verify_archive','finalize','upload_part'}
        if r.returncode!=255 or (r.stdout and config.get('mode')!='upload_part') or not retryable or attempt==2:
            break
        b.event('WORKER_TRANSPORT_RETRY', mode=config.get('mode'), part=config.get('part'), attempt=attempt+1,
                **invocation_diagnostics(r))
        b.require(remaining(ledger)>60, 'No bounded transport retry window')
        time.sleep(3)
    b.event("WORKER_INVOCATION_RESULT", mode=config.get("mode"), part=config.get("part"),
            **invocation_diagnostics(r))
    records = []
    for line in r.stdout.decode(errors="replace").splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        records.append(value)
        if "stage" in value or value.get("worker_operation_failed"):
            fields = dict(value)
            if "stage" in fields:
                fields["worker_stage"] = fields.pop("stage")
            b.event("WORKER_EVIDENCE", **fields)
    b.require(r.returncode == 0 and not any(v.get("worker_operation_failed") for v in records),
              "Owned-worker operation failed; inspect safe worker evidence")
    matches = [v for v in records if "result" in v]
    b.require(bool(records), "Worker returned no evidence")
    return matches[-1] if matches else records[-1]


def worker(b, row, config, timeout, ledger):
    b.require(remaining(ledger) > 40, "Worker lease nearly expired")
    job_id = uuid.uuid4().hex
    job_root = '/workspace/.' + LABEL + '-jobs'
    # Durable launch identity on EC2 comes before any SSH request.
    b.save('job-'+job_id+'.json', {'job_id':job_id,'mode':config.get('mode'),
                                 'deadline_utc':ledger['deadline_utc']})
    payload = worker_script(config, ledger)
    deadline = time.monotonic()+min(timeout, int(remaining(ledger))-25)
    submitted = False
    while time.monotonic() < deadline and remaining(ledger)>25:
        script = reader(job_root, job_id) if submitted else launcher(job_root, job_id, payload)
        try:
            r = subprocess.run(ssh_arguments(row,b), input=script.encode(),capture_output=True,timeout=25)
            records=[]
            for line in r.stdout.splitlines():
                try: records.append(json.loads(line))
                except ValueError: pass
            facts=records[-1] if records else {}
            if not submitted:
                b.event('DETACHED_JOB_LAUNCH',job_id=job_id,mode=config.get('mode'),**invocation_diagnostics(r))
                submitted=True  # Unknown launch outcome is reconciled by this same ID.
            if facts.get('state')=='absent':
                submitted=False
            elif facts.get('state')=='complete':
                result=facts['result']
                b.require(not result.get('worker_operation_failed'),'Detached worker operation failed')
                b.save('job-'+job_id+'-result.json',facts)
                b.event('DETACHED_JOB_COMPLETE',job_id=job_id,mode=config.get('mode'))
                return result
            elif facts.get('state')=='failed':
                b.save('job-'+job_id+'-result.json',facts)
                b.event('DETACHED_JOB_FAILED',job_id=job_id,error_type=facts.get('error_type'))
                raise ValueError('Detached worker failed; safe durable receipt retained')
        except subprocess.TimeoutExpired:
            submitted=True
            b.event('DETACHED_TRANSPORT_TIMEOUT',job_id=job_id,job_resubmitted=False)
        time.sleep(5)
    raise ValueError('Detached job observation window exceeded; do not resubmit without reconciliation')


def remaining(ledger):
    return (dt.datetime.fromisoformat(ledger["deadline_utc"]) - dt.datetime.now(dt.timezone.utc)).total_seconds()


def exercise(b, client, ledger):
    recovered = ROOT / "venv-manifest.json"
    b.require(remaining(ledger) >= (420 if recovered.is_file() else 900), "No bounded archive publication window")
    row = b.owned(client, ledger)
    b.require(row is not None, "Owned worker missing")
    # Recovery after a transport failure may leave an audited manifest on the
    # same owned worker. Reverify bytes/packages before using that completed build.
    if recovered.is_file():
        m = worker(b, row, {"mode": "verify_archive"}, timeout=180, ledger=ledger)["manifest"]
        b.require(m == json.loads(recovered.read_text()), "Recovered manifest differs from the worker")
    else:
        m = worker(b, row, {"mode": "build"}, timeout=500, ledger=ledger)["manifest"]
    b.save("venv-manifest.json", m)
    b.require(m["label"] == LABEL and 0 < m["archive_bytes"] < 12 * 1024**3
              and re.fullmatch(r"[0-9a-f]{64}", m["archive_sha256"]), "Unexpected built archive")
    import boto3
    from botocore.config import Config
    s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version="s3v4"))
    key = PREFIX + m["contract_sha256"] + "/venv.tar.zst"
    try:
        s3.head_object(Bucket=BUCKET, Key=key)
        raise ValueError("Artifact key exists; refusing overwrite")
    except s3.exceptions.ClientError as error:
        b.require(error.response["ResponseMetadata"]["HTTPStatusCode"] == 404, "Cannot verify fresh artifact key")
    upload = s3.create_multipart_upload(Bucket=BUCKET, Key=key, ContentType="application/zstd",
                                         Metadata={"sha256": m["archive_sha256"], "experiment": LABEL})
    upload_id = upload["UploadId"]
    part_size = 32 * 1024**2
    # Save upload identity before any signed URL or upload request leaves EC2.
    b.save('multipart-state.json', {'key':key,'upload_id':upload_id,'bytes':m['archive_bytes'],
                                  'part_size':part_size,'state':'uploading'})
    started = time.monotonic()
    parts = []
    completed = False
    configs = []
    for i, offset in enumerate(range(0, m["archive_bytes"], part_size), 1):
        configs.append({'part':i,'offset':offset,'bytes':min(part_size,m['archive_bytes']-offset)})
    for attempt in range(3):
        # S3, rather than the SSH return code, is authoritative for each part.
        receipts=[]
        for page in s3.get_paginator('list_parts').paginate(Bucket=BUCKET,Key=key,UploadId=upload_id):
            receipts.extend(page.get('Parts',[]))
        b.save('multipart-receipts.json',{'attempt':attempt,'parts':[
            {'PartNumber':p['PartNumber'],'ETag':p['ETag'],'Size':p['Size']} for p in receipts]})
        received={p['PartNumber']:p for p in receipts}
        missing=[dict(c) for c in configs if c['part'] not in received or received[c['part']]['Size']!=c['bytes']]
        if not missing:
            parts=[{'PartNumber':c['part'],'ETag':received[c['part']]['ETag']} for c in configs]
            break
        b.require(remaining(ledger)>600,'No safe batch upload window')
        for c in missing:
            url = s3.generate_presigned_url("upload_part", Params={"Bucket": BUCKET, "Key": key,
                "UploadId": upload_id, "PartNumber": c['part']}, ExpiresIn=min(3600,int(remaining(ledger))))
            c['url']=url
        result=worker(b,row,{'mode':'upload_batch','parts':missing},timeout=min(1800,int(remaining(ledger))-400),ledger=ledger)
        b.save('multipart-batch-'+str(attempt)+'.json',result)
    # Final authoritative reconciliation also covers completion of the last batch.
    receipts=[]
    for page in s3.get_paginator('list_parts').paginate(Bucket=BUCKET,Key=key,UploadId=upload_id):
        receipts.extend(page.get('Parts',[]))
    received={p['PartNumber']:p for p in receipts}
    b.require(len(received)==len(configs) and all(received.get(c['part'],{}).get('Size')==c['bytes'] for c in configs),
              'S3 has incomplete parts; retain upload identity for bounded recovery')
    parts=[{'PartNumber':c['part'],'ETag':received[c['part']]['ETag']} for c in configs]
    b.save('multipart-receipts.json',{'parts':[{'PartNumber':p['PartNumber'],'ETag':p['ETag'],'Size':p['Size']} for p in receipts]})
    s3.complete_multipart_upload(Bucket=BUCKET,Key=key,UploadId=upload_id,MultipartUpload={'Parts':parts})
    completed=True
    b.save('multipart-state.json',{'key':key,'upload_id':upload_id,'bytes':m['archive_bytes'],'state':'complete'})
    upload_seconds = time.monotonic() - started
    head = s3.head_object(Bucket=BUCKET, Key=key)
    b.require(head["ContentLength"] == m["archive_bytes"] and head["Metadata"].get("sha256") == m["archive_sha256"],
              "Uploaded S3 object differs")
    m.update(s3_bucket=BUCKET, s3_key=key, upload_seconds=round(upload_seconds, 3),
             upload_decimal_MB_per_second=round(m["archive_bytes"]/upload_seconds/10**6, 3))
    b.save("venv-manifest.json", m)
    s3.put_object(Bucket=BUCKET, Key=key.rsplit("/",1)[0]+"/manifest.json",
                  Body=json.dumps(m, indent=2).encode(), ContentType="application/json")
    b.event("S3_ARCHIVE_PUBLISHED", key=key, bytes=m["archive_bytes"], sha256=m["archive_sha256"],
            upload_seconds=m["upload_seconds"], decimal_MB_per_second=m["upload_decimal_MB_per_second"])
    if remaining(ledger)<360:
        b.event('RESTORE_SKIPPED',reason='Archive published; insufficient remaining original lease',
                remaining_seconds=round(remaining(ledger),1))
        return
    url = s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=1800)
    result = worker(b, row, {"mode": "restore", "url": url, "sha256": m["archive_sha256"],
                            "bytes": m["archive_bytes"]}, timeout=min(1500, int(remaining(ledger))-40), ledger=ledger)
    b.save("venv-restore-result.json", result)
    if remaining(ledger) > 750:
        b.smoke(b.endpoint(row), b.BOOTSTRAP.parent / "startup-speech.wav")
    else:
        b.event("REST_VIDEO_SMOKE_SKIPPED", reason="Protect two-hour deadline")
    b.snapshot(row)
    b.event("VENV_EXPERIMENT_COMPLETE", same_instance_restore=True, fresh_host_startup_measured=False)


def run(b, offer_id):
    import boto3
    s3 = boto3.client("s3", region_name="us-east-1")
    # Verify publisher permission before requesting GPU time.
    s3.put_object(Bucket=BUCKET, Key=PREFIX+"preflight.json", Body=b'{"purpose":"venv experiment preflight"}',
                  ContentType="application/json")
    original_freeze, original_cleanup, original_smoke = b.frozen_template, b.cleanup, b.smoke
    original_owned, original_snapshot = b.owned, b.snapshot
    ssh_key_attached, helper_probed = False, False
    def owned(client, ledger):
        nonlocal ssh_key_attached
        row = original_owned(client, ledger)
        if row and not ssh_key_attached:
            # Only after original_owned verifies label, machine, GPU and protected IDs.
            import requests
            pub = Path("/home/ec2-user/.ssh/id_ed25519.pub").read_text().strip()
            b.require(pub.startswith("ssh-ed25519 ") and "\n" not in pub, "Expected existing EC2 public key")
            response = requests.post(client.base_url + "/api/v0/instances/" + str(row["id"]) + "/ssh/",
                                     headers=client._headers(), json={"ssh_key": pub}, timeout=20, allow_redirects=False)
            b.event("TEST_SSH_KEY_ATTACH", instance_id=str(row["id"]), http_status=response.status_code,
                    key_sha256=hashlib.sha256(pub.encode()).hexdigest(), scope="Owned retry worker only")
            b.require(response.status_code == 200 and response.json().get("success") is True,
                      "Test-only SSH key attachment failed")
            ssh_key_attached = True
        return row
    def snapshot(row):
        nonlocal helper_probed
        sample = original_snapshot(row)
        if sample and not helper_probed:
            ledger = json.loads((ROOT / "ledger.json").read_text())
            result = worker(b, row, {"mode": "probe"}, timeout=40, ledger=ledger)
            b.save("worker-helper-probe.json", result)
            b.require(result.get("result") == "probe" and result.get("gpu") == "NVIDIA GeForce RTX 3090",
                      "Early helper probe failed")
            helper_probed = True
        return sample
    b.owned, b.snapshot = owned, snapshot
    original_save = b.save
    def save(name, value):
        if name == "ledger.json" and value.get("state") == "initialized":
            # Remove the baseline helper's two timer-alignment seconds, keeping
            # this experiment's durable deadline within the approved allowance.
            deadline = dt.datetime.fromisoformat(value["deadline_utc"]) - dt.timedelta(seconds=2)
            value["deadline_utc"] = deadline.isoformat()
        original_save(name, value)
    b.save = save
    patch = (HERE / "python_s3_venv_timing.patch").read_bytes()
    patch_hash = hashlib.sha256(patch).hexdigest()
    payload = base64.b64encode(patch).decode()
    def freeze(t):
        env, script, receipt = original_freeze(t)
        marker = 'cd "$REPO_DIR"'
        injected = marker + "\n" + "printf '%s' '" + payload + "' | base64 -d | git apply --check -\n"
        injected += "printf '%s' '" + payload + "' | base64 -d | git apply -\n"
        injected += "echo '[venv-experiment] dependency timing patch " + patch_hash + "'\n"
        script = script.replace(marker, injected, 1)
        subprocess.run(["bash", "-n"], input=script.encode(), check=True, capture_output=True)
        receipt.update(dependency_timing_patch_sha256=patch_hash,
                       run_onstart_sha256=hashlib.sha256(script.encode()).hexdigest(),
                       bandwidth_exception="Not used: replacement meets original 0.0015/GB caps",
                       selected_machine_id=APPROVED_MACHINE,
                       selected_offer_id="Resolved from fresh search; see offer.json",
                       download_cost_cap_per_GB=DOWNLOAD_CAP, upload_cost_cap_per_GB=UPLOAD_CAP)
        receipt["changes"].append("separate main/Kokoro timers; dependency pins/groups unchanged")
        return env, script, receipt
    b.frozen_template = freeze
    b.timer = lambda deadline: timer(b, json.loads((ROOT / "ledger.json").read_text())["deadline_utc"])
    # Reserve acceptance video work for the restored env; baseline phase only needs health.
    b.smoke = lambda *args: b.event("BASELINE_VIDEO_DEFERRED", reason="Validate after restoring identical dependency environment")
    def preserve_failure(client, ledger):
        # Keep diagnostic files on the same paid worker until its original expiry,
        # not indefinitely. A failed operation no longer destroys evidence instantly.
        active = subprocess.run(["systemctl", "is-active", LABEL + "-expiry.timer"], capture_output=True).returncode == 0
        if not active or remaining(ledger) <= 90:
            original_cleanup(client, ledger)
            return
        try:
            row = original_owned(client, ledger)
            if row:
                evidence = worker(b, row, {"mode": "diagnose"}, timeout=35, ledger=ledger)
                b.save("worker-failure-diagnostics.json", evidence)
        except Exception as error:
            b.event("FAILURE_DIAGNOSTIC_READ_FAILED", error_type=type(error).__name__)
        b.event("FAILED_WORKER_PRESERVED_UNTIL_EXPIRY", deadline_utc=ledger["deadline_utc"],
                remaining_seconds=round(max(0, remaining(ledger)), 1), new_instance_created=False)
    def cleanup(client, ledger):
        try:
            stages = [json.loads(line)["stage"] for line in (ROOT / "events.jsonl").read_text().splitlines()]
            if "BASELINE_FAILED" in stages:
                preserve_failure(client, ledger)
                return
            if "BASELINE_DIAGNOSTIC_COMPLETE" in stages and "BASELINE_FAILED" not in stages:
                b.smoke = original_smoke
                exercise(b, client, ledger)
        except Exception as error:
            b.event("VENV_EXPERIMENT_FAILED", error_type=type(error).__name__,
                    safe_detail=str(error) if type(error) is ValueError else "Details suppressed")
            preserve_failure(client, ledger)
            raise
        original_cleanup(client, ledger)
    b.cleanup = cleanup
    b.run(offer_id)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["run", "recover", "resume_baseline", "finish_restored", "expire", "cleanup", "status"])
    p.add_argument("--offer-id", type=int)
    a = p.parse_args()
    b = load_library()
    b.require(os.getuid() == 0, "EC2 root-owned helper required")
    if a.mode == "run":
        b.require(a.offer_id is not None, "Offer ID required")
        run(b, a.offer_id)
    else:
        b.require(ROOT.is_dir() and not ROOT.is_symlink(), "No experiment ledger")
        ledger = json.loads((ROOT / "ledger.json").read_text())
        if a.mode == "expire":
            b.require(remaining(ledger) <= 0, "Expiry too early")
        client = b.bootstrap()
        if a.mode == 'finish_restored':
            b.require(ledger.get('state')=='created' and remaining(ledger)>420,
                      'No bounded restored-runtime smoke window')
            b.SMOKE_PHASE_SECONDS=120
            b.require(subprocess.run(['systemctl','is-active',LABEL+'-expiry.timer'],capture_output=True).returncode==0,
                      'Original expiry must remain active')
            result=json.loads((ROOT/'venv-restore-result.json').read_text())
            b.require(result.get('pip_install_executed') is False and result.get('health',{}).get('ok') is True,
                      'Validated restore result required; never start a new GET here')
            try:
                row=b.owned(client,ledger)
                b.require(row is not None,'Owned worker absent')
                b.smoke(b.endpoint(row),b.BOOTSTRAP.parent/'startup-speech.wav')
                b.event('VENV_EXPERIMENT_COMPLETE',same_instance_restore=True,fresh_host_startup_measured=False)
            except Exception as error:
                b.event('RESTORED_VIDEO_SMOKE_FAILED',error_type=type(error).__name__,
                        safe_detail=str(error) if type(error) is ValueError else 'Details suppressed')
                raise
            finally:
                try:
                    row=b.owned(client,ledger)
                    if row:b.snapshot(row)
                finally:
                    b.cleanup(client,ledger)
        elif a.mode == 'resume_baseline':
            b.require(ledger.get('state')=='created' and remaining(ledger)>1800,
                      'No bounded same-instance baseline recovery window')
            b.require(subprocess.run(['systemctl','is-active',LABEL+'-expiry.timer'],capture_output=True).returncode==0,
                      'Original expiry must remain active')
            b.require(subprocess.run(['systemctl','is-active','musetalk-python-s3-venv-a3-run-20261011.service'],
                                    capture_output=True).returncode!=0,'Original controller is still active')
            stages=[json.loads(s) for s in (ROOT/'events.jsonl').read_text().splitlines()]
            b.require(not any(s.get('stage')=='DETACHED_JOB_LAUNCH' and s.get('mode')=='build' for s in stages)
                      and not (ROOT/'venv-manifest.json').exists(),'Archive work already started; reconcile instead')
            b.event('SAME_INSTANCE_BASELINE_OBSERVATION_RESUMED',new_instance=False,lease_extended=False)
            for attempt in range(160):
                b.require(remaining(ledger)>1800,'Insufficient publication/restore time reserve')
                row=b.owned(client,ledger)
                b.require(row is not None,'Owned worker absent')
                base=b.endpoint(row)
                code,body=b.probe(base,'/health') if base else (None,None)
                if code==200 and body and body.get('ok') is True and body.get('service')=='musetalk':
                    b.save('recovered-baseline-health.json',body)
                    b.event('RECOVERED_BASELINE_HEALTH_READY',full_call_acceptance=False)
                    exercise(b,client,ledger)
                    b.cleanup(client,ledger)
                    break
                time.sleep(15)
            else:
                raise ValueError('Bounded baseline recovery observation expired')
        elif a.mode == "recover":
            b.require(ledger.get('state')=='created' and remaining(ledger)>=(420 if (ROOT/'venv-manifest.json').exists() else 900),
                      "No live bounded recovery window")
            b.require(subprocess.run(['systemctl','is-active',LABEL+'-expiry.timer'],capture_output=True).returncode==0,
                      "Original expiry must still be active")
            row = b.owned(client, ledger)
            b.require(row is not None, "Owned retry worker absent")
            if not (ROOT/'venv-manifest.json').exists():
                info = worker(b, row, {'mode':'diagnose'}, timeout=40, ledger=ledger)
                mode = 'verify_archive' if info.get('manifest_exists') else 'finalize'
                m = worker(b, row, {'mode':mode}, timeout=300, ledger=ledger)['manifest']
                b.save('venv-manifest.json',m)
            b.event('SAME_INSTANCE_RECOVERY_STARTED', deadline_utc=ledger['deadline_utc'], lease_extended=False)
            exercise(b, client, ledger)
            b.cleanup(client, ledger)
        elif a.mode in {"expire", "cleanup"}:
            if ledger["state"] != "initialized":
                b.cleanup(client, ledger)
            state_path=ROOT/'multipart-state.json'
            if state_path.is_file():
                state=json.loads(state_path.read_text())
                if state.get('state')=='uploading':
                    b.require(state['key'].startswith(PREFIX),'Multipart cleanup outside owned prefix')
                    import boto3
                    s3=boto3.client('s3',region_name='us-east-1')
                    # GPU absence precedes abort; no subsequent new PUTs can survive.
                    for attempt in range(2):
                        try:
                            s3.abort_multipart_upload(Bucket=BUCKET,Key=state['key'],UploadId=state['upload_id'])
                        except s3.exceptions.ClientError as error:
                            b.require(error.response['Error']['Code']=='NoSuchUpload','Multipart cleanup failed')
                        if attempt==0: time.sleep(5)
                    state['state']='aborted_after_worker_cleanup'
                    b.save('multipart-state.json',state)
                    b.event('OWNED_MULTIPART_ABORTED_AFTER_CLEANUP',key=state['key'])
        else:
            row = b.owned(client, ledger)
            if row:
                b.snapshot(row)
            print(json.dumps({"ledger": ledger, "provider_status": row.get("actual_status") if row else None}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(json.dumps({"operation_failed": True, "error_type": type(error).__name__,
                          "safe_detail": str(error) if type(error) is ValueError else "Details suppressed"}), flush=True)
        raise SystemExit(2)
