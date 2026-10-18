#!/usr/bin/env python3
"""Owned-worker REST/native acceptance for the compiled uv screening candidate.

Run durably on EC2; no rentals, production registration or environment replacement.
Internal SSH forwarding is not direct-network/WebRTC/TURN acceptance.
"""
import datetime as dt
import json
import os
from pathlib import Path
import socket
import subprocess
import time

import dependency_optimization_ec2 as c


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def main():
    b = c.library()
    b.require(os.getuid() == 0, "Root-owned EC2 control plane required")
    os.umask(0o077)
    ledger = json.loads((c.ROOT/"ledger.json").read_text())
    b.require(dt.datetime.now(dt.timezone.utc) < dt.datetime.fromisoformat(ledger["deadline_utc"])-dt.timedelta(seconds=600), "Insufficient acceptance lease")
    candidate = json.loads((c.ROOT/"evidence/uv-compiled-screen-a3/receipt.json").read_text())
    b.require(candidate["state"] == "microbenchmark-passed" and not candidate["inventory_diff"], "Candidate screening not accepted")
    root = c.ROOT/"acceptance-a1"
    b.require(not root.exists(), "Acceptance already claimed; inspect existing service/receipt")
    root.mkdir(mode=0o700)
    rows = []
    tunnel = None
    candidate_attempted = False
    def save(name, value):
        (root/name).write_text(json.dumps(value, indent=2)+"\n")
    def event(stage, **fields):
        value = {"stage": stage, "utc": b.utc(), **fields}
        with (root/"events.jsonl").open("a") as out:
            out.write(json.dumps(value)+"\n")
        print(json.dumps(value), flush=True)
    try:
        client = b.bootstrap()
        row = b.owned(client, ledger)
        b.require(row and row["actual_status"] == "running", "Owned worker not running")
        ports = [free_port(), free_port()]
        b.require(ports[0] != ports[1], "Local forwarding port collision")
        ssh = c.ssh_arguments(row)
        command = ssh[:-2] + ["-N", "-o", "ExitOnForwardFailure=yes", "-L", f"127.0.0.1:{ports[0]}:127.0.0.1:8000",
                              "-L", f"127.0.0.1:{ports[1]}:127.0.0.1:8001", ssh[-2]]
        with (root/"tunnel.log").open("xb") as log:
            tunnel = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
        baseline = f"http://127.0.0.1:{ports[0]}"
        for _ in range(20):
            code, body = b.probe(baseline, "/health")
            if code == 200 and body and body.get("ok") is True:
                break
            b.require(tunnel.poll() is None, "Owned SSH tunnel terminated")
            time.sleep(1)
        else:
            raise ValueError("Baseline API not reachable over owned forwarding")
        audio = b.BOOTSTRAP.parent/"startup-speech.wav"
        for name in ("baseline", "uv-compiled"):
            case = root/name
            case.mkdir(mode=0o700)
            if name != "baseline":
                payload = '''import json,os,pathlib,subprocess,time
root=pathlib.Path(__ROOT__);root.mkdir(mode=0o700,exist_ok=False)
env=dict(os.environ)
for k in list(env):
 if k.startswith('LINGUA_'):env.pop(k)
env.update({'VENV_PATH':__VENV__,'PORT':'8001','HOST':'127.0.0.1','MUSETALK_RECIPE':'r5','MUSETALK_TAESD_TRT_BUILD':'0','MUSETALK_TAESD_TRT_STRICT':'1','MUSETALK_TRT_FALLBACK':'0','MUSETALK_RUNTIME_CONFIG_SOURCE':'injected','MUSETALK_RUNTIME_DIR':str(root/'runtime'),'WEBRTC_NATIVE_VP8_DIR':'/workspace/MuseTalk/.runtime/native_vp8','LINGUA_CONTROL_PLANE_BASE_URL':'','LINGUA_WORKER_TOKEN':'','LINGUA_WORKER_REGISTER_URL':'','LINGUA_WORKER_HEARTBEAT_URL':''})
with (root/'candidate-api.log').open('xb') as log:
 p=subprocess.Popen(['bash','scripts/run_musetalk_server.sh','--venv-path',__VENV__,'--host','127.0.0.1','--port','8001'],cwd='/workspace/MuseTalk',env=env,stdout=log,stderr=log,start_new_session=True)
stat=pathlib.Path('/proc')/str(p.pid)/'stat';data=stat.read_text();ticks=data[data.rfind(')')+2:].split()[19]
value={'pid':p.pid,'start_ticks':ticks,'worker_spawn_monotonic':time.monotonic(),'worker_spawn_epoch':time.time()}
(root/'candidate-pid.json').write_text(json.dumps(value));print(json.dumps(value))
'''.replace("__ROOT__", repr(c.WORKER_ROOT+"/acceptance-a1")).replace("__VENV__", repr(c.WORKER_ROOT+"/uv-compiled-screen-a3/venv"))
                compile(payload, '<owned-candidate-launch>', 'exec')
                started = time.monotonic()
                candidate_attempted = True
                launch = json.loads(c.worker(b, payload))
                save("candidate-launch.json", launch)
                base = f"http://127.0.0.1:{ports[1]}"
                until = time.monotonic()+90
                while time.monotonic() < until:
                    code, body = b.probe(base, "/health")
                    if code == 200 and body and body.get("ok") is True:
                        break
                    time.sleep(1)
                else:
                    raise ValueError("Candidate API health deadline; inspect preserved worker log")
                event("CANDIDATE_API_HEALTHY", dispatch_to_observed_health_seconds=time.monotonic()-started,
                      clock_domain="single EC2 monotonic; includes SSH launch transport and polling")
            else:
                base = baseline
            old_root = b.ROOT
            before = time.monotonic()
            try:
                b.ROOT = case
                b.SMOKE_PHASE_SECONDS = 120
                b.smoke(base, audio)
            finally:
                b.ROOT = old_root
            info = json.loads((case/"video-smoke.json").read_text())
            rows.append({"case": name, "smoke_seconds": time.monotonic()-before, **info})
            event("CASE_OUTPUT_PASS", case=name, video_bytes=info["video_bytes"], video_sha256=info["video_sha256"],
                  avatar_state_note="baseline may fetch from S3; candidate reuses prepared disk files; not a matched avatar-download timing")
        result = {"state": "passed", "cases": rows, "artifact_id": candidate["artifact_id"],
                  "scope": "same prepared avatar/input decoded REST audio+video; not WebRTC/TURN, direct connectivity or registration",
                  "throughput_certification": "PENDING_MATCHED_TIMED_GENERATIONS"}
    except BaseException as error:
        result = {"state": "failed", "cases": rows, "error_type": type(error).__name__,
                  "detail": str(error) if type(error) is ValueError else "Inspect protected acceptance logs"}
        event("ACCEPTANCE_FAILED", **{k:v for k,v in result.items() if k not in {"state","cases"}})
    finally:
        if candidate_attempted:
            payload = '''import json,pathlib,os,signal
root=pathlib.Path(ROOT);v=json.loads((root/'candidate-pid.json').read_text());pid=v['pid']
stat=pathlib.Path('/proc')/str(pid)/'stat'
if stat.exists():
 data=stat.read_text();assert data[data.rfind(')')+2:].split()[19]==v['start_ticks'] and os.getpgid(pid)==pid
 os.killpg(pid,signal.SIGTERM)
print(json.dumps({'candidate_stop_sent':True,'pid':pid}))
'''.replace("ROOT", repr(c.WORKER_ROOT+"/acceptance-a1"))
            try:save("candidate-stop.json", json.loads(c.worker(b, payload)))
            except BaseException:result["candidate_stop"] = "unconfirmed; owned rental expiry remains active"
        if tunnel:
            tunnel.terminate()
            try:tunnel.wait(timeout=10)
            except subprocess.TimeoutExpired:tunnel.kill();tunnel.wait()
        save("receipt.json", result)
    print(json.dumps({"result": result}), flush=True)
    return 0 if result["state"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
