"""Private stdin-fed jobs: transport loss never resubmits an existing job.

The payload (including signed tickets) is only sent to child stdin, never argv
or disk. Durable receipts contain results, not source/config or raw tracebacks.
"""
from pathlib import Path
import re


def job_path(root, job_id):
    if not re.fullmatch(r"[a-f0-9]{32}", job_id):
        raise ValueError("Invalid private job identifier")
    return str(Path(root) / job_id)


def launcher(root, job_id, payload):
    path = job_path(root, job_id)
    # The exclusive directory claim makes retrying this exact launch idempotent.
    child = """import json,os,sys,time,resource
from pathlib import Path
os.umask(0o077)
resource.setrlimit(resource.RLIMIT_CORE,(0,0))
job=Path(JOB_PATH)
try:
 exec(PAYLOAD,{'__name__':'detached_payload'})
 sys.stdout.flush()
 records=[]
 for line in (job/'job.log').read_text().splitlines():
  try: records.append(json.loads(line))
  except ValueError: pass
 matches=[v for v in records if 'result' in v]
 result=matches[-1] if matches else (records[-1] if records else {})
 receipt={'state':'complete','result':result,'completed_epoch':time.time()}
except BaseException as error:
 receipt={'state':'failed','error_type':type(error).__name__,'completed_epoch':time.time()}
tmp=job/'receipt.tmp'
tmp.write_text(json.dumps(receipt))
tmp.replace(job/'receipt.json')
""".replace("JOB_PATH", repr(path)).replace("PAYLOAD", repr(payload))
    return """import json,os,subprocess,sys
from pathlib import Path
os.umask(0o077)
root=Path(ROOT_PATH)
root.mkdir(mode=0o700,parents=True,exist_ok=True)
assert not root.is_symlink() and root.stat().st_uid==os.getuid()
job=Path(JOB_PATH)
try:
 job.mkdir(mode=0o700)
except FileExistsError:
 print(json.dumps({'state':'claimed','job_id':JOB_ID}),flush=True)
else:
 try:
  with (job/'job.log').open('xb') as log:
   p=subprocess.Popen([sys.executable,'-u','-B','-'],stdin=subprocess.PIPE,
       stdout=log,stderr=log,start_new_session=True,cwd=str(root))
   p.stdin.write(CHILD_SOURCE.encode());p.stdin.close()
  (job/'pid.json').write_text(json.dumps({'pid':p.pid}))
  print(json.dumps({'state':'started','job_id':JOB_ID,'pid':p.pid}),flush=True)
 except BaseException as error:
  (job/'receipt.json').write_text(json.dumps({'state':'failed','error_type':type(error).__name__}))
  print(json.dumps({'state':'failed','job_id':JOB_ID}),flush=True)
""".replace("ROOT_PATH", repr(str(root))).replace("JOB_PATH", repr(path)).replace("JOB_ID", repr(job_id)).replace("CHILD_SOURCE", repr(child))


def reader(root, job_id):
    path = job_path(root, job_id)
    return """import json
from pathlib import Path
job=Path(JOB_PATH)
if (job/'receipt.json').is_file():
 print((job/'receipt.json').read_text(),flush=True)
else:
 print(json.dumps({'state':'running' if job.exists() else 'absent'}),flush=True)
""".replace("JOB_PATH", repr(path))
