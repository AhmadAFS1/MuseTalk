#!/usr/bin/env python3
"""CPU filesystem measurements only; capacity is the CI host, not a Vast offer."""
import json
import os
import subprocess
from pathlib import Path
paths=['/','/opt/musetalk/venv','/opt/musetalk/app','/opt/character-runtime/app','/workspace/.venvs/comfy-h3',
       '/workspace/.venvs/musetalk_chin_tools','/workspace/ComfyUI','/workspace/minimax-h3-workflows']
rows={}
for path in paths:
    if not Path(path).exists():continue
    result=subprocess.run(['du','-sx','-B1',path],capture_output=True,text=True)
    rows[path]={'allocated_bytes':int(result.stdout.split()[0]),'du_exit_code':result.returncode}
f=os.statvfs('/workspace')
print(json.dumps({'schema':'lingua-character-image-footprint/1','measurement':'CPU container; df capacity belongs to CI host',
                  'allocated_paths':rows,'workspace_df':{'total_bytes':f.f_frsize*f.f_blocks,'available_bytes':f.f_frsize*f.f_bavail}},indent=2))
