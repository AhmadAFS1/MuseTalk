#!/usr/bin/env python3
"""Record and verify the immutable image runtime; never repairs it at boot."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SCHEMA = 'lingua-character-runtime/1'

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def command(args):
    return subprocess.check_output([str(x) for x in args], text=True, stderr=subprocess.PIPE).strip()

def snapshot(repo, workspace, model_lock):
    repo, workspace = Path(repo), Path(workspace)
    lock_path = repo/'character_factory/character_worker/locks/character_worker.lock.json'
    lock = json.loads(lock_path.read_text()); h3 = workspace/'minimax-h3-workflows'
    models = json.loads(Path(model_lock).read_text())
    for name, spec in models['files'].items():
        path = repo/name
        if path.stat().st_size != spec['size_bytes'] or digest(path) != spec['sha256']:
            raise ValueError(f'preparation weight differs: {name}')
    forbidden = ('models/musetalkV15/unet.pth', 'models/whisper/pytorch_model.bin')
    if any((repo/name).exists() for name in forbidden):
        raise ValueError('teacher/Whisper weights must not be included in the preparation image')
    source = command(['git','-C',repo,'rev-parse','HEAD'])
    tracked = command(['git','-C',repo,'ls-files','-z']).split('\0')
    source_files = {name:digest(repo/name) for name in tracked if name and (repo/name).is_file()}
    command(['git','-C',h3,'diff','--exit-code','HEAD','--'])
    command(['git','-C',workspace/'ComfyUI','diff','--exit-code','HEAD','--'])
    h3_rev = command(['git','-C',h3,'rev-parse','HEAD'])
    if h3_rev != lock['h3_workflows']['commit']:
        raise ValueError('H3 source revision differs from worker lock')
    comfy = json.loads((h3/'locks/comfyui.lock.json').read_text())['commit']
    if command(['git','-C',workspace/'ComfyUI','rev-parse','HEAD']) != comfy:
        raise ValueError('ComfyUI revision differs from H3 lock')
    environments = {}
    for name in ('musetalk_trt_stagewise','comfy-h3','musetalk_chin_tools'):
        py = workspace/'.venvs'/name/'bin/python'
        command([py,'-m','pip','check'])
        environments[name] = {'python':command([py,'-c','import platform; print(platform.python_version())']),
                              'packages':command([py,'-m','pip','freeze','--all']).splitlines()}
    # CUDA is deliberately hidden during image construction; boot doctor checks it.
    command([workspace/'.venvs/musetalk_trt_stagewise/bin/python','-c',
             'import torch,cv2,numpy,boto3,PIL,fastapi,multipart; from mmcv.ops import get_compiling_cuda_version; assert str(get_compiling_cuda_version()).startswith("12.1"); from musetalk.utils.face_parsing import FaceParsing; from musetalk.utils.preprocessing import get_landmark_and_bbox; from scripts.avatar_prepare_server import create_app'])
    command([workspace/'.venvs/musetalk_chin_tools/bin/python','-c','import mediapipe, cv2, numpy'])
    return {'schema':SCHEMA,'source_revision':source,'source_files':source_files,'worker_lock_sha256':digest(lock_path),
            'h3_revision':h3_rev,'comfyui_revision':comfy,
            'h3_model_profile':lock['h3_workflows']['model_profile'],
            'h3_helpers':{name:digest(workspace/'minimax-h3'/name) for name in lock['h3_workflows']['lib_copies']},
            'h3_model_paths_sha256':digest(workspace/'minimax-h3/extra_model_paths_h3.yaml'),
            'h3_locks':{str(p.relative_to(h3)):digest(p) for p in sorted((h3/'locks').glob('*')) if p.is_file()},
            'preparation_model_manifest_sha256':digest(model_lock),
            'preparation_models':models['files'],'environments':environments,
            'os_packages':command(['dpkg-query','-W','-f=${Package}=${Version}\n']).splitlines(),
            'preparation_mode':'avatar-preparation/1','teacher_renderer_included':False,
            'publication_schema':'lingua-character-release/2','qc_contract':'lingua-preprocessing-qc/1',
            'phone_student_unet':'unet_bk_tiny_r2_selected_int8w_b1',
            'phone_student_checkpoint_sha256':'a05d2d98f69b431252458d3b25f72e395d557ba693821baee91d11f5fe5a96cd',
            'source_fps':24,'phone_render_fps':25,'image_digest_location':'external pinned deployment descriptor'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['record','verify']);ap.add_argument('--repo',required=True);ap.add_argument('--workspace',default='/workspace');ap.add_argument('--manifest',default='/opt/character-runtime/manifest.json');ap.add_argument('--models',required=True);ap.add_argument('--expected-sha256');a=ap.parse_args()
    os.chdir(a.repo)
    target=Path(a.manifest)
    if a.mode=='verify' and a.expected_sha256 and digest(target)!=a.expected_sha256:
        raise ValueError('runtime manifest digest differs from deployment pin')
    actual=snapshot(a.repo,a.workspace,a.models)
    if a.mode=='record':
        target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(actual,sort_keys=True,indent=2)+'\n')
    elif json.loads(target.read_text())!=actual:
        raise ValueError('prebuilt runtime differs from manifest; repair requires a separately built image')
    print(json.dumps({'schema':SCHEMA,'verified':True,'sha256':digest(target),'source_revision':actual['source_revision']}))
if __name__=='__main__': main()
