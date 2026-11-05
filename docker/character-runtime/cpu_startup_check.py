#!/usr/bin/env python3
"""Actual image CPU startup/TLS checks. No CUDA, real generation or S3 claims."""
import argparse
import base64
import json
import os
from pathlib import Path
import socket
import ssl
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

REPO=Path('/opt/character-runtime/app')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',required=True);a=ap.parse_args()
    os.chdir(REPO)
    with tempfile.TemporaryDirectory(prefix='character-image-cpu-') as tmp:
        root=Path(tmp);cert,key=root/'cert.pem',root/'key.pem'
        subprocess.run(['openssl','req','-x509','-newkey','ec','-pkeyopt','ec_paramgen_curve:P-256','-nodes','-days','1',
                        '-subj','/CN=cpu.character.lingua.internal','-addext','subjectAltName=DNS:cpu.character.lingua.internal,IP:127.0.0.1',
                        '-addext','basicConstraints=critical,CA:false','-addext','extendedKeyUsage=serverAuth',
                        '-keyout',str(key),'-out',str(cert)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        env=dict(os.environ,CHARACTER_WORKER_REQUIRE_TLS='1',CHARACTER_WORKER_TOKEN='ephemeral-cpu-test-token',
                 CHARACTER_WORKER_TLS_CERT_B64=base64.b64encode(cert.read_bytes()).decode(),
                 CHARACTER_WORKER_TLS_KEY_B64=base64.b64encode(key.read_bytes()).decode())
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        worker=root/'worker';command=[sys.executable,str(REPO/'character_factory/character_worker/character_api.py'),
                                     '--fake','--fake-seconds','.01','--host','127.0.0.1','--port',str(port),'--root',str(worker),'--workspace','/workspace']
        context=ssl.create_default_context(cafile=str(cert));base='https://127.0.0.1:'+str(port)
        def request(path,body=None,token=None,ctx=context):
            req=urllib.request.Request(base+path,data=None if body is None else json.dumps(body).encode(),
                                       headers={'Content-Type':'application/json',**({'Authorization':'Bearer '+token} if token else {})})
            with urllib.request.urlopen(req,context=ctx,timeout=3) as response:return response.status,json.load(response)
        log=(root/'worker.log').open('w')
        proc=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT)
        try:
            end=time.monotonic()+20
            while True:
                if proc.poll() is not None:raise RuntimeError('actual image API startup failed')
                try:
                    _,health=request('/health')
                    if health['ready']:break
                except (OSError,urllib.error.URLError):pass
                if time.monotonic()>end:raise RuntimeError('actual image API did not become ready')
                time.sleep(.1)
            try:request('/characters');raise AssertionError('missing bearer accepted')
            except urllib.error.HTTPError as exc:assert exc.code==401
            payload={'character_id':'cpu_image_fixture','portrait':{'base64':base64.b64encode(b'fake-only').decode()},
                     'options':{'fake_seconds':.01,'max_attempts':1}}
            status,result=request('/createCharacter',payload,'ephemeral-cpu-test-token');assert status==202
            job_id=result['job_id'];end=time.monotonic()+10
            while True:
                _,job=request('/characters/'+job_id,token='ephemeral-cpu-test-token')
                if job['status']=='done':break
                assert job['status'] in ('queued','running'),job['status']
                assert time.monotonic()<end,'fake protocol job timed out'
                time.sleep(.1)
            assert job['result']['fake'] is True
            assert not (worker/'control-tls/server.key').exists()
            assert (worker/'control-tls/server.crt').stat().st_mode&0o777==0o600
            try:urllib.request.urlopen(base.replace('https:','http:')+'/health',timeout=3);raise AssertionError('HTTP accepted')
            except (OSError,urllib.error.URLError):pass
            try:request('/health',ctx=ssl.create_default_context());raise AssertionError('unpinned TLS accepted')
            except (OSError,urllib.error.URLError):pass
        finally:
            proc.terminate();proc.wait(timeout=10);log.close()
        missing=dict(env);missing.pop('CHARACTER_WORKER_TLS_CERT_B64');missing.pop('CHARACTER_WORKER_TLS_KEY_B64')
        invalid=subprocess.run(command,env=missing,capture_output=True,timeout=10)
        assert invalid.returncode!=0,'production API silently accepted missing TLS'
        check=[sys.executable,str(REPO/'character_factory/character_worker/runtime_manifest.py'),'verify','--repo',str(REPO),
               '--models',str(REPO/'docker/character-runtime/preparation-models.json'),'--expected-sha256','0'*64]
        invalid=subprocess.run(check,capture_output=True,timeout=10)
        assert invalid.returncode!=0 and b'manifest digest differs' in invalid.stderr
        manifest=json.load(open('/opt/character-runtime/manifest.json'))
        report={'schema':'lingua-character-image-cpu-qualification/1','status':'passed','source_revision':manifest['source_revision'],
                'python_version':sys.version.split()[0],'network':'none; loopback only',
                'checks':{k:'passed' for k in ('actual_api_main_tls_startup','trusted_tls_health','bearer_enforced','fake_create_poll',
                           'private_key_removed','plaintext_refused','untrusted_certificate_refused','missing_tls_refused','bad_manifest_pin_refused')},
                'gpu_qualification':'not_run','student_rendering':'not_run','s3_publication':'not_run'}
        Path(a.report).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
