"""Exercise real TLS and trusted certificate validation on the worker control port."""
import base64
import json
import os
from pathlib import Path
import ssl
import subprocess
import sys
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from unittest.mock import Mock,patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import character_api as api
import pipeline

class ControlTLSTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        for name in ('server','wrong'):
            subprocess.run(['openssl','req','-x509','-newkey','rsa:2048','-nodes','-days','1','-subj','/CN=localhost',
                            '-addext','subjectAltName=DNS:localhost,IP:127.0.0.1','-keyout',str(cls.root/(name+'.key')),
                            '-out',str(cls.root/(name+'.crt'))],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()
    def tls_env(self):
        return {'CHARACTER_WORKER_REQUIRE_TLS':'1',
                'CHARACTER_WORKER_TLS_CERT_B64':base64.b64encode((self.root/'server.crt').read_bytes()).decode(),
                'CHARACTER_WORKER_TLS_KEY_B64':base64.b64encode((self.root/'server.key').read_bytes()).decode()}
    def test_real_control_tls_validates_certificate_and_preserves_bearer_auth(self):
        env=self.tls_env();context=api.tls_context_for(self.root/'worker',env)
        self.assertNotIn('CHARACTER_WORKER_TLS_KEY_B64',env);self.assertFalse((self.root/'worker/control-tls/server.key').exists())
        self.assertEqual((self.root/'worker/control-tls/server.crt').stat().st_mode&0o777,0o600)
        manager=Mock();manager.worker_state.return_value={'ready':True,'boot':{}};manager.list.return_value=[]
        server=api.make_server('127.0.0.1',0,manager,[lambda:'bearer-secret'],False,True,tls_context=context)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start();url='https://127.0.0.1:'+str(server.server_address[1])
        trusted=ssl.create_default_context(cafile=str(self.root/'server.crt'))
        try:
            with urllib.request.urlopen(url+'/health',context=trusted,timeout=3) as response:self.assertEqual(json.load(response),{'ready':True,'boot':{}})
            with self.assertRaises(urllib.error.HTTPError) as unauthorized:urllib.request.urlopen(url+'/characters',context=trusted,timeout=3)
            self.assertEqual(unauthorized.exception.code,401)
            request=urllib.request.Request(url+'/characters',headers={'Authorization':'Bearer bearer-secret'})
            with urllib.request.urlopen(request,context=trusted,timeout=3) as response:self.assertEqual(response.status,200)
            wrong=ssl.create_default_context(cafile=str(self.root/'wrong.crt'))
            with self.assertRaises(urllib.error.URLError):urllib.request.urlopen(url+'/health',context=wrong,timeout=3)
            with self.assertRaises(Exception):urllib.request.urlopen(url.replace('https:','http:')+'/health',timeout=3)
        finally:server.shutdown();server.server_close();thread.join(3)
    def test_tls_required_partial_and_malformed_inputs_fail_closed(self):
        for env in ({'CHARACTER_WORKER_REQUIRE_TLS':'1'}, {'CHARACTER_WORKER_TLS_CERT_B64':'YQ=='},
                    {'CHARACTER_WORKER_TLS_CERT_B64':'bad','CHARACTER_WORKER_TLS_KEY_B64':'bad'}):
            with self.assertRaises(ValueError):api.tls_context_for(self.root/'invalid',env)
        self.assertIsNone(api.tls_context_for(self.root/'legacy',{}))
    def test_control_private_key_never_reaches_generation_subprocess_environment(self):
        with patch.dict(os.environ,{'CHARACTER_WORKER_TLS_KEY_B64':'private','CHARACTER_WORKER_TOKEN':'control'}):
            rt=pipeline.Runtime(workspace=self.root,root=self.root)
            self.assertNotIn('CHARACTER_WORKER_TLS_KEY_B64',rt.env())
if __name__=='__main__':unittest.main()
