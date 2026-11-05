import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
MODULE=Path(__file__).resolve().parents[1]/'runtime_manifest.py'
spec=importlib.util.spec_from_file_location('runtime_manifest',MODULE);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)

class RuntimeManifestTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name);self.repo=self.root/'app';self.workspace=self.root/'workspace';self.repo.mkdir();self.workspace.mkdir()
        h3=self.workspace/'minimax-h3-workflows';(h3/'locks').mkdir(parents=True);(h3/'locks/comfyui.lock.json').write_text(json.dumps({'commit':'comfy'}));(self.workspace/'minimax-h3').mkdir();(self.workspace/'minimax-h3/helper.py').write_text('original');(self.workspace/'minimax-h3/extra_model_paths_h3.yaml').write_text('paths');
        lock=self.repo/'character_factory/character_worker/locks/character_worker.lock.json';lock.parent.mkdir(parents=True);lock.write_text(json.dumps({'h3_workflows':{'commit':'h3','model_profile':'character-fl2va','lib_copies':['helper.py']}}))
        (self.repo/'source.py').write_text('source');(self.repo/'weight').write_bytes(b'weight');self.models=self.repo/'models.json';self.models.write_text(json.dumps({'files':{'weight':{'size_bytes':6,'sha256':mod.digest(self.repo/'weight')}}}))
        self.packages='package==1.0'
    def command(self,args):
        args=list(map(str,args))
        if args[-1]=='HEAD':return 'h3' if 'minimax-h3-workflows' in args[2] else ('comfy' if 'ComfyUI' in args[2] else 'source')
        if args[-1]=='-z':return 'source.py\0'
        if 'freeze' in args:return self.packages
        if 'platform.python_version' in args[-1]:return '3.10.12'
        return ''
    def snapshot(self):
        with patch.object(mod,'command',side_effect=self.command):return mod.snapshot(self.repo,self.workspace,self.models)
    def test_snapshot_records_actual_models_dependencies_and_no_teacher_claim(self):
        x=self.snapshot();self.assertEqual(x['h3_model_profile'],'character-fl2va');self.assertFalse(x['teacher_renderer_included']);self.assertEqual(x['publication_schema'],'lingua-character-release/2');self.assertEqual(x['phone_render_fps'],25)
        self.assertEqual(x['environments']['comfy-h3']['packages'],['package==1.0']);self.assertEqual(x['h3_helpers']['helper.py'],mod.digest(self.workspace/'minimax-h3/helper.py'))
    def test_same_git_revision_does_not_hide_source_or_package_drift(self):
        original=self.snapshot();(self.repo/'source.py').write_text('modified');self.assertNotEqual(self.snapshot(),original);(self.repo/'source.py').write_text('source');self.packages='package==2.0';self.assertNotEqual(self.snapshot(),original)
    def test_bad_weight_or_teacher_inclusion_fails_before_readiness(self):
        (self.repo/'weight').write_bytes(b'tamper')
        with self.assertRaisesRegex(ValueError,'weight differs'):self.snapshot()
        (self.repo/'weight').write_bytes(b'weight');teacher=self.repo/'models/musetalkV15/unet.pth';teacher.parent.mkdir(parents=True);teacher.write_bytes(b'teacher')
        with self.assertRaisesRegex(ValueError,'teacher/Whisper'):self.snapshot()
    def test_h3_and_helper_drift_cannot_match_the_recorded_runtime(self):
        original=self.snapshot();(self.workspace/'minimax-h3/helper.py').write_text('tampered');self.assertNotEqual(self.snapshot(),original)
        with patch.object(mod,'command',side_effect=subprocess.CalledProcessError(1,['git','diff'])):
            with self.assertRaises(subprocess.CalledProcessError):mod.snapshot(self.repo,self.workspace,self.models)
if __name__=='__main__':unittest.main()
