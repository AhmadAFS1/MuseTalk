"""Canary source-delivery fixtures only; no venv/pip/CUDA or live network."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'scripts'))
import dependency_fresh_canary_ec2 as canary
import dependency_overlap_worker as overlap
import dependency_s3_full_worker as full_s3
import dependency_link_screen_worker as link_screen
import dependency_final_screen_worker as final_worker
import dependency_final_screen_ec2 as final_control


class FinalScreenSafetyTests(unittest.TestCase):
    def test_final_screen_refuses_unowned_root_before_writes(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(final_worker.platform,'system',return_value='Linux'), \
             patch.object(final_worker.platform,'machine',return_value='x86_64'):
            root=Path(folder).resolve()
            with self.assertRaisesRegex(ValueError,'Exact owned'):
                final_worker.prepare(root,'2099-01-01T00:00:00+00:00')
            self.assertEqual(list(root.iterdir()),[])

    def test_final_screen_expired_lease_refused_before_context(self):
        with patch.object(final_worker,'context') as context:
            with self.assertRaisesRegex(ValueError,'lease'):
                final_worker.links(final_worker.ROOT,'2000-01-01T00:00:00+00:00')
            context.assert_not_called()

    def test_private_payloads_compile_without_dispatch(self):
        deadline='2099-01-01T00:00:00+00:00'
        for mode in ('prepare','links','acquire'):
            source=final_control.job_source(mode,deadline,*(({}, {'url':'private'}) if mode=='acquire' else ()))
            compile(source,'<fixture-only>','exec')
            self.assertIn('Path('+repr(final_control.WORKER_ROOT)+')',source)
        with self.assertRaises(ValueError):final_control.job_source('unknown',deadline)
        with self.assertRaises(ValueError):final_control.job_source('acquire',deadline)

    def test_final_stock_script_omits_clone_and_application(self):
        class Library:
            def frozen_template(self,template):
                return {'NVIDIA_VISIBLE_DEVICES':'all'},'must-not-run',{}
        env,script,receipt=final_control.freeze(Library(),{})
        self.assertNotIn('must-not-run',script)
        self.assertNotIn('git clone',script)
        self.assertNotIn('vast_onstart',script)
        self.assertFalse(receipt['production_template_modified'])
        self.assertEqual(env,{'NVIDIA_VISIBLE_DEVICES':'all'})

    def test_final_export_keeps_failure_and_cleanup_not_a_false_pass(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();state=root/'state';state.mkdir();tools=root/'tools';tools.mkdir()
            trial=state/'evidence/link-screen';trial.mkdir(parents=True)
            (trial/'receipt.json').write_text(json.dumps({'state':'failed','detail':'fixture'}))
            (state/'cleanup.json').write_text(json.dumps({'provider_absence_verified':True}))
            with patch.object(final_control,'ROOT',state),patch.object(final_control.c,'HERE',tools), \
                 patch.object(final_control.os,'chown'),contextlib.redirect_stdout(io.StringIO()):
                final_control.export(SimpleNamespace(utc=lambda:'fixture'))
            result=json.loads((tools/'final-screen-public-results.json').read_text())
            self.assertFalse(result['goal_complete'])
            self.assertEqual(result['trials']['link-screen']['state'],'failed')
            self.assertTrue(result['cleanup']['provider_absence_verified'])

    def test_final_export_refuses_signed_credentials(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder).resolve();state=root/'state';state.mkdir();tools=root/'tools';tools.mkdir()
            (state/'ledger.json').write_text(json.dumps({'unapproved':'X-Amz-Signature=fixture'}))
            with patch.object(final_control,'ROOT',state),patch.object(final_control.c,'HERE',tools):
                with self.assertRaisesRegex(ValueError,'Credential pattern'):
                    final_control.export(SimpleNamespace(utc=lambda:'fixture'))
            self.assertFalse((tools/'final-screen-public-results.json').exists())


class CanarySourceTests(unittest.TestCase):
    def fixture(self, root):
        repo = root / 'repo'; repo.mkdir()
        for name in canary.FILES:
            path = repo / name; path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('reviewed ' + name)
        return repo

    def execute(self, repo, worker_root, raw, sha):
        with patch.object(canary, 'WORKER_ROOT', str(worker_root)):
            code = canary.bootstrap_program(sha)
        code = code.replace("pathlib.Path('/workspace/MuseTalk')", f'pathlib.Path({str(repo)!r})')
        with patch.dict('os.environ', {'MUSETALK_CANARY_SOURCE_TICKET': 'https://private.example/redacted'}), \
             patch('urllib.request.urlopen', return_value=io.BytesIO(raw)), \
             contextlib.redirect_stdout(io.StringIO()):
            exec(compile(code, '<source-fixture>', 'exec'), {})

    def test_deterministic_allowlisted_source_and_recoverable_overlay(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); repo = self.fixture(root)
            raw, sha = canary.source_bundle(repo)
            self.assertEqual((raw, sha), canary.source_bundle(repo))
            self.assertEqual(set(json.loads(raw)['files']), set(canary.FILES))
            old = repo / canary.FILES[0]; old.write_text('original source')
            worker = root / 'worker'; self.execute(repo, worker, raw, sha)
            self.assertEqual(old.read_text(), 'reviewed ' + canary.FILES[0])
            self.assertEqual((worker / 'source-original' / canary.FILES[0]).read_text(), 'original source')
            self.assertEqual(json.loads((worker / 'source-overlay.json').read_text())['sha256'], sha)

    def test_corrupt_or_unreviewed_payload_cannot_write(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); repo = self.fixture(root)
            raw, sha = canary.source_bundle(repo)
            extra = json.loads(raw); extra['files']['../escape'] = next(iter(extra['files'].values()))
            extra = json.dumps(extra).encode()
            for payload, digest in ((raw + b'bad', sha), (extra, hashlib.sha256(extra).hexdigest())):
                with self.assertRaises(SystemExit):
                    self.execute(repo, root / 'worker', payload, digest)
                self.assertFalse((root / 'worker').exists())
                self.assertFalse((root / 'escape').exists())

    def test_source_symlink_rejected_before_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder).resolve(); repo = self.fixture(root)
            target = repo / canary.FILES[0]; target.unlink(); target.symlink_to(repo / canary.FILES[1])
            with self.assertRaisesRegex(ValueError, 'reviewed source'):
                canary.source_bundle(repo)


class OverlapSafetyTests(unittest.TestCase):
    def test_link_screen_refuses_unowned_root_before_writes(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(link_screen.platform,'system',return_value='Linux'), \
             patch.object(link_screen.platform,'machine',return_value='x86_64'):
            root=Path(folder).resolve()
            with self.assertRaisesRegex(ValueError,'Owned canary'):
                link_screen.execute(root,'2099-01-01T00:00:00+00:00')
            self.assertEqual(list(root.iterdir()),[])

    def test_unreviewed_restore_root_cannot_mutate(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(full_s3.platform,'system',return_value='Linux'), \
             patch.object(full_s3.platform,'machine',return_value='x86_64'):
            root=Path(folder).resolve()
            with self.assertRaisesRegex(ValueError,'experiment root'):
                full_s3.execute({}, {}, '2099-01-01T00:00:00+00:00',root=root)
            self.assertEqual(list(root.iterdir()),[])

    def ticket(self):
        return {'key':overlap.KEY,'sha256':overlap.SHA,'size':overlap.SIZE,
                'owner':overlap.fetch.OWNER,'url':'https://'+overlap.fetch.BUCKET+'.s3.amazonaws.com/'+overlap.KEY}

    def test_arbitrary_private_engine_targets_rejected(self):
        for key,value in (('key','other'),('sha256','0'*64),('url','https://example.com/engine'),('owner','other')):
            ticket=self.ticket();ticket[key]=value
            with self.assertRaises(ValueError): overlap.engine_ticket(ticket)

    def test_truncated_engine_not_published(self):
        class Response(io.BytesIO):
            status=200
            headers={'Content-Length':str(overlap.SIZE)}
        with tempfile.TemporaryDirectory() as folder, patch('urllib.request.urlopen',return_value=Response(b'short')):
            root=Path(folder)
            with self.assertRaisesRegex(ValueError,'SHA256'):
                overlap.download_engine(self.ticket(),root)
            self.assertEqual(list(root.iterdir()),[])


if __name__ == '__main__':
    unittest.main()
