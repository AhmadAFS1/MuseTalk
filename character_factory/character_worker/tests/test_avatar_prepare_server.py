"""Exercise the actual preparation HTTP surface without loading CUDA models."""
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock

from fastapi.testclient import TestClient

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from scripts.avatar_prepare_server import create_app


class PreparationApiTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        previous = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, previous)
        self.manager = Mock()
        self.manager.stats.return_value = {
            'mode': 'avatar-preparation/1', 'teacher_loaded': False,
            'whisper_loaded': False, 'renderer_available': False,
        }
        self.factory = Mock(return_value=self.manager)
        self.client = self.enterContext(TestClient(create_app(self.factory)))

    def post(self, identity='new_idle', data=b'source-video', **params):
        return self.client.post('/avatars/prepare', params={'avatar_id': identity, **params},
                                files={'video_file': ('untrusted-name.mp4', data, 'video/mp4')})

    def test_startup_exposes_preparation_and_no_live_rendering_routes(self):
        self.factory.assert_called_once_with()
        self.assertEqual(self.client.get('/stats').json(), self.manager.stats.return_value)
        paths = self.client.get('/openapi.json').json()['paths']
        self.assertEqual(set(paths), {'/stats', '/avatars/prepare'})
        for endpoint in ('/render', '/inference', '/sessions', '/render_batch'):
            self.assertEqual(self.client.post(endpoint).status_code, 404)

    def test_multipart_upload_and_options_reach_manager_without_teacher_input(self):
        received = {}
        def prepare(avatar_id, source, idle, batch_size, force):
            received.update(identity=avatar_id, source=source.read_bytes(), idle=idle.read_bytes(),
                            batch=batch_size, force=force)
            return {'status': 'success', 'avatar_id': avatar_id, 's3_uploaded': True,
                    'preprocessing_contract': 'avatar-preparation/1', 'teacher_executed': False}
        self.manager.prepare.side_effect = prepare
        response = self.client.post('/avatars/prepare',
            params={'avatar_id': 'new_talking', 'batch_size': 4, 'force_recreate': True},
            files={'video_file': ('../../outside.mp4', b'talking-source', 'video/mp4'),
                   'idle_video_file': ('idle.mp4', b'idle-source', 'video/mp4')})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(received, dict(identity='new_talking', source=b'talking-source',
                                       idle=b'idle-source', batch=4, force=True))
        self.assertFalse(response.json()['teacher_executed'])
        self.assertFalse(Path('../outside.mp4').exists())

    def test_invalid_identity_and_geometry_never_enter_preparation(self):
        for identity in ('../escape', 'bad/path', '.hidden', 'avatar..copy', 'a' * 129):
            with self.subTest(identity=identity):
                self.assertEqual(self.post(identity).status_code, 400)
        for params in ({'bbox_shift': 1}, {'batch_size': 0}, {'batch_size': 33}):
            with self.subTest(params=params):
                self.assertEqual(self.post(**params).status_code, 400)
        self.manager.prepare.assert_not_called()

    def test_failed_persistence_is_not_reported_as_success_or_leaked_to_response(self):
        self.manager.prepare.side_effect = RuntimeError('private credential material')
        response = self.post()
        self.assertEqual(response.status_code, 500)
        self.assertNotIn('private credential', response.text)
        self.assertNotIn('s3_uploaded', response.json())

    def test_overlapping_same_identity_requests_keep_their_own_source_bytes(self):
        entered = threading.Event()
        release = threading.Event()
        count_lock = threading.Lock()
        calls = 0
        def prepare(_avatar, source, _idle, _batch, _force):
            nonlocal calls
            with count_lock:
                calls += 1
                first = calls == 1
            if first:
                entered.set()
                if not release.wait(5):
                    raise TimeoutError('test did not release first request')
            return {'source': source.read_bytes().decode()}
        self.manager.prepare.side_effect = prepare
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(self.post, data=b'first-source')
            self.assertTrue(entered.wait(5))
            try:
                second = executor.submit(self.post, data=b'second-source').result(timeout=5)
            finally:
                release.set()
            self.assertEqual(first.result(timeout=5).json(), {'source': 'first-source'})
            self.assertEqual(second.json(), {'source': 'second-source'})


if __name__ == '__main__':
    unittest.main()
