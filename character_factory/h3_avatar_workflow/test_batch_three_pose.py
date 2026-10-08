"""Focused checks for batch planning and proof of S3 publication."""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import threading
import types
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from batch_three_pose import plan, process_jobs
from create_avatar import prompt as accepted_talking_prompt
from three_pose_s3_stage import publish_one
from three_pose_prompts import pose_prompt


class BatchThreePoseTest(unittest.TestCase):
    def test_four_language_roster_preserves_each_portraits_clothes_and_room(self):
        roster = json.loads((HERE / 'config/lumatalk_four_language_pilot_20260928.json').read_text())
        self.assertEqual(len(roster['avatars']), 16)
        for avatar in roster['avatars']:
            for pose in ('idle', 'talking', 'smiling'):
                with self.subTest(avatar=avatar['id'], pose=pose):
                    value = pose_prompt(avatar, pose)
                    self.assertIn(avatar['wardrobe'], value)
                    if pose == 'talking':
                        self.assertIn(avatar['background'], value)
                        self.assertNotIn('navy top', value)

    def test_plan_keeps_accepted_talking_prompt_and_rejects_changed_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            photos = root / 'photos'
            photos.mkdir()
            portrait = photos / 'jane.png'
            portrait.write_bytes(b'portrait bytes')
            profile = dict(id='jane', label='Japanese woman', age=26, gender='woman',
                           hair='straight black hair', tone='warm-light skin', facial_hair='none')
            metadata = root / 'meta.json'
            metadata.write_text(json.dumps({'images': {'jane.png': profile}}))
            args = argparse.Namespace(images=photos, output=root / 'out', metadata=metadata,
                                      workspace=Path('/workspace'), limit=None, only=None)
            entries = plan(args)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]['prompts']['talking'], accepted_talking_prompt(profile))
            self.assertEqual(set(entries[0]['prompts']), {'idle', 'talking', 'smiling'})
            profile['age'] = 27
            metadata.write_text(json.dumps({'images': {'jane.png': profile}}))
            with self.assertRaisesRegex(RuntimeError, 'changed'):
                plan(args)

    def test_s3_receipt_requires_exact_upload_ack_and_matching_stats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root / 'jane' / 'idle'
            out.mkdir(parents=True)
            (out / 'source.mp4').write_bytes(b'fixture video source')
            spec = dict(id='jane_12345678', pose='idle', signature='fixture-signature',
                        output=str(out), workspace=str(root))
            before = dict(enabled=True, bucket='test-bucket', prefix='avatars',
                          version='v15', upload_successes=0, last_upload_key=None)
            from common import sha
            avatar_id = f"{spec['id']}_idle_{sha(out / 'source.mp4')[:10]}"
            key = f'avatars/v15/{avatar_id}.tar.gz'
            after = dict(before, upload_successes=1, last_upload_key=key)
            module = types.ModuleType('prepare_musetalk_avatars')
            requests = []

            def successful_prepare(**kwargs):
                requests.append(kwargs)
                return dict(status='success', avatar_id=avatar_id,
                            already_prepared=False, s3_uploaded=True, s3_key=key)

            module.prepare_one = successful_prepare
            with patch.dict(sys.modules, {'prepare_musetalk_avatars': module}), \
                 patch('three_pose_s3_stage.stats', side_effect=[before, after]):
                receipt = publish_one(spec, 'http://fake-api')
            self.assertEqual(receipt['s3_uri'], f's3://test-bucket/{key}')
            self.assertTrue(requests[0]['force_recreate'])
            self.assertIsNone(requests[0]['idle_video'])
            self.assertEqual(receipt['upload_successes_after'], 1)

            (out / 's3.json').unlink()

            def failed_prepare(**kwargs):
                return dict(status='success', avatar_id=avatar_id,
                            already_prepared=False, s3_uploaded=False, s3_key=None)

            module.prepare_one = failed_prepare
            with patch.dict(sys.modules, {'prepare_musetalk_avatars': module}), \
                 patch('three_pose_s3_stage.stats', side_effect=[before, after]):
                with self.assertRaisesRegex(RuntimeError, 'not confirmed'):
                    publish_one(spec, 'http://fake-api')
            self.assertFalse((out / 's3.json').exists())

            module.prepare_one = successful_prepare
            unchanged_stats = dict(before, last_upload_key=key)
            with patch.dict(sys.modules, {'prepare_musetalk_avatars': module}), \
                 patch('three_pose_s3_stage.stats', side_effect=[before, unchanged_stats]):
                with self.assertRaisesRegex(RuntimeError, 'not confirmed'):
                    publish_one(spec, 'http://fake-api')
            self.assertFalse((out / 's3.json').exists())

    def test_repeated_photo_in_subfolders_gets_distinct_cache_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            photos = root / 'photos'
            for folder in ('set_a', 'set_b'):
                path = photos / folder / 'portrait.jpg'
                path.parent.mkdir(parents=True)
                path.write_bytes(b'same photo in both folders')
            args = argparse.Namespace(images=photos, output=root / 'out', metadata=None,
                                      workspace=Path('/workspace'), limit=None, only=None)
            entries = plan(args)
            self.assertEqual(len(entries), 2)
            self.assertEqual(len({x['id'] for x in entries}), 2)

    def test_published_identity_resumes_without_local_videos_or_api(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / 'jane'
            target.mkdir()
            entry = dict(id='jane', output=str(target), signature='unchanged')
            (target / 'published.json').write_text(json.dumps({
                'status': 'complete', 'signature': 'unchanged',
                'poses': {pose: {} for pose in ('idle', 'talking', 'smiling')},
            }))
            args = argparse.Namespace(stage='prepare', local_api=True, workspace=root)
            with patch('batch_three_pose.local_api', side_effect=AssertionError('API should not start')):
                self.assertEqual(process_jobs([entry], args), [])

    def test_real_multipart_client_and_s3_handshake_over_http(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            idle = root / 'jane' / 'idle'
            talking = root / 'jane' / 'talking'
            idle.mkdir(parents=True)
            talking.mkdir()
            (idle / 'source.mp4').write_bytes(b'idle fixture')
            (talking / 'source.mp4').write_bytes(b'talking fixture')
            spec = dict(id='jane_12345678', pose='talking', signature='http-fixture',
                        output=str(talking), workspace='/workspace')
            state = dict(enabled=True, bucket='test-bucket', prefix='avatars',
                         version='v15', upload_successes=0, last_upload_key=None)
            requests = []

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_args):
                    pass

                def send_json(self, value):
                    payload = json.dumps(value).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Content-Length', str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

                def do_GET(self):
                    self.send_json({'avatar_s3': state})

                def do_POST(self):
                    query = parse_qs(urlsplit(self.path).query)
                    body = self.rfile.read(int(self.headers['Content-Length']))
                    requests.append((query, body))
                    avatar_id = query['avatar_id'][0]
                    key = f'avatars/v15/{avatar_id}.tar.gz'
                    state.update(upload_successes=1, last_upload_key=key)
                    self.send_json(dict(status='success', avatar_id=avatar_id,
                                        already_prepared=False, s3_uploaded=True, s3_key=key))

            server = HTTPServer(('127.0.0.1', 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                receipt = publish_one(spec, f'http://127.0.0.1:{server.server_port}')
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
            self.assertEqual(len(requests), 1)
            query, body = requests[0]
            self.assertEqual(query['force_recreate'], ['true'])
            self.assertIn(b'name="video_file"', body)
            self.assertNotIn(b'name="idle_video_file"', body)
            self.assertIn(b'talking fixture', body)
            self.assertNotIn(b'idle fixture', body)
            self.assertEqual(receipt['s3_uri'], f"s3://test-bucket/{state['last_upload_key']}")


if __name__ == '__main__':
    unittest.main()
