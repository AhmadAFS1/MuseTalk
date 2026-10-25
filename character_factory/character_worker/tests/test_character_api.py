#!/usr/bin/env python3
"""Worker API tests with the fake pipeline (no GPU, no S3). Run: python3 -m unittest tests.test_character_api -v"""
from __future__ import annotations

import base64
import json
import os
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
import sys  # noqa: E402
sys.path.insert(0, str(HERE.parent))
import character_api  # noqa: E402
import pipeline  # noqa: E402

PNG_1X1 = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==')


def http(method: str, url: str, body=None, token: str | None = None, raw: bool = False):
    data = None if body is None else json.dumps(body).encode()
    headers = {'Content-Type': 'application/json'}
    if token:
        headers['Authorization'] = f'Bearer {token}'
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            payload = resp.read()
            return resp.status, (payload if raw else json.loads(payload or b'null'))
    except urllib.error.HTTPError as exc:
        payload = exc.read()
        return exc.code, (payload if raw else json.loads(payload or b'null'))


class WorkerApiTest(unittest.TestCase):
    token = 'test-token'

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix='character_api_'))
        os.environ['CHARACTER_WORKER_TOKEN'] = cls.token
        os.environ['CHARACTER_WORKER_READY_OVERRIDE'] = '1'
        cls.runtime = pipeline.Runtime(workspace=cls.tmp / 'ws', root=cls.tmp / 'root')
        cls.runtime.root.mkdir(parents=True)
        cls.manager = character_api.JobManager(cls.runtime, pipeline.FakePipeline(0.35).run_job, max_queue=3)
        cls.server = character_api.make_server('127.0.0.1', 0, cls.manager, character_api.token_sources_for(cls.runtime), False, True)
        cls.port = cls.server.server_address[1]
        cls.base = f'http://127.0.0.1:{cls.port}'
        cls.thread = threading.Thread(target=cls.server.serve_forever, kwargs={'poll_interval': 0.2}, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def body(self, **over):
        req = {'character_id': 'test_person', 'portrait': {'base64': base64.b64encode(PNG_1X1).decode(), 'filename': 'p.png'},
               'metadata': {'label': 'Test person', 'age': 30, 'gender': 'woman', 'hair': 'short black hair', 'tone': 'warm skin', 'facial_hair': 'none'},
               'poses': ['idle', 'talking', 'smiling'], 'options': {}}
        req.update(over)
        return req

    def wait_terminal(self, job_id: str, timeout: float = 15.0) -> dict:
        deadline = time.time() + timeout
        while time.time() < deadline:
            code, data = http('GET', f'{self.base}/characters/{job_id}', token=self.token)
            self.assertEqual(code, 200)
            if data['status'] in character_api.TERMINAL:
                return data
            time.sleep(0.1)
        self.fail(f'job {job_id} did not finish')

    def test_health_is_open_and_ready(self):
        code, data = http('GET', f'{self.base}/health')
        self.assertEqual(code, 200)
        self.assertTrue(data['ready'])
        self.assertEqual(data['service'], 'character-worker')
        self.assertIn('recipe', data)

    def test_auth_required(self):
        code, _ = http('POST', f'{self.base}/createCharacter', self.body())
        self.assertEqual(code, 401)
        code, _ = http('GET', f'{self.base}/characters', token='wrong')
        self.assertEqual(code, 401)

    def test_validation_errors(self):
        for bad in ({'portrait': {}}, {'poses': ['dance']}, {'options': {'nope': 1}}, {'options': {'max_attempts': 9}},
                    {'callback_url': 'ftp://x'}, {'metadata': {'pronouns': 'xe'}}, {'options': {'fake_outcome': 'done'}, 'portrait': {'url': 'file:///etc/passwd'}}):
            code, data = http('POST', f'{self.base}/createCharacter', self.body(**bad), token=self.token)
            self.assertEqual(code, 400, (bad, data))

    def test_full_job_lifecycle(self):
        code, data = http('POST', f'{self.base}/createCharacter', self.body(), token=self.token)
        self.assertEqual(code, 202, data)
        job_id = data['job_id']
        self.assertEqual(data['character_id'], 'test_person')
        final = self.wait_terminal(job_id)
        self.assertEqual(final['status'], 'done', final.get('error'))
        self.assertEqual(set(final['result']['published']['poses']), {'idle', 'talking', 'smiling'})
        self.assertTrue(final['result']['qc']['pass'])
        job_dir = self.runtime.root / 'jobs' / job_id
        self.assertTrue((job_dir / 'result.json').is_file())
        self.assertTrue((job_dir / 'upload.png').is_file(), 'base64 portrait must be persisted to disk, not kept in job.json')
        self.assertNotIn('base64', json.dumps(json.loads((job_dir / 'job.json').read_text())['request']['portrait']))
        code, listing = http('GET', f'{self.base}/characters/{job_id}/artifacts', token=self.token)
        self.assertEqual(code, 200)
        names = {a['path'] for a in listing['artifacts']}
        self.assertIn('result.json', names)
        code, raw = http('GET', f'{self.base}/characters/{job_id}/artifacts/result.json', token=self.token, raw=True)
        self.assertEqual(code, 200)
        self.assertEqual(json.loads(raw)['job_id'], job_id)
        code, _ = http('GET', f'{self.base}/characters/{job_id}/artifacts/../../etc/passwd', token=self.token)
        self.assertEqual(code, 404)

    def test_qc_failed_and_failed_outcomes(self):
        for outcome in ('qc_failed', 'failed'):
            code, data = http('POST', f'{self.base}/createCharacter', self.body(options={'fake_outcome': outcome}), token=self.token)
            self.assertEqual(code, 202, data)
            final = self.wait_terminal(data['job_id'])
            self.assertEqual(final['status'], outcome)
            self.assertTrue(final['error'])

    def test_cancel_queued_job_and_queue_order(self):
        first = http('POST', f'{self.base}/createCharacter', self.body(options={'fake_seconds': 1.5}), token=self.token)[1]
        second = http('POST', f'{self.base}/createCharacter', self.body(character_id='second_person'), token=self.token)[1]
        self.assertGreaterEqual(second['position'], 1)
        code, data = http('POST', f'{self.base}/characters/{second["job_id"]}/cancel', token=self.token)
        self.assertEqual(code, 200)
        self.assertEqual(data['status'], 'cancelled')
        self.assertEqual(self.wait_terminal(first['job_id'])['status'], 'done')
        self.assertEqual(http('GET', f'{self.base}/characters/{second["job_id"]}', token=self.token)[1]['status'], 'cancelled')

    def test_cancel_running_job(self):
        data = http('POST', f'{self.base}/createCharacter', self.body(options={'fake_seconds': 6}), token=self.token)[1]
        deadline = time.time() + 5
        while time.time() < deadline and http('GET', f'{self.base}/characters/{data["job_id"]}', token=self.token)[1]['status'] != 'running':
            time.sleep(0.05)
        code, resp = http('POST', f'{self.base}/characters/{data["job_id"]}/cancel', token=self.token)
        self.assertEqual(code, 200)
        self.assertEqual(self.wait_terminal(data['job_id'])['status'], 'cancelled')

    def test_drain_blocks_new_jobs(self):
        code, state = http('POST', f'{self.base}/worker/drain', token=self.token)
        self.assertEqual(code, 200)
        self.assertTrue(state['draining'])
        code, data = http('POST', f'{self.base}/createCharacter', self.body(), token=self.token)
        self.assertEqual(code, 503)
        http('POST', f'{self.base}/worker/undrain', token=self.token)
        code, _ = http('POST', f'{self.base}/createCharacter', self.body(), token=self.token)
        self.assertEqual(code, 202)

    def test_recovery_marks_interrupted_jobs(self):
        root = self.tmp / 'root2'
        (root / 'jobs' / 'j1').mkdir(parents=True)
        pipeline.write_json(root / 'jobs' / 'j1' / 'job.json', {'job_id': 'j1', 'character_id': 'x', 'status': 'running', 'stage': 'generate',
                                                                 'request': self.body(), 'created_utc': 'x', 'updated_utc': 'x'})
        (root / 'jobs' / 'j2').mkdir(parents=True)
        pipeline.write_json(root / 'jobs' / 'j2' / 'job.json', {'job_id': 'j2', 'character_id': 'y', 'status': 'queued', 'stage': 'queued',
                                                                 'request': self.body(), 'created_utc': 'x', 'updated_utc': 'x'})
        rt = pipeline.Runtime(workspace=self.tmp / 'ws', root=root)
        manager = character_api.JobManager(rt, pipeline.FakePipeline(0.1).run_job)
        self.assertEqual(manager.get('j1')['status'], 'failed')
        deadline = time.time() + 5
        while time.time() < deadline and manager.get('j2')['status'] not in character_api.TERMINAL:
            time.sleep(0.05)
        self.assertEqual(manager.get('j2')['status'], 'done')


if __name__ == '__main__':
    unittest.main()
