#!/usr/bin/env python3
"""Orchestrator tests: a fake Vast API plus the real worker API running the fake pipeline in-process.

Run: python3 -m unittest tests.test_orchestrator -v
"""
from __future__ import annotations

import base64
import json
import os
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'orchestrator'))
import character_api  # noqa: E402
import pipeline  # noqa: E402
from engine import Config, Orchestrator, WorkerClient  # noqa: E402
from job_store import JobStore  # noqa: E402
from vast_client import VastError, mapped_port, parse_query  # noqa: E402
import charfactory  # noqa: E402

PNG_1X1 = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==')


class FakeVast:
    """Enough of the Vast API for the engine: offers, create, list/get, destroy; instances boot after two listings."""

    def __init__(self, worker_port: int, boot_listings: int = 2):
        self.worker_port, self.boot_listings = worker_port, boot_listings
        self.instances: dict = {}
        self.next_id = 1000
        self.created: list = []
        self.destroyed: list = []
        self.fail_create: VastError | None = None

    def search_offers(self, query, limit=50, storage_gb=5.0, **_):
        assert query['gpu_ram']['gte'] == 12000.0, query
        return [dict(id=1, gpu_name='RTX 4090', gpu_ram=24564, cpu_ram=64000, disk_space=400, driver_version='580.82', cuda_max_good=13.0,
                     inet_down=900, reliability2=0.99, dph_total=0.45, geolocation='US', rented=False, compute_cap=890),
                dict(id=2, gpu_name='RTX 4090', gpu_ram=24564, cpu_ram=64000, disk_space=400, driver_version='550.54', cuda_max_good=12.4,
                     inet_down=900, reliability2=0.99, dph_total=0.30, geolocation='US', rented=False, compute_cap=890),
                dict(id=3, gpu_name='RTX 4070S', gpu_ram=12282, cpu_ram=30952, disk_space=250, driver_version='595.84', cuda_max_good=13.0,
                     inet_down=600, reliability2=0.99, dph_total=0.60, geolocation='FR', rented=False, compute_cap=890)]

    def create_instance(self, offer_id, *, template_hash, disk_gb, label, env=None, **_):
        if self.fail_create:
            raise self.fail_create
        iid = self.next_id
        self.next_id += 1
        self.instances[iid] = {'id': iid, 'label': label, 'actual_status': 'loading', 'cur_state': 'running', 'intended_status': 'running',
                               'public_ipaddr': '127.0.0.1', 'ports': {}, 'gpu_name': 'RTX 4090', 'dph_total': 0.45, '_listings': 0, 'start_date': time.time()}
        self.created.append({'offer_id': offer_id, 'template_hash': template_hash, 'disk_gb': disk_gb, 'label': label})
        return iid

    def list_instances(self):
        out = []
        for inst in self.instances.values():
            inst['_listings'] += 1
            if inst['_listings'] >= self.boot_listings:
                inst['actual_status'] = 'running'
                inst['ports'] = {'8010/tcp': [{'HostIp': '0.0.0.0', 'HostPort': str(self.worker_port)}, {'HostIp': '::', 'HostPort': str(self.worker_port)}]}
            out.append({k: v for k, v in inst.items() if not k.startswith('_')})
        return out

    def get_instance(self, iid):
        inst = self.instances.get(iid)
        return None if inst is None else {k: v for k, v in inst.items() if not k.startswith('_')}

    def destroy_instance(self, iid):
        self.destroyed.append(iid)
        self.instances.pop(iid, None)
        return True

    def set_label(self, iid, label):
        self.instances[iid]['label'] = label


class WebhookReceiver(BaseHTTPRequestHandler):
    received: list = []

    def log_message(self, *_):
        pass

    def do_POST(self):
        length = int(self.headers.get('Content-Length') or 0)
        WebhookReceiver.received.append({'headers': dict(self.headers), 'body': json.loads(self.rfile.read(length))})
        self.send_response(200)
        self.send_header('Content-Length', '2')
        self.end_headers()
        self.wfile.write(b'ok')


class OrchestratorTest(unittest.TestCase):
    token = 'worker-secret'

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix='charfactory_'))
        os.environ['CHARACTER_WORKER_TOKEN'] = cls.token
        os.environ['CHARACTER_WORKER_READY_OVERRIDE'] = '1'
        cls.worker_runtime = pipeline.Runtime(workspace=cls.tmp / 'ws', root=cls.tmp / 'worker_root')
        cls.worker_runtime.root.mkdir(parents=True)
        cls.manager = character_api.JobManager(cls.worker_runtime, pipeline.FakePipeline(0.4).run_job, max_queue=5)
        cls.worker = character_api.make_server('127.0.0.1', 0, cls.manager, character_api.token_sources_for(cls.worker_runtime), False, True)
        cls.worker_port = cls.worker.server_address[1]
        threading.Thread(target=cls.worker.serve_forever, kwargs={'poll_interval': 0.2}, daemon=True).start()
        cls.hooks = ThreadingHTTPServer(('127.0.0.1', 0), WebhookReceiver)
        threading.Thread(target=cls.hooks.serve_forever, kwargs={'poll_interval': 0.2}, daemon=True).start()
        cls.hook_url = f'http://127.0.0.1:{cls.hooks.server_address[1]}/hook'

    @classmethod
    def tearDownClass(cls):
        cls.worker.shutdown()
        cls.hooks.shutdown()

    def setUp(self):
        # retire_idle drains a worker before destroying it; the fake worker is shared by every test
        self.manager.set_draining(False)
        os.environ['CHARACTER_WORKER_READY_OVERRIDE'] = '1'

    def make(self, name: str, **over):
        settings = dict(template_hash='tpl-hash', max_workers=1, idle_ttl_s=5, ready_timeout_s=600, poll_interval_s=0, worker_port=8010,
                        worker_token=self.token, result_webhook=self.hook_url, result_webhook_token='hook-token', data_dir=self.tmp / name,
                        provision_retries=2, unreachable_polls_max=3)
        settings.update(over)
        cfg = Config(**settings)
        cfg.data_dir.mkdir(parents=True, exist_ok=True)
        store = JobStore(cfg.data_dir / 'db.sqlite')
        vast = FakeVast(self.worker_port)
        clock = {'offset': 0.0}
        orch = Orchestrator(cfg, store, vast, WorkerClient(self.token, 5), now=lambda: time.time() + clock['offset'], log=lambda m: None)
        return cfg, store, vast, orch, clock

    def enqueue(self, cfg, store, character_id='orch_person'):
        body = {'character_id': character_id, 'portrait': {'base64': base64.b64encode(PNG_1X1).decode(), 'filename': 'p.png'},
                'metadata': {'label': 'Orch person', 'age': 31, 'gender': 'man', 'hair': 'short hair', 'tone': 'olive skin', 'facial_hair': 'none'}}
        job, warnings = charfactory.enqueue(cfg, store, body)
        self.assertEqual(warnings, [])
        return job['job_id']

    def run_until(self, orch, store, job_id, statuses, ticks=60, sleep=0.15):
        for _ in range(ticks):
            orch.tick()
            job = store.get_job(job_id)
            if job['status'] in statuses:
                return job
            time.sleep(sleep)
        self.fail(f'job {job_id} stuck at {store.get_job(job_id)["status"]}')

    def test_full_lifecycle_provision_dispatch_complete_retire(self):
        cfg, store, vast, orch, clock = self.make('lifecycle')
        WebhookReceiver.received.clear()
        job_id = self.enqueue(cfg, store)
        self.assertTrue((cfg.data_dir / 'portraits' / f'{job_id}.png').is_file())
        orch.tick()                                   # provisions: offer 1 is the cheapest that passes the filters (offer 2 has an old driver)
        self.assertEqual(len(vast.created), 1)
        self.assertEqual(vast.created[0]['offer_id'], 1)
        self.assertEqual(vast.created[0]['template_hash'], 'tpl-hash')
        workers = store.list_workers()
        self.assertEqual(workers[0]['status'], 'booting')
        self.assertEqual(workers[0]['worker_id'], '1000')
        orch.tick()                                   # still loading: no port mapping yet
        self.assertEqual(store.get_worker('1000')['status'], 'booting')
        self.assertEqual(len(vast.created), 1, 'must not rent a second worker while one is booting')
        job = self.run_until(orch, store, job_id, ('running', 'done'))
        self.assertEqual(store.get_worker('1000')['status'], 'busy')
        self.assertTrue(job['remote_job_id'])
        job = self.run_until(orch, store, job_id, ('done',))
        self.assertEqual(job['result']['status'], 'done')
        self.assertEqual(set(job['result']['published']['poses']), {'idle', 'talking', 'smiling'})
        self.assertEqual(store.get_worker('1000')['status'], 'ready')
        self.assertEqual(len(WebhookReceiver.received), 1)
        hook = WebhookReceiver.received[0]
        self.assertEqual(hook['headers'].get('Authorization'), 'Bearer hook-token')
        self.assertEqual(hook['body']['job']['job_id'], job_id)
        self.assertEqual(hook['body']['job']['status'], 'done')
        orch.tick()
        self.assertEqual(vast.destroyed, [], 'idle worker must survive the TTL')
        clock['offset'] = 60                          # past idle_ttl_s with an empty queue
        orch.tick()
        self.assertEqual(vast.destroyed, [1000])
        self.assertEqual(store.get_worker('1000')['status'], 'dead')

    def test_lost_worker_requeues_job_and_rents_another(self):
        cfg, store, vast, orch, clock = self.make('lost')
        job_id = self.enqueue(cfg, store, 'lost_person')
        self.run_until(orch, store, job_id, ('running',))
        first = store.get_job(job_id)['worker_id']
        vast.instances.pop(int(first))                 # the host vanished mid-job
        orch.tick()
        job = store.get_job(job_id)
        self.assertEqual(job['status'], 'queued')
        self.assertEqual(job['attempts'], 1)
        self.assertEqual(store.get_worker(first)['status'], 'dead')
        job = self.run_until(orch, store, job_id, ('done',))
        self.assertNotEqual(job['worker_id'], first)
        self.assertEqual(len(vast.created), 2)

    def test_boot_failure_is_destroyed_then_renting_pauses(self):
        cfg, store, vast, orch, clock = self.make('bootfail', max_boot_failures=2, provision_backoff_s=3600, ready_timeout_s=10000)
        job_id = self.enqueue(cfg, store, 'boot_person')
        orch.tick()
        status_file = self.worker_runtime.root / 'status.json'
        os.environ['CHARACTER_WORKER_READY_OVERRIDE'] = '0'
        pipeline.write_json(status_file, {'phase': 'install', 'ready': False, 'failed': True, 'message': 'H3 weights checksum mismatch'})
        try:
            for _ in range(10):
                orch.tick()
            worker = store.get_worker('1000')
            self.assertEqual(worker['status'], 'dead')
            self.assertIn('boot failed', worker['notes'])
            self.assertEqual(vast.destroyed, [1000, 1001], 'two boot failures, then renting must pause')
            self.assertEqual(len(vast.created), 2)
            self.assertTrue(any(e['kind'] == 'provision.paused' for e in store.events()))
            self.assertEqual(store.get_job(job_id)['status'], 'queued')
        finally:
            os.environ['CHARACTER_WORKER_READY_OVERRIDE'] = '1'
            status_file.unlink(missing_ok=True)
        orch.tick()
        self.assertEqual(len(vast.created), 2, 'still paused inside the backoff window')
        clock['offset'] = 4000                        # backoff elapsed: renting resumes and the job completes
        self.run_until(orch, store, job_id, ('done',))
        self.assertEqual(len(vast.created), 3)

    def test_draining_idle_worker_is_destroyed(self):
        cfg, store, vast, orch, clock = self.make('draining')
        job_id = self.enqueue(cfg, store, 'drain_person')
        self.run_until(orch, store, job_id, ('done',))
        self.manager.set_draining(True)               # e.g. an operator drained it by hand
        orch.tick()
        self.assertEqual(store.get_worker('1000')['status'], 'dead')
        self.assertEqual(vast.destroyed, [1000])

    def test_create_timeout_is_adopted_by_label(self):
        cfg, store, vast, orch, clock = self.make('adopt')
        job_id = self.enqueue(cfg, store, 'adopt_person')
        vast.fail_create = VastError('timed out', None, None)
        orch.tick()
        pending = [w for w in store.list_workers() if w['worker_id'].startswith('pending-')]
        self.assertEqual(len(pending), 1)
        # the instance actually exists on Vast under our label
        vast.fail_create = None
        iid = vast.create_instance(1, template_hash='tpl-hash', disk_gb=200, label=pending[0]['label'])
        orch.tick()
        self.assertIsNone(store.get_worker(pending[0]['worker_id']))
        self.assertEqual(store.get_worker(str(iid))['status'], 'booting')
        self.run_until(orch, store, job_id, ('done',))

    def test_dry_run_never_rents(self):
        cfg, store, vast, orch, clock = self.make('dry', dry_run=True)
        self.enqueue(cfg, store, 'dry_person')
        orch.tick()
        self.assertEqual(vast.created, [])
        self.assertTrue(any(e['kind'] == 'provision.dry_run' for e in store.events()))

    def test_config_from_env_and_offer_filters(self):
        cfg = Config.from_env({'CHARFACTORY_TEMPLATE_HASH': 'abc', 'CHARFACTORY_MAX_WORKERS': '3', 'CHARFACTORY_MAX_DPH': '0.9',
                               'CHARFACTORY_DRY_RUN': 'true', 'CHARFACTORY_DATA_DIR': '/tmp/x', 'CHARFACTORY_EXTRA_ENV': '{"A": "1"}'})
        self.assertEqual((cfg.template_hash, cfg.max_workers, cfg.max_dph, cfg.dry_run, str(cfg.data_dir), cfg.extra_env), ('abc', 3, 0.9, True, '/tmp/x', {'A': '1'}))
        self.assertEqual(Config().validate()[0][:26], 'CHARFACTORY_TEMPLATE_HASH ')
        orch = Orchestrator(Config(max_dph=0.5), JobStore(':memory:'), FakeVast(1), log=lambda m: None)
        offers = FakeVast(1).search_offers(parse_query(Config().offer_query))
        self.assertEqual([o['id'] for o in offers if orch.offer_ok(o)], [1])
        self.assertEqual(mapped_port({'public_ipaddr': '1.2.3.4', 'ports': {'8010/tcp': [{'HostPort': '555'}]}}, 8010), ('1.2.3.4', 555))
        self.assertIsNone(mapped_port({'public_ipaddr': '1.2.3.4', 'ports': {}}, 8010))


if __name__ == '__main__':
    unittest.main()
