"""CPU-only negative coverage; never import real Torch or NVIDIA libraries."""
import ctypes as C
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

SOURCE = Path(__file__).with_name('watch_owned_single_leaf.py')
spec = importlib.util.spec_from_file_location('owned_watch_test_module', SOURCE)
w = importlib.util.module_from_spec(spec); spec.loader.exec_module(w)


class WatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.mps = self.root / 'mps'; self.mps.mkdir(mode=0o700)
        self.target = self.root / 'scripts' / 'repro_3090' / 'build_owned_taesd_fp32_a3.py'
        self.target.parent.mkdir(parents=True)

    def source(self, body=b'pass\n'):
        self.target.write_bytes(body)
        return hashlib.sha256(body).hexdigest()

    def nvml(self, status=0, count=1, pid=1234, memory=w.ALLOCATION_BYTES):
        observer = w.Nvml.__new__(w.Nvml); observer.handle = C.c_void_p(7)
        def result(handle, number, array):
            number._obj.value = count; array[0].pid = pid; array[0].memory = memory
            return status
        observer.lib = types.SimpleNamespace(nvmlDeviceGetComputeRunningProcesses_v3=Mock(side_effect=result))
        return observer

    def test_import_is_inert(self):
        script = "import ctypes,runpy,sys; ctypes.CDLL=lambda *a,**k: (_ for _ in ()).throw(AssertionError('library')); runpy.run_path(sys.argv[1],run_name='inert'); assert 'torch' not in sys.modules"
        subprocess.run([sys.executable, '-B', '-c', script, str(SOURCE)], check=True)

    def test_checked_bytes_not_second_read(self):
        digest = self.source(); body = w.checked_source(self.target, digest)
        self.target.write_bytes(b'raise AssertionError()\n'); exec(compile(body, str(self.target), 'exec'), {})

    def test_source_hash_path_rejections(self):
        digest = self.source()
        with self.assertRaises(w.Rejected): w.checked_source(self.target, '0' * 64)
        with self.assertRaises(w.Rejected): w.checked_source('relative.py', digest)
        link = self.root / 'link.py'; link.symlink_to(self.target)
        with self.assertRaises(w.Rejected): w.checked_source(link, digest)

    def test_exact_nvml_v3_abi(self):
        self.assertEqual(C.sizeof(w.Process), 24)
        self.assertEqual([getattr(w.Process, name).offset for name in ('pid', 'memory', 'gpu_instance', 'compute_instance')], [0, 8, 16, 20])

    def test_full_success_enumeration(self):
        self.assertEqual(self.nvml().rows(), [(1234, w.ALLOCATION_BYTES)])
        self.assertEqual(self.nvml(count=0).rows(), [])

    def test_error_or_truncation_never_partial(self):
        for status, count in ((7, 65), (4, 1), (999, 0), (0, 65)):
            with self.subTest(status=status, count=count), self.assertRaises(w.Rejected):
                self.nvml(status=status, count=count).rows()

    def test_stalled_nvml_query_fail_closed(self):
        release = threading.Event(); observer = self.nvml()
        observer.lib.nvmlDeviceGetComputeRunningProcesses_v3.side_effect = lambda *args: release.wait(1)
        try:
            with patch.object(w, 'QUERY_SECONDS', 0.01), self.assertRaisesRegex(w.Rejected, 'nvml_query_timeout'):
                observer.rows()
        finally:
            release.set()

    def test_nvml_thread_exception_never_success(self):
        observer = self.nvml(); observer.lib.nvmlDeviceGetComputeRunningProcesses_v3.side_effect = RuntimeError('driver')
        with self.assertRaises(RuntimeError): observer.rows()

    def test_singleton_is_not_first_new_pid(self):
        for rows in ([], [(1, w.ALLOCATION_BYTES), (2, w.ALLOCATION_BYTES)], [(True, w.ALLOCATION_BYTES)],
                     [(0, w.ALLOCATION_BYTES)], [(4, 0)], [(4, (1 << 64) - 1)]):
            with self.subTest(rows=rows), self.assertRaises(w.Rejected): w.sole_context_pid(rows)

    def test_bound_rejects_foreign_or_disappeared(self):
        w.verify_bound([(4, w.ALLOCATION_BYTES)], 4)
        for rows in ([], [(5, w.ALLOCATION_BYTES)], [(4, w.ALLOCATION_BYTES), (5, w.ALLOCATION_BYTES)]):
            with self.subTest(rows=rows), self.assertRaises(w.Rejected): w.verify_bound(rows, 4)

    def test_receipt_identity_not_allowlist(self):
        message = dict(container_pid=123, host_pid=456, cuda_uuid=w.UUID, allocation_bytes=w.ALLOCATION_BYTES)
        self.assertEqual(w.validate_receipt(message, 123, w.UUID), 456)
        for change in ({'container_pid': 124}, {'container_pid': True}, {'host_pid': True}, {'host_pid': 0},
                       {'cuda_uuid': 'GPU-other'}, {'allocation_bytes': 0}, {'extra': 1}):
            with self.subTest(change=change), self.assertRaises(w.Rejected):
                w.validate_receipt({**message, **change}, 123, w.UUID)

    def test_private_mps_must_remain_empty(self):
        w.quiet_mps(self.mps)
        self.mps.chmod(0o755)
        with self.assertRaises(w.Rejected): w.quiet_mps(self.mps)
        self.mps.chmod(0o700); (self.mps / 'socket').touch()
        with self.assertRaises(w.Rejected): w.quiet_mps(self.mps)

    def test_parent_death_signal_parent_race(self):
        fn = Mock(return_value=0)
        with patch.object(w.C, 'CDLL', return_value=types.SimpleNamespace(prctl=fn)), patch.object(w.os, 'getppid', side_effect=[100, 101]):
            with self.assertRaises(w.Rejected): w.parent_death_signal(100)
        fn.assert_called_once_with(1, w.signal.SIGTERM, 0, 0, 0)

    def test_cuda_uuid_exact_bytes(self):
        def write_uuid(output, device):
            output[:] = bytes.fromhex(w.UUID[4:].replace('-', '')); return 0
        lib = types.SimpleNamespace(cuDeviceGet=Mock(return_value=0), cuDeviceGetUuid_v2=Mock(side_effect=write_uuid))
        with patch.object(w.C, 'CDLL', return_value=lib): self.assertEqual(w.cuda_uuid(), w.UUID)
        lib.cuDeviceGetUuid_v2 = None
        with patch.object(w.C, 'CDLL', return_value=lib), self.assertRaises(w.Rejected): w.cuda_uuid()

    def leaf_setup(self, acknowledgments=b'01', uuid=None):
        digest = self.source(b'import torch\ntorch.cuda._lazy_init()\nassert len(_owned_context_retained)==1\n')
        report_r, report_w = os.pipe(); ack_r, ack_w = os.pipe()
        for fd in (report_r, report_w, ack_r, ack_w): self.addCleanup(os.close, fd)
        if acknowledgments: os.write(ack_w, acknowledgments)
        original = Mock(); cuda = types.SimpleNamespace(_lazy_init=original, device_count=lambda: 1)
        cuda.synchronize = lambda index: cuda._lazy_init()
        torch = types.SimpleNamespace(cuda=cuda, uint8='uint8')
        torch.empty = lambda size, **kwargs: (cuda._lazy_init(), object())[1]
        leaf_spec = dict(target=str(self.target), target_sha256=digest, arguments=[], uuid=w.UUID,
                         mps=str(self.mps), report_fd=report_w, ack_fd=ack_r, parent_pid=os.getpid())
        return leaf_spec, torch, report_r, original

    def test_deferred_init_two_acks_recursion_and_retention(self):
        leaf_spec, torch, report_r, original = self.leaf_setup()
        with patch.object(w, 'parent_death_signal'), patch.dict(sys.modules, torch=torch), patch.object(w, 'cuda_uuid', return_value=w.UUID), patch.object(w, 'Nvml', return_value=self.nvml()):
            w.leaf(leaf_spec)
        receipts = [json.loads(row) for row in os.read(report_r, 4096).splitlines()]
        self.assertEqual(receipts[0], dict(phase='BEGIN_INIT', container_pid=os.getpid()))
        self.assertEqual(w.validate_receipt(receipts[1], os.getpid(), w.UUID), 1234)
        self.assertEqual(original.call_count, 3)

    def test_wrong_cuda_uuid_cannot_bind(self):
        leaf_spec, torch, report_r, original = self.leaf_setup()
        with patch.object(w, 'parent_death_signal'), patch.dict(sys.modules, torch=torch), patch.object(w, 'cuda_uuid', return_value='GPU-other'), self.assertRaises(w.Rejected):
            w.leaf(leaf_spec)
        self.assertEqual(len(os.read(report_r, 4096).splitlines()), 1)

    def test_fork_identity_before_original_cuda_init(self):
        leaf_spec, torch, report_r, original = self.leaf_setup()
        with patch.object(w, 'parent_death_signal'), patch.dict(sys.modules, torch=torch), patch.object(w.os, 'getpid', side_effect=[100, 101]), self.assertRaises(w.Rejected):
            w.leaf(leaf_spec)
        original.assert_not_called()

    def test_default_off_and_exact_deadline_before_observer(self):
        arguments = ['--out', str(self.root / 'watch'), '--target', str(self.target), '--target-sha256', self.source(), '--deadline-utc', w.DEADLINE, '--']
        with patch.object(w, 'Nvml') as observer, self.assertRaises(w.Rejected): w.main(arguments)
        observer.assert_not_called()
        arguments[arguments.index(w.DEADLINE)] = '2026-10-10T02:45:00Z'
        with patch.object(w, 'Nvml') as observer, self.assertRaises(w.Rejected): w.main(['--enable', *arguments])
        observer.assert_not_called()

    def test_completed_nonzero_leaf_returncode_and_exit_race(self):
        digest = self.source(); host_pid, child_pid = 7654, 4321
        child = types.SimpleNamespace(pid=child_pid, returncode=1, poll=Mock(side_effect=[None, None, None, 1]))
        own = [(host_pid, w.ALLOCATION_BYTES)]
        observer = types.SimpleNamespace(rows=Mock(side_effect=[[], [], [], own, own, []]))
        messages = [dict(phase='BEGIN_INIT', container_pid=child_pid),
                    dict(container_pid=child_pid, host_pid=host_pid, cuda_uuid=w.UUID, allocation_bytes=w.ALLOCATION_BYTES)]
        fixed = w.dt.datetime(2026, 10, 9, 1, 0, tzinfo=w.dt.timezone.utc)
        clock = Mock(wraps=w.dt.datetime); clock.now.return_value = fixed
        output = self.root / 'watch.jsonl'
        argv = ['--enable', '--out', str(output), '--target', str(self.target), '--target-sha256', digest,
                '--deadline-utc', w.DEADLINE, '--']
        with patch.object(w.socket, 'gethostname', return_value=w.HOST), patch.object(w.os, 'getsid', return_value=123), \
             patch.object(w.os, 'getpgrp', return_value=123), patch.object(w.os, 'getpid', return_value=123), \
             patch.object(w.Path, 'is_file', return_value=True), patch.object(w.dt, 'datetime', clock), \
             patch.object(w.tempfile, 'tempdir', str(self.root)), \
             patch.object(w, 'Nvml', return_value=observer), patch.object(w.subprocess, 'Popen', return_value=child), \
             patch.object(w.select, 'select', return_value=([1], [], [])), patch.object(w.time, 'sleep'), \
             patch.object(w.os, 'read', side_effect=[(json.dumps(m) + '\n').encode() for m in messages]), \
             patch.object(w.os, 'write', return_value=1), patch.object(w.os, 'killpg') as kill:
            self.assertEqual(w.main(argv), 1, output.read_text() if output.exists() else 'no receipt')
        kill.assert_not_called()
        receipt = json.loads(output.read_text().splitlines()[-1])
        self.assertEqual((receipt['status'], receipt['returncode']), ('COMPLETE', 1))

    def test_bootstrap_real_main_inert_for_cpu_spawn(self):
        # A fake Torch module proves CPU spawn, not CUDA correctness or ownership.
        (self.root / 'torch.py').write_text('import types\ncuda=types.SimpleNamespace(_lazy_init=lambda:None)\n')
        (self.root / 'worker.py').write_text('def work(q): q.put(7)\n')
        body = b"import multiprocessing as mp\nfrom worker import work\nc=mp.get_context('spawn');q=c.Queue();p=c.Process(target=work,args=(q,));p.start();assert q.get(timeout=5)==7;p.join(5);assert p.exitcode==0\n"
        digest = self.source(body)
        leaf_spec = dict(target=str(self.target), target_sha256=digest, arguments=[], mps=str(self.mps), parent_pid=os.getpid())
        bootstrap = w.BOOTSTRAP.replace('m.leaf(json.loads(s))', 'm.parent_death_signal=lambda p:None;m.leaf(json.loads(s))')
        env = {**os.environ, 'PYTHONPATH': str(self.root)}
        subprocess.run([sys.executable, '-B', '-c', bootstrap, str(SOURCE), hashlib.sha256(SOURCE.read_bytes()).hexdigest(), json.dumps(leaf_spec)], env=env, check=True, timeout=15)


if __name__ == '__main__':
    unittest.main()
