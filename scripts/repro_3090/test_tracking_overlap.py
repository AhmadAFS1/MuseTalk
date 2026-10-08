"""Synthetic CPU concurrency/order tests; not real FaceMesh or pixel acceptance."""
from concurrent.futures import TimeoutError
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from chin_multistream.tracking_overlap import OrderedTrackingOverlap
from chin_multistream import worker
import chin_multistream_render as renderer
import runner
import report
from test_repro_3090 import aggregate_fixture


class FakeProcess:
    pid = 12345

    def __init__(self, release=None, kill_required=False):
        self.release = release
        self.kill_required = kill_required
        self.signals = []
        self.returncode = None

    def poll(self):
        return self.returncode

    def terminate(self):
        self.signals.append("terminate")
        if not self.kill_required:
            self.returncode = -15
            if self.release:
                self.release.set()

    def kill(self):
        self.signals.append("kill")
        self.returncode = -9
        if self.release:
            self.release.set()

    def wait(self, timeout):
        if self.returncode is None:
            raise subprocess.TimeoutExpired("synthetic owned tracker", timeout)
        return self.returncode


class FakeTracker:
    def __init__(self, block_frame=None, kill_required=False):
        self.started = threading.Event()
        self.release = threading.Event()
        self.proc = FakeProcess(self.release, kill_required)
        self.block_frame = block_frame
        self.calls = []
        self.count = 0
        self.active = False
        self.resets = 0

    def reset(self):
        if self.active:
            raise AssertionError("reset raced with track")
        self.count = 0
        self.resets += 1

    def track(self, frame, face, box):
        if self.active:
            raise AssertionError("two concurrent shared-memory writers")
        self.active = True
        try:
            self.calls.append((frame, bytes(face), box, self.count))
            self.count += 1
            if frame == self.block_frame:
                self.started.set()
                if not self.release.wait(timeout=2):
                    raise AssertionError("synthetic tracker not released")
            if self.proc.poll() is not None:
                raise RuntimeError("synthetic tracker terminated")
            return frame + self.count / 100., .001
        finally:
            self.active = False


class SyntheticBuffer(bytes):
    flags = SimpleNamespace(c_contiguous=True)


class SyntheticBatch(list):
    def copy(self):
        return list(self)


class FakeConnection:
    def __init__(self, messages):
        self.messages = iter(messages)
        self.sent = []

    def recv(self):
        return next(self.messages)

    def send(self, message):
        self.sent.append(message)


class TrackingOverlapTests(unittest.TestCase):
    def test_cli_default_off_explicit_forwarding_and_serial_rejection(self):
        args = renderer.parse_args(["--label", "synthetic"])
        self.assertFalse(args.tracking_overlap)
        args = renderer.parse_args(["--label", "synthetic", "--tracking-overlap"])
        self.assertTrue(args.tracking_overlap)
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            renderer.parse_args(["--label", "synthetic", "--mode", "serial", "--tracking-overlap"])
        base = dict(python="synthetic-python", loops=24, thermal_warmup_s=120)
        for selected in (False, True):
            command = runner.aggregate_command(SimpleNamespace(**base, tracking_overlap=selected), Path("synthetic-out"), "synthetic", 6, 2)
            self.assertEqual(command.count("--tracking-overlap"), int(selected))
            self.assertEqual(command[command.index("--min-timed-s") + 1], "60")
            self.assertEqual(command[command.index("--loops") + 1], "24")

    def test_report_rejects_ignored_or_mixed_worker_modes_without_weaker_fps_gate(self):
        data = aggregate_fixture(399.96)
        data["args"]["tracking_overlap"] = True
        with self.assertRaisesRegex(report.Invalid, "overlap mode mismatch"):
            report.aggregate(data, "T", 400)
        for repeat in data["repeats"]:
            for w in repeat["per_worker"].values():
                w["tracking_overlap"] = True
        result = report.aggregate(data, "T", 400)
        self.assertEqual(result["status"], "FAIL")
        self.assertTrue(result["tracking_overlap"])
        data["args"]["tracking_overlap"] = False
        with self.assertRaisesRegex(report.Invalid, "overlap mode mismatch"):
            report.aggregate(data, "T", 400)

    def test_single_flight_exact_return_and_normal_cleanup(self):
        tracker = FakeTracker()
        with OrderedTrackingOverlap(tracker) as helper:
            with self.assertRaises(RuntimeError):
                helper.finish()
            for i in range(4):
                helper.begin(i, bytes([i]), i)
                with self.assertRaises(RuntimeError):
                    helper.begin(i + 1, b"next", i + 1)
                self.assertEqual(helper.finish(), (i + (i + 1) / 100., .001))
                self.assertGreaterEqual(helper.last_call_s, 0.)
        helper.close()
        self.assertFalse(helper._thread.is_alive())
        self.assertEqual(tracker.proc.signals, [])
        with self.assertRaises(RuntimeError):
            helper.begin(0, b"closed", 0)

    def test_tracking_error_propagates_without_signaling_normal_process(self):
        tracker = FakeTracker()
        tracker.track = mock.Mock(side_effect=ValueError("synthetic tracking error"))
        with OrderedTrackingOverlap(tracker) as helper:
            helper.begin(0, b"face", 0)
            with self.assertRaisesRegex(ValueError, "synthetic tracking error"):
                helper.finish()
        self.assertEqual(tracker.proc.signals, [])
        self.assertFalse(helper._thread.is_alive())

    def test_timeout_and_exception_abort_only_exact_owned_process(self):
        for kill_required in (False, True):
            tracker = FakeTracker(block_frame=0, kill_required=kill_required)
            owned = tracker.proc
            helper = OrderedTrackingOverlap(tracker, timeout_s=.01, cleanup_timeout_s=.2)
            try:
                helper.begin(0, b"face", 0)
                self.assertTrue(tracker.started.wait(timeout=1))
                with self.assertRaises(TimeoutError):
                    helper.finish()
                # Replacing this reference cannot redirect cleanup to another process.
                replacement = FakeProcess()
                tracker.proc = replacement
            finally:
                helper.close()
            self.assertEqual(owned.signals, ["terminate", "kill"] if kill_required else ["terminate"])
            self.assertEqual(replacement.signals, [])
            self.assertFalse(helper._thread.is_alive())
        tracker = FakeTracker(block_frame=0)
        with self.assertRaisesRegex(ValueError, "compose failed"):
            with OrderedTrackingOverlap(tracker, cleanup_timeout_s=.2) as helper:
                helper.begin(0, b"face", 0)
                self.assertTrue(tracker.started.wait(timeout=1))
                raise ValueError("compose failed")
        self.assertEqual(tracker.proc.signals, ["terminate"])

    def test_process_swap_and_bad_timeouts_rejected(self):
        tracker = FakeTracker()
        with OrderedTrackingOverlap(tracker) as helper:
            tracker.proc = FakeProcess()
            with self.assertRaisesRegex(RuntimeError, "process changed"):
                helper.begin(0, b"face", 0)
        for value in (0., -1., float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                OrderedTrackingOverlap(tracker, timeout_s=value)

    def test_real_owned_subprocess_pipe_timeout_unblocks_thread_without_gpu(self):
        proc = subprocess.Popen([sys.executable, "-u", "-c", "import sys; sys.stdin.buffer.read()"],
                                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        started = threading.Event()

        def blocked_track(frame, face, box):
            started.set()
            return proc.stdout.readline(), 0.

        tracker = SimpleNamespace(proc=proc, track=blocked_track)
        helper = OrderedTrackingOverlap(tracker, timeout_s=.01, cleanup_timeout_s=1.)
        try:
            helper.begin(0, b"synthetic", 0)
            self.assertTrue(started.wait(timeout=1))
            with self.assertRaises(TimeoutError):
                helper.finish()
            helper.close()
            self.assertIsNotNone(proc.poll())
            self.assertFalse(helper._thread.is_alive())
        finally:
            helper.close()
            if proc.poll() is None:
                proc.kill()
                proc.wait(timeout=1)
            proc.stdin.close()
            proc.stdout.close()

    def run_synthetic_repeat(self, overlapped, *, prove_concurrency=False):
        tracker = FakeTracker(block_frame=2 if prove_concurrency else None)
        cfg = {"stream": 0, "identity": "synthetic", "tracking_overlap": overlapped}
        rcfg = {"loops": 2}
        n = 16
        data = {"cache": {"boxes": list(range(n))}, "p": list(range(n)),
                "g": [None] * n, "chin_delta": [None] * n}
        att = SimpleNamespace(frames=list(range(n)), d=data)
        rings = [SyntheticBatch(SyntheticBuffer(bytes([i])) for i in range(start, start + 8))
                 for start in (0, 8)]
        conn = FakeConnection([("b", slot, 0, loop, slot * 8) for loop in range(2) for slot in range(2)])
        composed = []

        def compose(frame, d, i, face):
            if prove_concurrency and i == 0 and not composed:
                # Frame2 tracking must start while frame0 is being composed;
                # release it here. A sequential implementation would deadlock/fail.
                self.assertTrue(tracker.started.wait(timeout=1))
                self.assertTrue(tracker.active)
                tracker.release.set()
            result = SyntheticBuffer(f"{i}:{d['g'][i]:.6f}:{d['chin_delta'][i]:.6f}:{face.hex()}".encode())
            composed.append(result)
            return result

        chin = SimpleNamespace(curves=lambda p, g: (p * .02, g * .01), corrected_refined=compose)
        np = SimpleNamespace(clip=lambda x, lo, hi: min(hi, max(lo, x)))
        with mock.patch.object(worker.paths, "N_FRAMES", n), \
                mock.patch.object(worker, "proc_cpu_s", return_value=0.), \
                mock.patch.object(worker, "rss_mib", return_value=0.):
            if overlapped:
                with OrderedTrackingOverlap(tracker) as helper:
                    result = worker._repeat(cfg, conn, 0, rcfg, att, tracker, rings, None, chin, np, helper)
            else:
                result = worker._repeat(cfg, conn, 0, rcfg, att, tracker, rings, None, chin, np)
        self.assertEqual(tracker.resets, 2)
        self.assertEqual(tracker.proc.signals, [])
        self.assertEqual(result["timing_ms"]["frames"], 32)
        self.assertEqual(result["timing_ms"]["batches"], 4)
        self.assertEqual(conn.sent, [("armed", 0, 0), ("r", 0), ("r", 1), ("r", 0), ("r", 1)])
        return result, composed, tracker.calls, data

    def test_actual_worker_loop_synthetic_order_filter_resets_and_hash_parity(self):
        serial, composed, calls, data = self.run_synthetic_repeat(False)
        parallel, pcomposed, pcalls, pdata = self.run_synthetic_repeat(True)
        self.assertEqual(composed, pcomposed)
        self.assertEqual(calls, pcalls)
        self.assertEqual(data, pdata)
        for name in ("raw_refined_sha256", "generated_faces_sha256"):
            self.assertEqual([c[name] for c in serial["clips"]], [c[name] for c in parallel["clips"]])
            self.assertEqual(serial["clips"][0][name], serial["clips"][1][name])
        self.assertFalse(serial["tracking_overlap"])
        self.assertTrue(parallel["tracking_overlap"])
        self.assertNotIn("tracking_overlap_wait_ms", serial["timing_ms"])
        self.assertIn("overlaps compose/filter", parallel["timing_semantics"])

    def test_actual_worker_loop_proves_overlap_without_timing_speed_claim(self):
        self.run_synthetic_repeat(True, prove_concurrency=True)


if __name__ == "__main__":
    unittest.main()
