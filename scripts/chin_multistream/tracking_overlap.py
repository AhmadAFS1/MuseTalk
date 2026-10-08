"""Default-off, single-flight adapter around the unchanged canonical Tracker.

Only the owning benchmark worker uses this helper. One thread calls Tracker.track
in input order; the main thread can compose the preceding frame concurrently.
There is never more than one outstanding call or shared-memory writer. Reset and
normal Tracker.close remain the caller's responsibility after finish().
"""
from __future__ import annotations

from concurrent.futures import Future, TimeoutError as FutureTimeoutError
import math
import queue
import subprocess
import threading
import time


class OrderedTrackingOverlap:
    def __init__(self, tracker, *, timeout_s=30., cleanup_timeout_s=5.):
        if not all(math.isfinite(x) and x > 0 for x in (timeout_s, cleanup_timeout_s)):
            raise ValueError("tracking timeouts must be finite and positive")
        self.tracker = tracker
        # Retain the exact owned Popen object, never look up or signal a PID/name.
        self._owned_proc = tracker.proc
        self.timeout_s = timeout_s
        self.cleanup_timeout_s = cleanup_timeout_s
        self._queue = queue.Queue(maxsize=1)
        self._pending = None
        self._closed = False
        self.last_call_s = None
        # A broken IPC/resize must not make interpreter shutdown wait forever.
        self._thread = threading.Thread(target=self._run, name="ordered-tracking", daemon=True)
        self._thread.start()

    def _run(self):
        while True:
            item = self._queue.get()
            if item is None:
                return
            future, frame, face, box = item
            try:
                t = time.perf_counter()
                result = self.tracker.track(frame, face, box)
                future.set_result((result, time.perf_counter() - t))
            except BaseException as exc:
                future.set_exception(exc)

    def begin(self, frame, face, box):
        if self._closed:
            raise RuntimeError("tracking adapter is closed")
        if self._pending is not None:
            raise RuntimeError("finish the outstanding tracking call first")
        if self.tracker.proc is not self._owned_proc:
            raise RuntimeError("owned tracking process changed")
        self._pending = Future()
        self._queue.put_nowait((self._pending, frame, face, box))

    def finish(self):
        if self._closed or self._pending is None:
            raise RuntimeError("no outstanding tracking call")
        future = self._pending
        try:
            result, elapsed = future.result(timeout=self.timeout_s)
            self.last_call_s = elapsed
            return result
        finally:
            # A timeout is still outstanding and must be aborted on close.
            if future.done():
                self._pending = None

    def _abort_owned_process(self):
        proc = self._owned_proc
        if proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=self.cleanup_timeout_s)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=self.cleanup_timeout_s)

    def close(self):
        if self._closed:
            return
        self._closed = True
        failure = None
        try:
            if self._pending is not None and not self._pending.done():
                self._abort_owned_process()
                try:
                    self._pending.result(timeout=self.cleanup_timeout_s)
                except FutureTimeoutError as exc:
                    failure = exc
                except BaseException:
                    pass  # The original tracking failure is propagated by finish().
        except BaseException as exc:
            failure = exc
        finally:
            # If a worker is irrecoverably stuck, leave it daemonized and fail;
            # the harness supervisor still owns termination of its worker process.
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                failure = failure or RuntimeError("tracking queue did not drain")
            self._thread.join(timeout=self.cleanup_timeout_s)
            self._pending = None
        if self._thread.is_alive():
            raise RuntimeError("owned tracking thread did not stop") from failure
        if failure is not None:
            raise RuntimeError("owned tracking cleanup failed") from failure

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
