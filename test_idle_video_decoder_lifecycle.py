"""Real-codec teardown checks for reusable idle/entry/return decoders.

These prove deterministic resource release. They do not claim to reproduce or
identify the native deadlock observed during a live WebRTC entry test.
"""
import json
import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

import av
import numpy as np

from scripts.webrtc_tracks import IdleVideoStreamTrack


class IdleDecoderLifecycleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.video = Path(cls.temp.name) / "decoder.mp4"
        with av.open(str(cls.video), "w") as container:
            stream = container.add_stream("libx264", rate=24)
            stream.width = 64
            stream.height = 64
            stream.pix_fmt = "yuv420p"
            stream.codec_context.thread_count = 1
            for index in range(64):
                pixels = np.full((64, 64, 3), index * 3, np.uint8)
                frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                for packet in stream.encode(frame):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_partial_decoder_stop_closes_all_owned_references_and_is_idempotent(self):
        track = IdleVideoStreamTrack(str(self.video), decode_threads=4)
        for _ in range(8):
            track.read_frame()
        container = track._container
        track.stop()
        self.assertIsNone(track._frame_iter)
        self.assertIsNone(track._stream)
        self.assertIsNone(track._container)
        self.assertEqual(track.readyState, "ended")
        with self.assertRaises((ValueError, AssertionError)):
            next(container.demux())
        track.stop()

    def test_reset_finishes_previous_generator_before_reopening_at_frame_zero(self):
        track = IdleVideoStreamTrack(str(self.video), decode_threads=4)
        try:
            first = track.read_frame().to_ndarray(format="rgb24")
            for _ in range(8):
                track.read_frame()
            previous = track._container
            original_iterator = track._frame_iter
            events = []

            class Iterator:
                def close(self):
                    events.append("iterator_closed")
                    original_iterator.close()
            track._frame_iter = Iterator()
            track.reset()
            self.assertEqual(events, ["iterator_closed"])
            with self.assertRaises((ValueError, AssertionError)):
                next(previous.demux())
            self.assertIsNot(track._container, previous)
            np.testing.assert_array_equal(track.read_frame().to_ndarray(format="rgb24"), first)
            self.assertEqual(track.get_timing()["source_frame_index"], 0)
        finally:
            track.stop()

    def test_real_native_decoder_workers_release_without_gc_across_many_stops_and_resets(self):
        # Isolate OS thread counts from unittest's other async executors. Retain
        # stopped track objects deliberately: their native codec workers must
        # already be gone, without deleting objects or waiting for cyclic GC.
        script = textwrap.dedent("""
            import gc, json, sys
            from pathlib import Path
            from scripts.webrtc_tracks import IdleVideoStreamTrack
            gc.disable()
            def threads(): return len(list(Path('/proc/self/task').iterdir()))
            baseline = threads()
            live_counts, stopped_counts, retained = [], [], []
            for cycle in range(20):
                track = IdleVideoStreamTrack(sys.argv[1], decode_threads=4)
                for _ in range(8): track.read_frame()
                live_counts.append(threads())
                if cycle % 2 == 0:
                    track.reset()
                    for _ in range(8): track.read_frame()
                track.stop()
                retained.append(track)
                stopped_counts.append(threads())
            assert min(live_counts) >= baseline + 4, (baseline, live_counts)
            assert all(n == baseline for n in stopped_counts), (baseline, stopped_counts)
            print(json.dumps({'baseline_threads': baseline, 'cycles': len(retained),
                              'peak_threads': max(live_counts), 'final_threads': threads()}))
        """)
        result = subprocess.run([sys.executable, "-c", script, str(self.video)],
            cwd=Path(__file__).resolve().parent, capture_output=True, text=True,
            timeout=20, env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        evidence = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(evidence["cycles"], 20)
        self.assertEqual(evidence["final_threads"], evidence["baseline_threads"])


if __name__ == "__main__":
    unittest.main()
