"""CPU regressions for the threaded-decoder/Python-log shutdown deadlock."""
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


class NativeFFmpegLoggingTest(unittest.TestCase):
    def run_child(self, source, *args):
        # A regression can hold the GIL indefinitely. subprocess.run kills and
        # reaps this isolated child on timeout, keeping the test runner usable.
        return subprocess.run(
            [sys.executable, "-u", "-c", textwrap.dedent(source), *map(str, args)],
            cwd=Path(__file__).resolve().parent,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
            capture_output=True, text=True, timeout=20,
        )

    def test_repeated_configuration_keeps_errors_native_after_dependency_reactivation(self):
        result = self.run_child("""
            import av
            from scripts.runtime_av_logging import configure_native_ffmpeg_logging
            # This is the process-wide side effect of torchvision.io imports.
            av.logging.set_level(av.logging.ERROR)
            first = configure_native_ffmpeg_logging('test')
            assert first['previous_python_level'] == av.logging.ERROR, first
            assert av.logging.get_level() is None
            configure_native_ffmpeg_logging('test-repeat')
            av.logging.set_level(av.logging.ERROR)
            configure_native_ffmpeg_logging('test-after-lazy-import')
            with av.logging.Capture() as captured:
                av.logging.log(av.logging.ERROR, 'native-test', 'native-error-visible\\n')
                av.logging.log(av.logging.INFO, 'native-test', 'info-not-visible\\n')
            assert not captured, captured
            assert av.logging.get_level() is None
        """)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("native-error-visible", result.stderr)
        self.assertNotIn("info-not-visible", result.stderr)

    def test_partial_frame_threaded_decoders_close_after_torchvision_import(self):
        with tempfile.TemporaryDirectory() as directory:
            video = Path(directory) / "source.mp4"
            with av.open(str(video), "w") as container:
                stream = container.add_stream("libx264", rate=24)
                stream.width = stream.height = 128
                stream.pix_fmt = "yuv420p"
                stream.codec_context.thread_count = 1
                for index in range(96):
                    pixels = np.full((128, 128, 3), index * 2, np.uint8)
                    frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                    for packet in stream.encode(frame):
                        container.mux(packet)
                for packet in stream.encode(None):
                    container.mux(packet)
            result = self.run_child("""
                import faulthandler, gc, json, sys
                from pathlib import Path
                import av
                import torchvision
                from scripts.runtime_av_logging import configure_native_ffmpeg_logging
                from scripts.webrtc_tracks import IdleVideoStreamTrack
                faulthandler.dump_traceback_later(12)
                configured = configure_native_ffmpeg_logging('decoder-test')
                gc.disable()
                def threads(): return len(list(Path('/proc/self/task').iterdir()))
                baseline = threads()
                retained = []
                for cycle in range(100):
                    track = IdleVideoStreamTrack(sys.argv[1], decode_threads=16)
                    for _ in range(24 + cycle % 32): track.read_frame()
                    if cycle % 10 == 0:
                        track.reset()
                        track.read_frame()
                    track.stop()
                    track.stop()
                    retained.append(track)
                    assert threads() == baseline, (cycle, threads(), baseline)
                # Count with the same watchdog lifetime as the baseline. Its
                # native thread can exit immediately after cancellation.
                final_threads = threads()
                faulthandler.cancel_dump_traceback_later()
                print(json.dumps({'cycles': len(retained), 'threads': final_threads,
                                  'baseline': baseline, 'logging': configured}))
            """, video)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        evidence = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(evidence["cycles"], 100)
        self.assertEqual(evidence["threads"], evidence["baseline"])
        self.assertEqual(evidence["logging"]["callback"], "ffmpeg_native")


if __name__ == "__main__":
    unittest.main()
