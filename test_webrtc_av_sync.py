import asyncio
import fractions
import json
import math
import sys
import tempfile
import threading
import time
import types
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np


class _Plane:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.data = b""

    def update(self, data):
        self.data = bytes(data)


class _AudioFrame:
    def __init__(self, *, format="s16", layout="mono", samples=0):
        channels = 2 if layout == "stereo" else 1
        self.format = format
        self.layout = layout
        self.samples = samples
        self.planes = [_Plane(samples * channels * 2)]
        self.pts = None
        self.sample_rate = None
        self.time_base = None


class _VideoFrame:
    @classmethod
    def from_ndarray(cls, _array, format=None):
        return cls()

    def reformat(self, *args, **kwargs):
        return self


def _install_media_stubs_if_needed():
    try:
        import aiortc  # noqa: F401
    except ModuleNotFoundError:
        aiortc = types.ModuleType("aiortc")

        class _Track:
            def __init__(self):
                self.readyState = "live"

            def stop(self):
                self.readyState = "ended"

        class _PeerConnection:
            def __init__(self, _configuration=None):
                self.connectionState = "new"

            def on(self, _event):
                return lambda callback: callback

            async def close(self):
                self.connectionState = "closed"

        class _Configuration:
            def __init__(self, iceServers=None):
                self.iceServers = iceServers or []

        class _IceServer:
            def __init__(self, urls=None, username=None, credential=None):
                self.urls = urls
                self.username = username
                self.credential = credential

        aiortc.VideoStreamTrack = _Track
        aiortc.MediaStreamTrack = _Track
        aiortc.RTCPeerConnection = _PeerConnection
        aiortc.RTCConfiguration = _Configuration
        aiortc.RTCIceServer = _IceServer
        aiortc.RTCRtpSender = object
        sys.modules["aiortc"] = aiortc

        mediastreams = types.ModuleType("aiortc.mediastreams")
        mediastreams.MediaStreamError = RuntimeError
        sys.modules["aiortc.mediastreams"] = mediastreams

    try:
        import av  # noqa: F401
    except ModuleNotFoundError:
        av = types.ModuleType("av")
        av.AudioFrame = _AudioFrame
        av.VideoFrame = _VideoFrame
        av.open = lambda _path: None
        sys.modules["av"] = av


_install_media_stubs_if_needed()

from scripts.webrtc_audio_timeline import prepare_webrtc_audio_timeline
from scripts import webrtc_tracks
from scripts.test_pose_webrtc import WallClockAudioTrack, WallClockVideoTrack


class VideoSyncClockLoggingTest(unittest.TestCase):
    def test_logs_actual_audio_start_vs_video_start_with_turn_context(self):
        clock = webrtc_tracks.VideoSyncClock(30, strict_fifo=True)
        clock.reset()
        clock.set_turn_context("request-123", "session-456")

        with patch("builtins.print") as print_mock:
            clock.release_playout(time.monotonic())
            clock.note_first_live_rtp_alignment(
                audio_target_seconds=7.0,
                video_rtp_seconds=7.0,
                correction_seconds=0.0,
                max_mismatch_seconds=1.0 / 30.0,
            )
            clock.mark_first_video_frame()
            clock.note_first_tts_transport_pts(7.0)
            clock.mark_first_audio_packet()

        payloads = []
        for call in print_mock.call_args_list:
            message = str(call.args[0])
            if message.startswith("📐 WEBRTC_AV_TIMING "):
                payloads.append(json.loads(message.split(" ", 2)[2]))

        self.assertEqual(
            [payload["event"] for payload in payloads],
            [
                "playout_gate_released",
                "first_live_video_frame",
                "first_tts_audio_packet",
                "av_start_summary",
            ],
        )
        summary = payloads[-1]
        self.assertEqual(summary["request_id"], "request-123")
        self.assertEqual(summary["session_id"], "session-456")
        self.assertEqual(summary["first_tts_audio_rtp_seconds"], 7.0)
        self.assertEqual(summary["first_live_video_rtp_seconds"], 7.0)
        self.assertEqual(summary["audio_rtp_minus_video_rtp_ms"], 0.0)
        self.assertTrue(summary["rtp_aligned"])
        self.assertGreaterEqual(summary["absolute_start_skew_ms"], 0.0)

        stats = clock.get_stats()
        self.assertEqual(stats["turn_request_id"], "request-123")
        self.assertEqual(stats["turn_session_id"], "session-456")
        self.assertIsNotNone(stats["first_audio_packet_unix_ms"])
        self.assertIsNotNone(stats["first_video_frame_unix_ms"])


def _write_tone_with_edge_silence(
    path: Path,
    *,
    sample_rate: int = 16_000,
    leading_seconds: float = 0.80,
    tone_seconds: float = 0.50,
    trailing_seconds: float = 1.20,
) -> None:
    leading = np.zeros(round(leading_seconds * sample_rate), dtype="<i2")
    tone_samples = round(tone_seconds * sample_rate)
    phase = np.arange(tone_samples, dtype=np.float64) / float(sample_rate)
    tone = np.round(np.sin(2.0 * math.pi * 440.0 * phase) * 12_000).astype("<i2")
    trailing = np.zeros(round(trailing_seconds * sample_rate), dtype="<i2")
    samples = np.concatenate([leading, tone, trailing])
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(samples.tobytes())


def _wave_duration(path: str | Path) -> float:
    with wave.open(str(path), "rb") as source:
        return source.getnframes() / float(source.getframerate())


class WebRTCAudioTimelineTest(unittest.TestCase):
    def test_edge_silence_is_removed_once_into_the_shared_media_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "tts_with_long_tail.wav"
            destination = Path(temp_dir) / "tts_shared_timeline.wav"
            _write_tone_with_edge_silence(source)

            timeline = prepare_webrtc_audio_timeline(
                source,
                output_path=destination,
                threshold_db=-40.0,
                leading_padding_seconds=0.08,
                trailing_padding_seconds=0.12,
                minimum_trim_seconds=0.10,
            )

            self.assertTrue(timeline.normalized)
            self.assertEqual(Path(timeline.media_path), destination)
            self.assertAlmostEqual(timeline.original_duration_seconds, 2.50, places=2)
            self.assertAlmostEqual(timeline.speech_start_seconds, 0.80, delta=0.02)
            self.assertAlmostEqual(timeline.speech_end_seconds, 1.30, delta=0.02)
            self.assertAlmostEqual(timeline.trim_start_seconds, 0.72, delta=0.02)
            self.assertAlmostEqual(timeline.trim_end_seconds, 1.42, delta=0.02)

            # This exact file is the one both MuseTalk generation and WebRTC
            # audio playout must consume; its decoded duration is authoritative.
            decoded_duration = _wave_duration(timeline.media_path)
            self.assertAlmostEqual(
                decoded_duration,
                timeline.media_duration_seconds,
                delta=1.0 / 16_000,
            )
            self.assertAlmostEqual(decoded_duration, 0.70, delta=0.04)

            second_pass = prepare_webrtc_audio_timeline(
                timeline.media_path,
                output_path=Path(temp_dir) / "must_not_be_created.wav",
                threshold_db=-40.0,
                leading_padding_seconds=0.08,
                trailing_padding_seconds=0.12,
                minimum_trim_seconds=0.10,
            )
            self.assertFalse(second_pass.normalized)
            self.assertEqual(second_pass.media_path, timeline.media_path)
            self.assertAlmostEqual(
                second_pass.media_duration_seconds,
                timeline.media_duration_seconds,
                delta=1.0 / 16_000,
            )

    def test_all_silent_audio_is_rejected_instead_of_starting_live_lipsync(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "silence.wav"
            samples = np.zeros(16_000, dtype="<i2")
            with wave.open(str(source), "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(16_000)
                output.writeframes(samples.tobytes())

            with self.assertRaisesRegex(ValueError, "no sustained activity"):
                prepare_webrtc_audio_timeline(source)


class ProofRecorderTimestampTest(unittest.IsolatedAsyncioTestCase):
    async def test_wall_delay_does_not_insert_audio_media_gap(self):
        class _JumpingClock:
            def __init__(self):
                self.values = [1.0, 1.2, 1.2]

            def elapsed(self):
                return self.values.pop(0)

        class _Frame:
            def __init__(self, pts):
                self.pts = pts
                self.time_base = fractions.Fraction(1, 48_000)
                self.sample_rate = 48_000
                self.samples = 960

        class _BurstSource:
            def __init__(self):
                self.frames = [_Frame(0), _Frame(960), _Frame(1_920)]

            async def recv(self):
                return self.frames.pop(0)

        # A recorder callback can run late while RTP frames wait in the relay.
        # The proof file must retain their sender timestamps instead of adding
        # a permanent wall-clock silence hole.
        track = WallClockAudioTrack(_BurstSource(), _JumpingClock())
        frames = [await track.recv() for _ in range(3)]

        self.assertEqual([frame.pts for frame in frames], [48_000, 48_960, 49_920])
        self.assertEqual(track.get_stats()["source_timestamp_anomalies"], 0)

    async def test_burst_delivery_preserves_sender_video_frame_spacing(self):
        class _FixedClock:
            @staticmethod
            def elapsed():
                return 1.25

        class _Frame:
            def __init__(self, pts):
                self.pts = pts
                self.time_base = fractions.Fraction(1, 90_000)

        class _BurstSource:
            def __init__(self):
                self.frames = [_Frame(0), _Frame(3_000), _Frame(6_000)]

            async def recv(self):
                return self.frames.pop(0)

        # All three frames arrive without advancing receiver wall time. The
        # recorder must retain the sender's 30 fps RTP spacing instead of
        # assigning duplicate encoder timestamps and dropping the burst.
        track = WallClockVideoTrack(_BurstSource(), _FixedClock())
        frames = [await track.recv() for _ in range(3)]

        self.assertEqual(
            [frame.pts for frame in frames],
            [112_500, 115_500, 118_500],
        )
        self.assertTrue(
            all(frame.time_base == fractions.Fraction(1, 90_000) for frame in frames)
        )

class _FakeIdleFrame:
    def __init__(self, source: str, index: int):
        self.source = source
        self.index = index
        self.pts = None
        self.time_base = None

    def reformat(self, *args, **kwargs):
        return self


class _FakeIdleTrack:
    def __init__(self, video_path, fps=None):
        self.video_path = str(video_path)
        self.fps = fps or 30
        self.next_index = 0
        self.last_index = None
        self.completed_cycles = 0

    def next_frame_starts_cycle(self):
        return False

    def last_read_started_cycle(self):
        return False

    def read_frame(self):
        frame = _FakeIdleFrame(self.video_path, self.next_index)
        self.last_index = self.next_index
        self.next_index += 1
        return frame

    def get_timing(self):
        return {
            "source_frame_index": self.last_index or 0,
            "source_frame_count": 300,
            "source_fps": self.fps,
            "completed_cycles": self.completed_cycles,
        }

    def stop(self):
        pass


class TimestampLockedAudioTrackTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Other test modules also provide intentionally small PyAV stubs. Patch
        # only for these unit tests so discovery order and optional dependencies
        # do not affect the result.
        self.original_audio_frame = webrtc_tracks.av.AudioFrame
        webrtc_tracks.av.AudioFrame = _AudioFrame

    def tearDown(self):
        webrtc_tracks.av.AudioFrame = self.original_audio_frame

    async def test_packet_pts_stays_contiguous_and_exact_eof_marks_shared_clock(self):
        class _ImmediateCoverageClock(webrtc_tracks.VideoSyncClock):
            async def wait_for_audio_coverage(self, target_seconds, timeout=None):
                return 0.0

        clock = _ImmediateCoverageClock(30, strict_fifo=True)
        clock.reset()
        clock.mark_started()
        t0 = clock.release_playout(time.monotonic() - 1.0)

        track = webrtc_tracks.SyncedAudioStreamTrack(
            "unused.wav",
            use_ffmpeg_convert=False,
            sync_clock=clock,
        )
        packet_bytes = track._samples_per_frame * track._bytes_per_sample
        track._audio_samples = b"\x01\x00" * (
            track._samples_per_frame * 3
        )
        self.assertEqual(len(track._audio_samples), packet_bytes * 3)
        track._fully_loaded = True
        track.signal_start(t0)

        frames = [await track.recv() for _ in range(3)]

        self.assertEqual([frame.pts for frame in frames], [0, 960, 1920])
        self.assertTrue(track.get_stats()["eof"])
        self.assertTrue(clock.audio_complete.is_set())
        self.assertAlmostEqual(clock.audio_media_seconds, 0.06, places=6)
        await track.wait_for_eof(timeout=0.01)

    async def test_timestamp_lock_does_not_pause_audio_for_video_coverage(self):
        class _NoCoverageWaitClock(webrtc_tracks.VideoSyncClock):
            async def wait_for_audio_coverage(self, target_seconds, timeout=None):
                raise AssertionError("timestamp-locked audio must not wait for video coverage")

        clock = _NoCoverageWaitClock(30, strict_fifo=True)
        clock.reset()
        clock.mark_started()
        t0 = clock.release_playout(time.monotonic() - 1.0)
        track = webrtc_tracks.SyncedAudioStreamTrack(
            "unused.wav",
            use_ffmpeg_convert=False,
            sync_clock=clock,
        )
        track._audio_samples = b"\x00\x00" * (track._samples_per_frame * 2)
        track._fully_loaded = True
        track.signal_start(t0)

        first = await track.recv()
        second = await track.recv()

        self.assertEqual(second.pts - first.pts, track._samples_per_frame)
        self.assertEqual(track.get_stats()["strict_audio_stalls"], 0)


class PersistentAudioTransportTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_audio_frame = webrtc_tracks.av.AudioFrame
        webrtc_tracks.av.AudioFrame = _AudioFrame

    def tearDown(self):
        webrtc_tracks.av.AudioFrame = self.original_audio_frame

    async def test_idle_tts_idle_uses_one_unbroken_sample_clock(self):
        transport = webrtc_tracks.SilenceAudioStreamTrack()
        first_idle = await transport.recv()
        # Keep the test instantaneous without changing media timestamps.
        transport._transport_start_time -= 1.0
        second_idle = await transport.recv()

        clock = webrtc_tracks.VideoSyncClock(30, strict_fifo=True)
        clock.reset()
        t0 = clock.release_playout(time.monotonic() - 1.0)
        source = webrtc_tracks.SyncedAudioStreamTrack(
            "unused.wav",
            use_ffmpeg_convert=False,
            sync_clock=clock,
        )
        source._audio_samples = b"\x01\x00" * (source._samples_per_frame * 2)
        source._fully_loaded = True
        transport.arm_source(source, sync_clock=clock, start_time=t0)

        # Prebuffer/release alone must not leak TTS ahead of the first live
        # video frame. The persistent sender keeps transmitting silence while
        # the staged source waits behind the shared start gate.
        gated_idle = await transport.recv()
        self.assertEqual(gated_idle.planes[0].data, b"\x00" * 1920)
        self.assertTrue(transport.get_stats()["source_armed"])
        clock.note_first_live_rtp_alignment(
            audio_target_seconds=0.06,
            video_rtp_seconds=0.06,
            correction_seconds=0.0,
            max_mismatch_seconds=1.0 / 30.0,
        )
        clock.mark_started()

        first_tts = await transport.recv()
        final_tts = await transport.recv()
        first_idle_after_tts = await transport.recv()

        frames = [
            first_idle,
            second_idle,
            gated_idle,
            first_tts,
            final_tts,
            first_idle_after_tts,
        ]
        self.assertEqual(
            [frame.pts for frame in frames],
            [0, 960, 1920, 2880, 3840, 4800],
        )
        self.assertEqual(first_idle.planes[0].data, b"\x00" * 1920)
        self.assertEqual(first_tts.planes[0].data, b"\x01\x00" * 960)
        self.assertEqual(first_idle_after_tts.planes[0].data, b"\x00" * 1920)
        self.assertTrue(clock.audio_complete.is_set())
        stats = transport.get_stats()
        self.assertEqual(stats["transport"], "persistent_timestamp_locked")
        self.assertEqual(stats["turns_started"], 1)
        self.assertEqual(stats["turns_completed"], 1)
        self.assertFalse(stats["source_active"])

    async def test_final_tts_packet_completes_on_the_following_silence_tick(self):
        transport = webrtc_tracks.SilenceAudioStreamTrack()
        clock = webrtc_tracks.VideoSyncClock(30, strict_fifo=True)
        clock.reset()
        clock.note_first_live_rtp_alignment(
            audio_target_seconds=0.0,
            video_rtp_seconds=0.0,
            correction_seconds=0.0,
            max_mismatch_seconds=1.0 / 30.0,
        )
        clock.mark_started()
        t0 = clock.release_playout(time.monotonic() - 1.0)
        source = webrtc_tracks.SyncedAudioStreamTrack(
            "unused.wav",
            use_ffmpeg_convert=False,
            sync_clock=clock,
        )
        source._audio_samples = b"\x01\x00" * source._samples_per_frame
        source._fully_loaded = True
        transport.arm_source(source, sync_clock=clock, start_time=t0)

        final_tts = await transport.recv()

        self.assertEqual(final_tts.planes[0].data, b"\x01\x00" * 960)
        self.assertTrue(transport.get_stats()["source_finishing"])
        self.assertFalse(source.get_stats()["eof"])
        self.assertFalse(clock.audio_complete.is_set())

        # Keep this unit test instantaneous without altering the sample clock.
        transport._transport_start_time -= 1.0
        first_idle = await transport.recv()

        self.assertEqual(first_idle.planes[0].data, b"\x00" * 1920)
        self.assertTrue(source.get_stats()["eof"])
        self.assertTrue(clock.audio_complete.is_set())
        self.assertEqual(transport.get_stats()["turns_completed"], 1)
        self.assertFalse(transport.get_stats()["source_active"])

    async def test_late_audio_callback_reanchors_instead_of_catching_up(self):
        transport = webrtc_tracks.SilenceAudioStreamTrack()
        await transport.recv()

        # Model a blocked event loop. The next callback is late by 500 ms and
        # must not make the following callback return immediately in a burst.
        transport._transport_start_time -= 0.5
        await transport.recv()
        started_at = time.monotonic()
        await transport.recv()
        elapsed = time.monotonic() - started_at

        self.assertGreaterEqual(elapsed, 0.012)
        self.assertGreaterEqual(transport.get_stats()["pace_reanchors"], 1)


class TimestampLockedVideoTrackTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_idle_track = webrtc_tracks.IdleVideoStreamTrack
        self.original_audio_sync_strategy = webrtc_tracks.WEBRTC_AUDIO_SYNC_STRATEGY
        webrtc_tracks.IdleVideoStreamTrack = _FakeIdleTrack
        webrtc_tracks.WEBRTC_AUDIO_SYNC_STRATEGY = "timestamp_locked"

    def tearDown(self):
        webrtc_tracks.IdleVideoStreamTrack = self.original_idle_track
        webrtc_tracks.WEBRTC_AUDIO_SYNC_STRATEGY = self.original_audio_sync_strategy

    @staticmethod
    def _make_track(*, source_fps=10, output_fps=30, max_queue=8):
        clock = webrtc_tracks.VideoSyncClock(source_fps, strict_fifo=True)
        track = webrtc_tracks.SwitchableVideoStreamTrack(
            "idle.mp4",
            source_fps=source_fps,
            output_fps=output_fps,
            max_queue=max_queue,
            prebuffer_seconds=0,
            adaptive_fps=False,
            sync_clock=clock,
        )
        return track, clock

    async def test_producer_audio_jump_does_not_skip_receiver_timeline_frames(self):
        track, clock = self._make_track()
        track.start_live()
        clock.release_playout(time.monotonic() - 1.0)

        empty_queue_frame = await asyncio.wait_for(track.recv(), timeout=0.15)

        self.assertEqual(empty_queue_frame.source, "idle.mp4")
        self.assertEqual(track.get_stats()["strict_video_stalls"], 0)

        for index in range(4):
            track._queue.put_nowait(_FakeIdleFrame("live", index))
        clock.mark_audio_progress(0.35)

        first_live = await asyncio.wait_for(track.recv(), timeout=0.15)

        # Producer-side progress jumped to 350 ms, but the receiver will still
        # hear the contiguous RTP samples from media time zero. Video therefore
        # begins at source frame zero instead of skipping ahead to frame three.
        self.assertEqual(first_live.source, "live")
        self.assertEqual(first_live.index, 0)
        self.assertEqual(track._live_source_consumed, 1)
        self.assertEqual(track.get_stats()["frames_dropped"], 0)
        self.assertEqual(track.get_stats()["strict_video_stalls"], 0)

        await track.recv()
        await track.recv()
        fourth_output = await track.recv()
        self.assertEqual(fourth_output.index, 1)

    async def test_underrun_refill_recovers_sequentially_without_frame_skip(self):
        track, clock = self._make_track()
        track.start_live()
        clock.release_playout(time.monotonic() - 1.0)
        track._queue.put_nowait(_FakeIdleFrame("live", 0))

        first_live = await track.recv()
        self.assertEqual(first_live.index, 0)

        # Let the receiver-visible video clock advance while inference is
        # empty.  This accumulates a source-frame deficit larger than one.
        for _ in range(9):
            track._last_ts -= 1.0
            held = await track.recv()
            self.assertEqual(held.index, 0)

        for index in range(1, 5):
            track._queue.put_nowait(_FakeIdleFrame("live", index))

        track._last_ts -= 1.0
        recovered = await track.recv()

        # Recovery must show frame one next.  The old deficit-based pop would
        # discard frames one and two and jump directly to frame three.
        self.assertEqual(recovered.index, 1)
        self.assertEqual(track._live_source_consumed, 2)
        self.assertEqual(track.get_stats()["frames_dropped"], 0)

    async def test_stop_with_full_queue_unblocks_late_producer(self):
        track, _clock = self._make_track(max_queue=1)
        generation_id = track.start_live()
        first = _FakeIdleFrame("live", 0)
        second = _FakeIdleFrame("live", 1)
        await track._push_video_frame(
            first,
            time.monotonic(),
            0.0,
            generation_id=generation_id,
        )

        blocked_push = asyncio.create_task(
            track._push_video_frame(
                second,
                time.monotonic(),
                0.0,
                generation_id=generation_id,
            )
        )
        await asyncio.sleep(0)
        self.assertFalse(blocked_push.done())

        track.stop()
        accepted = await asyncio.wait_for(blocked_push, timeout=0.1)

        self.assertFalse(accepted)
        self.assertFalse(track.get_stats()["live_active"])
        self.assertEqual(track._queue.qsize(), 0)

    async def test_neutral_waits_for_complete_audio_rtp_media_horizon(self):
        track, clock = self._make_track(source_fps=10, output_fps=30)
        generation_id = track.start_live()
        clock.release_playout(time.monotonic() - 1.0)
        for index in range(2):
            track._queue.put_nowait(_FakeIdleFrame("live", index))

        first_live = await track.recv()
        self.assertEqual(first_live.source, "live")
        track.signal_generation_complete(generation_id)
        clock.mark_audio_complete(0.20)

        # 200 ms at 30 fps requires six receiver-visible live video frames.
        remaining_live = [await track.recv() for _ in range(5)]
        self.assertTrue(all(frame.source == "live" for frame in remaining_live))
        self.assertTrue(track.get_stats()["audio_media_horizon_reached"])

        first_idle = await track.recv()
        self.assertEqual(first_idle.source, "idle.mp4")
        final_stats = track.get_stats()
        self.assertFalse(final_stats["live_active"])
        self.assertEqual(final_stats["last_live_output_frames"], 6)
        self.assertEqual(final_stats["last_required_live_output_frames"], 6)

    async def test_live_rtp_phase_anchors_to_persistent_audio_packet(self):
        track, clock = self._make_track(source_fps=10, output_fps=30)
        # Model independently free-running idle transports: video is at 3.0s,
        # while the next persistent audio packet is at 3.2s.
        track._rtp_frame_index = 90
        clock.publish_audio_transport_next_pts(3.2)
        generation_id = track.start_live()
        clock.release_playout(time.monotonic() - 1.0)
        for index in range(2):
            track._queue.put_nowait(_FakeIdleFrame("live", index))

        first_live = await track.recv()
        first_live_seconds = float(first_live.pts * first_live.time_base)
        track.signal_generation_complete(generation_id)
        clock.mark_audio_complete(0.20)
        remaining_live = [await track.recv() for _ in range(5)]
        first_idle = await track.recv()

        self.assertTrue(all(frame.source == "live" for frame in remaining_live))
        self.assertAlmostEqual(
            first_live_seconds,
            3.2,
            places=6,
        )
        self.assertAlmostEqual(
            float(first_idle.pts * first_idle.time_base),
            3.4,
            places=6,
        )
        stats = track.get_stats()
        self.assertEqual(stats["last_live_rtp_phase_correction_frames"], 6)
        self.assertAlmostEqual(
            stats["sync_clock"]["first_live_audio_target_seconds"],
            3.2,
            places=6,
        )
        self.assertAlmostEqual(
            stats["sync_clock"]["first_live_video_rtp_seconds"],
            3.2,
            places=6,
        )

    async def test_video_ahead_rebases_silent_audio_before_first_tts(self):
        track, clock = self._make_track(source_fps=10, output_fps=30)
        transport = webrtc_tracks.SilenceAudioStreamTrack(sync_clock=clock)
        track._rtp_frame_index = 96  # Video next PTS is 3.2 seconds.
        transport._timestamp = 144_000  # Audio next PTS is 3.0 seconds.
        transport._frames_sent = 150
        transport._transport_start_time = time.monotonic() - 3.0

        generation_id = track.start_live()
        release_time = clock.release_playout(time.monotonic() - 1.0)
        source = webrtc_tracks.SyncedAudioStreamTrack(
            "unused.wav",
            use_ffmpeg_convert=False,
            sync_clock=clock,
        )
        source._audio_samples = b"\x01\x00" * (source._samples_per_frame * 2)
        source._fully_loaded = True
        transport.arm_source(
            source,
            sync_clock=clock,
            start_time=release_time,
        )

        # Even an accidentally early start signal must not leak TTS before the
        # first-live RTP anchor is known.
        clock.mark_started()
        premature_audio = await transport.recv()
        self.assertEqual(premature_audio.pts, 144_000)
        self.assertEqual(source._read_position, 0)
        self.assertTrue(transport.get_stats()["source_armed"])
        self.assertIsNone(clock.first_live_video_rtp_seconds)

        track._queue.put_nowait(_FakeIdleFrame("live", 0))
        first_live = await track.recv()
        first_live_seconds = float(first_live.pts * first_live.time_base)
        self.assertAlmostEqual(first_live_seconds, 3.2, places=6)

        # Keep the unit test instantaneous; this changes wall pacing only, not
        # the persistent RTP/sample counters under test.
        transport._transport_start_time -= 1.0
        first_tts = await transport.recv()
        first_tts_seconds = first_tts.pts / float(transport.sample_rate)
        self.assertAlmostEqual(first_tts_seconds, 3.2, places=6)
        self.assertGreater(first_tts.pts, premature_audio.pts)
        self.assertGreaterEqual(
            first_tts.pts,
            premature_audio.pts + transport.samples,
        )

        sync_stats = clock.get_stats()
        self.assertAlmostEqual(
            sync_stats["first_tts_transport_pts_seconds"],
            first_live_seconds,
            places=6,
        )
        self.assertAlmostEqual(
            source.get_stats()["first_tts_transport_pts_seconds"],
            first_live_seconds,
            places=6,
        )
        self.assertAlmostEqual(
            sync_stats["audio_rtp_phase_correction_seconds"],
            0.18,
            places=6,
        )
        self.assertTrue(sync_stats["first_live_rtp_aligned"])
        self.assertLessEqual(
            sync_stats["first_live_rtp_abs_mismatch_seconds"],
            sync_stats["first_live_rtp_max_mismatch_seconds"],
        )

        # Absolute RTP rebasing must not change the turn-relative media horizon.
        track.signal_generation_complete(generation_id)
        clock.mark_audio_complete(0.20)
        remaining_live = [await track.recv() for _ in range(5)]
        first_idle = await track.recv()
        self.assertTrue(all(frame.source == "live" for frame in remaining_live))
        self.assertEqual(first_idle.source, "idle.mp4")
        self.assertAlmostEqual(
            float(first_idle.pts * first_idle.time_base),
            3.4,
            places=6,
        )

    async def test_subframe_video_lead_does_not_rebase_audio(self):
        track, clock = self._make_track(source_fps=10, output_fps=30)
        transport = webrtc_tracks.SilenceAudioStreamTrack(sync_clock=clock)
        track._rtp_frame_index = 91  # 3.033333 seconds.
        transport._timestamp = 144_960  # 3.02 seconds.
        transport._frames_sent = 151
        transport._transport_start_time = time.monotonic() - 3.02

        track.start_live()
        release_time = clock.release_playout(time.monotonic() - 1.0)
        source = webrtc_tracks.SyncedAudioStreamTrack(
            "unused.wav",
            use_ffmpeg_convert=False,
            sync_clock=clock,
        )
        source._audio_samples = b"\x01\x00" * source._samples_per_frame
        source._fully_loaded = True
        transport.arm_source(
            source,
            sync_clock=clock,
            start_time=release_time,
        )
        track._queue.put_nowait(_FakeIdleFrame("live", 0))

        first_live = await track.recv()
        transport._transport_start_time -= 1.0
        first_tts = await transport.recv()

        self.assertAlmostEqual(
            float(first_live.pts * first_live.time_base),
            91.0 / 30.0,
            places=6,
        )
        self.assertAlmostEqual(
            first_tts.pts / float(transport.sample_rate),
            3.02,
            places=6,
        )
        stats = clock.get_stats()
        self.assertIsNone(stats["audio_transport_rebase_target_seconds"])
        self.assertEqual(stats["audio_rtp_phase_correction_seconds"], 0.0)
        self.assertTrue(stats["first_live_rtp_aligned"])
        self.assertLessEqual(
            stats["first_live_rtp_abs_mismatch_seconds"],
            1.0 / 30.0,
        )

    async def test_blocked_old_producer_cannot_leak_into_the_next_generation(self):
        track, clock = self._make_track(max_queue=1)
        old_generation_id = track.start_live()
        old_head = _FakeIdleFrame("old_generation", 0)
        old_late = _FakeIdleFrame("old_generation", 1)
        await track._push_video_frame(old_head, time.monotonic(), 0.0)

        blocked_push = asyncio.create_task(
            track._push_video_frame(old_late, time.monotonic(), 0.0)
        )
        await asyncio.sleep(0)
        self.assertFalse(blocked_push.done())

        # end_live drains the full queue and wakes the old producer. Starting
        # the next turn before that task resumes exercises the ownership race.
        track.end_live()
        new_generation_id = track.start_live()
        self.assertFalse(await asyncio.wait_for(blocked_push, timeout=0.15))
        clock.release_playout(time.monotonic() - 1.0)

        stale_filtered = await asyncio.wait_for(track.recv(), timeout=0.15)
        self.assertEqual(stale_filtered.source, "idle.mp4")

        current = _FakeIdleFrame("current_generation", 0)
        await track._push_video_frame(current, time.monotonic(), 0.0)
        current_frame = await asyncio.wait_for(track.recv(), timeout=0.15)

        self.assertEqual(current_frame.source, "current_generation")
        self.assertNotEqual(current_frame.source, "old_generation")

        # Also cover an old callback whose coroutine does not begin until the
        # next generation is already active. Explicit callback ownership must
        # reject both its frame and its completion signal.
        late_old = _FakeIdleFrame("late_old_generation", 2)
        self.assertFalse(
            await track._push_video_frame(
                late_old,
                time.monotonic(),
                0.0,
                generation_id=old_generation_id,
            )
        )
        self.assertFalse(
            track.signal_generation_complete(old_generation_id)
        )
        self.assertFalse(track.get_stats()["generation_complete"])
        self.assertEqual(track._live_generation_id, new_generation_id)

    async def test_completion_idle_frame_zero_is_decoded_and_staged(self):
        track, _clock = self._make_track(source_fps=30, output_fps=30)
        track.start_live()

        staged = await track.stage_completion_idle_video(
            "completion_idle.mp4",
            pose_id="neutral_resting",
        )

        self.assertTrue(staged["staged"])
        self.assertTrue(track.get_stats()["completion_idle_staged"])
        pending = track._completion_idle_switch
        self.assertEqual(pending["first_frame"].source, "completion_idle.mp4")
        self.assertEqual(pending["first_frame"].index, 0)
        self.assertEqual(pending["idle_track"].next_index, 1)

        track.end_live()
        first_idle = await asyncio.wait_for(track.recv(), timeout=0.15)

        self.assertEqual(first_idle.source, "completion_idle.mp4")
        self.assertEqual(first_idle.index, 0)
        self.assertEqual(track.get_pose_status()["current_pose_id"], "neutral_resting")
        self.assertFalse(track.get_stats()["completion_idle_staged"])
        self.assertEqual(track._idle.next_index, 1)

    async def test_cancelled_completion_predecode_closes_eventual_decoder(self):
        entered = threading.Event()
        release = threading.Event()
        created = []

        class _BlockingIdleTrack(_FakeIdleTrack):
            def __init__(self, video_path, fps=None):
                super().__init__(video_path, fps=fps)
                self.stopped = False
                created.append(self)

            def read_frame(self):
                if self.video_path == "cancelled_completion.mp4":
                    entered.set()
                    release.wait(timeout=2.0)
                return super().read_frame()

            def stop(self):
                self.stopped = True

        original_idle_track = webrtc_tracks.IdleVideoStreamTrack
        webrtc_tracks.IdleVideoStreamTrack = _BlockingIdleTrack
        track = None
        try:
            track, _clock = self._make_track(source_fps=30, output_fps=30)
            staging = asyncio.create_task(
                track.stage_completion_idle_video(
                    "cancelled_completion.mp4",
                    pose_id="neutral_resting",
                )
            )
            self.assertTrue(await asyncio.to_thread(entered.wait, 1.0))

            staging.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await staging
            release.set()

            completion_idle = next(
                item for item in created
                if item.video_path == "cancelled_completion.mp4"
            )
            for _ in range(50):
                if completion_idle.stopped:
                    break
                await asyncio.sleep(0.01)
            self.assertTrue(completion_idle.stopped)
            self.assertIsNone(track._completion_idle_switch)
        finally:
            release.set()
            if track is not None:
                track.stop()
            webrtc_tracks.IdleVideoStreamTrack = original_idle_track

    async def test_track_close_during_predecode_discards_decoder(self):
        entered = threading.Event()
        release = threading.Event()
        created = []

        class _BlockingIdleTrack(_FakeIdleTrack):
            def __init__(self, video_path, fps=None):
                super().__init__(video_path, fps=fps)
                self.stopped = False
                created.append(self)

            def read_frame(self):
                if self.video_path == "closed_completion.mp4":
                    entered.set()
                    release.wait(timeout=2.0)
                return super().read_frame()

            def stop(self):
                self.stopped = True

        original_idle_track = webrtc_tracks.IdleVideoStreamTrack
        webrtc_tracks.IdleVideoStreamTrack = _BlockingIdleTrack
        track = None
        try:
            track, _clock = self._make_track(source_fps=30, output_fps=30)
            staging = asyncio.create_task(
                track.stage_completion_idle_video(
                    "closed_completion.mp4",
                    pose_id="neutral_resting",
                )
            )
            self.assertTrue(await asyncio.to_thread(entered.wait, 1.0))

            track.stop()
            release.set()
            result = await staging

            completion_idle = next(
                item for item in created
                if item.video_path == "closed_completion.mp4"
            )
            self.assertEqual(result["reason"], "track_closed")
            self.assertFalse(result["staged"])
            self.assertTrue(completion_idle.stopped)
            self.assertIsNone(track._completion_idle_switch)
        finally:
            release.set()
            if track is not None and not track._closed:
                track.stop()
            webrtc_tracks.IdleVideoStreamTrack = original_idle_track

    async def test_newer_completion_stage_supersedes_slower_predecode(self):
        slow_entered = threading.Event()
        release_slow = threading.Event()
        created = []

        class _RacingIdleTrack(_FakeIdleTrack):
            def __init__(self, video_path, fps=None):
                super().__init__(video_path, fps=fps)
                self.stopped = False
                created.append(self)

            def read_frame(self):
                if self.video_path == "slow_completion.mp4":
                    slow_entered.set()
                    release_slow.wait(timeout=2.0)
                return super().read_frame()

            def stop(self):
                self.stopped = True

        original_idle_track = webrtc_tracks.IdleVideoStreamTrack
        webrtc_tracks.IdleVideoStreamTrack = _RacingIdleTrack
        track = None
        try:
            track, _clock = self._make_track(source_fps=30, output_fps=30)
            slow_stage = asyncio.create_task(
                track.stage_completion_idle_video(
                    "slow_completion.mp4",
                    pose_id="neutral_resting",
                )
            )
            self.assertTrue(await asyncio.to_thread(slow_entered.wait, 1.0))

            fast_result = await track.stage_completion_idle_video(
                "fast_completion.mp4",
                pose_id="neutral_resting",
            )
            release_slow.set()
            slow_result = await slow_stage

            slow_idle = next(
                item for item in created
                if item.video_path == "slow_completion.mp4"
            )
            fast_idle = next(
                item for item in created
                if item.video_path == "fast_completion.mp4"
            )
            self.assertTrue(fast_result["staged"])
            self.assertFalse(slow_result["staged"])
            self.assertEqual(slow_result["reason"], "staging_superseded")
            self.assertTrue(slow_idle.stopped)
            self.assertFalse(fast_idle.stopped)
            self.assertEqual(
                track._completion_idle_switch["idle_video_path"],
                "fast_completion.mp4",
            )
        finally:
            release_slow.set()
            if track is not None:
                track.stop()
            webrtc_tracks.IdleVideoStreamTrack = original_idle_track

    async def test_audio_endpoint_supersedes_inflight_completion_predecode(self):
        entered = threading.Event()
        release = threading.Event()
        created = []

        class _BlockingIdleTrack(_FakeIdleTrack):
            def __init__(self, video_path, fps=None):
                super().__init__(video_path, fps=fps)
                self.stopped = False
                created.append(self)

            def read_frame(self):
                if self.video_path == "late_completion.mp4":
                    entered.set()
                    release.wait(timeout=2.0)
                return super().read_frame()

            def stop(self):
                self.stopped = True

        original_idle_track = webrtc_tracks.IdleVideoStreamTrack
        webrtc_tracks.IdleVideoStreamTrack = _BlockingIdleTrack
        track = None
        try:
            track, _clock = self._make_track(source_fps=30, output_fps=30)
            track.start_live()
            staging = asyncio.create_task(
                track.stage_completion_idle_video(
                    "late_completion.mp4",
                    pose_id="neutral_resting",
                )
            )
            self.assertTrue(await asyncio.to_thread(entered.wait, 1.0))

            # Simulate the audio endpoint arriving before decoder startup
            # finishes. The late result must not reinstall itself afterward.
            track.end_live()
            release.set()
            result = await staging

            late_idle = next(
                item for item in created
                if item.video_path == "late_completion.mp4"
            )
            self.assertFalse(result["staged"])
            self.assertEqual(result["reason"], "staging_superseded")
            self.assertTrue(late_idle.stopped)
            self.assertIsNone(track._completion_idle_switch)
            self.assertEqual(track._current_idle_video_path, "idle.mp4")
        finally:
            release.set()
            if track is not None:
                track.stop()
            webrtc_tracks.IdleVideoStreamTrack = original_idle_track


class AudioCompletionIdleHandoffTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.original_idle_track = webrtc_tracks.IdleVideoStreamTrack
        webrtc_tracks.IdleVideoStreamTrack = _FakeIdleTrack

    def tearDown(self):
        webrtc_tracks.IdleVideoStreamTrack = self.original_idle_track

    async def test_audio_completion_discards_lingering_live_frames_and_returns_idle(self):
        clock = webrtc_tracks.VideoSyncClock(30, strict_fifo=True)
        track = webrtc_tracks.SwitchableVideoStreamTrack(
            "idle.mp4",
            source_fps=30,
            output_fps=30,
            prebuffer_seconds=0,
            adaptive_fps=False,
            sync_clock=clock,
        )
        track.start_live()
        first_live = _FakeIdleFrame("live", 0)
        lingering_live = _FakeIdleFrame("live", 1)
        track._queue.put_nowait(first_live)
        track._queue.put_nowait(lingering_live)
        clock.release_playout(time.monotonic() - 1.0)

        first = await track.recv()
        self.assertEqual(first.source, "live")
        clock.mark_audio_complete(1.0 / 30.0)

        after_eof = await track.recv()

        self.assertEqual(after_eof.source, "idle.mp4")
        self.assertFalse(track.get_stats()["live_active"])
        self.assertEqual(track.get_stats()["queue_size"], 0)
        await track.wait_for_playback_complete(timeout=0.01)


if __name__ == "__main__":
    unittest.main()
