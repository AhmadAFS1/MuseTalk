"""CPU-only synthetic configuration/telemetry tests; never open a live session."""
import asyncio
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import live_isolated as plan
import live_client_evidence as client


class PlanTests(unittest.TestCase):
    def arguments(self):
        return ['/workspace/MuseTalk/models/test-native', '/workspace/MuseTalk/models/test-taesd',
                'a' * 20, '/workspace/MuseTalk/experiments/test_live_unique', ['test_api_avatar'], 'aiortc']

    def test_explicit_isolation_and_native_overrides(self):
        result = plan.prepare(*self.arguments())
        env = result['server_env']
        self.assertEqual(env['MUSETALK_UNET_STAGEWISE_CACHE_DIR'], self.arguments()[0])
        self.assertEqual(env['MUSETALK_TAESD_TRT_HW_COMPAT'], 'none')
        self.assertEqual(env['LINGUA_CONTROL_PLANE_ENABLED'], '0')
        self.assertEqual(env['LINGUA_CONTROL_PLANE_ENV_FILE'], '/dev/null')
        self.assertEqual(env['AVATAR_S3_ENABLED'], '0')
        self.assertEqual(env['MUSETALK_ENV_OVERRIDES_FILE'], '/dev/null')
        self.assertEqual(env['WEBRTC_LIFETIME_COUNTERS'], '1')
        self.assertEqual(env['WEBRTC_LIFETIME_SEND_RING'], '256')
        self.assertEqual(result['server_argv'][:2], ['env', '-i'])
        self.assertIn('scripts/box_guard.sh', result['server_argv'])
        self.assertFalse(any('SECRET' in k or 'TOKEN' in k or k.startswith('AWS_') for k in env))
        self.assertFalse(result['release_ready'])

    def test_cpu_actual_quota_disjoint_and_bounded(self):
        self.assertEqual(plan.prepare(*self.arguments())['observed_cpu_quota'], 18.43199)
        for server, peers in [('0-17', '18'), ('0-13', '12-15'), ('0-96', '1'), ('3-1', '4')]:
            with self.assertRaises(ValueError):
                plan.prepare(*self.arguments(), server_cpus=server, client_cpus=peers)

    def test_human_recording_and_unchanged_stage_gates(self):
        result = plan.prepare(*self.arguments())
        self.assertEqual(result['audio']['sha256'], plan.AUDIO_SHA)
        self.assertFalse(result['audio']['whole_corpus_is_human'])
        for stage, minimum, count in [('s0_n1', '10', 2), ('s0_n3', '10', 2),
                                      ('ramp_n5', '240', 8), ('ramp_n10', '240', 8),
                                      ('ramp_n15', '240', 8), ('soak', '3600', 64)]:
            argv = result['client_stage_argv'][stage]
            self.assertEqual(argv[argv.index('--min-steady-s') + 1], minimum)
            self.assertEqual(argv[argv.index('--wav-list') + 1].split(','), [plan.AUDIO] * count)
            self.assertNotIn('--chain', argv)
            self.assertNotIn('--record-streams', argv)
            self.assertNotIn(',', argv[argv.index('--levels') + 1])
        self.assertIn('SELECT_RAMP_PASSING_N', result['client_stage_argv']['soak'])

    def test_unsafe_or_ambiguous_paths_and_ids_rejected(self):
        for index, value in [(0, '/workspace/MuseTalk/models/../foreign'), (0, '/tmp/native'),
                             (1, '/workspace/MuseTalk/`evil`'), (2, 'guessed'), (4, ['a', 'a'])]:
            args = self.arguments()
            args[index] = value
            with self.assertRaises(ValueError):
                plan.prepare(*args)


class TelemetryTests(unittest.TestCase):
    def summary(self):
        return {'level': 1, 'server': {'lifetime_counters': True}, 'unmeasured': [],
                'streams': [{'stream': 0, 'turns_ok': 1, 'turns_posted': 1,
                             'first_frame': [{'matched': True, 'first_frame_s': .4}],
                             'server_ring': {'speaking_slots': 100, 'ring_gaps': 0, 'ring_missing_entries': 0},
                             'setup_timing': {'received_payloads': [{'negotiated': [{'payload_type': 99, 'mime_type': 'video/h264'}],
                                                                    'packets_by_payload': {'99': 200}}]}}]}

    def test_complete_codec_and_first_frame_evidence(self):
        result = client.additional_evidence(self.summary())
        self.assertEqual(result['video_codecs_by_stream'], {'0': ['video/h264']})
        self.assertIn('NVENC from wire codec alone', result['not_claimed'])

    def test_vp8_fallback_does_not_pass_explicit_h264_experiment(self):
        data = self.summary()
        data['streams'][0]['setup_timing']['received_payloads'][0]['negotiated'][0]['mime_type'] = 'video/vp8'
        with self.assertRaisesRegex(ValueError, 'wire codec'):
            client.additional_evidence(data)

    def test_missing_lifetime_ring_firstframe_or_codec_invalid(self):
        variants = []
        for field, value in [('first_frame', []), ('first_frame', [{'matched': False}]),
                             ('first_frame', [{'matched': True, 'first_frame_s': float('nan')}]),
                             ('server_ring', {'speaking_slots': 100, 'ring_gaps': 1, 'ring_missing_entries': 20}),
                             ('setup_timing', {}), ('turns_ok', 0)]:
            result = self.summary()
            result['streams'][0][field] = value
            variants.append(result)
        result = self.summary()
        result['server']['lifetime_counters'] = False
        variants.append(result)
        for result in variants:
            with self.assertRaises(ValueError):
                client.additional_evidence(result)

    def test_offhost_explicitly_invalid_even_with_good_data(self):
        with self.assertRaisesRegex(ValueError, 'offhost'):
            client.additional_evidence(self.summary(), 'offhost')

    def test_unknown_payload_or_negotiation_only_not_codec_proof(self):
        for counts in ({'105': 10}, {}, {'99': 0}):
            data = self.summary()
            data['streams'][0]['setup_timing']['received_payloads'][0]['packets_by_payload'] = counts
            with self.assertRaises(ValueError):
                client.additional_evidence(data)

    def test_codec_hook_preserves_packet_and_result_and_counts_actual_payload(self):
        class Receiver:
            async def receive(self, params):
                self.params = params
                return 'received'
            async def _handle_rtp_packet(self, packet, arrival_time_ms):
                self.packet = packet
                self.arrival = arrival_time_ms
                return 'handled'
        client.install_codec_hooks(Receiver, '1.14.0')
        receiver = Receiver()
        codec = SimpleNamespace(payloadType=99, mimeType='video/H264', clockRate=90000)
        params = SimpleNamespace(codecs=[codec])
        self.assertEqual(asyncio.run(receiver.receive(params)), 'received')
        packet = SimpleNamespace(payload_type=99, payload=b'do-not-record-encoded-data')
        self.assertEqual(asyncio.run(receiver._handle_rtp_packet(packet, 123)), 'handled')
        self.assertIs(receiver.packet, packet)
        self.assertEqual(receiver.arrival, 123)
        self.assertEqual(receiver._repro_codec_evidence['packets_by_payload'], {'99': 1})
        self.assertNotIn('do-not-record', json.dumps(receiver._repro_codec_evidence))
        with self.assertRaises(ValueError):
            client.install_codec_hooks(Receiver, '1.15.0')

    def test_missing_evidence_never_preserves_legacy_pass(self):
        class Stream:
            async def setup(self):
                pass
        module = SimpleNamespace(StreamClient=Stream, summarize_level=lambda: {'verdict': 'PASS'})
        client.install_client_hooks(module)
        result = module.summarize_level()
        self.assertEqual(result['canonical_verdict_before_strict_evidence'], 'PASS')
        self.assertEqual(result['verdict'], 'INVALID')

    def test_encoder_snapshot_is_safe_and_never_per_call_nvenc_proof(self):
        result = client.safe_encoder_snapshot({'secret': 'dont-copy', 'h264': {'impl': 'nvenc',
                        'credential': 'dont-copy', 'nvenc_sessions': {'opened': 3, 'open_failures': 1, 'private': 'dont-copy'}}})
        self.assertFalse(result['per_stream_encoder_proven'])
        self.assertEqual(result['nvenc_sessions']['open_failures'], 1)
        self.assertNotIn('dont-copy', json.dumps(result))

    def test_canonical_client_spawn_uses_adapter_without_source_edit(self):
        source = (plan.ROOT / 'load_test_webrtc_v2.py').read_text()
        self.assertIn('str(Path(__file__).resolve()), "--shard-worker"', source)
        self.assertIn('"setup_timing": result.get("timing")', source)
        self.assertIn('module.__file__ = str(Path(__file__).resolve())', Path(client.__file__).read_text())

    def test_original_278ms_gap_is_not_accepted_by_adapter_scorer(self):
        row = {'window_s': 45, 'server_fresh_fraction': .999, 'server_held_run_max': 1,
               'client_content_1s_min': 20, 'pts_join_matched': 1, 'P1_rate': True, 'P2_fresh': True,
               'P3_no_buffering': True, 'anchored_1s_min_frames': 20, 'gap_max_ms': 278.15,
               'gaps_over_100ms_per_10min': 0, 'gap_causes': {}}
        result = client.strict_report.live({'streams': [row], 'summary': {}}, 1, 10)
        self.assertEqual(result['status'], 'FAIL')

    def test_original_smoke_missing_telemetry_stays_invalid(self):
        path = plan.ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/portable/source_install_smoke/source_install_smoke_n01.json'
        if not path.exists():
            self.skipTest('frozen operator artifact not present in this checkout')
        with self.assertRaisesRegex(ValueError, 'lifetime'):
            client.additional_evidence(json.loads(path.read_text()))

    def test_client_configuration_refuses_offhost_recording_and_audio_mutation(self):
        spec = importlib.util.spec_from_file_location('canonical_client_test', plan.ROOT / 'load_test_webrtc_v2.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp:
            args = module.parse_args(['--trace-dir', tmp + '/traces', '--out-dir', tmp + '/out',
                                      '--levels', '1',
                                      '--base-url', 'http://127.0.0.1:8300', '--ring', '256', '--batch-size', '16',
                                      '--wav-list', '/test.wav,/test.wav', '--settle-s', '10', '--peers-per-shard', '1',
                                      '--client-cpus', '14-17', '--ignore-ice-servers'])
            with patch.object(client.strict_report, 'sha256', return_value=plan.AUDIO_SHA):
                client.verify_client_args(args)
                for name, value in [('base_url', 'http://92.49.17.100:25659'), ('chain', True),
                                    ('record_streams', '0'), ('ring', 64), ('min_fresh_fraction', .99),
                                    ('no_pin', True), ('levels', [1, 3]), ('poll_interval_s', 0),
                                    ('poll_interval_s', float('nan')), ('poll_interval_s', float('inf'))]:
                    changed = copy.deepcopy(args)
                    setattr(changed, name, value)
                    with self.assertRaises(ValueError):
                        client.verify_client_args(changed)
            with patch.object(client.strict_report, 'sha256', return_value='0' * 64), self.assertRaises(ValueError):
                client.verify_client_args(args)


if __name__ == '__main__':
    unittest.main()
