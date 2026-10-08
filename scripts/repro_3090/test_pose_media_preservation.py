import hashlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import preserve_production_pose_media as media
import run_isolated_live_stage as live
import warm_isolated_api as warm


class PreservationContracts(unittest.TestCase):
    def test_exact_shape_dtype_prefix_and_truncation(self):
        raw = bytes(512 * 896 * 3)
        class Proc:
            def __init__(self, value):
                self.stdout = io.BytesIO(value)
            def wait(self, timeout=None):
                return 0
            def poll(self):
                return 0
        with patch.object(media.subprocess, 'Popen', return_value=Proc(raw)):
            hashes = media.decoded_frame_hashes(Path('/test.mkv'), 512, 896)
        self.assertEqual(hashes, [hashlib.sha256(b'(896, 512, 3)|uint8|' + raw).hexdigest()])
        with patch.object(media.subprocess, 'Popen', return_value=Proc(raw[:-1])):
            with self.assertRaisesRegex(ValueError, 'truncated'):
                media.decoded_frame_hashes(Path('/test.mkv'), 512, 896)
        with self.assertRaisesRegex(ValueError, 'resolution'):
            media.decoded_frame_hashes(Path('/test.mkv'), 256, 256)

    def test_human_pcm_is_byte_exact(self):
        with patch.object(media.subprocess, 'check_output', return_value=b'pcm'):
            self.assertEqual(media.verify_audio(Path('/test.mkv'), b'pcm'), hashlib.sha256(b'pcm').hexdigest())
            with self.assertRaisesRegex(ValueError, 'human audio'):
                media.verify_audio(Path('/test.mkv'), b'other')

    def test_record_cannot_escape_or_follow_symlinks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / 'record'
            target.write_bytes(b'fixture')
            (root / 'alias').symlink_to(target)
            with patch.object(warm, 'ROOT', root):
                self.assertEqual(media.verify_record({'path': str(target), 'bytes': 7,
                                  'sha256': hashlib.sha256(b'fixture').hexdigest()}), target)
                for path in (root / 'alias', root / '..' / 'other'):
                    with self.assertRaises(ValueError):
                        media.safe_path(path)
                with self.assertRaisesRegex(ValueError, 'SHA mismatch'):
                    media.verify_record({'path': str(target), 'bytes': 7, 'sha256': '0' * 64})

    def test_source_report_hash_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'report.json'
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'terminal report SHA'):
                media.load_report(path)

    def test_reused_proof_requires_all48_exact_digests_and_pcm(self):
        data = {'results': [{'avatar_id': str(i), 'clip': {'frames_digest': 'f' * 64}} for i in range(48)]}
        proof = {'status': 'PASS_ALL48_MUXED_PIXEL_AND_HUMAN_PCM_INTEGRITY', 'report_sha256': media.REPORT_SHA,
                 'release_ready': False, 'cloud_mutations': False,
                 'poses': [{'avatar_id': str(i), 'frames_digest': 'f' * 64, 'original10s_pcm_sha256': 'a' * 64,
                            'all240_frame_hashes_match': True} for i in range(48)]}
        self.assertIs(media.validated_previous_proof(proof, data, 'a' * 64), proof)
        for key, value in [('frames_digest', 'b' * 64), ('original10s_pcm_sha256', 'b' * 64),
                           ('all240_frame_hashes_match', False)]:
            invalid = copy.deepcopy(proof)
            invalid['poses'][0][key] = value
            with self.assertRaises(ValueError):
                media.validated_previous_proof(invalid, data, 'a' * 64)
        with self.assertRaises(ValueError):
            media.validated_previous_proof({**proof, 'poses': proof['poses'][:-1]}, data, 'a' * 64)

    def test_attempt_does_not_rewrite_plan_or_existing_outputs(self):
        original = {'server_env': {'MUSETALK_RUNTIME_DIR': '/workspace/MuseTalk/experiments/test'},
                    'client_stage_argv': {'s0_n1': ['python', 'client', '--label', 's0_n1', '--levels', '1',
                                         '--out-dir', '/original/out', '--trace-dir', '/original/traces']}}
        before = json.dumps(original, sort_keys=True)
        argv = live.stage_argv(original, 's0_n1', 'h264_offer_v2')
        self.assertEqual(before, json.dumps(original, sort_keys=True))
        self.assertEqual(argv[argv.index('--label') + 1], 's0_n1_h264_offer_v2')
        self.assertTrue(argv[argv.index('--out-dir') + 1].endswith('/client_attempts/h264_offer_v2/s0_n1/lt2'))
        with self.assertRaises(ValueError):
            live.stage_argv(original, 's0_n1', '../unsafe')


if __name__ == '__main__':
    unittest.main()
