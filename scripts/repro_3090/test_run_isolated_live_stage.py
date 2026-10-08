"""CPU-only contracts for excluding tracing overhead from scored live runs."""
import tempfile
import unittest
from pathlib import Path

from run_isolated_live_stage import require_untraced_api, require_nvenc_level
from derive_live_codec_plan import derive, derive_nvenc
import json


class TracerGuardTests(unittest.TestCase):
    def fixture(self, root, tid, tracer):
        task = root / '123' / 'task' / str(tid)
        task.mkdir(parents=True)
        (task / 'status').write_text(f'Name:\tpython\nTracerPid:\t{tracer}\n')

    def test_all_threads_untraced(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, 123, 0)
            self.fixture(root, 124, 0)
            self.assertEqual(require_untraced_api(123, root)['threads_checked'], 2)

    def test_child_thread_traced_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, 123, 0)
            self.fixture(root, 124, 987)
            with self.assertRaises(ValueError):
                require_untraced_api(123, root)

    def test_missing_tracer_field_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.fixture(root, 123, 0)
            (root / '123/task/123/status').write_text('Name:\tpython\n')
            with self.assertRaises(ValueError):
                require_untraced_api(123, root)

    def test_codec_trial_changes_only_impl_and_output_root(self):
        root = Path(__file__).resolve().parents[2]
        parent = json.loads((root / 'docs/fps_comparisons/rtx3090_r5_20261008/native/isolated_live_v1_plan.json').read_text())
        result = derive(parent)
        changed = {k for k in parent['server_env'] if parent['server_env'][k] != result['server_env'][k]}
        self.assertEqual(changed, {'WEBRTC_H264_IMPL', 'MUSETALK_RUNTIME_DIR'})
        self.assertEqual(result['avatars'], parent['avatars'])
        self.assertEqual(result['audio'], parent['audio'])
        self.assertEqual(parent['server_env']['WEBRTC_H264_IMPL'], 'aiortc')
        for stage, args in result['client_stage_argv'].items():
            old = parent['server_env']['MUSETALK_RUNTIME_DIR']
            new = result['server_env']['MUSETALK_RUNTIME_DIR']
            self.assertEqual(args, [arg.replace(old, new) for arg in parent['client_stage_argv'][stage]])

    def test_nvenc_trial_changes_only_impl_and_output_root(self):
        root = Path(__file__).resolve().parents[2]
        parent = json.loads((root / 'docs/fps_comparisons/rtx3090_r5_20261008/native/isolated_live_x264tuned_v1_plan.json').read_text())
        result = derive_nvenc(parent)
        changed = {k for k in parent['server_env'] if parent['server_env'][k] != result['server_env'][k]}
        self.assertEqual(changed, {'WEBRTC_H264_IMPL', 'MUSETALK_RUNTIME_DIR'})
        self.assertEqual(result['avatars'], parent['avatars'])
        self.assertEqual(result['audio'], parent['audio'])
        self.assertEqual(parent['server_env']['WEBRTC_H264_IMPL'], 'x264tuned')
        old, new = parent['server_env']['MUSETALK_RUNTIME_DIR'], result['server_env']['MUSETALK_RUNTIME_DIR']
        self.assertEqual(result['server_argv'], [a.replace(old, new) if old in a else
                         ('WEBRTC_H264_IMPL=nvenc' if a == 'WEBRTC_H264_IMPL=x264tuned' else a)
                         for a in parent['server_argv']])
        for stage, args in result['client_stage_argv'].items():
            self.assertEqual(args, [a.replace(old, new) for a in parent['client_stage_argv'][stage]])

    def test_nvenc_missing_or_fallback_evidence_is_refused(self):
        row = {'server_encoder_after_level': {'h264': {'impl': 'nvenc', 'installed': True},
                'nvenc_sessions': {'peak': 3, 'opened': 4, 'denied': 0, 'open_failures': 0}}}
        require_nvenc_level(row, 3)
        with self.assertRaises(ValueError):
            require_nvenc_level(row, 5)
        row['server_encoder_after_level']['nvenc_sessions']['open_failures'] = 1
        with self.assertRaises(ValueError):
            require_nvenc_level(row, 3)
        with self.assertRaises(ValueError):
            require_nvenc_level({}, 1)


if __name__ == '__main__':
    unittest.main()
