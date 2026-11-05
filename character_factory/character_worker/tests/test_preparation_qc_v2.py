"""Release/2 proves measured preprocessing, never a fabricated teacher/student render."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import test_device_release as fixtures
release, resume, pipeline = fixtures.release, fixtures.resume, fixtures.pipeline


class PreparationReleaseTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DeviceReleaseTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)

    def spec(self):
        spec = self.fixture.spec()
        spec.update(release_schema=2, logical_job_id='logical-1',
                    qc={'pass': True, 'endpoint_set': {'pass': True},
                        'reports': {pose: {'pass': True, 'frames': 1} for pose in release.POSES}})
        spec.pop('smoke')
        return spec

    def test_manifest_last_measured_qc_without_any_teacher_claim(self):
        spec, s3 = self.spec(), fixtures.S3()
        decoded = []
        ref = release.publish(spec, s3, self.fixture.exporter, image_decoder=decoded.append)
        manifest = json.loads(s3.rows[ref['key']]['Body'])
        qc = json.loads(s3.rows[manifest['qc']['report']['key']]['Body'])
        provenance = json.loads(s3.rows[manifest['provenance']['key']]['Body'])
        self.assertEqual(s3.order[-1], ref['key'])
        self.assertEqual(manifest['schema'], 'lingua-character-release/2')
        self.assertEqual(manifest['logical_job_id'], 'logical-1')
        self.assertEqual(len(decoded), 3)
        self.assertEqual(qc['schema'], 'lingua-preprocessing-qc/1')
        self.assertEqual(qc['checks'], {name: 'passed' for name in release.QC_CHECKS})
        self.assertEqual(qc['phone_student_acceptance'], 'not_run_on_linux')
        self.assertNotIn('smoke', qc)
        self.assertNotIn('asset_smoke', provenance)
        self.assertFalse(provenance['teacher_render_required'])
        for pose in release.POSES:
            self.assertEqual(manifest['poses'][pose]['source']['frame_count'], 1)
            self.assertEqual(qc['poses'][pose]['images_decoded'], 2)
            self.assertEqual(qc['poses'][pose]['bundle_sha256'], manifest['poses'][pose]['bundle']['sha256'])
        self.assertEqual(self.fixture.publish(spec, s3, self.fixture.exporter), ref)

    def test_each_required_source_report_and_endpoint_gate_is_fail_closed(self):
        for defect in ('missing_pose', 'failed_pose', 'endpoint', 'top_level', 'frame_count'):
            with self.subTest(defect=defect):
                spec, s3 = self.spec(), fixtures.S3()
                if defect == 'missing_pose': spec['qc']['reports'].pop('smiling')
                elif defect == 'failed_pose': spec['qc']['reports']['talking']['pass'] = False
                elif defect == 'endpoint': spec['qc']['endpoint_set']['pass'] = False
                elif defect == 'top_level': spec['qc']['pass'] = False
                else: spec['qc']['reports']['idle']['frames'] = 2
                with self.assertRaises(ValueError):
                    self.fixture.publish(spec, s3, self.fixture.exporter)
                self.assertFalse(any(key.endswith('/release.json') for key in s3.rows))

    def test_archive_image_decode_failure_cannot_publish_passed_qc(self):
        s3 = fixtures.S3()
        with self.assertRaisesRegex(ValueError, 'injected decoder failure'):
            release.publish(self.spec(), s3, self.fixture.exporter,
                            image_decoder=Mock(side_effect=ValueError('injected decoder failure')))
        self.assertFalse(any(key.endswith(('/release.json', '/qc/report.json')) for key in s3.rows))

    def test_new_attempt_can_restore_same_logical_job_and_cannot_restore_another(self):
        s3, original, spec, planner = self.fixture.checkpoint_fixture()
        spec['descriptor']['release_id'] = original['identity']['release_id']
        spec['identity'] = {**spec['identity'], 'job_id': 'retry-2', 'release_id': 'retry-2'}
        spec['logical_job_id'] = original['identity']['job_id']
        result = self.fixture.restore(spec, s3, planner)
        self.assertEqual(result['seed'], 42)
        for wrong in (None, 'other-logical-job'):
            spec['logical_job_id'] = wrong
            with self.assertRaisesRegex(ValueError, 'logical job mismatch'):
                self.fixture.restore(spec, s3, planner)

    def test_same_attempt_cannot_contradict_checkpoint_logical_identity(self):
        s3, _original, spec, planner = self.fixture.checkpoint_fixture()
        spec['logical_job_id'] = 'other-logical-job'
        with self.assertRaisesRegex(ValueError, 'logical job mismatch'):
            self.fixture.restore(spec, s3, planner)

    def test_profile_only_h3_revision_change_needs_explicit_compatibility(self):
        s3, _original, spec, planner = self.fixture.checkpoint_fixture()
        spec['h3_commit'] = 'new-fl2va-profile'
        with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
            self.fixture.restore(spec, s3, planner)
        spec['h3_compatible_commits'] = ['h3-pinned']
        self.assertEqual(self.fixture.restore(spec, s3, planner)['seed'], 42)


class GenerationBudgetTest(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.DeviceReleaseTest()
        self.fixture.setUp()
        self.addCleanup(self.fixture.tearDown)
        self.root = self.fixture.root
        self.portrait = self.root / 'portrait.png'
        self.portrait.write_bytes(fixtures.png())
        self.ident = self.root / 'accepted_sources'
        for pose in release.POSES:
            (self.ident / pose).mkdir(parents=True)
            (self.ident / pose / 'source.mp4').write_bytes(b'accepted-source')
            (self.ident / pose / 'h3.json').write_text('{"generation_seconds":1}')
        self.rt = Mock(root=self.root, gpu_lock=self.root / 'gpu.lock')
        self.rt.preflight.return_value = []
        self.job = Mock(id='worker-job', dir=self.root / 'job', data={})
        self.job.dir.mkdir()

    def run_pipeline(self, options, restored=None, qc_pass=False):
        self.job.request = {'character_id': 'person', 'options': {**options, 'release_local': False, 'publish_records': False}}
        with patch.object(pipeline, 'host_facts', return_value={}), \
             patch.object(pipeline, 'ingest_portrait', return_value=self.portrait), \
             patch.object(pipeline, 'generate', return_value=self.ident) as generate, \
             patch.object(pipeline, 'restore_device_sources', return_value=restored), \
             patch.object(pipeline, 'qc_poses', return_value={'pass': qc_pass}), \
             patch.object(pipeline, 'callback'), \
             patch.object(pipeline, 'MuseTalkServer') as server, \
             patch.object(pipeline, 'prepare', return_value={'poses': {}}), \
             patch.object(pipeline, 'verify_s3', return_value={}):
            server.return_value.start.return_value = {}
            result = pipeline.run_job(self.rt, self.job)
        return result, generate, server

    def test_exhausted_generation_budget_never_calls_h3(self):
        result, generate, server = self.run_pipeline({'max_attempts': 0})
        self.assertEqual(result['status'], 'qc_failed')
        self.assertEqual(result['generation_attempts_used'], 0)
        generate.assert_not_called()
        server.assert_not_called()

    def test_remaining_attempts_are_not_reset_after_qc_failure(self):
        result, generate, server = self.run_pipeline({'max_attempts': 1})
        self.assertEqual(result['status'], 'qc_failed')
        self.assertEqual(result['generation_attempts_used'], 1)
        generate.assert_called_once()
        server.assert_not_called()

    def test_restored_sources_complete_preparation_with_zero_generation_budget(self):
        restored = {'seed': 42, 'source_checkpoint': {'key': 'accepted'}, 'identity_dir': str(self.ident)}
        result, generate, server = self.run_pipeline({'max_attempts': 0, 'source_checkpoint': {'key': 'accepted'}},
                                                    restored=restored, qc_pass=True)
        self.assertEqual(result['status'], 'done', result.get('error'))
        self.assertEqual(result['generation_attempts_used'], 0)
        generate.assert_not_called()
        server.assert_called_once_with(self.rt, self.job)
        self.assertGreaterEqual(server.return_value.stop.call_count, 1)


if __name__ == '__main__':
    unittest.main()
