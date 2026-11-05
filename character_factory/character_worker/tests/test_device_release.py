"""Publication must never advertise a partial or corrupt character release."""
import base64
import hashlib
import io
import json
from pathlib import Path
import struct
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
import zlib

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'tools'))
import character_api
import pipeline
import publish_device_release as release
import restore_device_sources as resume


def png(width=8, height=8):
    def chunk(kind, body):
        data = kind + body
        return struct.pack('>I', len(body)) + data + struct.pack('>I', zlib.crc32(data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, 8, 0, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress((b'\x00' + bytes(width)) * height)) + chunk(b'IEND', b''))


def bundle(out, avatar_id, mutation=None, extra=None):
    out.mkdir(parents=True, exist_ok=True)
    meta = {'avatar_id': avatar_id, 'unique_frames': 1, 'cycle_length': 2, 'ping_pong': True, 'fps': 24,
            'frame_width': 8, 'frame_height': 8, 'mask_width': 8, 'mask_height': 8,
            'frames': ['frames/0000.png'], 'masks': ['masks/0000.png'], 'coords': [[0, 0, 8, 8]],
            'mask_coords': [[0, 0, 8, 8]],
            'latents': {'file': 'latents_f16.bin', 'dtype': 'float16', 'shape': [1, 8, 32, 32]}}
    files = {'avatar.json': json.dumps(meta).encode(), 'latents_f16.bin': bytes(8 * 32 * 32 * 2),
             'frames/0000.png': png(), 'masks/0000.png': png()}
    if mutation:
        mutation(meta, files)
        files['avatar.json'] = json.dumps(meta).encode()
    path = out / f'{avatar_id}.tar'
    with tarfile.open(path, 'w', format=tarfile.USTAR_FORMAT) as archive:
        for name, data in files.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
        if extra:
            archive.addfile(extra)
    sidecar = {'avatar_id': avatar_id, 'bytes': path.stat().st_size, 'sha256': release.file_hash(path),
               'frames': 1, 'cycle_length': 2, 'fps': 24, 'width': 8, 'height': 8,
               'format': 'phone-avatar-bundle/1', 'exporter_commit': 'a' * 40}
    sidecar_path = out / f'{avatar_id}.json'
    sidecar_path.write_text(json.dumps(sidecar))
    return path, sidecar_path


class Conflict(Exception):
    response = {'Error': {'Code': 'PreconditionFailed'}}


class S3:
    def __init__(self, fail=None):
        self.rows = {}
        self.order = []
        self.fail = fail

    def put_object(self, **kwargs):
        key = kwargs['Key']
        if self.fail and self.fail in key:
            raise RuntimeError('injected transport failure')
        self.order.append(key)
        if key in self.rows:
            raise Conflict()
        body = kwargs['Body'].read()
        if kwargs['ChecksumSHA256'] != base64.b64encode(hashlib.sha256(body).digest()).decode():
            raise ValueError('bad checksum')
        self.rows[key] = dict(ContentLength=len(body), Metadata=kwargs['Metadata'],
                              ChecksumSHA256=kwargs['ChecksumSHA256'], Body=body)

    def head_object(self, **kwargs):
        return self.rows[kwargs['Key']]

    def get_object(self, **kwargs):
        return {'Body': io.BytesIO(self.rows[kwargs['Key']]['Body'])}


class DeviceReleaseTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # Dependency-free protocol tests; real publication invokes Pillow to decode all images.
        # Default arguments bind the function at import time, so pass a wrapped publish instead.
        self.publish = lambda *a, **kw: release.publish(*a, image_decoder=lambda tar: None, **kw)

    def tearDown(self):
        self.tmp.cleanup()

    def spec(self):
        source = self.root / 'source.mp4'
        source.write_bytes(b'already-QC-verified-source')
        portrait = self.root / 'portrait.png'
        portrait.write_bytes(png())
        return {'identity': {'character_id': 'lingua-character', 'job_id': 'charfactory_1', 'release_id': 'charfactory_1'},
                'worker_job_id': 'remote_1', 'repo': str(HERE.parents[1]), 'out': str(self.root / 'out'),
                'bucket': 'private-bucket', 'recipe_id': 'h3-recipe', 'qc': {'pass': True}, 'smoke': {'pass': True},
                'portrait': str(portrait), 'anchor': str(portrait), 'provenance': {'seed': 42},
                'poses': {p: {'avatar_id': 'new_' + p, 'cache': str(self.root / p), 'source': str(source),
                              'source_sha256': release.file_hash(source)} for p in release.POSES}}

    @staticmethod
    def exporter(repo, cache, avatar_id, out):
        return bundle(out, avatar_id)

    def test_complete_manifest_uploaded_last_and_retry_is_idempotent(self):
        s3 = S3()
        spec = self.spec()
        ref = self.publish(spec, s3, self.exporter)
        self.assertEqual(s3.order[-1], ref['key'])
        manifest = json.loads(s3.rows[ref['key']]['Body'])
        self.assertEqual(manifest['schema'], 'lingua-character-release/1')
        self.assertEqual(manifest['character_id'], spec['identity']['character_id'])
        self.assertEqual(set(manifest['poses']), set(release.POSES))
        for row in manifest['poses'].values():
            self.assertIn(row['bundle']['key'], s3.rows)
            self.assertIn(row['source']['key'], s3.rows)
        self.assertIn(manifest['qc']['report']['key'], s3.rows)
        self.assertIn(manifest['provenance']['key'], s3.rows)
        self.assertEqual(self.publish(spec, s3, self.exporter), ref)

    def test_failed_required_upload_never_commits_manifest(self):
        s3 = S3(fail='/sources/talking-')
        with self.assertRaisesRegex(RuntimeError, 'transport failure'):
            self.publish(self.spec(), s3, self.exporter)
        self.assertFalse(any(k.endswith('/release.json') for k in s3.rows))

    def test_corrupt_bundle_never_commits_manifest(self):
        def bad(repo, cache, avatar_id, out):
            return bundle(out, avatar_id, lambda meta, files: files.update({'latents_f16.bin': b'bad'}))
        s3 = S3()
        with self.assertRaisesRegex(ValueError, 'latent byte length'):
            self.publish(self.spec(), s3, bad)
        self.assertFalse(s3.rows)

    def test_source_hash_and_smoke_required(self):
        for change in ('source', 'smoke'):
            spec = self.spec()
            if change == 'source':
                spec['poses']['idle']['source_sha256'] = '0' * 64
            else:
                spec['smoke'] = {'pass': False}
            with self.assertRaises(ValueError):
                self.publish(spec, S3(), self.exporter)

    def test_cancel_before_commit_never_publishes_release(self):
        s3 = S3()
        calls = []
        def cancel():
            calls.append(True)
            if len(calls) == 8:
                raise pipeline.JobCancelled()
        with self.assertRaises(pipeline.JobCancelled):
            self.publish(self.spec(), s3, self.exporter, cancel)
        self.assertFalse(any(k.endswith('/release.json') for k in s3.rows))

    def test_validator_rejects_nonfinite_geometry_missing_file_and_symlink(self):
        cases = [lambda m, f: f.update({'latents_f16.bin': struct.pack('<e', float('nan')) + bytes(16382)}),
                 lambda m, f: m.update({'coords': [[-1, 0, 7, 8]]}),
                 lambda m, f: f.pop('masks/0000.png')]
        for mutate in cases:
            tar, sidecar = bundle(self.root, 'avatar', mutate)
            with self.assertRaises(ValueError):
                release.validate(tar, json.loads(sidecar.read_text()))
        info = tarfile.TarInfo('frames/evil')
        info.type, info.linkname = tarfile.SYMTYPE, '/etc/passwd'
        tar, sidecar = bundle(self.root, 'avatar', extra=info)
        with self.assertRaisesRegex(ValueError, 'unsafe'):
            release.validate(tar, json.loads(sidecar.read_text()))

    def test_api_rejects_partial_release_and_unsafe_identity(self):
        body = {'character_id': 'new_character', 'portrait': {'url': 'https://example.com/p.png'},
                'options': {'device_release': {'character_id': 'abc', 'job_id': 'job_1', 'release_id': 'job_1'}}}
        req, _ = character_api.validate_request(body)
        self.assertEqual(req['options']['device_release']['job_id'], 'job_1')
        for extra in ({'poses': ['talking']}, {'options': {'device_release': {'character_id': '../x'}}},
                      {'options': {**body['options'], 'smoke_render': False}}):
            with self.assertRaises(character_api.BadRequest):
                character_api.validate_request({**body, **extra})

    def test_source_checkpoint_survives_without_final_release(self):
        s3 = S3()
        spec = self.spec()
        ref = release.checkpoint(spec, s3)
        self.assertTrue(ref['key'].endswith('/source-checkpoint.json'))
        receipt = json.loads(s3.rows[ref['key']]['Body'])
        self.assertEqual(receipt['schema'], 'lingua-character-source-checkpoint/1')
        self.assertEqual(set(receipt['poses']), set(release.POSES))
        self.assertEqual(receipt['qc']['status'], 'passed')
        self.assertFalse(any(k.endswith('/release.json') for k in s3.rows))
        self.assertEqual(release.checkpoint(spec, s3), ref)

    def test_source_checkpoint_upload_failure_never_commits_receipt(self):
        s3 = S3(fail='/sources/smiling-')
        with self.assertRaises(RuntimeError):
            release.checkpoint(self.spec(), s3)
        self.assertFalse(any(k.endswith('/source-checkpoint.json') for k in s3.rows))

    def checkpoint_fixture(self):
        s3 = S3()
        spec = self.spec()
        spec['provenance'] = {'portrait_sha256': release.file_hash(Path(spec['portrait'])),
                              'lock': {'h3_workflows': {'commit': 'h3-pinned'}},
                              'plan': {'signature': 'fixed-inputs'},
                              'h3': {p: {'status': 'complete', 'seed': 42, 'generation_seconds': 1} for p in release.POSES}}
        ref = release.checkpoint(spec, s3)
        target = self.root / 'new-worker' / 'new_identity'
        restore_spec = {'identity': spec['identity'], 'descriptor': ref, 'bucket': spec['bucket'],
                        'recipe_id': spec['recipe_id'], 'portrait': spec['portrait'], 'h3_commit': 'h3-pinned',
                        'out': str(target.parent)}
        planner = lambda spec, seed: {'signature': 'fixed-inputs', 'output': str(target)}
        return s3, spec, restore_spec, planner

    def restore(self, spec, s3, planner):
        return resume.restore(spec, s3, planner=planner, anchor_validator=lambda s, a: None,
                              video_validator=lambda s, v, p: {'first_last_exact': True, 'first_rgb_sha256': 'shared-anchor'})

    def test_resume_from_previous_worker_checks_hashes_and_remaps_artifacts(self):
        s3, original, spec, planner = self.checkpoint_fixture()
        result = self.restore(spec, s3, planner)
        self.assertEqual(result['seed'], 42)
        for pose in release.POSES:
            target = Path(result['identity_dir']) / pose
            self.assertEqual(release.file_hash(target / 'source.mp4'), original['poses'][pose]['source_sha256'])
            record = json.loads((target / 'h3.json').read_text())
            self.assertEqual(record['artifacts'][str((target / 'source.mp4').resolve())], original['poses'][pose]['source_sha256'])

    def test_resume_refuses_foreign_bucket_job_recipe_inputs_and_corrupt_object(self):
        for kind in ('bucket', 'job', 'recipe', 'inputs', 'object'):
            with self.subTest(kind=kind):
                s3, original, spec, planner = self.checkpoint_fixture()
                if kind == 'bucket':
                    spec['descriptor']['bucket'] = 'foreign'
                elif kind == 'job':
                    spec['identity'] = {**spec['identity'], 'job_id': 'new-job'}
                elif kind == 'recipe':
                    spec['recipe_id'] = 'new-recipe'
                elif kind == 'inputs':
                    planner = lambda s, seed: {'signature': 'changed-prompt', 'output': str(self.root / 'new-target')}
                else:
                    key = next(k for k in s3.rows if '/sources/idle-' in k)
                    s3.rows[key]['Body'] = b'changed source'
                with self.assertRaises(ValueError):
                    self.restore(spec, s3, planner)

    def test_driver_resumes_then_reruns_qc_without_h3_generation(self):
        s3, original, spec, planner = self.checkpoint_fixture()
        restored = self.restore(spec, s3, planner)
        ident = Path(restored['identity_dir'])
        rt = pipeline.Runtime(workspace=self.root / 'workspace', root=self.root / 'worker')
        rt.workspace.mkdir(parents=True)
        job_dir = self.root / 'worker/jobs/new_worker_job'
        job_dir.mkdir(parents=True)
        pipeline.write_json(job_dir / 'job.json', {'job_id': 'new_worker_job', 'request': {
            'character_id': 'new_person', 'options': {'source_checkpoint': spec['descriptor'],
                'device_release': spec['identity'], 'release_local': False, 'publish_records': False}}})
        job = pipeline.Job(job_dir)
        published = {'poses': {p: {'avatar_id': 'new_' + p, 'source_sha256': original['poses'][p]['source_sha256']}
                               for p in release.POSES}}
        with (patch.object(rt, 'preflight', return_value=[]), patch.object(pipeline, 'host_facts', return_value={}),
              patch.object(pipeline, 'ingest_portrait', return_value=Path(spec['portrait'])),
              patch.object(pipeline, 'restore_device_sources', return_value=restored),
              patch.object(pipeline, 'generate') as generate,
              patch.object(pipeline, 'qc_poses', return_value={'pass': True}) as qc,
              patch.object(pipeline, 'MuseTalkServer') as server,
              patch.object(pipeline, 'prepare', return_value=published),
              patch.object(pipeline, 'smoke_render', return_value={'pass': True}),
              patch.object(pipeline, 'verify_s3', return_value={'pass': True}),
              patch.object(pipeline, 'publish_device_release', return_value={'key': 'verified-release'})):
            server.return_value.start.return_value = {'bucket': 'b', 'prefix': 'avatars', 'version': 'v15'}
            result = pipeline.run_job(rt, job)
        self.assertEqual(result['status'], 'done', result.get('error'))
        generate.assert_not_called()
        qc.assert_called_once()
        self.assertEqual(result['resumed_from'], spec['descriptor'])


if __name__ == '__main__':
    unittest.main()
