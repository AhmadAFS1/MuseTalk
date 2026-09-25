"""Browser catalog must select source-bound packages and never hide failures."""
import ast
import asyncio
import json
import os
from pathlib import Path
import tempfile
import unittest
from typing import Optional
from unittest.mock import patch

from character_factory.scripts.build_realtime_character import make_pose_set
from scripts.motion_transitions import IDLE, TALK, SMILE, MotionBank, file_hash, publish_bank
from scripts.pose_protocol import normalize_pose_set
from scripts.webrtc_pose_lab_catalog import PoseLabCatalogError, pose_lab_context
from test_motion_transitions import fixture


class PoseLabCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.environment = patch.dict(os.environ, {
            'WEBRTC_MOTION_ATLAS': '', 'WEBRTC_MOTION_ATLAS_DIR': str(self.root),
            'WEBRTC_MOTION_ALLOW_UNREVIEWED': '1'})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def package(self, name):
        folder = self.root / name
        folder.mkdir()
        atlas = fixture()
        atlas['status'] = 'candidate_requires_recorded_review'
        names = {IDLE:'idle', TALK:'talking', SMILE:'smiling'}
        for pose, short in names.items():
            path = folder / (short + '.mp4')
            path.write_bytes((name + pose).encode())
            atlas['sources'][pose].update(path=str(path), sha256=file_hash(path))
        publish_bank(folder/'motion-atlas.json', atlas)
        pose_set = normalize_pose_set(make_pose_set(name, atlas))
        (folder/'session-pose-set.json').write_text(json.dumps(pose_set))
        data = {'character_id':name, 'motion_atlas':str(folder/'motion-atlas.json'),
                'pose_set':str(folder/'session-pose-set.json'),
                'source_hashes':{names[p]:atlas['sources'][p]['sha256'] for p in names},
                'motion_atlas_content_sha256':MotionBank(atlas).routing_sha256,
                'prepared':[{'avatar_id':pose_set['poses'][p]['avatar_id'], 'status':'ready'} for p in names]}
        (folder/'character.json').write_text(json.dumps(data))
        return folder, data

    def test_each_character_exposes_exact_routing_and_three_cache_identity(self):
        for name in ('first','second'):
            folder, _ = self.package(name)
            result = pose_lab_context(name)
            bank = MotionBank(json.loads((folder/'motion-atlas.json').read_text()))
            self.assertEqual(result['selected_character'], name)
            self.assertEqual(result['expected_motion']['routing_sha256'], bank.routing_sha256)
            self.assertEqual(result['expected_motion']['source_hashes'],
                             {p:s['sha256'] for p,s in bank.sources.items()})
            self.assertEqual(len(set(p['avatar_id'] for p in result['pose_set']['poses'].values())),3)
        self.assertEqual(len(pose_lab_context()['characters']),2)

    def test_no_registry_is_legacy_but_configured_empty_never_falls_back(self):
        with self.assertRaises(PoseLabCatalogError):
            pose_lab_context()
        with patch.dict(os.environ, {'WEBRTC_MOTION_ATLAS_DIR':''}):
            self.assertEqual(pose_lab_context(), {})
            with self.assertRaises(PoseLabCatalogError) as error:
                pose_lab_context('unknown')
            self.assertEqual(error.exception.status_code,404)

    def test_unknown_and_path_inputs_are_not_filesystem_lookups(self):
        self.package('valid')
        for value in ('../valid','/tmp/valid','valid/file','', '..'):
            with self.subTest(value=value), self.assertRaises(PoseLabCatalogError) as error:
                pose_lab_context(value)
            self.assertEqual(error.exception.status_code,400)
        with self.assertRaises(PoseLabCatalogError) as error:
            pose_lab_context('absent')
        self.assertEqual(error.exception.status_code,404)

    def test_changed_source_routing_or_wire_identity_cannot_open_as_legacy(self):
        folder, data = self.package('selected')
        path = folder/'session-pose-set.json'
        original = path.read_bytes()
        wire = json.loads(original)
        wire['poses'][TALK]['avatar_id'] = 'other_character'
        path.write_text(json.dumps(wire))
        with self.assertRaisesRegex(PoseLabCatalogError, 'physical caches'):
            pose_lab_context('selected')
        path.write_bytes(original)
        data['motion_atlas_content_sha256'] = '0'*64
        (folder/'character.json').write_text(json.dumps(data))
        with self.assertRaisesRegex(PoseLabCatalogError, 'routing'):
            pose_lab_context('selected')
        data['motion_atlas_content_sha256'] = MotionBank(json.loads((folder/'motion-atlas.json').read_text())).routing_sha256
        (folder/'character.json').write_text(json.dumps(data))
        (folder/'idle.mp4').write_bytes(b'changed')
        with self.assertRaisesRegex(PoseLabCatalogError, 'sources'):
            pose_lab_context('selected')

    def test_review_gate_preparation_gate_and_explicit_single_bank(self):
        folder, data = self.package('selected')
        with patch.dict(os.environ, {'WEBRTC_MOTION_ALLOW_UNREVIEWED':'0'}):
            with self.assertRaisesRegex(PoseLabCatalogError, 'review'):
                pose_lab_context('selected')
        with patch.dict(os.environ, {'WEBRTC_MOTION_ATLAS_DIR':'', 'WEBRTC_MOTION_ATLAS':str(folder/'motion-atlas.json')}):
            self.assertEqual(pose_lab_context()['selected_character'],'selected')
        data['prepared'] = []
        (folder/'character.json').write_text(json.dumps(data))
        with self.assertRaisesRegex(PoseLabCatalogError, 'preparation'):
            pose_lab_context('selected')

    def test_unrelated_incomplete_package_does_not_hide_working_character(self):
        folder, _ = self.package('a_broken')
        (folder/'character.json').write_text('{partial')
        self.package('b_working')
        self.assertEqual(pose_lab_context()['selected_character'],'b_working')
        with self.assertRaisesRegex(PoseLabCatalogError, 'missing or invalid'):
            pose_lab_context('a_broken')

    def test_malformed_preparation_is_isolated_and_nonidle_default_is_rejected(self):
        broken, data = self.package('a_broken')
        data['prepared'] = ['not-an-object']
        (broken/'character.json').write_text(json.dumps(data))
        good, _ = self.package('b_working')
        self.assertEqual(pose_lab_context()['selected_character'], 'b_working')
        with self.assertRaisesRegex(PoseLabCatalogError, 'list of objects'):
            pose_lab_context('a_broken')
        manifest = good/'session-pose-set.json'
        value = json.loads(manifest.read_text())
        value['default_pose_id'] = TALK
        manifest.write_text(json.dumps(value))
        with self.assertRaisesRegex(PoseLabCatalogError, 'neutral idle'):
            pose_lab_context('b_working')

    def test_route_resolves_in_worker_and_reports_selection_failure(self):
        # Exercise the real route body without importing GPU model initialization.
        from fastapi import HTTPException
        node = next(n for n in ast.parse(Path('api_server.py').read_text()).body
                    if isinstance(n, ast.AsyncFunctionDef) and n.name == 'webrtc_pose_lab')
        node.decorator_list=[]
        captured=[]
        scope={'Optional':Optional, 'asyncio':asyncio, 'HTTPException':HTTPException,
               '_require_webrtc':lambda:None, 'HTMLResponse':lambda **kwargs:kwargs,
               'get_webrtc_pose_lab_html':lambda **kwargs:captured.append(kwargs) or 'html'}
        exec(compile(ast.Module(body=[node],type_ignores=[]),'api_server.py','exec'),scope)
        self.package('selected')
        self.assertEqual(asyncio.run(scope['webrtc_pose_lab']('selected')), {'content':'html'})
        self.assertEqual(captured[0]['selected_character'],'selected')
        with self.assertRaises(HTTPException) as error:
            asyncio.run(scope['webrtc_pose_lab']('../selected'))
        self.assertEqual(error.exception.status_code,400)


if __name__ == '__main__':
    unittest.main()
