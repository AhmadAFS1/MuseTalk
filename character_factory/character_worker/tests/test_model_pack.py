"""Shared model packs must select the declared plain student or chunked teacher."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts/ios'))
from pack_model_pack import select_models


class ModelPackTest(unittest.TestCase):
    def setUp(self):
        self.models = [{'name': 'student', 'kind': 'unet', 'file': 'student.mlmodelc'},
                       {'name': 'teacher_chunked', 'kind': 'unet'},
                       {'name': 'teacher_chunk1', 'kind': 'unet_chunk', 'chunk_of': 'teacher'},
                       {'name': 'teacher_chunk2', 'kind': 'unet_chunk', 'chunk_of': 'teacher'},
                       {'name': 'taesd', 'kind': 'taesd', 'batch': 1},
                       {'name': 'logmel', 'kind': 'whisper_logmel'},
                       {'name': 'encoder', 'kind': 'whisper_encoder'}]

    def test_student_pack_contains_no_teacher_chunks(self):
        selected = select_models(self.models, 'student')
        self.assertEqual([m['name'] for m in selected], ['student', 'taesd', 'logmel', 'encoder'])

    def test_existing_teacher_chunk_profile_remains_supported(self):
        selected = select_models(self.models, 'teacher')
        self.assertEqual([m['name'] for m in selected if m['kind'].startswith('unet')],
                         ['teacher_chunked', 'teacher_chunk1', 'teacher_chunk2'])

    def test_missing_unet_never_publishes_decoder_only_pack(self):
        with self.assertRaises(ValueError):
            select_models(self.models, 'missing')


if __name__ == '__main__':
    unittest.main()
