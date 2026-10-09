import contextlib
import io
import os
from pathlib import Path
import sys
import unittest
from unittest import mock

import inspect_a4_preparation_cpu as review


class Tests(unittest.TestCase):
    def argv(self, *extra):
        return ['review', '--comparison-sha256', 'a' * 64, '--bundle-manifest-sha256', 'b' * 64,
                '--out', str(review.BASE / 'a4_preparation_preprocessing_review_0449'), *extra]

    def test_default_off_before_heavy_imports(self):
        with mock.patch.object(sys, 'argv', self.argv()), mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            review.main()

    def test_visible_cuda_rejected_before_io(self):
        with mock.patch.object(sys, 'argv', self.argv('--execute')), mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '0'}), \
                mock.patch.object(review, 'sha') as digest, contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            review.main()
        digest.assert_not_called()

    def test_existing_output_rejected(self):
        with mock.patch.object(sys, 'argv', self.argv('--execute')), mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), \
                mock.patch.object(Path, 'exists', return_value=True), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            review.main()

    def test_comparison_hash_mismatch_before_decode(self):
        with mock.patch.object(sys, 'argv', self.argv('--execute')), mock.patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': ''}), \
                mock.patch.object(Path, 'exists', return_value=False), mock.patch.object(review, 'sha', return_value='changed'), \
                self.assertRaisesRegex(ValueError, 'comparison binding'):
            review.main()


if __name__ == '__main__':
    unittest.main()
