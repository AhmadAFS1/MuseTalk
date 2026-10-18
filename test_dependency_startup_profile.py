"""Stdlib-only tests: no pip installation, network requests or CUDA on macOS."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('profile', Path(__file__).parent/'scripts/dependency_startup_profile.py')
p = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(p)


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        p.OUT = Path(self.tmp.name)/'events.jsonl'
        p.RUN = 'test'

    def records(self):
        return [json.loads(s) for s in p.OUT.read_text().splitlines()]

    def test_urls_do_not_preserve_auth_path_query_fragment(self):
        value = p.safe_url('https://user:password@example.com/secret/token/torch-2.whl?X-Amz-Signature=private#secret')
        self.assertEqual(value, {'host':'example.com','file':'torch-2.whl'})

    def test_invalid_filename_redacted(self):
        self.assertEqual(p.safe_url('https://example.com/a%2Fb?token=foo')['file'], '[path-redacted]')

    def test_index_package_is_visible_without_index_credentials(self):
        self.assertEqual(p.safe_url('https://name:pass@example.com/simple/torch/?secret=x'),
                         {'host':'example.com','file':'torch'})

    def test_return_value_and_arguments_unchanged(self):
        fn = p.span_wrapper(lambda a, b=0: a+b, 'example')
        self.assertEqual(fn(3,b=5), 8)
        records = self.records()
        self.assertEqual([r['kind'] for r in records], ['example_start','example_end'])
        self.assertEqual(records[0]['span_id'], records[1]['span_id'])
        self.assertEqual(records[1]['outcome'], 'ok')

    def test_error_preserved_without_error_message(self):
        def bad():
            raise ValueError('super-secret-signed-url')
        with self.assertRaisesRegex(ValueError, 'super-secret'):
            p.span_wrapper(bad, 'bad')()
        self.assertNotIn('super-secret', p.OUT.read_text())
        self.assertEqual(self.records()[-1]['outcome'], 'ValueError')

    def test_logging_failure_does_not_fail_operation(self):
        p.OUT = Path(self.tmp.name)/'absent'/'x'
        self.assertEqual(p.span_wrapper(lambda:7, 'ok')(),7)

    def test_generator_is_consumed_and_failures_preserved(self):
        def values():
            yield 9
            raise RuntimeError('private')
        it = p.span_wrapper(values, 'gen', generator=True)()
        self.assertEqual(next(it),9)
        with self.assertRaises(RuntimeError):
            next(it)
        self.assertEqual(self.records()[-1]['outcome'], 'RuntimeError')

    def test_optional_hook_failure_is_recorded(self):
        p.patch('definitely_not_an_installed_module', 'missing', 'x')
        self.assertFalse(self.records()[0]['available'])

    def test_response_status_is_measured(self):
        class Response:
            status_code=404
            from_cache=False
        value=Response()
        self.assertIs(p.span_wrapper(lambda:value,'request')(),value)
        self.assertEqual(self.records()[-1]['status'],404)


class AnalysisTests(unittest.TestCase):
    def test_overlapping_spans_are_not_added_twice(self):
        spec=importlib.util.spec_from_file_location('analyze',Path(__file__).parent/'scripts/dependency_profile_analyze.py')
        a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
        self.assertEqual(a.union_seconds([(0,5),(3,8),(10,12),(4,6)]),10)


if __name__ == '__main__':
    unittest.main()
