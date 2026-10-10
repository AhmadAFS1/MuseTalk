import unittest
from unittest.mock import Mock, patch
import urllib.error

import run_docker_boot_sequence as sequence


class BootSequenceTests(unittest.TestCase):
    def test_only_owned_public_mapped_endpoint_allowed(self):
        row = {'public_ipaddr': '8.8.8.8', 'ports': {'8000/tcp': [{'HostPort': '12345'}]}}
        self.assertEqual(sequence.endpoint(row), 'http://8.8.8.8:12345')
        self.assertIsNone(sequence.endpoint({**row, 'ports': {}}))
        for ip in ('127.0.0.1', '169.254.169.254', '::1', '192.168.0.1'):
            with self.subTest(ip=ip), self.assertRaises(ValueError):
                sequence.endpoint({**row, 'public_ipaddr': ip})

    def test_http_errors_omit_body_and_do_not_retry(self):
        opener = Mock()
        opener.open.side_effect = urllib.error.HTTPError('http://8.8.8.8:12345/health', 500, 'synthetic-secret', {}, None)
        with patch.object(sequence.urllib.request, 'build_opener', return_value=opener):
            self.assertEqual(sequence.request('http://8.8.8.8:12345', '/health'), (500, None))
        self.assertEqual(opener.open.call_count, 1)

    def test_redirects_rejected(self):
        with self.assertRaises(ValueError):
            sequence.NoRedirect().redirect_request(None, None, None, None, None, 'http://other.example')


if __name__ == '__main__':
    unittest.main()
