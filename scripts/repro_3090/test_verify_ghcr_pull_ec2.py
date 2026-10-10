import hashlib
import io
import json
import unittest
import verify_ghcr_pull_ec2 as pull


class Response(io.BytesIO):
    status = 200

    def __init__(self, raw, headers=None):
        super().__init__(raw)
        self.headers = headers or {}


class DeploymentPullTests(unittest.TestCase):
    def fixture(self, *, scopes='read:packages', visibility='private', corrupt=False):
        manifest = json.dumps({'schemaVersion': 2, 'mediaType': pull.ACCEPT.split(', ')[0],
                               'layers': [{'size': 123}]}).encode()
        digest = 'sha256:' + hashlib.sha256(manifest).hexdigest()
        responses = iter([Response(json.dumps({'name': 'musetalk-rtx3090', 'visibility': visibility}).encode(),
                                 {'X-OAuth-Scopes': scopes}), Response(b'{"token":"synthetic-registry-token"}'),
                          Response(manifest + (b' ' if corrupt else b''), {'Docker-Content-Digest': digest})])
        return pull.IMAGE + '@' + digest, lambda *a, **k: next(responses)

    def test_exact_private_manifest_with_deployment_read_only_token(self):
        image, opener = self.fixture()
        result = pull.verify(image, 'AhmadAFS1', 'ghp_' + 's' * 40, opener)
        self.assertEqual(result['status'], 'PASS')
        self.assertEqual(result['compressed_layer_bytes'], 123)
        self.assertFalse(result['gpu_rental'])
        self.assertNotIn('ghp_', json.dumps(result))

    def test_broad_scope_public_package_or_changed_digest_is_rejected(self):
        for options in ({'scopes': 'read:packages, write:packages'}, {'visibility': 'public'}, {'corrupt': True}):
            image, opener = self.fixture(**options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                pull.verify(image, 'AhmadAFS1', 'ghp_' + 's' * 40, opener)

    def test_tag_foreign_image_invalid_token_or_user_never_reaches_network(self):
        image, _ = self.fixture()
        for reference, user, token in ((pull.IMAGE + ':latest', 'AhmadAFS1', 'ghp_' + 's' * 40),
                                      ('ghcr.io/other@sha256:' + 'a' * 64, 'AhmadAFS1', 'ghp_' + 's' * 40),
                                      (image, 'other', 'ghp_' + 's' * 40), (image, 'AhmadAFS1', 'invalid')):
            with self.assertRaises(ValueError):
                pull.verify(reference, user, token, lambda *a, **k: self.fail('network must not be called'))


if __name__ == '__main__':
    unittest.main()
