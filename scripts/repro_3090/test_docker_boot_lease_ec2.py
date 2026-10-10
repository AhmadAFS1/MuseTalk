"""Synthetic Docker lease guards; no credentials, network, timer or rental."""
import copy
import unittest
import docker_boot_lease_ec2 as lease
import candidate_ci_input_bridge as bridge


class DockerLeaseTests(unittest.TestCase):
    def offer(self):
        return {'gpu_name': 'RTX 3090', 'num_gpus': 1, 'verification': 'verified',
                'disk_space': 100, 'cpu_cores_effective': 24, 'cpu_ram': 64000,
                'dph_total': .25, 'dph_total_adj': .29, 'storage_cost': .1,
                'inet_down_cost': .0015, 'inet_up_cost': .0015}

    def proof(self):
        return {'schema': 'musetalk_full_candidate_verification_v1', 'source_revision': lease.SOURCE,
                'image': lease.IMAGE + '@sha256:' + 'a' * 64, 'published': True, 'serving_image': True,
                'workflow_conclusion': 'success', 'independent_pull': 'PASS', 'offline_cpu_check': 'PASS',
                'private_visibility': 'VERIFIED', 'anonymous_pull': 'DENIED', 'promotion_eligible': False}

    def test_lease_caps_both_directions_and_adjusted_price(self):
        self.assertTrue(lease.qualifying(self.offer()))
        for field, value in (('inet_down_cost', .00151), ('inet_up_cost', .00151),
                             ('dph_total_adj', .31), ('dph_total', float('nan')),
                             ('storage_cost', -.1), ('inet_down_cost', None), ('num_gpus', 2)):
            offer = self.offer()
            offer[field] = value
            with self.subTest(field=field, value=value):
                self.assertFalse(lease.qualifying(offer))
        self.assertEqual(lease.filters()['allocated_storage'], lease.DISK)

    def test_dependency_or_unverified_image_cannot_rent(self):
        self.assertEqual(lease.validate_verification(self.proof()), self.proof()['image'])
        for field, value in (('serving_image', False), ('workflow_conclusion', 'failure'),
                             ('independent_pull', 'NOT_RUN'), ('offline_cpu_check', 'NOT_RUN'),
                             ('private_visibility', 'unknown'), ('promotion_eligible', True),
                             ('source_revision', 'b' * 40), ('image', lease.IMAGE + ':latest')):
            proof = self.proof()
            proof[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                lease.validate_verification(proof)

    def test_runtime_payload_drops_registry_callbacks_and_code_overrides(self):
        value = lease.runtime_env({'AWS_ACCESS_KEY_ID': 'synthetic-id', 'AWS_SECRET_ACCESS_KEY': 'synthetic-value',
            'AVATAR_S3_BUCKET': 'synthetic-bucket', 'GH_TOKEN': 'not-allowed',
            'LINGUA_WORKER_TOKEN': 'not-allowed', 'PATH': 'not-allowed',
            'MUSETALK_SERVER_LAUNCHER': 'not-allowed', 'MUSETALK_CANDIDATE_STANDALONE': '0'})
        for name in ('GH_TOKEN', 'LINGUA_WORKER_TOKEN', 'PATH', 'MUSETALK_SERVER_LAUNCHER'):
            self.assertNotIn(name, value)
        self.assertEqual(value['MUSETALK_CANDIDATE_STANDALONE'], '1')
        self.assertEqual(value['LINGUA_CONTROL_PLANE_ENABLED'], '0')
        with self.assertRaises(ValueError):
            lease.runtime_env({})

    def test_build_input_identity_requires_hash_size_version_encryption(self):
        item = copy.deepcopy(bridge.INPUTS['weights.tar.gz'])
        head = {'ContentLength': item['size_bytes'], 'Metadata': {'sha256': item['sha256']},
                'VersionId': item['version_id'], 'ServerSideEncryption': 'AES256'}
        bridge.verify_head(head, item)
        for field, value in (('ContentLength', 1), ('Metadata', {}), ('VersionId', 'other'),
                             ('ServerSideEncryption', None)):
            bad = {**head, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                bridge.verify_head(bad, item)


if __name__ == '__main__':
    unittest.main()
