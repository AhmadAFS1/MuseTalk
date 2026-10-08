import copy
import unittest

import persist_native48_media_operator as media
import reserve_media_transfer_budget as budget


class OperatorContracts(unittest.TestCase):
    def test_private_head_requires_hash_size_encryption_version(self):
        row = {'bytes': 10, 'sha256': 'a' * 64}
        head = {'ContentLength': 10, 'Metadata': {'sha256': 'a' * 64},
                'ServerSideEncryption': 'AES256', 'VersionId': 'version'}
        media.validate_head(head, row)
        for key, value in [('ContentLength', 9), ('Metadata', {'sha256': 'b' * 64}),
                           ('ServerSideEncryption', 'none'), ('VersionId', None)]:
            with self.assertRaises(ValueError):
                media.validate_head({**head, key: value}, row)

    def test_reservation_stays_inside_original_authorization(self):
        data = {'schema': 'musetalk_experiment_budget_v1', 'total_usd': 30,
                'reservations': {'musetalk-r5-3090-dev-20261008': {'reserved_usd': 5.5847395833333335},
                                 'musetalk-r5-20261008-additional-aws-audit': {'reserved_usd': 2}}}
        before = copy.deepcopy(data)
        self.assertEqual(budget.reserve(data), (7.5847395833333335, 9.584739583333334))
        self.assertFalse(data['reservations'][budget.RESERVATION]['new_rental_authorized'])
        for invalid in ({**before, 'total_usd': 31}, {**before, 'reservations': {}}, data):
            with self.assertRaises(ValueError):
                budget.reserve(copy.deepcopy(invalid))


if __name__ == '__main__':
    unittest.main()
