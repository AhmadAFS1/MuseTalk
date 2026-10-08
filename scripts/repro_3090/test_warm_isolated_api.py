import unittest

import warm_isolated_api as warm


class WarmContracts(unittest.TestCase):
    def test_ready_requires_local_residency_not_historical_ready(self):
        row = {'avatar_id': 'fixture', 'status': 'ready', 'cached': True,
               'disk_prepared': True, 's3_enabled': False, 's3_restore_required': False}
        warm.ready_cache(row, 'fixture')
        for key in ('cached', 'disk_prepared'):
            with self.assertRaises(ValueError):
                warm.ready_cache({**row, key: False}, 'fixture')
        with self.assertRaises(ValueError):
            warm.ready_cache(row, 'different')

    def test_cloud_restore_rejected(self):
        row = {'avatar_id': 'fixture', 'status': 'ready', 'cached': True,
               'disk_prepared': True, 's3_enabled': True, 's3_restore_required': False}
        with self.assertRaises(ValueError):
            warm.ready_cache(row, 'fixture')

    def test_control_plane_and_ready_identity(self):
        state = {'control_plane_requested': False, 'control_plane_enabled': False,
                 'registered': False, 'local_ready': True, 'internal_port': 8300,
                 'instance_id': ''}
        warm.isolated_state(state)
        for key in ('control_plane_requested', 'control_plane_enabled', 'registered'):
            with self.assertRaises(ValueError):
                warm.isolated_state({**state, key: True})
        with self.assertRaises(ValueError):
            warm.isolated_state({**state, 'internal_port': 8000})


if __name__ == '__main__':
    unittest.main()
