import tempfile
import unittest
from pathlib import Path
from probe_traced_avatar_prepare import verify_artifacts


class AvatarArtifactsTests(unittest.TestCase):
    def make_cache(self, target, frames=480):
        for name in ('latents.pt', 'coords.pkl', 'mask_coords.pkl', 'avator_info.json'):
            (target / name).write_bytes(b'fixture')
        for kind in ('full_imgs', 'mask'):
            (target / kind).mkdir()
            for index in range(frames):
                (target / kind / f'{index:08d}.png').write_bytes(b'fixture')

    def test_canonical_historical_metadata_spelling(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            self.make_cache(target)
            result = verify_artifacts(target)
            self.assertEqual(result['frame_pngs'], 480)
            self.assertIn('avator_info.json', [row['name'] for row in result['artifacts']])
            (target / 'avator_info.json').rename(target / 'avatar_info.json')
            with self.assertRaises(AssertionError):
                verify_artifacts(target)

    def test_short_cycle_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            self.make_cache(target, 240)
            with self.assertRaises(AssertionError):
                verify_artifacts(target)

    def test_link_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)
            self.make_cache(target)
            (target / 'latents.pt').rename(target / 'real.pt')
            (target / 'latents.pt').symlink_to(target / 'real.pt')
            with self.assertRaises(AssertionError):
                verify_artifacts(target)


if __name__ == '__main__':
    unittest.main()
