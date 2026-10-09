"""Synthetic CPU provenance tests; no real TensorRT or image-quality inference."""
import copy
import hashlib
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_assemble_owned_unet_tail_a3 as fixtures
import unet_plan_overlay as overlay


class Tests(unittest.TestCase):
    write = fixtures.OverlayTests.write

    def setUp(self):
        fixtures.OverlayTests.setUp(self)

    def prepare(self, variant='portable_prefix', **kwargs):
        return overlay.prepare(self.native, self.portable, self.root / 'candidate', variant=variant,
            native_sha=hashlib.sha256((self.native / 'manifest.json').read_bytes()).hexdigest(),
            portable_sha=hashlib.sha256((self.portable / 'manifest.json').read_bytes()).hexdigest(), **kwargs)

    def test_import_inert_and_helper_pin(self):
        code = ('import sys;sys.path.insert(0,sys.argv[1]);import unet_plan_overlay;'
                'assert not any(k in sys.modules for k in ("torch","tensorrt","onnx"))')
        subprocess.run([sys.executable, '-B', '-c', code, str(Path(overlay.__file__).parent)],
                       cwd=self.root, check=True, timeout=5)
        self.assertEqual(set(self.root.iterdir()), {self.native, self.portable})
        self.assertEqual(overlay.checked_helpers().NATIVE_SHA,
                         'f66b46ca38d0e34af69ee5c01be52d93cc3d2f1ba3ac7b8a68ae0426c3658316')

    def test_each_fixed_variant_has_exact_literal_plan_provenance(self):
        for variant, portable_blocks in overlay.VARIANTS.items():
            before = {p: (p / 'manifest.json').read_bytes() for p in (self.native, self.portable)}
            target = self.root / variant
            m = overlay.prepare(self.native, self.portable, target, variant=variant,
                native_sha=hashlib.sha256(before[self.native]).hexdigest(),
                portable_sha=hashlib.sha256(before[self.portable]).hexdigest())
            self.assertEqual(set(m['blocks']), overlay.BLOCKS)
            self.assertFalse(m['complete']); self.assertFalse(m['quality_accepted']); self.assertFalse(m['release_ready'])
            self.assertNotIn('probe', m); self.assertNotIn('runtime', m)
            for block, entry in m['blocks'].items():
                source = self.portable if block in portable_blocks else self.native
                self.assertEqual(entry, self.manifests[source]['blocks'][block])
                self.assertEqual((target / 'bs16' / entry['engine_file']).read_bytes(),
                                 (source / entry['engine_file']).read_bytes())
                self.assertEqual(m['block_provenance'][block]['source_directory'], str(source))
            for p, raw in before.items(): self.assertEqual((p / 'manifest.json').read_bytes(), raw)

    def test_unknown_variant_before_output(self):
        with self.assertRaisesRegex(ValueError, 'preregistered'): self.prepare('arbitrary_all_int8')
        self.assertFalse((self.root / 'candidate').exists())

    def test_interface_and_precision_mismatch_before_output(self):
        original = copy.deepcopy(self.manifests[self.portable])
        for field, value in (('inputs', ['bad']), ('outputs', ['bad'])):
            bad = copy.deepcopy(original); bad['blocks']['prefix'][field] = value
            self.write(self.portable, bad)
            with self.subTest(field=field), self.assertRaises(ValueError): self.prepare()
            self.assertFalse((self.root / 'candidate').exists())
        bad = copy.deepcopy(original); bad['blocks']['prefix']['build_flags']['precision'] = 'int8'
        self.write(self.portable, bad)
        with self.assertRaises(ValueError): self.prepare()

    def test_selected_plan_corruption_before_output(self):
        (self.portable / 'prefix.plan').write_bytes(b'corrupt')
        with self.assertRaises(ValueError): self.prepare()
        self.assertFalse((self.root / 'candidate').exists())

    def test_duplicate_and_escaping_plan_names_before_output(self):
        for filename in ('../prefix.plan', self.manifests[self.native]['blocks']['up3']['engine_file']):
            bad = copy.deepcopy(self.manifests[self.portable]); bad['blocks']['prefix']['engine_file'] = filename
            self.write(self.portable, bad)
            with self.subTest(filename=filename), self.assertRaises(ValueError): self.prepare()
            self.assertFalse((self.root / 'candidate').exists())

    def test_existing_candidate_never_overwritten(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, 'fresh_output'): self.prepare()

    def test_graph_differences_are_preserved_not_relabelled(self):
        m = self.prepare()
        self.assertNotEqual(self.manifests[self.native]['blocks']['prefix']['onnx_sha256'],
                            m['blocks']['prefix']['onnx_sha256'])
        self.assertEqual(m['blocks']['prefix']['onnx_sha256'],
                         self.manifests[self.portable]['blocks']['prefix']['onnx_sha256'])


if __name__ == '__main__':
    unittest.main()
