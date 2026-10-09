"""Synthetic CPU provenance tests; no measured quality acceptance."""
import copy
import importlib.util
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('mixed', HERE / 'quality_mixed_decoder.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class MixedDecoder(unittest.TestCase):
    def fixture(self):
        env = {'engines': [{'manifest_sha256': 'a'*64}],
               'taesd': {'key': 'real-portable-key', 'fingerprint': {'hardware_compatibility_level': 'ampere_plus'}}}
        return env, {'engine_manifest_sha256': 'a'*64, 'taesd': copy.deepcopy(env['taesd'])}

    def test_exactly_one_metadata_check_changes(self):
        raw = (HERE / 'quality_envelope.py').read_bytes()
        adapted = m.adapted(raw)
        self.assertEqual(adapted.decode().replace(m.ROLE_REPLACEMENT, m.ROLE_CHECK).encode(), raw)
        with self.assertRaises(ValueError): m.adapted(raw + b'\n')

    def test_explicit_native_unet_portable_decoder(self):
        env, selection = self.fixture()
        m.decoder_role(env, 'candidate', 'none', selection)
        for mutation in ('unet', 'decoder', 'native_label', 'portable_unet', 'unknown_role'):
            bad = copy.deepcopy(env); role, compatibility = 'candidate', 'none'
            if mutation == 'unet': bad['engines'][0]['manifest_sha256'] = 'b'*64
            elif mutation == 'decoder': bad['taesd']['key'] = 'another'
            elif mutation == 'native_label': bad['taesd']['fingerprint'].pop('hardware_compatibility_level')
            elif mutation == 'portable_unet': compatibility = 'ampere_plus'
            else: role = 'unknown'
            with self.assertRaises(ValueError): m.decoder_role(bad, role, compatibility, selection)

    def test_reference_roles_never_adapted(self):
        env, selection = self.fixture()
        m.decoder_role(env, 'reference', 'ampere_plus', selection)
        with self.assertRaises(ValueError): m.decoder_role(env, 'reference', 'none', selection)

    def test_original_metric_registry_and_functions_unchanged(self):
        env, selection = self.fixture()
        core = m.core_for(selection)
        self.assertEqual(core.checks.sha256(core.__file__), m.CORE_SHA)
        self.assertEqual(len(core.METRICS), 99)
        self.assertTrue(callable(core.compare) and callable(core.global_metrics))

    def test_default_off_before_files(self):
        with self.assertRaisesRegex(ValueError, 'explicit mixed-candidate'):
            m.main(['--candidate','/missing','--inputs','/missing','--envelope','/missing',
                    '--selection','/missing','--out','/not-written','--envelope-sha256','a'*64,
                    '--selection-sha256','a'*64])

    def test_actual_builder_manifest_schema_and_wrong_device(self):
        manifest = {'complete': True, 'hardware_compatibility_level': 'none', 'batch': 16,
            'variant': 'srccache', 'gpu': 'NVIDIA GeForce RTX 3090', 'compute_capability': [8, 6],
            'blocks': dict.fromkeys(('prefix','down0rest','down1','down2','down3','mid','up0','up1','up2','up3','tail')),
            'probe': {'graph_equals_direct_enqueue': True, 'deterministic_run_to_run': True}}
        m.verify_native_manifest(manifest)
        for field, value in (('compute_capability', [8, 9]), ('compute_capability', '8.6'),
                             ('gpu', 'NVIDIA GeForce RTX 4070 SUPER'), ('complete', False)):
            wrong = copy.deepcopy(manifest); wrong[field] = value
            with self.assertRaises(ValueError): m.verify_native_manifest(wrong)


if __name__ == '__main__': unittest.main()
