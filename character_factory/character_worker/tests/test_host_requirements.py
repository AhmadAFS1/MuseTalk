"""The qualified 3090 exception must not widen admission to other Ampere cards."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / 'orchestrator'))
import host_requirements as host
from engine import ADA_QUERY, DEFAULT_QUERY, matches_hardware_query
from vast_client import parse_query

REQ = json.loads((HERE / 'locks/character_worker.lock.json').read_text())['host_requirements']


def memory(mib=128802):
    return {'host_ram_mib': mib, 'effective_ram_mib': mib, 'cgroup_limit_bytes': None, 'limit_source': 'fixture'}


class HostProfileTest(unittest.TestCase):
    def select(self, name='NVIDIA GeForce RTX 3090', cap='8.6', vram=24576, ram=128802, **kw):
        return host.select_profile(REQ, name, cap, vram, memory(ram), **kw)

    def test_actual_qualified_3090_needs_no_override(self):
        r = self.select()
        self.assertEqual(r['profile_id'], 'rtx3090-sm86-24gb-128gb-v1')
        self.assertEqual(r['qualification_status'], 'qualified')
        self.assertEqual((r['minimum_vram_mib'], r['minimum_ram_mib']), (24500, 128000))
        self.assertEqual(REQ['compute_capability']['min'], 8.9)

    def test_3090_low_memory_and_wrong_compute_are_rejected_even_with_override(self):
        for kwargs in ({'vram': 24499}, {'ram': 127999}, {'cap': '8.9'}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.select(**kwargs, allow_untested=True)

    def test_other_ampere_and_blackwell_are_not_promoted(self):
        for name, cap in [('NVIDIA GeForce RTX 3060', '8.6'), ('NVIDIA GeForce RTX 3070', '8.6'),
                          ('NVIDIA GeForce RTX 3090 Ti', '8.6'), ('NVIDIA A100', '8.0'), ('NVIDIA GeForce RTX 5090', '12.0')]:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.select(name=name, cap=cap)

    def test_existing_ada_profile_keeps_its_original_memory_requirements(self):
        self.assertEqual(self.select(name='NVIDIA GeForce RTX 4070 SUPER', cap='8.9', vram=12282, ram=30952)['qualification_status'], 'qualified')
        self.assertEqual(self.select(name='NVIDIA GeForce RTX 4090', cap='8.9', vram=24564, ram=64000)['qualification_status'], 'qualified')
        for vram, ram in [(11999, 64000), (12282, 28999)]:
            with self.assertRaises(ValueError):
                self.select(name='NVIDIA GeForce RTX 4070 SUPER', cap='8.9', vram=vram, ram=ram)

    def test_deliberate_untested_gpu_remains_explicitly_experimental(self):
        r = self.select(name='NVIDIA A100', cap='8.0', allow_untested=True)
        self.assertEqual(r['qualification_status'], 'experimental')

    def test_nonfinite_gpu_facts_fail_closed(self):
        for args in ({'cap': 'nan'}, {'vram': float('inf')}):
            with self.assertRaises(ValueError):
                self.select(**args)


class EffectiveMemoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.info = self.root / 'meminfo'
        self.cg = self.root / 'cgroup'
        self.cg.mkdir()
        self.info.write_text(f'MemTotal: {256000 * 1024} kB\n')

    def test_container_limit_is_enforced_before_large_host_ram(self):
        (self.cg / 'memory.max').write_text(str(64000 * 1024 * 1024))
        facts = host.memory_facts(self.info, self.cg)
        self.assertEqual(facts['effective_ram_mib'], 64000)
        with self.assertRaises(ValueError):
            host.select_profile(REQ, 'NVIDIA GeForce RTX 3090', '8.6', 24576, facts)

    def test_unlimited_and_absent_cgroup_use_actual_host_ram(self):
        self.assertEqual(host.memory_facts(self.info, self.cg)['effective_ram_mib'], 256000)
        (self.cg / 'memory.max').write_text('max\n')
        facts = host.memory_facts(self.info, self.cg)
        self.assertEqual(facts['effective_ram_mib'], 256000)
        self.assertIn('unlimited', facts['limit_source'])
        self.info.write_text(f'MemTotal: {64000 * 1024} kB\n')
        self.assertEqual(host.memory_facts(self.info, self.cg)['effective_ram_mib'], 64000)

    def test_v1_limit_and_invalid_limit(self):
        (self.cg / 'memory').mkdir()
        (self.cg / 'memory/memory.limit_in_bytes').write_text(str(128000 * 1024 * 1024))
        self.assertEqual(host.memory_facts(self.info, self.cg)['effective_ram_mib'], 128000)
        (self.cg / 'memory.max').write_text('invalid')
        with self.assertRaises(ValueError):
            host.memory_facts(self.info, self.cg)

    def test_real_cli_records_qualified_facts_and_rejects_capped_container(self):
        out = self.root / 'profile.json'
        (self.cg / 'memory.max').write_text(str(174866 * 1024 * 1024))
        argv = [sys.executable, str(HERE / 'host_requirements.py'), '--lock', str(HERE / 'locks/character_worker.lock.json'),
                '--gpu-name', 'NVIDIA GeForce RTX 3090', '--compute-capability', '8.6', '--vram-mib', '24576',
                '--meminfo', str(self.info), '--cgroup', str(self.cg), '--output', str(out)]
        result = subprocess.run(argv, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(out.read_text())['qualification_status'], 'qualified')
        (self.cg / 'memory.max').write_text(str(64000 * 1024 * 1024))
        self.assertNotEqual(subprocess.run(argv, capture_output=True).returncode, 0)


class OfferQualificationTest(unittest.TestCase):
    def test_default_query_is_exact3090_and_provider_rounding_has_actual_boot_guard(self):
        query = parse_query(DEFAULT_QUERY)
        self.assertEqual(query['gpu_name'], {'eq': 'RTX 3090'})
        self.assertEqual(query['compute_cap'], {'eq': 860})
        self.assertEqual(query['gpu_ram']['gte'], 24000)
        self.assertEqual(query['cpu_ram']['gte'], 128000)
        good = {'gpu_name': 'RTX 3090', 'compute_cap': 860, 'gpu_ram': 24576, 'cpu_ram': 128802}
        self.assertTrue(matches_hardware_query(good, query))
        for patch in ({'gpu_name': 'RTX 4090'}, {'compute_cap': 890}, {'gpu_ram': 23999}, {'cpu_ram': 64000}, {'cpu_ram': float('nan')}):
            with self.subTest(patch=patch):
                self.assertFalse(matches_hardware_query({**good, **patch}, query))

    def test_explicit_ada_query_remains_available(self):
        self.assertTrue(matches_hardware_query({'gpu_name': 'RTX 4070S', 'compute_cap': 890, 'gpu_ram': 12282, 'cpu_ram': 30952}, parse_query(ADA_QUERY)))


if __name__ == '__main__':
    unittest.main()
