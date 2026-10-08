"""Dependency diagnostic syntax/argument tests without Docker or a network."""
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dependency_preflight


class DependencyPreflightTests(unittest.TestCase):
    def test_apt_probe_preserves_exact_candidate_version(self):
        mock = "apt-get() { return 0; }; apt-cache() { printf 'Package:\\n  Installed: (none)\\n  Candidate: 1:2.3-4~22.04\\n'; };\n"
        result = subprocess.run(["bash", "-c", mock + dependency_preflight.APT_PROBE, "test", "python3", "curl"],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ["python3=1:2.3-4~22.04", "curl=1:2.3-4~22.04"])

    def test_apt_probe_rejects_unavailable_package(self):
        mock = "apt-get() { return 0; }; apt-cache() { printf '  Candidate: (none)\\n'; };\n"
        result = subprocess.run(["bash", "-c", mock + dependency_preflight.APT_PROBE, "test", "python3"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)

    def test_dependency_diagnostic_has_no_release_or_serving_claim(self):
        source = (Path(__file__).resolve().parents[1] / "Dockerfile.dependencies").read_text()
        self.assertIn("--skip-weights --no-selftest", source)
        self.assertIn("TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2", source)
        self.assertIn("dependency-diagnostic-no-models", source)
        self.assertIn("exit 64", source)
        self.assertRegex(dependency_preflight.BASE, r"^nvidia/cuda@sha256:[0-9a-f]{64}$")

    def test_size_experiment_is_fresh_stage_not_delete_in_shipping_layer(self):
        source = (Path(__file__).resolve().parents[1] / "Dockerfile.dependencies").read_text()
        self.assertEqual(source.count("FROM ${CUDA_BASE}"), 1)
        self.assertIn("FROM ${CUDA_RUNTIME_BASE} AS diagnostic", source)
        self.assertRegex(dependency_preflight.RUNTIME_BASE, r"^nvidia/cuda@sha256:[0-9a-f]{64}$")
        self.assertNotEqual(dependency_preflight.RUNTIME_BASE, dependency_preflight.BASE)
        self.assertIn("COPY --from=build /opt/musetalk /opt/musetalk", source)
        self.assertLess(source.index("prune_diagnostic.py --execute"), source.index("AS diagnostic"))
        self.assertIn("final-stage mmcv CUDA", source)
