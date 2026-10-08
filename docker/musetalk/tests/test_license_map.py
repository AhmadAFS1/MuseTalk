"""Offline notice/byte-binding tests; never download models or approve an image."""
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import build_public_license_map


class LicenseMapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.dossier = Path(self.temporary.name) / "dossier"
        shutil.copytree(ROOT / "notice_dossier_20261008", self.dossier)
        self.rules = Path(self.temporary.name) / "rules.json"
        shutil.copyfile(ROOT / "public_payload_assessment_rules.json", self.rules)

    def mutate(self, path, change):
        data = json.loads(path.read_text())
        change(data)
        path.write_text(json.dumps(data))

    def test_exact_ten_eligible_records_keep_exclusions_and_runtime_gates(self):
        result = build_public_license_map.assemble(self.dossier, self.rules)
        self.assertEqual(len(result["model_files"]), 10)
        self.assertTrue(all(x["upstream_redistribution_grant_identified"] for x in result["model_files"]))
        self.assertTrue(all(x["retained_notices"] for x in result["model_files"]))
        self.assertEqual(len(result["excluded_model_payloads"]["paths"]), 5)
        self.assertIn("PENDING_SPECIFIC_RUNTIME_GATES", result["final_image_publication_decision"])

    def test_rejects_changed_compvis_notice(self):
        (self.dossier / "supplemental/compvis_autoencoder_mit.txt").write_text("wrong notice")
        with self.assertRaises(ValueError):
            build_public_license_map.assemble(self.dossier, self.rules)

    def test_rejects_not_in_capture_even_with_reference_hash(self):
        self.mutate(self.dossier / "3090-model-byte-binding.json",
                    lambda d: d["models"][0].update(byte_binding="NOT_IN_CAPTURE"))
        with self.assertRaises(ValueError):
            build_public_license_map.assemble(self.dossier, self.rules)

    def test_rejects_wrong_captured_bytes_even_if_labeled_match(self):
        self.mutate(self.dossier / "3090-model-byte-binding.json",
                    lambda d: d["models"][0]["actual"].update(sha256="0" * 64))
        with self.assertRaises(ValueError):
            build_public_license_map.assemble(self.dossier, self.rules)

    def test_rejects_duplicate_captured_model(self):
        self.mutate(self.dossier / "3090-model-byte-binding.json",
                    lambda d: d["models"].append(d["models"][0]))
        with self.assertRaises(ValueError):
            build_public_license_map.assemble(self.dossier, self.rules)
