"""CPU-only contract tests. Fixtures are intentionally synthetic, not GPU evidence."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import context
import release


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.m = {
            "schema": release.SCHEMA, "status": "validated", "source_revision": "a" * 40,
            "cuda_base": "fixture.invalid/cuda@sha256:" + "b" * 64, "platform": "linux/amd64",
            "matrix": "cu121", "bundle_name": "rtx3090-r5-srcg50-int8", "redistribution_reviewed": True,
            "avatar_prep": True, "kokoro": False, "vp8_encoder": "native",
            "source_files": {"api_server.py": {}},
            "model_files": {"models/fixture/plan": {"public_redistribution": True, "license_id": "fixture"}},
            "notices": {"licenses/fixture.txt": {}}, "evidence": {"quality.json": {}},
            "archives": {"weights.tar.gz": {}, "native.tar.gz": {"sha256": "d" * 64}},
            "bundle_manifest_sha256": "c" * 64,
            "apt_packages": [name + "=1.0" for name in sorted(release.REQUIRED_APT)],
        }

    def tearDown(self):
        self.tmp.cleanup()

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}

    def manifest(self, value=None):
        path = self.root / "release.json"
        path.write_text(json.dumps(self.m if value is None else value))
        return path

    def test_valid_contract_schema(self):
        self.assertEqual(release.load_manifest(self.manifest())["matrix"], "cu121")

    def test_absent_or_unvalidated_release_rejected(self):
        for state in (None, "candidate", "incomplete", "rejected"):
            self.m["status"] = state
            with self.subTest(state=state), self.assertRaises(ValueError):
                release.load_manifest(self.manifest())

    def test_floating_base_wrong_arch_portable_bundle_rejected(self):
        for key, value in (("cuda_base", "nvidia/cuda:latest"), ("platform", "linux/arm64"),
                           ("bundle_name", "ampere-plus-r5-srcg50-int8"), ("matrix", "cu128")):
            m = {**self.m, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                release.load_manifest(self.manifest(m))

    def test_candidate_requires_nonpromotable_identity(self):
        self.m.update(status="candidate", promotion_eligible=False, candidate_reason="Synthetic test; native gate not met")
        self.assertEqual(release.load_manifest(self.manifest())["status"], "candidate")
        self.m["promotion_eligible"] = True
        with self.assertRaises(ValueError):
            release.load_manifest(self.manifest())

    def test_candidate_does_not_implicitly_pass_release_or_register(self):
        self.m.update(status="candidate", promotion_eligible=False, candidate_reason="Synthetic test")
        self.evidence()
        self.quality.update(decision="incomplete", quality_parity_with_reference="NOT_RUN")
        self.aggregate.update(status="NOT_RUN", limitation="Synthetic test has no GPU evidence")
        self.aggregate["T"] = []
        self.aggregate["SUST"] = []
        self.refresh_evidence()
        release.verify_evidence(self.root, self.m)
        with self.assertRaises(ValueError):
            release.stage(self.root, self.m, self.root, self.m["source_revision"], self.m["cuda_base"])
        settings = release.policy(self.m, {"engines": {"unet_stagewise": {"cache_dir": "models/native"}, "taesd_trt": {"dir": "models/taesd"}}})
        self.assertEqual(settings["LINGUA_WORKER_CALLBACK_REQUIRED"], "0")
        self.assertEqual(settings["LINGUA_CONTROL_PLANE_ENABLED"], "0")
        self.assertEqual(settings["LINGUA_CONTROL_PLANE_ENV_FILE"], "/dev/null")
        self.m["status"] = "validated"
        with self.assertRaises(ValueError):
            release.verify_evidence(self.root, self.m)

    def test_release_policy_overrides_unsafe_caller_and_secret_settings(self):
        unsafe = {"MUSETALK_UNET_STAGEWISE_PROBE_CHECK": "0", "MUSETALK_UNET_STAGEWISE_PROBE_TOL": "0.5",
                  "MUSETALK_TRT_FALLBACK": "1", "MUSETALK_VP8_FALLBACK": "1",
                  "LINGUA_CONTROL_PLANE_ENABLED": "0", "LINGUA_WORKER_CALLBACK_REQUIRED": "0",
                  "LINGUA_CONTROL_PLANE_ENV_FILE": "/workspace/stale.env", "TURN_ENV_FORCE": "0"}
        descriptor = {"engines": {"unet_stagewise": {"cache_dir": "models/native"}, "taesd_trt": {"dir": "models/taesd"}}}
        with patch.dict(os.environ, unsafe):
            settings = release.policy(self.m, descriptor)
        for key, value in unsafe.items():
            self.assertNotEqual(settings[key], value, key)
        self.assertEqual(settings["MUSETALK_UNET_STAGEWISE_PROBE_TOL"], "0")
        self.assertEqual(settings["MUSETALK_TRT_FALLBACK"], "0")
        self.assertEqual(settings["LINGUA_CONTROL_PLANE_ENABLED"], "1")
        self.assertEqual(settings["LINGUA_WORKER_CALLBACK_REQUIRED"], "1")

    def test_license_review_and_full_api_capability_required(self):
        for key in ("redistribution_reviewed", "avatar_prep"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                release.load_manifest(self.manifest({**self.m, key: False}))

    def test_model_redistribution_required(self):
        self.m["model_files"]["models/fixture/plan"]["public_redistribution"] = False
        with self.assertRaises(ValueError):
            release.load_manifest(self.manifest())

    def test_kokoro_cannot_defer_first_request_download(self):
        self.m["kokoro"] = True
        with self.assertRaises(ValueError):
            release.load_manifest(self.manifest())

    def test_apt_pins_reject_unpinned_and_injection(self):
        for value in ("curl", "curl=1;bad", "$(bad)", "--allow-unauthenticated=1"):
            m = {**self.m, "apt_packages": [*self.m["apt_packages"], value]}
            with self.subTest(value=value), self.assertRaises(ValueError):
                release.load_manifest(self.manifest(m))

    def test_bad_file_size_hash_missing_and_symlink(self):
        entry = self.write("model.bin", b"abcd")
        release.check_file(self.root, "model.bin", entry)
        for name, e in (("missing", entry), ("model.bin", {**entry, "size_bytes": 5}),
                         ("model.bin", {**entry, "sha256": "0" * 64})):
            with self.subTest(name=name, entry=e), self.assertRaises(ValueError):
                release.check_file(self.root, name, e)
        (self.root / "link").symlink_to(self.root / "model.bin")
        with self.assertRaises(ValueError):
            release.check_file(self.root, "link", entry)

    def test_unsafe_paths(self):
        for value in ("../escape", "/escape", "a/../b", "a//b", "", "./a"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                release.relative(value)

    def test_secret_detection_suppresses_value(self):
        secret = "AKIA" + "A" * 16
        self.write("leak.txt", secret.encode())
        with self.assertRaises(ValueError) as caught:
            release.scan_text(self.root / "leak.txt")
        self.assertNotIn(secret, str(caught.exception))

    def test_manifest_extra_field_cannot_hide_recognized_secret(self):
        self.m["accidental_key"] = "AKIA" + "Z" * 16
        with self.assertRaises(ValueError) as caught:
            release.load_manifest(self.manifest())
        self.assertNotIn(self.m["accidental_key"], str(caught.exception))

    def archive(self, name, kind="file", duplicate=False):
        path = self.root / "input.tar.gz"
        with tarfile.open(path, "w:gz") as tar:
            for _ in range(2 if duplicate else 1):
                item = tarfile.TarInfo(name)
                if kind == "link":
                    item.type = tarfile.SYMTYPE
                    item.linkname = "/etc/passwd"
                    tar.addfile(item)
                else:
                    item.size = 4
                    tar.addfile(item, io.BytesIO(b"test"))
        return path

    def test_safe_archive_and_duplicate_rejected(self):
        archive = self.archive("models/fixture/plan")
        target = self.root / "extract"
        release.extract(archive, target, {"models/fixture/plan"})
        self.assertEqual((target / "models/fixture/plan").read_bytes(), b"test")
        with self.assertRaises(ValueError):
            release.extract(self.archive("models/fixture/plan", duplicate=True), self.root / "dup", {"models/fixture/plan"})

    def test_archive_traversal_links_unlisted_and_missing_rejected(self):
        for name, kind in (("../escape", "file"), ("models/fixture/plan", "link"), ("unexpected", "file")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                release.extract(self.archive(name, kind), self.root / "extract", {"models/fixture/plan"})
        with self.assertRaises(ValueError):
            release.extract(self.archive("models/fixture/plan"), self.root / "missing", {"models/fixture/plan", "missing"})

    def evidence(self):
        self.m["quality_decision_file"] = "quality.json"
        self.m["aggregate_acceptance_file"] = "aggregate.json"
        self.quality = {"decision": "accepted", "bundle_sha256": "d" * 64,
                        "quality_parity_with_reference": "PASS", "strict_original_gates": "FAIL",
                        "visual_inspection_file": "visual.json"}
        window = {"elapsed_seconds": 60.0, "completed_valid_frames": 24000, "status": "PASS", "raw_report": "raw.json"}
        self.aggregate = {"gpu": "NVIDIA GeForce RTX 3090", "bundle_sha256": "d" * 64,
                          "T": [copy.deepcopy(window) for _ in range(2)], "SUST": [copy.deepcopy(window) for _ in range(5)]}
        self.refresh_evidence()

    def refresh_evidence(self):
        self.m["evidence"] = {name: self.write(name, json.dumps(value).encode()) for name, value in
                              (("quality.json", self.quality), ("aggregate.json", self.aggregate), ("visual.json", {}), ("raw.json", {}))}

    def test_exact_400_shared_denominator_and_inherited_strict_failure(self):
        self.evidence()
        release.verify_evidence(self.root, self.m)
        self.aggregate["SUST"][0]["completed_valid_frames"] = 23999
        self.refresh_evidence()
        with self.assertRaises(ValueError):
            release.verify_evidence(self.root, self.m)

    def test_nan_short_window_and_missing_window_rejected(self):
        for duration in (float("nan"), float("inf"), 59.9):
            self.evidence()
            self.aggregate["T"][0]["elapsed_seconds"] = duration
            self.refresh_evidence()
            with self.subTest(duration=duration), self.assertRaises(ValueError):
                release.verify_evidence(self.root, self.m)
        self.evidence()
        self.aggregate["SUST"].pop()
        self.refresh_evidence()
        with self.assertRaises(ValueError):
            release.verify_evidence(self.root, self.m)

    def test_missing_raw_report_wrong_gpu_quality_rejected(self):
        self.evidence()
        self.m["evidence"].pop("raw.json")
        with self.assertRaises(ValueError):
            release.verify_evidence(self.root, self.m)
        self.evidence()
        self.aggregate["gpu"] = "NVIDIA GeForce RTX 4070 SUPER"
        self.refresh_evidence()
        with self.assertRaises(ValueError):
            release.verify_evidence(self.root, self.m)
        self.evidence()
        self.quality["decision"] = "incomplete"
        self.refresh_evidence()
        with self.assertRaises(ValueError):
            release.verify_evidence(self.root, self.m)

    def test_context_allows_code_not_secrets_or_private_assets(self):
        for name in ("api_server.py", "scripts/vast_onstart.sh", "configs/recipes/r5.env", "musetalk/utils/utils.py"):
            self.assertTrue(context.allowed(name), name)
        for name in (".env", ".runtime/secrets.env", "character_factory/generated/private.json", "uploads/person.mp4",
                     "models/private.pt", "scripts/private.pem", "docker/musetalk/tests/test_release.py"):
            self.assertFalse(context.allowed(name), name)

    def test_immutable_setup_nonzero_checks_fail_without_install(self):
        # Exercise the actual shell function without its GPU/system-dependent main.
        source = (Path(__file__).resolve().parents[3] / "scripts/vast_onstart.sh").read_text()
        function = source.split("run_setup_if_needed() {", 1)[1].split("\n}\n", 1)[0]
        installer = self.root / "installer.sh"
        installer.write_text("exit 99\n")
        venv = self.root / "venv/bin"
        venv.mkdir(parents=True)
        (venv / "python").symlink_to(sys.executable)
        for rc in (11, 17):
            script = f'''set -e
IMMUTABLE_BOOT=1; AUTO_SETUP=0; HF_MAX_WORKERS=1
INSTALLER={installer}; VENV_PATH={venv.parent}; INSTALL_CHECK_RC={rc}
installer_group_args() {{ :; }}
run_install_check() {{ :; }}
env_flag_is_true() {{ [[ "$1" == 1 ]]; }}
log() {{ :; }}
die() {{ echo "$*"; exit 1; }}
run_setup_if_needed() {{{function}
}}
run_setup_if_needed
'''
            result = subprocess.run(["bash", "-c", script], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("rebuild instead of repairing", result.stdout)


if __name__ == "__main__":
    unittest.main()
