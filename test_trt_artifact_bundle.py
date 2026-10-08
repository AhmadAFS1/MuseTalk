#!/usr/bin/env python3
"""Unit tests for scripts/trt_artifact_bundle.py (CPU only, stdlib only, local-path URIs, no S3).

Covers the legacy repo-root sidecar layout (recipe legacy_int8, unchanged) and the --sidecar-dir
layout of the pinned bundles (recipe r5): relative engine-set symlinks, the restore stamp,
--skip-if-verified, --stage-dir and adopt. Everything runs in a TemporaryDirectory.

Run: PYTHONDONTWRITEBYTECODE=1 python3 -B -m unittest -v test_trt_artifact_bundle
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

REPO = Path(__file__).resolve().parent
TOOL = REPO / "scripts" / "trt_artifact_bundle.py"
MANIFEST = ".musetalk_trt_artifact_manifest.json"
SUMS = ".musetalk_trt_artifact_SHA256SUMS"
STAMP = ".musetalk_trt_artifact_restored.json"
SIDE = ".runtime/trt_artifacts/test-bundle"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else data.encode())


class BundleTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="mt_bundle_")
        self.tmp = Path(self._tmp.name)
        (self.tmp / "tmpdir").mkdir()

    def tearDown(self):
        self._tmp.cleanup()

    def run_tool(self, *argv):
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", TMPDIR=str(self.tmp / "tmpdir"))
        for key in ("MUSETALK_TRT_ARTIFACT_SHA256", "MUSETALK_TRT_ARTIFACT_STAGE_DIR"):
            env.pop(key, None)
        return subprocess.run([sys.executable, "-B", str(TOOL), *map(str, argv)], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=120)

    def assertRc(self, proc, rc=0):
        self.assertEqual(proc.returncode, rc, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}")

    # -- sources ---------------------------------------------------------------------------
    def r5_source(self) -> Path:
        """An r5-shaped tree: an engine-set folder of relative symlinks to real block files."""
        src = self.tmp / "r5src"
        write(src / "models/set_real/bs16/down1.int8.plan", os.urandom(50_000))
        write(src / "models/set_real/bs16/tail.plan", os.urandom(5_000))
        write(src / "models/set_alias/bs16/manifest.json", json.dumps({"complete": True}))
        for name in ("down1.int8.plan", "tail.plan"):
            os.symlink(f"../../set_real/bs16/{name}", src / "models/set_alias/bs16" / name)
        write(src / "calibration/c/a.pt", os.urandom(1_000))
        return src

    def create_r5(self, src: Path) -> Path:
        out = self.tmp / "r5.tar.gz"
        proc = self.run_tool("--repo-root", src, "--strict", "--sidecar-dir", SIDE, "create", "--output", out,
                             "--profile", "test-r5", "--required-files", "models/set_alias/bs16/manifest.json",
                             "--required-dirs", "models/set_real,models/set_alias,calibration/c",
                             "--keep-symlinks", "--compresslevel", "1")
        self.assertRc(proc)
        return out

    def fresh_repo(self, name="dst") -> Path:
        dst = self.tmp / name
        dst.mkdir()
        # the tracked legacy pair in the repo root must never be touched by a --sidecar-dir restore
        write(dst / MANIFEST, '{"legacy": true}\n')
        write(dst / SUMS, "legacy\n")
        return dst

    def restore(self, dst, archive, expected, *extra):
        return self.run_tool("--repo-root", dst, "--strict", "--sidecar-dir", SIDE, "restore", "--uri", archive,
                             "--expected-sha256", expected, "--stage-dir", dst / "tmp/stage", *extra)

    # -- legacy layout ---------------------------------------------------------------------
    def test_legacy_default_layout_unchanged(self):
        src = self.tmp / "legacy"
        write(src / "models/tensorrt_unet_static_bs8_20260529/unet_trt.ts", os.urandom(20_000))
        write(src / "models/tensorrt_unet_static_bs8_20260529/unet_trt_meta.json",
              json.dumps({"validation": {"passed": True}}))
        write(src / "calibration/vae_decoder/a.pt", os.urandom(1_000))
        write(src / "models/tensorrt/stagewise_int8_onnx_qdq_cache/x.plan", os.urandom(3_000))
        out = self.tmp / "legacy.tar.gz"
        self.assertRc(self.run_tool("--repo-root", src, "--strict", "create", "--output", out))
        self.assertTrue((src / MANIFEST).is_file(), "legacy create writes the root sidecars")
        dst = self.tmp / "legacy_dst"
        dst.mkdir()
        proc = self.run_tool("--repo-root", dst, "--strict", "restore", "--uri", out, "--expected-sha256", sha(out))
        self.assertRc(proc)
        self.assertEqual(json.loads((dst / MANIFEST).read_text())["profile"], "vae-int8-unet-trt-split8")
        self.assertFalse((dst / ".runtime").exists())
        self.assertRc(self.run_tool("--repo-root", dst, "--strict", "verify"))
        self.assertEqual(list((self.tmp / "tmpdir").iterdir()), [], "archive staged in TMPDIR and removed")

    # -- sidecar-dir layout ----------------------------------------------------------------
    def test_explicit_empty_optional_does_not_reintroduce_defaults(self):
        # Exercise a nonempty default even though today's optional defaults are
        # empty; this fails under the old `parse_csv(...) or defaults` behavior.
        spec = importlib.util.spec_from_file_location("bundle_optional_test", TOOL)
        bundle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bundle)
        src = self.r5_source()
        write(src / "models/legacy_optional/old.plan", b"unrelated legacy payload")
        with mock.patch.object(bundle, "DEFAULT_OPTIONAL_PATHS", ("models/legacy_optional",)):
            for empty in (False, True):
                out = self.tmp / ("empty.tar.gz" if empty else "default.tar.gz")
                argv = [str(TOOL), "--repo-root", str(src), "--strict", "--sidecar-dir", SIDE,
                        "create", "--output", str(out), "--required-files", "models/set_alias/bs16/manifest.json",
                        "--required-dirs", "models/set_real", *(["--optional-paths", ""] if empty else [])]
                with mock.patch.object(sys, "argv", argv):
                    args = bundle.parse_args()
                self.assertEqual(bundle.create_bundle(args), 0)
                manifest = json.loads((src / SIDE / MANIFEST).read_text())
                paths = {entry["path"] for entry in manifest["files"]}
                self.assertEqual("models/legacy_optional/old.plan" in paths, not empty)
                self.assertEqual(manifest["optional_paths"], [] if empty else ["models/legacy_optional"])

    def test_keep_symlinks_restore_into_sidecar_dir(self):
        src = self.r5_source()
        out = self.create_r5(src)
        self.assertFalse((src / MANIFEST).exists(), "create --sidecar-dir leaves the repo root alone")
        manifest = json.loads((src / SIDE / MANIFEST).read_text())
        self.assertEqual({e["path"] for e in manifest["files"] if "symlink" in e},
                         {"models/set_alias/bs16/down1.int8.plan", "models/set_alias/bs16/tail.plan"})
        dst = self.fresh_repo()
        proc = self.restore(dst, out, sha(out))
        self.assertRc(proc)
        link = dst / "models/set_alias/bs16/down1.int8.plan"
        self.assertTrue(link.is_symlink())
        self.assertEqual(os.readlink(link), "../../set_real/bs16/down1.int8.plan")
        self.assertEqual(sha(link), sha(src / "models/set_real/bs16/down1.int8.plan"))
        self.assertEqual((dst / MANIFEST).read_text(), '{"legacy": true}\n', "root sidecars untouched")
        self.assertEqual((dst / SUMS).read_text(), "legacy\n")
        stamp = json.loads((dst / SIDE / STAMP).read_text())
        self.assertEqual((stamp["archive_sha256"], stamp["mode"]), (sha(out), "restored"))
        self.assertEqual(list((dst / "tmp/stage").iterdir()), [], "staged archive removed")
        self.assertRc(self.run_tool("--repo-root", dst, "--strict", "--sidecar-dir", SIDE, "verify"))

    def test_skip_if_verified_and_redownload_on_drift(self):
        src = self.r5_source()
        out = self.create_r5(src)
        dst = self.fresh_repo()
        self.assertRc(self.restore(dst, out, sha(out), "--skip-if-verified"))
        # a reboot: stamp + files still verify -> no download (the URI does not even exist)
        proc = self.restore(dst, self.tmp / "gone.tar.gz", sha(out), "--skip-if-verified")
        self.assertRc(proc)
        self.assertIn("skipping download", proc.stdout)
        # a drifted file forces a download; with the archive gone that fails and clears the stamp
        write(dst / "models/set_real/bs16/tail.plan", b"drift")
        proc = self.restore(dst, self.tmp / "gone.tar.gz", sha(out), "--skip-if-verified")
        self.assertRc(proc, 1)
        self.assertFalse((dst / SIDE / STAMP).exists())
        self.assertEqual(list((dst / "tmp/stage").iterdir()), [], "no placeholder left after a failed fetch")
        self.assertRc(self.restore(dst, out, sha(out), "--skip-if-verified"))
        self.assertEqual(sha(dst / "models/set_real/bs16/tail.plan"), sha(src / "models/set_real/bs16/tail.plan"))

    def test_wrong_sha_refused_without_stamp(self):
        out = self.create_r5(self.r5_source())
        dst = self.fresh_repo()
        proc = self.restore(dst, out, "0" * 64)
        self.assertRc(proc, 1)
        self.assertIn("SHA256 mismatch", proc.stderr)
        self.assertFalse((dst / SIDE / STAMP).exists())
        self.assertFalse((dst / "models").exists(), "nothing extracted")

    def test_skip_if_verified_needs_sidecar_dir(self):
        out = self.create_r5(self.r5_source())
        dst = self.fresh_repo()
        proc = self.run_tool("--repo-root", dst, "restore", "--uri", out, "--expected-sha256", sha(out),
                             "--skip-if-verified")
        self.assertRc(proc, 1)
        self.assertIn("--skip-if-verified needs --sidecar-dir", proc.stderr)

    def test_adopt_binds_present_files_without_extracting(self):
        src = self.r5_source()
        out = self.create_r5(src)
        dst = self.tmp / "adopt"
        shutil.copytree(src / "models", dst / "models", symlinks=True)
        shutil.copytree(src / "calibration", dst / "calibration")
        proc = self.run_tool("--repo-root", dst, "--strict", "--sidecar-dir", SIDE, "adopt", "--uri", out,
                             "--expected-sha256", sha(out))
        self.assertRc(proc)
        stamp = json.loads((dst / SIDE / STAMP).read_text())
        self.assertEqual((stamp["archive_sha256"], stamp["mode"]), (sha(out), "adopted"))
        self.assertEqual((dst / SIDE / MANIFEST).read_bytes(), (src / SIDE / MANIFEST).read_bytes())
        # restore --skip-if-verified now trusts the adopted stamp
        proc = self.restore(dst, self.tmp / "gone.tar.gz", sha(out), "--skip-if-verified")
        self.assertRc(proc)
        # a local file that differs from the bundle is refused and leaves no stamp
        write(dst / "calibration/c/a.pt", b"different")
        proc = self.run_tool("--repo-root", dst, "--strict", "--sidecar-dir", SIDE, "adopt", "--uri", out,
                             "--expected-sha256", sha(out))
        self.assertRc(proc, 1)
        self.assertFalse((dst / SIDE / STAMP).exists())
        # wrong archive sha and a missing --sidecar-dir are refused
        proc = self.run_tool("--repo-root", dst, "--sidecar-dir", SIDE, "adopt", "--uri", out,
                             "--expected-sha256", "0" * 64)
        self.assertRc(proc, 1)
        proc = self.run_tool("--repo-root", dst, "adopt", "--uri", out, "--expected-sha256", sha(out))
        self.assertRc(proc, 1)
        self.assertIn("adopt needs --sidecar-dir", proc.stderr)


if __name__ == "__main__":
    unittest.main()
