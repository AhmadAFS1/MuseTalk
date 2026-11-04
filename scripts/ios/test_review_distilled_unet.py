"""Tiny verdict regressions: exercise acceptance logic without importing/allocating torch models."""
import ast
import copy
import hashlib
import json
import tempfile
import contextlib
import io
from types import SimpleNamespace
import math
from pathlib import Path
import unittest

SOURCE = Path(__file__).with_name("review_distilled_unet.py")
NAMES = {"CHECKS", "LIP_CORR_MARGIN", "LIP_DELTA_MARGIN_PX", "MOUTH_PSNR_MARGIN_DB",
         "MOUTH_SHARPNESS_MARGIN", "ERROR_FLICKER_MARGIN", "FLICKER_RATIO_MARGIN", "NEW_PROTOCOL", "NEW_R5_GATE_KEYS", "ARMS", "SCORED", "HEADER"}
FUNCTIONS = {"non_inferiority", "failures", "verdict", "sha256_file", "validate_new_manifest", "finite_metric_tree",
             "new_r5_verdict", "protocol_inputs", "summarize_new", "table", "numbers", "fmt"}
# Load production control logic without importing torch/CUDA/Core ML.
TREE = ast.parse(SOURCE.read_text())
BODY = [n for n in TREE.body if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in NAMES for t in n.targets))
        or (isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS)]
NS = {"math": math, "hashlib": hashlib, "json": json, "Path": Path}
exec(compile(ast.Module(body=BODY, type_ignores=[]), str(SOURCE), "exec"), NS)
# Reuse the real gate status helper and the frozen threshold dictionary without ML imports.
for filename, name in (("review_mixed_units.py", "r5_status"), ("review_ane_vs_fp16.py", "R5_GATES")):
    tree = ast.parse(SOURCE.with_name(filename).read_text())
    nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name == name)
             or (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, "exec"), NS)



def baseline():
    return {"frames": 250, "fps": 25, "face_box": [0, 0, 100, 100], "mouth_box": [10, 50, 80, 40],
            "aperture_frames": 250, "lip_corr": .998, "lip_mean_abs_delta_px": .2, "mouth_psnr_db": 46.,
            "mouth_sharpness_ratio": .99, "error_flicker_mouth": .8, "mouth_flicker_ratio": 1.}


class VerdictTests(unittest.TestCase):
    def test_equal_passes_and_any_visual_regression_fails(self):
        d = baseline()
        self.assertEqual(NS["non_inferiority"](d, d)["status"], "pass")
        for key, higher, margin in NS["CHECKS"]:
            self.assertEqual(margin, 0., key)
            x = dict(d)
            x[key] += -.00001 if higher else .00001
            out = NS["non_inferiority"](x, d)
            self.assertEqual(out["status"], "FAIL", key)
            self.assertAlmostEqual(out["checks"][key]["delta_vs_D"], x[key] - d[key])

    def test_missing_or_nonfinite_metric_cannot_pass(self):
        d = baseline()
        for key, _, _ in NS["CHECKS"]:
            x = dict(d)
            del x[key]
            self.assertEqual(NS["non_inferiority"](x, d)["status"], "incomplete", key)
            x[key] = math.nan
            self.assertEqual(NS["non_inferiority"](x, d)["status"], "FAIL", key)

    def test_comparability_and_landmark_coverage(self):
        d = baseline()
        for key, value in (("frames", 200), ("fps", 24), ("face_box", [0, 0, 90, 90]),
                           ("mouth_box", [10, 50, 70, 30]), ("aperture_frames", 249)):
            x = dict(d)
            x[key] = value
            self.assertEqual(NS["non_inferiority"](x, d)["status"], "incomplete", key)
        self.assertEqual(NS["non_inferiority"](d, None)["status"], "incomplete")

    def test_all_clips_and_r5_required(self):
        m = {"clip": "one", "r5_gates_S": {"aperture_corr_ge_0.97": True, "all_pass": True},
             "vs_D": {"S": NS["non_inferiority"](baseline(), baseline())}}
        self.assertEqual(NS["verdict"]([m], {"one": {}}, "S")[0], "PASS")
        self.assertEqual(NS["verdict"]([], {}, "S")[0], "INCOMPLETE")
        self.assertEqual(NS["verdict"]([m], {"one": {}, "two": {}}, "S")[0], "INCOMPLETE")
        bad = copy.deepcopy(m)
        bad["r5_gates_S"]["mouth_sharpness_ratio_ge_0.95"] = False
        self.assertEqual(NS["verdict"]([bad], {"one": {}}, "S")[0], "FAIL")
        no_landmarks = copy.deepcopy(m)
        del no_landmarks["r5_gates_S"]["aperture_corr_ge_0.97"]
        self.assertEqual(NS["verdict"]([no_landmarks], {"one": {}}, "S")[0], "INCOMPLETE")


class NewAvatarVerdictTests(unittest.TestCase):
    def clips(self):
        rows = []
        for i in range(30):
            for audio in ("one", "two"):
                m = {"clip": f"new_{i:02d}__{audio}", "frames": 250, "fps": 25,
                     "face_box": [0, 0, 256, 256], "mouth_box": [50, 150, 100, 50], "D": None,
                     "files": {"dir": f"new_{i:02d}__{audio}", "labelled": "AS.mp4"},
                     "vs_D": {a: {"status": "not_evaluated"} for a in ("N", "S", "S16")},
                     "landmark_coverage": {a: 250 for a in ("A", "N", "S", "S16")},
                     "latent_vs_A": {}, "error_flicker_mouth": {}, "error_flicker_face": {}, "aperture": {}}
                for arm in ("N", "S", "S16"):
                    m[f"r5_gates_{arm}"] = {k: True for k in NS["NEW_R5_GATE_KEYS"]}
                    m[f"r5_gates_{arm}"]["all_pass"] = True
                    m["latent_vs_A"][arm] = {"mae": .001, "max_abs": .1, "p95_abs": .01}
                    m[f"{arm}_vs_A"] = {"raw_face": {"mae_255": .1}, "mouth_roi": {"psnr_db": 60., "flicker_ratio": 1.},
                                          "face_box": {"flicker_ratio": 1.}, "mouth_sharpness": {"ratio": 1.}}
                    m["error_flicker_mouth"][arm] = .1
                    m["error_flicker_face"][arm] = .1
                    m["aperture"][f"{arm}_vs_A"] = {"frames": 250, "corr": .999, "lag_frames": 0,
                                                        "mean_abs_delta_px": .05}
                rows.append(m)
        return rows

    def test_complete_set_passes_without_D_and_arms_are_independent(self):
        rows = self.clips()
        expected = [m["clip"] for m in rows]
        for arm in ("N", "S", "S16"):
            self.assertEqual(NS["new_r5_verdict"](rows, expected, arm), ("PASS", []))
        rows[0]["r5_gates_N"]["aperture_corr_ge_0.97"] = False
        self.assertEqual(NS["new_r5_verdict"](rows, expected, "N")[0], "FAIL")
        self.assertEqual(NS["new_r5_verdict"](rows, expected, "S")[0], "PASS")
        self.assertEqual(NS["new_r5_verdict"](rows, expected, "S16")[0], "PASS")

    def test_missing_duplicate_wrong_workload_and_coverage_cannot_pass(self):
        rows = self.clips()
        expected = [m["clip"] for m in rows]
        for incomplete in (rows[:-1], rows[:-1] + [rows[0]], []):
            self.assertEqual(NS["new_r5_verdict"](incomplete, expected, "S")[0], "INCOMPLETE")
        for key, val in (("frames", 249), ("fps", 24)):
            bad = copy.deepcopy(rows)
            bad[0][key] = val
            self.assertEqual(NS["new_r5_verdict"](bad, expected, "S")[0], "INCOMPLETE")
        for arm in ("A", "S"):
            bad = copy.deepcopy(rows)
            bad[0]["landmark_coverage"][arm] = 249
            self.assertEqual(NS["new_r5_verdict"](bad, expected, "S")[0], "INCOMPLETE")
        bad = copy.deepcopy(rows)
        del bad[0]["r5_gates_S"]["aperture_lag_0"]
        self.assertEqual(NS["new_r5_verdict"](bad, expected, "S")[0], "INCOMPLETE")

    def test_nonfinite_or_missing_metrics_cannot_pass(self):
        rows = self.clips()
        expected = [m["clip"] for m in rows]
        for value in (math.nan, math.inf, None):
            bad = copy.deepcopy(rows)
            bad[0]["S_vs_A"]["mouth_roi"]["psnr_db"] = value
            self.assertEqual(NS["new_r5_verdict"](bad, expected, "S")[0], "INCOMPLETE")
        bad = copy.deepcopy(rows)
        bad[0]["aperture"]["S_vs_A"]["frames"] = 10
        self.assertEqual(NS["new_r5_verdict"](bad, expected, "S")[0], "INCOMPLETE")

    def test_summary_reports_independent_r5_and_explicit_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "unet.pth").write_bytes(b"teacher")
            (root / "musetalk.json").write_text("{}")
            student_json = root / "student.json"
            student_json.write_text("{}")
            rows = self.clips()
            rows[0]["r5_gates_S"]["aperture_corr_ge_0.97"] = False
            info = {"manifest_sha256": "a" * 64, "expected_clips": [r["clip"] for r in rows]}
            args = SimpleNamespace(new_manifest_info=info, out=root, unet_dir=root, student_json=student_json)
            with contextlib.redirect_stdout(io.StringIO()):
                NS["summarize_new"](rows, {"sha256": "b" * 64}, {"fp16_device": "test"}, args)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["protocol"], "new_static_avatars_r5_v1")
            self.assertEqual({a: v["verdict"] for a, v in summary["verdicts"].items()}, {"N": "PASS", "S": "FAIL", "S16": "PASS"})
            self.assertEqual(summary["acceptance"]["D_comparison"], "not_evaluated")
            self.assertEqual(summary["r5_gates"], {"aperture_corr_min": .97, "aperture_lag_frames": 0,
                                                 "aperture_mean_abs_delta_px_max": .5, "flicker_ratio_max": 1.05,
                                                 "sharpness_ratio_min": .95})
            self.assertIn("cannot prove quality for all future characters", " ".join(summary["limitations"]))
            self.assertIn("not Core ML", " ".join(summary["limitations"]))
            self.assertIn("head and body", (root / "summary.md").read_text())

    def test_legacy_still_requires_D(self):
        m = {"clip": "one", "r5_gates_S": {"aperture_corr_ge_0.97": True, "all_pass": True},
             "vs_D": {"S": NS["non_inferiority"](baseline(), None)}}
        self.assertEqual(NS["verdict"]([m], {"one": {}}, "S")[0], "INCOMPLETE")


class NewAvatarManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root / "frozen.json"
        self.weights = self.root / "student.pt"
        self.weights.write_bytes(b"fixed-student")
        self.avatars, self.audios = [], []
        self.m = {"schema_version": 1, "protocol": "new_static_avatars_r5_v1", "avatar_kind": "static_portrait",
                  "fps": 25, "frames": 250, "checkpoint_sha256": NS["sha256_file"](self.weights),
                  "avatars": [], "audio": []}
        for i in range(30):
            avatar = self.root / f"new_{i:02d}"
            (avatar / "full_imgs").mkdir(parents=True)
            (avatar / "mask").mkdir()
            files = {}
            for relative in ("coords.pkl", "mask_coords.pkl", "latents.pt", "full_imgs/0000.png",
                             "full_imgs/0001.png", "mask/0000.png", "mask/0001.png"):
                file = avatar / relative
                # Byte-only caches intentionally avoid torch/PIL/model imports in manifest checks.
                content = f"{i}/{relative.split('/')[0]}".encode()
                file.write_bytes(content)
                files[relative] = NS["sha256_file"](file)
            portrait = self.root / f"portrait_{i:02d}.png"
            portrait.write_bytes(f"new-portrait-{i}".encode())
            self.avatars.append(avatar)
            self.m["avatars"].append({"id": avatar.name, "portrait": portrait.name,
                                      "portrait_sha256": NS["sha256_file"](portrait), "cache_files": files})
        for name in ("one.wav", "two.wav"):
            file = self.root / name
            file.write_bytes(name.encode())
            self.audios.append(file)
            self.m["audio"].append({"file": name, "sha256": NS["sha256_file"](file)})

    def validate(self, fps=25, seconds=10., frames=None):
        self.path.write_text(json.dumps(self.m))
        return NS["validate_new_manifest"](self.path, self.avatars, self.audios, self.weights, fps, seconds, frames)

    def test_frozen_manifest_binds_all_60_clips_and_hashes(self):
        out = self.validate()
        self.assertEqual(len(out["expected_clips"]), 60)
        self.assertEqual(out["manifest_sha256"], NS["sha256_file"](self.path))
        self.assertEqual(out["inputs"]["checkpoint_sha256"], NS["sha256_file"](self.weights))

    def test_new_protocol_never_reads_legacy_D_and_requires_manifest(self):
        self.validate()
        args = SimpleNamespace(protocol="new-static-r5", new_avatar_manifest=self.path, facemesh_python=Path("unused"),
                               student_weights=self.weights, fps=25, seconds=10., frames=None,
                               mixed_metrics=self.root / "missing-old-D.json")
        d, info = NS["protocol_inputs"](args, self.avatars, self.audios)
        self.assertEqual(d, {})
        self.assertEqual(len(info["expected_clips"]), 60)
        args.new_avatar_manifest = None
        with self.assertRaisesRegex(ValueError, "requires --new-avatar-manifest"):
            NS["protocol_inputs"](args, self.avatars, self.audios)
        args.protocol = "legacy-r5-d"
        args.mixed_metrics.write_text(json.dumps({"clips": [{"clip": "old"}]}))
        self.assertEqual(NS["protocol_inputs"](args, self.avatars, self.audios), ({"old": {"clip": "old"}}, None))

    def test_changed_checkpoint_portrait_audio_or_cache_rejected(self):
        targets = [self.weights, self.root / self.m["avatars"][0]["portrait"], self.audios[0], self.avatars[0] / "latents.pt"]
        for file in targets:
            original = file.read_bytes()
            file.write_bytes(original + b"-changed")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                self.validate()
            file.write_bytes(original)

    def test_missing_cache_duplicate_identity_and_shortened_protocol_rejected(self):
        for fps, seconds, frames in ((24, 10., None), (25, 5., None), (25, 10., 12)):
            with self.assertRaisesRegex(ValueError, "complete 250-frame"):
                self.validate(fps, seconds, frames)
        saved = copy.deepcopy(self.m)
        del self.m["avatars"][0]["cache_files"]["mask/0001.png"]
        with self.assertRaisesRegex(ValueError, "cache_files must cover"):
            self.validate()
        self.m = copy.deepcopy(saved)
        self.m["avatars"][1]["id"] = self.m["avatars"][0]["id"]
        with self.assertRaisesRegex(ValueError, "30 unique manifest IDs"):
            self.validate()
        self.m = copy.deepcopy(saved)
        self.m["avatars"][1]["portrait"] = self.m["avatars"][0]["portrait"]
        self.m["avatars"][1]["portrait_sha256"] = self.m["avatars"][0]["portrait_sha256"]
        with self.assertRaisesRegex(ValueError, "30 distinct portrait"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
