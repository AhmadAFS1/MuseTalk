"""Tiny verdict regressions: exercise acceptance logic without importing/allocating torch models."""
import ast
import copy
import math
from pathlib import Path
import unittest

SOURCE = Path(__file__).with_name("review_distilled_unet.py")
NAMES = {"CHECKS", "LIP_CORR_MARGIN", "LIP_DELTA_MARGIN_PX", "MOUTH_PSNR_MARGIN_DB",
         "MOUTH_SHARPNESS_MARGIN", "ERROR_FLICKER_MARGIN", "FLICKER_RATIO_MARGIN"}
FUNCTIONS = {"non_inferiority", "failures", "verdict"}
# Load the production functions directly; full imports would pull CUDA/Core ML dependencies into tiny CPU tests.
TREE = ast.parse(SOURCE.read_text())
BODY = [n for n in TREE.body if (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in NAMES for t in n.targets))
        or (isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS)]
NS = {"math": math, "r5_status": lambda g: "pass" if g.get("aperture_corr_ge_0.97") else "incomplete"}
exec(compile(ast.Module(body=BODY, type_ignores=[]), str(SOURCE), "exec"), NS)


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


if __name__ == "__main__":
    unittest.main()
