"""CPU contact extraction tests; these do not judge visual quality."""
import copy
from fractions import Fraction
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest import mock
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import quality_contact_sheets as contact


def metric():
    aperture, seam, chin = [0.] * 240, [0.] * 240, [0.] * 240
    aperture[42], seam[75], chin[100], chin[5] = 5., 3., -7., 999.
    return {"lip_sync": {"series": {"A_px": aperture, "B_px": aperture}},
            "seam": {"series": {"band_mean": seam, "ring_mean": seam}},
            "chin": {"B": {"series_signed_error_px": chin}},
            "flicker": {"temporal_hf_power_gt_6hz": {"mouth_box": {"box_xyxy": [100, 200, 300, 400]},
                                                              "jaw_box": {"box_xyxy": [80, 300, 400, 500]}}}}


class ContactSheetTests(unittest.TestCase):
    def test_selection_uses_actual_series_and_original_chin_window(self):
        result = contact.select_frames({"portable_ref1": metric()})
        self.assertEqual([r["frame_index"] for r in result], [0, 42, 75, 100, 120, 239])
        self.assertFalse(any(x["selection"] == "reviewed_silence" for r in result for x in r["reasons"]))
        self.assertEqual(result[1]["canonical_time_seconds"], {"numerator": 42, "denominator": 24})
        self.assertTrue(any(r["selection"] == "largest_reported_absolute_chin_error" and r["signed_value"] == -7
                            for r in result[3]["reasons"]))

    def test_deterministic_first_frame_tie_and_missing_series_fail(self):
        m = metric()
        m["lip_sync"]["series"]["A_px"][41] = 5.
        self.assertIn(41, [r["frame_index"] for r in contact.select_frames({"r1": m})])
        for value in ([0.] * 239, [float("nan")] * 240, [None] * 240):
            m = metric()
            m["lip_sync"]["series"]["B_px"] = value
            with self.assertRaises(contact.checks.Invalid):
                contact.select_frames({"r1": m})

    def test_reviewed_silence_is_explicit_not_mouth_proxy(self):
        note = {"review_method": "reviewed_audio_intervals", "reviewer": "test reviewer", "evidence_note": "SYNTHETIC unit-test interval",
                "intervals": [{"start_frame": 10, "end_frame_exclusive": 20}]}
        rows = contact.select_frames({"r1": metric()}, note)
        self.assertIn(14, [r["frame_index"] for r in rows])
        for mutation in ("method", "attribution", "range"):
            bad = copy.deepcopy(note)
            if mutation == "method":
                bad["review_method"] = "small_aperture"
            elif mutation == "attribution":
                bad["reviewer"] = ""
            else:
                bad["intervals"][0]["end_frame_exclusive"] = 241
            with self.assertRaises(contact.checks.Invalid):
                contact.select_frames({"r1": metric()}, bad)

    def test_paired_crop_union_has_no_resizing(self):
        second = metric()
        second["flicker"]["temporal_hf_power_gt_6hz"]["mouth_box"]["box_xyxy"] = [90, 210, 320, 390]
        self.assertEqual(contact.crop_union({"r1": metric(), "native": second}, "mouth_box"), [90, 200, 320, 400])
        second["flicker"]["temporal_hf_power_gt_6hz"]["mouth_box"]["box_xyxy"][2] = 999
        with self.assertRaises(contact.checks.Invalid):
            contact.crop_union({"native": second}, "mouth_box")

    def test_fixture_selection_hash_and_bytes_are_required(self):
        row = {"path": "../../experiments/avatar_diversity_20260927/black_woman/source.mp4", "sha256": "a" * 64, "bytes": 1}
        self.assertEqual(contact.fixture_binding({"files": [row]}, "black_woman", "source.mp4"), row)
        with self.assertRaises(contact.checks.Invalid):
            contact.fixture_binding({"files": [row, row]}, "black_woman", "source.mp4")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.mp4"
            missing = []
            self.assertEqual(contact.checked_file(path, row, missing)["status"], "MISSING")
            self.assertEqual(missing, [str(path.resolve())])
            path.write_bytes(b"x")
            with self.assertRaises(contact.checks.Invalid):
                contact.checked_file(path, row, [])

    def test_exact_rgb_crop_pair_and_png_dimensions(self):
        a = bytes(range(18))  # 3 by 2 RGB
        b = bytes(range(18, 36))
        crop = contact.crop_rgb(a, 3, 2, [1, 0, 3, 2])
        self.assertEqual(crop, a[3:9] + a[12:18])
        pair = contact.pair_rgb([a, b], 3, 2)
        self.assertEqual(pair, a[:9] + b[:9] + a[9:] + b[9:])
        png = contact.png_bytes(6, 2, pair)
        self.assertEqual(png[:8], b"\x89PNG\r\n\x1a\n")
        self.assertEqual(struct.unpack(">II", png[16:24]), (6, 2))
        offset, data = 8, b""
        while offset < len(png):
            size = struct.unpack(">I", png[offset:offset + 4])[0]
            if png[offset + 4:offset + 8] == b"IDAT":
                data += png[offset + 8:offset + 8 + size]
            offset += size + 12
        self.assertEqual(zlib.decompress(data), b"\0" + pair[:18] + b"\0" + pair[18:])

    def test_probe_requires_observed_canonical_timestamp_resolution_coverage(self):
        probe = {"streams": [{"width": 512, "height": 896, "avg_frame_rate": "24/1"}],
                 "frames": [{"best_effort_timestamp_time": f"{i / 24:.6f}"} for i in range(240)]}
        with mock.patch.object(contact.subprocess, "run", return_value=mock.Mock(stdout=json.dumps(probe))):
            self.assertEqual(contact.probe_video("synthetic.mp4")["frames"], 240)
        for mutation in ("size", "timestamp", "count"):
            bad = copy.deepcopy(probe)
            if mutation == "size":
                bad["streams"][0]["height"] = 512
            elif mutation == "timestamp":
                bad["frames"][120]["best_effort_timestamp_time"] = "6.000000"
            else:
                bad["frames"].pop()
            with mock.patch.object(contact.subprocess, "run", return_value=mock.Mock(stdout=json.dumps(bad))):
                with self.assertRaises(contact.checks.Invalid):
                    contact.probe_video("synthetic.mp4")

    def test_no_overwrite_or_inference_during_missing_source_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(contact.checks.Invalid):
                contact.main(["--run", "missing", "--inputs", "missing", "--fixture-root", "missing", "--out", directory])
            with self.assertRaises(contact.checks.Invalid):
                contact.extract({"missing_sources": ["required.mp4"]}, Path(directory))


if __name__ == "__main__":
    unittest.main()
