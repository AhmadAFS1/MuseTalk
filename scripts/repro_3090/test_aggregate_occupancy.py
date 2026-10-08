import copy
import unittest
from analyze_aggregate_occupancy import analyze


class AggregateOccupancyTests(unittest.TestCase):
    def fixture(self):
        # Synthetic2loops,2880frames/80s;200fixed-bs16jobs,40half-full jobs.
        worker = {"frames": 480, "identity": "synthetic", "idle_frac": .001,
                  "per_frame_ms": {"tracking_ipc_ms": 14., "facemesh_ms": 13., "compose_ms": 8.}}
        return {"schema": "chin_multistream_v1", "args": {"streams": 6, "pack": 16, "decode_split": 8,
                "loops": 2, "encode": False, "compare_accepted": False},
                "backends": {"unet_describe": {"batch": 16}},
                "repeats": [{"repeat": 0, "frames": 2880, "wall_s": 80., "aggregate_fps": 36.,
                    "gpu": {"jobs": 200, "partial_jobs": 40, "chunks": 360, "busy_frac_events": .8},
                    "gpu_thread": {"credit_wait_frac": .1},
                    "per_worker": {str(i): copy.deepcopy(worker) for i in range(6)}}]}

    def test_padding_not_counted_as_completed_output(self):
        row = analyze(self.fixture())[0]
        self.assertEqual(row["full_recipe_fps"], 36.)
        self.assertEqual(row["useful_fixed_bs16_row_fraction"], .9)
        self.assertEqual(row["executed_padding_rows_inferred_from_fixed_batch"], 320)
        self.assertEqual(row["unchanged_400fps_gate"], "FAIL")
        self.assertEqual(row["workers"][0]["tracking_ipc_ms_per_frame"], 14.)

    def test_inconsistent_frames_jobs_workers_or_denominator_rejected(self):
        for field, value in (("frames", 3200), ("wall_s", 59.), ("aggregate_fps", 40.)):
            fixture = self.fixture()
            fixture["repeats"][0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                analyze(fixture)
        fixture = self.fixture()
        fixture["repeats"][0]["gpu"]["partial_jobs"] = 39
        with self.assertRaises(ValueError):
            analyze(fixture)
        fixture = self.fixture()
        fixture["repeats"][0]["per_worker"]["0"]["frames"] = 479
        with self.assertRaises(ValueError):
            analyze(fixture)


if __name__ == "__main__":
    unittest.main()
