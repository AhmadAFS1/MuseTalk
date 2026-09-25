"""CPU regression checks for immutable character rendering and package resume."""
import copy
from contextlib import ExitStack
import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from character_factory.scripts import generate_three_pose_videos as render


class RenderResumeTest(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "character"
        self.out.mkdir()
        self.image = self.root / "portrait.png"
        self.image.write_bytes(b"portrait-one")
        self.prompt = self.root / "prompts.json"
        self.pack = {"pack_id": "approved", "lineage": "accepted",
                     "poses": {p: {"seed": 1, "positive_prompt": "fixed pose", "negative_prompt": "zoom",
                                    "prompt_source": "approved", "frame_count": 9}
                               for p in ("idle", "talking", "smiling")}}
        self.prompt.write_text(json.dumps(self.pack))
        self.graph = self.root / "graph.json"
        self.graph.write_text(json.dumps({"model": {"inputs": {"unet_name": "accepted"}},
                                         "text": {"inputs": {"clip_name1": "accepted"}},
                                         "sigmas": {"inputs": {"sigmas": "1,0"}}}))
        self.comfy = self.root / "comfy"
        (self.comfy / "output").mkdir(parents=True)
        (self.comfy / "main.py").write_text("")
        (self.comfy / "output" / "generated.latent").write_bytes(b"latent")
        self.python = self.root / "python"
        self.python.write_text("")
        self.args = types.SimpleNamespace(image=self.image, output_dir=self.out, prompt_pack=self.prompt,
            guide_fit="center_crop", shared_anchor=True, poses=["idle", "talking", "smiling"],
            force=False, dry_run=False, port=18190, timeout_seconds=60)

    def fingerprint(self):
        return render.generation_fingerprint(self.image, self.prompt, self.graph,
            guide_fit=self.args.guide_fit, shared_anchor=self.args.shared_anchor)

    def completed(self, poses=("idle",)):
        guide = self.out / "guide-512x832.png"
        guide.write_bytes(b"guide")
        value = {"generation_fingerprint": self.fingerprint(), "prepared_guide": {"sha256": render.sha256_file(guide)},
                 "poses": {}}
        for pose in poses:
            path = self.out / (pose+".mp4")
            path.write_bytes(("verified-"+pose).encode())
            value["poses"][pose] = {"status": "completed", "generation_prompt_id": "original-"+pose,
                "delivery": {"sha256": render.sha256_file(path), "decoded_endpoint_rgb_sha256": "anchor"}}
        render.write_json(self.out / "manifest.json", value)
        return value

    def main_patches(self):
        for name, value in (("ACCEPTED_GRAPH_PATH", self.graph), ("COMFY_ROOT", self.comfy),
                            ("COMFY_PYTHON", self.python), ("GPU_LOCK", self.root / "gpu.lock")):
            self.stack.enter_context(patch.object(render, name, value))
        self.stack.enter_context(patch.object(render, "parse_args", return_value=self.args))
        self.stack.enter_context(patch.object(render.shutil, "which", return_value="available"))

    def test_changed_render_inputs_fail_before_any_output_changes_even_with_force(self):
        mutations = (lambda: self.image.write_bytes(b"another face"),
                     lambda: self.prompt.write_text(self.prompt.read_text()+" "),
                     lambda: self.graph.write_text(self.graph.read_text()+" "),
                     lambda: setattr(self.args, "guide_fit", "edge_pad"),
                     lambda: setattr(self.args, "shared_anchor", False))
        self.main_patches()
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                self.completed()
                before = {p.name: p.read_bytes() for p in self.out.iterdir()}
                mutate()
                self.args.force = True
                with patch.object(render, "prepare_guide") as prepare:
                    with self.assertRaisesRegex(render.RenderError, "Generation inputs differ"):
                        render.main()
                    prepare.assert_not_called()
                self.assertEqual(before, {p.name: p.read_bytes() for p in self.out.iterdir()})

    def test_completed_render_resumes_without_writing_or_starting_worker(self):
        self.completed(("idle", "talking", "smiling"))
        self.main_patches()
        before = {p.name: p.read_bytes() for p in self.out.iterdir()}
        with patch.object(render, "prepare_guide") as guide, patch.object(render, "start_server") as start:
            self.assertEqual(render.main(), 0)
        guide.assert_not_called()
        start.assert_not_called()
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.out.iterdir()})

    def test_corrupt_or_untracked_completed_video_is_rejected(self):
        self.completed()
        (self.out / "idle.mp4").write_bytes(b"corrupt")
        with self.assertRaisesRegex(render.RenderError, "missing or corrupt"):
            render.verified_resume(self.out, self.fingerprint())
        self.completed()
        (self.out / "talking.mp4").write_bytes(b"untracked")
        with self.assertRaisesRegex(render.RenderError, "lacks completed provenance"):
            render.verified_resume(self.out, self.fingerprint())

    def test_dry_run_cannot_destroy_completed_provenance(self):
        original = self.completed()
        self.main_patches()
        self.args.dry_run = True
        with self.assertRaisesRegex(render.RenderError, "Dry run cannot replace"):
            render.main()
        self.assertEqual(json.loads((self.out / "manifest.json").read_text()), original)

    def test_partial_resume_preserves_completed_record_and_only_renders_missing_poses(self):
        original = self.completed()
        self.main_patches()

        def prepare(source, destination, fit):
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(b"guide")
            return {"sha256": render.sha256_file(destination)}

        def submit(**kwargs):
            return kwargs["label"], {"outputs": {"save_images": {"images": []}}}, 1

        def package(frames, native, delivery, count, *args, **kwargs):
            delivery.write_bytes(("rendered-"+delivery.stem).encode())
            return {"sha256": render.sha256_file(delivery), "decoded_endpoint_rgb_sha256": "anchor"}

        with patch.object(render, "prepare_guide", side_effect=prepare), \
             patch.object(render, "build_generation_graph", return_value={}), \
             patch.object(render, "build_decode_graph", return_value={}), \
             patch.object(render, "start_server", return_value=(object(), None)), \
             patch.object(render, "stop_server"), \
             patch.object(render, "submit_and_wait", side_effect=submit) as submits, \
             patch.object(render, "history_item", return_value={"filename": "generated.latent"}), \
             patch.object(render, "package_frames", side_effect=package), \
             patch.object(render, "create_contact_sheet"):
            self.assertEqual(render.main(), 0)
        result = json.loads((self.out / "manifest.json").read_text())
        self.assertEqual(result["poses"]["idle"], original["poses"]["idle"])
        self.assertEqual(set(result["poses"]), {"idle", "talking", "smiling"})
        self.assertEqual([c.kwargs["label"] for c in submits.call_args_list],
                         ["talking-generation", "smiling-generation", "talking-decode", "smiling-decode"])
        self.assertTrue(result["delivery_policy"]["rendered_cross_clip_endpoints_exact"])
        render.verified_resume(self.out, self.fingerprint())


class PackageValidationTest(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np
        from character_factory.scripts import build_realtime_character as factory
        from scripts.motion_transitions import IDLE, TALK, SMILE
        self.factory = factory
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.sources = self.root / "sources"
        self.sources.mkdir()
        self.out = self.root / "package"
        self.out.mkdir()
        self.measurements = {"version": 1, "videos": {}}
        self.atlas = {"version": 1, "status": "candidate_requires_recorded_review", "fps": 24,
                      "sources": {}, "edges": {}, "exit_coverage": {}}
        self.hashes = {}
        for pose, name in ((IDLE, "idle"), (TALK, "talking"), (SMILE, "smiling")):
            path = self.sources / (name+".mp4")
            writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 24, (64,96))
            if not writer.isOpened():
                self.fail("Installed OpenCV cannot write the CPU test video")
            for _ in range(5): writer.write(np.zeros((96,64,3), dtype=np.uint8))
            writer.release()
            cap = cv2.VideoCapture(str(path))
            ok, frame = cap.read()
            cap.release()
            self.assertTrue(ok)
            endpoint = hashlib.sha256(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).tobytes()).hexdigest()
            self.hashes[name] = factory.file_hash(path)
            rows = [{"frame": i, "eye_mid_y_change_px": 0, "eye_scale_change_pct": 0,
                     "eye_line_roll_deg": 0, "lip_gap_px": 0} for i in range(5)]
            self.measurements["videos"][name] = {"sha256": self.hashes[name], "fps": 24,
                "frame_count": 5, "first_rgb_sha256": endpoint, "last_rgb_sha256": endpoint, "rows": rows}
            self.atlas["sources"][pose] = {"sha256": self.hashes[name], "path": str(path),
                                          "frame_count": 5, "width": 64, "height": 96}
            self.atlas["exit_coverage"][pose] = {"covered": 5, "total": 5}
            self.atlas["edges"][pose] = {target: [{"target_frame": (i+1)%5, "score": 0, "admissible": True}
                                                     for i in range(5)] for target in (IDLE,TALK,SMILE)}
        self.save_artifacts()

    def save_artifacts(self):
        render.write_json(self.out / "source-measurements.json", self.measurements)
        render.write_json(self.out / "motion-atlas.json", self.atlas)

    def validate(self):
        return self.factory.validate_artifacts(self.sources, self.measurements, self.atlas, self.hashes)

    def run_package(self, extra=None):
        argv = ["build_realtime_character.py", "--source-dir", str(self.sources),
                "--output-dir", str(self.out), "--character-id", "test_character"] + (extra or [])
        with patch("sys.argv", argv), patch.object(self.factory, "run") as run:
            self.factory.main()
        run.assert_not_called()
        return json.loads((self.out / "character.json").read_text())

    def test_cached_measurement_and_atlas_checks_read_actual_media(self):
        self.assertEqual(self.validate(), self.atlas["exit_coverage"])
        for name, mutate in (
            ("hash", lambda: self.measurements["videos"]["talking"].update(sha256="wrong")),
            ("row", lambda: self.measurements["videos"]["talking"]["rows"][1].update(frame=0)),
            ("nan", lambda: self.measurements["videos"]["talking"]["rows"][1].update(lip_gap_px=float("nan"))),
            ("count", lambda: self.measurements["videos"]["talking"].update(frame_count=50)),
            ("fps", lambda: self.measurements["videos"]["talking"].update(fps=20)),
            ("endpoint", lambda: self.measurements["videos"]["talking"].update(last_rgb_sha256="wrong")),
        ):
            with self.subTest(name=name):
                original = copy.deepcopy(self.measurements)
                mutate()
                with self.assertRaises(ValueError): self.validate()
                self.measurements = original

    def test_invalid_routes_or_incomplete_pose_coverage_cannot_publish(self):
        from scripts.motion_transitions import IDLE, TALK
        original = copy.deepcopy(self.atlas)
        mutations = (
            lambda: self.atlas["edges"][TALK][IDLE][2].update(admissible=False),
            lambda: [e.update(admissible=False) for e in self.atlas["edges"][IDLE][TALK]],
            lambda: self.atlas["edges"][IDLE][TALK][2].update(target_frame=500),
            lambda: self.atlas["exit_coverage"][TALK].update(covered=1),
            lambda: self.atlas["sources"][TALK].update(width=128))
        for mutation in mutations:
            self.atlas = copy.deepcopy(original)
            mutation()
            self.save_artifacts()
            with self.assertRaises(ValueError): self.run_package()
            self.assertFalse((self.out / "character.json").exists())

    def test_resume_preserves_preparation_and_accepts_only_review_metadata_changes(self):
        initial = self.run_package()
        initial["prepared"] = {"test_character_idle": {"status": "ready"}}
        initial["review_evidence"] = "received-webrtc.mp4"
        render.write_json(self.out / "character.json", initial)
        self.atlas["status"] = "reviewed"
        self.atlas["review"] = {"recording": "received-webrtc.mp4"}
        self.save_artifacts()
        resumed = self.run_package()
        self.assertEqual(resumed["prepared"], initial["prepared"])
        self.assertEqual(resumed["review_evidence"], initial["review_evidence"])
        self.assertEqual(resumed["status"], "reviewed")
        self.assertEqual(resumed["motion_atlas_content_sha256"], initial["motion_atlas_content_sha256"])
        self.atlas["edges"]["speaking_direct"]["neutral_resting"][0]["target_frame"] = 3
        self.save_artifacts()
        with self.assertRaisesRegex(ValueError, "atlas routes changed"):
            self.run_package()
        self.assertEqual(json.loads((self.out / "character.json").read_text()), resumed)

    def test_packaging_repairs_discovery_record_without_rewriting_atlas(self):
        atlas_path = self.out / "motion-atlas.json"
        original = atlas_path.read_bytes()
        self.run_package()
        registration_path = self.out / "motion-registration.json"
        registration_path.write_text("{interrupted old registration")
        self.run_package()
        registered = json.loads(registration_path.read_text())
        self.assertEqual(registered["atlas_sha256"], self.factory.file_hash(atlas_path))
        self.assertEqual(registered["source_hashes"], {p: v["sha256"] for p,v in self.atlas["sources"].items()})
        self.assertEqual(atlas_path.read_bytes(), original)

    def test_measurement_mutation_after_packaging_requires_new_version(self):
        initial = self.run_package()
        self.measurements["videos"]["talking"]["rows"][0]["lip_gap_px"] = 1
        self.save_artifacts()
        with self.assertRaisesRegex(ValueError, "measurement report changed"):
            self.run_package()
        self.assertEqual(json.loads((self.out / "character.json").read_text()), initial)

    def test_different_character_id_fails_before_any_packaged_files_change(self):
        self.run_package()
        before = {p.name: p.read_bytes() for p in self.out.iterdir()}
        with self.assertRaisesRegex(ValueError, "different character ID"):
            self.run_package(["--character-id", "another_character"])
        self.assertEqual(before, {p.name: p.read_bytes() for p in self.out.iterdir()})


if __name__ == "__main__": unittest.main()
