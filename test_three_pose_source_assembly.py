"""CPU-only selective reroll assembly checks; no LTX or MuseTalk workers."""
import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from character_factory.scripts import assemble_three_pose_sources as assembly
from character_factory.scripts.generate_three_pose_videos import (
    build_generation_graph, generation_fingerprint, normalized_interior_guides,
)
from scripts.motion_transitions import atomic_json, file_hash


class ThreePoseAssemblyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.media_temp = tempfile.TemporaryDirectory()
        cls.media = Path(cls.media_temp.name) / "source.mp4"
        cls.image = Path(cls.media_temp.name) / "portrait.png"
        pixels = np.zeros((832,512,3), dtype=np.uint8)
        cv2.imwrite(str(cls.image), pixels)
        writer = cv2.VideoWriter(str(cls.media), cv2.VideoWriter_fourcc(*"mp4v"), 24, (512,832))
        if not writer.isOpened(): raise RuntimeError("CPU test video writer unavailable")
        for _ in range(9): writer.write(pixels)
        writer.release()
        cls.metadata = assembly._decode_metadata(cls.media)

    @classmethod
    def tearDownClass(cls): cls.media_temp.cleanup()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "assembled"
        self.portrait = self.root / "portrait.png"
        shutil.copy2(self.image, self.portrait)
        self.base = self.root / "graph.json"
        graph = {name: {"inputs": {}} for name in ("pos", "neg", "noise", "image", "empty", "audio",
                                                        "portrait", "end_guide", "concat", "guider", "crop", "save")}
        atomic_json(self.base, graph)
        self.pack = {"pack_id": "approved_test", "poses": {p: {"seed": 1, "frame_count": 9,
                     "positive_prompt": "fixed "+p, "negative_prompt": "zoom", "prompt_source": "fixture"}
                     for p in assembly.POSES}}
        self.approved = self.root / "approved.json"
        atomic_json(self.approved, self.pack)
        self.original = self.make_generation("original", self.pack, assembly.POSES)
        reroll = copy.deepcopy(self.pack)
        reroll["pack_id"] = "idle_seed2"
        reroll["poses"]["idle"]["seed"] = 2
        self.reroll = self.make_generation("reroll", reroll, ("idle",))
        self.selection = {"idle": self.reroll, "talking": self.original, "smiling": self.original}

    def make_generation(self, name, pack, poses):
        directory = self.root / name
        (directory / "graphs").mkdir(parents=True)
        prompt_path = directory / "prompt-pack.json"
        atomic_json(prompt_path, pack)
        guide = directory / "guide-512x832.png"
        shutil.copy2(self.portrait, guide)
        fingerprint = generation_fingerprint(self.portrait, prompt_path, self.base,
                                              guide_fit="center_crop", shared_anchor=True)
        manifest = {"generation_fingerprint": fingerprint, "source_image": str(self.portrait),
                    "source_image_sha256": file_hash(self.portrait), "accepted_graph": str(self.base),
                    "prompt_pack_path": str(prompt_path), "prompt_pack_sha256": file_hash(prompt_path),
                    "prepared_guide": {"sha256": file_hash(guide), "guide_dimensions": [512,832]},
                    "workflow": {"model": "cpu_test", "resolution": [512,832], "fps": 24,
                                 "frames_by_pose": {p:9 for p in poses},
                                 "delivered_frames_by_pose": {p:9 for p in poses}}, "poses": {}}
        for pose in poses:
            video = directory / (pose+".mp4")
            shutil.copy2(self.media, video)
            profile = pack["poses"][pose]
            graph = build_generation_graph(json.loads(self.base.read_text()), profile, "same-image", name+pose, 9)
            atomic_json(directory / "graphs" / (pose+"-generation.json"), graph)
            manifest["poses"][pose] = {"status": "completed", "seed": profile["seed"],
                "positive_prompt": profile["positive_prompt"], "negative_prompt": profile["negative_prompt"],
                "delivery": {**self.metadata, "file": str(video), "sha256": file_hash(video),
                             "first_frame_replaced_with_shared_anchor": True}}
        guides_by_pose = {pose: normalized_interior_guides(pack["poses"][pose], 9) for pose in poses
                          if normalized_interior_guides(pack["poses"][pose], 9)}
        if guides_by_pose:
            manifest["workflow"]["interior_guides_by_pose"] = guides_by_pose
        path = directory / "manifest.json"
        atomic_json(path, manifest)
        return path

    def run_assembly(self):
        return assembly.assemble(self.selection, self.out, self.approved)

    def test_seed_only_reroll_copies_exact_bytes_and_preserves_separate_provenance(self):
        masters = {p: file_hash(p) for p in self.root.rglob("*") if p.is_file()}
        result = self.run_assembly()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["inputs"]["selections"]["idle"]["seed"], 2)
        self.assertEqual(result["inputs"]["selections"]["idle"]["reference_seed"], 1)
        self.assertEqual(result["inputs"]["selections"]["talking"]["generation_manifest"], str(self.original))
        self.assertFalse((self.out / "manifest.json").exists())
        self.assertNotIn("generation_fingerprint", result)
        for pose in assembly.POSES:
            self.assertEqual(file_hash(self.out / (pose+".mp4")), file_hash(self.media))
        self.assertEqual(masters, {p: file_hash(p) for p in masters})
        saved = (self.out / "assembly-provenance.json").read_bytes()
        self.assertEqual(self.run_assembly(), result)
        self.assertEqual((self.out / "assembly-provenance.json").read_bytes(), saved)

    def test_changed_selection_or_corrupt_output_requires_new_directory(self):
        self.run_assembly()
        self.selection["idle"] = self.original
        with self.assertRaisesRegex(ValueError, "Assembly inputs changed"):
            self.run_assembly()
        self.selection["idle"] = self.reroll
        (self.out / "idle.mp4").write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "Assembled output changed"):
            self.run_assembly()

    def test_changed_source_video_or_generation_graph_is_rejected_before_output(self):
        video = self.original.parent / "talking.mp4"
        original = video.read_bytes()
        video.write_bytes(original+b"changed")
        with self.assertRaisesRegex(ValueError, "changed provenance file"):
            self.run_assembly()
        video.write_bytes(original)
        graph_path = self.reroll.parent / "graphs" / "idle-generation.json"
        graph = json.loads(graph_path.read_text())
        graph["portrait"]["inputs"]["strength"] = .5
        atomic_json(graph_path, graph)
        with self.assertRaisesRegex(ValueError, "generation graph differs"):
            self.run_assembly()
        self.assertFalse(self.out.exists())

    def test_different_prompt_or_portrait_cannot_be_mixed(self):
        changed = copy.deepcopy(self.pack)
        changed["poses"]["idle"]["positive_prompt"] = "different breathing instruction"
        self.selection["idle"] = self.make_generation("changed_prompt", changed, ("idle",))
        with self.assertRaisesRegex(ValueError, "positive_prompt differs"):
            self.run_assembly()
        self.selection["idle"] = self.reroll
        manifest = json.loads(self.reroll.read_text())
        alternative = self.root / "other.png"
        cv2.imwrite(str(alternative), np.full((832,512,3), 80, dtype=np.uint8))
        manifest["source_image"] = str(alternative)
        manifest["source_image_sha256"] = file_hash(alternative)
        manifest["generation_fingerprint"]["source_image_sha256"] = file_hash(alternative)
        atomic_json(self.reroll, manifest)
        with self.assertRaisesRegex(ValueError, "different portrait/guide/graph"):
            self.run_assembly()
        self.assertFalse(self.out.exists())

    def test_partial_copy_failure_can_resume_identical_verified_inputs(self):
        original_copy = shutil.copyfile
        calls = []
        def copy(source, destination):
            calls.append(Path(source).name)
            if len(calls) == 2: raise OSError("Injected copy interruption")
            return original_copy(source, destination)
        with patch.object(assembly.shutil, "copyfile", side_effect=copy):
            with self.assertRaisesRegex(OSError, "Injected copy interruption"):
                self.run_assembly()
        before = json.loads((self.out / "assembly-provenance.json").read_text())
        self.assertEqual(before["status"], "assembling")
        result = self.run_assembly()
        self.assertEqual(result["status"], "complete")
        self.assertTrue(all((self.out / (p+".mp4")).is_file() for p in assembly.POSES))

    def select_experimental_idle(self):
        experimental = copy.deepcopy(self.pack)
        experimental.update(pack_id="experimental_interior_idle", approval_status="experimental_requires_review")
        experimental["poses"]["idle"]["interior_guides"] = [{"frame_idx": 4, "strength": .5}]
        self.reference = self.root / "experimental-reference.json"
        atomic_json(self.reference, experimental)
        self.selection["idle"] = self.make_generation("guided_idle", experimental, ("idle",))
        return experimental

    def test_guided_idle_mixes_with_original_other_poses_only_with_matching_reference(self):
        self.select_experimental_idle()
        with self.assertRaisesRegex(ValueError, "generation options differ"):
            self.run_assembly()
        self.assertFalse(self.out.exists())
        result = assembly.assemble(self.selection, self.out, self.reference)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["inputs"]["reference_prompt_pack"]["approval_status"], "experimental_requires_review")
        self.assertNotIn("approved_prompt_pack", result["inputs"])
        self.assertNotIn("approved_seed", result["inputs"]["selections"]["idle"])
        self.assertEqual(result["inputs"]["selections"]["talking"]["generation_manifest"], str(self.original))
        self.assertEqual(result["quality_status"], "requires_per_frame_validation_and_recorded_review")
        self.assertNotIn("interior_guides_by_pose", result["inputs"]["shared_workflow"])
        self.assertEqual(assembly.assemble(self.selection, self.out, self.reference), result)

    def test_missing_wrong_or_unexpected_guide_metadata_is_rejected(self):
        self.select_experimental_idle()
        path = self.selection["idle"]
        original = json.loads(path.read_text())
        for field in (None, {}, {"idle": []}, {"idle": [{"frame_idx": 4, "strength": .4}]},
                      {"idle": [{"frame_idx": 4.0, "strength": .5}]},
                      {"idle": [{"frame_idx": 4, "strength": True}]}, []):
            manifest = copy.deepcopy(original)
            if field is None:
                manifest["workflow"].pop("interior_guides_by_pose")
            else:
                manifest["workflow"]["interior_guides_by_pose"] = field
            atomic_json(path, manifest)
            with self.subTest(metadata=field), self.assertRaisesRegex(ValueError, "interior guide metadata"):
                assembly.assemble(self.selection, self.out, self.reference)
        atomic_json(path, original)
        other = json.loads(self.original.read_text())
        other["workflow"]["interior_guides_by_pose"] = {"talking": [{"frame_idx": 4, "strength": .5}]}
        atomic_json(self.original, other)
        with self.assertRaisesRegex(ValueError, "talking: workflow interior guide metadata"):
            assembly.assemble(self.selection, self.out, self.reference)
        self.assertFalse(self.out.exists())

    def test_interior_guide_graph_parameters_and_connections_must_match_exactly(self):
        self.select_experimental_idle()
        path = self.selection["idle"].parent / "graphs/idle-generation.json"
        original = json.loads(path.read_text())
        added = set(original) - set(json.loads(self.base.read_text()))
        self.assertTrue(added, "Fixture must contain actual interior conditioning nodes")
        for mutation in ("strength", "missing_node", "wrong_chain"):
            graph = copy.deepcopy(original)
            node = sorted(added)[0]
            if mutation == "strength":
                graph[node]["inputs"]["strength"] = .7
            elif mutation == "missing_node":
                del graph[node]
            else:
                graph["concat"]["inputs"]["video_latent"] = ["missing-guide-chain", 2]
            atomic_json(path, graph)
            with self.subTest(mutation=mutation), self.assertRaisesRegex(ValueError, "generation graph differs"):
                assembly.assemble(self.selection, self.out, self.reference)
        self.assertFalse(self.out.exists())

    def test_default_pack_keeps_legacy_provenance_shape(self):
        with patch.object(assembly, "APPROVED_PACK", self.approved):
            result = self.run_assembly()
        reference = result["inputs"]["approved_prompt_pack"]
        self.assertEqual(set(reference), {"path", "sha256", "pack_id"})
        self.assertNotIn("reference_prompt_pack", result["inputs"])
        self.assertEqual(result["inputs"]["selections"]["idle"]["approved_seed"], 1)

    def test_reference_flag_and_legacy_alias_select_same_explicit_pack(self):
        self.select_experimental_idle()
        argv = ["assemble"]
        for pose in assembly.POSES:
            argv += ["--"+pose+"-manifest", str(self.selection[pose])]
        argv += ["--output-dir", str(self.out)]
        with patch("sys.argv", argv + ["--reference-prompt-pack", str(self.reference)]), patch("builtins.print"):
            assembly.main()
        saved = (self.out / "assembly-provenance.json").read_bytes()
        with patch("sys.argv", argv + ["--approved-prompt-pack", str(self.reference)]), patch("builtins.print"):
            assembly.main()
        self.assertEqual((self.out / "assembly-provenance.json").read_bytes(), saved)

    def test_existing_master_directory_cannot_be_used_as_output(self):
        self.out = self.original.parent
        with self.assertRaisesRegex(ValueError, "differ from every generation"):
            self.run_assembly()


if __name__ == "__main__": unittest.main()
