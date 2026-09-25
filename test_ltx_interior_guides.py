"""Portable CPU tests for optional native LTX image guides; no media workers."""
import copy
import json
import tempfile
import types
import unittest
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

from character_factory.scripts import generate_three_pose_videos as render


def base_graph():
    """Small accepted-shape graph whose defaults already match the test input."""
    return {
        "model": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "unchanged-q4"}},
        "text": {"inputs": {"clip_name1": "unchanged-encoder"}},
        "pos": {"inputs": {"text": "unchanged positive"}},
        "neg": {"inputs": {"text": "unchanged negative"}},
        "noise": {"inputs": {"noise_seed": 195}},
        "image": {"inputs": {"image": "portrait.png"}},
        "empty": {"inputs": {"width": 512, "height": 832, "length": 241}},
        "audio": {"inputs": {"frames_number": 241, "frame_rate": 24}},
        "portrait": {"class_type": "LTXVAddGuide", "inputs": {
            "positive": ["fps", 0], "negative": ["fps", 1], "latent": ["empty", 0],
            "image": ["image", 0], "vae": ["vae", 0], "frame_idx": 0, "strength": 1.0}},
        "end_guide": {"class_type": "LTXVAddGuide", "inputs": {
            "positive": ["portrait", 0], "negative": ["portrait", 1],
            "latent": ["portrait", 2], "image": ["image", 0], "vae": ["vae", 0],
            "frame_idx": -1, "strength": 1.0}},
        "concat": {"inputs": {"video_latent": ["end_guide", 2], "audio_latent": ["audio", 0]}},
        "guider": {"inputs": {"positive": ["end_guide", 0], "negative": ["end_guide", 1], "cfg": 1.0}},
        "crop": {"inputs": {"positive": ["end_guide", 0], "negative": ["end_guide", 1], "latent": ["separate", 0]}},
        "save": {"inputs": {"filename_prefix": "fixture_idle", "samples": ["crop", 2]}},
        "sigmas": {"inputs": {"sigmas": "1.0,0.5,0.0"}},
        "vae": {"inputs": {"vae_name": "unchanged-vae"}},
    }


def profile():
    return {"seed": 195, "frame_count": 241, "positive_prompt": "unchanged positive",
            "negative_prompt": "unchanged negative", "prompt_source": "fixture"}


GUIDES = [{"frame_idx": 80, "strength": 1.0}, {"frame_idx": 160, "strength": 1.0}]


class InteriorGuideGraphTest(unittest.TestCase):
    def build(self, value, base=None, frame_count=241):
        return render.build_generation_graph(
            base if base is not None else base_graph(), value, "portrait.png", "fixture_idle", frame_count,
        )

    def test_absent_or_empty_configuration_preserves_accepted_graph_exactly(self):
        for value in (profile(), {**profile(), "interior_guides": []}):
            with self.subTest(value=value):
                base = base_graph()
                self.assertEqual(self.build(value, base), base)
                self.assertEqual(render.normalized_interior_guides(value, 241), [])

    def test_two_guides_form_ordered_conditioning_chain_without_changing_recipe(self):
        base = base_graph()
        before = copy.deepcopy(base)
        value = {**profile(), "interior_guides": copy.deepcopy(GUIDES)}
        original_profile = copy.deepcopy(value)
        graph = self.build(value, base)
        self.assertEqual(base, before)
        self.assertEqual(value, original_profile)
        previous = "portrait"
        for index, guide in enumerate(GUIDES):
            name = f"interior_guide_{index}"
            self.assertEqual(graph[name], {"class_type": "LTXVAddGuide", "inputs": {
                "positive": [previous, 0], "negative": [previous, 1], "latent": [previous, 2],
                "image": ["image", 0], "vae": ["vae", 0], **guide}})
            previous = name
        for field, output in (("positive", 0), ("negative", 1), ("latent", 2)):
            self.assertEqual(graph["end_guide"]["inputs"][field], [previous, output])
        # These five links must all consume the entire guide chain, including
        # crop metadata; otherwise guide frames can escape into delivered video.
        for node, field, output in (("concat", "video_latent", 2),
                ("guider", "positive", 0), ("guider", "negative", 1),
                ("crop", "positive", 0), ("crop", "negative", 1)):
            self.assertEqual(graph[node]["inputs"][field], ["end_guide", output])
        for name in before:
            if name != "end_guide":
                self.assertEqual(graph[name], before[name], name)
        self.assertEqual(graph["end_guide"]["inputs"]["frame_idx"], -1)
        self.assertEqual(graph["end_guide"]["inputs"]["strength"], 1.0)
        self.assertEqual(set(graph) - set(before), {"interior_guide_0", "interior_guide_1"})

    def test_single_image_indices_need_not_be_multiples_of_eight(self):
        value = {**profile(), "interior_guides": [
            {"frame_idx": 1, "strength": 1}, {"frame_idx": 239, "strength": .75}]}
        normalized = render.normalized_interior_guides(value, 241)
        self.assertEqual(normalized, [{"frame_idx": 1, "strength": 1.0},
                                      {"frame_idx": 239, "strength": .75}])
        normalized[0]["frame_idx"] = 8
        self.assertEqual(value["interior_guides"][0]["frame_idx"], 1)
        graph = self.build(value)
        self.assertEqual(graph["interior_guide_1"]["inputs"]["frame_idx"], 239)

    def test_invalid_guide_configuration_never_mutates_inputs(self):
        invalid = [None, {}, "80", [None], [{}], [{"frame_idx": 80}],
                   [{"frame_idx": 80, "strength": 1, "attention_mask": "extra"}],
                   [{"frame_idx": 80, "strength": 1}, {"frame_idx": 80, "strength": .5}]]
        invalid += [[{"frame_idx": index, "strength": 1}] for index in
                    (True, False, -1, 0, 240, 241, 9999, 80.0, "80", None)]
        invalid += [[{"frame_idx": 80, "strength": strength}] for strength in
                    (float("nan"), float("inf"), -float("inf"), 0, -1, True, False, 1.01, "1", None)]
        for guides in invalid:
            with self.subTest(guides=guides):
                value = {**profile(), "interior_guides": guides}
                base = base_graph()
                before = json.dumps({"base": base, "profile": value}, sort_keys=True)
                with self.assertRaises(render.RenderError):
                    render.normalized_interior_guides(value, 241)
                with self.assertRaises(render.RenderError):
                    self.build(value, base)
                self.assertEqual(json.dumps({"base": base, "profile": value}, sort_keys=True), before)

    def test_indices_are_validated_against_this_pose_native_length(self):
        value = {**profile(), "interior_guides": [{"frame_idx": 120, "strength": 1}]}
        self.build(value, frame_count=241)
        with self.assertRaisesRegex(render.RenderError, "endpoint"):
            self.build(value, frame_count=121)

    def test_conflicting_base_guide_node_cannot_be_silently_overwritten(self):
        base = base_graph()
        base["interior_guide_0"] = {"class_type": "OtherNode", "inputs": {"retained": True}}
        before = copy.deepcopy(base)
        with self.assertRaisesRegex(render.RenderError, "already contains"):
            self.build({**profile(), "interior_guides": GUIDES}, base)
        self.assertEqual(base, before)


class InteriorGuideCommandTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.out = self.root / "character"
        self.image = self.root / "portrait.png"
        self.image.write_bytes(b"unmodified portrait")
        self.graph = self.root / "base.json"
        self.graph.write_text(json.dumps(base_graph()))
        self.prompt = self.root / "prompt-pack.json"
        self.pack = {"pack_id": "experimental", "lineage": "accepted fixture",
                     "poses": {name: profile() for name in ("idle", "talking", "smiling")}}
        self.write_pack()
        self.comfy = self.root / "comfy"
        self.comfy.mkdir()
        (self.comfy / "main.py").write_text("")
        self.python = self.root / "python"
        self.python.write_text("")
        self.args = types.SimpleNamespace(image=self.image, output_dir=self.out,
            prompt_pack=self.prompt, poses=["idle"], force=False, dry_run=True,
            shared_anchor=True, guide_fit="center_crop", port=18000, timeout_seconds=1)

    def write_pack(self):
        self.prompt.write_text(json.dumps(self.pack))

    def fingerprint(self):
        return render.generation_fingerprint(self.image, self.prompt, self.graph,
            guide_fit="center_crop", shared_anchor=True)

    @contextmanager
    def main_fixture(self):
        with ExitStack() as stack:
            for name, value in (("ACCEPTED_GRAPH_PATH", self.graph), ("COMFY_ROOT", self.comfy),
                    ("COMFY_PYTHON", self.python), ("GPU_LOCK", self.root / "gpu.lock")):
                stack.enter_context(patch.object(render, name, value))
            stack.enter_context(patch.object(render, "parse_args", return_value=self.args))
            stack.enter_context(patch.object(render.shutil, "which", return_value="fixture-tool"))
            # Any accidental subprocess/network operation fails immediately.
            stack.enter_context(patch.object(render, "run", side_effect=AssertionError("Unexpected media command")))
            stack.enter_context(patch.object(render, "request_json", side_effect=AssertionError("Unexpected network request")))
            yield

    def prepare(self, source, destination, fit):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"same prepared portrait")
        return {"sha256": render.sha256_file(destination), "guide_dimensions": [512, 832]}

    def test_main_rejects_invalid_guides_before_creating_files_or_starting_workers(self):
        for guides in ([{"frame_idx": 0, "strength": 1}],
                       [{"frame_idx": 80, "strength": float("nan")}],
                       [{"frame_idx": 80, "strength": 1, "unsupported": True}]):
            with self.subTest(guides=guides):
                self.pack["poses"]["idle"]["interior_guides"] = guides
                self.write_pack()
                with self.main_fixture(), patch.object(render, "prepare_guide") as prepare, \
                        patch.object(render, "start_server") as start, \
                        patch.object(render, "write_json") as write, \
                        patch.object(render, "generation_fingerprint") as fingerprint:
                    with self.assertRaises(render.RenderError):
                        render.main()
                    for mock in (prepare, start, write, fingerprint):
                        mock.assert_not_called()
                self.assertFalse(self.out.exists())
                self.assertFalse((self.comfy / "input").exists())
                self.assertFalse((self.root / "gpu.lock").exists())

    def test_guide_configuration_changes_fingerprint_and_blocks_resume_even_with_force(self):
        self.out.mkdir()
        original = self.fingerprint()
        manifest = {"generation_fingerprint": original, "poses": {}}
        render.write_json(self.out / "manifest.json", manifest)
        original_bytes = (self.out / "manifest.json").read_bytes()
        self.pack["poses"]["idle"]["interior_guides"] = copy.deepcopy(GUIDES)
        self.write_pack()
        changed = self.fingerprint()
        self.assertNotEqual(original, changed)
        self.assertEqual(original["source_image_sha256"], changed["source_image_sha256"])
        self.assertEqual(original["accepted_graph_sha256"], changed["accepted_graph_sha256"])
        with self.assertRaisesRegex(render.RenderError, "Generation inputs differ"):
            render.verified_resume(self.out, changed)
        self.args.force = True
        with self.main_fixture(), patch.object(render, "prepare_guide") as prepare, \
                patch.object(render, "start_server") as start:
            with self.assertRaisesRegex(render.RenderError, "Generation inputs differ"):
                render.main()
        prepare.assert_not_called()
        start.assert_not_called()
        self.assertEqual((self.out / "manifest.json").read_bytes(), original_bytes)

    def test_default_dry_run_omits_optional_manifest_map(self):
        with self.main_fixture(), patch.object(render, "prepare_guide", side_effect=self.prepare), \
                patch.object(render, "start_server") as start:
            self.assertEqual(render.main(), 0)
        start.assert_not_called()
        manifest = render.load_json(self.out / "manifest.json")
        self.assertNotIn("interior_guides_by_pose", manifest["workflow"])
        graph = render.load_json(self.out / "graphs" / "idle-generation.json")
        self.assertFalse(any(name.startswith("interior_guide_") for name in graph))

    def test_partial_resume_preserves_completed_pose_and_guide_provenance(self):
        self.pack["poses"]["idle"]["interior_guides"] = copy.deepcopy(GUIDES)
        self.write_pack()
        with self.main_fixture(), patch.object(render, "prepare_guide", side_effect=self.prepare):
            self.assertEqual(render.main(), 0)
        manifest = render.load_json(self.out / "manifest.json")
        self.assertEqual(manifest["workflow"]["interior_guides_by_pose"], {"idle": GUIDES})
        # Model an already completed idle and a new request for only talking.
        # Actual resume verification runs against these bytes and fingerprints.
        video = self.out / "idle.mp4"
        video.write_bytes(b"completed idle fixture")
        completed = {"status": "completed", "generation_prompt_id": "original-idle",
                     "delivery": {"sha256": render.sha256_file(video)}}
        manifest["poses"]["idle"] = completed
        render.write_json(self.out / "manifest.json", manifest)
        original_video = video.read_bytes()
        self.args.poses = ["talking"]
        self.args.dry_run = False

        class WorkerBoundary(Exception):
            pass

        # Stop at the first worker boundary after the real manifest is persisted;
        # no network, model imports, generation, or decoder operations are needed.
        with self.main_fixture(), patch.object(render, "prepare_guide", side_effect=self.prepare), \
                patch.object(render, "start_server", side_effect=WorkerBoundary) as start, \
                patch.object(render, "stop_server"):
            with self.assertRaises(WorkerBoundary):
                render.main()
        start.assert_called_once()
        resumed = render.load_json(self.out / "manifest.json")
        self.assertEqual(resumed["workflow"]["interior_guides_by_pose"], {"idle": GUIDES})
        self.assertEqual(resumed["poses"]["idle"], completed)
        self.assertEqual(resumed["workflow"]["frames_by_pose"], {"idle": 241, "talking": 241})
        self.assertEqual(video.read_bytes(), original_video)
        talking_graph = render.load_json(self.out / "graphs" / "talking-generation.json")
        self.assertFalse(any(name.startswith("interior_guide_") for name in talking_graph))
        self.assertEqual(render.verified_resume(self.out, self.fingerprint()), resumed)


if __name__ == "__main__":
    unittest.main()
