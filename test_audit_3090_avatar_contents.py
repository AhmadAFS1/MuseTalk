import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import Mock

from scripts import audit_3090_avatar_contents as audit
from scripts.avatar_s3_store import AvatarS3Store
from test_avatar_s3_store import FakeS3Client


def publication(characters=1):
    result = {"characters": []}
    for index in range(characters):
        character = {"id": "char_%02d" % index, "poses": {}}
        for pose in sorted(audit.POSES):
            avatar = character["id"] + "_" + pose
            character["poses"][pose] = {
                "avatar_id": avatar, "s3_uri": "s3://audit-fixture/avatars/v15/" + avatar + ".tar.gz",
                "bytes": 123, "source_video_sha256": hashlib.sha256(b"source-video").hexdigest(),
                "metadata": {"avatar-id": avatar, "musetalk-version": "v15"},
            }
        result["characters"].append(character)
    return result


class FakeTensor:
    def __init__(self, shape, *, dtype="torch.float16", device="cpu", finite=True):
        self.shape, self.dtype, self.finite = shape, dtype, finite
        self.device = type("Device", (), {"type": device})()

    def split(self, count, dim=0):
        return [self]


class FakeTorch:
    __version__ = "test-double-not-runtime-evidence"

    def __init__(self, value=None):
        self.value = value if value is not None else FakeTensor((2, 1, 8, 32, 32))
        self.calls = []

    def load(self, path, **kwargs):
        self.calls.append(kwargs)
        return self.value

    @staticmethod
    def is_tensor(value):
        return isinstance(value, FakeTensor)

    @staticmethod
    def isfinite(value):
        return type("Finite", (), {"all": lambda self: self, "item": lambda self: value.finite})()


class FakeImage:
    __version__ = "test-double-not-runtime-evidence"
    opened = []

    @classmethod
    def open(cls, path):
        cls.opened.append(path)
        return cls()

    format, mode, width, height = "PNG", "RGB", 640, 360

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def verify(self):
        pass

    def load(self):
        pass


class AuditAvatarContentsTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.addCleanup(self.temporary.cleanup)
        self.item = audit.publication_items(publication(), expected_characters=1)[0]

    def fixture(self):
        source = self.root / "source" / self.item["avatar_id"]
        (source / "full_imgs").mkdir(parents=True)
        (source / "mask").mkdir()
        (source / "avator_info.json").write_text(json.dumps({"avatar_id": self.item["avatar_id"], "version": "v15", "bbox_shift": 0}))
        for name in ("coords.pkl", "mask_coords.pkl", "latents.pt"):
            (source / name).write_bytes(b"untrusted-fixture-never-pickle-loaded")
        (source / "input_video.mp4").write_bytes(b"source-video")
        for index in range(2):
            for folder in ("full_imgs", "mask"):
                (source / folder / ("%08d.png" % index)).write_bytes(b"PNG-test-double")
        client = FakeS3Client()
        store = AvatarS3Store(enabled=True, bucket=self.item["bucket"], prefix=self.item["prefix"],
                              version=self.item["version"], client=client, log_fn=lambda msg: None)
        self.assertTrue(store.upload_avatar_dir(self.item["avatar_id"], source))
        archive = client.objects[(self.item["bucket"], self.item["key"])]["body"]
        self.item["expected_archive_bytes"] = len(archive)
        return client, archive

    def test_complete_coverage_requires_16_by_3(self):
        self.assertEqual(len(audit.publication_items(publication(16))), 48)
        with self.assertRaisesRegex(ValueError, "character_coverage"):
            audit.publication_items(publication())
        manifest = publication(16)
        del manifest["characters"][0]["poses"]["idle"]
        with self.assertRaisesRegex(ValueError, "pose_coverage"):
            audit.publication_items(manifest)

    def test_duplicate_ids_and_wrong_canonical_mapping_rejected(self):
        manifest = publication(16)
        manifest["characters"][1]["id"] = manifest["characters"][0]["id"]
        with self.assertRaisesRegex(ValueError, "character_identity"):
            audit.publication_items(manifest)
        manifest = publication()
        manifest["characters"][0]["poses"]["idle"]["s3_uri"] = "s3://audit-fixture/avatars/v15/../wrong.tar.gz"
        with self.assertRaisesRegex(ValueError, "s3_mapping"):
            audit.publication_items(manifest, 1)

    def test_canonical_restore_hashes_and_cpu_contract(self):
        client, archive = self.fixture()
        tensor_module = FakeTorch()
        result = audit.audit_item(self.item, self.root / "isolated", client, tensor_module, FakeImage)
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["archive"]["sha256"], hashlib.sha256(archive).hexdigest())
        self.assertIsNone(result["archive"]["sha256_matches_publication"])
        self.assertEqual(tensor_module.calls, [{"map_location": "cpu", "weights_only": True}])
        contents = result["contents"]
        self.assertEqual(contents["latents"]["shape"], [2, 1, 8, 32, 32])
        self.assertEqual(contents["render_compatibility_status"], "NOT_TESTED")
        self.assertEqual(contents["visual_review_status"], "NOT_PERFORMED")
        self.assertEqual(contents["provenance"]["encoder_checkpoint_sha256"]["status"], "UNAVAILABLE_IN_CACHE_METADATA")
        self.assertFalse(contents["coordinate_pickles"]["coords.pkl"]["loaded"])
        self.assertEqual(len(contents["visual_contact_samples"]), 2)

    def test_existing_avatar_never_replaced(self):
        client, _ = self.fixture()
        target = self.root / "known" / self.item["avatar_id"]
        target.mkdir(parents=True)
        marker = target / "original.txt"
        marker.write_text("keep")
        client.download_file = Mock(side_effect=AssertionError("download must not start"))
        result = audit.audit_item(self.item, target.parent, client, FakeTorch(), FakeImage)
        self.assertEqual(result["reason"], "refuse_existing_avatar_target")
        self.assertEqual(marker.read_text(), "keep")
        client.download_file.assert_not_called()

    def test_archive_size_mismatch_fails_before_extraction(self):
        client, _ = self.fixture()
        self.item["expected_archive_bytes"] += 1
        result = audit.audit_item(self.item, self.root / "isolated", client, FakeTorch(), FakeImage)
        self.assertNotEqual(result["status"], "PASS")
        self.assertFalse(result["archive"]["size_matches_publication"])
        self.assertFalse((self.root / "isolated" / self.item["avatar_id"]).exists())

    def test_canonical_traversal_rejected(self):
        payload = io.BytesIO()
        with tarfile.open(fileobj=payload, mode="w:gz") as archive:
            member = tarfile.TarInfo(self.item["avatar_id"] + "/../escaped.txt")
            member.size = 3
            archive.addfile(member, io.BytesIO(b"bad"))
        client = FakeS3Client()
        client.objects[(self.item["bucket"], self.item["key"])] = {"body": payload.getvalue()}
        self.item["expected_archive_bytes"] = len(payload.getvalue())
        result = audit.audit_item(self.item, self.root / "isolated", client, FakeTorch(), FakeImage)
        self.assertEqual(result["reason"], "canonical_restore_failed")
        self.assertFalse((self.root / "isolated" / "escaped.txt").exists())

    def test_wrong_published_source_video_hash_fails(self):
        client, _ = self.fixture()
        self.item["expected_source_video_sha256"] = "0" * 64
        result = audit.audit_item(self.item, self.root / "isolated", client, FakeTorch(), FakeImage)
        self.assertEqual(result["reason"], "source_video_hash_mismatch")

    def test_latents_invalid_device_finite_dtype_shape_and_count(self):
        path = self.root / "latents.pt"
        path.write_bytes(b"fixture")
        cases = [
            (FakeTensor((2, 1, 8, 32, 32), device="cuda"), "not_on_cpu"),
            (FakeTensor((2, 1, 8, 32, 32), finite=False), "nonfinite"),
            (FakeTensor((2, 1, 8, 32, 32), dtype="torch.int64"), "dtype"),
            (FakeTensor((2, 1, 4, 32, 32)), "shape"),
            (FakeTensor((3, 1, 8, 32, 32)), "count"),
        ]
        for value, reason in cases:
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                audit.inspect_latents(path, FakeTorch(value), 2)

    def test_list_latents_supported_and_no_unsafe_retry(self):
        path = self.root / "latents.pt"
        path.write_bytes(b"fixture")
        result = audit.inspect_latents(path, FakeTorch([FakeTensor((1, 8, 32, 32))] * 2), 2)
        self.assertEqual(result["storage"], "list")
        module = FakeTorch()
        module.load = Mock(side_effect=TypeError("weights_only unsupported"))
        with self.assertRaises(TypeError):
            audit.inspect_latents(path, module, 2)
        module.load.assert_called_once_with(str(path), map_location="cpu", weights_only=True)

    def test_plan_only_never_imports_cloud_or_tensor_dependencies(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(publication(16)))
        report = audit.run_audit(manifest, self.root / "plan.json", self.root / "new-audit")
        self.assertEqual(report["status"], "PLAN_ONLY")
        self.assertEqual(report["expected_pose_caches"], 48)
        self.assertFalse((self.root / "new-audit").exists())

    def test_existing_root_and_output_rejected_before_imports(self):
        manifest = self.root / "manifest.json"
        manifest.write_text(json.dumps(publication(16)))
        with self.assertRaisesRegex(ValueError, "audit_root_must_not_exist"):
            audit.run_audit(manifest, self.root / "out.json", self.root)
        out = self.root / "out.json"
        out.write_text("keep")
        with self.assertRaisesRegex(ValueError, "output_must_not_exist"):
            audit.run_audit(manifest, out, self.root / "new")
        self.assertEqual(out.read_text(), "keep")

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("PIL"),
                         "real CPU torch/Pillow integration requires runtime dependencies")
    def test_real_cpu_torch_and_png_decode(self):
        import torch
        from PIL import Image
        tensor_path = self.root / "latents.pt"
        torch.save(torch.zeros((2, 1, 8, 32, 32), dtype=torch.float16, device="cpu"), tensor_path)
        self.assertTrue(audit.inspect_latents(tensor_path, torch, 2)["all_finite"])
        png = self.root / "00000000.png"
        Image.new("RGB", (16, 16)).save(png)
        self.assertTrue(audit.inspect_images([png], Image)["all_pngs_decoded"])


if __name__ == "__main__":
    unittest.main()
