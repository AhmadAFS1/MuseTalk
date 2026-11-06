import os
from pathlib import Path
import random
import shutil
import subprocess
import tempfile
import unittest

from repack_oci import (
    CONFIG, MANIFEST, MUSETALK, Member, docker_directory, docker_metadata_fixture, partition,
    read_layout, repack, root_inventory, split_musetalk_git_pack, stage_source, verify_config,
    verify_filesystems, walk, write_json_blob, write_layer, write_layout,
)


class RepackTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def test_hardlinks_stay_in_same_layer_and_directories_last(self):
        files = self.root / "files"
        (files / "a").mkdir(parents=True)
        (files / "z").mkdir()
        (files / "a" / "data").write_bytes(b"one")
        (files / "middle").write_bytes(b"two")
        os.link(files / "a" / "data", files / "z" / "data")
        self.assertEqual([[member.name for member in layer] for layer in partition(walk(files), 4100)],
                         [["a/data", "z/data"], ["middle"], ["z", "a", "."]])
        if hasattr(os, "listxattr"):
            self.assertEqual(set(docker_metadata_fixture(root_inventory(files))), {"a/data", "z/data"})

    def test_refuses_whiteout_in_flattened_tree_and_config_drift(self):
        (self.root / ".wh.removed").touch()
        with self.assertRaisesRegex(ValueError, "reserved OCI whiteout"):
            walk(self.root)
        with self.assertRaisesRegex(ValueError, "configuration changed"):
            verify_config({"config": {"Entrypoint": ["a"]}}, {"config": {"Entrypoint": ["b"]}})

    @unittest.skipUnless(hasattr(os, "listxattr"), "requires Linux filesystem metadata")
    def test_split_preserves_pack_only_unreachable_objects_and_rejects_unknown_files(self):
        repo = self.root / MUSETALK
        repo.mkdir(parents=True)

        def git(*args, **options):
            return subprocess.run(["git", "-C", repo, *args], check=True, stdout=subprocess.PIPE, **options).stdout

        git("init", "-q")
        data = random.Random(17)
        for filename in ("a", "b", "c"):
            (repo / filename).write_bytes(data.randbytes(700_000))
        git("add", "a", "b", "c")
        git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "fixture")
        head = git("rev-parse", "HEAD").decode().strip()
        unreachable = git("hash-object", "-w", "--stdin", input=b"unreferenced test object").strip()
        objects = git("cat-file", "--batch-all-objects", "--batch-check=%(objectname)")
        packdir = repo / ".git" / "objects" / "pack"
        git("pack-objects", "--window=0", str(packdir / "pack"), input=objects)
        pack = next(packdir.glob("*.pack"))
        git("index-pack", "--rev-index", str(pack))
        git("prune-packed")
        keep = pack.with_suffix(".keep")
        keep.touch()
        with self.assertRaisesRegex(ValueError, "unsupported file"):
            split_musetalk_git_pack(self.root, head, pack_limit="1m")
        keep.unlink()
        _, report = split_musetalk_git_pack(self.root, head, pack_limit="1m")
        self.assertGreater(len(list(packdir.glob("*.pack"))), 1)
        self.assertEqual(git("cat-file", "-p", unreachable.decode()), b"unreferenced test object")
        self.assertEqual(report["object_count"], len(objects.splitlines()))
        self.assertTrue(any(path.endswith(".rev") for path in report["old_pack_files_sha256"]))
        self.assertTrue(report["rootfs_equal_outside_pack_representation"])

    @unittest.skipUnless(os.geteuid() == 0 and shutil.which("umoci"),
                         "requires Linux umoci with root privileges")
    def test_linux_whiteouts_metadata_and_links_survive_oci_roundtrip(self):
        base, top = self.root / "base", self.root / "top"
        for path in (base / "a", base / "b", base / "z", top / "b"):
            path.mkdir(parents=True)
        (base / "remove-me").write_text("obsolete")
        (base / "b" / "old").write_text("also obsolete")
        (top / ".wh.remove-me").touch()
        (top / "b" / ".wh..wh..opq").touch()
        (top / "b" / "new").write_text("new")
        data = base / "a" / "data"
        data.write_text("a hardlinked file")
        os.chown(data, 1234, 2345)
        data.chmod(0o750)
        os.setxattr(data, "user.binary", b"\x00\xfftest")
        subprocess.run(["setfacl", "-m", "u:4321:r", data], check=True)
        subprocess.run(["setcap", "cap_net_bind_service=ep", data], check=True)
        os.utime(data, ns=(1680000000123456789, 1680000000123456789))
        os.link(data, base / "z" / "data")
        (base / "symlink").symlink_to("a/data")
        base.chmod(0o751)

        initial = self.root / "initial"
        (initial / "blobs" / "sha256").mkdir(parents=True)
        lower, first = write_layer(initial, base, walk(base))
        top_names = [".wh.remove-me", "b/.wh..wh..opq", "b/new"]
        upper, second = write_layer(initial, top, [Member(name, (top / name).lstat()) for name in top_names])
        config = {"os": "linux", "architecture": "amd64", "rootfs": {"type": "layers", "diff_ids": [first, second]},
                  "config": {"Entrypoint": ["/unused"], "Volumes": {"/b": {}}}}
        manifest = write_json_blob(initial, MANIFEST, {
            "schemaVersion": 2, "mediaType": MANIFEST, "config": write_json_blob(initial, CONFIG, config),
            "layers": [lower, upper],
        })
        write_layout(initial, "initial", manifest)

        original_bundle = self.root / "original-bundle"
        subprocess.run(["umoci", "unpack", "--image", f"{initial}:initial", original_bundle], check=True)
        source = original_bundle / "rootfs"
        self.assertFalse((source / "remove-me").exists())
        self.assertFalse((source / "b" / "old").exists())
        self.assertEqual((source / "b" / "new").read_text(), "new")
        self.assertEqual(len(os.listxattr(source / "a" / "data")), 3)
        self.assertEqual((source / "a" / "data").stat().st_mtime_ns, 1680000000123456789)
        candidate = self.root / "candidate"
        candidate_manifest, layers = repack(source, candidate, config, "candidate", target=4100)
        self.assertGreater(len(layers), 2)
        verify_config(config, read_layout(candidate, "candidate"))
        candidate_bundle = self.root / "candidate-bundle"
        subprocess.run(["umoci", "unpack", "--image", f"{candidate}:candidate", candidate_bundle], check=True)
        verify_filesystems(source, candidate_bundle / "rootfs")

        docker = self.root / "docker"
        docker_digest, config_digest = docker_directory(candidate, candidate_manifest, docker)
        staged = self.root / "staged"
        self.assertEqual(stage_source(docker, staged, docker_digest), read_layout(candidate, "candidate"))
        with self.assertRaisesRegex(ValueError, "differs from approved"):
            stage_source(docker, self.root / "bad", "sha256:" + "0" * 64)

        if shutil.which("skopeo") and shutil.which("docker"):
            archive = self.root / "docker-load.tar"
            image = "lingua-character-repack-fixture:ci"
            subprocess.run(["skopeo", "copy", f"dir:{docker}", f"docker-archive:{archive}:{image}"], check=True)
            subprocess.run(["docker", "load", "--input", archive], check=True)
            self.addCleanup(subprocess.run, ["docker", "image", "rm", image], check=True)
            actual = subprocess.check_output(["docker", "image", "inspect", image, "-f", "{{.Id}}"], text=True)
            self.assertEqual(actual.strip(), config_digest)


if __name__ == "__main__":
    unittest.main()
