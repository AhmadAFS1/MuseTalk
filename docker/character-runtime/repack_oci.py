#!/usr/bin/env python3
"""Re-layer an already locked Linux rootfs without rebuilding its runtime."""
import argparse
from dataclasses import dataclass
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile

GIB = 1024 ** 3
TARGET = GIB // 2
MAX_LAYER = 5 * GIB // 4
MAX_IMAGE = 26_000_000_000
BLOCK = 4 * 1024 * 1024
INDEX = "application/vnd.oci.image.index.v1+json"
MANIFEST = "application/vnd.oci.image.manifest.v1+json"
CONFIG = "application/vnd.oci.image.config.v1+json"
LAYER = "application/vnd.oci.image.layer.v1.tar+gzip"
DOCKER_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
DOCKER_CONFIG = "application/vnd.docker.container.image.v1+json"
DOCKER_LAYER = "application/vnd.docker.image.rootfs.diff.tar.gzip"
REF_ANNOTATION = "org.opencontainers.image.ref.name"
MUSETALK = Path("opt/character-runtime/app")


@dataclass(frozen=True)
class Member:
    name: str
    info: os.stat_result

    @property
    def inode(self):
        return self.info.st_dev, self.info.st_ino


def walk(root):
    def visit(name):
        path = root / name
        info = path.lstat()
        yield Member(name, info)
        if stat.S_ISDIR(info.st_mode):
            with os.scandir(path) as entries:
                children = sorted((entry.name for entry in entries), key=os.fsencode)
            for child in children:
                if child.startswith(".wh."):
                    raise ValueError(f"reserved OCI whiteout path in final rootfs: {path / child}")
                yield from visit(child if name == "." else f"{name}/{child}")

    return list(visit("."))


def partition(members, target=TARGET):
    directories, groups = [], {}
    for member in members:
        mode = member.info.st_mode
        if stat.S_ISDIR(mode):
            directories.append(member)
            continue
        if not any(predicate(mode) for predicate in (
            stat.S_ISREG, stat.S_ISLNK, stat.S_ISFIFO, stat.S_ISBLK, stat.S_ISCHR
        )):
            raise ValueError(f"unsupported OCI file type: {member.name}")
        groups.setdefault(member.inode, []).append(member)
    batch, estimate = [], 0
    for group in groups.values():
        info = group[0].info
        # Reserve record padding and metadata overhead for each path.
        cost = len(group) * 4096 + (info.st_size if stat.S_ISREG(info.st_mode) else 0)
        if batch and estimate + cost > target:
            yield batch
            batch, estimate = [], 0
        batch.extend(group)
        estimate += cost
    if batch:
        yield batch
    # Restore metadata after preceding layers implicitly create parent directories.
    yield list(reversed(directories))


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(BLOCK):
            digest.update(block)
    return digest.hexdigest()


def descriptor(media_type, digest, size):
    return {"mediaType": media_type, "digest": f"sha256:{digest}", "size": size}


def blob(layout, digest):
    prefix, sep, name = digest.partition(":")
    if prefix != "sha256" or not sep or len(name) != 64 or any(
        letter not in "0123456789abcdef" for letter in name
    ):
        raise ValueError("invalid OCI SHA-256")
    return layout / "blobs" / "sha256" / name


def read_json_blob(layout, reference):
    data = blob(layout, reference["digest"]).read_bytes()
    if (len(data) != reference["size"] or
            f"sha256:{hashlib.sha256(data).hexdigest()}" != reference["digest"]):
        raise ValueError("OCI metadata blob failed digest/size verification")
    return json.loads(data)


def read_layout(layout, tag):
    index = json.loads((layout / "index.json").read_bytes())
    matches = [entry for entry in index["manifests"]
               if entry.get("annotations", {}).get(REF_ANNOTATION) == tag]
    if len(matches) != 1:
        raise ValueError("expected one matching OCI manifest")
    manifest = read_json_blob(layout, matches[0])
    return read_json_blob(layout, manifest["config"])


def write_json_blob(layout, media_type, value):
    data = json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
    digest = hashlib.sha256(data).hexdigest()
    reference = descriptor(media_type, digest, len(data))
    blob(layout, reference["digest"]).write_bytes(data)
    return reference


def write_layout(layout, tag, manifest):
    (layout / "index.json").write_text(json.dumps({
        "schemaVersion": 2, "mediaType": INDEX,
        "manifests": [{**manifest, "annotations": {REF_ANNOTATION: tag}}],
    }))
    (layout / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}\n')


def stage_source(directory, layout, approved_digest):
    raw = (directory / "manifest.json").read_bytes()
    if f"sha256:{hashlib.sha256(raw).hexdigest()}" != approved_digest:
        raise ValueError("source registry manifest differs from approved digest")
    document = json.loads(raw)
    if document["mediaType"] != DOCKER_MANIFEST or document["schemaVersion"] != 2:
        raise ValueError("approved image must be a Docker schema 2 image")
    (layout / "blobs" / "sha256").mkdir(parents=True)

    def stage(reference, source_type, target_type):
        if reference["mediaType"] != source_type:
            raise ValueError("unsupported source config/layer media type")
        output = blob(layout, reference["digest"])
        original = directory / output.name
        if original.stat().st_size != reference["size"] or sha256_file(original) != output.name:
            raise ValueError("source config/layer blob differs from approved digest")
        os.link(original, output)
        return {**reference, "mediaType": target_type}

    manifest = write_json_blob(layout, MANIFEST, {
        "schemaVersion": 2, "mediaType": MANIFEST,
        "config": stage(document["config"], DOCKER_CONFIG, CONFIG),
        "layers": [stage(layer, DOCKER_LAYER, LAYER) for layer in document["layers"]],
    })
    write_layout(layout, "source", manifest)
    return read_layout(layout, "source")


def docker_directory(layout, manifest, directory):
    candidate = read_json_blob(layout, manifest)
    directory.mkdir()

    def stage(reference, media_type):
        original = blob(layout, reference["digest"])
        os.link(original, directory / original.name)
        return {**reference, "mediaType": media_type}

    config = stage(candidate["config"], DOCKER_CONFIG)
    raw = json.dumps({
        "schemaVersion": 2, "mediaType": DOCKER_MANIFEST, "config": config,
        "layers": [stage(layer, DOCKER_LAYER) for layer in candidate["layers"]],
    }, separators=(",", ":"), sort_keys=True).encode()
    (directory / "manifest.json").write_bytes(raw)
    (directory / "version").write_text("Directory Transport Version: 1.1\n")
    return f"sha256:{hashlib.sha256(raw).hexdigest()}", config["digest"]


def write_layer(layout, root, members, layer_limit=MAX_LAYER):
    destination = layout / "layer.tmp"
    raw = hashlib.sha256()
    with tempfile.TemporaryFile() as names:
        names.write(b"".join(os.fsencode(item.name) + b"\0" for item in members))
        names.seek(0)
        # GNU tar serializes uid/gid, pax mtime, ACLs and xattrs with file/link modes.
        command = [
            "tar", "--create", "--file=-", "--format=pax", "--numeric-owner",
            "--pax-option=exthdr.name=%d/PaxHeaders/%f,delete=atime,delete=ctime",
            "--xattrs", "--xattrs-include=*", "--acls", "--selinux",
            "--no-recursion", "--null", "--files-from=-",
        ]
        with destination.open("wb") as output, subprocess.Popen(
            command, cwd=root, stdin=names, stdout=subprocess.PIPE
        ) as tar:
            try:
                with gzip.GzipFile(filename="", fileobj=output, mtime=0, compresslevel=3) as compressed:
                    while block := tar.stdout.read(BLOCK):
                        raw.update(block)
                        compressed.write(block)
                        if output.tell() > layer_limit:
                            raise ValueError(
                                f"compressed layer exceeded {layer_limit} bytes; "
                                f"largest member: {max(members, key=lambda item: item.info.st_size).name}"
                            )
                if tar.wait():
                    raise RuntimeError("GNU tar failed while packing the verified rootfs")
            finally:
                if tar.poll() is None:
                    tar.kill()
    length = destination.stat().st_size
    if length > layer_limit:
        raise ValueError(f"compressed layer exceeded {layer_limit} bytes")
    digest = sha256_file(destination)
    reference = descriptor(LAYER, digest, length)
    destination.replace(blob(layout, reference["digest"]))
    return reference, f"sha256:{raw.hexdigest()}"


def repack(root, layout, source_config, tag, target=TARGET, layer_limit=MAX_LAYER):
    if (source_config.get("os"), source_config.get("architecture")) != ("linux", "amd64"):
        raise ValueError("only the approved linux/amd64 image may be re-layered")
    (layout / "blobs" / "sha256").mkdir(parents=True)
    layers, diff_ids = [], []
    total = 0
    members = walk(root)
    largest = sorted((member for member in members if stat.S_ISREG(member.info.st_mode)),
                     key=lambda member: member.info.st_size, reverse=True)[:12]
    for member in largest:
        print(f"Largest source file: {member.info.st_size} bytes {member.name!r}", flush=True)
    for batch in partition(members, target):
        layer, diff_id = write_layer(layout, root, batch, layer_limit)
        total += layer["size"]
        if total > MAX_IMAGE:
            raise ValueError("candidate image exceeds disk budget")
        layers.append(layer)
        diff_ids.append(diff_id)
        print(f"Packed layer {len(layers)}: {layer['size']} bytes", flush=True)
    config = {
        **source_config,
        "rootfs": {**source_config["rootfs"], "diff_ids": diff_ids},
        # Preserve original descriptive history while identifying only actual new layers.
        "history": [{**entry, "empty_layer": True} for entry in source_config.get("history", [])]
        + [{"created_by": "Lingua packaging-only re-layer"} for _ in layers],
    }
    manifest = write_json_blob(layout, MANIFEST, {
        "schemaVersion": 2, "mediaType": MANIFEST,
        "config": write_json_blob(layout, CONFIG, config), "layers": layers,
    })
    write_layout(layout, tag, manifest)
    return manifest, layers


def root_inventory(root, members=None):
    seen, rows = {}, {}
    for member in walk(root) if members is None else members:
        path = root / member.name
        info = member.info
        mode = info.st_mode
        first = seen.get(member.inode)
        kind = stat.S_IFMT(mode)
        if stat.S_ISDIR(mode):
            content = None
        elif first is not None:
            content = ("hardlink", first)
        elif stat.S_ISREG(mode):
            content = ("sha256", sha256_file(path))
        elif stat.S_ISLNK(mode):
            content = ("symlink", os.readlink(path))
        elif stat.S_ISFIFO(mode):
            content = ("fifo",)
        elif stat.S_ISBLK(mode) or stat.S_ISCHR(mode):
            content = ("device", os.major(info.st_rdev), os.minor(info.st_rdev))
        else:
            raise ValueError(f"unsupported OCI file type: {member.name}")
        if not stat.S_ISDIR(mode):
            seen.setdefault(member.inode, member.name)
        xattrs = {
            key: hashlib.sha256(os.getxattr(path, key, follow_symlinks=False)).hexdigest()
            for key in os.listxattr(path, follow_symlinks=False)
        }
        rows[member.name] = (
            kind, stat.S_IMODE(mode), info.st_uid, info.st_gid, info.st_mtime_ns, content, xattrs,
        )
    return rows


def verify_inventory(originals, copies):
    if originals.keys() != copies.keys():
        raise ValueError("source and candidate rootfs path sets differ")
    differing = [name for name, original in originals.items() if original != copies[name]]
    if differing:
        raise ValueError(f"rootfs metadata/content mismatch ({len(differing)}): {differing[:12]}")


def verify_filesystems(source, candidate):
    originals = root_inventory(source)
    verify_inventory(originals, root_inventory(candidate))
    return originals


def split_musetalk_git_pack(root, expected_revision, pack_limit="512m"):
    repo = root / MUSETALK
    objects = repo / ".git" / "objects"
    packs = objects / "pack"
    if any(path.is_symlink() or not path.is_dir() for path in (repo / ".git", objects, packs)):
        raise ValueError("pack normalization requires standalone MuseTalk Git storage")
    static_refs = [objects / "info" / filename for filename in ("alternates", "http-alternates", "packs")]
    if any(path.exists() or path.is_symlink() for path in static_refs):
        raise ValueError("pack normalization does not support alternates or static pack references")

    def files(directory, allow_reverse_index=False):
        entries = sorted(directory.iterdir())
        suffixes = "pack|idx|rev" if allow_reverse_index else "pack|idx"
        if not entries or any(
            not re.fullmatch(rf"pack-[0-9a-f]{{40}}\.(?:{suffixes})", entry.name)
            or entry.is_symlink() or not entry.is_file() or entry.stat().st_nlink != 1 for entry in entries
        ):
            raise ValueError("unsupported file in the MuseTalk pack directory")
        stems = {entry.stem for entry in entries if entry.suffix == ".pack"}
        if stems != {
            entry.stem for entry in entries if entry.suffix == ".idx"
        } or not {entry.stem for entry in entries if entry.suffix == ".rev"} <= stems:
            raise ValueError("MuseTalk pack files and all indexes must be paired")
        return entries

    def git(*args, **options):
        return subprocess.run(
            ["git", "-C", str(repo), "-c", "pack.writeReverseIndex=false", *args], check=True,
            stdout=subprocess.PIPE,
            env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_NO_LAZY_FETCH": "1", "GIT_ALLOW_PROTOCOL": ""},
            **options,
        ).stdout

    def identity():
        return (git("rev-parse", "HEAD").strip(), git("for-each-ref", "--format=%(refname) %(objectname)"),
                sha256_file(repo / ".git" / "index"))

    def inventory():
        return sorted(git(
            "cat-file", "--batch-all-objects", "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        ).splitlines())

    original_files = files(packs, allow_reverse_index=True)
    original_sizes = {str(entry.relative_to(root)): entry.stat().st_size for entry in original_files}
    pack_info = packs.stat()
    originals = root_inventory(root)
    source_identity = identity()
    if source_identity[0].decode() != expected_revision:
        raise ValueError("MuseTalk revision differs from approved packing source")
    for entry in original_files:
        if entry.suffix == ".rev":
            git("index-pack", "--verify", "--rev-index", str(entry.with_suffix(".pack")))
    object_rows = inventory()
    if not object_rows or any(
        not re.fullmatch(rb"[a-f0-9]{40} (?:blob|tree|commit|tag) [0-9]+", row) for row in object_rows
    ):
        raise ValueError("Git object inventory is missing or malformed")
    with tempfile.TemporaryDirectory(prefix="repack-git-", dir=root.parent) as temporary:
        temp = Path(temporary)
        git(
            "pack-objects", f"--max-pack-size={pack_limit}", "--window=0", "--threads=1", str(temp / "pack"),
            input=b"\n".join(row.split(b" ", 1)[0] for row in object_rows) + b"\n",
        )
        replacement = files(temp)
        if sum(entry.stat().st_size for entry in replacement) > 5_500_000_000:
            raise ValueError("replacement Git pack representation exceeds scratch budget")
        for entry in original_files:
            entry.unlink()
        for entry in replacement:
            entry.replace(packs / entry.name)
        os.utime(packs, ns=(pack_info.st_atime_ns, pack_info.st_mtime_ns))
    if identity() != source_identity or inventory() != object_rows:
        raise ValueError("MuseTalk Git identity or complete object inventory changed")
    git("fsck", "--full", "--no-dangling")
    normalized = root_inventory(root)
    old_paths = {str(entry.relative_to(root)) for entry in original_files}
    new_paths = {str(entry.relative_to(root)) for entry in files(packs)}
    verify_inventory({key: value for key, value in originals.items() if key not in old_paths},
                     {key: value for key, value in normalized.items() if key not in new_paths})

    def fingerprints(paths, rows):
        return {path: rows[path][-2][1] for path in sorted(paths)}

    report = {
        "old_pack_files_sha256": fingerprints(old_paths, originals),
        "new_pack_files_sha256": fingerprints(new_paths, normalized),
        "old_pack_files_bytes": original_sizes,
        "new_pack_files_bytes": {path: (root / path).stat().st_size for path in sorted(new_paths)},
        "all_object_inventory_sha256": hashlib.sha256(b"\n".join(object_rows) + b"\n").hexdigest(),
        "object_count": len(object_rows), "git_fsck": "passed",
        "head_refs_and_index_equal": True, "rootfs_equal_outside_pack_representation": True,
    }
    return normalized, report


def docker_metadata_fixture(rows):
    selected = set()
    for name, row in rows.items():
        content, xattrs = row[-2:]
        if xattrs:
            selected.add(name)
        if content and content[0] == "hardlink":
            selected.update((name, content[1]))
    return {name: row for name, row in rows.items() if name in selected}


def verify_docker_metadata(fixture):
    expected = json.loads(fixture.read_text())
    root = Path('/')
    names = [Member(name, (root / name).lstat()) for name in expected]
    actual = json.loads(json.dumps(root_inventory(root, names)))
    if actual != expected:
        mismatch = [name for name in expected if expected[name] != actual[name]]
        raise ValueError(f"Docker extraction changed {len(mismatch)} metadata paths: {mismatch[:12]}")
    print(json.dumps({"docker_extraction_metadata_paths": len(expected), "status": "passed"}))


def verify_config(source, candidate):
    def without_packaging(value):
        return {key: entry for key, entry in value.items() if key not in {"rootfs", "history"}}

    if without_packaging(source) != without_packaging(candidate):
        raise ValueError("image configuration changed apart from layer packaging")


def require_space(path, minimum):
    free = shutil.disk_usage(path).free
    print(f"Free disk: {free} bytes; require {minimum}", flush=True)
    if free < minimum:
        raise ValueError("insufficient disk headroom for bounded image repack")


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--verify-docker-metadata":
        verify_docker_metadata(Path(sys.argv[2]))
        return
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--source-digest", required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--docker-fixture", type=Path, required=True)
    parser.add_argument("--split-musetalk-pack", metavar="EXPECTED_COMMIT")
    parser.add_argument("--consume-source", action="store_true", help="Remove only the CI scratch OCI copy")
    args = parser.parse_args()
    if args.consume_source and (
        args.source.name != "source-image" or args.source.is_symlink()
        or not args.source.parent.name.startswith("character-repack.")
        or args.source.parent.is_symlink() or args.source.parent.stat().st_mode & 0o077
        or args.source.parent.resolve() != args.work.parent.resolve()
        or args.source.resolve() == args.work.resolve()
    ):
        parser.error("--consume-source requires the source-image scratch directory beside --work")
    args.work.mkdir()
    staged = args.work / "source-oci"
    source_bundle = args.work / "source-bundle"
    candidate_bundle = args.work / "candidate-bundle"
    output = args.work / "candidate"
    require_space(args.work, 69 * GIB)
    source_config = stage_source(args.source, staged, args.source_digest)
    subprocess.run(["umoci", "unpack", "--image", f"{staged}:source", str(source_bundle)], check=True)
    shutil.rmtree(staged)
    if args.consume_source:
        # The CI scratch copy is disposable after extracting the approved rootfs.
        shutil.rmtree(args.source)
    require_space(args.work, 55 * GIB)
    source_rootfs = source_bundle / "rootfs"
    if args.split_musetalk_pack:
        rows, git_representation = split_musetalk_git_pack(source_rootfs, args.split_musetalk_pack)
        require_space(args.work, 55 * GIB)
    else:
        rows, git_representation = root_inventory(source_rootfs), None
    manifest, layers = repack(source_rootfs, output, source_config, "candidate")
    verify_config(source_config, read_layout(output, "candidate"))
    require_space(args.work, 29 * GIB)
    subprocess.run(["umoci", "unpack", "--image", f"{output}:candidate", str(candidate_bundle)], check=True)
    verify_inventory(rows, root_inventory(candidate_bundle / "rootfs"))
    args.docker_fixture.write_text(json.dumps(docker_metadata_fixture(rows)))
    shutil.rmtree(source_bundle)
    shutil.rmtree(candidate_bundle)
    # Docker load may retain the archive, an extracted copy, and unpacked layers concurrently.
    require_space(args.work, 82 * GIB)
    docker_manifest, config = docker_directory(output, manifest, args.work / "docker-image")
    shutil.rmtree(output)
    args.report.write_text(json.dumps({
        "schema": "lingua-character-runtime-repack/1",
        "local_oci_manifest": manifest["digest"],
        "docker_manifest": docker_manifest,
        "config_digest": config,
        "rootfs_paths_verified": len(rows),
        "normalized_rootfs_and_original_config_equal": True,
        "git_storage_representation": git_representation,
        "layer_count": len(layers),
        "compressed_bytes": sum(layer["size"] for layer in layers),
        "largest_layer_bytes": max(layer["size"] for layer in layers),
        "layer_max_allowed_bytes": MAX_LAYER,
        "gpu_qualification": "not_run",
        "promotion_eligible": False,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
