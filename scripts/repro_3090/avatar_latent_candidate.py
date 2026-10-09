#!/usr/bin/env python3
"""Separate seed-123 canonical preparation experiment; never a warm-FPS claim.

Default is a CPU-only plan. --run invokes the unchanged canonical preparation
twice, under box_guard/watch, into an exclusive new tree. Never replaces a cache.
Only the explicitly listed original-manifest inputs are verified, not all 878.

Canonical path: unchanged source240x512x896 -> DWPose/S3FD boxes (+10 lower
margin), Lanczos256 crops -> native FP16 SD-VAE sampled masked then unmasked,
seed123 once before the frame loop; jaw/cheeks90 masks; original speech through
FP16 Whisper+positional encoding at24fps with2/2 padding. TAESD is a decoder,
not this encoder. Full preparation also regenerates audio, boxes and masks.
Their equality is reported separately before any latents-only interpretation.

The renderer's timed loop excludes this work; changed latent values do not
reduce its fixed tensor sizes or UNet MACs. Keep original cache and quality
bounds. A different new cache requires a separately identified quality/render
experiment, never replacement inputs for the already-frozen engine comparison.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import functools
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import runpy
import signal
import shutil
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
PARENT_SHA = "6d3ab6ef31605c2231605e03042e82361a27112589ef6d7f6f8ff8f4b016eea5"
IDS = ("black_man_short_beard", "black_woman", "east_asian_man_goatee",
       "middle_eastern_man_full_beard", "south_asian_woman", "white_man_clean_shaven")
FIXTURE_FILES = ("source.mp4", "speech.wav", "source_landmarks.npy", "cache.pt", "masks.npz",
                 "preparation.json", "spec.json", "render.json")
MODEL_FILES = ("sd-vae/config.json", "sd-vae/diffusion_pytorch_model.bin", "whisper/config.json",
               "whisper/preprocessor_config.json", "whisper/pytorch_model.bin", "dwpose/dw-ll_ucoco_384.pth",
               "face-parse-bisent/79999_iter.pth", "face-parse-bisent/resnet18-5c106cde.pth",
               "face_detection/s3fd.pth")
PINNED_CODE = ("character_factory/h3_avatar_workflow/prepare_stage.py",
               "character_factory/h3_avatar_workflow/common.py", "musetalk/models/unet.py",
               "musetalk/utils/audio_processor.py", "musetalk/utils/blending.py")
CODE_ROOTS = ("musetalk/models", "musetalk/utils", "character_factory/h3_avatar_workflow")
POLICY_ENV = "MUSETALK_REPRO_PREPARATION_POLICY"


def preparation_policy(fixed_cudnn=False, fixed_geometry=False):
    require(not (fixed_cudnn and fixed_geometry), "select one preparation policy only")
    policy = {"name": "prep_fixed_cudnn_selection_v1" if fixed_cudnn else "canonical_preparation_unmodified_v1",
            "non_historical_candidate": fixed_cudnn,
            "change": "cudnn.benchmark=False after original get_landmark_and_bbox returns" if fixed_cudnn else "none",
            "seed_policy": "unchanged canonical torch.manual_seed(123) before sequential frames",
            "early_seed_added": False, "deterministic_algorithms_changed": False,
            "cublas_workspace_config_added": False, "precision_tf32_attention_changed": False}
    if fixed_geometry:
        policy.update(name="prep_fixed_cudnn_selection_v2", non_historical_candidate=True,
                      change="cudnn.benchmark=False before every original inference_topdown call and after original get_landmark_and_bbox returns",
                      required_geometry_calls=240)
    return policy


def validate_policy_binding(spec, fixed_cudnn, fixed_geometry=False):
    policy = preparation_policy(fixed_cudnn, fixed_geometry)
    require(spec.get("preparation_policy") == policy and os.environ.get(POLICY_ENV) == policy["name"],
            "preparation policy spec/CLI/environment mismatch")
    return policy


def install_fixed_cudnn_wrapper(preprocessing, torch, observations):
    """Only replace the imported detector adapter, preserving its exact result.

    FaceAlignment/S3FD set benchmark=True internally. Resetting before the call
    is insufficient. No detector/model/seed/precision implementation is changed.
    A detector exception remains an exception and never earns a reset receipt.
    """
    original = preprocessing.get_landmark_and_bbox

    @functools.wraps(original)
    def fixed(*args, **kwargs):
        result = original(*args, **kwargs)
        before = bool(torch.backends.cudnn.benchmark)
        torch.backends.cudnn.benchmark = False
        observations.append({"after_original_return": True, "benchmark_before_reset": before,
                             "benchmark_after_reset": bool(torch.backends.cudnn.benchmark)})
        return result

    preprocessing.get_landmark_and_bbox = fixed
    return original


def install_fixed_geometry_wrapper(preprocessing, torch, observations):
    original = preprocessing.inference_topdown

    @functools.wraps(original)
    def fixed(*args, **kwargs):
        row = {"call_index": len(observations), "benchmark_before_reset": bool(torch.backends.cudnn.benchmark),
               "original_returned_successfully": False}
        torch.backends.cudnn.benchmark = False
        row["benchmark_before_original_call"] = bool(torch.backends.cudnn.benchmark)
        observations.append(row)
        result = original(*args, **kwargs)
        row["original_returned_successfully"] = True
        row["benchmark_after_original_return"] = bool(torch.backends.cudnn.benchmark)
        return result

    preprocessing.inference_topdown = fixed
    return original


@contextlib.contextmanager
def fixed_preparation_adapters(preprocessing, torch, post_rows, geometry_rows=None):
    original = install_fixed_cudnn_wrapper(preprocessing, torch, post_rows)
    original_geometry = None
    try:
        if geometry_rows is not None:
            original_geometry = install_fixed_geometry_wrapper(preprocessing, torch, geometry_rows)
        yield
    finally:
        preprocessing.get_landmark_and_bbox = original
        if original_geometry is not None:
            preprocessing.inference_topdown = original_geometry


def validate_policy_receipt(runtime, fixed_cudnn, fixed_geometry=False):
    require(runtime.get("preparation_policy") == preparation_policy(fixed_cudnn, fixed_geometry), "runtime preparation policy mismatch")
    if fixed_cudnn or fixed_geometry:
        rows = runtime.get("post_detector_cudnn_receipts", [])
        require(rows and all(row.get("after_original_return") is True
                             and row.get("benchmark_after_reset") is False for row in rows)
                and runtime.get("cudnn_benchmark_after_preparation") is False,
                "missing or ineffective post-detector cuDNN reset receipt")
    if fixed_geometry:
        rows = runtime.get("geometry_cudnn_receipts", [])
        require(len(rows) == 240 and all(row.get("call_index") == index
                and row.get("benchmark_before_original_call") is False
                and row.get("original_returned_successfully") is True
                and row.get("benchmark_after_original_return") is False for index, row in enumerate(rows)),
                "exactly 240 successful fixed-cuDNN geometry call receipts required")


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def rows_for(manifest):
    rows = {row["path"]: row for row in manifest["files"]}
    require(len(rows) == len(manifest["files"]), "duplicate original-manifest path")
    return rows


def verified(path, row):
    require(Path(path).is_file(), "missing pinned input: " + str(path))
    require(Path(path).stat().st_size == row["bytes"] and sha(path) == row["sha256"],
            "pinned input changed: " + str(path))
    return {"path": str(path), "bytes": row["bytes"], "sha256": row["sha256"]}


def verify_inputs(manifest_path, source_root, repo, identities):
    require(sha(manifest_path) == PARENT_SHA, "not the original frozen quality manifest")
    manifest = json.loads(Path(manifest_path).read_text())
    require(manifest.get("schema") == "repro_3090_inputs_v1" and len(manifest["files"]) == 878,
            "original 878-entry manifest required")
    rows = rows_for(manifest)
    bindings = {}
    for relative in (*PINNED_CODE, *("models/" + name for name in MODEL_FILES)):
        key = "../../../../" + relative
        require(key in rows, "missing canonical preparation pin: " + relative)
        bindings[relative] = verified(repo / relative, rows[key])
    for identity in identities:
        for name in FIXTURE_FILES:
            key = "../../../../../experiments/avatar_diversity_20260927/" + identity + "/" + name
            require(key in rows, "missing canonical source pin: " + key)
            bindings[identity + "/" + name] = verified(source_root / identity / name, rows[key])
        old = json.loads((source_root / identity / "preparation.json").read_text())
        require(old.get("seed") == 123 and old.get("encoder") == "Native MuseTalk FP16 SD-VAE",
                "historical preparation does not declare canonical seed/encoder")
        require(old.get("source_sha256") == bindings[identity + "/source.mp4"]["sha256"]
                and old.get("audio_sha256") == bindings[identity + "/speech.wav"]["sha256"],
                "historical preparation source/audio binding mismatch")
    return bindings


def code_snapshot(repo):
    # Original 878-file manifest did not pin VAE/preprocessing/all detector code.
    # Supplement it honestly: freeze these CURRENT source bytes, never call them
    # historical 4070-proven bytes. Recheck after every child, before comparison.
    paths = set()
    for directory in CODE_ROOTS:
        paths.update((repo / directory).rglob("*.py"))
    paths.update((repo / "scripts/box_guard.sh", HERE / "watch.py", Path(__file__)))
    return {str(path.relative_to(repo)): sha(path) for path in sorted(paths)}


def prepare_paths(source_root, out, repo, identities):
    require(not Path(out).exists() and not Path(out).is_symlink(), "candidate output already exists; never overwrite or resume")
    source_root, out, repo = (Path(p).resolve() for p in (source_root, out, repo))
    require(identities and len(set(identities)) == len(identities) and set(identities) <= set(IDS),
            "select unique canonical identities only")
    require(not out.exists() and not out.is_symlink(), "candidate output already exists; never overwrite or resume")
    require(out != source_root and not out.is_relative_to(source_root) and not source_root.is_relative_to(out),
            "candidate and reference roots must be disjoint")
    require(not out.is_relative_to(repo / "results") and not out.is_relative_to(repo / "models")
            and out != repo and not repo.is_relative_to(out), "never write a default cache/model/repository root")
    require(repo.name == "MuseTalk", "canonical prepare_stage requires workspace/MuseTalk layout")
    return source_root, out, repo


def child_environment(repo, uuid, fixed_cudnn=False, fixed_geometry=False):
    # No ambient recipe, PYTHONPATH, credentials, encoder override or seed knobs.
    values = {key: os.environ[key] for key in ("PATH", "LD_LIBRARY_PATH", "HOME") if key in os.environ}
    values.update(CUDA_VISIBLE_DEVICES=uuid, REPO_ROOT=str(repo), PYTHONDONTWRITEBYTECODE="1",
                  HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="4",
                  MUSETALK_BLEND_FIXED_POINT="1", MUSETALK_BLEND_SHRINK_MASK_BBOX="1",
                  MUSETALK_S3FD_PATH=str(repo / "models/face_detection/s3fd.pth"))
    values[POLICY_ENV] = preparation_policy(fixed_cudnn, fixed_geometry)["name"]
    return values


def gpu_identity(repo, uuid, idle):
    from safe_capture import capture
    raw = capture(["nvidia-smi", "--query-gpu=name,uuid,compute_cap,driver_version", "--format=csv,noheader,nounits"],
                  cwd=repo, stage="nvml_identity", timeout_s=20)
    rows = list(csv.reader(raw.splitlines()))
    require(len(rows) == 1 and len(rows[0]) == 4, "exactly one visible physical GPU required")
    name, actual, capability, driver = [s.strip() for s in rows[0]]
    require(name == "NVIDIA GeForce RTX 3090" and capability == "8.6" and actual == uuid,
            "exact RTX3090 UUID/compute capability mismatch")
    if idle:
        apps = capture(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                       cwd=repo, stage="nvml_workloads", timeout_s=20)
        require(not apps.strip(), "GPU busy: drain owned API/pipeline separately; wrapper stops nothing")
        # box_guard also checks cotenants. Reject a CPU-idle API before acquiring
        # its lease; report only PIDs, never process arguments containing secrets.
        require(not active_pipeline_pids(), "API/pipeline processes present; diagnostic preparation requires idle host")
    return {"name": name, "uuid": actual, "compute_capability": capability, "driver": driver}


def active_pipeline_pids(proc=Path("/proc")):
    tokens = (b"api_server.py", b"chin_multistream_render.py", b"production_pose_audit.py", b"trtexec",
              b"build_unet_stagewise.py", b"validate_unet_backend.py", b"build_unet_multi_avatar_corpus.py",
              b"benchmark_pipeline.py", b"prepare_stage.py")
    result = []
    for path in proc.glob("[0-9]*/cmdline"):
        try:
            args = path.read_bytes().split(b"\0")
        except FileNotFoundError:
            continue
        if int(path.parent.name) != os.getpid() and any(arg.rsplit(b"/", 1)[-1] in tokens for arg in args):
            result.append(int(path.parent.name))
    return sorted(result)


def host_identity():
    return {"hostname": socket.gethostname(), "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip()}


def stage_repeat(out, repeat, identity, reference, workspace, bindings, fixed_cudnn=False, fixed_geometry=False):
    target = out / repeat / identity
    target.mkdir(parents=True, exist_ok=False)
    for name in ("source.mp4", "speech.wav", "source_landmarks.npy"):
        source = reference / identity / name
        verified(source, bindings[identity + "/" + name])
        shutil.copyfile(source, target / name)
        verified(target / name, bindings[identity + "/" + name])
    spec = {"id": identity, "workspace": str(workspace), "output": str(target),
            "experiment": "canonical_native_fp16_seed123_repeat_not_accepted",
            "original_spec_sha256": bindings[identity + "/spec.json"]["sha256"], "preparation_seed": 123,
            "historical_preparation_versions": {key: json.loads((reference / identity / "preparation.json").read_text()).get(key)
                                                for key in ("torch", "numpy", "opencv")}}
    spec["preparation_policy"] = preparation_policy(fixed_cudnn, fixed_geometry)
    if fixed_cudnn or fixed_geometry:
        spec["experiment"] = spec["preparation_policy"]["name"] + "_native_fp16_seed123_repeat_not_accepted"
    spec["signature"] = hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    write_new(target / "spec.json", spec)
    return target


def array_comparison(left, right):
    import numpy as np
    a, b = np.asarray(left), np.asarray(right)
    result = {"a_shape": list(a.shape), "b_shape": list(b.shape), "a_dtype": str(a.dtype), "b_dtype": str(b.dtype),
              "a_values_sha256": hashlib.sha256(a.tobytes()).hexdigest(),
              "b_values_sha256": hashlib.sha256(b.tobytes()).hexdigest()}
    require(a.dtype.kind in "fiu" and b.dtype.kind in "fiu" and a.size and b.size, "invalid numeric cache array")
    require(np.isfinite(a).all() and np.isfinite(b).all(), "nonfinite cache array")
    result["exact"] = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
    if a.shape == b.shape:
        delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
        result.update(max_abs=float(delta.max()), mean_abs=float(delta.mean()), changed_values=int(np.count_nonzero(delta)))
    return result


def compare_caches(left, right):
    import numpy as np
    import torch
    # Reference pickle was SHA-checked against immutable canonical data; new
    # pickle is exclusively our pinned canonical worker's hash-checked output.
    a = torch.load(left / "cache.pt", map_location="cpu", weights_only=False)
    b = torch.load(right / "cache.pt", map_location="cpu", weights_only=False)
    require(set(a) == set(b) == {"latents", "audio", "boxes", "cropboxes"}, "unknown cache fields")
    for value in (a, b):
        require(tuple(value["latents"].shape) == (240, 8, 32, 32) and str(value["latents"].dtype) == "torch.float16",
                "canonical FP16 latent shape/dtype mismatch")
    result = {}
    for name in ("latents", "audio", "boxes", "cropboxes"):
        def array(value):
            return value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        result[name] = array_comparison(array(a[name]), array(b[name]))
    with np.load(left / "masks.npz", allow_pickle=False) as ma, np.load(right / "masks.npz", allow_pickle=False) as mb:
        require(set(ma.files) == set(mb.files) == {str(i) for i in range(240)}, "mask frame coverage mismatch")
        result["masks"] = {str(i): array_comparison(ma[str(i)], mb[str(i)]) for i in range(240)}
    result["all_exact"] = all(result[n]["exact"] for n in ("latents", "audio", "boxes", "cropboxes")) and all(
        row["exact"] for row in result["masks"].values())
    result["nonlatent_components_exact"] = all(result[n]["exact"] for n in ("audio", "boxes", "cropboxes")) and all(
        row["exact"] for row in result["masks"].values())
    return result


def check_prepared(path, bindings, identity):
    record = json.loads((path / "preparation.json").read_text())
    require(record.get("status") == "complete" and record.get("seed") == 123 and record.get("frames") == 240
            and record.get("encoder") == "Native MuseTalk FP16 SD-VAE", "canonical preparation incomplete")
    require(record["source_sha256"] == bindings[identity + "/source.mp4"]["sha256"]
            and record["audio_sha256"] == bindings[identity + "/speech.wav"]["sha256"], "prepared source changed")
    for name in ("cache.pt", "masks.npz"):
        require(record["artifacts"].get(str((path / name).resolve())) == sha(path / name), "prepared output hash mismatch")
    return record


def worker(spec_path, expected_uuid, fixed_cudnn=False, fixed_geometry=False):
    # This internal entry point may never initialize CUDA outside guard+watch.
    from watch import descends_from
    holder = Path("/workspace/.gpu_lease.holder")
    session = os.getsid(0)
    require(holder.is_file() and session != os.getpid() and descends_from(os.getpid(), session)
            and b"watch.py" in Path(f"/proc/{session}/cmdline").read_bytes(), "worker must be owned by box_guard/watch")
    holder_pid = re.search(r"(?:^|\s)pid=(\d+)(?:\s|$)", holder.read_text())
    require(holder_pid and descends_from(session, int(holder_pid[1])), "watch session not descended from lease holder")
    spec = json.loads(spec_path.read_text())
    policy = validate_policy_binding(spec, fixed_cudnn, fixed_geometry)
    repo = Path(spec["workspace"]) / "MuseTalk"
    gpu = gpu_identity(repo, expected_uuid, False)
    import torch
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1 and torch.__version__ == "2.5.1+cu121",
            "one CUDA device and canonical torch runtime required")
    require(torch.cuda.get_device_name(0) == gpu["name"] and os.environ.get("CUDA_VISIBLE_DEVICES") == expected_uuid,
            "CUDA/NVML binding mismatch")
    import cv2
    import numpy as np
    packages = {name: importlib.metadata.version(name) for name in ("torch", "numpy", "diffusers", "transformers")}
    for name in ("torchvision", "mmcv", "mmengine", "mmpose", "safetensors", "huggingface-hub", "Pillow"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "NOT_INSTALLED_UNDER_THIS_DISTRIBUTION_NAME"
    packages["opencv_import"] = cv2.__version__
    runtime = {"host": host_identity(), "gpu": gpu, "packages": packages,
               "cuda": torch.version.cuda, "interpreter": sys.executable,
               "seed_policy": "canonical torch.manual_seed(123) before sequential frames",
               "preparation_policy": policy}
    runtime_path = Path(spec["output"]) / "runtime.json"
    if not (fixed_cudnn or fixed_geometry):
        write_new(runtime_path, runtime)
    require({"torch": torch.__version__, "numpy": np.__version__, "opencv": cv2.__version__}
            == spec["historical_preparation_versions"], "preparation runtime differs from historical recorded versions")
    sys.path.insert(0, str(repo / "character_factory/h3_avatar_workflow"))
    sys.argv = [str(repo / "character_factory/h3_avatar_workflow/prepare_stage.py"), "--spec", str(spec_path)]
    if not (fixed_cudnn or fixed_geometry):
        runpy.run_path(sys.argv[0], run_name="__main__")
        return
    # Mirror canonical main's cwd/search path and first three imports in order,
    # under its inference context, before replacing only the detector adapter.
    # The unchanged main then imports the same cached modules. No model object,
    # seed, thread count or precision flag is changed here.
    os.chdir(repo)
    sys.path[:0] = [str(repo), str(repo / "scripts"), str(repo / "musetalk/utils")]
    observations = runtime["post_detector_cudnn_receipts"] = []
    geometry_rows = None
    if fixed_geometry:
        geometry_rows = runtime["geometry_cudnn_receipts"] = []
    try:
        with torch.inference_mode():
            from musetalk.models.vae import VAE  # noqa: F401
            from musetalk.models.unet import PositionalEncoding  # noqa: F401
            import musetalk.utils.preprocessing as preprocessing
        with fixed_preparation_adapters(preprocessing, torch, observations, geometry_rows):
            runpy.run_path(sys.argv[0], run_name="__main__")
        runtime["cudnn_benchmark_after_preparation"] = bool(torch.backends.cudnn.benchmark)
        validate_policy_receipt(runtime, fixed_cudnn, fixed_geometry)
    finally:
        runtime["cudnn_benchmark_after_preparation"] = bool(torch.backends.cudnn.benchmark)
        write_new(runtime_path, runtime)


def guarded_command(target, identity, repeat, uuid, fixed_cudnn=False, fixed_geometry=False):
    preparation_policy(fixed_cudnn, fixed_geometry)
    command = ["bash", "scripts/box_guard.sh", "run", "--wait-min", "0", "--min-avail-gb", "14",
            "--label", "latent_candidate_" + identity + "_" + repeat, "--", sys.executable,
            str(HERE / "watch.py"), "--out", str(target / "gpu_watch.jsonl"), "--", sys.executable,
            str(Path(__file__).resolve()), "--worker-spec", str(target / "spec.json"),
            "--expected-gpu-uuid", uuid]
    return command + (["--fixed-cudnn-preparation"] if fixed_cudnn else
                      ["--fixed-cudnn-geometry"] if fixed_geometry else [])


def allocation_deadline(value, hostname, now=None):
    """Optional exact-host UTC bound; never extends a rental or changes math."""
    require(bool(value) == bool(hostname), "deadline and exact hostname required together")
    if not value:
        return None
    end = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(end.tzinfo is not None and end.utcoffset() == dt.timedelta(0), "explicit UTC deadline required")
    require(socket.gethostname() == hostname, "wrong exact preparation host")
    require((end - (now or dt.datetime.now(dt.timezone.utc))).total_seconds() > 600,
            "preparation requires 600-second cleanup margin")
    return end


def run_guarded_child(command, repo, env, log, deadline=None):
    """Terminate the owned guard so its trap reaps its isolated GPU group."""
    timeout = None
    if deadline is not None:
        remaining = (deadline - dt.datetime.now(dt.timezone.utc)).total_seconds() - 120
        require(remaining > 0, "allocation cleanup cutoff reached")
        timeout = min(600, remaining)
    process = subprocess.Popen(command, cwd=repo, env=env, stdout=log,
                               stderr=subprocess.STDOUT, start_new_session=True)
    previous = signal.getsignal(signal.SIGTERM)

    def interrupted(signum, frame):
        raise InterruptedError("preparation interrupted; stopping owned guard")

    signal.signal(signal.SIGTERM, interrupted)
    try:
        print(json.dumps({"status": "RUNNING_PREPARATION_CHILD", "guard_pid": process.pid,
                          "timeout_s": timeout, "command": command}), flush=True)
        return process.wait(timeout=timeout)
    except BaseException:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                # Canonical guard's TERM trap has first received its cleanup
                # opportunity. Never target an operator/foreign process group.
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=10)
        raise
    finally:
        signal.signal(signal.SIGTERM, previous)


def execute(a, bindings, snapshot):
    source, out, repo = a.source_root, a.out, ROOT
    require(sys.platform == "linux", "GPU execution is Linux-only")
    host = host_identity()
    gpu = gpu_identity(repo, a.expected_gpu_uuid, True)
    deadline = allocation_deadline(a.deadline_utc, a.expected_hostname)
    out.mkdir(parents=True, exist_ok=False)
    report = {"schema": "canonical_avatar_latent_candidate_v1", "status": "INVALID_INCOMPLETE",
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "host": host, "gpu": gpu,
              "parent_manifest_sha256": PARENT_SHA, "validation_scope": "listed preparation/source inputs only; NOT full-original-878 PASS",
              "validated_inputs": bindings, "current_code_sha256": snapshot,
              "preparation_policy": preparation_policy(a.fixed_cudnn_preparation, a.fixed_cudnn_geometry),
              "resource_deadline_utc": deadline.isoformat() if deadline else None,
              "supplemental_current_code_is_historically_proven": False, "children": [], "identities": {},
              "visual_acceptance": "NOT_PERFORMED", "quality_acceptance": "NOT_EVALUATED",
              "warm_fps_speedup_claim": False, "frozen_quality_bounds_changed": False, "production_modified": False}
    write_new(out / "plan.json", report)
    try:
        for identity in a.identities:
            paths = []
            for repeat in ("repeat1", "repeat2"):
                if deadline is not None:
                    allocation_deadline(a.deadline_utc, a.expected_hostname)
                require(host_identity() == host and gpu_identity(repo, a.expected_gpu_uuid, True) == gpu, "host/GPU changed")
                require(code_snapshot(repo) == snapshot, "preparation code changed during experiment")
                target = stage_repeat(out, repeat, identity, source, repo.parent, bindings,
                                      a.fixed_cudnn_preparation, a.fixed_cudnn_geometry)
                command = guarded_command(target, identity, repeat, a.expected_gpu_uuid,
                                          a.fixed_cudnn_preparation, a.fixed_cudnn_geometry)
                with (target / "prepare.log").open("x") as log:
                    rc = run_guarded_child(command, repo, child_environment(repo, a.expected_gpu_uuid,
                                           a.fixed_cudnn_preparation, a.fixed_cudnn_geometry), log, deadline)
                report["children"].append({"identity": identity, "repeat": repeat, "returncode": rc, "path": str(target)})
                require(rc == 0, "canonical preparation child failed; inspect retained log, do not resume")
                require(host_identity() == host and gpu_identity(repo, a.expected_gpu_uuid, True) == gpu, "host/GPU changed")
                require(code_snapshot(repo) == snapshot, "preparation code changed during child")
                verify_inputs(a.inputs, source, repo, a.identities)
                check_prepared(target, bindings, identity)
                paths.append(target)
            runtimes = [json.loads((path / "runtime.json").read_text()) for path in paths]
            for runtime in runtimes:
                validate_policy_receipt(runtime, a.fixed_cudnn_preparation, a.fixed_cudnn_geometry)
            require(runtimes[0] == runtimes[1] and runtimes[0]["host"] == host and runtimes[0]["gpu"] == gpu,
                    "same-host repeat runtime changed")
            repeat_cmp = compare_caches(*paths)
            historical = compare_caches(source / identity, paths[0])
            report["identities"][identity] = {"same_host_repeat": repeat_cmp, "historical_vs_repeat1": historical,
                 "repeat_status": "PASS_EXACT" if repeat_cmp["all_exact"] else "FAIL_REPEAT_CHANGED",
                 "historical_status": "EXACT" if historical["all_exact"] else "DIFFERENT_NEW_CANDIDATE",
                 "latents_only_vs_historical": historical["nonlatent_components_exact"],
                 "runtime": runtimes[0],
                 "preparation_records": [str(p / "preparation.json") for p in paths]}
        report["status"] = "PASS_REPEAT_EXACT_DIAGNOSTIC_ONLY" if all(
            row["repeat_status"] == "PASS_EXACT" for row in report["identities"].values()) else "FAIL_REPEAT_CHANGED"
    except Exception as exc:
        report["failure_type"] = type(exc).__name__
        raise
    finally:
        report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        write_new(out / "comparison.json", report)
    return 0 if report["status"] == "PASS_REPEAT_EXACT_DIAGNOSTIC_ONLY" else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inputs", type=Path)
    p.add_argument("--source-root", type=Path, default=Path("/workspace/experiments/avatar_diversity_20260927"))
    p.add_argument("--out", type=Path)
    p.add_argument("--identity", action="append", choices=IDS)
    p.add_argument("--expected-gpu-uuid", required=True)
    p.add_argument("--run", action="store_true")
    p.add_argument("--deadline-utc", help="optional existing allocation expiry, paired with exact hostname")
    p.add_argument("--expected-hostname", help="optional exact owned host, paired with UTC deadline")
    policies = p.add_mutually_exclusive_group()
    policies.add_argument("--fixed-cudnn-preparation", action="store_true",
                   help="non-historical single-setting candidate: reset cuDNN benchmark=False after canonical detection")
    policies.add_argument("--fixed-cudnn-geometry", action="store_true",
                   help="non-historical v2: additionally reset cuDNN benchmark=False before each original DWPose inference")
    p.add_argument("--worker-spec", type=Path, help=argparse.SUPPRESS)
    a = p.parse_args(argv)
    require(re.fullmatch(r"GPU-[0-9a-fA-F]{8}(?:-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}", a.expected_gpu_uuid),
            "explicit physical GPU UUID required")
    if a.worker_spec:
        require(not a.run, "internal worker mode only")
        worker(a.worker_spec.resolve(), a.expected_gpu_uuid, a.fixed_cudnn_preparation, a.fixed_cudnn_geometry)
        return 0
    require(a.inputs is not None and a.out is not None, "--inputs and --out required")
    deadline = allocation_deadline(a.deadline_utc, a.expected_hostname)
    a.identities = a.identity or list(IDS)
    a.source_root, a.out, _ = prepare_paths(a.source_root, a.out, ROOT, a.identities)
    bindings = verify_inputs(a.inputs, a.source_root, ROOT, a.identities)
    snapshot = code_snapshot(ROOT)
    if not a.run:
        print(json.dumps({"status": "PLAN_ONLY", "identities": a.identities, "repeats": 2, "seed": 123,
                          "out": str(a.out), "validated_input_count": len(bindings), "full_original_878_verified": False,
                          "supplemental_current_code_count": len(snapshot), "gpu_execution": False,
                          "preparation_policy": preparation_policy(a.fixed_cudnn_preparation, a.fixed_cudnn_geometry),
                          "resource_deadline_utc": deadline.isoformat() if deadline else None,
                          "warm_fps_speedup_claim": False, "next": "append --run after owned API/pipeline is idle"}, indent=2))
        return 0
    return execute(a, bindings, snapshot)


if __name__ == "__main__":
    raise SystemExit(main())
