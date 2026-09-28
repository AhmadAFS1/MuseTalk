"""Gate: avatar memory layouts compose bit-identically to today's code on real avatars.

Baseline is scripts/api_avatar.py at BASE_COMMIT (today's code, default
layout). Each candidate is the working-tree api_avatar.py under a flag set.
For every avatar, every cycle position 0..N-1 plus wrap/mirror/negative
indices is composed three ways (plain, return_layers, alternate background)
with a deterministic per-position face, and every output array is compared by
SHA-256. Also checked on the whole prepared library: every mask's plane 0 from
the single-channel reader equals plane 0 of cv2.imread, and the 3 planes are
identical. Mutation safety is checked on the PNG store. CPU only (CUDA hidden).

    CUDA_VISIBLE_DEVICES= python verify_layout_bitexact.py [--quick]
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path("/workspace/MuseTalk")
HERE = Path(__file__).resolve().parent
os.chdir(ROOT)
sys.path[:0] = [str(HERE), str(ROOT), str(ROOT / "scripts")]

import memguard  # noqa: E402

memguard.set_oom_score_adj(1000)
WATCHDOG = memguard.Watchdog(kill_below_gb=3.5).start()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

cv2.setNumThreads(1)
memguard.force_cpu_torch_load()
from musetalk.utils import blending  # noqa: E402

BASE_COMMIT = "1564568"
FLAGS = (
    "MUSETALK_AVATAR_MASK_CHANNELS", "MUSETALK_AVATAR_MASK_STORE", "MUSETALK_AVATAR_FRAME_STORE",
    "MUSETALK_AVATAR_DECODED_LRU_FRAMES", "MUSETALK_AVATAR_PNG_READAHEAD", "MUSETALK_AVATAR_PLAN_FLOAT_ALPHA",
    "MUSETALK_CHEEK_ONLY_BLEND", "MUSETALK_FIXED_FACE_HEIGHT", "MUSETALK_FIXED_FACE_HEIGHT_AVATAR_IDS",
    "MUSETALK_SOURCE_MOUTH_BLEND", "MUSETALK_SOURCE_MOUTH_BLEND_AVATAR_IDS", "MUSETALK_SIDE_JAW_BLEND",
    "MUSETALK_SIDE_JAW_BLEND_AVATAR_IDS",
)
NEW_ALL = {"MUSETALK_AVATAR_FRAME_STORE": "png", "MUSETALK_AVATAR_MASK_STORE": "png",
           "MUSETALK_AVATAR_MASK_CHANNELS": "1", "MUSETALK_AVATAR_PLAN_FLOAT_ALPHA": "0"}
CONFIGS = {
    "new_default": {},
    "new_all": NEW_ALL,
    "mask1_decoded": {"MUSETALK_AVATAR_MASK_CHANNELS": "1"},
    "framepng_inline_lru1": {"MUSETALK_AVATAR_FRAME_STORE": "png", "MUSETALK_AVATAR_PNG_READAHEAD": "0",
                             "MUSETALK_AVATAR_DECODED_LRU_FRAMES": "1"},
}
# (avatar_id, extra env, configs to run, label)
AVATARS = [
    ("chinese_bob_pink_bedroom_talking_3373c10448", {}, "all", "standard"),
    ("chinese_bob_pink_bedroom_idle_d4b06da317", {}, ("new_default", "new_all"), "standard"),
    ("chinese_bob_pink_bedroom_smiling_dae31eb56e", {}, ("new_default", "new_all"), "standard"),
    ("chinese_bob_pink_bedroom_talking_3373c10448", {"MUSETALK_CHEEK_ONLY_BLEND": "1"},
     ("new_default", "new_all"), "cheek_only"),
    ("indian_realtime_talking_20f9845543", {}, "all", "standard"),
    ("japanese_realtime_talking_7d94520b7f", {}, ("new_default", "new_all"), "standard"),
    ("latina_guided_20260925_talking_84c5bc80b8", {}, "all", "standard"),
    ("japanese_baddie_ltx23_talking_v1", {}, ("new_default", "new_all"), "standard"),
    ("latina_baddie_ltx23_smiling_v1", {}, ("new_default", "new_all"), "standard"),
    ("japanese_relay64_talking_preview_20260925", {}, ("new_default", "new_all"), "standard"),
    ("codex_smoke", {}, ("new_default", "new_all"), "standard"),
    ("shared_indian_20260915", {}, ("new_default", "new_all"), "standard"),
    ("japanese_realtime_talking_7d94520b7f_fh1", {"MUSETALK_FIXED_FACE_HEIGHT": "1"},
     ("new_default", "new_all"), "fixed_face_height"),
    ("japanese_realtime_talking_7d94520b7f_fh1_lmix1",
     {"MUSETALK_FIXED_FACE_HEIGHT": "1", "MUSETALK_SOURCE_MOUTH_BLEND": "1"},
     ("new_default", "new_all"), "source_mouth_blend"),
    ("latina_guided_20260925_talking_84c5bc80b8_fh1_lmixx38y22",
     {"MUSETALK_FIXED_FACE_HEIGHT": "1", "MUSETALK_SIDE_JAW_BLEND": "1"},
     ("new_default", "new_all"), "side_jaw_blend"),
]
QUICK = {"chinese_bob_pink_bedroom_talking_3373c10448", "indian_realtime_talking_20f9845543",
         "latina_guided_20260925_talking_84c5bc80b8"}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def baseline_module():
    source = subprocess.run(["git", "show", f"{BASE_COMMIT}:scripts/api_avatar.py"], cwd=ROOT,
                            check=True, capture_output=True, text=True).stdout
    path = Path(os.environ.get("TMPDIR", "/tmp")) / f"api_avatar_{BASE_COMMIT}.py"
    path.write_text(source)
    return load_module(f"api_avatar_{BASE_COMMIT}", path), hashlib.sha256(source.encode()).hexdigest()


def set_env(env: dict) -> None:
    for name in FLAGS:
        os.environ.pop(name, None)
    os.environ.update(env)


def load_avatar(module, avatar_id: str, env: dict):
    set_env(env)
    stub = SimpleNamespace(model_dtype=torch.float16, runtime_dtype=torch.float16)
    return module.APIAvatar(avatar_id, "", 0, 8, stub, stub, None, None,
                            SimpleNamespace(version="v15"), preparation=False)


def sha(array) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256(array.tobytes())
    digest.update(str((array.shape, array.dtype.str)).encode())
    return digest.hexdigest()[:24]


def face_for(position: int):
    return np.random.RandomState(1000 + position).randint(0, 256, (256, 256, 3), dtype=np.uint8)


def positions_for(count: int):
    half = count // 2
    extra = [count, count + 1, 2 * count - 1, 3 * count + 7, -1, -count, half - 1, half, half + 1]
    return list(range(count)) + extra


def compose_hashes(avatar, positions, backgrounds):
    out = []
    for position in positions:
        face = face_for(position)
        out.append(sha(avatar.compose_frame(face, position)))
        layers = avatar.compose_frame(face, position, return_layers=True)
        out += [sha(layers["composed"]), sha(layers["raw"]), sha(layers["alpha"]["values"]),
                sha(np.asarray(layers["alpha"]["bounds"]))]
        background = backgrounds[position % len(backgrounds)]
        out.append(sha(avatar.compose_frame(face, position, background_frame=background)))
    return out


def mutation_checks(avatar, positions):
    """PNG store: shared decoded images must be immutable and stay exact."""
    frames = avatar.frame_list_cycle
    checks = {}
    paths = sorted((Path(avatar.full_imgs_path)).glob("*.png"))
    plan_hash_before = [sha(p["alpha_u8"]) for p in avatar._compose_plan_cycle if isinstance(p, dict)]
    first = avatar.compose_frame(face_for(0), 0)
    first_hash = sha(first)
    first[:] = 0  # a consumer scribbling on its own output must not leak into the cache
    checks["composed_output_is_private"] = sha(avatar.compose_frame(face_for(0), 0)) == first_hash
    raw = avatar.compose_frame(face_for(3), 3, return_layers=True)["raw"]
    try:
        raw[0, 0, 0] = 1
        checks["raw_layer_write_raises"] = False
    except ValueError:
        checks["raw_layer_write_raises"] = True
    cached = frames[5]
    try:
        cached[0, 0, 0] = 1
        checks["cached_frame_write_raises"] = False
    except ValueError:
        checks["cached_frame_write_raises"] = True
    for position in positions[:: max(1, len(positions) // 64)]:
        compose_hashes(avatar, [position], [np.zeros((10, 10, 3), np.uint8)])
    exact = all(sha(frames[i]) == sha(cv2.imread(str(paths[i]))) for i in range(len(paths)))
    checks["every_position_still_equals_imread_after_composes"] = exact
    plan_hash_after = [sha(p["alpha_u8"]) for p in avatar._compose_plan_cycle if isinstance(p, dict)]
    checks["plans_unchanged"] = plan_hash_before == plan_hash_after
    checks["store_stats"] = frames.stats()
    return checks


def mask_library_check():
    """Every mask in results/v15/avatars: 3 planes identical, plane-0 reader exact."""
    import api_avatar as new
    total = mismatched = non_identical_planes = 0
    per_avatar = {}
    started = time.perf_counter()
    for avatar_dir in sorted((ROOT / "results/v15/avatars").iterdir()):
        masks = sorted((avatar_dir / "mask").glob("*.png"))
        bad = 0
        for path in masks:
            reference = cv2.imread(str(path))
            plane = new._read_mask_plane0_required(str(path))
            total += 1
            if not (np.array_equal(reference[:, :, 0], reference[:, :, 1])
                    and np.array_equal(reference[:, :, 0], reference[:, :, 2])):
                non_identical_planes += 1
            if not np.array_equal(reference[:, :, 0], plane):
                mismatched += 1
                bad += 1
        per_avatar[avatar_dir.name] = {"masks": len(masks), "plane0_mismatch": bad}
    return {"masks_checked": total, "plane0_mismatches": mismatched,
            "masks_with_non_identical_planes": non_identical_planes,
            "avatars": len(per_avatar), "seconds": round(time.perf_counter() - started, 1)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default=str(HERE / "gate_bitexact.json"))
    args = parser.parse_args()

    baseline, baseline_sha = baseline_module()
    import api_avatar as new
    report = {"base_commit": BASE_COMMIT, "baseline_source_sha256": baseline_sha,
              "candidate_source_sha256": hashlib.sha256(Path(new.__file__).read_bytes()).hexdigest(),
              "blend_fixed_point": blending.MUSETALK_BLEND_FIXED_POINT,
              "configs": CONFIGS, "avatars": [], "verdict": None}
    backgrounds = [np.random.RandomState(9).randint(0, 256, (360, 640, 3), dtype=np.uint8), None]
    all_ok = True
    for avatar_id, extra, configs, label in AVATARS:
        if args.quick and avatar_id not in QUICK:
            continue
        configs = tuple(CONFIGS) if configs == "all" else configs
        if not memguard.wait_for_headroom(need_gb=0.9, floor_gb=4.0):
            report["avatars"].append({"avatar": avatar_id, "skipped": "RAM headroom"})
            continue
        entry = {"avatar": avatar_id, "variant": label, "extra_env": extra, "configs": {}}
        started = time.perf_counter()
        base = load_avatar(baseline, avatar_id, extra)
        count = len(base.coord_list_cycle)
        positions = positions_for(count)
        frame_hashes_before = [sha(f) for f in base.frame_list_cycle[: count // 2 + 1]]
        reference = compose_hashes(base, positions, backgrounds)
        with_float = None
        if avatar_id in QUICK and label == "standard":
            blending.MUSETALK_BLEND_FIXED_POINT = False
            with_float = compose_hashes(base, positions[::7], backgrounds)
            blending.MUSETALK_BLEND_FIXED_POINT = True
        entry["baseline_frames_untouched"] = frame_hashes_before == [
            sha(f) for f in base.frame_list_cycle[: count // 2 + 1]]
        entry.update(cycle_positions=count, unique_frames=len({id(f) for f in base.frame_list_cycle}),
                     positions_checked=len(positions), baseline_load_compose_s=round(time.perf_counter() - started, 1))
        del base
        gc.collect()
        for config in configs:
            started = time.perf_counter()
            candidate = load_avatar(new, avatar_id, {**extra, **CONFIGS[config]})
            got = compose_hashes(candidate, positions, backgrounds)
            mismatches = [i for i, (a, b) in enumerate(zip(reference, got)) if a != b]
            result = {"outputs_compared": len(got), "mismatches": len(mismatches) + abs(len(got) - len(reference)),
                      "first_mismatch_output": mismatches[0] if mismatches else None,
                      "digest": hashlib.sha256("".join(got).encode()).hexdigest()[:16],
                      "seconds": round(time.perf_counter() - started, 1)}
            if with_float is not None and config == "new_all":
                blending.MUSETALK_BLEND_FIXED_POINT = False
                float_got = compose_hashes(candidate, positions[::7], backgrounds)
                blending.MUSETALK_BLEND_FIXED_POINT = True
                result["float_blend_mismatches"] = sum(a != b for a, b in zip(with_float, float_got))
            if config == "new_all":
                result["mutation_safety"] = mutation_checks(candidate, positions)
                ms = result["mutation_safety"]
                result["mutation_ok"] = all(v for k, v in ms.items() if k != "store_stats")
                all_ok &= result["mutation_ok"]
            all_ok &= result["mismatches"] == 0 and result.get("float_blend_mismatches", 0) == 0
            entry["configs"][config] = result
            del candidate
            gc.collect()
        all_ok &= entry["baseline_frames_untouched"]
        report["avatars"].append(entry)
        print(json.dumps({k: entry[k] for k in ("avatar", "variant", "cycle_positions")}),
              {c: (r["mismatches"], r.get("mutation_ok")) for c, r in entry["configs"].items()}, flush=True)
    report["mask_library"] = mask_library_check()
    all_ok &= report["mask_library"]["plane0_mismatches"] == 0
    report["min_mem_available_gb"] = round(WATCHDOG.min_seen_gb, 2)
    report["verdict"] = "PASS" if all_ok and not any("skipped" in a for a in report["avatars"]) else "FAIL"
    set_env({})
    Path(args.out).write_text(json.dumps(report, indent=1))
    print("verdict", report["verdict"], "->", args.out)


if __name__ == "__main__":
    main()
