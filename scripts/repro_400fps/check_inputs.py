"""Input checks for scripts/repro_400fps/00_check.sh: files, byte-exact hashes and package versions.

Prints one line per check ("ok", "warn" or "MISSING") and exits 1 when any MISSING. CPU only. With --deep it also
hashes the large files: the avatars' mp4s and the 2.1 GB BEFORE .ts engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
ACC = Path("/workspace/experiments/avatar_diversity_20260927")
IDS = ["black_man_short_beard", "black_woman", "east_asian_man_goatee", "middle_eastern_man_full_beard", "south_asian_woman",
       "white_man_clean_shaven"]
FILES = ["source.mp4", "source_landmarks.npy", "cache.pt", "masks.npz", "faces.npz", "generated_landmarks.npy",
         "chin_delta.npy", "render.json", "speech.wav", "refined_raw.mp4"]
PINS = ["torch", "torch_tensorrt", "tensorrt-cu12", "nvidia-modelopt", "diffusers", "onnx", "transformers", "numpy",
        "opencv-python", "opencv-python-headless", "safetensors"]
bad = 0


def say(level, what, detail=""):
    global bad
    bad += level == "MISSING"
    print(f"{level:7s} {what}" + (f" ({detail})" if detail else ""), flush=True)


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true")
    ap.add_argument("--py", default="/workspace/.venvs/musetalk_trt_stagewise/bin/python")
    a = ap.parse_args()

    # six prepared avatars: every file, and the render's own recorded artifact hashes
    for i in IDS:
        d = ACC / i
        miss = [f for f in FILES if not (d / f).exists()]
        if miss:
            say("MISSING", f"avatar {i}", "missing " + ", ".join(miss) + "; copy the directory byte-exact from the original box")
            continue
        r = json.loads((d / "render.json").read_text())
        wrong = [Path(p).name for p, h in r.get("artifacts", {}).items()
                 if (a.deep or not p.endswith(".mp4")) and Path(p).exists() and sha(Path(p)) != h]
        say("MISSING" if wrong else "ok", f"avatar {i} artifacts match render.json", ("differ: " + ", ".join(wrong)) if wrong else
            ("mp4s not hashed; use --deep" if not a.deep else "all hashed"))
    # weights and the BEFORE engine the renders recorded
    r0 = json.loads((ACC / IDS[0] / "render.json").read_text())
    for rel, h in r0.get("model_and_blending_code_sha256", {}).items():
        p = REPO / rel.replace("MuseTalk/", "", 1)
        if rel.endswith(".ts"):
            if not p.exists():
                say("warn", f"BEFORE engine {rel}", "needed only for 30_benchmark.sh PAIR (the like-for-like baseline); build it with "
                    "scripts/unet_engine_store.py per docs/STARTUP.md §6 - a rebuilt engine is not bit-identical to the published one")
            elif a.deep:
                say("ok" if sha(p) == h else "warn", f"BEFORE engine sha256", "matches render.json" if sha(p) == h else "differs from render.json")
            else:
                say("ok", f"BEFORE engine present {rel}", "sha not checked; use --deep")
            continue
        if not p.exists():
            say("MISSING", rel, "scripts/install_musetalk.sh downloads weights")
        else:
            ok = sha(p) == h
            say("ok" if ok else "MISSING", f"{rel} sha256", "matches the renders" if ok else "differs from the renders")
    unet = REPO / "models/musetalkV15/unet.pth"
    say("ok" if unet.exists() and unet.stat().st_size == 3400074924 else "MISSING", "models/musetalkV15/unet.pth (3400074924 bytes)")
    # UNet capture corpus: exactly the manifest's files
    cd = REPO / "calibration/unet_multi_avatar_20260928"
    if not (cd / "manifest.json").exists():
        say("MISSING", "UNet capture corpus calibration/unet_multi_avatar_20260928",
            "copy it (218 MB, 352 main + 96 holdout) from the original box; rebuilding needs the 14 source avatars it names")
    else:
        m = json.loads((cd / "manifest.json").read_text())
        files = m.get("files", [])
        names = [f if isinstance(f, str) else f.get("file") or f.get("path") for f in files]
        missing = [n for n in names if n and not (cd / n).exists() and not (cd / Path(n).name).exists() and not (cd / "holdout" / Path(n).name).exists()]
        main = len(list(cd.glob("unet_io_*.pt")))
        hold = len(list((cd / "holdout").glob("unet_io_*.pt")))
        ok = not missing and main == 352 and hold == 96
        say("ok" if ok else "MISSING", f"UNet capture corpus: {main} main + {hold} holdout",
            "matches the manifest" if ok else f"expected 352 + 96, {len(missing)} manifest files missing")
    # package versions against the pinned constraints (the manifests' ONNX hashes depend on modelopt/diffusers/onnx)
    cons = {}
    for line in (REPO / "requirements/constraints-cu121.txt").read_text().splitlines():
        mm = re.match(r"^([A-Za-z0-9_.\-]+)={2,3}([^\s;#]+)", line.strip())
        if mm:
            cons[mm.group(1).lower().replace("_", "-")] = mm.group(2)
    try:
        freeze = subprocess.run([a.py, "-m", "pip", "freeze"], capture_output=True, text=True, timeout=120).stdout
    except Exception as e:  # pragma: no cover
        freeze = ""
        say("MISSING", f"pip freeze in {a.py}", repr(e))
    got = {}
    for line in freeze.splitlines():
        mm = re.match(r"^([A-Za-z0-9_.\-]+)={2,3}([^\s;]+)", line.strip())
        if mm:
            got[mm.group(1).lower().replace("_", "-")] = mm.group(2)
    for pkg in PINS:
        k = pkg.lower().replace("_", "-")
        if k not in cons:
            continue
        if k not in got:
            say("MISSING" if k in ("torch", "tensorrt-cu12", "nvidia-modelopt", "diffusers", "onnx") else "warn", f"{pkg}=={cons[k]}",
                "not installed in the main venv")
        else:
            say("ok" if got[k] == cons[k] else "warn", f"{pkg} {got[k]}", "pinned" if got[k] == cons[k] else f"constraints pin {cons[k]}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
