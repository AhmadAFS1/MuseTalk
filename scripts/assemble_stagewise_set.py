"""Assemble a stagewise UNet engine set from a base set plus per-block overlays (symlinks + merged manifest).

Same procedure the srcmix/srcv1 sets used (docs/.../unet_fp16/run_srccache.sh): every block engine is a
symlink to the real file of the set it comes from, the manifest's block entries are copied from those
sets, and the result is left un-finalised (complete=False, no probe) so that
  build_unet_stagewise.py --variant srccache --root <out> --blocks prefix
finalises it (loads the chain, CUDA-graph == direct check, determinism, probe hash).
  assemble_stagewise_set.py --base models/tensorrt_unet_stagewise_sm89_srcmix \
      --overlay models/tensorrt_unet_stagewise_sm89_nd0_thr_4e-06:down1,down2,up0,up1,up2,up3 --out models/...
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", required=True)
    ap.add_argument("--overlay", action="append", default=[], help="<root>:<block,block,...> (later wins)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    os.chdir(ROOT)
    bs = f"bs{args.batch}"
    base_dir = Path(args.base) / bs
    man = json.loads((base_dir / "manifest.json").read_text())
    srcs = {k: base_dir for k in man["blocks"]}
    blocks = dict(man["blocks"])
    calib = {}
    for ov in args.overlay:
        root, names = ov.split(":")
        d = Path(root) / bs
        m = json.loads((d / "manifest.json").read_text())
        if m.get("spec") != man.get("spec"):
            raise SystemExit(f"{d}: spec differs from the base set")
        for b in names.split(","):
            if b not in m.get("blocks", {}):
                raise SystemExit(f"{d}: no block {b}")
            blocks[b] = m["blocks"][b]
            srcs[b] = d
        if m.get("int8_calibration"):
            calib[str(d)] = {"blocks": names.split(","), **m["int8_calibration"]}
    out = Path(args.out) / bs
    out.mkdir(parents=True, exist_ok=True)
    for k, v in blocks.items():
        link = out / v["engine_file"]
        target = os.path.realpath(srcs[k] / v["engine_file"])
        if not os.path.exists(target):
            raise SystemExit(f"missing engine {target}")
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(target, link)
    man.update(blocks=blocks, complete=False, assembled_from={"base": args.base, "overlays": args.overlay},
               int8_calibration_by_overlay=calib)
    for k in ("probe", "runtime", "missing_blocks", "finalized_utc"):
        man.pop(k, None)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"assembled {out}: " + ", ".join(f"{k}<-{Path(str(srcs[k])).parent.name}" for k in blocks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
