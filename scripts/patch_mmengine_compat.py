#!/usr/bin/env python3
"""Backport MMEngine's TorchAdafactor registration to the pinned 0.10.4 wheel.

PyTorch 2.5 exposes an Adafactor optimizer. MMEngine 0.10.4 registers it under
the same name as transformers.Adafactor, breaking mmcv.ops and avatar prep.
Upstream 0.10.7 registers PyTorch's version as TorchAdafactor instead:
https://github.com/open-mmlab/mmengine/blob/v0.10.7/mmengine/optim/optimizer/builder.py
Keep the validated package pins; change only this optimizer registration.
"""
import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path

OLD = "            OPTIMIZERS.register_module(module=_optim)\n"
NEW = ('            if module_name == "Adafactor":\n'
       '                OPTIMIZERS.register_module(name="TorchAdafactor", module=_optim)\n'
       '            else:\n'
       '                OPTIMIZERS.register_module(module=_optim)\n')


def patched_source(source):
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                 and node.name == "register_torch_optimizers"]
    if len(functions) != 1:
        raise ValueError("Unexpected MMEngine optimizer registration")
    node = functions[0]
    lines = source.splitlines(keepends=True)
    block = "".join(lines[node.lineno - 1:node.end_lineno])
    if NEW in block:
        if block.count(NEW) != 1:
            raise ValueError("Ambiguous MMEngine compatibility patch")
        return source, False
    if block.count(OLD) != 1 or "TorchAdafactor" in block:
        raise ValueError("Unexpected MMEngine optimizer registration")
    changed = block.replace(OLD, NEW, 1)
    result = "".join(lines[:node.lineno - 1]) + changed + "".join(lines[node.end_lineno:])
    ast.parse(result)
    return result, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the patch without modifying the wheel")
    args = parser.parse_args()
    dist = importlib.metadata.distribution("mmengine")
    if dist.version != "0.10.4":
        raise ValueError("Compatibility backport is only reviewed for mmengine 0.10.4")
    path = Path(dist.locate_file("mmengine/optim/optimizer/builder.py"))
    if path.is_symlink() or not path.is_file():
        raise ValueError("Unexpected MMEngine source path")
    before = path.read_text()
    after, changed = patched_source(before)
    if args.check and changed:
        raise ValueError("MMEngine TorchAdafactor backport is missing")
    if changed:
        path.write_text(after)
    print(json.dumps({"schema": "mmengine_torch_adafactor_backport_v1", "version": dist.version,
                      "changed": changed, "sha256_before": hashlib.sha256(before.encode()).hexdigest(),
                      "sha256_after": hashlib.sha256(after.encode()).hexdigest(),
                      "scope": "optimizer registry names only; no inference or preprocessing math"}))


if __name__ == "__main__":
    main()
