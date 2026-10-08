"""Fixed locations and integrity checks shared by the parent, the workers and the checks."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
WORKFLOW = WORKTREE / "character_factory" / "h3_avatar_workflow"
WORKSPACE = Path(os.environ.get("MUSETALK_REPRO_WORKSPACE", "/workspace")).resolve()
ACCEPTED_ROOT = Path(os.environ.get("MUSETALK_REPRO_ACCEPTED", str(WORKSPACE / "experiments" / "avatar_diversity_20260927"))).resolve()
IDENTITIES = (
    "black_man_short_beard",
    "black_woman",
    "east_asian_man_goatee",
    "middle_eastern_man_full_beard",
    "south_asian_woman",
    "white_man_clean_shaven",
)
OUT_ROOT = Path(os.environ.get("MUSETALK_REPRO_CHIN_OUT", str(WORKTREE / "docs" / "fps_comparisons" / "4070s_300fps_impl_20260928" / "chin_multistream")))
FACEMESH_PY = WORKSPACE / "SoulX-FlashHead" / ".venv" / "bin" / "python"
BLENDING = WORKTREE / "musetalk" / "utils" / "blending.py"
N_FRAMES = 240
BATCH = 8
FRAME_SHAPE = (896, 512, 3)
FACE_SHAPE = (256, 256, 3)


def add_import_paths() -> None:
    """chin.py imports musetalk.utils.blending; the workflow dir holds chin/backend/common."""
    for p in (str(WORKFLOW), str(WORKTREE)):
        if p not in sys.path:
            sys.path.insert(0, p)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def accepted_render(identity: str) -> dict:
    return json.loads((ACCEPTED_ROOT / identity / "render.json").read_text())


def code_integrity(identities=IDENTITIES) -> dict:
    """The accepted recipe files must be byte-identical to what the accepted renders recorded."""
    ref = accepted_render(identities[0])
    files = {name: WORKFLOW / name for name in ref["code_sha256"]}
    got = {name: sha256_file(p) for name, p in files.items()}
    blend = sha256_file(BLENDING)
    ok = all(got[n] == ref["code_sha256"][n] for n in got) and \
        blend == ref["model_and_blending_code_sha256"]["MuseTalk/musetalk/utils/blending.py"]
    for ident in identities[1:]:
        r = accepted_render(ident)
        ok = ok and r["code_sha256"] == ref["code_sha256"]
    return {"files": got, "blending_py": blend, "matches_accepted_render_json": bool(ok)}
