"""Source-verified motion routing and bounded, optical-flow-aligned bridges.

The atlas is local configuration, never a client-supplied filesystem path.
Audio time and body-motion time deliberately remain independent.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import logging
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

IDLE = "neutral_resting"
TALK = "speaking_direct"
SMILE = "light_smile"
LOG = logging.getLogger(__name__)


def atomic_json(path, value):
    """Publish a complete JSON artifact; readers never observe a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent,
                prefix="."+path.name+".", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def publish_bank(path, manifest):
    """Publish bank then its small discovery record, with exact content binding."""
    MotionBank(manifest)
    path = Path(path)
    atomic_json(path, manifest)
    atomic_json(path.with_name("motion-registration.json"), {
        "version": 1,
        "source_hashes": {pose: manifest["sources"][pose]["sha256"]
                          for pose in (IDLE, TALK, SMILE)},
        "atlas_sha256": file_hash(path),
    })


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_video(path):
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        cap.release()
    if not frames or not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Invalid motion video: {path}")
    return frames, fps


def flow_blend(old, new, progress):
    """Warp both faces toward intermediate geometry before blending appearance.

    This is a local CPU operation. It does not synthesize identity or guarantee
    perceptual quality; the atlas and recorded output still require review.
    """
    import cv2
    if old.shape != new.shape:
        raise ValueError("Motion bridge requires identical frame geometry")
    if progress <= 0:
        return old.copy()
    if progress >= 1:
        return new.copy()
    h, w = old.shape[:2]
    size = (max(32, w // 4), max(32, h // 4))
    a = cv2.cvtColor(cv2.resize(old, size), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(cv2.resize(new, size), cv2.COLOR_BGR2GRAY)
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    forward = dis.calc(a, b, None)
    backward = dis.calc(b, a, None)
    scale = np.array([w / size[0], h / size[1]], np.float32)
    forward = cv2.resize(forward, (w, h)) * scale
    backward = cv2.resize(backward, (w, h)) * scale
    # Cap displacements rather than allow unreliable hair/background flow to
    # fold the entire image. Large mismatches are rejected by atlas building.
    forward = np.clip(forward, -32, 32)
    backward = np.clip(backward, -32, 32)
    x, y = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    t = float(progress)
    left = cv2.remap(old, x - t * forward[..., 0], y - t * forward[..., 1],
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    right = cv2.remap(new, x - (1-t) * backward[..., 0], y - (1-t) * backward[..., 1],
                      cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return np.clip(left.astype(np.float32)*(1-t) + right.astype(np.float32)*t,
                   0, 255).astype(np.uint8)


class MotionBank:
    def __init__(self, manifest):
        self.manifest = manifest
        routing = {key: value for key, value in manifest.items() if key not in ("status", "review")}
        self.routing_sha256 = hashlib.sha256(json.dumps(routing, sort_keys=True,
                        separators=(",", ":")).encode()).hexdigest()
        if manifest.get("version") != 1:
            raise ValueError("Unsupported motion atlas version")
        self.sources = manifest["sources"]
        if set(self.sources) != {IDLE, TALK, SMILE}:
            raise ValueError("Motion atlas requires idle, talking, and smiling")
        self.fps = float(manifest["fps"])
        self.bridge_seconds = float(manifest.get("bridge_seconds", .3))
        self.short_seconds = float(manifest.get("short_reply_seconds", 3))
        if not (1 <= self.fps <= 60 and .05 <= self.bridge_seconds <= .5
                and 0 <= self.short_seconds <= 10):
            raise ValueError("Invalid motion timing")
        for pose, source in self.sources.items():
            if int(source["frame_count"]) < 2:
                raise ValueError("Motion source has no usable cycle")
            for target in self.sources:
                edges = manifest["edges"][pose][target]
                if len(edges) != source["frame_count"]:
                    raise ValueError("Incomplete motion exit coverage")
                for edge in edges:
                    if not (0 <= int(edge["target_frame"]) < self.count(target)):
                        raise ValueError("Motion target frame outside source")
                    if not math.isfinite(float(edge["score"])):
                        raise ValueError("Non-finite transition score")
        if not all(edge["admissible"] for edge in manifest["edges"][IDLE][IDLE]):
            raise ValueError("Idle must have a closed-mouth recovery at every phase")
        self.eye_blend = self._validate_eye_blend()
        self.current_phoneme = self._validate_current_phoneme()

    def _validate_current_phoneme(self):
        profile = self.manifest.get("current_phoneme")
        if profile is None:
            return None
        if (not isinstance(profile, dict)
                or set(profile) != {"method", "source_hashes"}
                or profile.get("method") != "current_similarity_v1"):
            raise ValueError("Unsupported current phoneme blend profile")
        if self.eye_blend is None:
            raise ValueError("Current phoneme blending requires source-bound eye geometry")
        if len({(s["height"], s["width"]) for s in self.sources.values()}) != 1:
            raise ValueError("Current phoneme sources require identical frame dimensions")
        if profile.get("source_hashes") != {p: s["sha256"] for p, s in self.sources.items()}:
            raise ValueError("Current phoneme source hash mismatch")
        return profile

    def blend_current(self, old_raw, new_raw, current_composed, old_alpha, new_alpha,
                      progress, old_pose, old_frame, new_pose, new_frame):
        """Align raw bodies while keeping this generation frame's articulation.

        Used only inside a speaking job. Entry and recovery retain their separate
        playback policies; this method must never fade speech against stale lips.
        """
        if self.current_phoneme is None:
            raise ValueError("Current phoneme blending is not enabled for this bank")
        from scripts.motion_current_phoneme import current_phoneme_blend
        for pose, index, image in ((old_pose, old_frame, old_raw),
                                   (new_pose, new_frame, new_raw)):
            if (pose not in self.sources or isinstance(index, (bool, np.bool_))
                    or not isinstance(index, (int, np.integer))
                    or not 0 <= index < self.count(pose)):
                raise ValueError("Current phoneme blend requires exact original source indices")
            source = self.sources[pose]
            if image.shape[:2] != (source["height"], source["width"]):
                raise ValueError("Current phoneme geometry does not match supplied frame size")
        return current_phoneme_blend(old_raw, new_raw, current_composed,
            old_alpha, new_alpha, progress, self.eye_blend["frames"][old_pose][old_frame],
            self.eye_blend["frames"][new_pose][new_frame])

    def _validate_eye_blend(self):
        profile = self.manifest.get("eye_blend")
        if profile is None:
            return None
        if not isinstance(profile, dict) or profile.get("method") != "incoming_roi_v1":
            raise ValueError("Unsupported motion eye blend profile")
        hashes, frames = profile.get("source_hashes"), profile.get("frames")
        if not isinstance(hashes, dict) or not isinstance(frames, dict):
            raise ValueError("Motion eye blend needs source-bound frame geometry")
        if set(hashes) != set(self.sources) or set(frames) != set(self.sources):
            raise ValueError("Motion eye blend must cover all three sources")
        from scripts.motion_eye_blend import validate_eye_points
        for pose, source in self.sources.items():
            if hashes[pose] != source["sha256"]:
                raise ValueError("Motion eye geometry source hash mismatch")
            width, height = source.get("width"), source.get("height")
            if not isinstance(width, int) or not isinstance(height, int) or min(width, height) < 32:
                raise ValueError("Motion eye geometry requires original source dimensions")
            rows = frames[pose]
            if not isinstance(rows, list) or len(rows) != self.count(pose):
                raise ValueError("Incomplete motion eye geometry coverage")
            for points in rows:
                validate_eye_points(points, width, height)
        return profile

    def blend(self, old, new, progress, old_pose, old_frame, new_pose, new_frame):
        """Blend an exact pair; optional source-bound eyes keep incoming appearance."""
        if self.eye_blend is None:
            return flow_blend(old, new, progress)
        from scripts.motion_eye_blend import incoming_eye_blend
        old_pose, new_pose = self.canonical(old_pose), self.canonical(new_pose)
        if not (isinstance(old_frame, (int, np.integer)) and isinstance(new_frame, (int, np.integer))
                and 0 <= old_frame < self.count(old_pose) and 0 <= new_frame < self.count(new_pose)):
            raise ValueError("Motion eye blend requires exact original source indices")
        for pose, image in ((old_pose, old), (new_pose, new)):
            source = self.sources[pose]
            if image.shape[:2] != (source["height"], source["width"]):
                raise ValueError("Motion eye geometry does not match the supplied frame size")
        return incoming_eye_blend(old, new, progress,
                                 self.eye_blend["frames"][old_pose][old_frame],
                                 self.eye_blend["frames"][new_pose][new_frame])

    def canonical(self, pose):
        return pose if pose in self.sources else IDLE

    def count(self, pose):
        return int(self.sources[self.canonical(pose)]["frame_count"])

    def edge(self, pose, index, target):
        pose, target = self.canonical(pose), self.canonical(target)
        return self.manifest["edges"][pose][target][int(index) % self.count(pose)]

    def compatible(self, paths):
        # Prepared source MP4s must be exact copies of the atlas source. Reject
        # crop/encode changes instead of silently applying stale geometry.
        for pose, source in self.sources.items():
            if pose not in paths or file_hash(paths[pose]) != source["sha256"]:
                return False
        return True

    def plan(self, total, generation_fps, initial_pose, initial_frame, requested):
        """Compile physical source choices; semantic speech labels stay public."""
        if total < 1 or not math.isfinite(generation_fps) or generation_fps <= 0:
            raise ValueError("Invalid generation timeline")
        pose = self.canonical(initial_pose)
        source = float(initial_frame % self.count(pose))
        duration = total / generation_fps
        bridge_n = max(2, math.ceil(self.bridge_seconds * generation_fps))
        # More than one bridge must fit before enabling an expressive source.
        short = duration < self.short_seconds
        frames, switches = [], []
        cooldown_n = bridge_n + math.ceil(.25 * generation_fps)
        cooldown_until = bridge_n
        terminal_start = max(0, total - bridge_n - math.ceil(.2 * generation_fps))
        for n in range(total):
            desired = IDLE
            if not short and bridge_n <= n < terminal_start:
                cue = requested[0]
                for item in requested:
                    if n >= total * item["at_permille"] / 1000:
                        cue = item
                desired = self.canonical(cue["pose_id"])
            idx = math.floor(source + 1e-8) % self.count(pose)
            blend = 0
            if desired != pose and n >= cooldown_until:
                # The compositor blends from its previous composed frame. The
                # next source index has not been displayed and may already be
                # in a different blink phase (especially with 24→20Hz sampling).
                anchor_index = frames[-1]["source_frame"] if frames else idx
                edge = self.edge(pose, anchor_index, desired)
                # Never enter motion with an uncovered interrupted-return phase.
                eligible = all(e["admissible"] for e in self.manifest["edges"][desired][IDLE])
                # A semantic cue must leave room for its whole entry/cooldown,
                # followed by the mandatory terminal bridge and idle phonemes.
                enough_time = n + cooldown_n <= terminal_start or desired == IDLE
                if edge["admissible"] and eligible and enough_time:
                    outgoing = pose
                    pose, idx = desired, int(edge["target_frame"])
                    source = float(idx)
                    blend = bridge_n
                    cooldown_until = n + cooldown_n
                    switches.append({"frame": n, "from": outgoing, "to": pose,
                                     "anchor_frame": anchor_index,
                                     "target_frame": idx, "score": edge["score"]})
            frames.append({"pose_id": pose, "source_frame": idx,
                           "crossfade_frames": blend})
            source += self.fps / generation_fps
        return {"frames": frames, "switches": switches, "short_reply": short,
                "duration_seconds": duration, "switch_policy": "matched_motion_v1",
                "status": "compiled", "generation_fps": generation_fps,
                "total_generation_frames": total,
                "bridge_seconds": self.bridge_seconds}


@lru_cache(maxsize=64)
def load_bank(path, serialized):
    # Read once and key the exact contents: this filesystem can preserve mtime
    # across rapid writes, including a change to recorded-review status.
    return MotionBank(json.loads(serialized))


def configured_bank(paths):
    """Select an exact character bank from a file or a directory registry."""
    if not {IDLE, TALK, SMILE}.issubset(paths):
        return None
    path = os.environ.get("WEBRTC_MOTION_ATLAS", "").strip()
    registry = os.environ.get("WEBRTC_MOTION_ATLAS_DIR", "").strip()
    candidates = [Path(path)] if path else []
    if registry:
        candidates += sorted(Path(registry).glob("*/motion-atlas.json"))
    if not candidates:
        return None
    hashes = {pose: file_hash(paths[pose]) for pose in (IDLE, TALK, SMILE)}
    matched = None
    for candidate in dict.fromkeys(candidates):
        # Explicit configuration errors are fatal. Registry entries are isolated:
        # an unfinished/damaged *other* character cannot take this one offline.
        explicit = bool(path and candidate == Path(path))
        registration = candidate.with_name("motion-registration.json")
        indexed_match = False
        try:
            if registration.exists():
                index = json.loads(registration.read_text())
                if index.get("version") != 1 or set(index["source_hashes"]) != set(hashes):
                    raise ValueError("Invalid motion registration")
                if index["source_hashes"] != hashes:
                    continue
                indexed_match = True
            serialized = candidate.read_text()
            manifest = json.loads(serialized)
            source_hashes = {p: manifest["sources"][p]["sha256"] for p in hashes}
        except (OSError, ValueError, KeyError, TypeError) as exc:
            if explicit or indexed_match:
                raise ValueError(f"Unreadable selected motion atlas: {candidate}") from exc
            LOG.warning("Skipping invalid motion registry entry %s: %s", candidate, exc)
            continue
        if source_hashes != hashes:
            if indexed_match:
                raise ValueError(f"Motion registration does not match its atlas: {candidate}")
            continue
        if indexed_match and hashlib.sha256(serialized.encode()).hexdigest() != index["atlas_sha256"]:
            raise ValueError(f"Motion registration is stale; republish bank: {candidate}")
        # Fully validate only selected banks, preserving hard failure on any
        # selected geometry/review corruption rather than falling back silently.
        bank = load_bank(str(candidate), serialized)
        if matched is not None:
            raise ValueError("Multiple motion atlases match this character; keep one active version")
        matched = bank
    if matched is None:
        return None
    if (matched.manifest.get("status") != "reviewed"
            and os.environ.get("WEBRTC_MOTION_ALLOW_UNREVIEWED", "0") != "1"):
        raise ValueError("Motion atlas needs recorded review; pilot tests require WEBRTC_MOTION_ALLOW_UNREVIEWED=1")
    return matched
