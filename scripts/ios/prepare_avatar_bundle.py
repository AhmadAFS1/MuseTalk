#!/usr/bin/env python
"""Export one avatar cache plus one speech clip for the on-device lipsync view (ios/MuseTalkBench).

From a MuseTalk v1.5 avatar directory (full_imgs/, mask/, coords.pkl, mask_coords.pkl,
latents.pt: 2N entries = N video frames followed by the same frames reversed) it writes,
into <out>/avatar/:
  avatar.json                 geometry, fps, cycle length, file lists, audio description
  frames/NNNN.jpg             the N unique full frames (JPEG q92; the cycle maps i -> i<N ? i : 2N-1-i)
  masks/NNNN.png              the N unique soft masks, 8-bit single channel
  latents_f16.bin             [N, 8, 32, 32] UNet input latents (masked + reference) per unique frame
  audio_f32.bin               the clip as 16 kHz mono float32 (what the log-mel model takes)
  audio.m4a                   the same seconds of the clip for playback
  windows_ref_f16.bin         [frames, 50, 384] the repo's per-frame Whisper windows (AudioProcessor,
                              torch fp32) at the avatar fps: the check for the Swift gather
  check/NNNN_{composed,face}.png   a few frames composed in Python with the same Core ML models the
                              phone runs (chunked int8w UNet on the ANE + Core ML TAESD): the check
                              for the Swift frame loop

Usage (repo root, .venv-ct):
  python scripts/ios/prepare_avatar_bundle.py --avatar /path/to/latina_guided_... --audio data/audio/ai-assistant.mpga
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "ios"))

from musetalk.utils.audio_processor import AudioProcessor  # noqa: E402
from musetalk.utils.blending import get_image_blending  # noqa: E402
from render_evidence import load_avatar, to_bgr_u8  # noqa: E402

MANIFEST = REPO_ROOT / "models" / "ios_bench" / "coreml" / "manifest.json"


def main() -> None:
    import librosa
    from transformers import WhisperModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--avatar", type=Path, required=True)
    ap.add_argument("--audio", type=Path, default=REPO_ROOT / "data" / "audio" / "ai-assistant.mpga")
    ap.add_argument("--seconds", type=float, default=30.0, help="clip length (<= 30 s, one Whisper segment)")
    ap.add_argument("--fps", type=int, default=0, help="render fps (0 = the avatar video's fps)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "ios" / "MuseTalkBench" / "HarnessBundle")
    ap.add_argument("--unet", default="unet_anec0_int8w_b1", help="chunk parent used for the check frames")
    ap.add_argument("--check-frames", default="0,24,48,96")
    ap.add_argument("--whisper-dir", type=Path, default=REPO_ROOT / "models" / "whisper")
    args = ap.parse_args()

    out = args.out / "avatar"
    if out.exists():
        shutil.rmtree(out)
    (out / "frames").mkdir(parents=True)
    (out / "masks").mkdir()
    (out / "check").mkdir()

    frames, masks, coords, mask_coords, latents = load_avatar(args.avatar)
    total = len(frames)
    n = total // 2 if total % 2 == 0 else total
    # the cache is a ping-pong cycle: entry 2N-1-i must equal entry i
    symmetric = total % 2 == 0 and all(
        np.array_equal(frames[i], frames[total - 1 - i]) and np.array_equal(masks[i], masks[total - 1 - i])
        and [int(v) for v in coords[i]] == [int(v) for v in coords[total - 1 - i]]
        and [int(v) for v in mask_coords[i]] == [int(v) for v in mask_coords[total - 1 - i]]
        and torch.equal(latents[i], latents[total - 1 - i])
        for i in range(0, total // 2, max(1, total // 40)))
    if not symmetric:
        n = total
    print(f"avatar {args.avatar.name}: {total} cache entries, {'ping-pong, ' if symmetric else 'NOT symmetric, '}{n} unique; "
          f"frame {frames[0].shape[1]}x{frames[0].shape[0]}, mask {masks[0].shape[1]}x{masks[0].shape[0]}", flush=True)

    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate",
                            "-of", "csv=p=0", str(args.avatar / "input_video.mp4")], capture_output=True, text=True).stdout.strip()
    num, den = (probe.split("/") + ["1"])[:2]
    video_fps = round(float(num) / float(den)) if probe else 25
    fps = args.fps or video_fps

    for i in range(n):
        cv2.imwrite(str(out / "frames" / f"{i:04d}.jpg"), frames[i], [cv2.IMWRITE_JPEG_QUALITY, 92])
        m = masks[i][:, :, 0] if masks[i].ndim == 3 else masks[i]
        cv2.imwrite(str(out / "masks" / f"{i:04d}.png"), m)
    lat = latents[:n].to(torch.float16).contiguous().numpy()
    (out / "latents_f16.bin").write_bytes(lat.tobytes())
    print(f"  wrote {n} frames (jpg), {n} masks (png), latents {lat.shape}", flush=True)

    # audio: 16 kHz float32 for the log-mel model, and a playable cut of the same seconds
    wav, _ = librosa.load(str(args.audio), sr=16000)
    seconds = min(args.seconds, 30.0, len(wav) / 16000)
    samples = int(round(seconds * 16000))
    wav = wav[:samples].astype(np.float32)
    (out / "audio_f32.bin").write_bytes(wav.tobytes())
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(args.audio), "-t", f"{seconds:.3f}", "-vn", "-c:a", "aac",
                    "-b:a", "128k", str(out / "audio.m4a")], check=True)
    print(f"  audio {args.audio.name}: {seconds:.2f} s, {samples} samples; playback audio.m4a", flush=True)

    # the repo's per-frame windows (reference for the Swift gather)
    processor = AudioProcessor(feature_extractor_path=str(args.whisper_dir))
    whisper = WhisperModel.from_pretrained(str(args.whisper_dir), attn_implementation="eager").eval()
    seg = np.zeros(16000 * 30, dtype=np.float32)
    seg[:samples] = wav
    mel = processor.feature_extractor(seg, return_tensors="pt", sampling_rate=16000).input_features
    with torch.inference_mode():
        feats, num_frames = processor.encode_whisper_feature([mel], torch.device("cpu"), torch.float32, whisper, samples, fps=fps)
        windows = processor.build_audio_prompts(whisper_feature=feats, num_frames=num_frames, fps=fps).float()
    (out / "windows_ref_f16.bin").write_bytes(windows.to(torch.float16).contiguous().numpy().tobytes())
    print(f"  windows {tuple(windows.shape)} at {fps} fps", flush=True)

    # check frames composed in Python with the phone's Core ML models
    import coremltools as ct

    man = json.loads(MANIFEST.read_text())
    units = ct.ComputeUnit.CPU_AND_NE
    chunks = [ct.models.MLModel(man[f"{args.unet}_chunk{i}"]["path"], compute_units=units) for i in (1, 2)]
    taesd = ct.models.MLModel(man["taesd_fp16_b1"]["path"], compute_units=units)
    names2 = [x.name for x in chunks[1].get_spec().description.input]
    check = [int(x) for x in args.check_frames.split(",") if int(x) < windows.shape[0]]
    t_ms = []
    for i in check:
        j = i % total
        feed = {"sample": latents[j:j + 1].numpy().astype(np.float16), "audio": windows[i:i + 1].numpy().astype(np.float16)}
        t0 = time.perf_counter()
        o1 = chunks[0].predict(feed)
        pred = chunks[1].predict({k: ({**feed, **o1})[k] for k in names2})["pred_latents"]
        img = np.asarray(taesd.predict({"latents": np.asarray(pred, dtype=np.float16)})["faces"], dtype=np.float32)
        t_ms.append((time.perf_counter() - t0) * 1000)
        face = to_bgr_u8(img)[0]
        x1, y1, x2, y2 = coords[j]
        # compose onto the JPEG the app ships (the PNG master would hide the blend behind JPEG noise)
        u = j if j < n else total - 1 - j
        base = cv2.imread(str(out / "frames" / f"{u:04d}.jpg"))
        composed = get_image_blending(base, cv2.resize(face, (x2 - x1, y2 - y1)), coords[j], masks[j], mask_coords[j])
        cv2.imwrite(str(out / "check" / f"{i:04d}_face.png"), face)
        cv2.imwrite(str(out / "check" / f"{i:04d}_composed.png"), composed)
    print(f"  check frames {check}: Core ML chain {np.mean(t_ms[1:]) if len(t_ms) > 1 else t_ms[0]:.1f} ms/frame on the Mac", flush=True)

    meta = {
        "avatar_id": args.avatar.name,
        "fps": fps,
        "video_fps": video_fps,
        "frame_width": int(frames[0].shape[1]),
        "frame_height": int(frames[0].shape[0]),
        "unique_frames": n,
        "cycle_length": total,
        "ping_pong": symmetric,
        "frames": [f"frames/{i:04d}.jpg" for i in range(n)],
        "masks": [f"masks/{i:04d}.png" for i in range(n)],
        "mask_width": int(masks[0].shape[1]),
        "mask_height": int(masks[0].shape[0]),
        "coords": [[int(v) for v in coords[i]] for i in range(n)],
        "mask_coords": [[int(v) for v in mask_coords[i]] for i in range(n)],
        "latents": {"file": "latents_f16.bin", "shape": [n, 8, 32, 32], "dtype": "float16"},
        "audio": {"source": args.audio.name, "seconds": seconds, "sample_rate": 16000, "samples": samples,
                  "waveform": "audio_f32.bin", "playback": "audio.m4a",
                  "windows_ref": {"file": "windows_ref_f16.bin", "shape": list(windows.shape), "dtype": "float16"},
                  "frames": int(windows.shape[0])},
        "window": {"padding_left_steps": 4, "padding_right_steps": 12, "steps": 10, "layers": 5, "feature_fps": 50,
                   "layout": "(step, layer) -> 50 rows of 384"},
        "check": {"frames": check, "unet": args.unet, "units": "CPU_AND_NE",
                  "note": "composed in Python with the same Core ML models onto the shipped JPEG frame; face = raw 256 px decoder output (BGR png)"},
        "blend": {"fixed_point": True, "shrink_mask_bbox": True, "face_resize": "cv2 INTER_LINEAR"},
    }
    (out / "avatar.json").write_text(json.dumps(meta, indent=2) + "\n")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print(f"wrote {out / 'avatar.json'}; avatar bundle {size / 2**20:.0f} MB", flush=True)


if __name__ == "__main__":
    main()
