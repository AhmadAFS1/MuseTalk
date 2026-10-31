#!/usr/bin/env python
"""Export MuseTalk's audio front end (Whisper-tiny) to Core ML.

MuseTalk conditions the UNet on Whisper-tiny *encoder* features: the log-mel
spectrogram of a 30 s segment goes through the 4-layer encoder, the five hidden
states (embeddings + one per layer, the last after the final layer norm) are
stacked, and `AudioProcessor.build_audio_prompts` gathers the 50x384 window per
video frame. Two models come out of here, mirroring `musetalk/utils/audio_processor.py`:

  whisper_logmel_fp32   waveform [1, 480000] (30 s at 16 kHz, zero padded) -> log-mel [1, 80, 3000]
                        (WhisperFeatureExtractor's torch path, written with conv1d so it
                        converts without the complex-number dialect; fp32, meant for the CPU)
  whisper_encoder_fp16  log-mel [1, 80, 3000] -> features [1, 1500, 5, 384]
                        (= torch.stack(encoder(...).hidden_states, dim=2), the server's layout)

Usage (repo root, .venv-ct):
  python scripts/ios/export_whisper_coreml.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

OUT_DIR = REPO_ROOT / "models" / "ios_bench" / "coreml"
WHISPER_DIR = REPO_ROOT / "models" / "whisper"
SAMPLES_30S = 30 * 16000


class LogMel(torch.nn.Module):
    """WhisperFeatureExtractor._torch_extract_fbank_features without torch.stft.

    STFT as a strided conv1d with the windowed DFT basis (center=True reflect
    padding like torch.stft), power spectrum, mel filter bank, log10, the
    8 dB dynamic-range floor, and the (x + 4) / 4 normalisation.
    """

    def __init__(self, fe):
        super().__init__()
        n_fft, hop = fe.n_fft, fe.hop_length
        self.hop, self.pad = hop, n_fft // 2
        window = torch.hann_window(n_fft, dtype=torch.float64)
        n = torch.arange(n_fft, dtype=torch.int64)
        k = torch.arange(n_fft // 2 + 1, dtype=torch.int64)
        # k*n mod N in integers, then float64: a float32 angle of ~1e3 rad is only good to
        # 1e-4 rad, and that leakage lands in bins 80 dB down, which the floor keeps.
        angle = 2 * math.pi * ((k[:, None] * n[None, :]) % n_fft).to(torch.float64) / n_fft
        basis = (torch.cat([torch.cos(angle), -torch.sin(angle)], 0) * window).to(torch.float32)  # [2*201, 400]
        self.register_buffer("basis", basis[:, None, :])  # conv1d weight [402, 1, 400]
        self.register_buffer("mel_filters", torch.from_numpy(fe.mel_filters.T.astype(np.float32)))  # [80, 201]
        self.n_bins = n_fft // 2 + 1

    def forward(self, waveform):  # [1, 480000]
        x = F.pad(waveform[:, None, :], (self.pad, self.pad), mode="reflect")
        spec = F.conv1d(x, self.basis, stride=self.hop)[..., :-1]  # [1, 402, 3000]
        power = spec[:, :self.n_bins] ** 2 + spec[:, self.n_bins:] ** 2
        mel = torch.matmul(self.mel_filters, power)  # [1, 80, 3000]
        log_spec = torch.clamp(mel, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.amax(dim=(1, 2), keepdim=True) - 8.0)
        return (log_spec + 4.0) / 4.0


class EncoderStack(torch.nn.Module):
    """log-mel [1, 80, 3000] -> [1, 1500, 5, 384], the stacked hidden states MuseTalk windows."""

    def __init__(self, encoder):
        super().__init__()
        self.conv1, self.conv2 = encoder.conv1, encoder.conv2
        self.register_buffer("pos", encoder.embed_positions.weight.detach().clone())
        self.layers = encoder.layers
        self.layer_norm = encoder.layer_norm

    def forward(self, input_features):
        h = F.gelu(self.conv1(input_features))
        h = F.gelu(self.conv2(h)).permute(0, 2, 1) + self.pos
        states = [h]
        for layer in self.layers:
            out = layer(h, None)
            h = out[0] if isinstance(out, tuple) else out
            states.append(h)
        states[-1] = self.layer_norm(h)
        return torch.stack(states, dim=2)


def check_against_transformers(stack: EncoderStack, encoder, mel: torch.Tensor) -> float:
    """The export must reproduce `encoder(..., output_hidden_states=True).hidden_states`."""
    with torch.no_grad():
        ref = torch.stack(encoder(mel, output_hidden_states=True).hidden_states, dim=2)
        ours = stack(mel)
    err = (ref - ours).abs().amax(dim=(0, 1, 3))  # per hidden-state slot
    print(f"  hidden-state slots max |diff| vs transformers: {[f'{e:.2e}' for e in err.tolist()]}", flush=True)
    if err.max() > 1e-4:
        raise SystemExit("EncoderStack does not match transformers' hidden_states; fix the slot layout")
    return float(err.max())


def convert(traced, inputs, out_name, precision, target):
    import coremltools as ct

    t0 = time.time()
    model = ct.convert(traced, inputs=inputs, outputs=[ct.TensorType(name=out_name)], convert_to="mlprogram",
                       minimum_deployment_target=target, compute_precision=precision,
                       compute_units=ct.ComputeUnit.ALL)
    print(f"  converted in {time.time() - t0:.0f}s", flush=True)
    return model


def main() -> None:
    import coremltools as ct
    import librosa
    from transformers import AutoFeatureExtractor, WhisperModel

    ap = argparse.ArgumentParser()
    ap.add_argument("--whisper-dir", type=Path, default=WHISPER_DIR)
    ap.add_argument("--audio", type=Path, default=REPO_ROOT / "data" / "audio" / "ai-assistant.mpga")
    ap.add_argument("--target", default="iOS17", choices=["iOS17", "iOS18"])
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    args = ap.parse_args()
    target = getattr(ct.target, args.target)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.out_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    def record(name, path, extra):
        size = sum(p.stat().st_size for p in Path(path).rglob("*") if p.is_file())
        manifest[name] = {"path": str(path), "bytes": size, **extra}
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"  saved {path} ({size / 2**20:.1f} MB)", flush=True)

    fe = AutoFeatureExtractor.from_pretrained(str(args.whisper_dir))
    whisper = WhisperModel.from_pretrained(str(args.whisper_dir), attn_implementation="eager").eval()
    encoder = whisper.encoder.requires_grad_(False)

    # Real audio, the same first 30 s segment AudioProcessor.get_audio_feature builds.
    wav, _ = librosa.load(str(args.audio), sr=16000)
    seg = np.zeros(SAMPLES_30S, dtype=np.float32)
    seg[:min(len(wav), SAMPLES_30S)] = wav[:SAMPLES_30S]
    ref_mel = fe(seg, return_tensors="pt", sampling_rate=16000).input_features  # [1, 80, 3000]

    # --- log-mel ---------------------------------------------------------------
    print("log-mel: torch module vs WhisperFeatureExtractor", flush=True)
    logmel = LogMel(fe).eval()
    with torch.no_grad():
        ours = logmel(torch.from_numpy(seg)[None])
    print(f"  max |diff| {float((ours - ref_mel).abs().max()):.2e} (fp32 torch)", flush=True)
    traced = torch.jit.trace(logmel, torch.from_numpy(seg)[None])
    model = convert(traced, [ct.TensorType(name="waveform", shape=(1, SAMPLES_30S), dtype=np.float32)],
                    "logmel", ct.precision.FLOAT32, target)
    cm = np.asarray(model.predict({"waveform": seg[None]})["logmel"], dtype=np.float32)
    logmel_err = float(np.abs(cm - ref_mel.numpy()).max())
    print(f"  Core ML (fp32, CPU) max |diff| vs extractor {logmel_err:.2e}", flush=True)
    path = args.out_dir / "whisper_logmel_fp32.mlpackage"
    model.save(str(path))
    record("whisper_logmel_fp32", path, {"kind": "whisper_logmel", "variant": "fp32", "batch": 1,
                                         "max_abs_err_vs_torch": logmel_err})

    # --- encoder ---------------------------------------------------------------
    print("encoder: stacked hidden states", flush=True)
    stack = EncoderStack(encoder).eval()
    check_against_transformers(stack, encoder, ref_mel)
    with torch.no_grad():
        ref_feat = stack(ref_mel)
    traced = torch.jit.trace(stack, ref_mel, check_trace=False)
    model = convert(traced, [ct.TensorType(name="logmel", shape=(1, 80, 3000), dtype=np.float16)],
                    "features", ct.precision.FLOAT16, target)
    cm = np.asarray(model.predict({"logmel": ref_mel.numpy().astype(np.float16)})["features"], dtype=np.float32)
    diff = np.abs(cm - ref_feat.numpy())
    print(f"  Core ML fp16 vs fp32 torch: max |diff| {diff.max():.3f}, mean {diff.mean():.4f}, "
          f"ref |x| mean {np.abs(ref_feat.numpy()).mean():.3f}", flush=True)
    path = args.out_dir / "whisper_encoder_fp16.mlpackage"
    model.save(str(path))
    record("whisper_encoder_fp16", path, {"kind": "whisper_encoder", "variant": "fp16", "batch": 1,
                                          "max_abs_err_vs_fp32": float(diff.max()),
                                          "mean_abs_err_vs_fp32": float(diff.mean())})


if __name__ == "__main__":
    main()
