import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musetalk.utils.utils import load_all_model
from scripts.trt_runtime import load_vae_trt_decoder


def load_reference_vae(device: str | torch.device = "cuda"):
    vae, unet, pe = load_all_model(
        unet_model_path="./models/musetalkV15/unet.pth",
        vae_type="sd-vae",
        device=device,
    )
    del unet, pe

    resolved_device = torch.device(device)
    vae.vae = vae.vae.half().to(resolved_device).eval()
    vae.runtime_dtype = torch.float16
    return vae


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compare MuseTalk PyTorch VAE decode against the active TRT backend."
    )
    parser.add_argument(
        "--avatar-id",
        default="test_avatar",
        help="Prepared avatar id under results/v15/avatars/<avatar_id>.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="How many cached avatar latents to compare in one decode call.",
    )
    parser.add_argument(
        "--trt-dir",
        default="./models/tensorrt_altenv_bs32",
        help="Directory containing vae_decoder_trt.ts and metadata.",
    )
    parser.add_argument(
        "--backend",
        default=os.getenv("MUSETALK_VAE_BACKEND", "trt"),
        help="Backend to activate for comparison, e.g. trt or trt_stagewise.",
    )
    parser.add_argument(
        "--output-dir",
        default="./tmp/vae_backend_validation",
        help="Directory for JSON report and debug PNGs.",
    )
    return parser.parse_args()


def load_cached_latents(avatar_id: str) -> torch.Tensor:
    latents_path = Path(f"./results/v15/avatars/{avatar_id}/latents.pt")
    latents = torch.load(latents_path, map_location="cpu")
    if isinstance(latents, list):
        latents = torch.stack(latents, dim=0)
    if latents.dim() == 5 and latents.shape[1] == 1:
        latents = latents.squeeze(1)
    if latents.dim() != 4 or latents.shape[1] != 8:
        raise RuntimeError(
            f"Unexpected latent cache shape {tuple(latents.shape)} from {latents_path}"
        )
    # MuseTalk cached UNet inputs are [masked_latent(4), ref_latent(4)].
    # The VAE decode path expects the predicted 4-channel latent output.
    return latents[:, :4, :, :].contiguous()


def tensor_to_bgr_uint8(image_tensor: torch.Tensor) -> np.ndarray:
    image = image_tensor.detach().cpu().permute(0, 2, 3, 1).float().numpy()
    image = (image * 255.0).round().clip(0, 255).astype(np.uint8)
    return image[..., ::-1]


def summarize_tensor(name: str, tensor: torch.Tensor) -> dict:
    return {
        f"{name}_shape": list(tensor.shape),
        f"{name}_dtype": str(tensor.dtype),
        f"{name}_min": float(tensor.min().item()),
        f"{name}_max": float(tensor.max().item()),
        f"{name}_mean": float(tensor.mean().item()),
    }


def decode_backend_tensor(
    backend,
    latents: torch.Tensor,
    scaling_factor: float,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    if hasattr(backend, "decode"):
        return backend.decode(
            latents=latents,
            scaling_factor=scaling_factor,
            output_dtype=output_dtype,
        )

    output = backend(latents)
    if isinstance(output, (tuple, list)):
        output = output[0]
    if output_dtype is not None and output.dtype != output_dtype:
        output = output.to(dtype=output_dtype)
    return output


def compare_vae_backend_outputs(
    vae,
    backend,
    batch_latents: torch.Tensor,
    backend_name: str = "trt",
    output_dir: Path | None = None,
    extra_report_fields: dict | None = None,
) -> dict:
    with torch.no_grad():
        pytorch_image = vae.decode_latents_tensor(batch_latents)

    with torch.no_grad():
        backend_image = decode_backend_tensor(
            backend=backend,
            latents=batch_latents,
            scaling_factor=vae.scaling_factor,
            output_dtype=vae.runtime_dtype,
        )

    abs_diff = (pytorch_image.float() - backend_image.float()).abs()
    pytorch_bgr = tensor_to_bgr_uint8(pytorch_image)
    backend_bgr = tensor_to_bgr_uint8(backend_image)
    diff_bgr = (
        np.abs(pytorch_bgr.astype(np.int16) - backend_bgr.astype(np.int16))
        .clip(0, 255)
        .astype(np.uint8)
    )

    outputs = {}
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs = {
            "pytorch_face": str((output_dir / "pytorch_face.png").resolve()),
            f"{backend_name}_face": str((output_dir / f"{backend_name}_face.png").resolve()),
            "abs_diff": str((output_dir / "abs_diff.png").resolve()),
        }
        cv2.imwrite(outputs["pytorch_face"], pytorch_bgr[0])
        cv2.imwrite(outputs[f"{backend_name}_face"], backend_bgr[0])
        cv2.imwrite(outputs["abs_diff"], diff_bgr[0])

    report = {
        **summarize_tensor("pytorch", pytorch_image),
        **summarize_tensor(backend_name, backend_image),
        "mae": float(abs_diff.mean().item()),
        "max_abs": float(abs_diff.max().item()),
        "outputs": outputs,
    }
    if extra_report_fields:
        report.update(extra_report_fields)
    return report


def main():
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This validation script requires CUDA.")

    vae = load_reference_vae(device=device)

    cached_latents = load_cached_latents(args.avatar_id)
    batch_latents = cached_latents[: args.batch_size].to(device=device, dtype=torch.float16)

    os.environ["MUSETALK_TRT_ENABLED"] = "1"
    os.environ["MUSETALK_VAE_BACKEND"] = args.backend
    os.environ["MUSETALK_TRT_FALLBACK"] = "0"
    if args.backend.strip().lower() in {"trt", "tensorrt"}:
        os.environ["MUSETALK_TRT_DIR"] = str(Path(args.trt_dir).resolve())
    backend = load_vae_trt_decoder(
        device=device,
        scaling_factor=vae.scaling_factor,
        vae_module=vae.vae,
    )
    if backend is None:
        raise RuntimeError("TensorRT backend did not activate.")

    report = compare_vae_backend_outputs(
        vae=vae,
        backend=backend,
        batch_latents=batch_latents,
        backend_name="trt",
        output_dir=output_dir,
        extra_report_fields={
        "avatar_id": args.avatar_id,
        "backend": args.backend,
        "trt_dir": str(Path(args.trt_dir).resolve()),
        "batch_size": int(args.batch_size),
        },
    )

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
