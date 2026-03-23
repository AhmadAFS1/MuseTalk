import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tensorrt_export import (
    VAEDecodeTRTWrapper,
    parse_precision,
    require_torch_tensorrt,
    resolve_torch_executed_ops,
)
from scripts.validate_vae_backend import (
    compare_vae_backend_outputs,
    load_cached_latents,
    load_reference_vae,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile the MuseTalk VAE to TRT in memory and compare its output against PyTorch before any save/load step."
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
        help="Exact batch size to compile and validate.",
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Precision to use for the in-memory TRT compile.",
    )
    parser.add_argument(
        "--workspace-gb",
        type=float,
        default=1.0,
        help="TensorRT workspace size in GB.",
    )
    parser.add_argument(
        "--min-block-size",
        type=int,
        default=3,
        help="Torch-TensorRT min_block_size.",
    )
    parser.add_argument(
        "--output-dir",
        default="./tmp/vae_trt_inmemory_validation",
        help="Directory for the validation report and debug PNGs.",
    )
    parser.add_argument(
        "--torch-executed-op",
        action="append",
        default=[],
        help=(
            "Keep selected ops on the PyTorch side during TRT compilation. "
            "Supported: native_group_norm, group_norm, scaled_dot_product_attention."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This validation script requires CUDA.")

    precision = parse_precision(args.precision)
    torch_executed_ops = resolve_torch_executed_ops(args.torch_executed_op)
    require_torch_tensorrt()
    import torch_tensorrt

    vae = load_reference_vae(device=device)
    vae.vae = vae.vae.to(device=device, dtype=precision).eval()
    vae.runtime_dtype = precision

    batch_size = max(1, int(args.batch_size))
    latents = load_cached_latents(args.avatar_id)[:batch_size].to(device=device, dtype=precision)

    wrapper = VAEDecodeTRTWrapper(vae.vae, float(vae.scaling_factor)).eval().to(
        device=device,
        dtype=precision,
    )

    torch.cuda.empty_cache()
    inputs = [
        torch_tensorrt.Input(
            min_shape=(batch_size, 4, 32, 32),
            opt_shape=(batch_size, 4, 32, 32),
            max_shape=(batch_size, 4, 32, 32),
            dtype=precision,
        )
    ]
    compiled = torch_tensorrt.compile(
        wrapper,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions={precision},
        workspace_size=int(float(args.workspace_gb) * (1 << 30)),
        min_block_size=int(args.min_block_size),
        require_full_compilation=not bool(torch_executed_ops),
        pass_through_build_failures=False,
        torch_executed_ops=torch_executed_ops,
    )

    report = compare_vae_backend_outputs(
        vae=vae,
        backend=compiled,
        batch_latents=latents,
        backend_name="trt_inmem",
        output_dir=Path(args.output_dir),
        extra_report_fields={
            "avatar_id": args.avatar_id,
            "batch_size": batch_size,
            "precision": args.precision,
            "mode": "in_memory_compile",
            "torch_executed_ops": list(args.torch_executed_op),
        },
    )

    report_path = Path(args.output_dir) / "report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
