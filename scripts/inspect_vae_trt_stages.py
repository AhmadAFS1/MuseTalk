import argparse
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.tensorrt_export import (
    parse_precision,
    require_torch_tensorrt,
    resolve_torch_executed_ops,
)
from scripts.validate_vae_backend import load_cached_latents, load_reference_vae, summarize_tensor


LATENT_H = 32
LATENT_W = 32


class ScalePostQuantStage(torch.nn.Module):
    def __init__(self, vae_module: torch.nn.Module, scaling_factor: float):
        super().__init__()
        self.post_quant_conv = vae_module.post_quant_conv
        self.register_buffer(
            "scaling_factor_tensor",
            torch.tensor(float(scaling_factor), dtype=torch.float32),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        scaled = latent / self.scaling_factor_tensor.to(device=latent.device, dtype=latent.dtype)
        return self.post_quant_conv(scaled)


class DecoderConvInStage(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.conv_in = decoder.conv_in

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.conv_in(sample)


class DecoderMidBlockStage(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.mid_block = decoder.mid_block
        self.upscale_dtype = next(iter(decoder.up_blocks.parameters())).dtype

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.mid_block(sample, None)
        return sample.to(self.upscale_dtype)


class DecoderUpBlockStage(torch.nn.Module):
    def __init__(self, up_block: torch.nn.Module):
        super().__init__()
        self.up_block = up_block

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.up_block(sample, None)


class DecoderPostprocessStage(torch.nn.Module):
    def __init__(self, decoder: torch.nn.Module):
        super().__init__()
        self.conv_norm_out = decoder.conv_norm_out
        self.conv_act = decoder.conv_act
        self.conv_out = decoder.conv_out

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)
        return sample


class OutputNormalizeStage(torch.nn.Module):
    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return (sample / 2 + 0.5).clamp(0, 1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compile MuseTalk VAE decoder stages one by one in TRT and report the first stage that diverges from PyTorch."
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
        help="Exact batch size to feed each stage.",
    )
    parser.add_argument(
        "--precision",
        choices=["fp16", "fp32"],
        default="fp16",
        help="Precision to use for stage compilation.",
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
        "--fail-mae",
        type=float,
        default=0.05,
        help="Threshold used to flag the first divergent stage.",
    )
    parser.add_argument(
        "--output-json",
        default="./tmp/vae_trt_stage_inspection/report.json",
        help="Where to write the stage comparison report.",
    )
    parser.add_argument(
        "--stage",
        action="append",
        default=[],
        help=(
            "Restrict inspection to one or more named stages. "
            "Examples: decoder_mid_block, decoder_up_block_0."
        ),
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


def compare_tensors(stage_name: str, pytorch_tensor: torch.Tensor, trt_tensor: torch.Tensor) -> dict:
    abs_diff = (pytorch_tensor.float() - trt_tensor.float()).abs()
    return {
        "stage": stage_name,
        **summarize_tensor("pytorch", pytorch_tensor),
        **summarize_tensor("trt", trt_tensor),
        "mae": float(abs_diff.mean().item()),
        "max_abs": float(abs_diff.max().item()),
    }


def compile_stage(
    stage_module,
    sample_input,
    precision,
    workspace_gb,
    min_block_size,
    torch_executed_ops,
):
    import torch_tensorrt

    sample_input = sample_input.contiguous()
    shape = tuple(int(dim) for dim in sample_input.shape)
    inputs = [
        torch_tensorrt.Input(
            min_shape=shape,
            opt_shape=shape,
            max_shape=shape,
            dtype=precision,
        )
    ]
    torch.cuda.empty_cache()
    return torch_tensorrt.compile(
        stage_module,
        ir="dynamo",
        inputs=inputs,
        enabled_precisions={precision},
        workspace_size=int(float(workspace_gb) * (1 << 30)),
        min_block_size=int(min_block_size),
        require_full_compilation=not bool(torch_executed_ops),
        pass_through_build_failures=False,
        torch_executed_ops=torch_executed_ops,
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("This stage inspection script requires CUDA.")

    precision = parse_precision(args.precision)
    require_torch_tensorrt()
    torch_executed_ops = resolve_torch_executed_ops(args.torch_executed_op)
    requested_stages = {name.strip() for name in args.stage if name.strip()}

    vae = load_reference_vae(device=device)
    vae.vae = vae.vae.to(device=device, dtype=precision).eval()
    vae.runtime_dtype = precision

    batch_size = max(1, int(args.batch_size))
    latents = load_cached_latents(args.avatar_id)[:batch_size].to(device=device, dtype=precision)

    decoder = vae.vae.decoder
    stages = [
        ("scale_post_quant", ScalePostQuantStage(vae.vae, vae.scaling_factor)),
        ("decoder_conv_in", DecoderConvInStage(decoder)),
        ("decoder_mid_block", DecoderMidBlockStage(decoder)),
        ("decoder_up_block_0", DecoderUpBlockStage(decoder.up_blocks[0])),
        ("decoder_up_block_1", DecoderUpBlockStage(decoder.up_blocks[1])),
        ("decoder_up_block_2", DecoderUpBlockStage(decoder.up_blocks[2])),
        ("decoder_up_block_3", DecoderUpBlockStage(decoder.up_blocks[3])),
        ("decoder_postprocess", DecoderPostprocessStage(decoder)),
        ("output_normalize", OutputNormalizeStage()),
    ]
    if requested_stages and not any(name in requested_stages for name, _ in stages):
        raise RuntimeError(
            f"No matching stages for --stage values: {sorted(requested_stages)}"
        )

    current = latents
    report_rows = []
    first_bad_stage = None

    for stage_name, stage_module in stages:
        stage_module = stage_module.eval().to(device=device, dtype=precision)
        with torch.no_grad():
            pytorch_output = stage_module(current)

        inspect_this_stage = not requested_stages or stage_name in requested_stages
        if not inspect_this_stage:
            current = pytorch_output
            continue

        try:
            compiled_stage = compile_stage(
                stage_module=stage_module,
                sample_input=current,
                precision=precision,
                workspace_gb=args.workspace_gb,
                min_block_size=args.min_block_size,
                torch_executed_ops=torch_executed_ops,
            )
            with torch.no_grad():
                trt_output = compiled_stage(current.contiguous())
                if isinstance(trt_output, (tuple, list)):
                    trt_output = trt_output[0]
            if trt_output.dtype != pytorch_output.dtype:
                trt_output = trt_output.to(dtype=pytorch_output.dtype)

            row = compare_tensors(stage_name, pytorch_output, trt_output)
            row["compile_failed"] = False
            row["passed"] = bool(row["mae"] <= float(args.fail_mae))
        except Exception as exc:
            row = {
                "stage": stage_name,
                "compile_failed": True,
                "passed": False,
                "error": f"{type(exc).__name__}: {exc}",
            }

        report_rows.append(row)
        if first_bad_stage is None and not row.get("passed", False):
            first_bad_stage = stage_name

        current = pytorch_output

    payload = {
        "avatar_id": args.avatar_id,
        "batch_size": batch_size,
        "precision": args.precision,
        "fail_mae": float(args.fail_mae),
        "torch_executed_ops": list(args.torch_executed_op),
        "requested_stages": sorted(requested_stages),
        "first_bad_stage": first_bad_stage,
        "stages": report_rows,
    }

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
