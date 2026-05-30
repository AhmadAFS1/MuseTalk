import argparse
import json
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from musetalk.models.vae import VAE  # noqa: E402
from scripts.validate_vae_backend import load_cached_latents, load_calibration_latents  # noqa: E402


def _measure_cuda(fn, iterations: int, warmup: int) -> tuple[float, object]:
    result = None
    for _ in range(max(0, warmup)):
        result = fn()
    torch.cuda.synchronize()

    started_at = time.perf_counter()
    for _ in range(max(1, iterations)):
        result = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - started_at) / max(1, iterations), result


def _stage_inputs(vae, latents: torch.Tensor, up_block_index: int) -> torch.Tensor:
    decoder = vae.vae.decoder
    current = latents / vae.scaling_factor
    current = vae.vae.post_quant_conv(current)
    current = decoder.conv_in(current)
    current = decoder.mid_block(current, None)
    current = current.to(dtype=vae.runtime_dtype).contiguous()
    for index in range(up_block_index):
        current = decoder.up_blocks[index](current, None)
        current = current.to(dtype=vae.runtime_dtype).contiguous()
    return current


def _profile_resnet(resnet, input_tensor: torch.Tensor, iterations: int, warmup: int) -> tuple[dict, torch.Tensor]:
    timings: dict[str, float] = {}

    def step_norm1():
        return resnet.norm1(input_tensor)

    timings["norm1"], hidden = _measure_cuda(step_norm1, iterations, warmup)

    def step_act1():
        return resnet.nonlinearity(hidden)

    timings["act1"], hidden = _measure_cuda(step_act1, iterations, warmup)

    if resnet.upsample is not None:
        raise RuntimeError("This profiler does not expect per-resnet upsample in the VAE up block.")
    if resnet.downsample is not None:
        raise RuntimeError("This profiler does not expect per-resnet downsample in the VAE up block.")

    def step_conv1():
        return resnet.conv1(hidden)

    timings["conv1"], hidden = _measure_cuda(step_conv1, iterations, warmup)

    def step_norm2():
        return resnet.norm2(hidden)

    timings["norm2"], hidden = _measure_cuda(step_norm2, iterations, warmup)

    def step_act2():
        return resnet.nonlinearity(hidden)

    timings["act2"], hidden = _measure_cuda(step_act2, iterations, warmup)

    def step_dropout():
        return resnet.dropout(hidden)

    timings["dropout"], hidden = _measure_cuda(step_dropout, iterations, warmup)

    def step_conv2():
        return resnet.conv2(hidden)

    timings["conv2"], hidden = _measure_cuda(step_conv2, iterations, warmup)

    shortcut = input_tensor
    if resnet.conv_shortcut is not None:
        def step_shortcut():
            return resnet.conv_shortcut(input_tensor)

        timings["conv_shortcut"], shortcut = _measure_cuda(step_shortcut, iterations, warmup)

    def step_residual_add():
        return (shortcut + hidden) / resnet.output_scale_factor

    timings["residual_add"], output = _measure_cuda(step_residual_add, iterations, warmup)
    return timings, output.contiguous()


def _profile_up_block(block, input_tensor: torch.Tensor, iterations: int, warmup: int) -> dict:
    report: dict = {
        "block_class": type(block).__name__,
        "input_shape": list(input_tensor.shape),
        "input_dtype": str(input_tensor.dtype),
        "children": [],
    }

    def full_block():
        return block(input_tensor, None)

    full_time, _ = _measure_cuda(full_block, iterations, warmup)
    report["full_block_avg_s"] = full_time

    current = input_tensor
    child_total = 0.0
    for index, resnet in enumerate(block.resnets):
        def run_resnet():
            return resnet(current, None)

        child_time, output = _measure_cuda(run_resnet, iterations, warmup)
        internals, current = _profile_resnet(resnet, current, iterations, warmup)
        child_total += child_time
        report["children"].append(
            {
                "name": f"resnets.{index}",
                "class": type(resnet).__name__,
                "input_shape": list(output.shape),
                "avg_s": child_time,
                "internal_avg_s": internals,
                "internal_sum_s": sum(internals.values()),
            }
        )

    if block.upsamplers is not None:
        for index, upsampler in enumerate(block.upsamplers):
            def run_upsampler():
                return upsampler(current)

            child_time, current = _measure_cuda(run_upsampler, iterations, warmup)
            child_total += child_time
            report["children"].append(
                {
                    "name": f"upsamplers.{index}",
                    "class": type(upsampler).__name__,
                    "output_shape": list(current.shape),
                    "avg_s": child_time,
                }
            )

    report["child_sum_s"] = child_total
    return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Profile internals of one MuseTalk VAE decoder up block."
    )
    parser.add_argument("--avatar-id", default="test_avatar_2")
    parser.add_argument("--calibration-dir", default="./calibration/vae_decoder")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--up-block-index", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument(
        "--output-json",
        default="./tmp/vae_up_block_profile/up_block_3_bs16.json",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for VAE up-block profiling.")

    device = torch.device("cuda:0")
    vae = VAE(model_path="./models/sd-vae/", use_float16=True)
    vae.vae = vae.vae.half().to(device).eval()
    vae.runtime_dtype = torch.float16
    if args.calibration_dir:
        latents = load_calibration_latents(args.calibration_dir, args.batch_size)
    else:
        latents = load_cached_latents(args.avatar_id)[: args.batch_size]
    latents = latents.to(device=device, dtype=vae.runtime_dtype).contiguous()

    with torch.no_grad():
        block_input = _stage_inputs(vae, latents, args.up_block_index)
        block = vae.vae.decoder.up_blocks[args.up_block_index].eval()
        report = _profile_up_block(block, block_input, args.iterations, args.warmup)

    report.update(
        {
            "avatar_id": args.avatar_id,
            "calibration_dir": args.calibration_dir,
            "batch_size": args.batch_size,
            "up_block_index": args.up_block_index,
            "iterations": args.iterations,
            "warmup": args.warmup,
        }
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))
    print(f"Wrote profile report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
