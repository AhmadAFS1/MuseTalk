"""ONNX-parser stagewise FP16 TensorRT UNet for MuseTalk v1.5 (300 fps plan items 2.2 / 2.3a / 2.3b).

The UNet2DConditionModel is split at its 11 top-level blocks (conv_in, down0-3, mid, up0-3,
conv_norm_out+act+conv_out). Each block is exported to ONNX in RAM (never written to disk) and
built into its own FP16 TensorRT engine through the ONNX parser (GroupNorm -> InstanceNorm ->
Myelin fusions that the torch_tensorrt path does not get). At runtime the 11 engines are chained
exactly like UNet2DConditionModel.forward, with the skip tensors in dedicated static buffers:

  x -> head -> h0 -> down0 -> (d0r0, d0r1, d0ds) -> down1 -> ... -> down3 -> (d3r0, d3r1)
    -> mid -> m -> up0(m, d2ds, d3r0, d3r1) -> up1(.., d1ds, d2r0, d2r1) -> up2(.., d0ds, d1r0, d1r1)
    -> up3(.., h0, d0r0, d0r1) -> tail -> out

- One raw IExecutionContext per block, created with USER_MANAGED device memory and pointed at a
  single shared scratch arena (the contexts run back to back on one stream).
- Multi-input / multi-output bindings by name (i<k> / o<k> -> spec keys).
- `execute_async_v3` for all 11 contexts, captured once into one torch.cuda.CUDAGraph; every call
  copies its inputs into the static input buffers, replays, and returns a CLONE of the output.
- The t=0 time embedding is computed once at build time and baked into each block.
- The engine batch is fixed (MUSETALK_UNET_STAGEWISE_BATCH, 16 or 8); smaller batches are padded
  (only the real rows are returned), larger batches are split into engine-batch chunks.

Selected by MUSETALK_UNET_BACKEND=trt_stagewise in scripts/trt_runtime.load_unet_trt_backend; the
default (unset / trt) path is unchanged. Engines live under
  $MUSETALK_UNET_STAGEWISE_CACHE_DIR (default models/tensorrt_unet_stagewise_sm89)/bs<N>/
with manifest.json (TensorRT version, GPU, per-block ONNX hash, build flags, probe output hash).
Build them with scripts/build_unet_stagewise.py (under scripts/box_guard.sh).
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import torch

logger = logging.getLogger("unet_stagewise_trt")

ROOT = Path(__file__).resolve().parent.parent
BLOCK_ORDER = ["head", "down0", "down1", "down2", "down3", "mid", "up0", "up1", "up2", "up3", "tail"]
# Source-prefix cache variant (engine-set manifest "variant": "srccache"): conv_in and down_blocks[0].resnets[0]
# see only the avatar latents (t=0 is baked; audio first enters at down_blocks[0].attentions[0]), so their
# outputs h0 (skip r0) and a0 are computed once per source frame by a separate "prefix" engine and gathered
# per batch; the per-frame chain starts at "down0rest" (the exact remainder of CrossAttnDownBlock2D.forward).
SRCCACHE_BLOCK_ORDER = ["down0rest", "down1", "down2", "down3", "mid", "up0", "up1", "up2", "up3", "tail"]
PREFIX_SPEC = {"name": "prefix", "inputs": ["x"], "outputs": ["h0", "a0"]}
VARIANTS = ("default", "srccache")


def block_order(variant: Optional[str] = None) -> list:
    """Engine names of a set: its chain blocks plus the prefix for the srccache variant."""
    if variant in (None, "default"):
        return list(BLOCK_ORDER)
    if variant == "srccache":
        return ["prefix"] + SRCCACHE_BLOCK_ORDER
    raise RuntimeError(f"Unknown stagewise UNet variant {variant!r}")
MANIFEST_SCHEMA = "musetalk_unet_stagewise_trt_v1"
LATENT_CHW = (8, 32, 32)
AUDIO_TD = (50, 384)
OUT_CHW = (4, 32, 32)
PROBE_SEED = 20260928


# --------------------------------------------------------------------------- env
def default_cache_root() -> Path:
    raw = os.getenv("MUSETALK_UNET_STAGEWISE_CACHE_DIR", "").strip()
    path = Path(raw) if raw else Path("models/tensorrt_unet_stagewise_sm89")
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def requested_stagewise_batch() -> int:
    raw = os.getenv("MUSETALK_UNET_STAGEWISE_BATCH", "16").strip() or "16"
    batch = int(raw)
    if batch <= 0:
        raise RuntimeError(f"Invalid MUSETALK_UNET_STAGEWISE_BATCH={raw!r}")
    return batch


def _env_on(name: str, default: str) -> bool:
    return os.getenv(name, default).strip().lower() not in {"0", "false", "no", "off", ""}


# --------------------------------------------------------------------------- block wrappers
def _import_nn():
    import torch.nn as nn

    return nn


def make_block_wrappers(model, emb: torch.Tensor) -> dict:
    """Per-block nn.Modules with exactly the inputs each block uses (ONNX input order = spec order)."""
    nn = _import_nn()

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_in = model.conv_in

        def forward(self, x):
            return self.conv_in(x)

    class Down(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block
            self.register_buffer("emb", emb.clone())

        def forward(self, h, ehs):
            _, res = self.block(hidden_states=h, temb=self.emb, encoder_hidden_states=ehs)
            return tuple(res)  # res[-1] is the block output h; returned once only

    class DownPlain(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block
            self.register_buffer("emb", emb.clone())

        def forward(self, h):
            _, res = self.block(hidden_states=h, temb=self.emb)
            return tuple(res)

    class Mid(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block
            self.register_buffer("emb", emb.clone())

        def forward(self, h, ehs):
            return self.block(h, self.emb, encoder_hidden_states=ehs)

    class Up(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block
            self.register_buffer("emb", emb.clone())

        def forward(self, h, r0, r1, r2, ehs):
            return self.block(hidden_states=h, temb=self.emb, res_hidden_states_tuple=(r0, r1, r2),
                              encoder_hidden_states=ehs, upsample_size=None)

    class UpPlain(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block
            self.register_buffer("emb", emb.clone())

        def forward(self, h, r0, r1, r2):
            return self.block(hidden_states=h, temb=self.emb, res_hidden_states_tuple=(r0, r1, r2),
                              upsample_size=None)

    class Tail(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = model.conv_norm_out
            self.act = model.conv_act
            self.conv_out = model.conv_out

        def forward(self, h):
            return self.conv_out(self.act(self.norm(h)))

    down0 = model.down_blocks[0]

    class Prefix(nn.Module):
        """conv_in + down_blocks[0].resnets[0]: depends only on the latent (source-prefix cache)."""

        def __init__(self):
            super().__init__()
            self.conv_in = model.conv_in
            self.resnet0 = down0.resnets[0]
            self.register_buffer("emb", emb.clone())

        def forward(self, x):
            h0 = self.conv_in(x)
            return h0, self.resnet0(h0, self.emb)

    class Down0Rest(nn.Module):
        """The rest of CrossAttnDownBlock2D.forward after resnets[0], op for op (diffusers 0.30)."""

        def __init__(self):
            super().__init__()
            self.block = down0
            self.register_buffer("emb", emb.clone())

        def forward(self, a0, ehs):
            b = self.block
            h = b.attentions[0](a0, encoder_hidden_states=ehs, cross_attention_kwargs=None, attention_mask=None,
                                encoder_attention_mask=None, return_dict=False)[0]
            out = (h,)
            for resnet, attn in list(zip(b.resnets, b.attentions))[1:]:
                h = resnet(h, self.emb)
                h = attn(h, encoder_hidden_states=ehs, cross_attention_kwargs=None, attention_mask=None,
                         encoder_attention_mask=None, return_dict=False)[0]
                out = out + (h,)
            if b.downsamplers is not None:
                for downsampler in b.downsamplers:
                    h = downsampler(h)
                out = out + (h,)
            return out

    wrappers = {"head": Head(), "tail": Tail(), "mid": Mid(model.mid_block), "prefix": Prefix(),
                "down0rest": Down0Rest()}
    for i, block in enumerate(model.down_blocks):
        cross = bool(getattr(block, "has_cross_attention", False))
        wrappers[f"down{i}"] = Down(block) if cross else DownPlain(block)
    for i, block in enumerate(model.up_blocks):
        cross = bool(getattr(block, "has_cross_attention", False))
        wrappers[f"up{i}"] = Up(block) if cross else UpPlain(block)
    return {k: v.eval() for k, v in wrappers.items()}


def chain_spec(model, variant: str = "default") -> list[dict]:
    """Input/output tensor keys of every block, mirroring UNet2DConditionModel.forward.

    variant "srccache": head+down0 are replaced by down0rest(a0, ehs); h0/a0 are chain inputs
    (produced per source frame by PREFIX_SPEC).
    """
    spec = _default_chain_spec(model)
    if variant in (None, "default"):
        return spec
    if variant != "srccache":
        raise RuntimeError(f"Unknown stagewise UNet variant {variant!r}")
    d0 = next(s for s in spec if s["name"] == "down0")
    if d0["inputs"] != ["h0", "ehs"] or len(model.down_blocks[0].resnets) < 1:
        raise RuntimeError(f"srccache needs a cross-attention down0 fed by conv_in; got {d0}")
    spec = ([{"name": "down0rest", "inputs": ["a0", "ehs"], "outputs": list(d0["outputs"])}]
            + [s for s in spec if s["name"] not in ("head", "down0")])
    assert [s["name"] for s in spec] == SRCCACHE_BLOCK_ORDER, [s["name"] for s in spec]
    return spec


def _default_chain_spec(model) -> list[dict]:
    spec = [{"name": "head", "inputs": ["x"], "outputs": ["h0"]}]
    res_stack = ["h0"]
    cur = "h0"
    for i, block in enumerate(model.down_blocks):
        cross = bool(getattr(block, "has_cross_attention", False))
        n_out = len(block.resnets) + (1 if block.downsamplers is not None else 0)
        outs = [f"d{i}r{j}" for j in range(len(block.resnets))]
        if block.downsamplers is not None:
            outs.append(f"d{i}ds")
        assert len(outs) == n_out
        spec.append({"name": f"down{i}", "inputs": [cur] + (["ehs"] if cross else []), "outputs": outs})
        res_stack += outs
        cur = outs[-1]
    spec.append({"name": "mid", "inputs": [cur, "ehs"], "outputs": ["m"]})
    cur = "m"
    for i, block in enumerate(model.up_blocks):
        cross = bool(getattr(block, "has_cross_attention", False))
        n = len(block.resnets)
        res = res_stack[-n:]
        res_stack = res_stack[:-n]
        spec.append({"name": f"up{i}", "inputs": [cur] + res + (["ehs"] if cross else []), "outputs": [f"u{i}"]})
        cur = f"u{i}"
    spec.append({"name": "tail", "inputs": [cur], "outputs": ["out"]})
    assert [s["name"] for s in spec] == BLOCK_ORDER, [s["name"] for s in spec]
    assert not res_stack, res_stack
    return spec


@torch.no_grad()
def time_embedding_t0(model, batch: int, device) -> torch.Tensor:
    """emb for timestep 0 exactly as UNet2DConditionModel.forward computes it (batch-expanded), row 0."""
    timesteps = torch.tensor([0], device=device).expand(batch)
    t_emb = model.time_proj(timesteps).to(dtype=model.dtype)
    emb = model.time_embedding(t_emb, None)
    return emb[:1].detach().clone()


@torch.no_grad()
def trace_block_inputs(model, spec: list[dict], wrappers: dict, latent: torch.Tensor, ehs: torch.Tensor):
    """Run the wrappers in chain order (eager). Returns (tensors by key, final output)."""
    tensors = {"x": latent, "ehs": ehs}
    if any("a0" in blk["inputs"] for blk in spec):
        tensors["h0"], tensors["a0"] = wrappers["prefix"](latent)
    for blk in spec:
        args = [tensors[k] for k in blk["inputs"]]
        out = wrappers[blk["name"]](*args)
        outs = list(out) if isinstance(out, (tuple, list)) else [out]
        assert len(outs) == len(blk["outputs"]), (blk["name"], len(outs))
        for k, t in zip(blk["outputs"], outs):
            tensors[k] = t
    return tensors, tensors["out"]


def export_block_onnx(wrapper, args: list[torch.Tensor], opset: int = 17) -> bytes:
    n_out = None
    with torch.no_grad():
        ref = wrapper(*args)
        n_out = len(ref) if isinstance(ref, (tuple, list)) else 1
    buf = io.BytesIO()
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            tuple(args),
            buf,
            opset_version=opset,
            input_names=[f"i{k}" for k in range(len(args))],
            output_names=[f"o{k}" for k in range(n_out)],
            do_constant_folding=True,
        )
    return buf.getvalue()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def build_engine_from_onnx(
    onnx_bytes: bytes,
    *,
    opt_level: int = 3,
    workspace_gb: float = 2.0,
    timing_cache: Optional[bytes] = None,
    use_timing_cache: bool = True,
    log_severity: str = "ERROR",
    int8: bool = False,
):
    """ONNX bytes -> (serialized FP16 engine bytes, build seconds, updated timing cache bytes).

    int8=True additionally sets BuilderFlag.INT8 for ONNX graphs that carry explicit Q/DQ nodes
    (modelopt fake-quant export); layers without Q/DQ stay FP16.
    """
    import tensorrt as trt

    trt_logger = trt.Logger(getattr(trt.Logger, log_severity))
    builder = trt.Builder(trt_logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, trt_logger)
    if not parser.parse(onnx_bytes):
        errors = " | ".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError("ONNX parse failed: " + errors[:1200])
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        config.set_flag(trt.BuilderFlag.INT8)
    config.builder_optimization_level = int(opt_level)
    # Build-time only: timing iterations per tactic (TensorRT default 1). More iterations make the
    # kernel choice less sensitive to single noisy samples on a power-capped GPU.
    avg_iters = int(os.getenv("MUSETALK_TRT_AVG_TIMING_ITERS", "0") or 0)
    if avg_iters > 0:
        config.avg_timing_iterations = avg_iters
    cache = None
    if use_timing_cache:
        cache = config.create_timing_cache(timing_cache or b"")
        config.set_timing_cache(cache, ignore_mismatch=True)
    else:
        config.set_flag(trt.BuilderFlag.DISABLE_TIMING_CACHE)
    started = time.time()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.time() - started
    if serialized is None:
        raise RuntimeError("TensorRT engine build failed")
    engine_bytes = bytes(serialized)
    del serialized
    cache_out = bytes(config.get_timing_cache().serialize()) if use_timing_cache else None
    del parser, network, config, builder
    return engine_bytes, build_s, cache_out


def build_flags_record(opt_level: int, workspace_gb: float, use_timing_cache: bool, opset: int = 17) -> dict:
    return {
        "precision": "fp16",
        "builder_flags": ["FP16"],
        "builder_optimization_level": int(opt_level),
        "workspace_gb": float(workspace_gb),
        "timing_cache": bool(use_timing_cache),
        "onnx_opset": int(opset),
        "onnx_exporter": "torch.onnx.export (TorchScript), do_constant_folding=True, in RAM",
        "network": "explicit batch, static shapes, ONNX parser",
    }


# --------------------------------------------------------------------------- runtime chain
def _trt_dtype_to_torch(dtype) -> torch.dtype:
    import tensorrt as trt

    mapping = {trt.DataType.HALF: torch.float16, trt.DataType.FLOAT: torch.float32}
    if dtype not in mapping:
        raise RuntimeError(f"Unsupported TensorRT IO dtype {dtype}")
    return mapping[dtype]


class StageChain:
    """The 11 deserialized engines, their contexts, the static IO buffers and one shared arena."""

    def __init__(self, engines: dict, spec: list[dict], device: torch.device, require_io: bool = True,
                 required_keys: tuple = ("x", "ehs", "out")):
        import tensorrt as trt

        self.trt = trt
        self.device = device
        self.spec = spec
        self.engines = engines
        self.buffers: dict[str, torch.Tensor] = {}
        self.contexts = []
        # IO buffers, shaped from the engines themselves
        for blk in spec:
            eng = engines[blk["name"]]
            for i in range(eng.num_io_tensors):
                tname = eng.get_tensor_name(i)
                kind, idx = tname[0], int(tname[1:])
                key = blk["inputs"][idx] if kind == "i" else blk["outputs"][idx]
                shape = tuple(eng.get_tensor_shape(tname))
                dtype = _trt_dtype_to_torch(eng.get_tensor_dtype(tname))
                if key in self.buffers:
                    if tuple(self.buffers[key].shape) != shape or self.buffers[key].dtype != dtype:
                        raise RuntimeError(f"Stagewise UNet tensor {key}: {shape}/{dtype} vs "
                                           f"{tuple(self.buffers[key].shape)}/{self.buffers[key].dtype}")
                else:
                    # Normal (non-inference) tensors: callers may construct the chain inside
                    # torch.inference_mode() and later update the buffers in place outside it.
                    with torch.inference_mode(False):
                        self.buffers[key] = torch.zeros(shape, device=device, dtype=dtype)
        if require_io:
            for key in required_keys:
                if key not in self.buffers:
                    raise RuntimeError(f"Stagewise UNet chain has no {key!r} tensor")
        first = self.buffers.get("x", next(iter(self.buffers.values())))
        self.batch = int(first.shape[0])
        # one scratch arena shared by every context (they execute back to back on one stream)
        sizes = {}
        for name, eng in engines.items():
            size = getattr(eng, "device_memory_size_v2", None)
            sizes[name] = int(size if size is not None else eng.device_memory_size)
        self.arena_bytes = max(sizes.values()) if sizes else 0
        self.device_memory_sizes = sizes
        with torch.inference_mode(False):
            self.arena = torch.empty(max(1, self.arena_bytes), device=device, dtype=torch.uint8)
        for blk in spec:
            eng = engines[blk["name"]]
            ctx = eng.create_execution_context(trt.ExecutionContextAllocationStrategy.USER_MANAGED)
            if ctx is None:
                raise RuntimeError(f"Could not create execution context for {blk['name']}")
            if hasattr(ctx, "set_device_memory"):
                ctx.set_device_memory(self.arena.data_ptr(), self.arena_bytes)
            else:  # pragma: no cover - TensorRT < 10.1
                ctx.device_memory = self.arena.data_ptr()
            for i in range(eng.num_io_tensors):
                tname = eng.get_tensor_name(i)
                kind, idx = tname[0], int(tname[1:])
                key = blk["inputs"][idx] if kind == "i" else blk["outputs"][idx]
                if not ctx.set_tensor_address(tname, self.buffers[key].data_ptr()):
                    raise RuntimeError(f"set_tensor_address failed for {blk['name']}:{tname}")
            self.contexts.append((blk["name"], ctx))
        self.graph = None
        self.stream = None

    def enqueue(self, stream_handle: int) -> None:
        for name, ctx in self.contexts:
            if not ctx.execute_async_v3(stream_handle):
                raise RuntimeError(f"execute_async_v3 failed for stagewise UNet block {name}")

    def enqueue_current(self) -> None:
        # Non-graph path: enqueue on a private stream joined to the caller's stream
        # (TensorRT adds stream syncs when enqueued on the legacy default stream).
        current = torch.cuda.current_stream(self.device)
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=self.device)
        self.stream.wait_stream(current)
        self.enqueue(self.stream.cuda_stream)
        current.wait_stream(self.stream)

    def capture_graph(self) -> None:
        current = torch.cuda.current_stream(self.device)
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(current)
        with torch.cuda.stream(side):
            # TensorRT needs one enqueue before capture (lazy init / deferred updates)
            self.enqueue(side.cuda_stream)
            self.enqueue(side.cuda_stream)
        current.wait_stream(side)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        # Capture with capture_begin/capture_end instead of torch.cuda.graph(): the context manager
        # runs gc.collect() on entry, which can finalize torch.compile cudagraph-tree objects created
        # under inference_mode (e.g. the compiled TAESD) and fail with an inference-tensor error.
        # Nothing is allocated during this capture (TensorRT runs in the user-managed arena), so the
        # pre-capture collection is not needed.
        # capture_begin updates the CUDA generator's graph-safe state tensors in place; if an earlier
        # capture in this process (e.g. torch.compile cudagraph trees) ran under inference_mode those
        # are inference tensors, which may only be updated inside inference_mode. Capturing inside
        # inference_mode is valid for both kinds, and the TensorRT enqueue has no autograd.
        capture_stream = torch.cuda.Stream(device=self.device)
        capture_stream.wait_stream(current)
        with torch.cuda.stream(capture_stream), torch.inference_mode(True):
            graph.capture_begin(capture_error_mode="thread_local")
            try:
                self.enqueue(capture_stream.cuda_stream)
            finally:
                graph.capture_end()
        current.wait_stream(capture_stream)
        torch.cuda.synchronize(self.device)
        self.graph = graph

    def run(self) -> None:
        if self.graph is not None:
            # replay() also advances the generator's graph-safe state in place (see capture_graph).
            with torch.inference_mode(True):
                self.graph.replay()
        else:
            self.enqueue_current()

    def activation_bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.buffers.values())


def deserialize_engines(engine_dir: Path, manifest: dict):
    import tensorrt as trt

    trt_logger = trt.Logger(trt.Logger.ERROR)
    runtime = trt.Runtime(trt_logger)
    engines = {}
    for name in block_order(manifest.get("variant")):
        entry = manifest["blocks"][name]
        path = engine_dir / entry["engine_file"]
        data = path.read_bytes()
        if entry.get("engine_sha256") and _env_on("MUSETALK_UNET_STAGEWISE_VERIFY_SHA", "1"):
            if sha256_bytes(data) != entry["engine_sha256"]:
                raise RuntimeError(f"Stagewise UNet engine hash mismatch: {path}")
        engine = runtime.deserialize_cuda_engine(data)
        del data
        if engine is None:
            raise RuntimeError(f"Could not deserialize stagewise UNet engine {path}")
        engines[name] = engine
    return runtime, engines


def probe_inputs(batch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic CPU probe batch (latents and audio features, fp16)."""
    gen = torch.Generator().manual_seed(PROBE_SEED)
    latent = torch.randn((batch,) + LATENT_CHW, generator=gen, dtype=torch.float32).to(torch.float16)
    audio = torch.randn((batch,) + AUDIO_TD, generator=gen, dtype=torch.float32).to(torch.float16)
    return latent, audio


def tensor_sha256(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().contiguous().cpu().numpy().tobytes()).hexdigest()


# --------------------------------------------------------------------------- backend
class _UNetOutput:
    def __init__(self, sample: torch.Tensor):
        self.sample = sample


class StagewiseTrtUnetBackend(torch.nn.Module):
    """Call-compatible with TrtUnetBackend: model(latent, timesteps, encoder_hidden_states=...).sample.

    Timesteps are baked (t=0), exactly like the shipping torch_tensorrt export wrapper.
    """

    name = "tensorrt_unet_stagewise"

    def __init__(self, chain: StageChain, runtime, manifest: dict, engine_dir: Path, device: torch.device,
                 use_graph: bool = True):
        super().__init__()
        self._chain = chain
        self._runtime = runtime  # keeps the TensorRT runtime alive with its engines
        self.manifest = manifest
        self.engine_dir = engine_dir
        self.device = device
        self.batch = chain.batch
        self.variant = manifest.get("variant", "default") or "default"
        self._prefix = None  # srccache: StageChain of the prefix engine (set by load)
        self.runtime_dtype = chain.buffers["x" if self.variant == "default" else "a0"].dtype
        self.dtype = self.runtime_dtype
        self.opt_batch = self.batch
        self.batch_range = None  # any batch: padded or split onto the engine batch
        self.use_graph = bool(use_graph)
        self.calls = 0
        self.padded_rows = 0

    # -- loading
    @classmethod
    def load(cls, engine_dir: Path, device: torch.device, use_graph: Optional[bool] = None,
             probe_check: Optional[bool] = None, require_complete: bool = True) -> "StagewiseTrtUnetBackend":
        import tensorrt as trt

        manifest_path = engine_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Stagewise UNet manifest not found: {manifest_path} "
                "(build with scripts/build_unet_stagewise.py)"
            )
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != MANIFEST_SCHEMA:
            raise RuntimeError(f"Unexpected stagewise UNet manifest schema in {manifest_path}")
        if require_complete and not manifest.get("complete"):
            raise RuntimeError(f"Stagewise UNet engine set is incomplete: {manifest_path}")
        if manifest.get("tensorrt_version") != trt.__version__:
            raise RuntimeError(
                f"Stagewise UNet engines were built with TensorRT {manifest.get('tensorrt_version')}, "
                f"runtime is {trt.__version__}"
            )
        cap = list(torch.cuda.get_device_capability(device))
        if manifest.get("compute_capability") != cap:
            raise RuntimeError(
                f"Stagewise UNet engines were built for sm{manifest.get('compute_capability')}, device is sm{cap}"
            )
        started = time.time()
        runtime, engines = deserialize_engines(engine_dir, manifest)
        variant = manifest.get("variant", "default") or "default"
        required = ("x", "ehs", "out") if variant == "default" else ("h0", "a0", "ehs", "out")
        chain = StageChain(engines, manifest["spec"], device, required_keys=required)
        prefix_chain = None
        if variant == "srccache":
            prefix_chain = StageChain({"prefix": engines["prefix"]}, [PREFIX_SPEC], device,
                                      required_keys=("x", "h0", "a0"))
        if chain.batch != int(manifest["batch"]):
            raise RuntimeError(f"Stagewise UNet engine batch {chain.batch} != manifest {manifest['batch']}")
        if use_graph is None:
            use_graph = _env_on("MUSETALK_UNET_STAGEWISE_CUDAGRAPH", "1")
        backend = cls(chain, runtime, manifest, engine_dir, device, use_graph=use_graph)
        backend._prefix = prefix_chain
        backend.warmup()
        if probe_check is None:
            probe_check = _env_on("MUSETALK_UNET_STAGEWISE_PROBE_CHECK", "1")
        if probe_check:
            backend.check_probe()
        backend.load_seconds = time.time() - started
        logger.info("Stagewise TensorRT UNet bs%d loaded from %s in %.1fs (arena %.1f MiB, graph=%s)",
                    backend.batch, engine_dir, backend.load_seconds, chain.arena_bytes / 2**20, backend.use_graph)
        return backend

    @torch.no_grad()
    def warmup(self) -> None:
        with torch.inference_mode(False):
            inputs = ["x", "ehs"] if self.variant == "default" else ["h0", "a0", "ehs"]
            chains = [self._chain] + ([self._prefix] if self._prefix is not None else [])
            for key in inputs:
                self._chain.buffers[key].normal_()
            if self._prefix is not None:
                self._prefix.buffers["x"].normal_()
            for ch in chains:
                ch.enqueue_current()
            torch.cuda.synchronize(self.device)
            for ch in chains:
                if self.use_graph and ch.graph is None:
                    ch.capture_graph()
                ch.run()
            out = self._chain.buffers["out"]
            torch.cuda.synchronize(self.device)
            if not torch.isfinite(out).all():
                raise RuntimeError("Stagewise UNet warmup produced non-finite output")
            for key in inputs:
                self._chain.buffers[key].zero_()
            if self._prefix is not None:
                self._prefix.buffers["x"].zero_()

    @torch.no_grad()
    def run_probe(self) -> torch.Tensor:
        lat, aud = probe_inputs(self.batch)
        return self(lat.to(self.device), None, encoder_hidden_states=aud.to(self.device)).sample

    def check_probe(self) -> None:
        probe = self.manifest.get("probe") or {}
        expected = probe.get("output_sha256")
        if not expected:
            raise RuntimeError("Stagewise UNet manifest has no probe output hash")
        out = self.run_probe()
        got = tensor_sha256(out)
        if got == expected:
            self.probe_status = "exact"
            return
        tol = float(os.getenv("MUSETALK_UNET_STAGEWISE_PROBE_TOL", "0") or 0)
        ref_path = self.engine_dir / probe.get("output_file", "probe_output.pt")
        max_abs = None
        if ref_path.exists():
            ref = torch.load(ref_path, map_location="cpu", weights_only=True)
            max_abs = float((out.float().cpu() - ref.float()).abs().max())
        if max_abs is not None and max_abs <= tol:
            self.probe_status = f"within_tol:{max_abs}"
            logger.warning("Stagewise UNet probe hash differs but max_abs %.3g <= tol %.3g", max_abs, tol)
            return
        raise RuntimeError(
            f"Stagewise UNet probe output mismatch (sha {got[:12]} != {expected[:12]}, max_abs={max_abs}); "
            "rebuild with scripts/build_unet_stagewise.py or set MUSETALK_UNET_STAGEWISE_PROBE_TOL"
        )

    # -- execution
    def _run_chunk(self, latent: torch.Tensor, audio: torch.Tensor, out: torch.Tensor) -> None:
        n = int(latent.shape[0])
        buf = self._chain.buffers
        if self.variant == "srccache":
            h0, a0 = self._prefix_rows(latent)
            self._run_cached_chunk(h0, a0, audio, out)
            return
        buf["x"][:n].copy_(latent)
        buf["ehs"][:n].copy_(audio)
        if n < self.batch:
            # pad rows repeat the chunk's own rows (like the scheduler's padding); rows are independent
            # in the UNet, so the pad content never changes a real row's result.
            self.padded_rows += self.batch - n
            fill = n
            while fill < self.batch:
                take = min(n, self.batch - fill)
                buf["x"][fill:fill + take].copy_(latent[:take])
                buf["ehs"][fill:fill + take].copy_(audio[:take])
                fill += take
        self._chain.run()
        # static output is overwritten by the next replay: hand back a copy
        out.copy_(buf["out"][:n])

    # -- source-prefix cache (variant "srccache")
    @staticmethod
    def _fill_padded(dst: torch.Tensor, src: torch.Tensor) -> None:
        """dst[:n] = src; remaining rows repeat src (rows are independent, pads never change real rows)."""
        n = int(src.shape[0])
        dst[:n].copy_(src)
        fill = n
        while fill < dst.shape[0]:
            take = min(n, dst.shape[0] - fill)
            dst[fill:fill + take].copy_(src[:take])
            fill += take

    def _prefix_rows(self, latent: torch.Tensor):
        """h0, a0 for <= batch latent rows via the prefix engine (fresh tensors)."""
        if self._prefix is None:
            raise RuntimeError("This stagewise UNet engine set has no prefix engine (variant != srccache)")
        n = int(latent.shape[0])
        pb = self._prefix.buffers
        self._fill_padded(pb["x"], latent.to(device=self.device, dtype=pb["x"].dtype))
        self._prefix.run()
        return pb["h0"][:n].clone(), pb["a0"][:n].clone()

    @torch.no_grad()
    def precompute_prefix(self, latents: torch.Tensor):
        """Per-source-frame cache: latents [N,8,32,32] -> (h0 [N,320,32,32], a0 [N,320,32,32]) on device."""
        latents = latents.to(device=self.device, dtype=self.runtime_dtype)
        total = int(latents.shape[0])
        h0 = torch.empty((total,) + tuple(self._prefix.buffers["h0"].shape[1:]), device=self.device,
                         dtype=self._prefix.buffers["h0"].dtype)
        a0 = torch.empty((total,) + tuple(self._prefix.buffers["a0"].shape[1:]), device=self.device,
                         dtype=self._prefix.buffers["a0"].dtype)
        for start in range(0, total, self.batch):
            end = min(total, start + self.batch)
            h, a = self._prefix_rows(latents[start:end])
            h0[start:end].copy_(h)
            a0[start:end].copy_(a)
        return h0, a0

    def _run_cached_chunk(self, h0: torch.Tensor, a0: torch.Tensor, audio: torch.Tensor, out: torch.Tensor) -> None:
        n = int(a0.shape[0])
        buf = self._chain.buffers
        self._fill_padded(buf["h0"], h0)
        self._fill_padded(buf["a0"], a0)
        self._fill_padded(buf["ehs"], audio)
        if n < self.batch:
            self.padded_rows += self.batch - n
        self._chain.run()
        out.copy_(buf["out"][:n])

    @torch.no_grad()
    def forward_cached(self, h0: torch.Tensor, a0: torch.Tensor, *,
                       encoder_hidden_states: torch.Tensor) -> _UNetOutput:
        """UNet output from cached (h0, a0) rows; bit-identical to forward() on the same latents."""
        if self.variant != "srccache":
            raise RuntimeError("forward_cached needs a srccache engine set")
        total = int(a0.shape[0])
        audio = encoder_hidden_states.to(device=self.device, dtype=self.runtime_dtype)
        out = torch.empty((total,) + OUT_CHW, device=self.device, dtype=self._chain.buffers["out"].dtype)
        self.calls += 1
        for start in range(0, total, self.batch):
            end = min(total, start + self.batch)
            self._run_cached_chunk(h0[start:end], a0[start:end], audio[start:end], out[start:end])
        return _UNetOutput(out)

    @torch.no_grad()
    def forward(self, latent: torch.Tensor, timesteps=None, *,
                encoder_hidden_states: Optional[torch.Tensor] = None) -> _UNetOutput:
        del timesteps  # baked t=0, as the shipping TensorRT UNet export
        if encoder_hidden_states is None:
            raise RuntimeError("Stagewise TensorRT UNet requires encoder_hidden_states")
        if tuple(latent.shape[1:]) != LATENT_CHW or tuple(encoder_hidden_states.shape[1:]) != AUDIO_TD:
            raise RuntimeError(f"Stagewise UNet input shape mismatch: latent={tuple(latent.shape)} "
                               f"audio={tuple(encoder_hidden_states.shape)}")
        total = int(latent.shape[0])
        if int(encoder_hidden_states.shape[0]) != total:
            raise RuntimeError("Stagewise UNet latent/audio batch mismatch")
        latent = latent.to(device=self.device, dtype=self.runtime_dtype)
        audio = encoder_hidden_states.to(device=self.device, dtype=self.runtime_dtype)
        out = torch.empty((total,) + OUT_CHW, device=self.device, dtype=self._chain.buffers["out"].dtype)
        self.calls += 1
        for start in range(0, total, self.batch):
            end = min(total, start + self.batch)
            self._run_chunk(latent[start:end], audio[start:end], out[start:end])
        return _UNetOutput(out)

    def describe(self) -> dict:
        return {
            "name": self.name,
            "engine_dir": str(self.engine_dir),
            "batch": self.batch,
            "cuda_graph": self.use_graph and self._chain.graph is not None,
            "arena_mib": self._chain.arena_bytes / 2**20,
            "activation_mib": self._chain.activation_bytes() / 2**20,
            "device_memory_sizes_mib": {k: v / 2**20 for k, v in self._chain.device_memory_sizes.items()},
            "probe_status": getattr(self, "probe_status", None),
        }


def engine_dir_for_batch(batch: int, root: Optional[Path] = None) -> Path:
    return (root or default_cache_root()) / f"bs{int(batch)}"


def load_stagewise_unet_backend(device: torch.device) -> StagewiseTrtUnetBackend:
    batch = requested_stagewise_batch()
    return StagewiseTrtUnetBackend.load(engine_dir_for_batch(batch), device=device)
