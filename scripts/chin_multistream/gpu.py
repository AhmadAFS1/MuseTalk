"""Backend setup from the worktree and the single GPU issue loop.

setup_backends() is character_factory/h3_avatar_workflow/backend.setup() with the tree switched
from /workspace/MuseTalk to the worktree (same env file, same overrides, same load order), plus an
optional layer of candidate flags applied last.

GpuIssuer.submit() runs render_stage.generate()'s math:
    z      = unet(latents[i:i+8], stamp, encoder_hidden_states=audio[i:i+8]).sample
    pixels = decoder.decode(z, sf, torch.float16)
    finite = torch.isfinite(pixels).all()            (checked after the batch lands; raises like render_stage)
    u8     = pixels.float().mul(255).round().clamp(0,255).to(torch.uint8).flip(1).permute(0,2,3,1).contiguous()
but asynchronously: latents/audio are uploaded once before timing (the same bytes .cuda() would copy),
the uint8 result goes to a pinned host buffer with a non-blocking copy and a CUDA event, and the
caller submits the next batch before waiting on the current one (double buffering). With pack=16,
two stream-aligned 8-frame chunks share one UNet call; the TAESD decode is by default still run per
8-frame chunk (decode_split=8), each chunk post-processed before the next decode is issued (the
compiled TAESD replays a CUDA graph whose output buffer the next call reuses).
"""
from __future__ import annotations

import gc
import os
import sys
import time
from collections import deque
from pathlib import Path

from . import paths

PRESETS = {  # "replay" is filled in below
    # backend.setup() verbatim: TensorRT FP16 bs8 .ts UNet + compiled TAESD
    "baseline": dict(env={}, pack=8, unet_names=("tensorrt_unet_multi", "tensorrt_unet"), decoder_names=("taesd",)),
    # .ts UNet with manual CUDA-graph capture (plan item 1.2)
    "ts_manualgraph": dict(env={"MUSETALK_TRT_UNET_CUDAGRAPHS": "manual"}, pack=8,
                           unet_names=("tensorrt_unet_multi", "tensorrt_unet"), decoder_names=("taesd",)),
    # ONNX-parser stagewise FP16 UNet, engine batch 16 + compiled TAESD
    "stagewise16": dict(env={"MUSETALK_UNET_BACKEND": "trt_stagewise", "MUSETALK_UNET_STAGEWISE_BATCH": "16"}, pack=16,
                        unet_names=("tensorrt_unet_stagewise",), decoder_names=("taesd",)),
    # stagewise bs16 UNet + TensorRT TAESD (strict: no silent fallback to compiled TAESD)
    "stagewise16_taesdtrt": dict(env={"MUSETALK_UNET_BACKEND": "trt_stagewise", "MUSETALK_UNET_STAGEWISE_BATCH": "16",
                                      "MUSETALK_TAESD_BACKEND": "trt", "MUSETALK_TAESD_TRT_STRICT": "1",
                                      "MUSETALK_TAESD_TRT_BUILD": "0"}, pack=16,
                                 unet_names=("tensorrt_unet_stagewise",), decoder_names=("taesd_trt",)),
}


PRESETS["replay"] = dict(env={}, pack=8, unet_names=(), decoder_names=())
# CPU-only: the accepted run's saved faces are "generated" (no CUDA); validates the worker path and
# measures the CPU-side ceiling of N workers.


def setup_backends(extra_env: dict, taesd_warmup_batches: str = "8"):
    """backend.setup(), rooted at the worktree. Returns (unet, decoder, scaling_factor, env_record)."""
    muse = paths.WORKTREE
    os.chdir(muse)
    sys.path[:0] = [str(muse), str(muse / "scripts")]
    env_file = Path(os.environ.get("MUSETALK_REPRO_RUNTIME_ENV", str(muse / ".runtime" / "musetalk_trt_local_sm89.env"))).resolve()
    applied = {}
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            key, value = line.split("=", 1)
            os.environ[key] = value
            applied[key] = value
    base = dict(MUSETALK_TRT_FALLBACK="0", MUSETALK_VAE_BACKEND="taesd", MUSETALK_TAESD_WARMUP_BATCHES=taesd_warmup_batches)
    os.environ.update(base)
    os.environ.update(extra_env)
    import torch
    from scripts.vae_fast_decoder import load_taesd_decoder
    from scripts.trt_runtime import load_unet_trt_backend

    t = time.perf_counter()
    decoder = load_taesd_decoder(device=torch.device("cuda:0"), runtime_dtype=torch.float16, force=True)
    if decoder is None:
        raise RuntimeError("TAESD decoder did not load")
    gc.collect()
    torch.cuda.empty_cache()
    unet = load_unet_trt_backend(device=torch.device("cuda:0"), force=True)
    if unet is None:
        raise RuntimeError("TensorRT UNet did not load")
    load_s = time.perf_counter() - t
    env_record = dict(env_file=str(env_file), env_file_values=applied, recipe_overrides=base, candidate_flags=dict(extra_env),
                      effective={k: v for k, v in sorted(os.environ.items()) if k.startswith(("MUSETALK_", "HLS_"))})
    return unet, decoder, .18215, env_record, load_s


def describe_backends(unet, decoder) -> dict:
    out = dict(unet_name=getattr(unet, "name", None), unet_class=type(unet).__name__,
               decoder_name=getattr(decoder, "name", None), decoder_class=type(decoder).__name__,
               decoder_compile_enabled=getattr(decoder, "compile_enabled", None),
               decoder_compile_mode=getattr(decoder, "compile_mode", None))
    if hasattr(unet, "describe"):
        try:
            out["unet_describe"] = unet.describe()
        except Exception as exc:  # pragma: no cover
            out["unet_describe"] = repr(exc)
    inner = getattr(unet, "backends_by_batch", None)
    if inner is not None:
        out["unet_inner"] = {k: dict(cls=type(v).__name__, cudagraphs_mode=getattr(v, "cudagraphs_mode", None),
                                     batch_range=getattr(v, "batch_range", None)) for k, v in inner.items()}
    elif hasattr(unet, "cudagraphs_mode"):
        out["unet_cudagraphs_mode"] = unet.cudagraphs_mode
    if hasattr(decoder, "meta"):
        out["decoder_trt_key"] = (decoder.meta or {}).get("key")
        out["decoder_trt_gate"] = (decoder.meta or {}).get("gate")
        out["decoder_trt_plan_sha256"] = (decoder.meta or {}).get("decoder_plan_sha256")
    return out


class Job:
    __slots__ = ("chunks", "buf", "ev", "t_submit")

    def __init__(self, chunks, buf, ev, t_submit):
        self.chunks, self.buf, self.ev, self.t_submit = chunks, buf, ev, t_submit


class GpuIssuer:
    def __init__(self, unet, decoder, sf, latents: dict, audio: dict, pack: int, decode_split: int, depth: int):
        import torch

        self.torch = torch
        self.unet, self.decoder, self.sf = unet, decoder, sf
        self.lat, self.aud = latents, audio
        self.pack, self.decode_split, self.depth = int(pack), int(decode_split), int(depth)
        # Source-prefix cache (stagewise engine set variant "srccache"): conv_in + down0.resnets[0] depend only on
        # the source latent, so they are computed once per identity frame here and gathered per batch
        # (forward_cached == forward bit for bit; see docs/.../unet_fp16/srccache_exact.py).
        self.prefix = None
        if getattr(unet, "variant", "default") == "srccache":
            with torch.inference_mode():
                self.prefix = {ident: unet.precompute_prefix(lat) for ident, lat in latents.items()}
            torch.cuda.synchronize()
        self.stamp = torch.tensor([0], device="cuda")
        k = self.pack // paths.BATCH
        self.free_bufs = [dict(u8=torch.empty((self.pack,) + paths.FACE_SHAPE, dtype=torch.uint8, pin_memory=True),
                               flag=torch.zeros((k,), dtype=torch.bool, pin_memory=True)) for _ in range(self.depth + 1)]
        for b in self.free_bufs:
            b["u8_np"] = b["u8"].numpy()
        self.reset_stats()

    def reset_stats(self):
        self.stats = dict(jobs=0, partial_jobs=0, chunks=0, submit_host_ms=0., sync_wait_ms=0., deliver_host_ms=0.,
                          gpu_job_ms=0., gpu_unet_ms=0., gpu_decode_post_d2h_ms=0.)

    def post(self, pixels):
        # render_stage.generate(), minus the trailing .cpu().numpy() (done as a pinned async copy)
        return pixels.float().mul(255).round().clamp(0, 255).to(self.torch.uint8).flip(1).permute(0, 2, 3, 1).contiguous()

    def submit(self, chunks):
        """chunks: list of (identity, base, meta) with 1..pack/8 entries."""
        torch = self.torch
        t = time.perf_counter()
        buf = self.free_bufs.pop()
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        ev[0].record()
        B8 = paths.BATCH
        if len(chunks) == 1:
            ident, base, _ = chunks[0]
            aud = self.aud[ident][base:base + B8]
        else:
            aud = torch.cat([self.aud[ident][base:base + B8] for ident, base, _ in chunks])
        if self.prefix is not None:
            if len(chunks) == 1:
                h0 = self.prefix[ident][0][base:base + B8]
                a0 = self.prefix[ident][1][base:base + B8]
            else:
                h0 = torch.cat([self.prefix[i][0][b:b + B8] for i, b, _ in chunks])
                a0 = torch.cat([self.prefix[i][1][b:b + B8] for i, b, _ in chunks])
            lat = None
            z = self.unet.forward_cached(h0, a0, encoder_hidden_states=aud).sample
            del h0, a0
        else:
            if len(chunks) == 1:
                lat = self.lat[ident][base:base + B8]
            else:
                lat = torch.cat([self.lat[ident][base:base + B8] for ident, base, _ in chunks])
            z = self.unet(lat, self.stamp, encoder_hidden_states=aud).sample
        ev[1].record()
        k = len(chunks)
        B = paths.BATCH
        if self.decode_split:
            for c in range(k):
                pixels = self.decoder.decode(z[c * B:(c + 1) * B], self.sf, torch.float16)
                buf["flag"][c:c + 1].copy_(torch.isfinite(pixels).all().view(1), non_blocking=True)
                u8 = self.post(pixels)
                del pixels
                buf["u8"][c * B:(c + 1) * B].copy_(u8, non_blocking=True)
                del u8
        else:
            pixels = self.decoder.decode(z, self.sf, torch.float16)
            buf["flag"][:k].copy_(torch.isfinite(pixels).view(k, -1).all(1), non_blocking=True)
            u8 = self.post(pixels)
            del pixels
            buf["u8"][:k * B].copy_(u8, non_blocking=True)
            del u8
        ev[2].record()
        del z, lat, aud
        self.stats["submit_host_ms"] += (time.perf_counter() - t) * 1000
        self.stats["jobs"] += 1
        self.stats["chunks"] += k
        if k * B < self.pack:
            self.stats["partial_jobs"] += 1
        return Job(chunks, buf, ev, t)

    def wait(self, job):
        """Block until the job's faces are in pinned host memory; return a (k*8,256,256,3) view."""
        t = time.perf_counter()
        job.ev[2].synchronize()
        self.stats["sync_wait_ms"] += (time.perf_counter() - t) * 1000
        k = len(job.chunks)
        if not bool(job.buf["flag"][:k].all()):
            raise RuntimeError("Non-finite generated pixels")
        self.stats["gpu_job_ms"] += job.ev[0].elapsed_time(job.ev[2])
        self.stats["gpu_unet_ms"] += job.ev[0].elapsed_time(job.ev[1])
        self.stats["gpu_decode_post_d2h_ms"] += job.ev[1].elapsed_time(job.ev[2])
        return job.buf["u8_np"][:k * paths.BATCH]

    def release(self, job):
        self.free_bufs.append(job.buf)

    def warmup(self, ident, iters=4):
        """render_stage runs 4 warm unet+decode batches before timing; here the whole issue path is warmed."""
        k = self.pack // paths.BATCH
        shapes = [k] + ([1] if k > 1 else [])
        for kk in shapes:
            for _ in range(iters):
                job = self.submit([(ident, c * paths.BATCH, None) for c in range(kk)])
                self.wait(job)
                self.release(job)
        self.torch.cuda.synchronize()
        self.reset_stats()


class ReplayIssuer:
    """Same interface as GpuIssuer; returns the accepted faces.npz rows instead of running the GPU."""

    def __init__(self, faces: dict, pack: int, depth: int):
        self.faces, self.pack, self.depth = faces, int(pack), int(depth)
        self.reset_stats()

    reset_stats = GpuIssuer.reset_stats

    def submit(self, chunks):
        import numpy as np

        t = time.perf_counter()
        host = np.concatenate([self.faces[ident][base:base + paths.BATCH] for ident, base, _ in chunks])
        self.stats["submit_host_ms"] += (time.perf_counter() - t) * 1000
        self.stats["jobs"] += 1
        self.stats["chunks"] += len(chunks)
        return Job(chunks, host, None, t)

    def wait(self, job):
        return job.buf

    def release(self, job):
        pass


def run_repeat(issuer: GpuIssuer, streams, conns, rings, rep, loops, ring_slots, on_message):
    """One timed repeat. streams: list of identity names (index = stream id).

    Scheduling: round-robin over streams that have remaining batches AND a free ring slot (a credit),
    so a slow stream never blocks the GPU while another stream can take work. Per-stream order is
    always base 0,8,...,232 of loop 0, then loop 1, ... exactly one batch after another.
    """
    from multiprocessing.connection import wait as conn_wait

    issuer.reset_stats()
    n = len(streams)
    per_clip = paths.N_FRAMES // paths.BATCH
    total = loops * per_clip
    next_idx = [0] * n
    free = [list(range(ring_slots)) for _ in range(n)]
    k_full = issuer.pack // paths.BATCH
    inflight = deque()
    rr = 0
    credit_wait_ms = 0.
    min_free_seen = ring_slots

    def handle(s, msg):
        if msg[0] == "r":
            free[s].append(msg[1])
        else:
            on_message(s, msg)

    def drain(timeout=0.0):
        ready = conn_wait(conns, timeout=timeout)
        for c in ready:
            s = conns.index(c)
            while c.poll():
                handle(s, c.recv())
        return bool(ready)

    def available():
        return sum(min(len(free[s]), total - next_idx[s]) for s in range(n))

    def take(k):
        nonlocal rr
        chunks = []
        while len(chunks) < k:
            progressed = False
            for step in range(n):
                s = (rr + step) % n
                if next_idx[s] < total and free[s]:
                    idx = next_idx[s]
                    next_idx[s] += 1
                    slot = free[s].pop()
                    loop, base = divmod(idx, per_clip)
                    chunks.append((streams[s], base * paths.BATCH, (s, slot, rep, loop, base * paths.BATCH)))
                    rr = (s + 1) % n
                    progressed = True
                    break
            if not progressed:
                break
        return chunks

    t0 = time.perf_counter()
    while True:
        drain(0.0)
        while len(inflight) < issuer.depth:
            avail = available()
            if avail == 0:
                break
            remaining = sum(total - next_idx[s] for s in range(n))
            if avail < k_full and inflight and remaining > avail:
                break  # wait for credits to fill the pack while the GPU still has work
            chunks = take(k_full)
            if not chunks:
                break
            inflight.append(issuer.submit(chunks))
        if inflight:
            job = inflight.popleft()
            host = issuer.wait(job)
            t = time.perf_counter()
            for c, (_, _, (s, slot, r, loop, base)) in enumerate(job.chunks):
                rings[s][slot][:] = host[c * paths.BATCH:(c + 1) * paths.BATCH]
                conns[s].send(("b", slot, r, loop, base))
            issuer.stats["deliver_host_ms"] += (time.perf_counter() - t) * 1000
            issuer.release(job)
            min_free_seen = min(min_free_seen, min(len(f) for f in free))
            continue
        if all(next_idx[s] >= total for s in range(n)):
            break
        t = time.perf_counter()
        drain(30.0)
        credit_wait_ms += (time.perf_counter() - t) * 1000
    t_gpu_done = time.perf_counter()
    stats = dict(issuer.stats)
    stats.update(credit_wait_ms=credit_wait_ms, min_free_slots_seen=min_free_seen, t0=t0, t_gpu_done=t_gpu_done)
    return stats, drain
