# Why Ditto throughput is shared across concurrent calls

Measured on 2026-09-16, NVIDIA GeForce RTX 4070 SUPER, driver 595.84,
12,282 MiB VRAM. Ditto commit: `c3e47eee2e626500017a0556b470d6d4182f85e8`.
This supplements the same-avatar comparison and corrects the scope of earlier
capacity claims. No production inference code or GPU settings were changed.

## Finding

The installed TensorRT implementation generates approximately 32 frames/second
for one unpaced online session. Starting more independent SDKs does not multiply
GPU throughput: two and three peers contend for the same device, yielding about
29 aggregate delivered FPS including startup and completion overhead. Three
20 FPS speakers require at least 60 generated FPS, approximately twice the
measured capacity, plus operating headroom.

The single-peer WebRTC result of 25 FPS was capped by audio delivery and video
pacing. It did not establish the unpaced ceiling. New runs removed both pacing
and all encoding/transport, keeping the original model configuration and same
input. They still produced only about 32–33 FPS.

| Diagnostic | Generated frames | Elapsed seconds | Generated FPS |
|---|---:|---:|---:|
| Online unpaced, first run | 260 | 8.098 | 32.11 |
| Online unpaced, warm run | 260 | 8.077 | 32.19 |
| Offline unpaced, first run | 250 | 7.614 | 32.83 |
| Offline unpaced, warm run | 250 | 7.538 | 33.16 |

See `profile-online.json`, `profile-offline.json`, and `profile_ditto.py`.
Model loading and avatar preparation are excluded. Output is counted as frames
arrive at an in-memory sink; MP4 writing and WebRTC are absent. Online padding
produces 260 frames, so online and offline counts are reported separately.
The warm online interval between first and last generated frames is 33.10 FPS;
even removing startup does not reveal a hidden 60+ FPS renderer.

## Where time goes

Each output frame runs motion stitching, a warp network, a decoder, and CPU
compositing. TensorRT accelerates these operations but does not remove them.
The final 384×672 portrait is composited from a **512×512 decoded face**.

After the online run, each engine was measured alone for 20 calls after five
warmups, reusing its existing input/output allocation. Calls include synchronous
host/device copies and a completion synchronization, but exclude buffer setup.
These are host wall timings, not GPU kernel-only measurements.

| Isolated engine operation | Mean time |
|---|---:|
| Warp, once per output frame | 9.23 ms |
| Decoder, once per output frame | 15.63 ms |
| One motion diffusion step | 3.41 ms |
| HuBERT audio feature call | 6.82 ms |

Warp plus decoder alone costs about 24.86 ms in this sequential diagnostic,
equivalent to approximately 40 FPS before other work. This is evidence of
substantial rendering cost, not a mathematical upper bound for a redesigned,
batched pipeline. Pipeline workers overlap, contend, and wait for each other;
their instrumented call durations must not be added as exclusive GPU time.
In particular, the roughly 25–26 ms motion-step wall times during the concurrent
pipeline include contention; the isolated motion-step time is only 3.41 ms.

The source exposes additional overhead:

- `core/utils/tensorrt_utils.py`, `TRTWrapper.setup`: frees and reallocates
  fixed-shape device I/O buffers on each model invocation.
- `TRTWrapper.infer`: synchronous host-to-device and device-to-host copies;
  execution defaults to stream 0. Multiple Python worker threads therefore
  do not establish independent, efficiently overlapping CUDA work.
- `core/models/warp_network.py` and `decoder.py`: pass NumPy arrays between
  stages and copy outputs on the CPU. Measured tensor sizes imply about 19 MiB
  of transfers per rendered frame across warp and decoder: 8 MiB source input,
  4 MiB warp output, that same 4 MiB uploaded to decoder, and 3 MiB RGB output.
  That is roughly 608 MiB/s at 32 FPS. This is not evidence of PCIe bandwidth
  saturation; allocations, synchronization and redundant transfers are the
  concerns.
- `stream_pipeline_online.py`, `StreamSDK.__init__`: every SDK instantiates
  its own models and execution contexts. The concurrency harness uses one SDK
  per peer, with no shared model pool or cross-session batching.

No experiment here quantifies how much faster buffer reuse, device-resident
intermediates, new CUDA streams, or batched engines would be. Those are supported
optimization candidates, not demonstrated speedups.

## Why 10 online steps do not mean five times the speed

The motion model predicts 80-frame windows. Offline overlap is 10 frames,
advancing 70 frames per window at 50 steps. Online overlap is 70 frames,
advancing only 10 frames per window at 10 steps. Asymptotically that is roughly
50/70 = 0.714 versus 10/10 = 1 diffusion evaluation per newly advanced frame,
before startup and tail handling.

The instrumented clip confirms **200 motion engine evaluations offline versus
260 online**. Both still render every frame through warp and decoder. Online
settings reduce response latency and support chunked input; they do not make
this full pipeline five times faster.

## What the earlier live-peer tests establish

| Independent speaking peers | Received FPS per peer | Delivered aggregate FPS including startup/tail | Peak device memory | Increment above background |
|---:|---:|---:|---:|---:|
| 1 | 25.00 | 23.36 | 5,065 MiB | 2,578 MiB |
| 2 | 15.58–15.69 | 29.17 | 7,459 MiB | 4,972 MiB |
| 3 | 10.24–10.42 | 29.29 | 9,849 MiB | 7,362 MiB |
| 4 | No completed session | n/a | 10,687 MiB sampled before failure | n/a |

The fourth SDK failed during TensorRT deserialization with CUDA OOM. Its sampled
memory peak is not the failed allocation size. An unrelated OmniVoice service
held about 2,478 MiB, with total idle device usage of 2,487 MiB. It was left
running. Background residency explains part of the memory limit; its complete
compute activity was not separately monitored, so these are not clean-card
capacity certifications.

All three completed tests used actual local aiortc video RTP connections and
200 ms audio chunks driving Ditto. They were **video-only loopback tests**, not
full audio/video FaceTime calls or Internet/TURN tests. Each peer had its own SDK
but used the same image and audio. No independent-avatar/state-isolation check
or long-duration soak was performed.

The receiver limits each peer to 250 decoded frames; sender pacing is 25 FPS.
The final harness sends 251 frames to drain the receiver's final buffered frame.
The preceding 249-frame timeout was a harness drain issue, not a measured
inference failure. The earlier claim that H.264 consumed a startup frame was
not verified: the harness did not force or record the negotiated codec.

The producer generates 260 frames and its completion is included in aggregate
wall time. Consequently 29 aggregate delivered FPS and the 32 unpaced generated
FPS are related but not identical metrics. At three peers the synchronous audio
feeding also falls behind its intended 200 ms schedule due to GPU contention.

## Hardware and comparison qualifications

The current power limit is **180 W**, with **220 W default/max** reported by
`nvidia-smi`. Prior loaded runs peaked near 177–179 W. This is a plausible
contributor to reduced throughput; no power-limit change or controlled A/B was
performed, so its performance contribution is unquantified. Earlier reports
should not be read as stock-power RTX 4070 SUPER results.

The observed practical result is one simultaneous 25 FPS speaker for the tested
configuration and workload. It is not a universal Ditto or TensorRT limit.

The user's 8–10 MuseTalk calls on an RTX 3090 are not a matched benchmark:
different hardware, rendering models, scheduling, batch sizes, and possibly
speech duty cycles apply. MuseTalk's installed scheduler can batch requests;
this Ditto harness duplicates batch-1 pipelines. Available VRAM cannot predict
how many continuously speaking peers meet a latency/FPS target.

The earlier 3090 residency prediction of 8–9 SDKs is only a linear extrapolation
of roughly 2,578 MiB initially plus 2,392 MiB per extra SDK. Neither residency
nor throughput has been verified on a 3090. Likewise, dividing this card's
29 FPS by eight does not predict a 3090's FPS. That earlier illustration should
not be treated as a hardware forecast.

## Next experiments supported by these findings

1. Reuse fixed-shape buffers and keep warp/decoder intermediates on the GPU;
   verify output equivalence and remeasure unpaced throughput.
2. Profile GPU kernels with a CUDA timeline before changing stream scheduling.
3. Evaluate shared immutable engines with isolated session contexts, then
   batch-compatible exports/engines and a fair cross-session scheduler.
4. Re-run on the actual 3090 with matched power settings, distinct avatars,
   simultaneous speech, audio RTP, negotiated codec capture, and a sustained run.

Achieving three 20 FPS speakers requires approximately 60 FPS of useful
aggregate generation. The experiments do not yet demonstrate that target or
guarantee it can be reached with these optimizations.

## Evidence and reproduction

The neighboring JSONs contain the original C1–C4 telemetry and C1–C3 peer reports.
The checked-in scripts preserve the tested harness and profiler. They reference
the existing `/workspace` installation and virtual environments; they are not
a portable standalone deployment. The offline profile predates addition of the
isolated-engine diagnostic and therefore has no `isolated` section.

Fixtures under `/workspace/benchmarks/same-avatar`:

- `shared.png` SHA-256: `cb7f390757c2d1a5e782a5fe0a0660d5d0fc848d54fc1f6eed72e4a3757feac9`
- `audio.wav` SHA-256: `999011601ba01d0c4cba544222997ce421f219419e4e3e9471a4ca056742ba22`

Run from `/workspace/ditto-talkinghead`, with sufficient free VRAM:

```sh
/workspace/.venvs/ditto/bin/python /workspace/benchmarks/same-avatar/profile_ditto.py online
/workspace/.venvs/ditto/bin/python /workspace/benchmarks/same-avatar/profile_ditto.py offline
```

These diagnostics generate fresh frames without opening a model server. No
MuseTalk, SoulX or OmniVoice environment was modified for this investigation.
