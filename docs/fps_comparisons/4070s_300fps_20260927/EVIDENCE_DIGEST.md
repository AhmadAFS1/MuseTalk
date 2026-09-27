# Evidence digest — MuseTalk 300 fps on RTX 4070 SUPER (from workflow 1, 2026-09-27)

Full structured results: `wf1_result.json` in this folder (readers: model-path, scheduler-serving,
cpu-post-chin, audio-tts, history; probes: unet, taesd, cpu). Probe scripts + JSON under
`unet_probe/`, `taesd_probe/`, `cpu_probe/`, `audio_tts/` (and a serving loopback harness written by
the scheduler reader). Tags: [M] measured now on this box, [D] from repo docs, [I] inferred.

## Target
15 streams x 20 fps = 300 fps aggregate (3.33 GPU-ms/frame); stretch 20 streams = 400 fps (2.5 ms).
User REQUIRES: TAESD decoder + native avatar encoder + 100% chin alignment + refined seam + expressive H3
source. User style (memory): attempt quality-risky levers, gate on labelled video + numeric gates, keep a
one-line env-flag rollback, end each implementation round with a labelled comparison video.

## Where the "160 fps" comes from
Offline render harness numbers, not live WebRTC: diversity batch 148–171 fps, refined-seam chin 163–169,
chin100 v1 184.9/193.4, pipelined standard 250.8, serial TAESD 204–215 [D]. Live multi-stream WebRTC with
TAESD has never been measured cleanly beyond 3 peers. The user's own 10-stream wall test (19:35–19:39 UTC
today, pid 3305906) showed only ~120–128 of ~177 output frames per stream fresh (49–77 held/duplicated),
avg_gpu_batch 36–40 ms, callback max 103–145 ms, 5.4–8.9 cores, >1000 threads, RSS 7.1 GB (high-water
10.5 GB); the process died at 19:39:24 with 0.8 GB RAM available (likely OOM) [D: wall_api.log].
CAVEAT: workflow-1 probes were running on the GPU/RAM at the same time and likely contaminated that run and
contributed to the OOM.

## GPU — measured
- UNet: 177.8 GFLOP/frame [M]; 849.9M params, 84% of params in 8²/4² stages; FLOPs 36% @32², 31% @16²,
  27% @8² [M]. Op split: conv3x3 ~49–56%, linear GEMM ~23%, attention core ~5% [M].
- Shipping TRT FP16 static bs8 (torch_tensorrt TorchScript): 24.0–24.6 ms/bs8 = 3.0–3.08 ms/frame,
  ~58 TFLOPS (~41% of 142 TFLOPS FP16-acc peak) [M]. All convs/GEMMs already FP16-accumulate kernels [M].
  1250 kernels/call; 31% of time outside tensor cores (Myelin pointwise 22%, reformat 5%, reductions 4%);
  308 of 1798 layers come from torch_tensorrt decomposing GroupNorm [M].
- bs16 today = split into 2x bs8 (49.3 ms, no amortization) [M].
- Native bs16 FP16 torch_tensorrt engine (built in memory): 45.34 ms = 2.83 ms/frame (-8.7%); with CUDA
  graph 44.63 ms = 2.79 ms/frame; mae vs torch 0.00196 [M]. Build: 401 s cold, host RSS peak 11.0 GB [M].
- ONNX-parser FP16 path (sum of 11 per-block engines, slightly pessimistic): 20.58 ms/bs8 = 2.57 ms/frame,
  -14% vs shipping [M, estimate]. Per-block torch_tensorrt is 1.09–1.53x slower than ONNX path [M].
- torch_tensorrt.runtime.set_cudagraphs_mode(True): UNet 23.3–23.4 ms (-3.9 to -5.2%), bit-exact, host
  time per call 4.5 ms -> <0.3 ms [M]. Unverified: output-buffer reuse across calls.
- Two execution contexts sharing weights on two streams: -5% [M]. torch_tensorrt runtime serializes to one
  engine stream [M]. Second CUDA stream for TAESD: 0.6–1.7% [M]. GPU is power-capped (200–218 W of 220 W,
  SM 2505–2625 MHz under load) so FLOP cuts convert to throughput [M].
- FP8: DEAD on this card + TRT 10.3. FP8 Q/DQ convs get no FP8 kernel (fallback FP16 conv + emulation):
  1.25–6.9x slower; FP8 GEMM kernels exist but transformer blocks still lose; consumer-Ada FP8 w/ FP32 acc
  = same peak as FP16 w/ FP16 acc [M]. (SoulX found the same class of result with TRT 10.9 Conv3d [D].)
- INT8 Q/DQ: real INT8 kernels. ResNet blocks 1.8–2.1x, 64-token transformer 1.49x, 1024-token transformer
  1.02x [M]. Whole-UNet INT8 (sum of blocks) 14.36 ms/bs8 = 1.80 ms/frame (-40%); with up1 kept FP16
  ~15.8 ms [M/I]. QUALITY: naive INT8 all-layers latent rel-L2 7.07%, PSNR 41.8 dB (min 39.4) vs FP16 —
  ~18x shipping TRT error, fails repo UNet gate (mae_max 0.01, max_abs 0.5); SmoothQuant 6.75%;
  conv-only 6.09%; up1 most sensitive (21.7% on its own output) [M, 64 frames, one avatar].
- TAESD live (compiled max-autotune = default compile + cudagraphs on this 56-SM card; cudnn.benchmark on):
  0.739 ms/frame [M] (doc's 0.86 was with cudnn.benchmark off). Larger batch doesn't help.
- TAESD TensorRT FP16: full 0.486 ms/frame; STAGED EXACT CROP (crop feature maps at stage inputs; bit-exact
  max_abs 0) + uint8 BGR NHWC output: 0.345 ms/frame (2.1x vs live) [M]. Engine ~3 MB, builds ~15 s in
  memory. Latent-space crop is never exact (receptive field 18.1 latent rows) [M]. Crop start row must be
  PER AVATAR: standard avatars read from row 110–111, fixed_face_height (_fh1) avatars from row 85 [M];
  presets R=80 (0.403 ms) / R=104 (0.350 ms). INT8 TAESD staged crop 0.147 ms/frame, PSNR 46.3 dB,
  max_abs 0.18 (risky, needs video) [M].
- END-TO-END GPU PATH (bs8, sustained, real inputs, excludes audio/compose/serving) [M]:
  live-equivalent (TRT UNet + compiled TAESD + post + pinned D2H) 30.93 ms = 258.7 fps;
  + TRT staged-crop uint8 TAESD 26.93 ms = 297.0 fps;
  + UNet cudagraphs mode 26.27 ms = 304.6 fps (3.28 ms/frame; ~1.5% margin).
- GPU handoff: 18 us/frame; GPU compose (grid_sample resize + blend) 3–5 us/frame vs CPU 0.35–0.47 ms [M].
- Whisper-tiny: 14 ms per 8 s turn warm on GPU (encoder ~3 ms; ~0.02 GPU-ms/frame) — negligible. Doc's
  0.88 s = cold librosa first call (1.2 s), not warmed by _warm_runtime_paths [M].
- Host: TRT call releases GIL but the issuing thread spin-waits 100% of a core (24 ms CPU per bs8 call) [M];
  with other Python threads busy, enqueue grew 4.5 -> 14 ms [M]. TAESD compiled with cudagraph trees
  crashes if called from a second thread (thread-affine) [M].

## GPU — projections (verify these)
- Lossless FP16 stack (bs16 native + cudagraph measured 2.79 ms; ONNX-path rebuild est. further -7..-14%)
  UNet ~2.25–2.79 ms/frame + TAESD TRT crop 0.35 + ~0.05 misc => 2.65–3.19 ms/frame => ~314–377 fps
  GPU ceiling (low end fully measured components; high end relies on the ONNX sum-of-blocks estimate).
- Mixed INT8 UNet (if quality passes): ~1.8–2.0 ms + 0.35 => ~2.2–2.4 ms => ~415–455 fps.
- 400 fps with margin likely needs INT8 (quality-recovered via mixed precision / QAT / distillation) or
  UNet block pruning + distillation (SoulX precedent: decoder bypass+distill worked; pruning the
  GENERATOR broke lip sync) [D].
- Closed / not worth it: FP8; batches >16 (eager curve flat); 2nd CUDA stream; 2 sessions on saturated GPU
  (+1.3% SoulX); DeepCache-style reuse (single-step model, audio enters every transformer); source-only
  caching (≤3.3% FLOPs); latent-space crop; ROI 224/192 (user rejected as blurry).

## Serving path — findings
- One uvicorn process, one asyncio loop owning all aiortc PeerConnections; one 'hls-gpu-scheduler'
  thread; 10-thread compose pool; everything under one GIL [I-code].
- Scheduler does 3 torch.cuda.synchronize per batch (HLS_GPU_STAGE_SYNC_TIMING default on,
  MUSETALK_VAE_DECODE_TIMING_SYNC default on) and a blocking D2H; batch N+1 is never queued before batch
  N's D2H; the WebRTC handoff blocks the scheduler thread on run_coroutine_threadsafe(...).result()
  (8–10 ms GPU idle per 31.5 ms batch => single-process live ceiling ~195–200 fps) [I from logs + code].
  Under load push handoff p50 reached 75 ms/batch [M loopback].
- Idle sessions: no GPU, but idle mp4 decode (512x896, ~5 ms/output frame) runs INSIDE recv() on the event
  loop => loop saturates at ~180–186 fps total output [M loopback]. Pre-decoded shared idle yuv cache fixes:
  loop 98% -> 26% for 15 idle sessions [M]. Motion entry/return builders re-decode from frame 0 each turn
  boundary (up to ~1.2 CPU-s per turn end) [I].
- Encode/transport demand = ALL connected sessions x playback fps; GPU demand = speaking frames only.
  Playback transport may be 30 fps (20 generated, duplicated) — raises encode cost 1.5x [I, open question].
- H.264: api_server's NVENC patch is DEAD CODE on aiortc 1.14 (H264Encoder._encode_frame hardcodes libx264;
  log says codec=libx264 despite "encoder set to h264_nvenc"). aiortc default libx264 = preset medium,
  ~28 threads/encoder, 11.5–26 CPU-ms/frame => ~7.9 cores at 300 fps, 10.8 at 400 [M].
  NVENC: 0.12–0.21 CPU-ms/frame but driver caps at 12 concurrent sessions, ~200 MB VRAM each [M].
  x264 ultrafast 1 thread 1.2 ms; veryfast 1 thread 3.9 ms [M].
- VP8 native (libvpx 1.13.1, cpu-used -6, 2.5 Mbps): 4.0 ms/frame at 1 thread; aiortc's default 2 threads
  costs 35–45% more CPU under concurrency. 15 streams: 1.85 cores (1 thr); 20 streams: 2.79 cores [M].
- PyAV bgr24 -> yuv420p: 1.51–1.72 ms/frame at width 512 (slow path; holds GIL ~0.35 ms); cv2
  COLOR_BGR2YUV_I420 0.14–0.21 ms, ≤1–2 LSB diff [M].
- RTP+SRTP+UDP ~0.12 ms/video frame; Opus 6.7 ms CPU per stream-second [M].
- Single-process GIL ceiling for per-frame CPU chain: ~570–665 fps with chin, 820–990 standard, at ~6
  cores. At 400 fps with 20 streams in ONE process: 48 ms mean per-frame work vs 50 ms period, 19 late;
  4 processes x 5 streams: 12.6–23.8 ms, 0 late [M synthetic combined load].
- Combined synthetic load incl. chin stand-in + FaceMesh per stream + native VP8 1-thread + RTP + Opus:
  10.6 logical CPUs at 300 fps, 12.5–13.1 at 400 (of 32) [M]. Plus GPU-feeding spin thread(s) ~1–3.5 [I].
- Scheduler: HLS_SCHEDULER_MAX_BATCH=8, buckets [8], round-robin cursor, no deadline awareness; strict FIFO
  400-frame queue with 30 s push timeout can stall the shared scheduler thread [I-code].
- GPU waste: exact_silence turns and WEBRTC_RAW_IDLE_POSE neutral frames still run UNet+TAESD then discard
  the output (hls_gpu_scheduler.py ~1763–1782) [I-code].
- Kokoro TTS runs IN-PROCESS on CPU (KOKORO_TTS_DEVICE=cpu, single RLock, ~8.8 audio-s/s max); at 15
  speakers it would take 30–85% of the CPU [M] => must be off-box / pre-synthesized for capacity tests.
- Per-turn setup decodes audio 4–5 times via ffmpeg/PyAV (p50 140 ms/turn); latency, not throughput [D].
- Historical: RTX 6000 Ada live plateau ~112 fps equalled batch/avg_gpu_batch (model-bound by SD-VAE INT8);
  3090 live/implied ratio 0.80–0.85; strict smooth 20 fps never exceeded 4 streams on any GPU [D].
  Those serving overheads were hidden by the slow decoder and are ALL still present [I].

## Chin path (user-required; offline only today)
- Not in the live server at all (no generated-face tracking/warp in api_server/scheduler/api_avatar) [I-code].
- Cost: FaceMesh refined tracking 2.98 ms wall / ~3.3 core-ms per frame, deterministic, stateful per
  stream (one graph per stream, ~31 MB, 33 threads each; first frame ~14 ms) [M]; refined chin compose
  (mask + blend + warp) ~2.3–3.9 ms [M/D]. Standard path total ~1.5–1.8 core-ms/frame; chin seam-v1
  ~7–8.4 core-ms/frame as written [M].
- Threads plateau ~830 fps/process (GIL, ~1.2 ms Python per frame); processes: 2046 fps @8, 2424 @16 [M].
- Bit-exact numba nogil ports proven for blend (0.50->0.19 ms) and warp (0.71–0.91->0.19 ms), 0 mismatches
  over 720 frames / 3 identities; mask port not written [M].
- FaceMesh pool scaling: 292 fps (1 proc), 626 (2), 995 (4), 1633 (8); machine saturates ~1900 fps [M].
- 3-tap symmetric jaw filter needs next frame => ≤1 frame (50 ms) latency, no throughput cost; requires
  per-stream ORDERED actors (today's free compose pool can reorder) [I-code].
- Chin caches 8.5 MB per source frame (~2 GB per 240-frame pose) — must be trimmed to ~1.1 MB (drop unused
  arrays, dedup cycle halves) for multi-avatar serving in 30 GB RAM [M].
- NumPy 2 (FaceMesh venv) vs NumPy 1.23.5 (TRT venv) dtype promotion changes lip_y => pixel drift; compose
  must stay in the TRT venv or port with explicit dtypes [M].
- Cheaper tracking (crop / half-res / unrefined / static mode) all deviate 0.3–2.4 px on jaw/lip, same scale
  as the corrected error => not acceptable as exact; GPU landmark net would spend the binding GPU budget.

## Capacity model
- GPU demand = simultaneous speakers x 20 fps. Binomial p99: 15 sessions @50% duty ≈ 12 speakers = 240 fps;
  20 sessions @50% ≈ 300 fps; 20 @40% ≈ 260 fps [I]. No repo doc gives the product's real duty cycle.
- Run-ahead: whole-turn audio + 400-frame queue lets the GPU absorb bursts (latency, not stutter, if the
  scheduler is deadline-aware).

## Environment constraints (hard)
- Disk: root overlay 1.2–1.4 GB free (100%), shrinking (shared with other agents/sessions). One UNet
  TRT .ts artifact = 2.21 GB; full UNet FP16 ONNX ~1.7 GB. A duplicate 2.2 GB copy exists at
  MuseTalk/models/trt_downloaded_backup_20260915; ~570 MB of SoulX .bundle files sit in /workspace.
  Deleting anything is the USER's decision.
- RAM: 30 GB (cgroup max 31.1 GB; oom_kill counter 12). api_server RSS 7–10.5 GB at 10 sessions; >1000
  threads. A native bs16 engine build peaks ~11 GB host RSS. Other sessions (a Codex VS Code session
  running run_wall_api.sh; a SoulX session) share the box and the GPU.
- VRAM: 12 GB; server ~5–6 GB. Dual 8+16 engines OOMed even on 24 GB historically; NVENC 200 MB/session.
- Toolchain pinned: torch 2.5.1+cu121, torch_tensorrt 2.5.0, TRT 10.3, modelopt 0.23.2 (onnx<1.18).
  SoulX venv has torch 2.7.1 + TRT 10.9 (upgrade path exists in a separate venv).
- Quality gates in repo: UNet capture gate mae_max ≤0.01, max_abs ≤0.5 (scripts/validate_unet_backend.py,
  MUSETALK_UNET_CALIBRATION_CAPTURE); chin exactness gates (verify_baseline.py 480 frames zero diff,
  pixel_checks, SHA sequential==pipelined); TAESD ROI mae 0.011–0.016 accepted; SyncNet only an
  uncalibrated relative diagnostic; human visual review on labelled same-speed video is the arbiter.
- WebRTC acceptance: smooth 20 fps = avg frame interval ~0.050 s, max interval ≤0.100 s, zero strict
  stalls, no held/duplicated frames beyond a small budget; aggregate fps = concurrency / avg interval.
