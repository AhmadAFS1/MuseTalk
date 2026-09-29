export const meta = {
  name: 'musetalk-300fps-understand',
  description: 'Map every per-frame cost in the MuseTalk serving path on the RTX 4070S and probe the key levers for a 300 fps feasibility analysis',
  phases: [
    { title: 'Read', detail: 'parallel read-only subsystem maps + history mining' },
    { title: 'Probe', detail: 'serialized GPU probes (UNet, TAESD) then CPU probe' },
  ],
}

const SCRATCH = '/tmp/claude-0/-workspace/866169db-0af2-4cde-9a98-79a398efeda7/scratchpad/throughput300'

const CONTEXT = `
## Goal context (shared by every agent)
The user runs a MuseTalk (v1.5) real-time talking-head server at /workspace/MuseTalk on ONE RTX 4070 SUPER
(12 GB, sm_89 Ada, 220 W, driver 595.84 / CUDA 13.2) with a Ryzen 9 7950X (16C/32T), 30 GB RAM.
They want to know whether ~300 fps AGGREGATE generation (15 concurrent WebRTC streams x 20 fps; stretch
20 streams = 400 fps) is achievable on this machine, and then a full plan. THIS TURN IS ANALYSIS + PLAN ONLY:
nothing in the repo is implemented or changed.

Budget arithmetic: 300 fps = 3.33 ms of GPU time per frame (aggregate), and ~(32 threads x 1000 / 300) CPU-ms per
frame spread across cores.

Known state (from docs/musetalk_4070s_pipeline_component_analysis_2026-09-22.md and experiments/*):
- Live server env: MUSETALK_VAE_BACKEND=taesd (compiled TAESD, scripts/vae_fast_decoder.py), MUSETALK_UNET_BACKEND=trt,
  static batch-8 TensorRT FP16 UNet via torch_tensorrt TorchScript (models/tensorrt_unet_sm89_bs8_local/unet_trt.ts,
  ~850M-param SD1.5-shaped UNet: block_out 320/640/1280/1280, in_channels 8, cross_attention_dim 384, latent 32x32,
  whisper encoder_hidden_states [50,384]). Larger batches are SPLIT into bs8 calls (no amortization; bs16 = 2x bs8).
  No native bs16 engine exists on this host.
- Measured @bs8: UNet TRT ~24-26 ms (~3.1 ms/frame), TAESD compiled ~6.85 ms (~0.86 ms/frame), post <1 ms.
  Offline "pipelined standard" harness: 250.8 fps (GPU-bound). With 100% chin alignment (FaceMesh tracking + masks +
  warp, CPU): 184.9 / 193.4 fps. Diversity batch warm render: 148-171 fps. Serial TAESD: 204-215 fps.
  These are GPU-to-composite render numbers EXCLUDING audio features/TTS, encoding, scheduling, WebRTC.
- Live multi-stream WebRTC capacity with TAESD on this 4070S has NEVER been measured beyond 2-3 streams.
  Historically (RTX 6000 Ada, 2026-06-08) live WebRTC aggregate plateaued at ~112 fps well below model-path capacity,
  so the serving path (scheduler/encode/WebRTC/GIL) is a real ceiling, not just the model.
- Chin alignment (user-required, "TAESD + 100% chin alignment") is validated OFFLINE only
  (experiments/chin_fps_validation_20260927, character_factory/h3_avatar_workflow/chin.py); not yet in the live API.
- Prior SoulX-FlashHead work on this same GPU (/workspace/SoulX-FlashHead/docs/research/PRO_40FPS_ITERATION_LOG_2026-09-21.md)
  closed: 2-session concurrency (+1.3%), FP8 Q/DQ Conv3d (no FP8 Conv3d kernel in TRT 10.9), some CUDA-graph cases.
  It succeeded with block-bypass + distillation of a decoder. Different model, but the same card.

## Hard guardrails
- READ-ONLY on /workspace. Do not edit, move, or delete any repo file. Put any script or output you create under
  ${SCRATCH}/ (create subdirs as needed).
- DISK: the root overlay has only ~2.6 GB free (99% full). Never write a file >100 MB; keep your total writes <400 MB;
  no ONNX export of the full UNet, no serialized full-model TRT engines. Delete your large temporaries when done.
- A live api_server.py (pid 3240171, port 8000, ~5 GB VRAM, normally idle) is running. Never kill/signal/restart it and
  never send it generation requests. (Reading files it uses is fine.)
- Python: /workspace/.venvs/musetalk_trt_stagewise/bin/python (torch 2.5.1+cu121, TensorRT 10.3, torch_tensorrt 2.5,
  modelopt 0.23.2). MediaPipe FaceMesh lives in /workspace/SoulX-FlashHead/.venv/bin/python.
- Label every number you report with its provenance: [measured-now], [doc:<path>], or [inferred-from-code].
`

const READER_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string', description: '5-15 sentence narrative of how this subsystem works and what limits throughput' },
    per_frame_costs: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          stage: { type: 'string' },
          location: { type: 'string', description: 'file:line or function' },
          cost_ms_per_frame: { type: 'string', description: 'number or range, amortized per output frame; say "unknown" if not known' },
          resource: { type: 'string', enum: ['gpu', 'cpu', 'cpu-gil', 'io', 'mixed'] },
          scales_with: { type: 'string', description: 'what it scales with: frames, streams, turns, frame area, etc.' },
          provenance: { type: 'string' },
        },
        required: ['stage', 'location', 'cost_ms_per_frame', 'resource', 'provenance'],
      },
    },
    levers: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          description: { type: 'string' },
          expected_gain: { type: 'string' },
          quality_risk: { type: 'string', enum: ['none', 'low', 'medium', 'high', 'unknown'] },
          status: { type: 'string', enum: ['untried', 'tried-open', 'tried-closed', 'shipped'] },
          evidence: { type: 'string' },
        },
        required: ['name', 'description', 'expected_gain', 'quality_risk', 'status', 'evidence'],
      },
    },
    constraints: { type: 'array', items: { type: 'string' }, description: 'hard limits: VRAM, GIL, static shapes, ordering, latency, etc.' },
    open_questions: { type: 'array', items: { type: 'string' } },
    key_files: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'per_frame_costs', 'levers', 'constraints', 'open_questions', 'key_files'],
}

const PROBE_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    measurements: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          name: { type: 'string' },
          config: { type: 'string' },
          value: { type: 'string' },
          unit: { type: 'string' },
          n_runs: { type: 'string' },
          notes: { type: 'string' },
        },
        required: ['name', 'config', 'value', 'unit'],
      },
    },
    conclusions: { type: 'array', items: { type: 'string' } },
    contamination_flags: { type: 'array', items: { type: 'string' }, description: 'any sign another GPU/CPU workload overlapped the timing' },
    scripts_written: { type: 'array', items: { type: 'string' } },
    could_not_measure: { type: 'array', items: { type: 'string' } },
  },
  required: ['summary', 'measurements', 'conclusions', 'contamination_flags', 'scripts_written', 'could_not_measure'],
}

const READERS = [
  {
    key: 'model-path',
    prompt: `You are mapping the GPU MODEL PATH of MuseTalk for a 300 fps feasibility study.
Read: MuseTalk/musetalk/models/unet.py, musetalk/models/vae.py, scripts/trt_runtime.py, scripts/vae_fast_decoder.py,
scripts/tensorrt_export.py, scripts/select_unet_trt_profile.py, scripts/validate_unet_backend.py, the UNet TRT meta json files
under models/tensorrt_unet_*/, .runtime/*.env, and docs/musetalk_4070s_pipeline_component_analysis_2026-09-22.md plus
docs/fps_comparisons/4070s_20260922/*.json (unet_probe*.json, capacity.json, component_profile.json, taesd.json).
Determine precisely:
1. How the UNet is invoked per batch (timesteps, positional encoding of audio, dtype, how bs>8 is split, whether the
   torch_tensorrt TorchScript wrapper adds host overhead, whether CUDA graphs are used, which CUDA stream).
2. How TAESD is invoked (compile mode, warmup, batch shapes, whether output is cropped, dtype, the latent scaling convention).
3. UNet compute: count FLOPs per frame at 32x32 latent. Do this ANALYTICALLY or with torch.utils.flop_counter on the
   'meta' device / CPU (do NOT use the GPU in this task). Break down: convs vs attention projections/FF GEMMs vs
   attention score/softmax, per resolution stage (32x32, 16x16, 8x8, 4x4). Also params per stage.
4. Using RTX 4070 SUPER peak rates (look them up from your knowledge: 56 SMs; FP16 tensor dense w/ FP16 accumulate
   ~142 TFLOPS, w/ FP32 accumulate ~71 TFLOPS on GeForce; FP8 ~284 dense (half with FP32 acc); INT8 ~284 TOPS;
   ~504 GB/s), compute the achieved TFLOPS of the measured ~24-25 ms bs8 TRT UNet and the theoretical floor for FP16
   and FP8/INT8. State assumptions.
5. Which parts of the UNet output are actually used (only lower-face rows ~104..256 of the 256 decode are blended),
   and whether any computation is source-only (depends only on the avatar frame, not audio) and thus cacheable per
   cycle frame.
6. TAESD architecture (no GroupNorm?) and whether a row-cropped decode can be made EXACT with a halo; its receptive field.
List every lever you see in this path (FP8/INT8 UNet, native bs16/bs24 engine, TRT version upgrade, CUDA graphs,
direct TRT runtime instead of TorchScript wrapper, UNet block pruning + distillation, DeepCache-style cross-frame
feature reuse, TAESD crop/TRT, etc.) with evidence.`,
  },
  {
    key: 'scheduler-serving',
    prompt: `You are mapping the SERVING PATH (scheduler, threading, WebRTC) of MuseTalk for a 300 fps feasibility study
(15 concurrent WebRTC streams x 20 fps). Read: MuseTalk/scripts/hls_gpu_scheduler.py (all of it, especially
_run_generation_batch, batch formation, fixed_batch_sizes, pinned staging, frame callbacks, compose threading),
api_server.py (the WebRTC stream endpoint(s), session setup, how audio is turned into features, frame_callback,
push_bgr_frames_batch, WEBRTC_BATCH_FRAME_CALLBACK), scripts/webrtc_tracks.py, scripts/webrtc_native_vp8.py,
scripts/webrtc_manager.py, scripts/webrtc_motion_playback.py, scripts/avatar_manager_parallel.py,
scripts/concurrent_gpu_manager.py, scripts/runtime_cpu_tuning.py, scripts/avatar_cache.py, and start_params.md /
current_start_param_reference.md for how the server is launched (profiles like throughput_record).
Determine precisely:
1. The thread/process model: which work runs on the asyncio loop, which in threads, which holds the GIL, how many
   compose workers, how encode happens (PyAV libvpx/libx264 vs native VP8 module), and whether GPU work for batch N+1
   overlaps CPU work for batch N in the LIVE path (the offline harness overlaps; does the server?).
2. How frames from multiple streams are combined into a GPU batch (max_combined_batch_size, fairness, padding waste
   when a batch is partially full), and what batch sizes are warmed.
3. What happens during idle/silence: is the UNet run for idle frames or are raw source frames passed through
   (exact_silence, WEBRTC_RAW_IDLE_POSE, idle_track)? This decides whether GPU fps demand = streams x 20 x speaking-duty.
4. Per-frame CPU costs on the live path (color conversion, resize, blend, encode, RTP packetization, SRTP) with any
   timing you can find in logs/docs. Look at /workspace/benchmarks/musetalk-webrtc-load*.json and
   /workspace/experiments/chinese_bob_webrtc_20260927/*.
5. Every serialization point that would cap aggregate throughput below GPU capacity (single scheduler thread, GIL
   hot loops, per-frame asyncio.run_coroutine_threadsafe, per-frame numpy copies, locks).
6. Why the RTX 6000 Ada live WebRTC plateaued at ~112 fps (read load_test_webrtc_rtx6000ada_*.md,
   current_cross_server_throughput_findings.md, docs/webrtc_load_test_findings_2026-06-07.md) and whether those causes
   still exist in today's code.
List levers (multi-process sharding with CUDA MPS, moving encode to NVENC H.264 — note h264_nvenc IS available in
ffmpeg here —, larger cross-stream batches, GPU-side compose, zero-copy, removing per-frame Python work, etc.).`,
  },
  {
    key: 'cpu-post-chin',
    prompt: `You are mapping CPU-SIDE POST-PROCESSING of MuseTalk for a 300 fps feasibility study. The user REQUIRES
"TAESD + 100% chin alignment" going forward. Read: MuseTalk/scripts/api_avatar.py (compose_frame and everything it calls),
musetalk/utils/blending.py (get_image_blending_with_plan, fixed-point path, and the new chin code added in commit
c284891 — use 'git -C /workspace/MuseTalk show c284891 -- musetalk/utils/blending.py scripts/api_avatar.py'),
scripts/benchmark_compose_frame.py, /workspace/experiments/chin_fps_validation_20260927/{README.md,run.py,
fast_geometry.py,tracker_worker.py,taesd_pipelined_results.json}, /workspace/experiments/chin_geometry_ab_20260927/PERFORMANCE.md,
/workspace/MuseTalk/character_factory/h3_avatar_workflow/{chin.py,tracker_worker.py,render_stage.py,WORKFLOW.md},
/workspace/experiments/avatar_diversity_20260927/{README.md,MUSETALK_TESTS.md}.
Determine precisely:
1. Per-frame CPU cost of each post-GPU step, standard path vs 100% chin path: GPU->CPU copy, resize to bbox, blend,
   FaceMesh generated-face tracking, dynamic lip guard/mask, chin warp, the one-frame-lookahead jaw filter, color
   conversion for the encoder. Use the measured numbers in those reports.
2. Which parts are exact-cacheable per source frame (source jaw cap, source lip hull, baseline alpha) vs necessarily
   dynamic per generated frame.
3. Is the chin path integrated in the live api_server path today? If not, what integration would look like and what it
   costs per frame.
4. How the chin path parallelizes: GIL-free (OpenCV/numpy release GIL?) vs Python-bound, and the core count needed at
   300 fps and 400 fps on a 7950X (16C/32T), leaving room for encode + WebRTC + TTS/audio.
5. Could the dynamic tracking be replaced by something cheaper yet equivalent (e.g., landmarks from the UNet latent,
   a tiny GPU landmark net, tracking at lower res, reusing source landmarks + a delta)? Flag which are exact vs
   approximate.
List levers with gains and quality risk.`,
  },
  {
    key: 'audio-tts',
    prompt: `You are mapping the AUDIO FEATURE and TTS path of MuseTalk for a 300 fps feasibility study (15-20 concurrent
conversational WebRTC streams on one RTX 4070 SUPER). Read: MuseTalk/musetalk/utils/audio_processor.py (or wherever
Whisper features are computed; grep for 'whisper', 'feature_extractor', 'get_audio_feature', 'get_whisper_chunk'),
api_server.py (how an uploaded / TTS audio turn becomes whisper chunks and frames; per-turn setup latency;
time_to_live_ready), scripts/kokoro_tts.py, scripts/webrtc_audio_timeline.py, scripts/realtime_inference.py, and
docs on chatterbox/kokoro (docs/chatterbox_autoscaling_plan.md). Also check what TTS the live server uses and whether
TTS runs on THIS GPU/CPU or elsewhere.
Determine precisely:
1. Whisper model size, where it runs (GPU/CPU, dtype), how audio is windowed (30 s padding?), and the cost per second
   of audio (doc says 0.88 s for 8 s audio, ~9x realtime — find what dominates: mel on CPU, model, padding, Python loops).
   At 15 streams talking continuously that is 15 s of audio per wall second: would Whisper become a GPU/CPU
   bottleneck or contend with the UNet? Could it be batched across streams / run on CPU cores / made incremental?
2. TTS: if Kokoro/other runs locally, its GPU/CPU cost per second of speech and VRAM, and whether it must share the 4070S.
3. Per-turn fixed costs (setup, feature extraction, first-batch latency) and how they interact with batching.
4. The conversational duty cycle: during a call the avatar speaks only part of the time. Find any code/docs that
   quantify speaking vs idle time, and whether idle frames cost GPU (raw idle passthrough vs UNet).
List levers (batched whisper, whisper on CPU, FP16/TRT whisper, streaming features, TTS placement) with evidence.
You may run small CPU-only timing of the whisper feature path if it is cheap and safe (no GPU, <2 min), writing only
under the scratchpad.`,
  },
  {
    key: 'history',
    prompt: `You are the HISTORIAN for a MuseTalk 300 fps feasibility study. Mine every prior throughput document so the plan
does not repeat closed experiments. Read (skim large ones smartly with grep, then read relevant sections fully):
MuseTalk/archive_hls_throughput_experiment_history.md, archive_hls_throughput_architecture_notes.md,
current_cross_server_throughput_findings.md, current_model_backend_acceleration_plan.md, current_model_backend_findings.md,
current_model_backend_execution_plan.md, current_unet_trt_throughput_findings_2026-05-29.md, current_tensorrt_environment_plan.md,
CPU_OPTIMIZATION_ANALYSIS.md, docs/musetalk_quantization_optimization_plan.md, docs/next_bottleneck_vae_late_block_plan_2026-06-11.md,
docs/webrtc_generation_optimization_results_2026-07-03.md, docs/webrtc_load_test_findings_2026-06-07.md,
docs/v100_webrtc_load_test_2026-05-22.md, docs/gpu_vram_budgeting.md, docs/musetalk_model_pipeline_breakdown.md,
load_test_webrtc_rtx6000ada_*.md, docs/musetalk_4070s_pipeline_component_analysis_2026-09-22.md,
/workspace/experiments/chin_fps_validation_20260927/README.md, /workspace/experiments/musetalk_sdvae_vs_taesd_new_avatars_20260925/README.md,
and /workspace/SoulX-FlashHead/docs/research/PRO_40FPS_ITERATION_LOG_2026-09-21.md (a different model on THIS SAME GPU;
extract transferable lessons on FP8, INT8, concurrency, CUDA graphs, TRT versions, distillation, block bypass).
Also 'git -C /workspace/MuseTalk log --stat -30' for context.
Produce: (a) a chronological table of every throughput lever tried on MuseTalk, on which GPU, result, and whether it is
closed/open and why; (b) the recurring gap between model-path fps and live WebRTC fps across GPUs and its documented
causes; (c) quality gates that exist (mae thresholds, syncnet at models/syncnet/latentsync_syncnet.pt, visual A/B
conventions, user decisions like "user judged SD-VAE vs TAESD essentially the same", "user rejected 224x224 as too
blurry", "user requires TAESD + 100% chin alignment"); (d) any user preferences that constrain the plan. Put (a)-(d) in
the summary/levers/constraints fields; include every tried lever in 'levers' with status.`,
  },
]

const UNET_PROBE = `You are a GPU PROBE agent measuring the MuseTalk UNet on the RTX 4070 SUPER for a 300 fps feasibility study.
Work in ${SCRATCH}/unet_probe/. Before each timing block run 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'
and record it; if utilization from another process is >5%, mark contamination and retry once later.
Load the real MuseTalk v1.5 UNet the way the repo does (read MuseTalk/musetalk/models/unet.py and
MuseTalk/docs/fps_comparisons/4070s_20260922/unet_probe.py or similar harnesses in that folder for how they built inputs;
real UNet I/O captures may exist under /workspace/benchmarks/same-avatar/unet-captures/ — use one if present, else
realistic random latents [B,8,32,32] fp16 and audio [B,50,384]). Use CUDA events, >=10 warmup, median of >=30.
Measure:
1. PyTorch FP16 eager UNet at bs 8, 16, 24, 32: ms/batch and ms/frame (amortization curve). Peak VRAM.
2. Per-top-level-block GPU time at bs8 and bs16 (conv_in, each down block, mid, each up block, conv_out) using CUDA events
   or torch.profiler; then within the most expensive blocks split resnets vs transformer (attn1/attn2/ff). Report the
   share of GEMM-like (linear/1x1) vs 3x3 conv vs attention-core time.
3. The shipping TRT artifact (/workspace/MuseTalk/models/tensorrt_unet_sm89_bs8_local/unet_trt.ts, load it with
   torch_tensorrt / torch.jit.load as scripts/trt_runtime.py does) at bs8, and bs16 via two bs8 calls; also measure host
   overhead: time of enqueue vs GPU time, and whether two bs8 calls on two CUDA streams overlap at all.
   Also try capturing the TRT module call in a torch.cuda.CUDAGraph and report if it works and its ms.
4. FP8 feasibility on this card: (a) torch._scaled_mm FP8 e4m3 vs FP16 torch.matmul TFLOPS at the UNet's actual
   linear shapes (M = B*tokens for tokens 1024/256/64 at bs8 and bs16; K,N from the 320/640/1280 channel attention
   projections and GEGLU FF), (b) build tiny TensorRT 10.3 engines IN MEMORY (do not serialize to disk; ONNX files must be
   <20 MB) for ONE representative ResNet block (GroupNorm+SiLU+3x3 conv+residual at 320ch 32x32 and 1280ch 8x8) and ONE
   transformer block, in FP16 vs INT8 (Q/DQ) vs FP8 (Q/DQ e4m3) via modelopt or hand-inserted QuantizeLinear/DequantizeLinear,
   and report per-layer kernel/tactic names if obtainable (IEngineInspector) to prove whether real FP8/INT8 conv and
   GEMM kernels are selected on sm_89 in TRT 10.3, and their speed vs FP16.
5. Optionally, if VRAM and time allow (<25 min, free VRAM >= 5 GB, and NOTHING >100 MB written to disk): build an
   in-memory torch_tensorrt FP16 UNet at a native bs16 static shape and time it vs 2x bs8. Skip if risky; say so.
Report achieved TFLOPS where you can (FLOPs per frame: compute with torch.utils.flop_counter.FlopCounterMode).
Free GPU memory at the end (exit the processes).`

const TAESD_PROBE = `You are a GPU PROBE agent measuring the TAESD decoder path of MuseTalk on the RTX 4070 SUPER for a 300 fps
feasibility study. Work in ${SCRATCH}/taesd_probe/. Before each timing block record 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader';
flag contamination if another process is active. Read MuseTalk/scripts/vae_fast_decoder.py and use the vendored weights at
MuseTalk/models/taesd/ with the repo's exact convention (TAESD consumes the scaled latent, output /2+0.5).
Use real post-UNet latents if you can find saved ones (grep docs/fps_comparisons/4070s_20260922/*.py for how they produced
them; /workspace/benchmarks/same-avatar/unet-captures may hold UNet outputs), else random N(0,1) latents [B,4,32,32].
CUDA events, >=10 warmup, median of >=30. Measure:
1. TAESD decode eager and torch.compile(max-autotune) at bs 8, 16, 24, 32 (ms/batch, ms/frame), and with
   'reduce-overhead' (CUDA graphs) mode.
2. ROW-CROPPED decode: MuseTalk only blends rows ~104..256 of the 256x256 output. Compute TAESD's receptive field, then
   decode only latent rows [13-h : 32] for halo h (latent row 13 = pixel 104) and compare the kept pixel rows to the
   full decode: report max_abs difference per halo (expect EXACT 0 above some halo if TAESD has no global ops like
   GroupNorm), and ms/frame saved.
3. Whether TAESD on a second CUDA stream overlaps with a concurrently running UNet (use the PyTorch FP16 UNet from
   MuseTalk/musetalk/models/unet.py or a stand-in heavy matmul loop if loading the UNet is too heavy): report the combined
   wall time vs serial sum.
4. The GPU->CPU handoff: time of converting decoded [B,3,256,256] fp16 to uint8 BGR NHWC on GPU then D2H into pinned
   memory, at bs16; and alternatively the cost if the final resize-to-bbox + alpha blend were done ON GPU (write a quick
   torch implementation with a realistic 512x832 frame and ~170x210 bbox to estimate) vs the CPU path.
5. An optional TensorRT FP16 build of TAESD (small model, in memory only) at bs16: ms/frame.
Free GPU memory at the end.`

const CPU_PROBE = `You are a CPU/ENCODER PROBE agent for a MuseTalk 300 fps feasibility study on a Ryzen 9 7950X (16C/32T) +
RTX 4070 SUPER. Work in ${SCRATCH}/cpu_probe/. Record 'uptime' load average and nvidia-smi utilization before each block and
flag contamination. Measure, using realistic frame sizes from the live avatars (512x832 portrait; also 384x672):
1. Video encoding per frame at 20 fps realtime settings, 2.5 Mbps target: (a) libvpx VP8 via PyAV (realtime deadline,
   cpu-used as the repo uses — read MuseTalk/scripts/webrtc_tracks.py and scripts/webrtc_native_vp8.py for the actual
   settings; if the native VP8 module in MuseTalk/.runtime/native_vp8 is importable from the musetalk_trt_stagewise venv,
   time it too), (b) libx264 ultrafast/zerolatency, (c) h264_nvenc low-latency preset via PyAV or ffmpeg pipe. Report
   ms/frame single-thread and aggregate encode throughput when running 15 and 20 concurrent encoders (threads/processes),
   and CPU cores consumed.
2. The standard compose path: time MuseTalk/scripts/benchmark_compose_frame.py if runnable without the GPU server (read it
   first; do not start the api server), else re-implement resize-to-bbox + fixed-point alpha blend on a 512x832 frame and
   ~170x210 bbox and time it; show scaling across 8/16/24 threads.
3. MediaPipe FaceMesh (refined) per-frame cost on a 256x256 or bbox-sized face crop using
   /workspace/SoulX-FlashHead/.venv/bin/python, single process and scaling across 4/8/12 processes (aggregate frames/s).
   Use any face image from /workspace/experiments/chin_fps_validation_20260927/ or the avatar caches.
4. Whisper-tiny feature extraction cost for 8 s of audio on CPU vs the doc's 0.88 s figure (read MuseTalk audio_processor to
   replicate; use data/audio/yongen.wav). Only if it can be done quickly; do not use more than ~1 GB VRAM if you use the GPU.
From the results, compute the CPU core budget required at 300 fps and 400 fps for: compose + chin path + encode +
WebRTC packetization (estimate) and say whether a 7950X fits it and with how much headroom.`

phase('Read')
const readersP = parallel(READERS.map(r => () =>
  agent(`${CONTEXT}\n\n## Your task (${r.key})\n${r.prompt}`, { label: `read:${r.key}`, phase: 'Read', schema: READER_SCHEMA })
    .then(res => res ? { key: r.key, ...res } : null)))

// GPU probes must not overlap each other; run them serially, then the CPU probe (which would perturb host timing).
const probesP = (async () => {
  const unet = await agent(`${CONTEXT}\n\n## Your task\n${UNET_PROBE}`, { label: 'probe:unet', phase: 'Probe', schema: PROBE_SCHEMA })
  log('UNet probe done; starting TAESD probe')
  const taesd = await agent(`${CONTEXT}\n\n## Your task\n${TAESD_PROBE}`, { label: 'probe:taesd', phase: 'Probe', schema: PROBE_SCHEMA })
  log('TAESD probe done; starting CPU/encoder probe')
  const cpu = await agent(`${CONTEXT}\n\n## Your task\n${CPU_PROBE}`, { label: 'probe:cpu', phase: 'Probe', schema: PROBE_SCHEMA })
  return { unet, taesd, cpu }
})()

const [readers, probes] = await Promise.all([readersP, probesP])
return { readers: readers.filter(Boolean), probes }
