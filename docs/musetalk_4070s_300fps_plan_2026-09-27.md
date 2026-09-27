# MuseTalk on RTX 4070 SUPER: path to 300 fps (15 × 20 fps streams)

Date: 2026-09-27. Status: **analysis and plan only — nothing in the serving code was changed.**

How this was produced:
- Measurements: probe runs on this box on 2026-09-27. They covered UNet (TRT, eager, per-block ONNX/INT8/FP8 engines built in memory), TAESD (compiled, TRT, staged crop, INT8), CPU/encoder load (VP8, x264, NVENC, FaceMesh, compose, RTP/Opus, Whisper) and a loopback aiortc serving harness.
- Reading: read-only passes over the model path, scheduler/WebRTC, CPU/chin path, audio/TTS, and every prior throughput document.
- Planning: four independent plans (exact-first, serving-first, model-acceleration, capacity/product) were scored and merged, attacked by three adversarial reviewers (GPU arithmetic; serving/CPU/RAM/VRAM; completeness vs your constraints), and corrected by a final editor.
- Evidence: all probe JSON, scripts and raw results are in `docs/fps_comparisons/4070s_300fps_20260927/`.

> **Incident during the analysis.** Probe runs (in-memory TRT engine builds peaking at ~11 GB host RSS, plus a loopback load test) overlapped your 10-stream wall test between 19:35 and 19:39 UTC. That server (pid 3305906) died at 19:39:24 with 0.8 GB RAM available. The probes very likely contributed to that OOM, and they also contaminated that run's GPU timings. The box-guard item 0.1 and decision D2 exist so this cannot recur.

## Executive summary

**Can this machine do 300 fps (15 × 20 fps, all speaking, full TAESD + 100% chin + refined seam recipe)?**
- **Plausibly, without quality loss, but it is not proven.**
- Projected delivered throughput is **~311 fps (range 265–338)** after the serving fixes, TRT TAESD and an ONNX-parser stagewise FP16 UNet at bs16.
- Two numbers decide it, and neither has been measured yet:
  1. The chained stagewise FP16 UNet at bs16 must sustain **≤ 2.51 ms/frame** (item 2.2, a one-session bench with no backend code).
  2. The live path must deliver **≥ 0.90** of the GPU ceiling after Phase 1.
- On today's evidence: **roughly a coin flip for lossless 300**. It is good odds for 300 with one lossy backstop that passes your video review.

**Where the "160 fps" comes from:**
- It is the offline one-stream render harness with chin alignment (148–171 fps; 163–169 with the refined seam).
- It is not live capacity, it is per-stream CPU-chain bound, and it was rendered at 24 fps.
- The live server today is structurally capped at **~174–203 fps**, even without chin:
  - it does 3 CUDA syncs per batch, a blocking D2H, and blocks the scheduler thread on the WebRTC handoff;
  - it decodes the idle mp4 on the asyncio loop, which saturates at ~183 fps;
  - the H.264 path actually runs libx264 medium (the NVENC patch is dead code on aiortc 1.14), about 8 cores at 300 fps.
- Your contaminated 10-stream run got ~136–145 fresh fps.

**What the GPU can do (measured):**
- Shipping UNet: 3.0 ms/frame. TAESD: 0.74–0.85 ms/frame. Ceiling ~253–259 fps.
- The UNet has headroom: it runs at ~41% of the FP16 peak, and 31% of its time is outside tensor cores, largely because torch_tensorrt decomposes GroupNorm.
- Native bs16 measured 2.79 ms/frame.
- ONNX-parser per-block engines measured 2.57 ms/frame at bs8.
- TRT TAESD measured 0.49 ms/frame at full height.
- **FP8 is dead on this card** (no FP8 conv kernels in TRT 10.3; FP8 has the same peak as FP16-accumulate on GeForce Ada).
- **INT8 kernels are real** (1.8–2.1× on ResNet blocks). Naive INT8 quality is ~5 dB short on the rows the blend uses.
- The exact TAESD row-crop (0.345 ms) **cannot be used with the chin recipe**, because the FaceMesh tracker reads the full generated face.

**Conversational sessions (not a wall):**
- 15 sessions at 50% speaking duty need ~240 fps at p99.
- That is reachable after Phase 1 + TRT TAESD (~264 fps expected) **without touching the UNet**.

**400 fps (20 streams all speaking):** not reachable losslessly. It needs a mixed-INT8 UNet that passes review (not there today), plus INT8 TAESD or tracker-row fill, plus the process split.

**CPU is not the limit.** The chin path at 300 fps measured ~11–14 of 32 logical CPUs. **The per-process GIL is**: moving work out of one Python process (Phase 4) is probable at 15 streams with chin and required at 20.

**Box prerequisites that block the decisive measurements:**
- Disk has 1.3 GB free; one UNet engine is 2.2 GB.
- RAM available is ~12 GB, while `/dev/shm/soulx-lfs-state-20260919` holds 7.9 GB and the 15-session server budget is 9–14.5 GB.
- There is no box-wide GPU lease yet, and the Codex and SoulX sessions share the machine.

**First moves, in order:**
1. **D1/D2:** free disk and `/dev/shm`, and agree a GPU lease.
2. **Phase 0:** box guard, CUDA-event telemetry, golden-SHA replay, load harness v2 with correct freshness counters, a 20 fps chin reference, and a clean layered baseline.
3. **Item 2.2:** the go/no-go UNet bench.
4. **Phase 1:** serving fixes, all bit-exact and flag-gated.

**Effort:** lossless 15-stream sign-off is about 5–7 weeks without the process split and 8–10 with it.

**Decisions only you can make:** see §9. The most consequential are D1/D2 (disk and RAM), D4 (does "15 streams" mean a 100%-speaking wall or conversational duty), D14 (do 1–3 LSB FP16 runtime changes count as "no quality change") and D18 (re-accept the chin recipe at 20 fps).

---

**Evidence tags**

| Tag | Meaning |
|---|---|
| [M] | Workflow-1 probe measurement. Files are listed in §10. |
| [M-now] | Read-only box check made while planning on 2026-09-27: df, free, du, ps, lscpu, stat, nvidia-smi query, git status. |
| [D] | Repo or experiment doc, or a log. |
| [C file:line] | Read in the source code. |
| [I] | Inference, with the arithmetic shown. |

- **fps** means fresh generated frames per second, summed over all streams.
- **Required recipe** means TAESD decoder + MuseTalk native avatar encoder (prepared SD-VAE latents, no runtime cost [D chin_fps README:3,49]) + 100% chin alignment + refined seam + expressive H3 source.
- **Live efficiency e** means delivered fps ÷ GPU ceiling fps.

---

## 1. Verdict

**300 fps (15 × 20 fps, all speaking at once, full required recipe) is projected, not measured.**
- **Expected: about 311 fps delivered, range 265–338** [I, §5].
- The expected case clears 300 by about 4%. The low case misses.

**It depends on two things that nobody has measured:**
1. **The ONNX-parser stagewise FP16 UNet at bs16 must run at ≤2.51 ms/frame sustained.**
   - Projected: 2.32 / 2.40 / 2.67 ms [I, §3.1].
2. **Live in-process efficiency must reach ≥0.90 at saturation after Phase 1.**
   - Assumed: 0.88 / 0.93 / 0.97 [I].
   - Every live ratio on record is lower:
     - 0.53–0.56 today, and that run was contaminated [D/I];
     - 0.80–0.85 on the 3090, with the old serial path [D].

**What is measured and what is projected**

| Piece | Value | Status |
|---|---|---|
| Shipping bs8 .ts UNet + runtime CUDA graph | 23.43 ms/bs8 = 2.93 ms/frame | [M p6]. A 60-batch power-capped burst (~1.5 s), not a ≥60 s run |
| torch_tensorrt native bs16 + graph | 44.63 ms/bs16 = 2.79 ms/frame | [M ttrt_bs16]. Two rounds of 30 calls |
| ONNX-parser FP16, 11 per-block engines, bs8 | 20.575 ms sum = 2.57 ms/frame | [M]. Each block was timed alone with a sync per call; the blocks were never chained and never run sustained |
| bs16 factor for the ONNX path | 0.95 (0.92–1.0) | [I] from 5 micro-blocks. 3 of the 10 micro-runs had your server (pid 3305906) resident on the GPU [M gpu_log] |
| Full-height TRT TAESD | 0.486 ms/frame, FP16 NCHW output | [M p5]. The uint8 BGR output that item 2.1 ships was never built at full height, so 0.50–0.55 [I] |
| Other GPU (H2D + Whisper) | 0.08 (0.05–0.10) ms/frame | [D: H2D ~0.5 ms/bs8, docs/musetalk_4070s_pipeline_component_analysis_2026-09-22.md:80; M: Whisper 0.02] |
| Live efficiency after P1 | 0.88 / 0.93 / 0.97 | [I]. The top value is the offline pipelined harness, 250.8 / 258.7 = 0.97 [D/M] |
| Chin CPU chain, 15 streams in one process | 0 late frames | [M C_chin15]. Threads only; chin was a lighter stand-in; no GPU feeder, no asyncio or aiortc |

**Closest-to-measured stack** (torch_tensorrt bs16 + graph, full-height TRT TAESD):
- 2.79 + 0.51 + 0.08 = 3.38 ms → 296 fps ceiling → 256–291 delivered [I].
- It misses 300 at any efficiency.

**Lower-risk intermediate** (ONNX stagewise at bs8 + graph, item 2.3a):
- 2.53 + 0.51 + 0.08 = 3.12 ms → 321 ceiling → 298 expected (265–316) [I].
- The bs8 per-block sum is the most directly measured part of the ONNX gain.

**The single measurement that most changes the conclusion is item 2.2.**
- **What it is:** the 11 FP16 per-block engines chained exactly as the shipping stagewise backend, at bs16 and bs8, timed for ≥60 s at the power cap on a quiet box.
- **Why it decides:** it moves the GPU ceiling anywhere between 291 and 349 fps [I, §5].
  - ≤2.51 ms: lossless 300 is physically available, and the rest is serving engineering.
  - >2.64 ms: lossless 300 is gone, however good the serving path gets (K2).
- **Cost:** no backend code, and about one build-and-bench session.
- **Second most important:** the saturated in-process efficiency at the Phase 1 exit. It decides whether Phase 4 (the process split) is needed for 300.

**Go threshold for item 2.2** [I: UNet_max = 1000·e/300 − (0.51 TAESD + 0.08 other)]

| Live efficiency e | 0.85 | 0.88 | 0.90 | 0.93 | 0.95 | 0.97 |
|---|---|---|---|---|---|---|
| Max UNet ms/frame for 300 fps | 2.24 | 2.34 | 2.41 | 2.51 | 2.58 | 2.64 |

- **Go:** ≤2.51 ms, which gives lossless 300 at the expected efficiency.
- **Marginal:** 2.51–2.64 ms. This needs a measured e ≥0.95, or a lossy backstop (6.1 or 6.2).
- **No-go (K2):** >2.64 ms.
- The Phase 2 exit, K2 and item 2.2 all use this same rule, fed with the measured Phase 1 efficiency.

**Odds** [I, judgment, not computed]:
- Lossless 300 is roughly a coin flip on today's evidence.
- Both deciding quantities sit inside their projected ranges, but neither has been measured. The efficiency needed (0.90) is above every live ratio on record.
- The lossy backstops add margin only if they pass your video review. Their pass rates are unknown; for PTQ-only INT8 the rate is likely low (see 400 below).

**If the product is independent conversations rather than a wall:**
- 15 sessions at 50% speaking duty need 240 fps at p99. 20 sessions at 50% duty need 300 fps [I, binomial, §5].
- About 264 fps is expected after P1 + TRT TAESD, with no UNet rebuild [I: 2.93 + 0.51 + 0.08 = 3.52 ms → 284 × 0.93]. That serves 17 sessions at 50% duty (p99) [I].

**400 fps (20 × 20, all speaking): not reachable losslessly. The lossy paths need INT8 quality that does not exist yet.**
- **Best lossless:** 311 expected, 338 in the high case [I].
- **Mixed-INT8 UNet alone** (recipe v1, up0 and up1 kept in FP16) plus full-height FP16 TAESD: 2.49 ms → **373 expected**, short of 400 [I].
- **400 needs mixed-INT8 plus a second lossy lever:**
  - with INT8 TAESD at full height: 424 expected (357–465);
  - with tracker-row fill and the staged crop: 397 (335–434), just short [I].
- **INT8 quality today** [M quant_accuracy.json]:
  - The probe used 64 frames of one avatar, calibrated on the same avatar and decoded with SD-VAE, not TAESD.
  - On the rows the blend uses, naive INT8 gives 40.3 dB mean / 37.3 dB min, against a proposed gate of ≥45 / ≥42.
  - SmoothQuant made those rows worse: 39.9 / 35.9 dB.
  - The best PTQ variant (conv-only) reaches 6.09% latent rel-L2 and 41.1 / 38.3 dB on used rows.
  - Recipe v1's own error has never been measured. QAT or distillation is the realistic path.
- **It also needs Phase 4 (the process split)** [M D/F/G]:
  - 20 streams in one process with PyAV conversion: 19 late frames.
  - Same with cv2 conversion: 0 late frames, but p99 work was 49.5 ms against a 50 ms frame period.
  - 4 processes × 5 streams: 0 late frames at 12.6–23.8 ms.

**Binding ceilings, in the order they bind as each one is removed:**
1. **The serial single-process serving path (binds today).**
   - The scheduler does 3 syncs per batch plus a blocking D2H, and the handoff blocks it for 8–10 ms per 31.5–36 ms batch. That gives a structural ceiling of 174–203 fps [I: 8/(0.036+0.010); 8/(0.0315+0.008)].
   - Idle mp4 decode runs on the event loop. The loop saturates at 180–186 fps of total output, while a pre-decoded cache delivered 299.7 fps at 15 streams and 399.7 at 20 [M serving].
2. **UNet GPU time under the 220 W cap.**
   - After P2 the UNet is 2.40 of 2.99 ms per frame, 80% [I].
   - Power limit = maximum limit = 220 W [M-now]. SM clock under load is 2505–2625 MHz [M].
   - FP8 is a dead end on this card [M].
3. **The per-process GIL once chin runs live.**
   - The chin chain tops out at 567–665 fps per process [M U_ceiling_*].
   - The only 15-stream evidence (0 late; 34 ms mean / 40 ms p99 per 50 ms period) comes from a threads-only load.
     - Its chin stand-in cost about +1.5 ms, against 3.7–5.1 core-ms of refined chin beyond FaceMesh as written [M/D].
     - Per-frame work still inflated 4.5× versus one stream (7.58 → 34.1 ms) [M].
   - **Phase 4 is probable for 300 with chin live.** Item 3.0c decides.
4. **Host RAM (30 GB, shared).**
   - MemAvailable is 12.0 GB right now, with no MuseTalk server running. /dev/shm holds 8.9 GB, and Pylance has 4.4 GB RSS [M-now].
   - The in-process budget for 15 sessions is 9–14.5 GB [I, §3]. Adding the 4 GB guard, it does not fit, so **D2 is a hard prerequisite for any 15-stream work**.
   - The server has a 10.5 GB startup transient [D]. The cgroup oom_kill counter reads 12 [M-now].
5. **Disk:** 1.3 GB free [M-now] against 2.21 GB per .ts engine [M].
6. **NVENC:** capped at 12 sessions, using 2.47 GB of VRAM at 12 [M]. Encoding stays on the CPU with VP8.

**CPU is not binding in aggregate, but it has less headroom than the logical-CPU count suggests.**
- **Measured:** 10.6 logical CPUs busy at 300 fps [M C_chin15]; 12.1–13.1 at 400 [M D/F/G].
- **Added:** 0.5–3 for GPU feeding and aiortc [I].
- **Total:** 11–14 at 300 fps, 13–16 at 400.
- **Physical cores:** the 32 CPUs are SMT siblings on 16 cores; CPU n and CPU n+16 share core n [M-now lscpu].
  - Against about 20 core-equivalents [I: 16 × ~1.25], that is 55–70% at 300 and 65–80% at 400.
  - Contention already shows: FaceMesh CPU per frame rose from 3.4 to 13.2 ms under the C_chin15 load [M results_fm.jsonl; fm_during_C_chin15.json].

**Corrections to the evidence digest.** All were verified in files. The first one changes the verdict.
1. **The staged exact-crop TAESD (0.345 ms) is not usable with the required chin recipe.**
   - The tracker pastes the whole raw 256×256 generated face into the frame before FaceMesh sees it [C experiments/chin_fps_validation_20260927/run.py:38-41, called at :139; same in h3_avatar_workflow/backend.py Tracker].
   - Rows above R therefore change the landmarks and so the chin warp. The p7 row scan covered only the blend mask [M].
   - So the recipe needs full-height TAESD. The digest's "304.6 fps measured GPU path" becomes about **284 fps** for this recipe [I: 23.43/8 + 0.51 + 0.08 = 3.52 ms].
   - The crop comes back only as lossy item 6.1.
2. **CUDA-graph mode does not free the feeder core.**
   - `graph_no_spinner` still burns 23.5 ms of CPU per bs8 call [M gil.json].
   - The probe issues calls back to back with no per-call sync [C unet_probe/probe_gil.py:61-67], so this is launch-queue back-pressure, not a sync spin.
   - A blocking event (item 1.3) recovers the core only if depth-2 pipelining bounds the work in flight. Measure feeder CPU per batch in the real loop.
3. **ONNX-path bs16 factor and clocks.**
   - **bs16 factor.** Measured bs16 ÷ (2 × bs8) per micro-block [M trt_blocks_*.json]: res320 1.024, res1280 0.808, up1 0.946, tf320 1.069, tf1280 0.865. The simple mean is 0.942 [I]. Use 0.95 (range 0.92–1.0), not the torch_tensorrt bs16+graph factor of 0.9055.
   - **Clocks.** The per-block engines ran as short sync-per-call bursts of 0.03–4.8 ms, so they may have run at higher clocks than a sustained whole network (sustained 2520–2565 MHz [M p6]).
     - The 2790–2805 MHz "idle" samples are taken after 1.2 s of idle [C unet_probe/common.py:19-23]. They say nothing about clocks during the runs.
     - Treat the clock penalty as 0–9% [I: 2790/2565 = 1.088].
   - **What that means for the UNet:**
     - expected ratios plus the full penalty: 2.62 ms (−6%);
     - no bs16 gain plus the full penalty: 2.78 ms, no gain over the 2.79 fallback [I].
   - **No sum-of-blocks discount.**
     - The same-session eager block sum and full forward differ by only 0.6% (37.35 vs 37.11 ms [M eager.json]).
     - The earlier 39.46 ms came from a probe without cudnn.benchmark [C probe_trt_topblocks.py vs probe_eager.py:11].
     - Stagewise ships the block boundaries anyway.
4. **The "duplicate" engine is not a duplicate.**
   - The backup `models/trt_downloaded_backup_20260915/tensorrt_unet_static_bs8_20260529/unet_trt.ts` is 2,220,655,318 B, dated 2026-07-10.
   - The live engine is the symlink `models/tensorrt_unet_sm89_bs8_local/unet_trt.ts` → `../tensorrt_unet_static_bs8_20260529/unet_trt.ts`, 2,210,490,890 B, dated 2026-09-15 [M-now].
   - The backup is still a cleanup candidate; that is your call (D1).
5. **"Sustained" means 60-batch bursts.**
   - The p6 configs are N = 60 batches each, about 1.5–1.9 s; 48 s in total, with the GPU going from 40 to 66 °C [M p6_combined.py:17 and JSON timestamps].
   - No ms figure in this plan comes from a ≥60 s run. Items 0.8 and 2.2 provide the first ones.
6. **The digest's "bit-exact" for torch_tensorrt runtime cudagraphs mode is not recorded.**
   - max_abs 0 was measured for a manual `torch.cuda.CUDAGraph` capture (23.65 ms) [M trt.json ts_cudagraph_torch].
   - The runtime mode (23.79 ms in the short run, 23.43 in p6) has no accuracy field [M].

---

## 2. Where the "160 fps" number comes from vs what live WebRTC delivers today

| Number | What it is | Source |
|---|---|---|
| 163.4 / 168.8 fps | Refined-seam 100% chin, Latina / Japanese; one stream, warm offline harness, 24 fps playback | [D experiments/chin_seam_refinement_20260927/README.md:69-72] |
| 148–171 fps | Avatar diversity batch, one warm run per identity | [D experiments/avatar_diversity_20260927/README.md:30-39] |
| 182.8–197.7 fps | Chin100 v1, pipelined | [D chin_fps_validation_20260927/README.md:43] |
| 250.8 fps | Pipelined standard compose (no chin) | [D same README:38] |
| 204–219.6 fps | Serial TAESD | [D same README:26] |
| 258.7 fps | GPU-only live-equivalent path at bs8 (TRT UNet + compiled TAESD + post + pinned D2H), in a 60-batch power-capped burst | [M taesd_probe/p6_combined.json] |

**What the ~160 means.**
- It is **one stream in one process**, with FaceMesh IPC done one frame at a time: about 3 ms of tracking plus 2.3–3.9 ms of refined compose per frame, in series, with GPU overlap [M/D].
- It excludes TTS and audio features, scheduling, WebRTC and encoding. The READMEs say so.
- It was rendered at **24 fps** from 24 fps sources [C run.py:18-19; D pose-set.json fps 24.0]. Live generation runs at 20 fps, so item 0.10 is needed.
- It is a per-stream CPU-chain rate, not a GPU or box capacity. Across processes the FaceMesh pool reaches 995 fps with 4 processes and 1633 with 8 [M].

**What live WebRTC delivers today.**
- **Never measured cleanly** with TAESD beyond 3 peers [D].
- **Your 10-stream wall run** (19:35–19:39 UTC, pid 3305906), contaminated by probes running at the same time:
  - Per stream "played: 120, duplicated: 57". That is 68–72% fresh, or **136–145 fresh fps against 200 fps of demand** [D wall_api.log:2216-2219; I: 10 × 20 × (120..128)/177].
  - avg_gpu_batch was 36–37 ms, of which the UNet took 26–27 ms and the VAE 8 ms per bs8. max_callback reached 103 ms [D wall_api.log:2127-2172].
  - RSS went 2.18 GB idle → 2.24 GB with 10 sessions connected → 6.45 GB after turn 1 → 7.17 GB in turn 2. The high-water mark was 10.5 GB at startup [D wall_api.log RESOURCE lines 47, 506, 1450, 1893].
  - Threads reached 1037–1212. Available RAM fell to 0.8 GB and the process died [D].
- **That run was not the required recipe.**
  - The live path has no chin code at all [C: no chin symbols in api_server.py, hls_gpu_scheduler.py or api_avatar.py].
  - It encoded H.264 with libx264 medium, even though the log said "encoder set to h264_nvenc" [D wall_api.log:4,882]. `run_local_api.sh` does not set `WEBRTC_VP8_ENCODER=native` [C experiments/chinese_bob_webrtc_20260927/run_local_api.sh; api_server.py:4524-4545].
- **Live efficiency today:**
  - observed 0.53–0.56 of the 258.7 fps GPU path (contaminated) [I];
  - structural single-process ceiling 0.68–0.78 [I: 174–203/258.7].

---

## 3. Budget: today vs target

All figures are GPU ms per frame, with 100% of streams speaking. Where three values are given they are low / expected / high ms. The high case includes half the clock penalty, and that term also stands in for thermal soak, since both are the same sustained-clock mechanism and are not stacked.

| Resource | Today (live, standard compose, no chin) | After P1 (serving fixes + UNet graph, bs8) | After P2 (TRT TAESD full height + ONNX stagewise bs16 + graph) | After P3/P4 (chin live; split if needed) | P6 lossy (mixed-INT8 UNet + INT8 TAESD) |
|---|---|---|---|---|---|
| UNet | 3.00–3.08 [M: 24.03–24.62 / 8] | 2.93 [M: 23.43 / 8, burst] | **2.32 / 2.40 / 2.67** [I, §3.1]; worst 2.78. Fallbacks: ONNX bs8 2.52 / 2.53 / 2.67 [I]; torch_tensorrt bs16 2.79 [M] | same | 1.84 / 1.91 / 2.12 [I]; worst 2.21 |
| TAESD + post + D2H | 0.85 [M: (30.93−24.03)/8 and (30.26−23.43)/8] | 0.85 | 0.50 / 0.51 / 0.55 [I from M 0.486 FP16 NCHW; u8 full height never built] | same | 0.20 / 0.21 / 0.25 [I: 0.147 × 0.486/0.345; only the crop was measured] |
| Other GPU (H2D + Whisper) | 0.08 | 0.05 / 0.08 / 0.10 | same | same | same |
| **Total → ceiling** | **3.95 → 253 fps** | **3.86 (3.83–3.88) → 259 (258–261)** | **2.99 (2.87–3.32) → 334 (301–349)** | same | **2.19 (2.09–2.47) → 456 (405–479)** |
| Live efficiency | 0.53–0.56 observed; ≤0.78 structural | 0.97 / 0.93 / 0.88 [I; see note below] | same | same in-process [I]; could fall with in-process chin (GIL); P4 may restore it but that is unmeasured | same |
| Delivered (high / exp / low) | 136–145 observed | 253 / 241 / 227 | 338 / 311 / 265 (standard compose) | **338 / 311 / 265** if efficiency holds | 465 / 424 / 357 |
| CPU (logical, of 32 SMT; about 20 core-equivalents) | 5.4–8.9 at 10 streams [D]; x264 medium alone 7.85 at 300 fps [M] | 3.3 at 300 fps (standard path, VP8 on 1 thread) [M A_std15] + 0–1 feeder [M gil / I] | same | 10.6 at 300 fps [M C] + 0.5–3 [I] = **11–14** (55–70% of cores [I]); 400 fps: **13–16** (65–80%) | same |
| Host RAM | RSS 7.1 GB at 10 sessions; startup high-water 10.5 GB; 0.43 GB per session after turn 1 [D; I: (6450−2184)/10], still rising turn to turn (+0.7 GB) [D] | + idle YUV cache 0.44 GB per 638-frame avatar [I: 638 × 0.69 MB]; run-ahead cap 100 frames = 69 MB per stream vs 275 MB at 400 [I] | builds: per-block peak unmeasured (estimated 3–5 GB [I], measured on one block first in item 2.2); torch_tensorrt bs16 build 11.0 GB [M] | **in-process, 15 sessions, 1 avatar: 9.0 GB bottom-up** [I, see below] **vs 11–14.5 GB top-down** [D/I]; budget 14 GB until item 0.9 measures it. **+1.4–1.8 GB per extra avatar** [I]. P4 adds process bases (measure in 4.x) | same |
| VRAM (12 GB) | 5.0–6.2 GB [M trt.json 4994 MiB; D RESOURCE 5.96–6.22 GB] | same | −1.7 GB if the eager UNet is freed (item 0.6) [I: 849.9M × 2 B]; +0.3 GB bs16 activations [I from M 151 MB at bs8]; 11 stagewise contexts must share device memory (measure in 2.2); a build needs +1.9 GB [M] | FaceMesh uses no VRAM [M]; an NVENC hybrid would add up to 2.47 GB [M] | −0.4 GB for v1 INT8 weights [I: 821 MB of FP16 weights in the INT8 blocks ÷ 2] |
| Disk (1.3 GB free [M-now]) | – | 0 | FP16 stagewise engines about 1.62 GB [I: sum of per-block ONNX sizes, M]; TAESD engine 3 MB [M]; needs D1, or in-RAM builds (D15) | per-round video ≤150 MB each [M-now: past rounds 86–150 MB]; calibration corpus 0.13 GB [I] | INT8 engines about 1.21 GB for v1 [I: 821/2 + 797 MB] |

**Where the 0.97 / 0.93 / 0.88 efficiency comes from** [I]. The 0.97 is the offline pipelined harness (250.8 / 258.7 [D/M]). The 0.93 and 0.88 are assumptions for a fixed live path. The 3090's live 0.80–0.85 [D] was measured with the serial syncs and blocking handoff that P1 removes. If P1 reaches only that level, delivered at the P2 ceiling is 284 fps at 0.85 [I: 334 × 0.85].

**Bottom-up RAM for 15 sessions, one avatar, in-process** [I]. 9.0 GB total:

| Component | GB | Source |
|---|---|---|
| Base | 2.2 | [D] |
| Per-session non-queue | 15 × 0.15 = 2.25 | [I: 0.43 measured − 0.275 for a full 400-frame queue] |
| Run-ahead as I420, capped at 100 frames | 15 × 0.069 = 1.04 | [I] |
| Trimmed chin cache | 0.70 | [M/I] |
| Idle cache | 0.44 | [I] |
| FaceMesh pool | 1.67 | [I: 6 × 0.2 + 15 × 0.031 (M)] |
| Turn-to-turn growth | 0.7 | [D 6450 → 7165] |

**Top-down RAM** [D/I]. The 3090 history shows 11 GB at 8 streams [D]. The serving review estimates 14.5 GB [I: using 0.43 GB per session with no queue credit, plus tracker frames].

**Distinct avatars that fit** [I]. After D2, about 2–6 distinct avatars fit on the box:
- Available RAM: about 12.0 + 7.9 + 0.65 = 20.5 GB, minus the 4 GB guard.
- The 15-session server takes 9–14.5 GB of that.
- Each extra avatar takes 1.4–1.8 GB: chin cache + idle cache + 240–480 frames × 1.376 MB.
- **15 distinct avatars do not fit.**

### 3.1 How the UNet estimates were built, and what must not be stacked

**ONNX stagewise path at bs16** (the shipping layout of item 2.3b).
- Start: 20.575 ms per bs8, the sum of 11 per-block engines [M], so 2.572 ms per frame.
- Multiply by, for low / expected / high:
  - **bs16 factor:** 0.92 / 0.95 / 1.0.
    - 0.92 is the torch_tensorrt whole-UNet ratio 45.34/49.29 [M].
    - 0.95 is about the ONNX micro-block mean of 0.942 [I], rounded up because torch_tensorrt micro-blocks (mean 0.931) under-predicted their own whole-UNet ratio (0.920) [M].
    - 1.0 means no gain.
  - **CUDA graph:** 0.98 / 0.984 / 0.995 [M: 44.63/45.34 at bs16; raw TRT at bs8 23.62/24.04 = 0.9825].
  - **Sum-of-blocks:** 1.0. The stagewise backend keeps the block boundaries (§1, correction 3).
  - **Clock:** 1.0 / 1.0 / 1.044, half of 2790/2565 [I].
- **Result: 2.32 / 2.40 / 2.67 ms** [I].
- The worst stacked case (full clock penalty and no bs16 gain) is 2.78 ms, equal to the fallback. That is K2.

**ONNX stagewise at bs8** (item 2.3a): 2.572 × 0.98 / 0.9825 / (0.995 × 1.044) = **2.52 / 2.53 / 2.67 ms** [I].

**Mixed-INT8 recipe v1.** INT8 in down0–3, mid, up2 and up3; FP16 in up0, up1, head and tail.
- Per bs8: 16.30 ms [I from M per-block: the INT8 sum of 14.36, with the INT8 up0/up1/tail times swapped for their FP16 times].
- Per frame: 2.04 ms, times the same factors → **1.84 / 1.91 / 2.12 ms**; worst case 2.21 [I].
- A lossier recipe that also puts up0 in INT8 (7.2% block error) would be 15.82 ms per bs8 [I: 14.36 − 2.23 + 3.69], or 1.78 ms per frame optimistic. That is not v1.

**TAESD, full height, uint8 output: 0.50 / 0.51 / 0.55 ms** [I].
- 0.486 ms standalone with FP16 NCHW output [M].
- The crop's in-pipeline overhead was 0.346 → 0.354 ms, about +2% [M p5/p6].
- The high end allows for the uint8 variant never having been built.

**Other: 0.05 / 0.08 / 0.10 ms** [D/M]. The low end assumes H2D overlaps on a copy stream, which no item does yet.

**Overlaps that must not be stacked:**
- **CUDA graph gain at bs16** is −1.6% (45.34 → 44.63 ms [M]), not the bs8 figure.
- **Idle-gap fixes** all close the same idle gap: turning syncs off, double buffering, the non-blocking handoff and the idle cache. Together they are capped at an efficiency of about 0.97 [D offline harness].
- **Two execution contexts** (−5% at bs8, raw TRT [M trt_raw 45.66 vs 48.06 ms]) and a second stream fill the same idle SMs that bs16 fills. Expect 0–3% at bs16 [I]. Not counted.
- **Clock penalty and thermal soak** are one mechanism. Only the clock term is counted.
- **Demand reductions** (skipping discarded GPU work, speaking duty cycle) reduce the frames needed, not the cost per frame. They are kept out of the waterfall.

---

## 4. The plan

**Conventions**
- **Lever classes:**
  - **E0, bit-exact:** the SHA-256 of every pre-encoder frame is identical to the current baseline.
  - **E1, FP16 runtime change:** within 1–3 LSB; passes the numeric gates and G-TRACK.
  - **L, lossy:** attempted, then gated on video.
- **Flags:**
  - Every new flag goes into one overlay file, `.runtime/musetalk_300fps.env`. One lever is one line, so rolling back is deleting or flipping that line.
  - The overlay is sourced by a **new candidate launcher**, `experiments/throughput300_candidate/run_candidate_api.sh`, with its own port (for example 8300), its own log and a `oom_score_adj` of 1000. It sources the overlay after the generated `.runtime/musetalk_trt_local_sm89.env`, which says "Do not edit by hand" [C].
  - Your `experiments/chinese_bob_webrtc_20260927/run_local_api.sh` is exec'd by `run_wall_api.sh`, which the Codex session also uses [C run_wall_api.sh:7]. It stays untouched.
- **Baselines are versioned by backend flag set.**
  - After each accepted E1 lever (2.1 TRT TAESD, 2.3 stagewise UNet), recapture the golden SHAs (item 0.3) and the offline chin reference with the new backend.
  - P3's E0 gates run against both the old and new baselines until P2 lands.
  - The offline harness hardwires compiled TAESD today [C h3_avatar_workflow/backend.py:41] and must accept the new backends (item 2.6).
- **Work on a branch or worktree.**
  - `api_server.py`, `templates/webrtc_player.py` and `templates/webrtc_wall.py` have uncommitted edits.
  - `docs/WEBRTC_WALL_AUDIO.md` and `scripts/test_webrtc_wall_audio_browser.py` are new and untracked. Another session (Codex) is working now [M-now git status].
  - Coordinate before touching them (D13).
- **Every round ends with the labelled comparison video** described in §6.5.
- **Box hygiene rule:** any process that initializes CUDA, loads the UNet or TAESD, builds an engine, runs a golden-SHA replay or drives multi-stream traffic goes through `box_guard.sh` and the lease (item 0.1).
  - Items below are tagged [GPU], [BUILD] or [LOAD] accordingly.
  - At today's 12.0 GB MemAvailable, **D2 blocks every server start, the item 2.2 bench and every load test**.

### Phase 0 — prerequisites and a truthful baseline (no output change; about 6–7 days)

**0.1 Box guard, GPU lease and disk ledger.**
- **Change:** a new stdlib-only `MuseTalk/scripts/box_guard.sh`. It generalises `throughput300/cpu_probe/status.sh` and is modelled on SoulX `benchmarks/tiny_vae_20260926/decoder_bakeoff/wait_quiet.sh`. There is no `wait_lease.sh` [M-now].
- **Preflight checks:**
  - No foreign GPU compute apps; utilization ≤5% and memory used ≤600 MiB for 30 s.
  - Load average ≤2 for 60 s.
  - `pgrep` for co-tenants: api_server on another port, SoulX `dev_server` / `drive_wall.py`, the Codex `run_wall_api.sh`, probe scripts.
- **RAM thresholds (MemAvailable):**
  - ≥14 GB before a server start (covers the 10.5 GB transient [D] plus margin) until item 0.6 lands.
  - ≥16 GB before a torch_tensorrt build (11.0 GB peak [M]).
  - ≥ the measured per-block peak + 4 GB before a stagewise build.
  - ≥ the item 0.9 server budget + 4 GB before a load test.
- **Disk ledger:**
  - Before any write: at least the artifact size + 0.5 GB free.
  - Budget per phase for engines, the calibration corpus (0.13 GB), per-round videos (≤150 MB) and profiler traces (nsys runs are capture-range limited).
  - Write to a temp path, then rename atomically.
- **Runtime protection:**
  - `oom_score_adj=1000` on build, probe, test-client and candidate-server processes.
  - A watchdog kills the child if MemAvailable drops below 3 GB (below 4 GB during load tests).
  - Snapshot `/sys/fs/cgroup/memory.events` (oom_kill), nvidia-smi and `du /dev/shm` before and after every run. Any rise in oom_kill invalidates the run.
- **Lease:** a `flock` on `/workspace/.gpu_lease`. The SoulX and Codex sessions are asked to honour it (D2).
- **Effort / deps:** 0.5 d / D2.

**0.2 CUDA-event timing and capacity telemetry.**
- **Change:**
  - `HLS_GPU_EVENT_TIMING=1` replaces `_sync_gpu_for_stage_timing` (hls_gpu_scheduler.py:2398) as the timing source.
  - Export: GPU busy fraction, idle gap between batches, time blocked in the callback, batch fill (actual/padded), jobs per batch, feeder-thread CPU per batch.
  - Add asyncio loop-lag p50/p99 to the RESOURCE lines.
  - A `/stats/capacity` endpoint (`MUSETALK_CAPACITY_TELEMETRY=1`) that extends `get_stats` (:385).
- **Rollback:** set the flags to 0.
- **Gate:** event timings agree with sync timings within 2% on a single stream.
- **Effort:** 0.5–1 d.

**0.3 Golden capture and replay harness [GPU].**
- **Change:**
  - `MUSETALK_CAPTURE_DIR` (frame-capped) records the SHA-256 of every decoded face and every composed BGR frame before YUV conversion.
  - Hashes are computed in memory. Any video dump is encoded straight to lossless x264 (`-qp 0`) or FFV1, never raw BGR: 2 identities × 20 s × 20 fps × 1.376 MB = 1.1 GB [I], which is nearly all the free disk.
  - A new `scripts/replay_scheduler_exactness.py` drives `HLSGpuScheduler` with fixed avatars and WAVs.
  - Baselines are stored under `baselines/<flagset-hash>/`.
- **Gate:** the capture reproduces across 2 runs.
- **Keep `/tmp/torchinductor_root`** (835 MB [M-now]). Deleting it changes compiled-TAESD numerics [D].
- **Effort:** 1–1.5 d.

**0.4 Load harness v2** (`load_test_webrtc.py`).
- **Defaults:** `--musetalk-fps 20 --playback-fps 20 --batch-size 8`. Today's defaults are 15 / 30 / 2 [C :1057-1064]. The committed wall template defaults to 15 / 30 [C templates/webrtc_wall.py:87-88]; pin it to 20 / 20, coordinating with the Codex edits (D13).
- **Inputs:** `--audio-dir` with distinct WAVs; `--turns N`; a `--chain` mode that pre-queues each stream's next turn, so there are no idle gaps.
- **Modes:** sustained, burst, duty, soak, churn, barge-in.
- **Counters, fixed.** Per track, poll `/webrtc/sessions/stats` (api_server.py:4238) once per second:
  - fresh = Δ`frames_played`, which counts only freshly popped frames [C webrtc_tracks.py:2216];
  - held = Δ`frames_duplicated` [C :2255];
  - stall time = Δ`strict_video_stall_seconds`, plus server send gaps.
  - `start_live()` zeroes every counter at each turn [C webrtc_tracks.py:1622-1629]. So add server-side monotonic lifetime counters behind `WEBRTC_LIFETIME_COUNTERS=1`, or stitch the resets in the client.
  - Assert output_fps == 20 in every run.
- **Also:**
  - Record server-side send timestamps.
  - Shard the client at ≤5 peers per process, pinned to physical cores (§6.2), with its own loop-lag metric.
  - Record only 1–2 observer peers.
  - Log a per-second count of concurrent speakers.
- **Lift the group cap:** the cap is a hardcoded `count > 12` check at api_server.py:3539 and :3982. Make it an env var, `WEBRTC_GROUP_MAX_COUNT` (default 12).
- **Why:** the current client records receive intervals only (:444-466 [C]). Held frames still arrive on cadence, so it cannot see them.
- **Gate:** a synthetic 10% hold is detected, and a turn boundary does not corrupt the deltas.
- **Effort:** 2 d.

**0.5 Audio corpus and no local TTS.**
- **Corpus:** at least 30 distinct pre-synthesized WAVs of 3–24 s with natural pauses and 2+ voices, plus concatenated 60–300 s versions for S1. Made off-box or in a quiet window.
- **`MUSETALK_DISABLE_LOCAL_TTS=1`** makes `/webrtc/tts/kokoro` (api_server.py:3930) return 503. Local Kokoro would take 30–85% of the CPU [M audio_tts].
- **Warm librosa** in `_warm_runtime_paths` (avatar_manager_parallel.py:423). A cold first call costs 1.2 s [M].
- **Effort:** 0.5 d.

**0.6 Host RAM and VRAM hygiene in the model process (E0).**
- **Step 1, read-only:** trace in the code where the 10.5 GB startup RSS spike (settling to 2.18 GB [D]) comes from.
  - Is the FP32 `unet.pth` (3.4 GB [M-now]) held while TRT is active?
  - Is the 2.2 GB .ts deserialized with extra copies?
- **Step 2 [GPU], after D2:** a load-only run under `/usr/bin/time -v` with the guard and watchdog.
- **Fixes:**
  - `MUSETALK_SKIP_TORCH_UNET_WITH_TRT=1`: mmap or skip the load.
  - `MUSETALK_FREE_EAGER_UNET=1`: drop `self.eager_unet_model` (avatar_manager_parallel.py:170) after `_activate_unet_backend` (:334). Saves about 1.7 GB of VRAM [I]. Keep it at 0 when calibration capture is needed.
- **Gate:** golden SHA; start, turn and teardown; startup RSS no more than 4 GB above steady state.
- **Effort:** 1 d.

**0.7 Multi-avatar UNet capture corpus [GPU].**
- **Settings:** `MUSETALK_UNET_CALIBRATION_CAPTURE=1`, `MUSETALK_UNET_CALIBRATION_DIR`, `MUSETALK_UNET_CALIBRATION_MAX_BATCHES=16` (read at hls_gpu_scheduler.py:246-252; capture function `_capture_unet_calibration_batch`, :1528).
- **Coverage:** 8 identities × 2 voices × 16 bs8 batches ≈ 125 MB [I]. Hold out 3 identities.
- **Disk:** use only avatars and research caches already on disk. Do not create new production avatars for the corpus (about 290–310 MB per pose [M-now du]).
- **Needed by:** G-UNET for bs16 (2.3b) and INT8 (6.2).
- **Effort / deps:** 0.5 d / 0.1, D1 ledger.

**0.8 Layered clean baseline, and the first sustained GPU measurement [LOAD].**
- **What:** today's code (TAESD, standard compose, current env) on a quiet box, after D2.
- **L1 first:** `HLS_NULL_SINK=1`, unpaced, for ≥5 min at the power cap, logging clocks and power at 1 Hz. This is the first ≥60 s measurement of the shipping bs8 GPU path and settles R5 for today's engine.
- **Then:** 1/3/5/8/10/12/14 streams; burst plus 5 minutes sustained; 3 repeats.
  - L2: + compose.
  - L3: + handoff to a fake consumer.
  - L4: full aiortc loopback.
  - L5: off-box client, if one is available.
- **Output:** per-layer efficiency measured at saturation (demand above the ceiling), plus the baseline labelled video.
- **Redirect rule:** if L1 efficiency is below 0.90, the loss is inside the scheduler. Do items 1.1–1.5 first and profile with py-spy.
- **Effort / deps:** 1 d / D2, D3.

**0.9 RAM attribution and soak (new) [LOAD].**
- **Change:** smaps plus tracemalloc snapshots at 1 / 5 / 10 / 15 sessions, with 1 and 3 avatars, over ≥5 turns. Attribute RSS to the base, per session, per avatar, per turn (including the lazy pose loads at turn start, +1.0 GB at the first turn [D]), queues and allocator arenas.
- **Output:** the server RAM budget that the guard, Phase 3/4 exits and admission (7.1) use.
- **Effort:** 1 d.

**0.10 20 fps offline reference and re-acceptance (new) [GPU].**
- **Why:** the accepted recipe was generated and reviewed at 24 fps:
  - audio features at fps=24 [C prepare_stage.py:42,49];
  - encode at `-r 24` [C run.py:18-19];
  - 240 frames per clip [C run.py:92].
  - Live runs at 20 fps. The 3-tap jaw filter then spans 150 ms instead of 125 ms, and the 24 fps px baselines do not apply.
- **Change:** a `--fps 20` mode in `experiments/chin_fps_validation_20260927/run.py`: audio features at fps=20, and the same source-frame mapping the live pose router uses.
- **Output:** render the certification identities, and ask you to re-accept chin100 + refined seam at 20 fps (D18). That render becomes Column A of every video, and its chin error becomes the gate baseline.
- **Effort:** 1 d, plus your review.

**Phase 0 exit:**
- The guard is in place, and D2 is resolved.
- The harness sees held frames across turn boundaries.
- Per-layer saturated efficiency is measured, as is the ≥5 min L1 run.
- The RAM budget is attributed.
- The golden SHA reproduces.
- The 20 fps reference is accepted.
- The baseline video exists.

### Phase 1 — serving path up to the GPU ceiling, in-process (E0 unless marked; about 8–10 days)

**1.1 Turn off the timing syncs (E0).**
- **Change:** `HLS_GPU_STAGE_SYNC_TIMING=0` (default on, hls_gpu_scheduler.py:230) and `MUSETALK_VAE_DECODE_TIMING_SYNC=0` (musetalk/models/vae.py:30; the sync is at :156-157).
- **Rollback:** set both to 1.
- **Gate:** golden SHA.
- **Effort:** 0.25 d.

**1.2 UNet CUDA graph (E0 only once its gate passes).**
- **Change:** `MUSETALK_TRT_UNET_CUDAGRAPHS=manual|runtime|0` in `TrtUnetBackend` (trt_runtime.py:1425; forward at :1543).
  - **Default `manual`:** a `torch.cuda.CUDAGraph` capture of the call. Measured 23.65 ms per bs8 with max_abs 0 [M trt.json].
  - **`runtime`:** `torch_tensorrt.runtime.set_cudagraphs_mode(True)`. Measured 23.79 ms in the short run and 23.43 in p6, with no accuracy recorded [M].
  - Both reuse a static output buffer, so clone the output (131 KB at bs16 [I]) wherever it outlives the next call, including in calibration capture.
- **Gain:** about −2.5% of UNet time; host enqueue 4.5 ms → under 0.3 ms [M].
- **Gate:** max_abs 0 on 128 frames for the mode shipped; an overwrite test with two batches; golden SHA at depth 1 and 2.
- **Effort:** 0.5 d.

**1.3 Blocking wait (E0).**
- **Change:** `HLS_GPU_BLOCKING_WAIT=1` uses `torch.cuda.Event(blocking=True)`, or puts the device in blocking-sync mode.
- **Gain:** up to about 1 core, but only once graph mode (1.2) and depth-2 pipelining (1.4) bound the work in flight. The measured 23.5 ms of CPU per call is launch back-pressure [M gil.json; C probe_gil.py].
- **Gate:** feeder CPU per batch (item 0.2) falls; GPU busy unchanged; SHA.
- **Effort:** 0.25 d / deps 1.2, 1.4.

**1.4 Double-buffered async GPU loop (E0).**
- **Change:**
  - Split `_run_generation_batch` (:1297) into submit(slot) and collect(slot), with per-slot pinned staging (today `_get_staging_buffers` at :2486 keeps one set per shape) and a pinned output ring.
  - Add `decode_latents_async` next to `decode_latents` (vae.py:147).
  - `_run_loop` (:998) submits N+1 before collecting N.
  - UNet and TAESD launches stay on the one scheduler thread; compiled TAESD graph trees are thread-affine [M].
  - Whisper already encodes in the job-prep path [C hls_gpu_scheduler.py:566-571]. Move it onto its own stream (`MUSETALK_WHISPER_STREAM=1`).
  - Capture every CUDA graph at warmup before the prep threads start, or use `capture_error_mode='thread_local'`.
- **Flag / rollback:** `HLS_GPU_PIPELINE_DEPTH=2` / `=1`.
- **Gain:** offline, serial 204.6–215 fps went to pipelined 250.8 fps (+17–23%) with SHA-identical output [D].
- **Gate:** golden SHA at depth 1 vs 2; per-stream frame order monotonic.
- **Effort / deps:** 2 d / 1.1, 1.2.

**1.5 Non-blocking handoff (E0).**
- **Change:**
  - In `frame_batch_callback` (api_server.py:5495-5535), replace `run_coroutine_threadsafe(...).result()` (:5517/:5527) with `loop.call_soon_threadsafe` into a bounded per-track deque, plus a depth counter read at selection time.
  - PyAV BGR → yuv420p conversion moves into the compose workers; `push_bgr_frames_batch` (webrtc_tracks.py:1946-1973) accepts pre-converted frames.
  - Never block the scheduler on a full strict-FIFO queue (webrtc_tracks.py:62; api_server.py:5375-5407).
  - Skip the crossfade `source_frame.copy()` (hls_gpu_scheduler.py:2141) when no crossfade is active.
- **Flag / rollback:** `WEBRTC_NONBLOCKING_HANDOFF=1` / `=0`.
- **Gain:** removes 8–10 ms of blocking per batch [M/D].
- **Gate:** SHA of I420 frames per stream; A/V offset unchanged; sequence numbers monotonic.
- **Effort / deps:** 1–1.5 d / 1.4.

**1.6 cv2 I420 conversion (E1, separate flag).**
- **Change:** `WEBRTC_YUV_CONVERTER=cv2`, with `cv2.setNumThreads(1)` in the workers.
- **Gain:** 1.51–1.72 → 0.14–0.21 ms per frame. It moved 20 single-process streams from 19 late frames to 0 [M].
- **Gate:** ≤2 LSB max and ≤0.5 LSB mean (measured 0.44 [M]); video.
- **Rollback:** `=pyav`.
- **Effort / deps:** 0.25 d / D7.

**1.7 Shared pre-decoded idle and pose YUV cache (E0).**
- **Change:** `IdleVideoStreamTrack.read_frame` (webrtc_tracks.py:730/748), `_advance_idle_frame` (:1479) and the motion entry/return builders (webrtc_motion_playback.py:196/198, 456) index a per-clip yuv420p cache.
  - In Phase 4 the cache lives in read-only /dev/shm, shared by all workers.
- **Flag / rollback:** `WEBRTC_IDLE_FRAME_CACHE=1` / `=0`.
- **Gain:** 15 idle sessions: loop 98% → 26% busy; 182.7 → 299.7 fps aggregate; 399.7 fps at 20 streams [M]. Push p50 75 → 8.5 ms [M].
- **Cost:** 0.44 GB per 638-frame avatar [I].
- **Gate:** idle frame SHAs; visual check of idle → live → idle.
- **Effort:** 1 d.

**1.8 Make the encoder match the config.**
- **VP8:** `WEBRTC_VP8_ENCODER=native` (webrtc_native_vp8.py:263), plus `WEBRTC_NATIVE_VP8_THREADS=1`, which sets `cfg.g_threads` (:123).
- **H.264-only clients:**
  - Replace the dead `enable_h264_nvenc` (api_server.py:273-325) with an override of `H264Encoder._encode_frame`: x264 ultrafast or veryfast on 1 thread.
  - Fix the misleading log line.
- **NVENC hybrid: not recommended.** It costs up to 2.47 GB of VRAM at 12 sessions and 32% GPU utilization at saturation [M nvenc_util_*]. If you still want it (D5), put it behind a `WEBRTC_NVENC_MAX_SESSIONS` semaphore.
- **Rollback:** `=pyav`, threads unset, `WEBRTC_H264_IMPL=aiortc`.
- **Gain:** VP8 at 15 streams 2.53 → 1.85 cores, at 20 streams 4.00 → 2.79 [M]; x264 7.85 → 0.69 cores at 300 fps [M].
- **Gate (G-ENC):** a receiver recording, and PSNR of decoded vs pre-encode frames within 0.2 dB of baseline [I, proposed].
- **Effort / deps:** 1 d / D5.

**1.9 Deadline-aware scheduling, run-ahead cap, prebuffer-sized startup slices (E0).**
- **Change:** in `_select_jobs_locked` (:1027):
  - keep the startup-fairness rounds 1–2 (:1038-1063);
  - replace rounds 3–5 with earliest-deadline-first on slack (queued playback seconds);
  - skip jobs whose slack exceeds `HLS_SCHEDULER_MAX_RUNAHEAD_S` (default 5) or whose queue is full;
  - make the startup slice equal the prebuffer size, `HLS_SCHEDULER_STARTUP_SLICE_SIZE=10`, packed across jobs into full batches.
- **Flag / rollback:** `HLS_SCHEDULER_POLICY=edf` / `roundrobin`.
- **Gain:** smooth streams at capacity; queue RAM 69 MB per stream instead of 275 MB [I]; a burst start of 15 streams needs 150 frames before the last first frame, instead of 240 [I].
- **Gate:** per-stream SHA and order; a burst at N = capacity + 2 shows fewer held frames than round-robin; first-live latency per §6.4.
- **Effort / deps:** 1–1.5 d / 1.5, D6.

**1.10 Skip the GPU for frames whose output is discarded (E0).**
- **Change:** exact_silence jobs and `WEBRTC_RAW_IDLE_POSE` frames (hls_gpu_scheduler.py:1763-1782) are routed to raw compose; `compose_frame` (api_avatar.py:1321) gets a `res_frame=None` path.
- **Flag:** `HLS_SKIP_GPU_FOR_RAW=1`.
- **Gain:** demand only.
- **Gate:** SHA on an exact-silence turn.
- **Effort / deps:** 0.5 d / D12.

**1.11 Thread hygiene (E0).**
- **Change:**
  - Idle decoder threads 16 → 1–2 (0 once item 1.7 is in).
  - `cv2.setNumThreads(1)` in the workers; torch intra-op threads ≤4.
  - Per-turn ffmpeg in its own executor.
  - Do not re-enable aggressive global caps (GPU utilization fell to 37% [D CPU_OPTIMIZATION_ANALYSIS.md]).
- **Flag:** `MUSETALK_THREAD_CAPS=1`.
- **Gain:** 1037–1212 threads [D] → about 600–900 [I: the torch-free loopback harness alone ran 518 threads at 15 streams and 674 at 20 (M serving)]. Measure.
- **Gate:** GPU busy does not drop; SHA.
- **Effort:** 0.25 d.

**1.12 Allocator hygiene (E0, new).**
- **Change:** `MALLOC_ARENA_MAX=4` plus a fixed `MALLOC_MMAP_THRESHOLD_` in the candidate launcher, as one overlay line (alternative: jemalloc via `LD_PRELOAD`).
- **Why:** RSS rose about 0.07 GB per session per turn in the wall run and never came back down [D: (7165−6450)/10].
- **Gate:** the item 0.9 soak shows a flat RSS slope; SHA.
- **Effort:** 0.25 d.

**Phase 1 exit gate** (standard compose, chin not yet live), under **saturation**:
- N ≥14 all speaking (280 fps demand against a 259 ceiling [I]), or an unpaced L1/L2 run.
- Fresh ≥99% for streams within capacity; max interval ≤100 ms; 0 stall seconds.
- Saturated efficiency = generated fps ÷ GPU-event ceiling. Golden SHA matches.
- Labelled video: P0 vs P1.
- **Kill / redirect (K1):**
  - <0.85: stop GPU work, profile, and pull Phase 4 forward.
  - 0.85–0.90: continue P2, but Phase 4 becomes a prerequisite for 300.
  - ≥0.90: continue in-process.

### Phase 2 — FP16 GPU levers (E1; needs D1 or in-RAM builds, plus D2 and a D3 window; about 6–8 days)

**2.1 TRT TAESD at full height, uint8 BGR NHWC output (E1) [BUILD].**
- **Change:** a `TaesdTrtBackend` in scripts/vae_fast_decoder.py, next to `TaesdVaeDecodeBackend` (:61, `_raw_decode` :128).
  - Runs in bs8 sub-batches (0.486 ms at bs8 vs 0.502 at bs16 [M]).
  - The fused uint8 output drops the post kernel.
  - It removes the thread-affinity constraint of the compiled decoder [M].
- **Persist the 3 MB engine** rather than rebuilding at startup. Fingerprint it by plan hash plus a probe-batch output hash checked at load; refuse, or fall back via the flag, on mismatch.
- **Flag / rollback:** `MUSETALK_TAESD_BACKEND=trt` / `=compiled`.
- **Gain:** 0.85 → 0.51 ms (0.50–0.55 [I]); ceiling 259 → 284 fps [I].
- **Risk:** max_abs 0.0085 vs eager, about 2 LSB [M p5]. It feeds a stateful tracker, hence G-TRACK.
- **Gate:** G-TAESD, G-TRACK, chin gates on 3 identities, video.
- **Effort:** 1.5 d.

**2.2 Go/no-go bench of the exact shipping artifact [GPU][BUILD]. Do this before writing backend code.**
- **Setup:** quiet box, server stopped, after D2.
- **Build:**
  - Export and build the 11 FP16 per-block ONNX engines **one after another, in RAM**. Each ONNX is ≤0.49 GB [M].
  - Measure host RSS on the first block with `/usr/bin/time -v` before continuing (estimated 3–5 GB [I]).
  - Build at bs16 and bs8.
  - Measure the build time at optimization level 5. The 184 s total was for bs8 at the default level [M]; the timing-cache benefit on the ONNX network is unknown, since the 401 → 44 s figure was for the torch_tensorrt network [M].
  - Reuse `$S/unet_probe/tt16_timing_cache.bin` (already preserved in the repo evidence folder).
- **Measure:**
  - Chain the engines as the stagewise backend prototype, with one CUDA graph over all enqueues.
  - Time for ≥60 s (ideally 5 min) at the power cap, with clocks and power at 1 Hz.
  - Record VRAM for all 11 contexts, which must share device memory.
  - Record G-UNET against torch on the item 0.7 corpus.
- **No torch_tensorrt arm:** compare against the measured 44.63 ms per bs16 [M]. As a same-conditions control, run the shipping bs8 .ts for 60 s only if the RAM guard allows.
- **Write only JSON.**
- **Go criteria (§1 table):** bs16 ≤2.51 ms go, 2.51–2.64 marginal, >2.64 K2. The bs8 result decides the fallback 2.3a (go if ≤2.60 ms, i.e. at least 7% under 2.79).
- **Effort / deps:** 1 d / 0.1, D2, D3.

**2.3a ONNX stagewise FP16 UNet at bs8 (E1).** 2.3b adds bs16 on top.
- **Change:** a new backend in trt_runtime.py, following `_TensorRtOnnxStage` (:493) and `StagewiseTrtVaeDecodeBackend` (:687):
  - a raw `IExecutionContext` per block, created without device memory and pointed at one shared activation arena;
  - **multi-input and multi-output bindings**, because UNet blocks carry skip tensors and today's class supports only single-IO;
  - `execute_async_v3`;
  - one `torch.cuda.CUDAGraph` over all enqueues. Raw TRT with a graph is bit-exact [M]; replay host time is 0.89 ms [M].
  - The time embedding for t=0 is computed once.
  - The same layout serves INT8 later: each INT8 block is ≤0.97 GB, while a monolithic Q/DQ ONNX would exceed the 2 GB protobuf limit [I].
- **Selection:** `MUSETALK_UNET_BACKEND=trt_stagewise`, `MUSETALK_UNET_STAGEWISE_BATCH=8|16`, `MUSETALK_UNET_STAGEWISE_CACHE_DIR`, in `load_unet_trt_backend` (:1641).
- **Gain:** 2.93 → 2.53 ms at bs8 (2.52–2.67) [I]; ceiling 321.
- **Rollback:** `MUSETALK_UNET_BACKEND=trt`, which keeps today's bs8 .ts.
- **Gate (G-UNET):** `scripts/validate_unet_backend.py --fail-mae 0.01 --fail-max-abs 0.5` on the multi-avatar corpus. Also report rel-L2 vs torch as the FP16 noise floor (the shipping engine is 0.38% [M]). Plus G-TRACK and video.
- **Effort / deps:** 2–3 d / 2.2 go, 0.7, D1 or D15.

**2.3b bs16 on the same backend (E1).**
- **Change:** set `MUSETALK_UNET_STAGEWISE_BATCH=16`, `HLS_SCHEDULER_MAX_BATCH=16` and `HLS_SCHEDULER_FIXED_BATCH_SIZES=16`. The startup slice stays at 10, packed (item 1.9). Load only bs16; an 8 + 16 pair OOMed even on 24 GB [D].
- **Gain:** 2.40 (2.32–2.67) ms [I].
- **Gate:** G-UNET, pairing two bs8 captures into one bs16 input; G-TRACK; video.
- **Build twice and keep the faster engine** (tactic variance is about ±5% [D SoulX]). Persist the timing cache.
- **Effort:** 1 d.

**2.4 Fallback: torch_tensorrt bs16 (E1) [BUILD].**
- **Change:** `scripts/tensorrt_export.py --batch-sizes 16 --require-valid-unet --validate-unet-capture-dir …`, with `MUSETALK_TRT_UNET_PATHS=16:<path>`.
- **Gain:** 2.79 ms [M]; mae 0.00196 [M].
- **Risk:** a 2026-05-29 bs16 artifact failed the gate [D].
- **Cost:** 2.7 GB of disk and ≥16 GB MemAvailable [M 11.0 GB peak].
- **When:** only if both 2.3a and 2.3b fail. With this fallback, 300 is out of reach losslessly (§1).
- **Effort:** 1 d.

**2.5 Optional extras (measure before adopting).**
- **Two raw-TRT execution contexts sharing weights** (`MUSETALK_UNET_CONTEXTS=2`): −5% at bs8 [M trt_raw]. This is distinct from the closed torch_tensorrt two-stream lever, which serializes. Try it only if item 0.2 shows more than 3% idle SMs at bs16.
- **A fused per-slot graph** covering UNet + TAESD + pack + D2H: only if item 0.2 shows more than 2% bubbles.
- **An H2D copy stream** (`HLS_H2D_COPY_STREAM=1`): measure; it would move "Other" toward 0.05.
- **Effort:** 1–2 d.

**2.6 Re-baseline after each E1 acceptance (new).**
- **Change:** recapture the golden SHAs and the 20 fps offline chin reference with the accepted backend, and extend `run.py` / `backend.py` to take `MUSETALK_TAESD_BACKEND=trt` and the stagewise UNet.
- **Effort:** 0.5 d.

**Phase 2 exit (go/no-go):**
- In-server CUDA events under saturation show all-in GPU ms/frame ≤ 3.333 × e_P1, where e_P1 is the measured Phase 1 efficiency (for example ≤3.10 ms at e = 0.93).
- The Phase 1 criteria still pass at a higher fps.
- Labelled video: compiled vs TRT TAESD; .ts bs8 vs stagewise bs8/bs16; frame-aligned offline, plus one live capture.

### Phase 3 — the required recipe live: 100% chin + refined seam (about 13–18 days; the offline items 3.0a, 3.0c and 3.1–3.3 can start during P1)

**3.0a Chin assets for production avatars (new; critical).**
- **Why:** live chin needs per-source-frame inputs that production avatars do not have [C chin.py:56-63, 229-240]:
  - FaceMesh source landmarks `d['p']`;
  - jaw masks built with FaceParsing cheek widths 90, a +10 px lower box margin and `mode='jaw'` [C prepare_stage.py:32,40,60];
  - the refined-mask data, cropboxes and boxes.
  - The production archive carries latents, frame PNGs, masks and coordinates, but no landmarks [D BATCH_THREE_POSE.md:105-115]. chinese_bob has no landmarks anywhere [M-now].
- **Change:** at avatar prepare or load time, run source FaceMesh per physical pose clip, matching `track_stage.py` (mediapipe 0.10.9, `static_image_mode=False`, same sequential order). Build the chin masks, then store and version the assets with the avatar archive.
- **Flag:** `MUSETALK_CHIN_ASSETS=build|load|off`.
- **Gates:**
  - Parity with the research cache for japanese_new and latina_new (zero diff).
  - Your visual acceptance of the chin recipe on each certification identity, since new identities need individual review [D WORKFLOW.md:11-13].
  - The rig's `_fh1` avatars are Sep 25–26 realtime/guided caches [M-now], so replace them with H3 identities unless one is confirmed H3 expressive (D20).
- **Effort:** 2–3 d, plus your review.

**3.0c Combined in-process bench: this decides whether Phase 4 is needed for 300 (new) [LOAD].**
- **Change:** re-run the synthetic combined load (cpu_probe/combined_bench.py) with:
  - a GPU-feeder thread that holds the GIL as the real one does;
  - an asyncio aiortc loop;
  - a 6-process tracker pool with real IPC and several graphs per process (only one graph per process was ever measured);
  - the real refined chin with the item 3.2 numba kernels, not the +1.5 ms stand-in.
- **Decision:** at 15 streams, 0 late frames with p99 work ≤40 ms and feeder stalls <3% keeps P4 conditional. Anything worse makes P4 required for 300.
- **Effort:** 1–1.5 d.

**3.1 Trim the chin caches (E0).**
- **Change:** drop the unused arrays in `SourceMask` (chin.py:125) and `RefinedMask` (:186); dedup the cycle halves; store once in read-only /dev/shm.
- **Gain:** 8.5 → 1.1 MB per source frame [M/I].
- **Flag:** `MUSETALK_CHIN_CACHE_TRIM=1`.
- **Gate:** `verify_baseline.py`, 480 frames zero diff.
- **Effort:** 1 d.

**3.2 numba nogil kernels (E0).**
- **Change:**
  - Blend (blending.py:304): 0.50 → 0.19 ms. `warp_roi` (chin.py:92): 0.71–0.91 → 0.19 ms. Both bit-exact over 720 frames and 3 identities [M].
  - `RefinedMask.current` port (:204): about 0.70 → 0.25 ms [I], not yet proven.
  - Keep NumPy 1.23 dtype rules.
- **Flag:** `MUSETALK_CHIN_NUMBA=1`.
- **Gate:** `verify_baseline.py` and pixel_checks.
- **Effort:** 2 d.

**3.3 FaceMesh tracker pool.**
- **Change:** 4–6 processes in the SoulX venv (mediapipe 0.10.9), with one stateful graph per speaking stream and IPC batched per chunk.
  - **The tracker input is pasted in the TRT venv:** OpenCV 4.9 / NumPy 1.23.5, not the SoulX venv's OpenCV 5.0 / NumPy 2.2.6 [M-now].
  - The pasted 896×512 frame goes through a /dev/shm slot, as in run.py [C :26-47]: 413 MB/s at 300 fps [I: 1.376 MB × 300].
  - Record a checksum of the SoulX venv's mediapipe, numpy and opencv at the start of each run, and fail on change. That venv belongs to another project.
- **Flag:** `MUSETALK_CHIN_TRACKER_PROCS=6`.
- **Gain:** capacity 995 fps with 4 processes, 1633 with 8 [M].
- **Gate:** landmarks identical to the replay reference (item 3.4).
- **Effort:** 1.5 d.

**3.4 Ordered per-stream post actors and live chin.**
- **Change:**
  - Call `chin.corrected_refined` (chin.py:234) from the compose path (api_avatar.py:1321), through one ordered actor per stream.
  - The existing per-job re-sequencing already restores output order [C hls_gpu_scheduler.py:1868-1897]. Ordered actors are needed for tracker state and the one-frame lookahead; keep the re-sequencing.
  - **Tracker reset semantics:** reset the FaceMesh graph per turn and on barge-in, as run.py does per clip [C run.py:37,102]. The first detection frame costs about 14 ms [M]. Record the rule in D11.
  - The 3-tap filter holds one frame across chunks and edge-pads at turn edges.
- **Flag:** `MUSETALK_CHIN_ALIGN=refined100` / `off`.
- **Gate:**
  - Pre-encode SHAs equal a **replay reference**: the offline harness driven with the live per-turn source-index sequence (forward/reverse cycles, pose switches), the live reset points, the same faces and the turn-edge padding. Not run.py's 0..239 clip.
  - `verify_baseline.py`; pixel_checks; sequential == pipelined.
  - Chin-target error within ±0.05 px of the 20 fps reference (0.10); the 24 fps values are 1.07 / 1.01 px [D].
  - Video.
- **Effort / deps:** 4–5 d / 3.0a, 3.1–3.3, 1.5, D11.

**3.5 Fallback policies (D11).**
- **Cases:** FaceMesh loses the face; pose crossfades; motion atlas; raw layers; barge-in.
- **Flag:** `MUSETALK_CHIN_FALLBACK=standard|hold`.
- **Gate:** visual review of the companion-midclip-stop case.
- **Effort:** 1–1.5 d.

**3.6 Exact warp skip where the chin delta is zero (E0).**
- **Flag:** `MUSETALK_CHIN_WARP_SKIP=1`.
- **Gain:** 0–21% of frames skip the warp [M].
- **Effort:** 0.5 d.

**3.7 Chin telemetry and sampled invariants (new).**
- **Change:**
  - Per-stream counters: `chin_applied`, `chin_fallback{reason}`, `warp_skipped`.
  - A sampled runtime check (1 frame in N): protected lip pixels unchanged and minimum map Jacobian >0.25. `corrected_refined` does not assert these; only `aligned()` does [C chin.py:84-86, 234-240]. Violations are counted and alerted.
- **Flag:** `MUSETALK_CHIN_INVARIANT_SAMPLE=30`.
- **Effort:** 0.5 d.

**Phase 3 exit:**
- 15 all-speaking streams with chin on, at the P2 GPU cost.
- Frames SHA-equal to the replay reference.
- Chin applied on 100% of speaking frames, excluding the categories approved in D11; 0 invariant violations.
- Loop lag p99 ≤20 ms; GPU idle attributable to GIL waits <3%.
- RSS within the item 0.9 budget, with MemAvailable ≥4 GB.
- Labelled video.
- If these limits fail at 15 streams, Phase 4 is required for 300.

### Phase 4 — process split (required for 20 streams / 400; probable for 15 streams with chin; about 12–17 days)

**Why probable at 15 streams:**
- The only one-process 15-stream chin evidence is threads-only (C_chin15), with per-frame work inflated 4.5× [M].
- Switch-interval tuning made it worse: 881–1055 late frames [M H].
- Crossfade and `frame_batch_callback` run on the scheduler thread today [C hls_gpu_scheduler.py:1905-1921].
- Item 3.0c decides.

**Target architecture**
```
 browser ─HTTP─► front (control only) ─► owner media worker (unix socket)
┌── gen: GPU process (TRT venv) ─────────────────────────────────────────────┐
│ T-gpu: the only thread issuing UNet/TAESD work. EDF select → pinned slot k │
│   → H2D → UNet (graph) → TAESD-TRT full height, uint8 → async D2H → event  │
│   (depth 2). Owns pose choice and motion-bank crossfade state per session. │
│ T-collect: blocking event → face ring[stream]                              │
│ T-prep: Whisper on its own stream (graphs captured at warmup, before it)   │
└──────┬─────────────────────────────────────────────────────────────────────┘
       │ faces 196 KB/frame (/dev/shm ring per stream)
┌──────▼── post pool ×3–4 (TRT venv, NumPy 1.23.5, OpenCV 4.9, no torch) ─┐  ┌─ tracker pool ×4–6 (SoulX venv) ──┐
│ ordered actor per stream: paste face into source frame → tracker req   │─►│ one FaceMesh graph per stream;    │
│ → 3-tap filter → refined mask/blend/warp (numba) → I420                 │◄─│ receives pre-pasted 1.38 MB frames│
└──────┬──────────────────────────────────────────────────────────────────┘  └───────────────────────────────────┘
       │ I420 0.69 MB/frame (/dev/shm ring per session, ≤1.5 s)
┌──────▼── media workers ×3–4 (≤5 sessions each, no torch) ─────────────────┐
│ aiortc PCs · playout · shared idle YUV cache · VP8 1-thread · Opus · RTP  │
└───────────────────────────────────────────────────────────────────────────┘
 Shared read-only in /dev/shm: avatar frames, masks, chin caches, idle YUV.
```

**Invariants**
1. One thread issues UNet and TAESD work. Whisper is the one exception: it runs on its own stream in the prep thread. All graphs are captured at warmup, or with `capture_error_mode='thread_local'`.
2. The GPU process never waits on a media process. Backpressure comes from ring occupancy, read at selection time.
3. Per-stream order is kept end to end.
4. Pixel math, including the tracker-input paste, runs on NumPy 1.23.5 and OpenCV 4.9.
5. Run-ahead is stored as faces: 23.5 MB per stream for 6 s, vs 83 MB as I420 [I]. Compose happens at most 1.5 s ahead.
6. Idle playback never decodes and never touches the GPU.
7. Every stage sits behind a flag, and 0 restores the in-process path.
8. Session state that selects GPU work (pose, motion bank) lives in the GPU process. Media workers send pose and barge-in commands over the control socket.

**4.1 Shared-memory rings.**
- **Change:** a new `scripts/shm_ring.py` with seqlock rings.
- **Load:** faces 59 MB/s + pasted tracker frames 413 MB/s + I420 206 MB/s at 300 fps; 79 + 550 + 275 MB/s at 400 [I].
- **Gate:** fuzz test with 0 torn reads.
- **Effort:** 2–3 d.

**4.2 Post pool processes.**
- **Change:** move compose, `_apply_webrtc_pose_crossfade` (:2008), chin and I420 conversion out of the GPU process. Split the compose and chin code out of `api_avatar.py`, which imports torch at module level [C :8], so post workers stay torch-free.
- **Flag:** `HLS_POST_WORKERS=4` / `0`.
- **Gain:** chin post capacity goes from 830 fps with threads to more than 2000 fps with 8 processes [M].
- **Gate:** golden SHA through the pool.
- **Effort:** 3–4 d.

**4.3 Media workers.**
- **Change:** 3–4 workers of ≤5 sessions each, owning the PeerConnections, playout, VP8, Opus and RTP. The front routes /offer, /ice, /stream and the pose endpoints to the owner worker. Pose, barge-in and crossfade commands travel over the control socket to the GPU process.
- **Flag:** `MUSETALK_MEDIA_WORKERS=4` / `0`.
- **Gate:** A/V skew, barge-in and multipose regression scenarios pass; added latency ≤1 frame.
- **Effort / deps:** 7–10 d (api_server.py is 6167 lines [M-now]) / D13.

**Phase 4 exit:**
- 20 streams with chin on show 0 late frames and loop lag p99 ≤20 ms in every process.
- RSS within the budget re-measured with item 0.9's method, with MemAvailable ≥4 GB.
- Output SHA-equal to the in-process path; added latency ≤1 frame.
- Labelled video.
- **Kill (K10):** added latency above 1 frame, or A/V, barge-in or multipose regressions still failing after 2 iterations. Then stop and ship the in-process capacity.

### Phase 5 — sign-off

Run the §6 protocol at 15 and 16 streams, then 20 (stretch). Report one real-network run (TURN, NACK) separately from certification (D21). About 1–2 d.

### Phase 6 — lossy levers ("attempt, then gate visually"; each behind one flag; in this order)

**6.1 Tracker-row fill with the TRT staged-crop TAESD (L).**
- **Change:**
  - Decode only rows ≥R, with R per avatar: 104 for standard avatars, 80 for `_fh1` [M p7/p8]. Derive R from the avatar's mask plan at load and assert it.
  - Fill the tracker's rows below R from the source face crop.
- **Flags:** `MUSETALK_TAESD_BACKEND=trt_crop_trackfill`, `MUSETALK_TAESD_CROP_ROW=auto`.
- **Gain:** 0.51 → 0.36 ms (0.35–0.41 [M: 0.354 in pipeline; 0.403 at R=80]); expected ceiling 334 → 352 [I].
- **Gate:**
  - G-TRACK.
  - Poison test: fill the rows below R with 255; the composite may change only through the landmarks.
  - Video.
- **Effort:** 1.5–2 d.

**6.2 Mixed-INT8 UNet (L), on the stagewise backend.**
- **Q0 (new):** rerun the accuracy probe with **TAESD** decode, on **held-out avatars**, with used-row metrics, for naive INT8 and recipe v1. Do this before scheduling the rest.
- **Q1, corpus:** from item 0.7. Compute metrics only on the rows the blend uses: decoded rows ≥R, latent rows ≥13 [M p7; D].
- **Q2, offline sensitivity sweep:** ModelOpt 0.23.2 fake-quant, per block, per layer class, then per layer inside the 3 most sensitive blocks. Rank by ms saved per unit of error (§10 table).
- **Q3, recipe v1:**
  - INT8: the ResNets and GEMMs of down0–3, mid, up2 and up3.
  - FP16: up0 (7.2% block error), up1 (21.7%), the 1024-token transformers (INT8 only 1.02× faster [M]), conv_in and conv_out.
  - The 7 INT8 blocks still carry 0.5–4.7% block error each [M].
  - Estimated UNet: 1.91 (1.84–2.12) ms [I].
- **Q4, recovery ladder.** Stop at the first step that passes:
  1. calibrators: max, then entropy, percentile, MSE;
  2. per-layer FP16 carve-outs;
  3. SmoothQuant α sweep 0.5–0.9, used-row metrics only (the default made used rows worse [M]);
  4. AdaRound or BRECQ one block at a time;
  5. QAT by distillation from the FP16 TRT teacher. Scale-only QAT (about 4–6 GB) fits on the 4070S with the server stopped; full-weight QAT (about 8.5 GB + activations) needs a rented GPU (D9).
- **Flags:** `MUSETALK_UNET_STAGEWISE_QRECIPE=<name>`, `MUSETALK_UNET_STAGEWISE_INT8_BLOCKS=…`. An empty value means FP16.
- **Gates:** **the L1–L8 lossy gates (§6.4) with your thresholds (D17).** G-UNET is reported for information only: INT8 fails it by construction (inferred RMSE about 0.059 vs the 0.01 MAE gate [I, wf1]).
- **Gain on its own:** about 373 fps expected [I]. Not 400 by itself.

**6.3 INT8 TAESD at full height (L).**
- **Change:** `MUSETALK_TAESD_PRECISION=int8`. A variant keeps the final 256² stage in FP16.
- **Gain:** 0.51 → about 0.21 ms [I]. Only the crop was measured: 46.3 dB, max_abs 0.18 [M p9]. Combined with 6.2: 424 expected [I].
- **Risk:** it will stress G-TRACK.
- **Effort:** 1 d.

**6.4 In-utterance pause gating (≥500 ms pauses).**
- **Flag:** `MUSETALK_PAUSE_GATING_MS=0|500`.
- **Gain:** 0–18% of speaking frames; 0% on Kokoro turns [M]. Demand only.
- **Constraint:** it conflicts with the rule at hls_gpu_scheduler.py:1771, so only with D8.

**6.5 UNet block pruning plus distillation (L; for 400 only; XL).**
- **Flag:** `MUSETALK_UNET_WEIGHTS=<pruned ckpt>`. An empty value means the stock weights.
- **Candidate cut:** about 33% of FLOPs [M analytic]; estimated 1.15–1.3× faster [I].
- **Risk:** in SoulX, pruning the generator broke lip sync (correlation 0.44–0.58) [D].
- **Kill:** at the first lip-sync regression.
- **Deps:** D9, D10.

**6.6 TRT 10.9 build environment.**
- **Status:** the TRT 10.9 on the box is runtime-only [D PROVENANCE.txt]. A builder needs 4–5 GB of free disk [D/I].
- **Spike first:** build 3 blocks and compare. Adopt only at ≥5% gain.

### Phase 7 — capacity productization (only if the product is independent sessions; about 3 d)

**7.1 Admission module.**
- **Change:** a new `scripts/capacity_admission.py`, hooked into `_require_accepting_new_sessions` (api_server.py:985) and group creation. It publishes through `worker_control_plane._payload()` (:242) and sets `LINGUA_WORKER_DEFAULT_CAPACITY` (:71).
- **Admit a new session only if:**
  - P(speakers > C_eff) ≤ 1e-3, using a Poisson-binomial over each session's duty (prior 0.5), where a group counts as one correlated unit;
  - C_eff = floor(0.95 × measured sustained fresh fps / 20);
  - MemAvailable ≥4 GB, and free VRAM ≥1 GB.
- **Flag:** `MUSETALK_CAPACITY_ADMISSION=1`.

**7.2 Turn-start deferral.**
- **Flag:** `MUSETALK_TURN_DEFER_MS=0|300`.
- **Rule:** delay a new turn's start by up to 300 ms when speakers ≥ C_eff + 2 and the smallest speaking buffer is under 1 s. Frame rate never drops mid-turn.

**7.3 SLO monitors:** first-frame p95, fresh fraction, held-run length, loop lag, RSS slope, chin coverage.

**Order and effort** [I: sum of item estimates, with P3's offline items overlapping P1]:
- **Sequence:** P0 (6–7 d) → P1 (8–10 d; 3.0a, 3.0c and 3.1–3.3 run alongside it) → P2 (6–8 d) and the rest of P3 in parallel → P4 (12–17 d; probable) → P5 (1–2 d).
- **Lossless 15-stream sign-off:** about 5–7 weeks without Phase 4, 8–10 weeks with it.
- **400 fps:** adds P4 plus P6, i.e. 2–6 weeks plus training compute.

---

## 5. Projected fps waterfall

Cumulative, with 100% of streams speaking. Delivered = ceiling × efficiency, pairing the low-ms case with e = 0.97, expected with 0.93, and high-ms with 0.88. All values [I] from the component sources in §3, unless marked otherwise.

| # | Step | GPU ms/frame (low / exp / high) | Ceiling fps | Delivered (high / exp / low) | Basis |
|---|---|---|---|---|---|
| 0 | Today, live (standard compose, no chin, x264) | 3.95 | 253 | 136–145 observed (contaminated); structural ≤174–203 | ceiling [M p6 + I other]; delivered [D] |
| 1 | + P1 serving fixes + UNet graph (bs8) | 3.83 / 3.86 / 3.88 | 261 / 259 / 258 | 253 / 241 / 227 | [M p6 30.26 ms] |
| 2 | + TRT TAESD, full height | 3.48 / 3.52 / 3.58 | 287 / 284 / 279 | 279 / 264 / 246 | TAESD standalone [M]; u8 [I] |
| 3a | + torch_tensorrt bs16 + graph (fallback 2.4) | 3.34 / 3.38 / 3.44 | 299 / 296 / 291 | 291 / 275 / 256 | **closest to measured; misses 300** |
| 3a′ | ONNX stagewise bs8 + graph (2.3a), instead of 3a | 3.07 / 3.12 / 3.32 | 326 / 321 / 301 | 316 / 298 / 265 | per-block bs8 sum [M]; chaining and clocks [I] |
| 3b | ONNX stagewise bs16 + graph (2.3b) | 2.87 / 2.99 / 3.32 | 349 / 334 / 301 | **338 / 311 / 265** | bs16 factor [I]; settled by 2.2 |
| 4 | + P3 chin live / P4 split | unchanged (chin uses no GPU [D]) | unchanged | 338 / 311 / 265 if efficiency holds | in-process chin unmeasured; 3.0c decides |
| 5 | + 2 raw-TRT contexts / H2D copy stream / TRT 10.9 | −0–5% | – | upside only, not counted | unmeasured |
| – | **End of lossless.** 300 at the expected case, not at the low case. 400 not reached. | | | | |
| 6 | (L) 6.1 tracker-row fill + crop | 2.72 / 2.84 / 3.18 | 368 / 352 / 314 | 357 / 327 / 277 | crop [M]; tracker effect unknown |
| 7 | (L) 6.2 mixed-INT8 v1, FP16 full-height TAESD | 2.39 / 2.49 / 2.77 | 419 / 401 / 361 | 406 / 373 / 318 | per-block [M]; quality **fails today** |
| 8 | (L) 6.2 + 6.1 | 2.24 / 2.34 / 2.63 | 447 / 427 / 381 | 434 / 397 / 335 | [I] |
| 9 | (L) 6.2 + 6.3 INT8 TAESD at full height | 2.09 / 2.19 / 2.47 | 479 / 456 / 405 | 465 / 424 / 357 | full-height INT8 TAESD never measured |

The worst stacked lossless case (full clock penalty, no bs16 gain): 2.78 + 0.55 + 0.10 = 3.43 ms → 291 ceiling [I]. That is K2.

**Efficiency needed to deliver 300** [I: 300/ceiling]:

| Ceiling (fps) | 296 | 301 | 321 | 334 | 349 |
|---|---|---|---|---|---|
| Efficiency needed | 1.01 (impossible) | 1.00 | 0.93 | 0.90 | 0.86 |

**Sessions supported at each expected capacity.** Each cell is the largest number of sessions for which the p99 (or p99.9) of concurrent speakers × 20 fps ≤ capacity [I, binomial, computed].

| Capacity (fps) | Speakers | 40% duty | 50% duty | 60% duty | 50% duty at p99.9 | 100% (wall) |
|---|---|---|---|---|---|---|
| 241 (P1) | 12 | 18 | 15 | 14 | 14 | 12 |
| 264 (P1 + TRT TAESD) | 13 | 20 | 17 | 15 | 15 | 13 |
| 298 (ONNX bs8) | 14 | 22 | 19 | 16 | 16 | 14 |
| 311 (ONNX bs16 + P3) | 15 | 24 | 20 | 18 | 18 | 15 |
| 373 (6.2) | 18 | 30 | 25 | 22 | 22 | 18 |
| 424 (6.2 + 6.3) | 21 | 36 | 30 | 26 | 27 | 21 |

The wall's group flow (`POST /webrtc/groups/{id}/stream`, api_server.py:4116) sends the same audio to every tile, so a wall is 100% correlated [C]. Run-ahead absorbs bursts for independent calls, but not for a wall start.

---

## 6. Validation protocol

### 6.1 Box conditions (every stage)
- `box_guard.sh` passes before and after each stage. D2 is resolved.
- Only the candidate server runs on the GPU, inside an approved lease window, with your live server stopped.
- No SoulX or Codex GPU processes. The GPU sits at 0% for 10 s before each stage.
- MemAvailable ≥4 GB throughout; oom_kill unchanged; test disk writes within the ledger.
- `ps` and `nvidia-smi` snapshots every 10 s. The run is **contaminated** if a foreign GPU process appears or foreign CPU exceeds 1 core; repeat it.

### 6.2 Rig
- **Audio:** the item 0.5 corpus, distinct per session, with `MUSETALK_DISABLE_LOCAL_TTS=1`.
  - S1 uses concatenated long turns or `--chain`, so every stream speaks for the whole window.
- **Content:** the required recipe on H3 expressive sources.
  - At least 3 certification identities, each with chin assets (3.0a) and your chin acceptance (D20).
  - chinese_bob is included only after that acceptance.
  - 5 sessions per identity.
- **Client:**
  - Off-box if possible (D16).
  - On-box: pin by **physical core**. The client gets cores 12–15 = CPUs {12–15, 28–31}; the server gets {0–11, 16–27} [M-now lscpu: CPU n+16 shares core n].
  - Shard at ≤5 peers per client process.
  - The run is invalid if client loop-lag p99 exceeds 20 ms.
  - On-box loopback receivers showed p99 gaps of 62–108 ms even at light load [M], so cadence is judged on server-side send timestamps as well as the client.
- **Certification is loopback / LAN only.** The real-network run is reported separately (D21).

### 6.3 Scenarios
| ID | What | Demand |
|---|---|---|
| S0 | Smoke test, N = 1 and 3. It also produces the round's video. | – |
| S1 | Burst / wall: N all speaking, long or chained turns, ≥5 min at the power cap. Ramp 10 → 12 → 14 → 15 → 16 → 17 → 20, with 20 s warm-up and 60 s cool-down per stage. | 300 fps at N = 15 |
| S2 | Conversational: N = 15 and 20 independent sessions, Poisson turns at 50% and 60% duty, 10–15 min. Log the realized speaker histogram against the binomial. | 240–340 fps |
| S3 | Soak: S2 for 60 min (leaks, RSS slope, clocks). | – |
| S4 | Churn and admission (Phase 7). | – |
| S5 | Barge-in: 20% of turns aborted; return to idle ≤0.5 s under load (0.35–0.40 s measured unloaded [D]). | – |

Each scenario runs 3 times. Report the minimum and the median.

### 6.4 Metrics, pass criteria and quality gates

**Metrics.**
- **Per stream, per second** (lifetime monotonic counters, item 0.4):
  - fresh = Δframes_played; held = Δframes_duplicated; stall = Δstrict_video_stall_seconds;
  - queue_underruns; run-ahead seconds;
  - client interval average, p99 and maximum; server send gaps;
  - first-frame latency split into setup / queue / first block;
  - A/V offset; RTCP loss;
  - `chin_applied`, `chin_fallback`, `warp_skipped`, invariant violations.
- **Server:** generated speaking fps (scheduler side); per-second concurrent speaker count; GPU busy fraction from events; batch fill; GPU ms per frame; idle gaps; feeder CPU per batch; loop lag in each process; cores, RSS and threads per process; VRAM; power and SM clock at 1 Hz.

**Pass criteria for "300 fps / 15 streams certified"** (S1 at N = 15, 3 of 3 repeats, ≥180 s of steady state):
- **Throughput:**
  - Scheduler-side generated speaking frames ≥297 per second, sustained.
  - Concurrent speakers = 15 for ≥99% of the window.
  - output_fps = 20 asserted.
- **Per stream, within turns:**
  - fresh ≥99.5% of output slots after the prebuffer; no held run longer than 2 frames;
  - 0 stall seconds; 0 underruns after the prebuffer;
  - average interval 0.050 ± 0.001 s; maximum ≤0.100 s on server send timestamps;
  - A/V within `WEBRTC_AUDIO_MAX_LEAD_SECONDS` 0.15 and MAX_LAG 0.25 (webrtc_tracks.py:2789-2793).
  - Prebuffer and idle time are reported separately.
- **First live frame:**
  - S1 burst: p95 ≤1.0 s, p99 ≤1.3 s [I: 15 × 10 frames / 311 fps = 0.48 s queue + 0.14 s setup (D p50) + ~0.12 s first tracked frames (10 × 11.5 ms, M) + 0.05 s lookahead ≈ 0.79 s for the last stream].
  - S2: p95 ≤0.8 s [I, proposed].
- **Margin:** S1 passes at N = 16, or GPU busy is ≤95% at 15.
- **Health:** loop lag p99 ≤20 ms in every process; MemAvailable ≥4 GB; oom_kill unchanged; VRAM ≤10.5 GB; no restarts.
- **Recipe:** chin applied on 100% of speaking frames, excluding the categories approved in D11; 0 sampled invariant violations.
- **Exactness:** a pre-encode tap on 2 streams is SHA-equal to the replay reference (E0 builds), or passes the E1 gates.
- **"15 conversational sessions certified":** S2 at 60% duty for 10 min, with the same per-stream criteria.
- **400 / 20 streams:** the same criteria at N = 20.

**Quality gates**

| Gate | Applies to | Criterion |
|---|---|---|
| G-EXACT | every E0 lever | SHA-256 of every pre-encoder frame identical to the current versioned baseline (item 0.3 replay, ≥240 frames × 3 identities). Chin: `verify_baseline.py` 480 frames zero diff, protected lip pixels unchanged, sequential == pipelined [D]. |
| G-TAESD | 2.1 | Against compiled TAESD: ≤3 LSB max, ≤0.2 LSB mean [I, proposed]. |
| G-UNET | 2.3a, 2.3b, 2.4 (FP16 only) | `validate_unet_backend.py` mae_max ≤0.01, max_abs ≤0.5 [D]. Also report rel-L2 vs torch (shipping 0.38% [M]) as the FP16 noise floor. Reported, not gating, for 6.2. |
| G-TRACK | anything that changes generated-face pixels (2.1, 2.3, 6.x) | FaceMesh jaw + lip deviation vs the reference run: mean ≤0.05 px, p99 ≤0.15 px [I, proposed]. Chin-target error within ±0.05 px of the 20 fps reference (0.10). |
| G-ENC | 1.8 | PSNR of decoded vs pre-encode frames within 0.2 dB of baseline [I, proposed], plus the receiver recording. |
| L1–L8 | lossy 6.x (thresholds are D17) | See the list below. Every metric is also reported FP16-vs-FP16 as the noise floor. |

**L1–L8 lossy gates**, all on used rows:
- **L1:** latent rel-L2 vs FP16 (naive INT8 7.07% [M]).
- **L2:** decoded face PSNR with TAESD: mean ≥45 dB, worst frame ≥42 dB. Naive INT8 is 40.3 / 37.3 dB [M]. Mouth-ROI MAE ≤0.006.
- **L3:** FaceMesh inner-lip opening correlation ≥0.97, mean |Δ| ≤0.5 px.
- **L4:** chin error ≤ FP16 + 0.2 px.
- **L5:** mouth-sharpness ratio 0.95–1.05.
- **L6:** temporal flicker ≤1.05× FP16.
- **L7:** SyncNet delta, reported only.
- **L8:** your verdict on the video.

### 6.5 Video deliverable (every round)
- **Output:** `experiments/throughput300_r<N>_<date>/`, encoded straight from memory, and moved off-box when a round is done (disk ledger).
- **Layout:** labelled side-by-side at the **same playback speed (20 fps)**.
  - Column A: baseline, the 20 fps accepted reference (item 0.10) from the versioned baseline.
  - Column B: candidate, from the live-path pre-encode tap.
  - Full frame on top; nearest-neighbour 3× mouth zoom below.
  - Flags and gate numbers burned in, plus an optional |A−B| × 8 panel.
- **Content:**
  - Lossless rounds: two certification identities, 10–20 s each. Lossy rounds: all review identities.
  - One receiver recording from the multi-stream run, labelled "not frame-aligned".
- **Tools:** `make_pair_comparisons.py` and `package_review.py` (chin_fps_validation_20260927), and the drawtext labelling from chin_seam_refinement_20260927.
- **Also:** a contact sheet and a gate JSON. Lossless rounds say "SHA-identical" and show a black diff panel.

---

## 7. Risks and kill criteria

### Kill criteria
- **K1. Saturated Phase 1 efficiency.**
  - <0.85: stop GPU work, profile, and pull Phase 4 forward.
  - 0.85–0.90: continue, but Phase 4 becomes required for 300.
- **K2. Item 2.2 measures the chained stagewise bs16 above 2.64 ms/frame sustained.**
  - Lossless then tops out at about 298 (ONNX bs8, if it passes) or 275 (row 3a) expected.
  - 300 all-speaking then depends on 6.1 or 6.2, or on redefining the target by duty cycle (D4).
  - Between 2.51 and 2.64: marginal. Proceed only with a measured e ≥0.95 or a lossy backstop.
  - Report either outcome to you before continuing.
- **K3. TRT TAESD fails G-TRACK.**
  - Revert to compiled TAESD: 2.40 + 0.85 + 0.08 = 3.33 ms → 300 ceiling → about 279 delivered, so 300 is lost [I].
  - Item 6.1 falls with it. Only INT8 UNet remains.
- **K4. bs16 fails G-UNET twice.** Stay on ONNX stagewise bs8 (321 ceiling, about 298 expected). If that also fails, stay on the .ts bs8 with TRT TAESD (284 ceiling).
- **K5. INT8.** The ladder, including one QAT round, finds no recipe with UNet ≤2.03 ms/frame [I: 400 at e = 0.93 needs ≤2.33 ms total, minus INT8 TAESD 0.21 and other 0.08] that passes L1–L6 and your video review. Close INT8 and the 400 target, or go to 6.5 only with D10.
- **K6. Live chin is not SHA-equal to the replay reference after 2 iterations.** Stop and fix dtype or tracker-state parity first.
- **K7. Any SHA break in an E0 lever** blocks the merge until it is explained.
- **K8. OOM or contamination** invalidates the run. If your server is killed and our process is implicated, stop and write a post-mortem.
- **K9. Pruning:** stop at the first lip-sync regression.
- **K10. Phase 4:** added latency above 1 frame, or A/V, barge-in or multipose regressions after 2 iterations. Stop and ship the in-process capacity.
- **K11. Chin assets:** the 3.0a assets do not match the research caches for japanese_new / latina_new. Stop before live chin.

### Other risks
| # | Risk | Mitigation |
|---|---|---|
| R1 | Hidden serial costs keep efficiency below 0.9 | Saturated layered baseline (0.8); K1 |
| R2 | bs16 fails the gate on TRT 10.3 (a 05-29 artifact did [D]) | ONNX bs8 fallback; multi-avatar corpus |
| R3 | The ONNX gain is partly burst-clock state (per-block runs were short sync-per-call bursts) | Chained ≥60 s bench in 2.2 first; K2 |
| R4 | INT8 quality: 5 dB short on used rows [M] | Q0 re-probe → ladder → QAT; lossy gates |
| R5 | Sustained clocks and thermal soak: every ms figure comes from ~1.5–2 s bursts [M] | ≥5 min L1 in 0.8; ≥60 s in 2.2 |
| R6 | RAM: 12.0 GB available now, 8.9 GB in /dev/shm, Pylance 4.4 GB [M-now]; 15-session budget 9–14.5 GB [I] | D2 as a hard prerequisite; 0.9; guard; watchdog |
| R7 | VRAM: 11 stagewise contexts; NVENC 2.47 GB at 12 sessions [M]; dual engines OOM [D] | Shared device memory; VP8; free the eager UNet |
| R8 | Chin math is valid only on tracked source frames | 3.5, 3.7 counters, D11 |
| R9 | SMT: 11–14 logical CPUs busy is 55–70% of about 20 core-equivalents [I]; FaceMesh CPU per frame rose 3.4 → 13.2 ms under load [M]; 33 threads per graph [M] | Physical-core pinning; shard; measure in 3.0c |
| R10 | Real networks unmeasured [M] | Separate report in Phase 5 (D21) |
| R11 | Multipose, barge-in and A/V regressions; no tests/ directory | Scripted scenarios every round; flags |
| R12 | Latency: 47.8 ms bs16 batches [I: 16 × 2.99] × depth 2, + 50 ms chin lookahead, + prebuffer | 10-frame packed startup slices; first-frame SLOs |
| R13 | Codex session actively editing api_server.py and the templates [M-now] | Worktree; D13 |
| R14 | Disk fills mid-write | Ledger; atomic renames; in-memory hashing; off-box videos |
| R15 | TRT tactic variance ±5% [D]; rebuilds at startup can change numerics | Persist engines and the timing cache; fingerprint check at load (2.1, 2.3, D15) |
| R16 | Padding waste at low load with a bs16-only engine | Irrelevant at saturation; watch latency at low load |
| R17 | The recipe was accepted at 24 fps; live is 20 fps [C] | Item 0.10; D18 |
| R18 | FaceMesh depends on another project's venv (SoulX) | Checksum per run (3.3); a dedicated venv after D1 |
| R19 | Production avatars lack chin assets [C/D] | 3.0a; K11; D20 |

---

## 8. Closed levers — do not retry

| Lever | Evidence |
|---|---|
| FP8 (UNet or TAESD) on TRT 10.3 / sm_89 | No FP8 conv kernel; 1.25–6.9× slower; FP8 with FP32 accumulate has the same peak as FP16 with FP16 accumulate [M]; SoulX saw the same on TRT 10.9 [D] |
| Raising the power cap | Power limit = maximum limit = 220 W [M-now] |
| Batches >16 | Eager bs16 → bs32 only −4.4% per frame [M]; bucket profiles beyond 8/16 gave nothing [D] |
| Larger TAESD batches | bs16 slower per frame than bs8 (0.502 vs 0.486 [M]) |
| Second CUDA stream for TAESD; two torch_tensorrt UNets on two streams | 0.6–1.7% [M]; torch_tensorrt serializes onto one engine stream [M]. Raw-TRT two contexts (−5% at bs8 [M]) is the measured exception, kept as optional 2.5 |
| Two server replicas or MPS on the saturated GPU | +1.3% [D SoulX]; 5–6 GB VRAM each |
| DeepCache / cross-frame reuse; source-only caching | Single-step model with audio in every transformer [C]; ≤3.3% of FLOPs [M] |
| Latent-space TAESD crop | Never exact; receptive field 18.1 latent rows [M] |
| **Staged TAESD crop as an exact lever under chin** | The tracker reads the full generated face [C run.py:38-41,139]. Reopened only as lossy item 6.1 |
| Forced inductor max-autotune for TAESD; cudnn.benchmark off | Skipped on 56 SMs and no gain when forced [M]; benchmark off is 11% slower [M] |
| "Unfused attention" hunting | Attention already runs as fused MHA, 5.3% of engine time [M] |
| TorchScript PTQ INT8 | No scales; illegal memory access [D]. Use ModelOpt ONNX Q/DQ only |
| Dynamic-shape UNet TRT; torch.compile UNet | Failed the gate (mae 0.0122, max_abs 1.94) [D] |
| Dropping generator blocks without distillation | SoulX lip-sync correlation 0.44–0.58 [D] |
| ROI 224/192; lowering fps | Rejected as blurry [D]; 20 fps required |
| Whisper on CPU, shorter windows, cross-stream batching | Net negative; 10–24% feature error; <1% of GPU [M] |
| Cheaper FaceMesh (crop, half-res, unrefined, static mode); tracking subsampling or a causal filter | 0.3–2.4 px jaw/lip deviation [M]; violates the accepted recipe [D] |
| GPU landmark net or GPU compose | Spends the binding GPU budget while CPU has headroom [I] |
| GIL switch-interval tuning | 881–1055 late frames, worse [M] |
| Global thread caps / worker-pool count tuning | GPU util fell to 37%; 82 → 77% [D] |
| NVENC for every session | 12-session cap; 656 ms latency spikes at 12 [M]; 2.47 GB VRAM [M] |
| On-box TTS during capacity tests | 30–85% of CPU [M] |
| Sum-of-blocks discount for the stagewise UNet | Block boundaries ship; same-session sum vs full forward differs by only 0.6% [M] |

---

## 9. Decisions needed from you

1. **D1 Disk.** 1.3 GB free [M-now]. This blocks engine persistence for 2.3 / 2.4 / 6.x; D15 is the in-RAM alternative. Deleting is your call.
   - **Candidates:**
     - `models/trt_downloaded_backup_20260915/`: 2.2 GB, a Jul 10 build, not the live engine.
     - Two SoulX `.bundle` files in /workspace: 2 × 285 MB.
     - `/root/.cache/uv`: 582 MB. `/root/.cache/pip`: 61 MB.
     - `models/syncnet`: 1.4 GB, diagnostic only; could move off-box.
   - **Keep:**
     - `models/tensorrt_unet_static_bs8_20260529` (the live engine and rollback).
     - `models/musetalkV15/unet.pth`.
     - `/tmp/torchinductor_root` (835 MB).
     - SoulX `trt-10.9.0.34`.
   - **Needed free:** ≥2.7 GB for a .ts; ≥2.2 GB for FP16 stagewise engines; ≥4–5 GB for a TRT 10.9 builder.
2. **D2 RAM and box sharing. This is a hard prerequisite for 0.6 step 2, 0.8, 0.9, 2.2 and every load test.**
   - MemAvailable is 12.0 GB now [M-now].
   - `/dev/shm/soulx-lfs-state-20260919` holds 7.9 GB; `musetalk-corrected-benchmark` 654 MB; `musetalk-browser-pilot-20260925` 191 MB; `musetalk-nose-capture-tmp` 146 MB [M-now].
   - Freeing the first two raises available RAM to about 20.5 GB [I]. Stopping the 4.4 GB Pylance server during windows adds more.
   - Adopt one box-wide GPU lease that both the SoulX and Codex sessions honour.
3. **D3 Maintenance windows.** Your live server must be stopped during builds and the item 2.2 bench (about 1 h), and during load tests (1–2 h per phase).
4. **D4 What "15 streams" means.** Certify at 100% simultaneous speech (the wall; this plan's default), or at a duty cycle and percentile, e.g. 20 sessions at 50% duty = 300 fps at p99? This decides whether 2.3b and Phase 6 are needed at all.
5. **D5 Codec and transport.**
   - Transport: 20 fps (recommended).
   - Encoder: native VP8 on 1 thread for everyone (recommended), with x264 ultrafast/veryfast on 1 thread for H.264-only clients. The NVENC hybrid is not recommended (2.47 GB of VRAM, SM contention [M]).
   - Bitrate: 2.5 Mbps was the tested rate [M].
6. **D6 Prebuffer, run-ahead and overload.**
   - Run-ahead cap: 5 s proposed.
   - Prebuffer: the wall uses 0.5 s (10 frames) [D]; the code default is 2.0 s [C webrtc_tracks.py:35].
   - Under overload: delay new turns (recommended), or let every stream degrade.
7. **D7 cv2 I420 conversion** (≤2 LSB, mean 0.44 [M]): accept it, or keep PyAV conversion in the workers?
8. **D8 Which lossy arms to attempt:** 6.1, 6.2, 6.3, and 6.4 (which conflicts with the "do not gate quiet frames" rule).
9. **D9 Training compute** for QAT, distillation or pruning: the local 4070S (server down), or a rented 48–80 GB GPU (weights and avatar data would leave the box)?
10. **D10 400 fps:** go or no-go on pruning plus distillation (6.5) if INT8 stalls around 373 fps.
11. **D11 Chin fallback and tracker reset rules.**
    - When FaceMesh loses the face, at pose switches and at barge-in: standard compose for the frame, or hold the previous delta?
    - Reset the tracker per turn (proposed)?
    - Which fallback categories are excluded from "100% chin"?
12. **D12** How often do clients upload exact silence or listening turns? This sets the value of item 1.10.
13. **D13** Approve the Phase 4 scope, and coordinate with the Codex session's active edits (api_server.py, templates, new wall-audio files).
14. **D14** Do E1 runtime changes that stay within 1–3 LSB and pass G-TRACK count as "no quality change"? This covers TRT TAESD and the stagewise UNet. **The 300 verdict depends on this.**
15. **D15 Engine persistence.** Write engines to disk (needs D1; recommended), or build in RAM at startup?
    - In-RAM building is offered only after 0.6 lands and the per-block peak is measured.
    - It requires a persisted timing cache plus a load-time fingerprint check, so the engine you approved on video is the one that serves.
16. **D16** Is a second machine available as an off-box load generator? Strongly preferred.
17. **D17** Lossy gate thresholds L2–L6, and the review set (proposed: 8 identities, including 3 bearded).
18. **D18 (new)** Re-accept chin100 + refined seam at 20 fps on the item 0.10 labelled video. The accepted recipe was reviewed at 24 fps.
19. **D19 (new)** Where does TTS run in production for independent sessions? Capacity is certified with local Kokoro disabled; local Kokoro at 15 speakers would take 30–85% of the CPU [M].
20. **D20 (new)** Which H3 identities are the certification set?
    - Approve generating chin assets (3.0a) for production avatars, and review each identity's chin result (chinese_bob included).
    - Confirm whether any `_fh1` avatar is H3 expressive.
21. **D21 (new)** Is certification loopback / LAN only, with a separate real-network report (proposed), or must a TURN/NACK run pass the same criteria?

---

## 10. Appendix: evidence index

All probe outputs, scripts and the raw workflow results are kept in `docs/fps_comparisons/4070s_300fps_20260927/` (written as `$S` below; paths are relative to `MuseTalk/`).

**Digest and raw data**
- `$S/EVIDENCE_DIGEST.md`
- `$S/wf1_result.json`: readers and probes.
- `$S/plan/plan_*.txt`: per-plan extracts.
- `$S/wf2_result.json`: the four competing plans' scores, the three adversarial reviews, and the editor's applied/rejected changes.

**UNet**
- `$S/unet_probe/trt.json`: shipping bs8 24.62 ms; host enqueue 4.57 ms; `ts_cudagraph_torch` 23.65 ms with max_abs 0; `ts_runtime_cudagraphs_mode` 23.79 ms with no accuracy recorded.
- `trt_raw.json`: raw bs8 24.04 → graph 23.62 ms, max_abs 0; 2 contexts 45.66 vs 48.06 ms; 151 MB device memory; 1250 kernels per call.
- `trt_topblocks_bs8.json`: per-block FP16 sum 20.575 ms, INT8 sum 14.36 ms, per-block rel-L2, ONNX sizes, build seconds (sum 184 s); idle samples 2790–2805 MHz.
- `eager.json`: same-session top-level block sum 37.35 ms vs full forward 37.11 ms; 177.8 GFLOP per frame.
- `trt_blocks_res320.json`, `trt_blocks_res1280_res2560_tf320_tf1280.json`: ONNX-path bs8 → bs16 micro-blocks (0.449 → 0.920, 0.393 → 0.635, 0.581 → 1.099, 0.805 → 1.721, 0.491 → 0.849 ms). The res320 bs8/bs16 and res1280 bs8 runs had pid 3305906 (4972 MiB) resident.
- `ttrt_bs16.json`: 45.34 / 44.63 ms; mae 0.00196; 11.03 GB RSS; 44.3 s warm build.
- `tt16_timing_cache.bin`: 8 MB TRT timing cache from the torch_tensorrt bs16 build, preserved for item 2.2.
- `quant_accuracy.json`:
  - naive INT8: 7.07%; full 41.8 / 39.4 dB; lower half 40.3 / 37.3 dB.
  - SmoothQuant: 6.75%; lower half 39.9 / 35.9 dB.
  - conv-only: 6.09%; lower half 41.1 / 38.3 dB.
  - 64 eval frames of one avatar; SD-VAE decode.
- `gil.json` + `probe_gil.py:53-75`: 23.5–24.2 ms CPU per call; back-to-back calls with no per-call sync.
- `common.py:19-23`: `gpu_state` samples after 1.2 s of idle.

**Per-block reference, bs8** [M, trt_topblocks_bs8.json]

| Block | FP16 ms | INT8 ms | INT8 block rel-L2 | FP16 ONNX MB |
|---|---|---|---|---|
| down0 | 2.59 | 2.17 | 4.1% | 19 |
| down1 | 1.80 | 1.21 | 3.4% | 68 |
| down2 | 1.76 | 1.15 | 4.1% | 263 |
| down3 | 0.36 | 0.22 | 0.5% | 119 |
| mid | 0.60 | 0.40 | 1.1% | 183 |
| up0 | 1.01 | 0.55 | 7.2% | 309 |
| up1 | 3.69 | 2.23 | 21.7% | 487 |
| up2 | 3.88 | 2.49 | 4.6% | 133 |
| up3 | 4.78 | 3.86 | 4.7% | 35 |
| head | 0.03 | – | – | 0.04 |
| tail | 0.07 | 0.06 | – | 0.02 |

**TAESD**
- `$S/taesd_probe/p5_trt_taesd.json`: full-height 0.486 (bs8) / 0.502 (bs16) ms, FP16 output only. Crop 0.349 FP16 and 0.346 uint8. max_abs 0.0085.
- `p6_combined.json` / `p6_combined.py:17`: N = 60 batches per config; 30.93 / 26.93 / 30.26 / 26.26 / 23.43 / 24.03 ms per bs8; 48 s in total; 40 → 66 °C.
- `p7_rows.json`: the first row the blend reads, per avatar. Blend only.
- `p8_crop_rows.json`: R = 0 / 80 / 85 / 104 / 110 timings.
- `p9_int8.json`: INT8 crop 0.147 ms, 46.3 dB, max_abs 0.18.

**CPU and serving**
- `$S/cpu_probe/results_combined.jsonl`: A–I and U runs (C_chin15 0 late; D_chin20 19 late; F 0 late at p99 49.5 ms; G 0 late; H 881–1055 late).
- `combined_bench.py:23`: the chin stand-in is "+1.5 ms".
- `fm_during_C_chin15.json`: 15 processes, 13.2 ms CPU per frame, 11.46 ms latency.
- `results_fm.jsonl`: 3.43 ms CPU per frame for 1 process.
- `results_enc_conc.jsonl`, `nvenc_util_paced.txt` (2470 MiB), `nvenc_util_sat.txt` (32% GPU).
- `$S/serving/webrtc_transport_load_results.jsonl`: idle-decode ceiling about 183 fps; cache 299.7 / 399.7 fps; 518 / 674 threads.
- `$S/cpu-post-chin/`, `$S/audio_tts/`, `$S/model_path/`.

**Docs, logs and code outside MuseTalk**
- `/workspace/experiments/chinese_bob_webrtc_20260927/`:
  - `wall_api.log` (lines 4, 47, 506, 882, 1450, 1893–2219);
  - `run_local_api.sh` (your live launcher; leave it alone), `run_wall_api.sh:7`;
  - `pose-set.json` (fps 24.0).
- `/workspace/experiments/chin_fps_validation_20260927/`: README.md; `run.py` (encode `-r 24` at :18-19, Tracker at :26-47, reset at :102, 240 frames at :92); `make_pair_comparisons.py`; `package_review.py`.
- `/workspace/experiments/chin_seam_refinement_20260927/README.md:69-72`; `/workspace/experiments/avatar_diversity_20260927/README.md`.
- `MuseTalk/docs/musetalk_4070s_pipeline_component_analysis_2026-09-22.md:80` (H2D ~0.5 ms per bs8), `CPU_OPTIMIZATION_ANALYSIS.md`, `current_cross_server_throughput_findings.md`.
- `MuseTalk/character_factory/h3_avatar_workflow/`: `chin.py`, `prepare_stage.py`, `track_stage.py`, `tracker_worker.py`, `backend.py`, `WORKFLOW.md`, `BATCH_THREE_POSE.md`.
- `/workspace/SoulX-FlashHead/benchmarks/tiny_vae_20260926/decoder_bakeoff/wait_quiet.sh`; `/workspace/SoulX-FlashHead/.restored-deps-20260926/trt-10.9.0.34/PROVENANCE.txt`.

**Code anchors (MuseTalk)**
- `scripts/hls_gpu_scheduler.py`: 193-205, 230, 246-252, 385, 566-571, 998, 1027, 1297, 1528, 1763-1782, 1771, 1868-1897, 1905-1921, 2008, 2141, 2398, 2486.
- `api_server.py`: 273-325, 985, 3539, 3930, 3982, 4116, 4238, 4524-4545, 5375-5407, 5495-5535.
- `scripts/webrtc_tracks.py`: 35, 62, 730, 748, 1479, 1599-1632, 1946-1973, 2216, 2255, 2318-2335, 2789-2793.
- `musetalk/models/vae.py`: 30, 147, 156.
- `scripts/trt_runtime.py`: 493, 687, 1425, 1543, 1641.
- `scripts/vae_fast_decoder.py`: 61, 128.
- `scripts/webrtc_native_vp8.py`: 123, 263.
- `scripts/avatar_manager_parallel.py`: 168-170, 334, 423.
- `scripts/api_avatar.py`: 8, 1321.
- `musetalk/utils/blending.py`: 304.
- `character_factory/h3_avatar_workflow/chin.py`: 56-63, 84-86, 92, 125, 186, 204, 229-240.
- `scripts/worker_control_plane.py`: 71, 242.
- `load_test_webrtc.py`: 444-466, 1057-1064.
- `templates/webrtc_wall.py`: 87-88 (HEAD).

**Box state read while planning** [M-now]
- `df /`: 1.3 GB free. `/dev/shm`: 8.9 GB used.
- `free`: 11,973 MB available; no MuseTalk server running; Pylance 4.4 GB RSS.
- nvidia-smi: GPU idle, 1 MiB used; power limit 220 W = maximum limit.
- `memory.events`: oom_kill 12.
- `lscpu`: 16 physical cores; CPU n+16 shares core n.
- Venvs: the TRT venv has NumPy 1.23.5 and OpenCV 4.9.0.80; the SoulX venv has NumPy 2.2.6 and OpenCV 5.0.0.93.
- `git status`: api_server.py and two templates modified; two new untracked wall-audio files.
- Engine files: sizes and dates as in §1, correction 4.
