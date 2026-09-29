arm=B_garbage15 overrides=/workspace/MuseTalk-perf300/experiments/live15_r5/garbage15.env:/workspace/MuseTalk-perf300/experiments/live15_r5/serve.env:/workspace/MuseTalk-perf300/experiments/live15_r5/common.env stages=ramp levels=15
git 130a62488e1d336e0537c464ed6bdff1dc6510e4 dirty=10
coturn_pid_before=3554991 alive=turnserver
--- /workspace/MuseTalk-perf300/experiments/live15_r5/garbage15.env
# Event-loop fixes found by the live test (frames unchanged), i.e. the production configuration:
#  - freeze the long-lived heap after startup and after each avatar load (gen-2 GC pauses 140 ms -> ~20 ms);
#  - take the per-turn nvidia-smi snapshot, GET /stats, /health and /worker/state off the event loop;
#  - build every avatar's idle clip into the idle frame cache at avatar warm time, with room for all 15 clips
#    (~2.2 GB), instead of at the avatar's first session create (a GIL-heavy decode while other streams play);
#  - queue live frames as packed I420 arrays, so no av.VideoFrame/VideoFormat reference cycle lives in the FIFO
#    long enough to reach generation 2 (gen-2 collections every ~10 s at 15 streams, ~30 ms each).
MUSETALK_GC_FREEZE=1
MUSETALK_OFFLOOP_DIAGNOSTICS=1
WEBRTC_IDLE_FRAME_CACHE_WARM=1
WEBRTC_IDLE_FRAME_CACHE_MAX_MB=2400
WEBRTC_QUEUE_PACKED_I420=1
# diagnostic: GC log + garbage types of every gen-2 collection
MUSETALK_GC_LOG=1
MUSETALK_GC_GARBAGE_TYPES=1
--- /workspace/MuseTalk-perf300/experiments/live15_r5/serve.env
# Arm B additions (exact output by design, CPU-tested): used as MUSETALK_ENV_OVERRIDES_FILE=serve.env:common.env
WEBRTC_NONBLOCKING_HANDOFF=1
WEBRTC_HANDOFF_CONVERT_THREADS=2
WEBRTC_IDLE_FRAME_CACHE=1
WEBRTC_IDLE_FRAME_CACHE_MAX_MB=1536
MUSETALK_THREAD_CAPS=1
HLS_SKIP_CROSSFADE_COPY=1
HLS_GPU_STAGE_SYNC_TIMING=0
MUSETALK_VAE_DECODE_TIMING_SYNC=0
--- /workspace/MuseTalk-perf300/experiments/live15_r5/common.env
# Arm A = the r5 engines + the scheduler shape they need + measurement + rig settings + RAM guard.
# Parsed (never sourced) by scripts/run_musetalk_server.sh via MUSETALK_ENV_OVERRIDES_FILE; one lever per line.
MUSETALK_UNET_BACKEND=trt_stagewise
MUSETALK_UNET_STAGEWISE_BATCH=16
MUSETALK_UNET_STAGEWISE_CACHE_DIR=/workspace/MuseTalk/models/tensorrt_unet_stagewise_sm89_srcg50
MUSETALK_TAESD_BACKEND=trt
MUSETALK_TAESD_TRT_BUILD=0
MUSETALK_TAESD_TRT_STRICT=1
HLS_SCHEDULER_FIXED_BATCH_SIZES=16
HLS_SCHEDULER_MAX_BATCH=16
HLS_SCHEDULER_STARTUP_SLICE_SIZE=8
MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=16
MUSETALK_TAESD_WARMUP_BATCHES=16
MUSETALK_TRT_FALLBACK=0
HLS_MAX_PENDING_JOBS=24
WEBRTC_LIFETIME_COUNTERS=1
HLS_GPU_EVENT_TIMING=1
WEBRTC_DEADLINE_PACING=1
MUSETALK_DISABLE_LOCAL_TTS=1
AVATAR_S3_ENABLED=0
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
LINGUA_DRAIN_TIMEOUT_SECONDS=30
AVATAR_CACHE_MAX_MEMORY_MB=9000
AVATAR_CACHE_TTL_SECONDS=14400
MUSETALK_AVATAR_MASK_CHANNELS=1
MUSETALK_AVATAR_PLAN_FLOAT_ALPHA=0
MUSETALK_AVATAR_MASK_STORE=png
MUSETALK_AVATAR_FRAME_STORE=png
coturn_pid_after=3554991 alive=turnserver
