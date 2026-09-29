#!/usr/bin/env bash
# Step 0: check that this machine has everything the 350/400 fps reproduction needs. Read-only; exits 1 on a blocker.
#   scripts/repro_400fps/00_check.sh [--deep]      --deep also hashes the avatars' mp4s and the 2.1 GB BEFORE engine
source "$(dirname "$0")/lib.sh"
fail=0
# each check runs in a subshell, so a check string can never exit this script
need() { if ( eval "$2" ) >/dev/null 2>&1; then log "ok      $1"; else log "MISSING $1 ($3)"; fail=1; fi; }
warn() { if ( eval "$2" ) >/dev/null 2>&1; then log "ok      $1"; else log "warn    $1 ($3)"; fi; }

log "GPU: $(nvidia-smi --query-gpu=name,compute_cap,memory.total,power.limit,driver_version --format=csv,noheader 2>/dev/null)"
need "GPU visible" "nvidia-smi -L" "NVIDIA driver / GPU"
warn "RTX 4070 SUPER (sm_89), the GPU the numbers were measured on" "nvidia-smi --query-gpu=name --format=csv,noheader | grep -q '4070 SUPER'" \
     "other GPUs work, but the engines rebuild for them and the fps differ"
warn "power limit 220 W (published runs)" "[ \$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | cut -d. -f1) -ge 220 ]" \
     "a lower cap lowers every fps number"
need "main venv $PY" "test -x $PY" "scripts/install_musetalk.sh --matrix cu121 --with-legacy-int8 (TensorRT + modelopt extras)"
need "torch 2.5.1 with CUDA" "$PY -c 'import torch,sys; sys.exit(0 if torch.__version__.startswith(\"2.5.1\") and torch.cuda.is_available() else 1)'" \
     "the engines and gates were made with torch 2.5.1+cu121"
need "TensorRT 10.3" "$PY -c 'import tensorrt,sys; sys.exit(0 if tensorrt.__version__.startswith(\"10.3\") else 1)'" "tensorrt-cu12==10.3.0"
need "modelopt (INT8 Q/DQ export)" "$PY -c 'import modelopt.torch.quantization'" "nvidia-modelopt==0.23.2"
warn "torch_tensorrt 2.5.0 (BEFORE .ts engine, PAIR step only)" "$PY -c 'import torch_tensorrt,sys; sys.exit(0 if torch_tensorrt.__version__.startswith(\"2.5.0\") else 1)'" \
     "needed only for the like-for-like baseline"
need "FaceMesh venv $FACEMESH_PY (fixed path)" "$FACEMESH_PY -c 'import mediapipe'" \
     "scripts/install_musetalk.sh --with-chin-tools --chin-venv /workspace/SoulX-FlashHead/.venv, or symlink that path to your chin venv"
need "ffmpeg with libx264" "grep -q libx264 <(ffmpeg -hide_banner -encoders 2>/dev/null)" "apt install ffmpeg"
need "flock + setsid (box_guard)" "command -v flock && command -v setsid" "util-linux"
need "writable /workspace (GPU lease file)" "touch /workspace/.repro_write_test && rm -f /workspace/.repro_write_test" "box_guard keeps its lease in /workspace"
need "box_guard" "test -x scripts/box_guard.sh" "scripts/box_guard.sh"
need "chin recipe code matches the accepted renders" "$PY -c 'import sys; sys.path[:0]=[\"scripts\"]; from chin_multistream import paths; sys.exit(0 if paths.code_integrity()[\"matches_accepted_render_json\"] else 1)'" \
     "character_factory/h3_avatar_workflow/*.py or musetalk/utils/blending.py changed"
need "INT8 recipe $RECIPE" "test -s $RECIPE" "part of this repo"
need "package and its tools are committed (a fresh checkout has them)" \
     "git ls-files --error-unmatch scripts/repro_400fps/lib.sh $RECIPE scripts/int8_layer_study.py scripts/video_lineage.py scripts/build_unet_stagewise.py" \
     "commit them on this branch before copying the repo elsewhere"
if [ -e "$RUNTIME_ENV" ]; then
  warn "$RUNTIME_ENV equals the published runs' values" "diff <(grep -v '^#' $PKG/musetalk_trt_local_sm89.env | sort) <(grep -v '^#' $RUNTIME_ENV | grep . | sort)" \
       "differs: the harness applies it to every run, so results may not match"
else
  log "note    $RUNTIME_ENV is absent: the first harness step installs it from $PKG/musetalk_trt_local_sm89.env"
fi
warn "no seed timing cache (the published INT8 blocks were built cold)" "test ! -e docs/fps_comparisons/4070s_300fps_20260927/unet_probe/tt16_timing_cache.bin" \
     "build_unet_stagewise.py seeds from it silently; move it away for a like-for-like rebuild"
warn "published r5 set $PUBLISHED_R5 (to compare a rebuild against)" "test -e $PUBLISHED_R5/bs16/manifest.json" \
     "optional; copy with cp -L / rsync -L (it is a symlink farm), same GPU, driver and TensorRT only"
warn "published r2 video capture (40_videos.sh without an r2 rebuild)" "ls docs/fps_comparisons/4070s_300fps_impl_20260928/chin_multistream/V_srcmix_taesdtrt/stream00_*_faces.npz" \
     "gitignored arrays; otherwise build r2 (10_build_engines.sh --set r2) and capture it (30_benchmark.sh <r2 root> V)"
"$PY" "$PKG/check_inputs.py" "$@" --py "$PY" | while read -r line; do log "$line"; done
[ "${PIPESTATUS[0]}" -eq 0 ] || fail=1
avail=$(awk '/MemAvailable/{printf "%d", $2/1048576}' /proc/meminfo)
disk=$(df -BG --output=avail "$REPO" | tail -1 | tr -dc 0-9)
shm=$(df -BG --output=avail /dev/shm | tail -1 | tr -dc 0-9)
log "RAM available ${avail} GB (INT8 build ~10.5 GB of MemAvailable, 15 streams ~9.6 GB, 6 streams ~7.7 GB; box_guard kills below 3 GB)"
log "disk free ${disk} GB (a fresh engine set ~1.1 GB of plans; captures ~1 GB per quality/video round)"
log "/dev/shm free ${shm} GB (the harness maps each avatar's ~0.6 GB arena there; docker: --shm-size=8g or more)"
[ "${avail:-0}" -ge 14 ] || log "warn    less than 14 GB RAM available: builds and the 15-stream run will wait in box_guard"
[ "${disk:-0}" -ge 4 ] || { log "MISSING disk: need >= 4 GB free"; fail=1; }
[ "${shm:-0}" -ge 5 ] || { log "MISSING /dev/shm: need >= 5 GB free for the six avatar arenas"; fail=1; }
[ $fail -eq 0 ] && log "all prerequisites present" || log "fix the MISSING items above first"
exit $fail
