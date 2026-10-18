#!/usr/bin/env bash
# Candidate Vast On-start contents. Requires reviewed code on GitHub and EC2
# launch injection described in docs/VAST_LOCKED_STARTUP_ROLLOUT_PLAN_20261016.md.
# No credentials belong in this file. Linux worker only; do not run on macOS.
set +x
set -Eeuo pipefail

bootstrap_log() { printf '[bootstrap] [%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*"; }

# A PID file left on /workspace by a previous container is not a live API.
# Only a running process whose command line is the MuseTalk API chain blocks an
# in-place code update; unreadable or non-numeric content stays ambiguous.
api_pid_file_is_stale() {
  local pid_file="$1" pid="" cmdline=""
  [[ -f "$pid_file" && ! -L "$pid_file" ]] || return 1
  pid="$(tr -d '[:space:]' < "$pid_file" 2>/dev/null)" || return 1
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  if [[ -d /proc/self ]]; then
    [[ -d "/proc/$pid" ]] || return 0  # Permission-independent on Linux.
  else
    kill -0 "$pid" 2>/dev/null || return 0
  fi
  cmdline="$(tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null || true)"
  [[ "$cmdline" == *api_server.py* || "$cmdline" == *run_musetalk_server.sh* \
     || "$cmdline" == *run_trt_stagewise_server.sh* ]] && return 1
  return 0
}
bootstrap_die() { bootstrap_log "ERROR: $*" >&2; exit 2; }
# Stage -> epoch stamps for the control plane (worker_control_plane.py reports
# them as metadata.startup_timings); one file per boot, truncated at start.
stamp_stage() {
  [[ -n "${MUSETALK_STARTUP_TIMINGS_FILE:-}" ]] || return 0
  printf '%s=%s\n' "$1" "$(date +%s.%N | cut -c1-14)" >> "$MUSETALK_STARTUP_TIMINGS_FILE" 2>/dev/null || true
}
file_sha256() { { sha256sum 2>/dev/null || shasum -a 256; } < "$1" | cut -d' ' -f1; }

# Workers check out a sparse, blob-filtered tree: the revision's own spec drops
# ~3 GB of showcase media (generated/, assets/ banks, docs/) that no runtime
# path reads. The guard makes a wrong spec fail in seconds, never as a silent
# 3 GB lazy fetch or a per-request 404 later.
WORKER_SPARSE_SPEC=configs/worker-sparse-checkout
REQUIRED_WORKER_PATHS=(api_server.py download_weights.sh scripts/vast_onstart.sh scripts/install_musetalk.sh
  scripts/vast_server_ctl.sh scripts/run_musetalk_server.sh scripts/lib configs/recipes/r5.env configs/trt_bundles
  configs/dependencies requirements/server.in calibration/unet_portable_bs8 calibration/vae_decoder
  musetalk/whisper/whisper/assets/mel_filters.npz third_party_wheels templates/webrtc_pose_lab.py
  assets/segmind_motion_references)
DENIED_WORKER_PATHS=(generated assets/ltx23_pose_banks character_factory/generated docs)

apply_worker_sparse_spec() {
  local repo="$1"
  git -C "$repo" config core.sparseCheckout true
  git -C "$repo" config core.sparseCheckoutCone false
  mkdir -p "$repo/.git/info"
  if ! git -C "$repo" show "$MUSETALK_SOURCE_REVISION:$WORKER_SPARSE_SPEC" > "$repo/.git/info/sparse-checkout.new" 2>/dev/null; then
    rm -f "$repo/.git/info/sparse-checkout.new"
    git -C "$repo" config core.sparseCheckout false
    bootstrap_log "Revision has no $WORKER_SPARSE_SPEC; full (lazy) checkout"
    return 1
  fi
  mv -f "$repo/.git/info/sparse-checkout.new" "$repo/.git/info/sparse-checkout"
}

verify_worker_checkout() {
  local repo="$1" path skipped
  [[ "$(git -C "$repo" config --get core.sparseCheckout)" == true ]] || return 0
  skipped="$(git -C "$repo" ls-files -t | grep -c '^S ' || true)"
  (( skipped > 0 )) || bootstrap_die 'Sparse spec excluded nothing; refusing a full worker checkout'
  for path in "${DENIED_WORKER_PATHS[@]}"; do
    [[ ! -e "$repo/$path" ]] || bootstrap_die "Excluded path present in worker checkout: $path"
  done
  for path in "${REQUIRED_WORKER_PATHS[@]}"; do
    [[ -e "$repo/$path" ]] || bootstrap_die "Required worker path missing from sparse checkout: $path"
  done
  bootstrap_log "Sparse worker checkout: $skipped paths skipped; spec $(file_sha256 "$repo/.git/info/sparse-checkout" | cut -c1-16); $(git --version)"
}

checkout_code() {
  local current=""
  if [[ -e "$REPO_DIR" || -L "$REPO_DIR" ]]; then
    [[ -d "$REPO_DIR/.git" && ! -L "$REPO_DIR" && ! -L "$REPO_DIR/.git" ]] \
      || bootstrap_die 'Existing repo path is not a normal Git checkout; left unchanged'
    [[ "$(stat -c %u "$REPO_DIR")" == "$(id -u)" ]] || bootstrap_die 'Foreign checkout; left unchanged'
    [[ "$(git -C "$REPO_DIR" remote get-url origin)" == "$REPO_URL" ]] \
      || bootstrap_die 'Existing checkout has a different origin; left unchanged'
    git -C "$REPO_DIR" diff --quiet -- && git -C "$REPO_DIR" diff --cached --quiet -- \
      || bootstrap_die 'Tracked local edits exist; left unchanged'
    current="$(git -C "$REPO_DIR" rev-parse HEAD)"
    if [[ "$current" == "$MUSETALK_SOURCE_REVISION" ]]; then
      bootstrap_log 'Pinned checkout already present; no clone/fetch'
      return
    fi
    # A live API or ambiguous PID file is not permission for an in-place update.
    local pid_file="${PID_FILE:-/workspace/logs/musetalk/api_server_${PORT}.pid}"
    if [[ -e "$pid_file" || -L "$pid_file" ]]; then
      api_pid_file_is_stale "$pid_file" \
        || bootstrap_die 'API PID file exists; stop/inspect worker before changing code'
      bootstrap_log 'Stale API PID file from a previous container; continuing with code update'
    fi
    # A blob-filtered checkout keeps its filter for this fetch (promisor remote).
    git -C "$REPO_DIR" -c core.hooksPath=/dev/null fetch --depth 1 origin "$MUSETALK_SOURCE_REVISION"
    local sparse=0
    apply_worker_sparse_spec "$REPO_DIR" && sparse=1
    git -C "$REPO_DIR" -c core.hooksPath=/dev/null checkout --detach --no-overwrite-ignore "$MUSETALK_SOURCE_REVISION"
    if (( sparse )); then git -C "$REPO_DIR" sparse-checkout reapply; fi
    verify_worker_checkout "$REPO_DIR"
  else
    STAGE_DIR="$(mktemp -d /workspace/MuseTalk.stage.XXXXXX)"
    git -C "$STAGE_DIR" init -q
    git -C "$STAGE_DIR" remote add origin "$REPO_URL"
    git -C "$STAGE_DIR" -c core.hooksPath=/dev/null fetch --depth 1 --filter=blob:none origin "$MUSETALK_SOURCE_REVISION"
    apply_worker_sparse_spec "$STAGE_DIR" || true
    git -C "$STAGE_DIR" -c core.hooksPath=/dev/null checkout --detach "$MUSETALK_SOURCE_REVISION"
    verify_worker_checkout "$STAGE_DIR"
    [[ ! -e "$REPO_DIR" && ! -L "$REPO_DIR" ]] || bootstrap_die 'Repo path appeared during staging'
    mv -T "$STAGE_DIR" "$REPO_DIR"
    STAGE_DIR=""
  fi
  [[ "$(git -C "$REPO_DIR" rev-parse HEAD)" == "$MUSETALK_SOURCE_REVISION" ]] \
    || bootstrap_die 'Checkout revision differs from requested deployment'
}

configure_locked_boot() {
  local artifact
  export MUSETALK_DEPENDENCY_INSTALL_MODE=locked-wheelhouse-uv
  export MUSETALK_DEPENDENCY_CONTRACT="$REPO_DIR/configs/dependencies/rtx3090-cp310-cu121-screening.json"
  [[ -f "$MUSETALK_DEPENDENCY_CONTRACT" && -f "$REPO_DIR/scripts/install_locked_dependencies.py" ]] \
    || bootstrap_die 'This revision lacks the reviewed locked installer; publish code before using this template'
  artifact="$(python3 -B -c 'import json,re,sys;v=json.load(open(sys.argv[1]))["artifact_id"];assert isinstance(v,str) and re.fullmatch("[a-f0-9]{64}",v);print(v)' "$MUSETALK_DEPENDENCY_CONTRACT" 2>/dev/null)" \
    || bootstrap_die 'Reviewed dependency contract has no valid artifact identity'
  export MUSETALK_DEPENDENCY_CACHE=/workspace/.cache/musetalk-dependencies
  # Package CDNs are limited per connection (fresh 3090s measured 1-5 MB/s per
  # stream while the same hosts pulled Hugging Face at ~74 MB/s): run 8 artifact
  # workers, ranged 4-way for files >= 64 MiB, up to 16 open connections.
  export MUSETALK_DEPENDENCY_WORKERS="${MUSETALK_DEPENDENCY_WORKERS:-8}"
  export MUSETALK_DEPENDENCY_RANGE_STREAMS="${MUSETALK_DEPENDENCY_RANGE_STREAMS:-4}"
  export MUSETALK_DEPENDENCY_MAX_CONNECTIONS="${MUSETALK_DEPENDENCY_MAX_CONNECTIONS:-16}"
  # The stock image's /venv/main already holds 2.8 GB of the pinned GPU wheels
  # (torch, triton, nvidia-*), file-for-file identical; the installer verifies
  # every file against the contract-pinned seed index before reusing it.
  if [[ -d /venv/main/lib/python3.10/site-packages ]]; then
    export MUSETALK_DEPENDENCY_SEED_SITE=/venv/main/lib/python3.10/site-packages
  fi
  # The locked installer owns its venv. An inherited generic VENV_PATH (e.g.
  # the legacy musetalk_trt_stagewise venv) would be refused as foreign, so it
  # is ignored; only the explicit locked override is honoured.
  local locked_venv="${MUSETALK_LOCKED_VENV_PATH:-/workspace/.venvs/musetalk_locked_$artifact}"
  if [[ -n "${VENV_PATH:-}" && "$VENV_PATH" != "$locked_venv" ]]; then
    bootstrap_log "Ignoring inherited VENV_PATH; locked venv is $locked_venv"
  fi
  export VENV_PATH="$locked_venv"
  export SETUP_CLEAN=0 SETUP_FULL_STACK=1 SETUP_MATRIX=cu121 AUTO_SETUP=1
  export DOWNLOAD_SYNCNET_WEIGHTS=0  # Training/offline-metrics weight only; 1.4 GB per fresh worker.
  export WORKSPACE=/workspace
  export PROFILE=throughput_record PORT
  export MUSETALK_RECIPE="${MUSETALK_RECIPE:-r5}"
  # The compiled-TAESD self-test timing only matters for the fast recipe.
  if [[ "$MUSETALK_RECIPE" == r5 ]]; then export MUSETALK_SELFTEST_SKIP_COMPILE=1; fi
  # Phase 7's import smoke and the server's strict start verification cover these.
  export ONSTART_POST_VALIDATE_IMPORTS="${ONSTART_POST_VALIDATE_IMPORTS:-0}"
  export STARTUP_TIMEOUT_SECONDS="${STARTUP_TIMEOUT_SECONDS:-1800}"
  # Keep read URLs in a pipe, not a template literal, argv or a temporary file.
  if [[ -n "${MUSETALK_DEPENDENCY_TICKETS_JSON:-}" ]]; then
    export -n MUSETALK_DEPENDENCY_TICKETS_JSON
    exec 8< <(printf '%s' "$MUSETALK_DEPENDENCY_TICKETS_JSON")
    unset MUSETALK_DEPENDENCY_TICKETS_JSON
  else
    [[ -f "$VENV_PATH/.musetalk-locked-dependencies.json" \
       && -f "$MUSETALK_DEPENDENCY_CACHE/$artifact/manifest.json" ]] \
      || bootstrap_die 'Fresh locked installation requires EC2-issued dependency tickets; AWS secret reference alone is insufficient'
    exec 8</dev/null
    bootstrap_log 'No fresh ticket supplied: only a fully validated existing environment may be reused'
  fi
  export MUSETALK_DEPENDENCY_TICKET_FD=8 MUSETALK_BOOTSTRAP_TICKET_FD=8
}

# The ~1 GB r5 engine bundle does not depend on the venv: fetch it in the
# background from an EC2-presigned URL while dependencies install. Onstart's
# restore still verifies the pinned sha256 and falls back to S3 on any failure.
# Extract + verify + stamp the prefetched bundle in the background too (stdlib-only
# tool, image interpreter). Onstart's restore then takes its --skip-if-verified path
# (seconds) instead of copying/extracting 1 GB on the critical path. Any failure
# leaves the verified archive for onstart's normal restore.
restore_prefetched_bundle() {
  local dir="$1" sha="$2" desc sidecar
  [[ -f "$REPO_DIR/scripts/trt_artifact_bundle.py" ]] || return 0
  desc="$(grep -l "\"sha256\": \"$sha\"" "$REPO_DIR"/configs/trt_bundles/*.json 2>/dev/null | head -n 1)"
  [[ -n "$desc" ]] || return 0
  sidecar="$(python3 -B -c 'import json,sys;print(json.load(open(sys.argv[1]))["sidecar_dir"])' "$desc" 2>/dev/null)" || return 0
  [[ -n "$sidecar" ]] || return 0
  local started=$SECONDS
  if python3 -B "$REPO_DIR/scripts/trt_artifact_bundle.py" --repo-root "$REPO_DIR" --strict --sidecar-dir "$sidecar" \
       restore --uri "$dir/$sha.tar.gz" --expected-sha256 "$sha" --stage-dir "$REPO_DIR/tmp/trt_artifact_stage" \
       >"$dir/restore.log" 2>&1; then
    rm -f "$dir/$sha.tar.gz"
    bootstrap_log "r5 bundle restored in the background ($((SECONDS - started))s)"
  else
    bootstrap_log "Background r5 restore failed; onstart restores from the verified archive"
  fi
}

# Download priority on fresh workers. The dependency wheels gate the whole boot
# (Phase 3), while the weights (needed at Phase 4) and the r5 bundle (needed at
# server start) have slack. Started together, the many-connection weight download
# starved the wheels: on c5-a (2026-10-18) acquisition moved 40 MB in its first
# 35 s. The prefetches below therefore wait for this gate, which
# install_locked_dependencies.py removes once acquisition ends (even on failure);
# the wait is bounded so a stuck installer never blocks them.
DOWNLOAD_GATE="${MUSETALK_DOWNLOAD_GATE_PATH:-/workspace/.mtboot/deps-gate}"
arm_download_gate() {
  rm -f "$DOWNLOAD_GATE"
  [[ "${MUSETALK_DOWNLOAD_GATE_ENABLED:-1}" != 0 ]] || return 0
  [[ -f "$REPO_DIR/.runtime/install_state.json" ]] && return 0  # Installed before: nothing to prioritize.
  mkdir -p "${DOWNLOAD_GATE%/*}"
  date -u +%Y-%m-%dT%H:%M:%SZ > "$DOWNLOAD_GATE"
  export MUSETALK_DOWNLOAD_GATE="$DOWNLOAD_GATE"
  bootstrap_log 'Download gate armed: weight and r5 prefetches wait for dependency acquisition'
}

wait_for_download_gate() {
  local what="$1" waited=0 limit="${MUSETALK_DOWNLOAD_GATE_WAIT_SECONDS:-180}"
  [[ -e "$DOWNLOAD_GATE" ]] || return 0
  while [[ -e "$DOWNLOAD_GATE" ]] && (( waited < limit )); do sleep 1; waited=$(( waited + 1 )); done
  bootstrap_log "$what: download gate $([[ -e "$DOWNLOAD_GATE" ]] && echo "still closed after ${limit}s; proceeding" || echo "opened after ${waited}s")"
}

start_r5_prefetch() {
  [[ -n "${MUSETALK_R5_PREFETCH_JSON:-}" ]] || return 0
  local dir="${MUSETALK_R5_PREFETCH_DIR:-/workspace/.cache/r5-prefetch}" url sha size fields
  fields="$(python3 -B -c 'import json,os,re
d=json.loads(os.environ["MUSETALK_R5_PREFETCH_JSON"])
assert re.fullmatch("[a-f0-9]{64}",d["sha256"]) and d["url"].startswith("https://") and int(d["bytes"])>0
print(d["sha256"],int(d["bytes"]));print(d["url"])' 2>/dev/null)" || { bootstrap_log 'Ignoring malformed r5 prefetch ticket'; return 0; }
  export -n MUSETALK_R5_PREFETCH_JSON
  unset MUSETALK_R5_PREFETCH_JSON
  read -r sha size <<< "$(head -n 1 <<< "$fields")"
  url="$(sed -n 2p <<< "$fields")"
  export MUSETALK_R5_PREFETCH_DIR="$dir"
  mkdir -p "$dir"
  if compgen -G "$REPO_DIR/.runtime/trt_artifacts/*/.musetalk_trt_artifact_restored.json" >/dev/null; then
    bootstrap_log 'r5 bundle already restored on this disk; no prefetch'
    return 0
  fi
  touch "$dir/$sha.pending"
  if [[ -f "$dir/$sha.ok" && -f "$dir/$sha.tar.gz" ]]; then
    # Verified archive from an earlier boot: just restore it in the background.
    ( restore_prefetched_bundle "$dir" "$sha"; rm -f "$dir/$sha.pending" ) 8<&- 9<&- &
    return 0
  fi
  bootstrap_log "r5 bundle prefetch queued ($size bytes)"
  (
    wait_for_download_gate 'r5 bundle prefetch'
    started=$SECONDS
    # Multi-connection ranged fetch (8 streams; single-connection S3 measured
    # 1.36 MB/s on a slow host). URL on stdin only; publishes after sha256.
    if [[ -f "$REPO_DIR/scripts/fetch_ranged_url.py" ]] \
       && printf '%s\n' "$url" | python3 -B "$REPO_DIR/scripts/fetch_ranged_url.py" --sha256 "$sha" --bytes "$size" \
            --out "$dir/$sha.tar.gz" --streams 8; then
      touch "$dir/$sha.ok"
      restore_prefetched_bundle "$dir" "$sha"
    elif printf 'url = "%s"\n' "$url" | curl -fsS --retry 3 --retry-delay 2 --max-time 1800 -K - -o "$dir/$sha.tar.gz.part" \
       && [[ "$(wc -c < "$dir/$sha.tar.gz.part" | tr -d " ")" == "$size" ]] \
       && [[ "$(file_sha256 "$dir/$sha.tar.gz.part")" == "$sha" ]]; then
      mv -f "$dir/$sha.tar.gz.part" "$dir/$sha.tar.gz"
      touch "$dir/$sha.ok"
      restore_prefetched_bundle "$dir" "$sha"
      bootstrap_log "r5 bundle prefetch done ($((SECONDS - started))s)"
    else
      rm -f "$dir/$sha.tar.gz.part"
      bootstrap_log "r5 bundle prefetch failed after $((SECONDS - started))s; onstart falls back to S3"
    fi
    rm -f "$dir/$sha.pending"
  ) 8<&- 9<&- &
}

# The image's own interpreter already has the pinned huggingface_hub, so the big
# Hugging Face weights start downloading now, overlapping dependency install.
# hf_xet goes to a --target dir (hash-pinned); the image venv is never modified.
start_weights_prefetch() {
  [[ "${MUSETALK_WEIGHTS_PREFETCH:-1}" != 0 ]] || return 0
  local py=/venv/main/bin/python dir=/workspace/.mtboot/weights-prefetch tools=/workspace/.mtboot/hftools
  [[ -x "$py" && -x /venv/main/bin/huggingface-cli ]] && "$py" -c 'import huggingface_hub' 2>/dev/null \
    || { bootstrap_log 'Image has no huggingface_hub; weights download in installer Phase 4'; return 0; }
  [[ -f "$REPO_DIR/.runtime/install_state.json" ]] && return 0  # Installed before; Phase 4 re-validates only.
  mkdir -p "$dir"
  rm -f "$dir/rc"
  touch "$dir/pending"
  export MUSETALK_WEIGHTS_PREFETCH_DIR="$dir"
  bootstrap_log 'Weight prefetch queued (image interpreter)'
  (
    if [[ ! -d "$tools/hf_xet" ]]; then
      "$py" -m pip install --disable-pip-version-check --no-cache-dir --no-deps --require-hashes \
        --target "$tools" -r "$REPO_DIR/configs/dependencies/weights-prefetch-tools.txt" >"$dir/tools.log" 2>&1 \
        || bootstrap_log 'hf_xet unavailable for prefetch; plain HTTP downloads'
    fi
    wait_for_download_gate 'Weight prefetch'
    started=$SECONDS
    cd "$REPO_DIR"
    rc=0
    PATH="/venv/main/bin:$PATH" PYTHONPATH="$tools" DOWNLOAD_WEIGHTS_PREFETCH_ONLY=1 \
      DOWNLOAD_MUSETALK_V1_WEIGHTS=0 DOWNLOAD_AVATAR_PREP_WEIGHTS=1 DOWNLOAD_TAESD_WEIGHTS=1 \
      DOWNLOAD_KOKORO_WEIGHTS=0 DOWNLOAD_SYNCNET_WEIGHTS=0 \
      bash download_weights.sh >"$dir/download.log" 2>&1 || rc=$?
    echo "$rc" > "$dir/rc"
    rm -f "$dir/pending"
    bootstrap_log "Weight prefetch finished rc=$rc ($((SECONDS - started))s)"
  ) 8<&- 9<&- &
}

# System packages (python3.10-venv, ffmpeg, coturn, ...) install in the
# background while wheels download; the locked installer waits before it
# creates the venv. Fresh workers only.
start_apt_prefetch() {
  [[ "$(id -u)" == 0 ]] && command -v apt-get >/dev/null 2>&1 || return 0
  [[ -f "$REPO_DIR/.runtime/install_state.json" ]] && return 0
  local dir=/workspace/.mtboot/apt list="$REPO_DIR/scripts/lib/apt_packages.txt"
  [[ -f "$list" ]] || return 0
  mkdir -p "$dir"
  rm -f "$dir/rc"
  touch "$dir/pending"
  export MUSETALK_APT_PREFETCH_DIR="$dir"
  bootstrap_log 'Background apt started'
  (
    started=$SECONDS rc=0
    mapfile -t packages < <(grep -v -E '^\s*(#|$)' "$list")
    { DEBIAN_FRONTEND=noninteractive apt-get update -y \
        && DEBIAN_FRONTEND=noninteractive apt-get install -y "${packages[@]}"; } >"$dir/apt.log" 2>&1 || rc=$?
    echo "$rc" > "$dir/rc"
    rm -f "$dir/pending"
    bootstrap_log "Background apt finished rc=$rc ($((SECONDS - started))s)"
  ) 8<&- 9<&- &
}

configure_runtime_auth() {
  # EC2 role credentials do not automatically exist on Vast. The launcher must
  # supply a reviewed worker identity/runtime payload, never the publisher role.
  [[ -n "${AWS_ACCESS_KEY_ID:-}" && -n "${AWS_SECRET_ACCESS_KEY:-}" ]] \
    || bootstrap_die 'Worker AWS read identity missing; EC2 must inject runtime access separately'
  export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
  case "${MUSETALK_RUNTIME_CONFIG_SOURCE:-secrets-manager}" in
    secrets-manager)
      : "${MUSETALK_AWS_SECRET_ID:?Supply the runtime secret reference through launch environment}"
      export MUSETALK_AWS_SECRET_REGION="${MUSETALK_AWS_SECRET_REGION:-us-east-1}"
      export MUSETALK_SECRETS_STRICT=true MUSETALK_SECRETS_VERIFY_S3=1
      ;;
    injected)
      [[ -n "${AVATAR_S3_BUCKET:-}" && -n "${TRT_ARTIFACT_S3_BUCKET:-}" ]] \
        || bootstrap_die 'EC2-injected runtime S3 configuration missing'
      export MUSETALK_AWS_SECRET_ID=""
      ;;
    *) bootstrap_die 'Unknown runtime configuration source' ;;
  esac
}

bootstrap_cleanup() {
  # Only the exact mktemp staging checkout from this invocation is disposable.
  if [[ -n "${STAGE_DIR:-}" && "$STAGE_DIR" == /workspace/MuseTalk.stage.* \
        && "${STAGE_DIR%/*}" == /workspace && -d "$STAGE_DIR" && ! -L "$STAGE_DIR" ]]; then
    rm -rf -- "$STAGE_DIR"
  fi
}

main() {
  [[ "$(uname -s)" == Linux ]] || bootstrap_die 'This bootstrap runs on a Linux Vast worker, not macOS'
  umask 077
  mkdir -p /workspace
  exec > >(tee -a /workspace/bootstrap.log) 2>&1
  bootstrap_log "start boot_id=$(cat /proc/sys/kernel/random/boot_id 2>/dev/null || echo unknown)"
  export MUSETALK_STARTUP_TIMINGS_FILE="${MUSETALK_STARTUP_TIMINGS_FILE:-/workspace/.mtboot/startup_timings}"
  mkdir -p "$(dirname "$MUSETALK_STARTUP_TIMINGS_FILE")"
  : > "$MUSETALK_STARTUP_TIMINGS_FILE"
  stamp_stage bootstrap_start
  trap 'bootstrap_log "Failed at line $LINENO (exit $?)"' ERR
  STAGE_DIR="";trap bootstrap_cleanup EXIT
  command -v flock >/dev/null 2>&1 || bootstrap_die 'Stock template must provide flock'
  exec 9>/tmp/musetalk-onstart.lock
  flock -n 9 || { bootstrap_log 'Another bootstrap owns the lock; exiting'; return 0; }
  : "${MUSETALK_SOURCE_REVISION:?EC2 must supply a reviewed full Git commit SHA}"
  [[ "$MUSETALK_SOURCE_REVISION" =~ ^[a-f0-9]{40}$ ]] || bootstrap_die 'Expected a full lowercase Git commit SHA'
  PORT="${PORT:-8000}"
  [[ "$PORT" =~ ^[0-9]{1,5}$ ]] && (( 10#$PORT > 0 && 10#$PORT <= 65535 )) || bootstrap_die 'Invalid API port'
  REPO_DIR=/workspace/MuseTalk
  REPO_URL=https://github.com/AhmadAFS1/MuseTalk.git
  export GIT_TERMINAL_PROMPT=0
  command -v python3 >/dev/null 2>&1 || bootstrap_die 'Stock PyTorch template must provide host python3'
  configure_runtime_auth
  if ! command -v git >/dev/null 2>&1; then
    apt-get update -y
    DEBIAN_FRONTEND=noninteractive apt-get install -y git ca-certificates
  fi
  local started="$SECONDS"
  checkout_code
  bootstrap_log "Code ready ($((SECONDS-started))s); revision=$MUSETALK_SOURCE_REVISION"
  stamp_stage code_ready
  start_apt_prefetch
  arm_download_gate
  start_r5_prefetch
  start_weights_prefetch
  configure_locked_boot
  cd "$REPO_DIR"
  # Parent retains the bootstrap lock while it waits; daemons must not inherit
  # it and keep future on-start attempts locked out after this shell exits.
  bash scripts/vast_onstart.sh 9<&-
  exec 8<&-
  bootstrap_log "Complete ($((SECONDS-started))s after checkout phase began)"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then main "$@"; fi
