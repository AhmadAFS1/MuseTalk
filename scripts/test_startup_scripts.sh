#!/usr/bin/env bash
# CPU-only tests for the MuseTalk launch chain (startup rework 2026-09-28, docs/STARTUP.md).
#
#   bash scripts/test_startup_scripts.sh [-k FILTER] [--keep]
#
# No GPU, no torch, no real server, no network, no writes outside a temp dir. Every scenario
# runs inside a throw-away repo skeleton ($TMPDIR/musetalk_startup_test.*) that holds copies
# of the launch-chain scripts, stub api_server.py / legacy launcher / installer / engine store
# / ctl, and a fake venv whose bin/python is a torch-free system python.
#
# Groups:
#   syntax      bash -n on every touched shell script, ast-parse of the Python components
#   parser      env-file parser + only-if-unset layering (scripts/lib/musetalk_env_layers.sh)
#   launcher    run_musetalk_server.sh with a STUB resolver: precedence caller > overrides >
#               resolved, pass-through of unknown knobs, fast300 / legacy dispatch (dry),
#               --print-env / --validate-only side effects, VP8 preflight, lever validation
#   ctl         vast_server_ctl.sh start/verify/status/stop against a stub server process
#               (health via file:// URLs, nothing listens on a socket)
#   onstart     vast_onstart.sh phases + VAST_ONSTART COMPLETE/FAILED markers with stubs
#   relay       run_webrtc_relay_api_server.sh final exec + layering
#   shim        setup_musetalk.sh flag translation
#   integration the REAL resolver (scripts/musetalk_host_profile.py) with injected host facts
#               and a fake venv: eager-vs-trt selection, bucket coupling, verify-log exit codes
#               (skipped with a notice while the resolver is not in the tree)

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
FILTER=""
KEEP=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    -k) FILTER="$2"; shift 2 ;;
    --keep) KEEP=1; shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown option $1" >&2; exit 2 ;;
  esac
done

TEST_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/musetalk_startup_test.XXXXXX")"
cleanup() {
  local pid
  for pid in $(cat "$TEST_ROOT"/pids 2>/dev/null || true); do
    kill "$pid" >/dev/null 2>&1 || true
  done
  if (( KEEP )); then
    echo "kept $TEST_ROOT"
  else
    rm -rf "$TEST_ROOT"
  fi
}
trap cleanup EXIT

PASS=0
FAIL=0
SKIP=0
declare -a FAILED_NAMES=()
CURRENT=""
CASE_FAILED=0

# ---------------------------------------------------------------- torch-free python
pick_python() {
  local candidate
  for candidate in "${MUSETALK_TEST_PYTHON:-}" /usr/bin/python3.10 /usr/bin/python3 python3; do
    [[ -n "$candidate" ]] || continue
    command -v "$candidate" >/dev/null 2>&1 || continue
    if "$candidate" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2>/dev/null; then
      command -v "$candidate"
      return 0
    fi
  done
  return 1
}
TEST_PY="$(pick_python)" || { echo "no python >= 3.8 found" >&2; exit 2; }
if "$TEST_PY" -c 'import importlib.util as u, sys; sys.exit(0 if u.find_spec("torch") else 1)' 2>/dev/null; then
  echo "NOTE: $TEST_PY can import torch; the tests never do (stubs are stdlib-only)" >&2
fi
mkdir -p "$TEST_ROOT/bin"
ln -s "$TEST_PY" "$TEST_ROOT/bin/python3"
cat > "$TEST_ROOT/bin/nvidia-smi" <<'EOF'
#!/usr/bin/env bash
# fake nvidia-smi for the launch-chain tests
case "$*" in
  *query-gpu*) echo "0, NVIDIA GeForce RTX 4070 SUPER, 8.9, 12282, 500, 220.00, 220.00, 595.84" ;;
  *query-compute-apps*) ;;
  *) echo "fake nvidia-smi" ;;
esac
EOF
chmod +x "$TEST_ROOT/bin/nvidia-smi"
SAFE_PATH="$TEST_ROOT/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"

# ---------------------------------------------------------------- harness
begin() {
  CURRENT="$1"
  CASE_FAILED=0
}
selected() {
  [[ -z "$FILTER" || "$1" == *"$FILTER"* ]]
}
fail() {
  CASE_FAILED=1
  printf '    assertion failed: %s\n' "$*"
}
end() {
  if (( CASE_FAILED )); then
    FAIL=$((FAIL + 1))
    FAILED_NAMES+=("$CURRENT")
    printf 'FAIL %s\n' "$CURRENT"
  else
    PASS=$((PASS + 1))
    printf 'PASS %s\n' "$CURRENT"
  fi
}
skip() {
  SKIP=$((SKIP + 1))
  printf 'SKIP %s (%s)\n' "$1" "$2"
}
assert_eq() {
  # assert_eq ACTUAL EXPECTED WHAT
  if [[ "$1" != "$2" ]]; then
    fail "$3: expected [$2] got [$1]"
  fi
}
assert_rc() {
  # assert_rc ACTUAL EXPECTED WHAT
  if [[ "$2" == "nonzero" ]]; then
    [[ "$1" != "0" ]] || fail "$3: expected a non-zero exit, got 0"
  elif [[ "$1" != "$2" ]]; then
    fail "$3: expected exit $2 got $1"
  fi
}
assert_file_contains() {
  if [[ ! -f "$1" ]] || ! grep -Eq -- "$2" "$1"; then
    fail "$3: /$2/ not in ${1##*/}"
    [[ -f "$1" ]] && tail -n 12 "$1" | sed 's/^/      | /'
  fi
}
assert_file_lacks() {
  if [[ -f "$1" ]] && grep -Eq -- "$2" "$1"; then
    fail "$3: /$2/ unexpectedly in ${1##*/}"
  fi
}
assert_exists() {
  [[ -e "$1" ]] || fail "$2: missing $1"
}
assert_missing() {
  [[ ! -e "$1" ]] || fail "$2: unexpected $1"
}

# JSON env dump helpers (stub api_server / stubs write {"argv":[...], "env":{...}})
dump_get() {
  "$TEST_PY" - "$1" "$2" <<'PY'
import json, sys
try:
    data = json.load(open(sys.argv[1]))
except Exception:
    print("<nodump>")
    sys.exit(0)
env = data.get("env", {})
print(env.get(sys.argv[2], "<unset>"))
PY
}
dump_argv() {
  "$TEST_PY" - "$1" <<'PY'
import json, sys
try:
    print(" ".join(json.load(open(sys.argv[1])).get("argv", [])))
except Exception:
    print("<nodump>")
PY
}
json_get() {
  # json_get FILE dotted.path
  "$TEST_PY" - "$1" "$2" <<'PY'
import json, sys
try:
    cur = json.load(open(sys.argv[1]))
    for part in sys.argv[2].split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    print(cur if not isinstance(cur, (dict, list)) else json.dumps(cur, sort_keys=True))
except Exception:
    print("<missing>")
PY
}
# --print-env output: KEY='value'  # source
pe_get() {
  "$TEST_PY" - "$1" "$2" "${3:-value}" <<'PY'
import shlex, sys
want, field = sys.argv[2], sys.argv[3]
for line in open(sys.argv[1]):
    line = line.rstrip("\n")
    if not line.startswith(want + "="):
        continue
    body, _, source = line[len(want) + 1:].partition("  # ")
    try:
        value = shlex.split(body)[0] if body.strip() else ""
    except ValueError:
        value = body
    print(value if field == "value" else source)
    sys.exit(0)
print("<unset>")
PY
}

# ---------------------------------------------------------------- sandbox builders
write_fake_venv() {
  local venv="$1" sp
  sp="$venv/lib/python3.10/site-packages"
  mkdir -p "$venv/bin" "$sp"
  ln -sf "$TEST_PY" "$venv/bin/python"
  local pkg
  for pkg in "torch 2.5.1+cu121" "torchvision 0.20.1+cu121" "tensorrt 10.3.0" "tensorrt_cu12 10.3.0" \
             "torch_tensorrt 2.5.0" "triton 3.1.0" "aiortc 1.14.0" "av 16.1.0" "cffi 2.1.1" \
             "diffusers 0.30.2" "fastapi 0.135.1"; do
    set -- $pkg
    mkdir -p "$sp/$1-$2.dist-info"
    printf 'Metadata-Version: 2.1\nName: %s\nVersion: %s\n' "$1" "$2" > "$sp/$1-$2.dist-info/METADATA"
  done
}
FAKE_VENV="$TEST_ROOT/venv"
write_fake_venv "$FAKE_VENV"

write_facts() {
  # write_facts FILE [gpus_json] [ram_available_mb]
  local file="$1" gpus="${2:-}" avail="${3:-20000}"
  if [[ -z "$gpus" ]]; then
    gpus='[{"index": 0, "name": "NVIDIA GeForce RTX 4070 SUPER", "compute_capability": "8.9", "memory_total_mib": 12282, "memory_used_mib": 500, "power_limit_w": 220.0, "power_default_limit_w": 220.0, "driver_version": "595.84"}]'
  fi
  cat > "$file" <<EOF
{
 "gpus": $gpus,
 "selected_gpu_index": 0,
 "cpu": {"nproc": 32, "affinity": 32, "cgroup_quota_cpus": null, "effective": 32},
 "ram": {"mem_total_mb": 30000, "mem_available_mb": $avail, "cgroup_limit_mb": null,
         "effective_total_mb": 30000, "effective_available_mb": $avail},
 "disk_free_gb": 50.0,
 "machine": "x86_64",
 "venv": {"python_version": "3.10.12", "torch": "2.5.1+cu121", "torch_cuda_tag": "cu121",
          "tensorrt": "10.3.0", "torch_tensorrt": "2.5.0", "triton": "3.1.0",
          "aiortc": "1.14.0", "av": "16.1.0", "cffi": "2.1.1"}
}
EOF
}

write_stub_api_server() {
  cat > "$1/api_server.py" <<'PY'
"""Stub api_server for the launch-chain tests: dump env, optionally emulate a server."""
import json, os, signal, sys, time

dump = os.environ.get("MT_TEST_ENV_DUMP")
if dump:
    with open(dump, "w") as fh:
        json.dump({"argv": sys.argv[1:], "env": dict(os.environ), "cwd": os.getcwd()}, fh)
if os.environ.get("MT_TEST_SERVER_MODE", "exit") != "serve":
    sys.exit(0)
vae = os.environ.get("MT_TEST_SERVER_VAE", "taesd")
unet = os.environ.get("MT_TEST_SERVER_UNET", "")
print(f"Starting MuseTalk API Server pid={os.getpid()}...", flush=True)
print("VAE decode backend: PyTorch" if vae == "pytorch" else f"VAE decode backend active: {vae}", flush=True)
print("UNet backend: PyTorch" if unet in ("", "eager", "pytorch") else f"UNet backend active: {unet}", flush=True)
health = os.environ["MT_TEST_HEALTH_FILE"]

def stop(*_):
    try:
        os.remove(health)
    except FileNotFoundError:
        pass
    sys.exit(0)

signal.signal(signal.SIGTERM, stop)
signal.signal(signal.SIGINT, stop)
with open(health, "w") as fh:
    fh.write("ok")
deadline = time.time() + float(os.environ.get("MT_TEST_SERVER_MAX_S", "90"))
while time.time() < deadline:
    time.sleep(0.2)
stop()
PY
}

write_stub_resolver() {
  # A stand-in for scripts/musetalk_host_profile.py that follows the interface contract.
  cat > "$1/scripts/musetalk_host_profile.py" <<'PY'
"""STUB resolver for the launch-chain tests (not the real component A)."""
import argparse, json, os, re, sys, time

ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
r = sub.add_parser("resolve")
for flag in ("--repo-root", "--venv", "--out", "--report"):
    r.add_argument(flag, required=True)
r.add_argument("--recipe", default="fast")
v = sub.add_parser("verify-log")
v.add_argument("--log", required=True)
v.add_argument("--offset", type=int, default=0)
v.add_argument("--expect-vae", required=True)
v.add_argument("--expect-unet", default="any")
v.add_argument("--timeout", type=float, default=30)
args = ap.parse_args()
env = os.environ

if args.cmd == "resolve":
    dump = env.get("MT_TEST_RESOLVER_DUMP")
    if dump:
        with open(dump, "w") as fh:
            json.dump({"argv": sys.argv[1:], "env": dict(env)}, fh)
    forced = int(env.get("MT_TEST_RESOLVER_EXIT", "0") or 0)
    if forced:
        json.dump({"errors": ["stub: forced failure"], "warnings": [], "decisions": [], "emitted": {}},
                  open(args.report, "w"))
        print("stub resolver: forced failure", file=sys.stderr)
        sys.exit(forced)
    buckets = env.get("HLS_SCHEDULER_FIXED_BATCH_SIZES", "8")
    emitted = {
        "MUSETALK_RECIPE": args.recipe,
        "MUSETALK_VAE_BACKEND": "taesd",
        "MUSETALK_TAESD_COMPILE": "1",
        "MUSETALK_TRT_FALLBACK": "0",
        "MUSETALK_TRT_ENABLED": "0",
        "MUSETALK_UNET_BACKEND": "eager",
        "MUSETALK_TRT_UNET_ENABLED": "0",
        "HLS_SCHEDULER_FIXED_BATCH_SIZES": buckets,
        "MUSETALK_TAESD_WARMUP_BATCHES": buckets,
        "MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES": buckets,
        "HLS_SCHEDULER_MAX_BATCH": str(max(int(b) for b in buckets.split(","))),
        "WEBRTC_VP8_ENCODER": "pyav",
        "WEBRTC_SYNC_MODE": "strict_fifo",
        "MUSETALK_TEST_LAYER_A": "resolved",
        "MUSETALK_TEST_LAYER_B": "resolved",
        "MUSETALK_TEST_LAYER_C": "resolved",
        "MUSETALK_TEST_LAYER_D": "resolved",
        "MUSETALK_TEST_QUOTED": "it's a 'quoted' $HOME value",
    }
    for key in list(emitted):
        if key in env:
            emitted[key] = env[key]  # the caller's value wins and is reported
    for key, value in json.loads(env.get("MT_TEST_RESOLVER_EXTRA", "{}")).items():
        emitted[key] = env.get(key, value)
    with open(args.out, "w") as fh:
        fh.write("# stub resolved env\n")
        for key, value in emitted.items():
            fh.write(f"{key}='" + value.replace("'", "'\\''") + "'\n")
    json.dump({"facts": {}, "decisions": [], "warnings": [], "errors": [], "emitted": emitted},
              open(args.report, "w"), indent=1)
    print(f"stub resolver: recipe={args.recipe}", file=sys.stderr)
    sys.exit(0)

# verify-log
if env.get("MT_TEST_VERIFY_STRICT_CHOICES") == "1" and (
        args.expect_vae != "taesd" or args.expect_unet not in ("trt", "eager", "any")):
    print("stub verify-log: unsupported choice", file=sys.stderr)
    sys.exit(2)
calls = env.get("MT_TEST_VERIFY_CALLS")
if calls:
    with open(calls, "a") as fh:
        fh.write(f"{args.expect_vae} {args.expect_unet} {args.offset}\n")
deadline = time.time() + args.timeout
vae = unet = None
while True:
    with open(args.log, "rb") as fh:
        fh.seek(args.offset)
        text = fh.read().decode("utf-8", "replace")
    m = re.search(r"VAE decode backend active: (\S+)", text)
    vae = m.group(1) if m else ("pytorch" if "VAE decode backend: PyTorch" in text else None)
    m = re.search(r"UNet backend active: (\S+)", text)
    if m:
        unet = {"tensorrt_unet": "trt", "tensorrt_unet_multi": "trt",
                "tensorrt_unet_stagewise": "trt_stagewise"}.get(m.group(1), m.group(1))
    elif "UNet backend: PyTorch" in text:
        unet = "eager"
    if (vae and unet) or time.time() >= deadline:
        break
    time.sleep(0.2)
if not (vae and unet):
    print(f"stub verify-log: not found (vae={vae} unet={unet})")
    sys.exit(3)
ok = args.expect_vae in ("any", vae) and args.expect_unet in ("any", unet)
print(f"stub verify-log: found vae={vae} unet={unet} -> {'match' if ok else 'MISMATCH'}")
sys.exit(0 if ok else 1)
PY
}

write_stub_legacy() {
  cat > "$1/scripts/run_trt_stagewise_server.sh" <<'EOF'
#!/usr/bin/env bash
# stub legacy launcher: record args + env, optionally act as the server
"$(dirname "$0")/../stub_record.py" "${MT_TEST_LEGACY_DUMP:-/dev/null}" "$@"
if [[ "${MT_TEST_SERVER_MODE:-exit}" == serve ]]; then
  MT_TEST_SERVER_VAE="${MT_TEST_SERVER_VAE:-tensorrt_stagewise_int8_mixed}" \
  MT_TEST_SERVER_UNET="${MT_TEST_SERVER_UNET:-tensorrt_unet_multi}" \
  MT_TEST_ENV_DUMP= exec python3 "$(dirname "$0")/../api_server.py"
fi
EOF
  cat > "$1/stub_record.py" <<'PY'
#!/usr/bin/env python3
import json, os, sys
path = sys.argv[1]
if path and path != "/dev/null":
    mode = "a" if os.environ.get("MT_TEST_RECORD_APPEND") == "1" else "w"
    with open(path, mode) as fh:
        json.dump({"argv": sys.argv[2:], "env": dict(os.environ), "cwd": os.getcwd()}, fh)
        fh.write("\n")
PY
  chmod +x "$1/stub_record.py" "$1/scripts/run_trt_stagewise_server.sh"
}

make_repo() {
  # make_repo NAME [real] -> prints the path. "real" copies the real resolver + engine keys.
  local repo="$TEST_ROOT/$1" mode="${2:-stub}" f
  mkdir -p "$repo/scripts/lib" "$repo/configs/recipes" "$repo/.runtime" "$repo/models"
  for f in run_musetalk_server.sh run_webrtc_relay_api_server.sh vast_server_ctl.sh vast_onstart.sh \
           run_turnserver_tcp_relay.sh lib/musetalk_env_layers.sh; do
    cp "$REAL_REPO/scripts/$f" "$repo/scripts/$f"
  done
  cp "$REAL_REPO/setup_musetalk.sh" "$repo/setup_musetalk.sh"
  cp "$REAL_REPO/scripts/setup_musetalk.sh" "$repo/scripts/setup_musetalk.sh"
  cp "$REAL_REPO/configs/recipes/fast300.env" "$repo/configs/recipes/fast300.env"
  : > "$repo/scripts/__init__.py"
  write_stub_api_server "$repo"
  write_stub_legacy "$repo"
  if [[ "$mode" == real ]]; then
    for f in musetalk_host_profile.py musetalk_engine_keys.py; do
      [[ -f "$REAL_REPO/scripts/$f" ]] && cp "$REAL_REPO/scripts/$f" "$repo/scripts/$f"
    done
  else
    write_stub_resolver "$repo"
  fi
  printf '%s' "$repo"
}

# Environment for one launcher run: a clean env (env -i) so this shell's exports never leak in.
base_env() {
  # base_env REPO -> prints NAME=VALUE words for env -i
  local repo="$1"
  printf '%s\n' \
    "PATH=$SAFE_PATH" "HOME=$TEST_ROOT/home" "TMPDIR=$TEST_ROOT" "LANG=C.UTF-8" \
    "WORKSPACE=$TEST_ROOT" "MUSETALK_ENV_OVERRIDES_FILE=$repo/.runtime/musetalk_overrides.env" \
    "MUSETALK_HOST_FACTS_JSON=$TEST_ROOT/facts.json" "MUSETALK_NVIDIA_SMI=$TEST_ROOT/bin/nvidia-smi" \
    "MUSETALK_UNET_ENGINE_STORE=$TEST_ROOT/empty_store" "MT_TEST_ENV_DUMP=$TEST_ROOT/api_env.json" \
    "MT_TEST_RESOLVER_DUMP=$TEST_ROOT/resolver_env.json" "MT_TEST_LEGACY_DUMP=$TEST_ROOT/legacy_env.json"
}
mkdir -p "$TEST_ROOT/home" "$TEST_ROOT/empty_store"
write_facts "$TEST_ROOT/facts.json"

run_launcher() {
  # run_launcher REPO OUTFILE [NAME=VALUE ...] -- [launcher args]; returns the exit code
  local repo="$1" out="$2" rc=0
  shift 2
  local -a extra=()
  while [[ $# -gt 0 && "$1" != "--" ]]; do
    extra+=("$1")
    shift
  done
  [[ "${1:-}" == "--" ]] && shift
  rm -f "$TEST_ROOT/api_env.json" "$TEST_ROOT/resolver_env.json" "$TEST_ROOT/legacy_env.json"
  local -a envv=()
  mapfile -t envv < <(base_env "$repo")
  env -i "${envv[@]}" "${extra[@]}" timeout 120 bash "$repo/scripts/run_musetalk_server.sh" \
    --host 127.0.0.1 --port 18001 --venv-path "$FAKE_VENV" --repo-root "$repo" "$@" \
    > "$out" 2> "$out.err" || rc=$?
  cat "$out.err" >> "$out.all" 2>/dev/null || true
  return "$rc"
}

# ================================================================ syntax
if selected syntax; then
  begin "syntax: bash -n on the launch chain"
  for f in scripts/run_musetalk_server.sh scripts/lib/musetalk_env_layers.sh scripts/vast_server_ctl.sh \
           scripts/vast_onstart.sh scripts/run_webrtc_relay_api_server.sh scripts/run_turnserver_tcp_relay.sh \
           scripts/run_trt_stagewise_server.sh scripts/test_startup_scripts.sh setup_musetalk.sh \
           scripts/setup_musetalk.sh \
           scripts/install_musetalk.sh download_weights.sh \
           docs/startup_rework_20260928/impl/launch/gpu_sequence.sh \
           docs/startup_rework_20260928/impl/launch/run-musetalk-local-trt.sh.proposed; do
    if [[ -f "$REAL_REPO/$f" ]]; then
      bash -n "$REAL_REPO/$f" 2> "$TEST_ROOT/bashn.err" || fail "bash -n $f: $(cat "$TEST_ROOT/bashn.err")"
    elif [[ "$f" != scripts/install_musetalk.sh ]]; then
      fail "missing $f"
    fi
  done
  end

  begin "syntax: python components parse (ast, no bytecode written)"
  for f in scripts/musetalk_host_profile.py scripts/musetalk_engine_keys.py scripts/unet_engine_store.py \
           scripts/musetalk_selftest.py; do
    [[ -f "$REAL_REPO/$f" ]] || continue
    "$TEST_PY" -c 'import ast, sys; ast.parse(open(sys.argv[1]).read(), sys.argv[1])' "$REAL_REPO/$f" \
      2> "$TEST_ROOT/ast.err" || fail "ast.parse $f: $(tail -n 2 "$TEST_ROOT/ast.err")"
  done
  end

  begin "syntax: TURN relay default range widened to 49160-49460"
  assert_file_contains "$REAL_REPO/scripts/run_turnserver_tcp_relay.sh" 'TURN_INTERNAL_RELAY_MIN_PORT:-49160' "min port"
  assert_file_contains "$REAL_REPO/scripts/run_turnserver_tcp_relay.sh" 'TURN_INTERNAL_RELAY_MAX_PORT:-49460' "max port"
  end

  begin "syntax: ctl health timeout default 900 s"
  assert_file_contains "$REAL_REPO/scripts/vast_server_ctl.sh" 'STARTUP_TIMEOUT_SECONDS:-900' "timeout"
  end
fi

# ================================================================ parser
if selected parser; then
  begin "parser: KEY=VALUE forms, quotes, comments, rejects"
  out="$(env -i PATH="$SAFE_PATH" bash -c '
    set -Eeuo pipefail
    source "$1"
    check() {
      local rc=0; mt_env_parse_line "$1" || rc=$?
      if (( rc == 0 )); then printf "%s|0|%s|%s\n" "$2" "$MT_ENV_KEY" "$MT_ENV_VALUE"; else printf "%s|%s\n" "$2" "$rc"; fi
    }
    check "A=1" a
    check "export B=two" b
    check "C='\''a b'\''" c
    check "D='\''it'\''\\'\'''\''s'\''" d
    check "E=\"x y\"" e
    check "F=v   # inline comment" f
    check "G=a b" g
    check "H=\$(touch /tmp/pwned)" h
    check "1BAD=x" i
    check "   # just a comment" j
    check "" k
    check "K=" l
    check "M=a;b" m
    check "N=#literal" n
    check "BASH_ENV=/x" o
    check "Q=8:/abs/unet_trt.ts" q
  ' _ "$REAL_REPO/scripts/lib/musetalk_env_layers.sh")"
  printf '%s\n' "$out" > "$TEST_ROOT/parser.out"
  assert_file_contains "$TEST_ROOT/parser.out" '^a\|0\|A\|1$' "bare"
  assert_file_contains "$TEST_ROOT/parser.out" '^b\|0\|B\|two$' "export"
  assert_file_contains "$TEST_ROOT/parser.out" '^c\|0\|C\|a b$' "single quotes"
  assert_file_contains "$TEST_ROOT/parser.out" "^d\\|0\\|D\\|it's$" "quote idiom"
  assert_file_contains "$TEST_ROOT/parser.out" '^e\|0\|E\|x y$' "double quotes"
  assert_file_contains "$TEST_ROOT/parser.out" '^f\|0\|F\|v$' "inline comment"
  assert_file_contains "$TEST_ROOT/parser.out" '^g\|2$' "unquoted space rejected"
  assert_file_contains "$TEST_ROOT/parser.out" '^h\|2$' "command substitution rejected"
  assert_file_contains "$TEST_ROOT/parser.out" '^i\|2$' "bad name rejected"
  assert_file_contains "$TEST_ROOT/parser.out" '^j\|1$' "comment skipped"
  assert_file_contains "$TEST_ROOT/parser.out" '^k\|1$' "blank skipped"
  assert_file_contains "$TEST_ROOT/parser.out" '^l\|0\|K\|$' "empty value"
  assert_file_contains "$TEST_ROOT/parser.out" '^m\|2$' "metachar rejected"
  assert_file_contains "$TEST_ROOT/parser.out" '^n\|0\|N\|#literal$' "hash inside word"
  assert_file_contains "$TEST_ROOT/parser.out" '^o\|2$' "deny-listed key"
  assert_file_contains "$TEST_ROOT/parser.out" '^q\|0\|Q\|8:/abs/unet_trt.ts$' "path value"
  [[ ! -e /tmp/pwned ]] || fail "a file value executed code"
  end

  begin "parser: only-if-unset, first overrides file wins, missing files skipped, empty caller value kept"
  mkdir -p "$TEST_ROOT/layers"
  printf 'X1=ovr1\nX2=ovr1\nX3=ovr1\nX2=ovr1_dup\n' > "$TEST_ROOT/layers/one.env"
  printf 'X2=ovr2\nX4=ovr2\nbroken line\n' > "$TEST_ROOT/layers/two.env"
  printf "X1='res'\nX4='res'\nX5='res'\n" > "$TEST_ROOT/layers/res.env"
  out="$(env -i PATH="$SAFE_PATH" X1=caller X3= \
    MUSETALK_ENV_OVERRIDES_FILE="$TEST_ROOT/layers/one.env:$TEST_ROOT/layers/missing.env:$TEST_ROOT/layers/two.env" \
    bash -c '
      set -Eeuo pipefail
      source "$1"
      mt_env_record_caller
      mt_env_load_overrides /nonexistent-repo 2>/dev/null
      mt_env_load_file "$2" resolved
      for k in X1 X2 X3 X4 X5; do printf "%s=%s|%s\n" "$k" "${!k-<unset>}" "${MT_ENV_SOURCE[$k]:-none}"; done
      printf "files=%s\n" "${#MT_ENV_FILES_USED[@]}"
      printf "warnings=%s\n" "${#MT_ENV_WARNINGS[@]}"
    ' _ "$REAL_REPO/scripts/lib/musetalk_env_layers.sh" "$TEST_ROOT/layers/res.env")"
  printf '%s\n' "$out" > "$TEST_ROOT/layers.out"
  assert_file_contains "$TEST_ROOT/layers.out" '^X1=caller\|caller$' "caller beats overrides and resolved"
  assert_file_contains "$TEST_ROOT/layers.out" "^X2=ovr1\\|overrides:$TEST_ROOT/layers/one.env$" "first overrides file wins (and first line in a file)"
  assert_file_contains "$TEST_ROOT/layers.out" '^X3=\|caller$' "empty caller value is still 'set'"
  assert_file_contains "$TEST_ROOT/layers.out" "^X4=ovr2\\|overrides:$TEST_ROOT/layers/two.env$" "second overrides file"
  assert_file_contains "$TEST_ROOT/layers.out" '^X5=res\|resolved$' "resolved fills the rest"
  assert_file_contains "$TEST_ROOT/layers.out" '^files=2$' "missing file skipped"
  assert_file_contains "$TEST_ROOT/layers.out" '^warnings=1$' "invalid line warned"
  end

  begin "parser: example overrides and fast300 recipe parse cleanly when uncommented"
  sed -nE 's/^#([A-Z][A-Z0-9_]*=)/\1/p' "$REAL_REPO/configs/musetalk_overrides.env.example" > "$TEST_ROOT/ex_ovr.env"
  sed -nE 's/^#([A-Z][A-Z0-9_]*=)/\1/p' "$REAL_REPO/configs/recipes/fast300.env" > "$TEST_ROOT/ex_f300.env"
  out="$(env -i PATH="$SAFE_PATH" bash -c '
    set -Eeuo pipefail; source "$1"
    mt_env_load_file "$2" a; printf "ovr=%s/%s\n" "$MT_ENV_LAST_LOADED" "$MT_ENV_LAST_INVALID"
    mt_env_load_file "$3" b; printf "f300=%s/%s\n" "$(( MT_ENV_LAST_LOADED + MT_ENV_LAST_KEPT ))" "$MT_ENV_LAST_INVALID"
  ' _ "$REAL_REPO/scripts/lib/musetalk_env_layers.sh" "$TEST_ROOT/ex_ovr.env" "$TEST_ROOT/ex_f300.env")"
  [[ "$out" =~ ovr=([0-9]+)/0 ]] && (( BASH_REMATCH[1] >= 60 )) || fail "overrides example: $out"
  [[ "$out" =~ f300=([0-9]+)/0 ]] && (( BASH_REMATCH[1] >= 10 )) || fail "fast300 recipe: $out"
  if grep -Eq '^[A-Z][A-Z0-9_]*=' "$REAL_REPO/configs/recipes/fast300.env"; then
    fail "fast300.env must ship with every lever commented out"
  fi
  end
fi

# ================================================================ launcher (stub resolver)
if selected launcher; then
  L="$(make_repo launcher)"
  OVR="$L/.runtime/musetalk_overrides.env"

  begin "launcher: precedence caller > overrides > resolved (print-env sources + exec'd env)"
  printf 'MUSETALK_TEST_LAYER_A=ovr\nMUSETALK_TEST_LAYER_B=ovr\n' > "$OVR"
  printf 'MUSETALK_TEST_LAYER_B=ovr2\nMUSETALK_TEST_LAYER_C=ovr2\n' > "$L/.runtime/second.env"
  rc=0
  run_launcher "$L" "$TEST_ROOT/pe1.out" MUSETALK_TEST_LAYER_A=caller \
    "MUSETALK_ENV_OVERRIDES_FILE=$OVR:$L/.runtime/second.env" -- --print-env || rc=$?
  assert_rc "$rc" 0 "print-env"
  assert_eq "$(pe_get "$TEST_ROOT/pe1.out" MUSETALK_TEST_LAYER_A)" caller "A value"
  assert_eq "$(pe_get "$TEST_ROOT/pe1.out" MUSETALK_TEST_LAYER_A source)" caller "A source"
  assert_eq "$(pe_get "$TEST_ROOT/pe1.out" MUSETALK_TEST_LAYER_B)" ovr "B value (first file wins)"
  assert_eq "$(pe_get "$TEST_ROOT/pe1.out" MUSETALK_TEST_LAYER_B source)" "overrides:$OVR" "B source"
  assert_eq "$(pe_get "$TEST_ROOT/pe1.out" MUSETALK_TEST_LAYER_C)" ovr2 "C value"
  assert_eq "$(pe_get "$TEST_ROOT/pe1.out" MUSETALK_TEST_LAYER_D)" resolved "D value"
  assert_eq "$(pe_get "$TEST_ROOT/pe1.out" MUSETALK_TEST_LAYER_D source)" resolved "D source"
  assert_missing "$TEST_ROOT/api_env.json" "print-env must not exec the server"
  assert_missing "$L/.runtime/musetalk_resolved.env" "print-env must not touch the canonical resolved env"
  assert_missing "$L/.runtime/musetalk_launch_18001.json" "print-env must not write launch state"
  # the resolver saw the overrides (it must compute dependents from them)
  assert_eq "$(dump_get "$TEST_ROOT/resolver_env.json" MUSETALK_TEST_LAYER_B)" ovr "resolver sees overrides"
  assert_file_contains "$TEST_ROOT/resolver_env.json" 'MUSETALK_ENV_OVERRIDE_KEYS' "source hints passed to the resolver"
  rc=0
  run_launcher "$L" "$TEST_ROOT/run1.out" MUSETALK_TEST_LAYER_A=caller \
    "MUSETALK_ENV_OVERRIDES_FILE=$OVR:$L/.runtime/second.env" -- || rc=$?
  assert_rc "$rc" 0 "launch"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" MUSETALK_TEST_LAYER_A)" caller "exec'd A"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" MUSETALK_TEST_LAYER_B)" ovr "exec'd B"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" MUSETALK_TEST_LAYER_C)" ovr2 "exec'd C"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" MUSETALK_TEST_LAYER_D)" resolved "exec'd D"
  assert_eq "$(dump_argv "$TEST_ROOT/api_env.json")" "--host 127.0.0.1 --port 18001" "api_server argv"
  assert_exists "$L/.runtime/musetalk_resolved.env" "canonical resolved env written on launch"
  assert_exists "$L/.runtime/musetalk_resolved.json" "canonical resolved report written on launch"
  end

  begin "launcher: resolved value with quotes round-trips exactly"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" MUSETALK_TEST_QUOTED)" "it's a 'quoted' \$HOME value" "quoted value"
  end

  begin "launcher: launch state JSON (expectations, sources, secrets redacted)"
  state="$L/.runtime/musetalk_launch_18001.json"
  assert_exists "$state" "launch state"
  assert_eq "$(json_get "$state" schema)" musetalk_launch_v1 "schema"
  assert_eq "$(json_get "$state" expect.vae)" taesd "expect vae"
  assert_eq "$(json_get "$state" expect.unet)" eager "expect unet"
  assert_eq "$(json_get "$state" env.MUSETALK_TEST_LAYER_A.source)" caller "state source"
  rc=0
  run_launcher "$L" "$TEST_ROOT/run_secret.out" TURN_PASS=supersecret WEBRTC_TURN_PASS=supersecret -- || rc=$?
  assert_rc "$rc" 0 "launch with secrets"
  assert_file_lacks "$state" supersecret "secret in launch state"
  assert_file_lacks "$TEST_ROOT/run_secret.out" supersecret "secret in launcher log"
  end

  begin "launcher: unknown knobs pass through untouched (caller + overrides)"
  printf 'WEBRTC_FUTURE_KNOB=abc\nHLS_GPU_PIPELINE_DEPTH=2\n' > "$OVR"
  rc=0
  run_launcher "$L" "$TEST_ROOT/run2.out" MUSETALK_FUTURE_KNOB=42 WEBRTC_IDLE_FRAME_CACHE=1 -- || rc=$?
  assert_rc "$rc" 0 "launch"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" MUSETALK_FUTURE_KNOB)" 42 "caller unknown knob"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" WEBRTC_FUTURE_KNOB)" abc "overrides unknown knob"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" HLS_GPU_PIPELINE_DEPTH)" 2 "300 fps lever from overrides"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" WEBRTC_IDLE_FRAME_CACHE)" 1 "300 fps lever from caller"
  assert_file_contains "$TEST_ROOT/run2.out" 'levers:.*HLS_GPU_PIPELINE_DEPTH=.2.\(overrides:' "levers line shows source"
  rc=0
  run_launcher "$L" "$TEST_ROOT/pe2.out" -- --print-env || rc=$?
  assert_eq "$(pe_get "$TEST_ROOT/pe2.out" WEBRTC_FUTURE_KNOB source)" "overrides:$OVR" "print-env lists overrides-only keys"
  assert_file_contains "$TEST_ROOT/pe2.out" '^#MUSETALK_TAESD_BACKEND=  # unset' "unset lever listed"
  end

  begin "launcher: stale caller keys dropped, overrides may set them"
  printf 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True\n' > "$OVR"
  rc=0
  run_launcher "$L" "$TEST_ROOT/run3.out" PYTORCH_CUDA_ALLOC_CONF=caller_value MUSETALK_CPU_TUNING=1 -- || rc=$?
  assert_rc "$rc" 0 "launch"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" PYTORCH_CUDA_ALLOC_CONF)" "expandable_segments:True" "overrides value used"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" MUSETALK_CPU_TUNING)" "<unset>" "caller MUSETALK_CPU_TUNING unset"
  end

  begin "launcher: bucket override reaches the resolver (dependents follow)"
  printf 'HLS_SCHEDULER_FIXED_BATCH_SIZES=16\n' > "$OVR"
  rc=0
  run_launcher "$L" "$TEST_ROOT/run4.out" -- || rc=$?
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" MUSETALK_TAESD_WARMUP_BATCHES)" 16 "warmup follows buckets"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" HLS_SCHEDULER_MAX_BATCH)" 16 "max batch follows buckets"
  end

  begin "launcher: invalid overrides line warns, the rest applies"
  printf 'GOOD_KNOB_X=1\nthis is not valid\nBAD=$(id)\n' > "$OVR"
  rc=0
  run_launcher "$L" "$TEST_ROOT/run5.out" -- || rc=$?
  assert_rc "$rc" 0 "launch"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" GOOD_KNOB_X)" 1 "valid line applied"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" BAD)" "<unset>" "expansion line ignored"
  assert_file_contains "$TEST_ROOT/run5.out.err" 'WARNING: .*musetalk_overrides.env:2 ignored' "warned with line number"
  end

  begin "launcher: resolver hard error (exit 2) stops the launch"
  : > "$OVR"
  rc=0
  run_launcher "$L" "$TEST_ROOT/run6.out" MT_TEST_RESOLVER_EXIT=2 -- || rc=$?
  assert_rc "$rc" nonzero "launch"
  assert_file_contains "$TEST_ROOT/run6.out.err" 'Resolver refused' "message"
  assert_file_contains "$TEST_ROOT/run6.out.err" 'stub: forced failure' "report errors surfaced"
  assert_missing "$TEST_ROOT/api_env.json" "server not started"
  end

  begin "launcher: missing resolver is fatal"
  rc=0
  run_launcher "$L" "$TEST_ROOT/run7.out" "MUSETALK_RESOLVER=$L/scripts/nope.py" -- || rc=$?
  assert_rc "$rc" nonzero "launch"
  assert_file_contains "$TEST_ROOT/run7.out.err" 'Resolver not found' "message"
  end

  begin "launcher: fast300 dispatch passes --recipe fast300 and exports the recipe"
  printf 'MUSETALK_RECIPE=fast300\n' > "$OVR"
  rc=0
  run_launcher "$L" "$TEST_ROOT/run8.out" -- || rc=$?
  assert_rc "$rc" 0 "launch"
  assert_file_contains "$TEST_ROOT/resolver_env.json" '"--recipe", "fast300"' "resolver argv"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" MUSETALK_RECIPE)" fast300 "exported recipe"
  assert_file_contains "$TEST_ROOT/run8.out" 'recipe=fast300 \(source=overrides:' "summary names the source"
  end

  begin "launcher: legacy_int8 (one overrides line) execs the old chain with the same args, no resolver, no overrides"
  printf 'MUSETALK_RECIPE=legacy_int8\nMUSETALK_TEST_LAYER_A=from_overrides\n' > "$OVR"
  rc=0
  run_launcher "$L" "$TEST_ROOT/run9.out" -- --profile throughput_record || rc=$?
  assert_rc "$rc" 0 "legacy dispatch"
  assert_eq "$(dump_argv "$TEST_ROOT/legacy_env.json")" \
    "--host 127.0.0.1 --port 18001 --venv-path $FAKE_VENV --repo-root $L --profile throughput_record" "legacy argv"
  assert_eq "$(dump_get "$TEST_ROOT/legacy_env.json" MUSETALK_RECIPE)" legacy_int8 "recipe exported"
  assert_eq "$(dump_get "$TEST_ROOT/legacy_env.json" MUSETALK_TEST_LAYER_A)" "<unset>" "overrides not applied to legacy"
  assert_missing "$TEST_ROOT/resolver_env.json" "resolver not run for legacy"
  assert_missing "$TEST_ROOT/api_env.json" "api_server not exec'd directly"
  rc=0
  run_launcher "$L" "$TEST_ROOT/run9b.out" MUSETALK_LAUNCHER_DRY_RUN=1 -- || rc=$?
  assert_rc "$rc" 0 "legacy dry run"
  assert_file_contains "$TEST_ROOT/run9b.out" 'DRY RUN: exec bash .*run_trt_stagewise_server.sh' "dry run line"
  assert_missing "$TEST_ROOT/legacy_env.json" "dry run must not exec"
  rc=0
  run_launcher "$L" "$TEST_ROOT/run9c.out" -- --print-env || rc=$?
  assert_rc "$rc" 0 "legacy print-env"
  assert_file_contains "$TEST_ROOT/run9c.out" "^MUSETALK_RECIPE='legacy_int8'" "legacy print-env"
  assert_missing "$TEST_ROOT/legacy_env.json" "print-env must not exec legacy"
  rc=0
  run_launcher "$L" "$TEST_ROOT/run9d.out" MUSETALK_RECIPE=fast -- || rc=$?
  assert_rc "$rc" 0 "caller recipe beats the overrides line"
  assert_exists "$TEST_ROOT/api_env.json" "fast path taken"
  end

  begin "launcher: unsupported recipe is refused"
  : > "$OVR"
  rc=0
  run_launcher "$L" "$TEST_ROOT/run10.out" MUSETALK_RECIPE=turbo -- || rc=$?
  assert_rc "$rc" nonzero "launch"
  assert_file_contains "$TEST_ROOT/run10.out.err" 'Unsupported MUSETALK_RECIPE=turbo' "message"
  end

  begin "launcher: --validate-only resolves to .validate files, no exec, no launch state"
  rm -f "$L/.runtime/musetalk_launch_18001.json" "$L/.runtime/musetalk_resolved.validate.env"
  before="$(stat -c %Y "$L/.runtime/musetalk_resolved.env" 2>/dev/null || echo none)"
  sleep 1
  rc=0
  run_launcher "$L" "$TEST_ROOT/run11.out" -- --validate-only || rc=$?
  assert_rc "$rc" 0 "validate-only"
  assert_missing "$TEST_ROOT/api_env.json" "no exec"
  assert_missing "$L/.runtime/musetalk_launch_18001.json" "no launch state"
  assert_exists "$L/.runtime/musetalk_resolved.validate.env" "validate env"
  assert_eq "$(stat -c %Y "$L/.runtime/musetalk_resolved.env" 2>/dev/null || echo none)" "$before" "canonical resolved env untouched"
  assert_file_contains "$TEST_ROOT/run11.out" 'Validation-only checks passed' "message"
  end

  begin "launcher: native VP8 preflight failure is fatal, MUSETALK_VP8_FALLBACK=1 -> pyav"
  rc=0
  run_launcher "$L" "$TEST_ROOT/run12.out" WEBRTC_VP8_ENCODER=native -- || rc=$?
  assert_rc "$rc" nonzero "native without artifact"
  assert_file_contains "$TEST_ROOT/run12.out.err" 'native VP8 preflight failed' "message"
  rc=0
  run_launcher "$L" "$TEST_ROOT/run12b.out" WEBRTC_VP8_ENCODER=native MUSETALK_VP8_FALLBACK=1 -- --print-env || rc=$?
  assert_rc "$rc" 0 "fallback"
  assert_eq "$(pe_get "$TEST_ROOT/run12b.out" WEBRTC_VP8_ENCODER)" pyav "fell back to pyav"
  assert_eq "$(pe_get "$TEST_ROOT/run12b.out" WEBRTC_VP8_ENCODER source)" launcher:vp8_fallback "fallback source"
  cat > "$L/scripts/webrtc_native_vp8.py" <<'PY'
import json, os
def configure_vp8_encoder(label):
    with open(os.environ["MT_TEST_VP8_DUMP"], "w") as fh:
        json.dump({"env": dict(os.environ), "argv": [label]}, fh)
    return {"encoder": "native"}
PY
  rc=0
  run_launcher "$L" "$TEST_ROOT/run12c.out" WEBRTC_VP8_ENCODER=native CUDA_VISIBLE_DEVICES=0 \
    "MT_TEST_VP8_DUMP=$TEST_ROOT/vp8.json" -- || rc=$?
  assert_rc "$rc" 0 "preflight ok"
  assert_eq "$(dump_get "$TEST_ROOT/vp8.json" CUDA_VISIBLE_DEVICES)" "" "preflight hides CUDA"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" WEBRTC_VP8_ENCODER)" native "native kept"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" CUDA_VISIBLE_DEVICES)" 0 "server CUDA env untouched"
  rm -f "$L/scripts/webrtc_native_vp8.py"
  end

  begin "launcher: values the server would reject fail fast"
  for bad in "WEBRTC_H264_IMPL=bogus" "MUSETALK_TRT_UNET_CUDAGRAPHS=bogus" "HLS_SCHEDULER_POLICY=fifo"; do
    rc=0
    run_launcher "$L" "$TEST_ROOT/run13.out" "$bad" -- --validate-only || rc=$?
    assert_rc "$rc" nonzero "$bad"
  done
  rc=0
  run_launcher "$L" "$TEST_ROOT/run13b.out" WEBRTC_H264_IMPL=x264tuned MUSETALK_TRT_UNET_CUDAGRAPHS=manual \
    HLS_SCHEDULER_POLICY=edf -- --validate-only || rc=$?
  assert_rc "$rc" 0 "valid lever values"
  end

  begin "launcher: trt_stagewise needs a complete manifest when fallback is off"
  rc=0
  run_launcher "$L" "$TEST_ROOT/run14.out" MUSETALK_UNET_BACKEND=trt_stagewise -- --validate-only || rc=$?
  assert_rc "$rc" nonzero "missing manifest"
  assert_file_contains "$TEST_ROOT/run14.out.err" 'manifest.json is missing or incomplete' "message"
  mkdir -p "$L/models/tensorrt_unet_stagewise_sm89/bs16"
  printf '{"schema": "musetalk_unet_stagewise_trt_v1", "batch": 16, "complete": true}\n' \
    > "$L/models/tensorrt_unet_stagewise_sm89/bs16/manifest.json"
  rc=0
  run_launcher "$L" "$TEST_ROOT/run14b.out" MUSETALK_UNET_BACKEND=trt_stagewise -- --print-env || rc=$?
  assert_rc "$rc" 0 "complete manifest"
  assert_file_contains "$TEST_ROOT/run14b.out" '^# expect: vae=taesd unet=trt_stagewise' "expectation"
  assert_file_contains "$TEST_ROOT/run14b.out.err" 'engine batch 16 != HLS_SCHEDULER_FIXED_BATCH_SIZES=8' "bucket warning"
  rc=0
  run_launcher "$L" "$TEST_ROOT/run14c.out" MUSETALK_TAESD_BACKEND=trt -- --print-env || rc=$?
  assert_file_contains "$TEST_ROOT/run14c.out" '^# expect: vae=taesd_trt unet=eager' "TAESD TRT expectation"
  end
fi

# ================================================================ ctl (stub server, no sockets)
if selected ctl; then
  C="$(make_repo ctl)"
  mkdir -p "$TEST_ROOT/ctl/logs"
  echo '{"metrics": {"active_requests": 0, "active_sessions_local": 0, "queue_depth": 0}}' > "$TEST_ROOT/ctl/state.json"
  : > "$TEST_ROOT/ctl/drain"
  run_ctl() {
    # run_ctl OUT COMMAND [NAME=VALUE ...]
    local out="$1" cmd="$2" rc=0
    shift 2
    local -a envv=()
    mapfile -t envv < <(base_env "$C")
    env -i "${envv[@]}" REPO_ROOT="$C" VENV_PATH="$FAKE_VENV" PORT=18002 HOST=127.0.0.1 \
      LOG_DIR="$TEST_ROOT/ctl/logs" \
      HEALTH_URL="file://$TEST_ROOT/ctl/health" WORKER_STATE_URL="file://$TEST_ROOT/ctl/state.json" \
      DRAIN_URL="file://$TEST_ROOT/ctl/drain" MT_TEST_HEALTH_FILE="$TEST_ROOT/ctl/health" \
      MT_TEST_SERVER_MODE=serve MT_TEST_SERVER_MAX_S=60 MT_TEST_ENV_DUMP= \
      LINGUA_CONTROL_PLANE_ENV_FILE="$TEST_ROOT/none.env" VAST_SERVER_CTL_LOAD_TURN_ENV=0 \
      WEBRTC_RELAY_ENABLED=0 WEBRTC_TURN_AUTOSTART=0 PUBLIC_IPADDR=127.0.0.1 VAST_TCP_PORT_8000=18002 \
      STARTUP_TIMEOUT_SECONDS=30 POLL_INTERVAL_SECONDS=0.2 DRAIN_TIMEOUT_SECONDS=5 \
      MUSETALK_VERIFY_TIMEOUT_SECONDS=5 "$@" \
      timeout 90 bash "$C/scripts/vast_server_ctl.sh" "$cmd" > "$out" 2>&1 || rc=$?
    return "$rc"
  }
  ctl_pid() {
    if [[ -f "$TEST_ROOT/ctl/logs/api_server_18002.pid" ]]; then
      tr -d '[:space:]' < "$TEST_ROOT/ctl/logs/api_server_18002.pid"
    fi
  }
  ctl_cleanup() {
    local pid
    pid="$(ctl_pid)"
    [[ -n "$pid" ]] && kill "$pid" >/dev/null 2>&1 || true
    rm -f "$TEST_ROOT/ctl/health" "$TEST_ROOT/ctl/logs/api_server_18002.pid"
  }

  begin "ctl: start -> health -> verify PASS (strict), status summary, stop"
  rc=0
  run_ctl "$TEST_ROOT/ctl1.out" start MT_TEST_SERVER_VAE=taesd MT_TEST_SERVER_UNET= || rc=$?
  assert_rc "$rc" 0 "start"
  pid="$(ctl_pid)"
  [[ -n "$pid" ]] && echo "$pid" >> "$TEST_ROOT/pids"
  assert_file_contains "$TEST_ROOT/ctl1.out" 'launcher=.*run_musetalk_server.sh recipe=fast' "default launcher"
  assert_file_contains "$TEST_ROOT/ctl1.out" 'Recipe verification passed: vae=taesd unet=eager' "verify pass"
  assert_file_contains "$TEST_ROOT/ctl/logs/api_server_18002.verify" 'result=PASS' "verify state"
  rc=0
  run_ctl "$TEST_ROOT/ctl1s.out" status || rc=$?
  assert_rc "$rc" 0 "status"
  assert_file_contains "$TEST_ROOT/ctl1s.out" 'recipe=fast \(source=' "status recipe"
  assert_file_contains "$TEST_ROOT/ctl1s.out" 'vae=taesd .*unet=eager' "status backends"
  assert_file_contains "$TEST_ROOT/ctl1s.out" 'last_verify: .*result=PASS' "status verify"
  rc=0
  run_ctl "$TEST_ROOT/ctl1t.out" stop || rc=$?
  assert_rc "$rc" 0 "stop"
  assert_file_contains "$TEST_ROOT/ctl1t.out" 'Requested drain' "drain before stop"
  sleep 0.5
  [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null && fail "server still running after stop"
  ctl_cleanup
  end

  begin "ctl: strict verify mismatch stops the server and fails start"
  rc=0
  run_ctl "$TEST_ROOT/ctl2.out" start MT_TEST_SERVER_VAE=pytorch || rc=$?
  assert_rc "$rc" nonzero "start"
  pid="$(ctl_pid)"
  assert_file_contains "$TEST_ROOT/ctl2.out" 'recipe verification failed \(MUSETALK_VERIFY_RECIPE=strict\)' "message"
  assert_file_contains "$TEST_ROOT/ctl2.out" 'Server stopped' "stopped"
  assert_file_contains "$TEST_ROOT/ctl/logs/api_server_18002.verify" 'result=FAIL' "verify state"
  assert_missing "$TEST_ROOT/ctl/health" "server gone"
  ctl_cleanup
  end

  begin "ctl: warn mode keeps the server up; off skips; overrides file can set the mode"
  printf 'MUSETALK_VERIFY_RECIPE=warn\n' > "$C/.runtime/musetalk_overrides.env"
  rc=0
  run_ctl "$TEST_ROOT/ctl3.out" start MT_TEST_SERVER_VAE=pytorch || rc=$?
  assert_rc "$rc" 0 "warn start"
  assert_file_contains "$TEST_ROOT/ctl3.out" 'MUSETALK_VERIFY_RECIPE=warn; server left running' "warn message"
  assert_exists "$TEST_ROOT/ctl/health" "still serving"
  run_ctl "$TEST_ROOT/ctl3t.out" stop || true
  ctl_cleanup
  : > "$C/.runtime/musetalk_overrides.env"
  rc=0
  run_ctl "$TEST_ROOT/ctl3b.out" start MT_TEST_SERVER_VAE=pytorch MUSETALK_VERIFY_RECIPE=off || rc=$?
  assert_rc "$rc" 0 "off start"
  assert_file_contains "$TEST_ROOT/ctl3b.out" 'Recipe verification disabled' "off message"
  run_ctl "$TEST_ROOT/ctl3c.out" stop || true
  ctl_cleanup
  end

  begin "ctl: verifier without taesd_trt/trt_stagewise choices -> conservative retry"
  printf 'MUSETALK_TAESD_BACKEND=trt\nMUSETALK_TAESD_TRT_BUILD=1\n' > "$C/.runtime/musetalk_overrides.env"
  rc=0
  run_ctl "$TEST_ROOT/ctl4.out" start MT_TEST_SERVER_VAE=taesd MT_TEST_VERIFY_STRICT_CHOICES=1 \
    "MT_TEST_VERIFY_CALLS=$TEST_ROOT/ctl/verify_calls" || rc=$?
  assert_rc "$rc" 0 "start"
  assert_file_contains "$TEST_ROOT/ctl4.out" 'retrying with vae=taesd unet=eager' "retry"
  assert_file_contains "$TEST_ROOT/ctl/verify_calls" '^taesd eager ' "second call"
  run_ctl "$TEST_ROOT/ctl4t.out" stop || true
  ctl_cleanup
  : > "$C/.runtime/musetalk_overrides.env"
  end

  begin "ctl: verify reads the log from the recorded byte offset (old lines ignored)"
  printf 'VAE decode backend active: taesd\nUNet backend: PyTorch\n' >> "$TEST_ROOT/ctl/logs/api_server_18002.log"
  rc=0
  run_ctl "$TEST_ROOT/ctl5.out" start MT_TEST_SERVER_VAE=pytorch || rc=$?
  assert_rc "$rc" nonzero "old matching lines must not satisfy verify"
  ctl_cleanup
  end

  begin "ctl: legacy_int8 skips verification"
  rc=0
  run_ctl "$TEST_ROOT/ctl6.out" start MUSETALK_RECIPE=legacy_int8 || rc=$?
  assert_rc "$rc" 0 "legacy start"
  assert_file_contains "$TEST_ROOT/ctl6.out" 'verification skipped for legacy_int8' "skip"
  run_ctl "$TEST_ROOT/ctl6t.out" stop || true
  ctl_cleanup
  end
fi

# ================================================================ onstart (all stubs)
if selected onstart; then
  O="$(make_repo onstart)"
  cat > "$O/scripts/install_musetalk.sh" <<'EOF'
#!/usr/bin/env bash
MT_TEST_RECORD_APPEND=1 "$(dirname "$0")/../stub_record.py" "$MT_TEST_INSTALL_LOG" "$@"
if [[ "${1:-}" == --check ]]; then exit "${MT_TEST_CHECK_RC:-0}"; fi
exit "${MT_TEST_INSTALL_RC:-0}"
EOF
  cat > "$O/scripts/unet_engine_store.py" <<'PY'
import json, os, sys
with open(os.environ["MT_TEST_ENSURE_LOG"], "a") as fh:
    fh.write(" ".join(sys.argv[1:]) + "\n")
with open(os.environ["MT_TEST_ENSURE_LOG"] + ".env", "a") as fh:
    fh.write("store=%s remote=%s\n" % (os.environ.get("MUSETALK_TAESD_TRT_ENGINE_STORE", ""),
                                       os.environ.get("MUSETALK_UNET_ENGINE_REMOTE", "")))
sys.exit(int(os.environ.get("MT_TEST_ENSURE_RC", "0")))
PY
  cat > "$O/scripts/vast_server_ctl.sh" <<'EOF'
#!/usr/bin/env bash
"$(dirname "$0")/../stub_record.py" "$MT_TEST_CTL_DUMP" "$@"
exit "${MT_TEST_CTL_RC:-0}"
EOF
  cat > "$O/scripts/trt_artifact_bundle.py" <<'PY'
import os, sys
open(os.environ["MT_TEST_LEGACY_STEPS"], "a").write("restore " + " ".join(sys.argv[1:]) + "\n")
PY
  cat > "$O/scripts/select_unet_trt_profile.py" <<'PY'
import os, sys
open(os.environ["MT_TEST_LEGACY_STEPS"], "a").write("select " + " ".join(sys.argv[1:]) + "\n")
out = sys.argv[sys.argv.index("--output") + 1]
open(out, "w").write("MUSETALK_VAE_BACKEND=trt_stagewise\n")
PY
  chmod +x "$O/scripts/install_musetalk.sh" "$O/scripts/vast_server_ctl.sh"
  run_onstart() {
    # run_onstart NAME [NAME=VALUE ...]; sets ON_RC and ON_DIR
    local name="$1"
    shift
    ON_DIR="$TEST_ROOT/onstart_$name"
    mkdir -p "$ON_DIR"
    ON_RC=0
    local -a envv=()
    mapfile -t envv < <(base_env "$O")
    env -i "${envv[@]}" REPO_ROOT="$O" VENV_PATH="$FAKE_VENV" ONSTART_LOG="$ON_DIR/onstart.log" \
      SETUP_WEBRTC_TURN=0 ONSTART_POST_VALIDATE_IMPORTS=0 AUTO_SETUP=1 \
      MT_TEST_INSTALL_LOG="$ON_DIR/install.jsonl" MT_TEST_ENSURE_LOG="$ON_DIR/ensure.log" \
      MT_TEST_CTL_DUMP="$ON_DIR/ctl.json" MT_TEST_LEGACY_STEPS="$ON_DIR/legacy_steps.log" \
      TRT_ARTIFACT_S3_BUCKET= "$@" \
      timeout 60 bash "$O/scripts/vast_onstart.sh" > "$ON_DIR/stdout" 2>&1 || ON_RC=$?
    touch "$ON_DIR/ensure.log" "$ON_DIR/install.jsonl" "$ON_DIR/legacy_steps.log"
  }
  marker_count() { grep -c "$2" "$1/onstart.log" 2>/dev/null || true; }

  begin "onstart: fast default -> check OK, ensure unet_ts, ctl start without legacy profile env, COMPLETE"
  run_onstart fast1
  assert_rc "$ON_RC" 0 "onstart"
  assert_eq "$(marker_count "$ON_DIR" 'VAST_ONSTART COMPLETE')" 1 "COMPLETE marker"
  assert_eq "$(marker_count "$ON_DIR" 'VAST_ONSTART FAILED')" 0 "no FAILED marker"
  assert_file_contains "$ON_DIR/install.jsonl" '"argv": \["--check", "--venv"' "install check"
  assert_eq "$(grep -c '"--check"' "$ON_DIR/install.jsonl")" "$(wc -l < "$ON_DIR/install.jsonl" | tr -d ' ')" "no install when check passes"
  assert_file_contains "$ON_DIR/ensure.log" '^ensure --kind unet_ts --provision auto$' "ensure unet_ts"
  assert_eq "$(wc -l < "$ON_DIR/ensure.log" | tr -d ' ')" 1 "only unet_ts for fast"
  assert_eq "$(dump_argv "$ON_DIR/ctl.json")" start "ctl start"
  assert_eq "$(dump_get "$ON_DIR/ctl.json" MUSETALK_TRT_PROFILE_ENV_FILE)" "<unset>" "no legacy profile env in the fast chain"
  assert_eq "$(dump_get "$ON_DIR/ctl.json" PROFILE)" "<unset>" "PROFILE not exported into the fast chain"
  assert_eq "$(wc -l < "$ON_DIR/legacy_steps.log" | tr -d ' ')" 0 "no legacy restore/selector"
  assert_file_contains "$ON_DIR/onstart.log" 'recipe=fast \(source=default\)' "recipe logged"
  end

  begin "onstart: ctl failure -> exactly one FAILED marker"
  run_onstart fail1 MT_TEST_CTL_RC=1
  assert_rc "$ON_RC" nonzero "onstart"
  assert_eq "$(marker_count "$ON_DIR" 'VAST_ONSTART FAILED')" 1 "FAILED marker"
  assert_eq "$(marker_count "$ON_DIR" 'VAST_ONSTART COMPLETE')" 0 "no COMPLETE"
  end

  begin "onstart: AUTO_SETUP=0 + check exit 10 -> FAILED, no install; exit 11 -> continue"
  run_onstart auto0 AUTO_SETUP=0 MT_TEST_CHECK_RC=10
  assert_rc "$ON_RC" nonzero "exit 10"
  assert_eq "$(marker_count "$ON_DIR" 'VAST_ONSTART FAILED')" 1 "FAILED marker"
  assert_file_contains "$ON_DIR/onstart.log" 'needs a clean install' "message"
  assert_eq "$(grep -vc '"--check"' "$ON_DIR/install.jsonl" || true)" 0 "no install with AUTO_SETUP=0"
  run_onstart auto0b AUTO_SETUP=0 MT_TEST_CHECK_RC=11
  assert_rc "$ON_RC" 0 "exit 11 tolerated with AUTO_SETUP=0"
  end

  begin "onstart: AUTO_SETUP=1 repair in place (11), clean install (10 / SETUP_CLEAN=1)"
  run_onstart repair MT_TEST_CHECK_RC=11
  assert_rc "$ON_RC" 0 "repair"
  assert_file_contains "$ON_DIR/install.jsonl" '"argv": \["--venv", "[^"]*"\]' "install without --clean"
  assert_file_lacks "$ON_DIR/install.jsonl" '"--clean"' "no clean on repair"
  run_onstart clean MT_TEST_CHECK_RC=10
  assert_file_contains "$ON_DIR/install.jsonl" '"--clean"' "clean on exit 10"
  run_onstart clean2 SETUP_CLEAN=1
  assert_file_contains "$ON_DIR/install.jsonl" '"--clean"' "clean on SETUP_CLEAN=1"
  run_onstart imports SETUP_CHECK_IMPORTS=1
  assert_file_contains "$ON_DIR/install.jsonl" '"--check", "--check-imports", "--venv"' "SETUP_CHECK_IMPORTS=1"
  run_onstart groups SETUP_FULL_STACK=1 SETUP_CHIN_TOOLS=1 SETUP_KOKORO=0 SETUP_SKIP_WEIGHTS=1 MT_TEST_CHECK_RC=11
  assert_file_contains "$ON_DIR/install.jsonl" '"--check", "--venv", "[^"]*", "--with-avatar-prep", "--without-kokoro", "--with-chin-tools"' "groups on check"
  assert_file_contains "$ON_DIR/install.jsonl" '"--with-avatar-prep", "--without-kokoro", "--with-chin-tools".*"--skip-weights"' "groups on install"
  end

  begin "onstart: fast300 provisions unet_ts + taesd_trt + unet_stagewise"
  run_onstart f300 MUSETALK_RECIPE=fast300 MUSETALK_UNET_STAGEWISE_PROVISION=adopt
  assert_rc "$ON_RC" 0 "onstart"
  assert_file_contains "$ON_DIR/ensure.log" '^ensure --kind unet_ts --provision auto$' "unet_ts"
  assert_file_contains "$ON_DIR/ensure.log" '^ensure --kind taesd_trt --provision auto$' "taesd_trt"
  assert_file_contains "$ON_DIR/ensure.log" '^ensure --kind unet_stagewise --provision adopt$' "unet_stagewise"
  end

  begin "onstart: no usable engine is fatal only with MUSETALK_UNET_MODE=trt"
  run_onstart noeng MT_TEST_ENSURE_RC=3
  assert_rc "$ON_RC" 0 "auto mode tolerates exit 3"
  assert_file_contains "$ON_DIR/onstart.log" 'No usable unet_ts engine' "warning"
  printf 'MUSETALK_UNET_MODE=trt\n' > "$O/.runtime/musetalk_overrides.env"
  run_onstart noeng_trt MT_TEST_ENSURE_RC=3
  assert_rc "$ON_RC" nonzero "trt mode (from the overrides file)"
  assert_file_contains "$ON_DIR/ensure.log" '--require' "--require passed"
  assert_eq "$(marker_count "$ON_DIR" 'VAST_ONSTART FAILED')" 1 "FAILED marker"
  : > "$O/.runtime/musetalk_overrides.env"
  printf 'MUSETALK_UNET_MODE=trt\n' > "$O/.runtime/musetalk_overrides.env"
  run_onstart noeng_trt2 MT_TEST_ENSURE_RC=2
  assert_rc "$ON_RC" nonzero "trt mode, ensure --require exit 2"
  : > "$O/.runtime/musetalk_overrides.env"
  run_onstart ensure_fail MT_TEST_ENSURE_RC=1
  assert_rc "$ON_RC" 0 "a failed ensure is non-fatal in auto mode"
  assert_file_contains "$ON_DIR/onstart.log" 'exited 1 .*continuing \(non-fatal\)' "warning"
  run_onstart provoff MUSETALK_UNET_ENGINE_PROVISION=off
  assert_eq "$(wc -l < "$ON_DIR/ensure.log" | tr -d ' ')" 0 "provision=off skips the store"
  end

  begin "onstart: engine-store roots from the overrides file reach ensure (caller still wins)"
  # (base_env exports MUSETALK_UNET_ENGINE_STORE itself, so the TAESD store root is the probe here)
  printf 'MUSETALK_TAESD_TRT_ENGINE_STORE=/engines/taesd\nMUSETALK_UNET_ENGINE_REMOTE=file:///share\n' \
    > "$O/.runtime/musetalk_overrides.env"
  run_onstart storeroot
  assert_rc "$ON_RC" 0 "onstart"
  assert_file_contains "$ON_DIR/ensure.log.env" '^store=/engines/taesd remote=file:///share$' "overrides forwarded"
  run_onstart storeroot_caller MUSETALK_TAESD_TRT_ENGINE_STORE=/caller/taesd
  assert_file_contains "$ON_DIR/ensure.log.env" '^store=/caller/taesd remote=file:///share$' "caller wins"
  : > "$O/.runtime/musetalk_overrides.env"
  end

  begin "onstart: legacy_int8 (overrides line) -> old restore + selector, legacy ctl env, no ensure"
  printf 'MUSETALK_RECIPE=legacy_int8\n' > "$O/.runtime/musetalk_overrides.env"
  run_onstart legacy MUSETALK_TRT_ARTIFACT_RESTORE=auto
  assert_rc "$ON_RC" 0 "onstart"
  assert_file_contains "$ON_DIR/onstart.log" 'TRT artifact restore skipped' "restore path ran (auto, no URI)"
  assert_file_contains "$ON_DIR/legacy_steps.log" '^select --output ' "selector ran"
  assert_eq "$(wc -l < "$ON_DIR/ensure.log" | tr -d ' ')" 0 "no engine store for legacy"
  assert_eq "$(dump_get "$ON_DIR/ctl.json" MUSETALK_RECIPE)" legacy_int8 "recipe to ctl"
  assert_eq "$(dump_get "$ON_DIR/ctl.json" PROFILE)" throughput_record "PROFILE for legacy"
  assert_file_contains "$ON_DIR/ctl.json" '"MUSETALK_TRT_PROFILE_ENV_FILE": "[^"]*musetalk_trt_best.env"' "profile env for legacy"
  assert_file_contains "$ON_DIR/install.jsonl" '"--with-legacy-int8"' "legacy install group"
  : > "$O/.runtime/musetalk_overrides.env"
  end

  begin "onstart: invalid setting dies with a FAILED marker (die inside a function)"
  run_onstart badval SETUP_KOKORO=maybe
  assert_rc "$ON_RC" nonzero "onstart"
  assert_eq "$(marker_count "$ON_DIR" 'VAST_ONSTART FAILED')" 1 "FAILED marker"
  run_onstart badrecipe MUSETALK_RECIPE=turbo
  assert_eq "$(marker_count "$ON_DIR" 'VAST_ONSTART FAILED')" 1 "FAILED marker for a bad recipe"
  end
fi

# ================================================================ relay wrapper
if selected relay; then
  R="$(make_repo relay)"
  printf 'TURN_PUBLIC_IP=203.0.113.9\nTURN_PUBLIC_PORT=3478\nTURN_PUBLIC_TRANSPORT=udp\nTURN_LISTEN_PORT=3478\nTURN_PASS=relaysecret\n' \
    > "$TEST_ROOT/turn.env"
  run_relay() {
    local out="$1" rc=0
    shift
    local -a envv=()
    mapfile -t envv < <(base_env "$R")
    # explicit Vast port mappings: never read this host's /proc/1/environ values
    env -i "${envv[@]}" REPO_ROOT="$R" TURN_ENV_FILE="$TEST_ROOT/turn.env" PUBLIC_IPADDR=203.0.113.9 \
      VAST_TCP_PORT_1455=41455 VAST_UDP_PORT_3478=43478 "$@" \
      timeout 60 bash "$R/scripts/run_webrtc_relay_api_server.sh" --host 127.0.0.1 --port 18003 \
        --venv-path "$FAKE_VENV" --repo-root "$R" > "$out" 2>&1 || rc=$?
    return "$rc"
  }

  begin "relay: execs run_musetalk_server.sh; overrides still set WEBRTC_SYNC_MODE for the fast chain"
  rm -f "$TEST_ROOT/api_env.json"
  printf 'WEBRTC_SYNC_MODE=latest_frame\n' > "$R/.runtime/musetalk_overrides.env"
  rc=0
  run_relay "$TEST_ROOT/relay1.out" || rc=$?
  assert_rc "$rc" 0 "relay launch"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" WEBRTC_SYNC_MODE)" latest_frame "overrides beat the relay defaults"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" WEBRTC_ICE_TRANSPORT_POLICY)" relay "relay policy"
  assert_eq "$(dump_get "$TEST_ROOT/api_env.json" WEBRTC_TURN_URLS)" \
    "turn:203.0.113.9:3478?transport=udp,turn:203.0.113.9:41455?transport=tcp" "turn urls"
  assert_file_lacks "$TEST_ROOT/relay1.out" relaysecret "TURN secret not logged"
  end

  begin "relay: legacy recipe keeps the old WebRTC defaults and reaches the legacy launcher"
  rm -f "$TEST_ROOT/legacy_env.json"
  printf 'MUSETALK_RECIPE=legacy_int8\n' > "$R/.runtime/musetalk_overrides.env"
  rc=0
  run_relay "$TEST_ROOT/relay2.out" || rc=$?
  assert_rc "$rc" 0 "relay legacy"
  assert_eq "$(dump_get "$TEST_ROOT/legacy_env.json" WEBRTC_SYNC_MODE)" strict_fifo "legacy default kept"
  assert_eq "$(dump_get "$TEST_ROOT/legacy_env.json" WEBRTC_AUDIO_PREBUFFER_SECONDS)" 0.0 "legacy audio prebuffer"
  end

  begin "relay: MUSETALK_SERVER_LAUNCHER override"
  : > "$R/.runtime/musetalk_overrides.env"
  printf '#!/usr/bin/env bash\n"%s/stub_record.py" "%s/custom_launcher.json" "$@"\n' "$R" "$TEST_ROOT" > "$R/custom_launcher.sh"
  rc=0
  run_relay "$TEST_ROOT/relay3.out" MUSETALK_SERVER_LAUNCHER="$R/custom_launcher.sh" || rc=$?
  assert_rc "$rc" 0 "custom launcher"
  assert_eq "$(dump_argv "$TEST_ROOT/custom_launcher.json")" \
    "--host 127.0.0.1 --port 18003 --venv-path $FAKE_VENV --repo-root $R" "args forwarded"
  end
fi

# ================================================================ setup_musetalk.sh shim
if selected shim; then
  S="$(make_repo shim)"
  begin "shim: legacy flags translated to install_musetalk.sh"
  printf '#!/usr/bin/env bash\n"%s/stub_record.py" "%s/shim.json" "$@"\n' "$S" "$TEST_ROOT" > "$S/scripts/install_musetalk.sh"
  rc=0
  env -i PATH="$SAFE_PATH" bash "$S/setup_musetalk.sh" --venv-path /v --full-stack --install-modelopt \
    --artifact-dir /a --clean --skip-apt --skip-weights --python-bin python3.10 --skip-modelopt --with-kokoro \
    > "$TEST_ROOT/shim.out" 2>&1 || rc=$?
  assert_rc "$rc" 0 "shim"
  assert_eq "$(dump_argv "$TEST_ROOT/shim.json")" \
    "--venv /v --with-legacy-int8 --clean --skip-apt --skip-weights --python python3.10 --with-kokoro --with-avatar-prep" "translated args"
  assert_file_contains "$TEST_ROOT/shim.out" 'ignoring --artifact-dir' "artifact-dir notice"
  rc=0
  env -i PATH="$SAFE_PATH" bash "$S/scripts/setup_musetalk.sh" --install-avatar-prep-deps --venv-path /w \
    > "$TEST_ROOT/shim2.out" 2>&1 || rc=$?
  assert_rc "$rc" 0 "scripts/ shim"
  assert_eq "$(dump_argv "$TEST_ROOT/shim.json")" "--venv /w --with-avatar-prep" "scripts/ shim args"
  end
fi

# ================================================================ integration (real resolver)
if selected integration; then
  if [[ ! -f "$REAL_REPO/scripts/musetalk_host_profile.py" ]]; then
    skip "integration: real resolver" "scripts/musetalk_host_profile.py not in the tree yet"
  else
    I="$(make_repo integration real)"
    STORE="$TEST_ROOT/store"
    mkdir -p "$STORE"
    ikey() {
      local -a envv=()
      mapfile -t envv < <(base_env "$I")
      env -i "${envv[@]}" "$FAKE_VENV/bin/python" -B "$I/scripts/musetalk_host_profile.py" engine-key \
        --kind unet_ts --repo-root "$I" --venv "$FAKE_VENV" 2>/dev/null | tail -n 1 | tr -d '[:space:]'
    }
    make_fake_engine() {
      # make_fake_engine KEY -> prints the .ts path. A validated unet_ts store entry with a tiny fake
      # .ts that carries the embedded device string; written through musetalk_engine_keys when that
      # module is present so the record matches the real schema, else per the contract.
      local key="$1" dir="$STORE/$1/bs8"
      mkdir -p "$dir"
      printf 'FAKE\0000%%8%%9%%0%%NVIDIA GeForce RTX 4070 SUPER\0FAKE' > "$dir/unet_trt.ts"
      printf '{"batch_range": [8, 8], "validation": {"passed": true}}\n' > "$dir/unet_trt_meta.json"
      printf '{"passed": true, "mae_max": 0.0026, "max_abs_max": 0.40}\n' > "$dir/validation.json"
      if [[ -f "$I/scripts/musetalk_engine_keys.py" ]]; then
        "$TEST_PY" -B - "$I/scripts" "$TEST_ROOT/facts.json" "$dir" <<'PY' || fail "fake engine via musetalk_engine_keys"
import json, os, sys
sys.path.insert(0, sys.argv[1])
import musetalk_engine_keys as k
facts = json.load(open(sys.argv[2]))
directory = sys.argv[3]
engine = os.path.join(directory, "unet_trt.ts")
fp = k.make_fingerprint("unet_ts", facts, 8, "built", engine_bytes=os.path.getsize(engine),
                        embedded_device="0%8%9%0%NVIDIA GeForce RTX 4070 SUPER")
fp["validation"] = {"passed": True, "status": "passed", "engine_key": fp["engine_key"],
                    "validated_utc": k.utc_now(), "mae_max": 0.0026, "max_abs_max": 0.40,
                    "capture_dir": "calibration/unet_portable_bs8", "files": 16}
k.write_fingerprint(directory, fp)
PY
      else
        cat > "$dir/fingerprint.json" <<EOF
{"schema": "musetalk_unet_engine_v1", "kind": "unet_ts", "engine_key": "$key",
 "gpu_name": "NVIDIA GeForce RTX 4070 SUPER", "compute_capability": "8.9",
 "tensorrt_version": "10.3.0", "torch_tensorrt_version": "2.5.0", "torch_version": "2.5.1+cu121",
 "batch": 8, "engine_file": "unet_trt.ts", "engine_bytes": $(stat -c %s "$dir/unet_trt.ts"),
 "engine_sha256": null, "embedded_device": "0%8%9%0%NVIDIA GeForce RTX 4070 SUPER",
 "source": "built", "created_utc": "2026-09-28T00:00:00Z",
 "validation": {"passed": true, "status": "passed", "engine_key": "$key", "mae_max": 0.0026,
                "max_abs_max": 0.40, "capture_dir": "x", "files": 16}}
EOF
      fi
      printf '%s' "$dir/unet_trt.ts"
    }

    begin "integration: no engine -> eager UNet, TAESD, fallback off, coupled 8/8/8/8/8"
    rc=0
    run_launcher "$I" "$TEST_ROOT/i1.out" -- --print-env || rc=$?
    assert_rc "$rc" 0 "print-env"
    assert_eq "$(pe_get "$TEST_ROOT/i1.out" MUSETALK_RECIPE)" fast "recipe"
    assert_eq "$(pe_get "$TEST_ROOT/i1.out" MUSETALK_VAE_BACKEND)" taesd "vae"
    assert_eq "$(pe_get "$TEST_ROOT/i1.out" MUSETALK_UNET_BACKEND)" eager "unet"
    assert_eq "$(pe_get "$TEST_ROOT/i1.out" MUSETALK_TRT_UNET_ENABLED)" 0 "trt unet disabled"
    assert_eq "$(pe_get "$TEST_ROOT/i1.out" MUSETALK_TRT_FALLBACK)" 0 "fallback"
    assert_eq "$(pe_get "$TEST_ROOT/i1.out" MUSETALK_TRT_ENABLED)" 0 "legacy VAE TRT off"
    for k in HLS_SCHEDULER_FIXED_BATCH_SIZES MUSETALK_TAESD_WARMUP_BATCHES MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES \
             HLS_SCHEDULER_MAX_BATCH HLS_SCHEDULER_STARTUP_SLICE_SIZE; do
      assert_eq "$(pe_get "$TEST_ROOT/i1.out" "$k")" 8 "$k"
    done
    assert_eq "$(pe_get "$TEST_ROOT/i1.out" WEBRTC_VP8_ENCODER)" pyav "vp8 default"
    assert_file_contains "$TEST_ROOT/i1.out" '^# expect: vae=taesd unet=eager' "expectation"
    end

    begin "integration: validated engine for this key -> TRT UNet with absolute bs8 path"
    key="$(ikey)"
    [[ -n "$key" ]] || key="sm89-nvidia-geforce-rtx-4070-super-trt10.3.0-tt2.5.0"
    ts="$(make_fake_engine "$key")"
    rc=0
    run_launcher "$I" "$TEST_ROOT/i2.out" "MUSETALK_UNET_ENGINE_STORE=$STORE" -- --print-env || rc=$?
    assert_rc "$rc" 0 "print-env"
    assert_eq "$(pe_get "$TEST_ROOT/i2.out" MUSETALK_UNET_BACKEND)" trt "unet"
    assert_eq "$(pe_get "$TEST_ROOT/i2.out" MUSETALK_TRT_UNET_ENABLED)" 1 "trt unet enabled"
    assert_eq "$(pe_get "$TEST_ROOT/i2.out" MUSETALK_TRT_UNET_PATHS)" "8:$ts" "engine path"
    assert_file_contains "$TEST_ROOT/i2.out" '^# expect: vae=taesd unet=trt' "expectation"
    end

    begin "integration: caller buckets 16 -> coupled warmups 16, max 16, slice 8"
    rc=0
    run_launcher "$I" "$TEST_ROOT/i3.out" "MUSETALK_UNET_ENGINE_STORE=$STORE" HLS_SCHEDULER_FIXED_BATCH_SIZES=16 -- --print-env || rc=$?
    assert_rc "$rc" 0 "print-env"
    assert_eq "$(pe_get "$TEST_ROOT/i3.out" MUSETALK_UNET_BACKEND)" trt "unet"
    assert_eq "$(pe_get "$TEST_ROOT/i3.out" MUSETALK_TAESD_WARMUP_BATCHES)" 16 "taesd warmup"
    assert_eq "$(pe_get "$TEST_ROOT/i3.out" MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES)" 16 "stagewise warmup"
    assert_eq "$(pe_get "$TEST_ROOT/i3.out" HLS_SCHEDULER_MAX_BATCH)" 16 "max batch"
    assert_eq "$(pe_get "$TEST_ROOT/i3.out" HLS_SCHEDULER_STARTUP_SLICE_SIZE)" 8 "slice"
    assert_eq "$(pe_get "$TEST_ROOT/i3.out" HLS_SCHEDULER_FIXED_BATCH_SIZES source)" caller "bucket source"
    end

    begin "integration: overrides buckets 12 with an engine in auto mode -> eager (not a multiple of 8)"
    printf 'HLS_SCHEDULER_FIXED_BATCH_SIZES=12\n' > "$I/.runtime/musetalk_overrides.env"
    rc=0
    run_launcher "$I" "$TEST_ROOT/i4.out" "MUSETALK_UNET_ENGINE_STORE=$STORE" -- --print-env || rc=$?
    assert_rc "$rc" 0 "print-env"
    assert_eq "$(pe_get "$TEST_ROOT/i4.out" MUSETALK_UNET_BACKEND)" eager "unet"
    assert_eq "$(pe_get "$TEST_ROOT/i4.out" MUSETALK_TAESD_WARMUP_BATCHES)" 12 "warmup follows the overrides buckets"
    rc=0
    run_launcher "$I" "$TEST_ROOT/i4b.out" "MUSETALK_UNET_ENGINE_STORE=$STORE" MUSETALK_UNET_MODE=trt -- --print-env || rc=$?
    assert_rc "$rc" nonzero "trt mode with invalid buckets is a hard error"
    : > "$I/.runtime/musetalk_overrides.env"
    end

    begin "integration: MUSETALK_UNET_MODE=trt without an engine is a hard error"
    rc=0
    run_launcher "$I" "$TEST_ROOT/i5.out" MUSETALK_UNET_MODE=trt -- --print-env || rc=$?
    assert_rc "$rc" nonzero "launch"
    assert_file_contains "$TEST_ROOT/i5.out.err" 'Resolver refused' "message"
    end

    begin "integration: caller trt_stagewise is respected (no .ts paths emitted)"
    mkdir -p "$I/models/tensorrt_unet_stagewise_sm89/bs16"
    printf '{"schema": "musetalk_unet_stagewise_trt_v1", "batch": 16, "complete": true}\n' \
      > "$I/models/tensorrt_unet_stagewise_sm89/bs16/manifest.json"
    rc=0
    run_launcher "$I" "$TEST_ROOT/i6.out" "MUSETALK_UNET_ENGINE_STORE=$STORE" MUSETALK_UNET_BACKEND=trt_stagewise -- --print-env || rc=$?
    assert_rc "$rc" 0 "print-env"
    assert_eq "$(pe_get "$TEST_ROOT/i6.out" MUSETALK_UNET_BACKEND)" trt_stagewise "unet"
    assert_eq "$(pe_get "$TEST_ROOT/i6.out" MUSETALK_UNET_BACKEND source)" caller "unet source"
    [[ "$(pe_get "$TEST_ROOT/i6.out" MUSETALK_TRT_UNET_PATHS)" == "<unset>" ]] || fail "resolver emitted .ts paths for trt_stagewise"
    end

    begin "integration: fast300 with every lever commented = fast (+ recipe name)"
    rc=0
    run_launcher "$I" "$TEST_ROOT/i7.out" MUSETALK_RECIPE=fast300 -- --print-env || rc=$?
    assert_rc "$rc" 0 "print-env"
    assert_eq "$(pe_get "$TEST_ROOT/i7.out" MUSETALK_RECIPE)" fast300 "recipe"
    assert_eq "$(pe_get "$TEST_ROOT/i7.out" MUSETALK_TAESD_BACKEND)" "<unset>" "no TAESD TRT lever"
    assert_eq "$(pe_get "$TEST_ROOT/i7.out" WEBRTC_VP8_ENCODER)" pyav "vp8 stays pyav"
    assert_eq "$(pe_get "$TEST_ROOT/i7.out" HLS_GPU_PIPELINE_DEPTH)" "<unset>" "no serving lever"
    end

    begin "integration: overrides value is reported by the resolver (the report shows the truth)"
    printf 'AVATAR_CACHE_MAX_MEMORY_MB=1234\n' > "$I/.runtime/musetalk_overrides.env"
    rc=0
    run_launcher "$I" "$TEST_ROOT/i8.out" -- || rc=$?
    assert_rc "$rc" 0 "launch"
    assert_eq "$(dump_get "$TEST_ROOT/api_env.json" AVATAR_CACHE_MAX_MEMORY_MB)" 1234 "effective"
    assert_eq "$(json_get "$I/.runtime/musetalk_resolved.json" emitted.AVATAR_CACHE_MAX_MEMORY_MB)" 1234 "report"
    assert_eq "$(json_get "$I/.runtime/musetalk_launch_18001.json" env.AVATAR_CACHE_MAX_MEMORY_MB.source)" \
      "overrides:$I/.runtime/musetalk_overrides.env" "launch state source"
    : > "$I/.runtime/musetalk_overrides.env"
    end

    begin "integration: no GPU is a hard error unless MUSETALK_ALLOW_NO_GPU=1"
    write_facts "$TEST_ROOT/facts_nogpu.json" "[]"
    rc=0
    run_launcher "$I" "$TEST_ROOT/i9.out" "MUSETALK_HOST_FACTS_JSON=$TEST_ROOT/facts_nogpu.json" -- --print-env || rc=$?
    assert_rc "$rc" nonzero "no gpu"
    rc=0
    run_launcher "$I" "$TEST_ROOT/i9b.out" "MUSETALK_HOST_FACTS_JSON=$TEST_ROOT/facts_nogpu.json" MUSETALK_ALLOW_NO_GPU=1 -- --print-env || rc=$?
    assert_rc "$rc" 0 "allowed"
    assert_eq "$(pe_get "$TEST_ROOT/i9b.out" MUSETALK_UNET_BACKEND)" eager "eager without GPU"
    end

    begin "integration: verify-log exit codes 0 match / 1 mismatch / 3 not found"
    vlog="$TEST_ROOT/verify.log"
    printf 'old junk\n' > "$vlog"
    off="$(stat -c %s "$vlog")"
    printf '✅ VAE decode backend active: taesd\n✅ UNet backend active: tensorrt_unet_multi\n' >> "$vlog"
    verify() {
      local rc=0
      env -i PATH="$SAFE_PATH" "$FAKE_VENV/bin/python" -B "$I/scripts/musetalk_host_profile.py" verify-log \
        --log "$vlog" --offset "$1" --expect-vae "$2" --expect-unet "$3" --timeout 1 > "$TEST_ROOT/verify.out" 2>&1 || rc=$?
      printf '%s' "$rc"
    }
    assert_eq "$(verify "$off" taesd trt)" 0 "match"
    assert_eq "$(verify "$off" taesd eager)" 1 "unet mismatch"
    printf '✅ VAE decode backend active: taesd\nℹ️  UNet backend: PyTorch\n' > "$vlog"
    assert_eq "$(verify 0 taesd eager)" 0 "eager wording"
    assert_eq "$(verify 0 taesd any)" 0 "any"
    printf 'ℹ️  VAE decode backend: PyTorch\nℹ️  UNet backend: PyTorch\n' > "$vlog"
    assert_eq "$(verify 0 taesd any)" 1 "silent SD-VAE fallback is a mismatch"
    printf 'nothing here\n' > "$vlog"
    assert_eq "$(verify 0 taesd any)" 3 "not found"
    end
  fi
fi

# ================================================================ summary
echo
printf 'test_startup_scripts: %d passed, %d failed, %d skipped\n' "$PASS" "$FAIL" "$SKIP"
if (( FAIL > 0 )); then
  printf '  failed: %s\n' "${FAILED_NAMES[@]}"
  exit 1
fi
exit 0
