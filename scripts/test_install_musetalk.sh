#!/usr/bin/env bash
# test_install_musetalk.sh - CPU-only tests for the installer component:
#   scripts/install_musetalk.sh (--plan / --check / install-mode guards), scripts/musetalk_install_state.py,
#   the download_weights.sh TAESD + Kokoro additions, and requirements/*.
#
# No network, no GPU, no torch: fake venvs (python3.10 -m venv --without-pip + synthetic dist-info),
# injected host facts (MUSETALK_HOST_FACTS_JSON), stubbed huggingface-cli/curl/gdown/pip. Everything is
# created under one temp dir (TMPDIR) and removed at exit. Optional last test: --check against the real
# server venv (read-only) when it exists; skip with TEST_LIVE_VENV=0.
#
# Usage: bash scripts/test_install_musetalk.sh      (exit 0 = all tests passed)
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$SCRIPT_DIR/.." && pwd)"
PY310="$(command -v python3.10 || true)"
[[ -n "$PY310" ]] || { echo "SKIP: python3.10 not found"; exit 0; }
T="$(mktemp -d "${TMPDIR:-/tmp}/musetalk-install-test-XXXXXX")"
trap 'rm -rf "$T"' EXIT
export STEP_LOG_ROOT="$T/logs"
unset MUSETALK_HOST_FACTS_JSON MUSETALK_NVIDIA_SMI CUDA_VISIBLE_DEVICES PIP_CONSTRAINT || true

PASS=0
FAIL=0
N=0
log() { printf '[test_install %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
ok() { PASS=$((PASS + 1)); printf 'PASS %s\n' "$1"; }
bad() { FAIL=$((FAIL + 1)); printf 'FAIL %s: %s\n' "$1" "$2"; }

# run NAME EXPECTED_EXIT cmd... ; output kept in $T/out.<N> ($LAST_OUT)
run() {
  local name="$1" expected="$2"; shift 2
  N=$((N + 1)); LAST_OUT="$T/out.$N"
  local status=0
  "$@" >"$LAST_OUT" 2>&1 || status=$?
  if [[ "$status" == "$expected" ]]; then ok "$name (exit $status)"; else bad "$name" "exit $status, expected $expected; tail: $(tail -n 5 "$LAST_OUT" | tr '\n' '|')"; fi
}
has() {  # has NAME PATTERN  (grep -E in $LAST_OUT)
  if grep -Eq -- "$2" "$LAST_OUT"; then ok "$1"; else bad "$1" "pattern not found: $2"; fi
}
hasnt() {
  if grep -Eq -- "$2" "$LAST_OUT"; then bad "$1" "unexpected pattern: $2"; else ok "$1"; fi
}

# ------------------------------------------------------------------------- fixtures
FAKE="$T/repo"
mkdir -p "$FAKE/scripts/lib" "$FAKE/requirements"
cp "$REPO/scripts/install_musetalk.sh" "$REPO/scripts/musetalk_install_state.py" "$REPO/scripts/install_native_vp8.py" \
   "$REPO/scripts/native_vp8_manifest.json" "$REPO/scripts/musetalk_selftest.py" "$FAKE/scripts/"
cp "$REPO/scripts/lib/step_logging.sh" "$FAKE/scripts/lib/"
cp "$REPO/requirements/"* "$FAKE/requirements/"
cp "$REPO/download_weights.sh" "$FAKE/"
MODEL_FILES=(musetalkV15/musetalk.json musetalkV15/unet.pth sd-vae/config.json sd-vae/diffusion_pytorch_model.bin
             whisper/config.json whisper/pytorch_model.bin whisper/preprocessor_config.json
             face-parse-bisent/79999_iter.pth face-parse-bisent/resnet18-5c106cde.pth
             taesd/config.json taesd/diffusion_pytorch_model.safetensors)
for rel in "${MODEL_FILES[@]}"; do mkdir -p "$FAKE/models/$(dirname "$rel")"; echo x >"$FAKE/models/$rel"; done

printf '{"gpus":[{"index":0,"name":"NVIDIA GeForce RTX 4070 SUPER","compute_capability":"8.9"}]}\n' >"$T/facts_89.json"
printf '{"gpus":[{"index":0,"name":"NVIDIA RTX PRO 6000 Blackwell Workstation Edition","compute_capability":"12.0"}]}\n' >"$T/facts_120.json"
printf '{"gpus":[]}\n' >"$T/facts_none.json"

MODULES=(torch torchvision torchaudio torch_tensorrt tensorrt onnx diffusers transformers cv2 numpy aiortc av cffi
         fastapi uvicorn boto3 librosa soundfile numba imageio ffmpeg omegaconf multipart
         kokoro misaki spacy en_core_web_sm espeakng_loader)

# make_fake_venv DIR MATRIX [python]: venv without pip + dist-info for every pin of the matrix
make_fake_venv() {
  local dir="$1" matrix="$2" py="${3:-$PY310}"
  "$py" -m venv --without-pip "$dir" >/dev/null
  "$PY310" - "$dir" "$FAKE/requirements/constraints-$matrix.txt" "${MODULES[@]}" <<'PY'
import glob, os, re, sys
venv, constraints, modules = sys.argv[1], sys.argv[2], sys.argv[3:]
sp = glob.glob(os.path.join(venv, "lib", "python3.*", "site-packages"))[0]
def dist(name, version):
    d = os.path.join(sp, f"{re.sub(r'[-.]+', '_', name)}-{version}.dist-info")
    os.makedirs(d, exist_ok=True)
    open(os.path.join(d, "METADATA"), "w").write(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
for line in open(constraints):
    line = line.split("#", 1)[0].strip()
    m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*===?\s*(\S+)$", line)
    if m:
        dist(m.group(1), m.group(2))
dist("en_core_web_sm", "3.8.0")
dist("mmcv", "2.1.0")
for mod in modules:
    os.makedirs(os.path.join(sp, mod), exist_ok=True)
    open(os.path.join(sp, mod, "__init__.py"), "w").write("")
PY
}
set_dist_version() {  # set_dist_version VENV NAME NEWVERSION (renames + rewrites METADATA)
  "$PY310" - "$@" <<'PY'
import glob, os, re, sys
venv, name, new = sys.argv[1:4]
sp = glob.glob(os.path.join(venv, "lib", "python3.*", "site-packages"))[0]
key = re.sub(r"[-_.]+", "_", name).lower()
for d in glob.glob(os.path.join(sp, "*.dist-info")):
    stem = os.path.basename(d)[:-10]
    if re.sub(r"[-_.]+", "_", stem.rsplit("-", 1)[0]).lower() == key:
        target = os.path.join(sp, f"{stem.rsplit('-', 1)[0]}-{new}.dist-info")
        os.rename(d, target)
        open(os.path.join(target, "METADATA"), "w").write(f"Metadata-Version: 2.1\nName: {name}\nVersion: {new}\n")
PY
}
remove_dist() {
  local sp; sp="$(echo "$1"/lib/python3.*/site-packages)"
  rm -rf "$sp"/"$2"-*.dist-info
}

INSTALL=(bash "$FAKE/scripts/install_musetalk.sh" --repo-root "$FAKE")
HELPER=("$PY310" -B "$FAKE/scripts/musetalk_install_state.py")

# ------------------------------------------------------------------------- 1. syntax
log "1. syntax"
run "bash -n install_musetalk.sh" 0 bash -n "$REPO/scripts/install_musetalk.sh"
run "bash -n download_weights.sh" 0 bash -n "$REPO/download_weights.sh"
run "bash -n test_install_musetalk.sh" 0 bash -n "$REPO/scripts/test_install_musetalk.sh"
run "py_compile helper + selftest" 0 "$PY310" -c "
import py_compile, sys
for f in sys.argv[1:]:
    py_compile.compile(f, cfile='$T/x.pyc', doraise=True)
" "$REPO/scripts/musetalk_install_state.py" "$REPO/scripts/musetalk_selftest.py"
run "helper and selftest never import torch at module level" 0 "$PY310" -c "
import ast, sys
for f in sys.argv[1:]:
    tree = ast.parse(open(f).read())
    top = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
    names = [a.name for n in top for a in n.names] + [n.module or '' for n in top if isinstance(n, ast.ImportFrom)]
    bad = [n for n in names if n.split('.')[0] in ('torch', 'tensorrt', 'torch_tensorrt')]
    assert not bad, (f, bad)
" "$REPO/scripts/musetalk_install_state.py" "$REPO/scripts/musetalk_selftest.py"

# ------------------------------------------------------------------------- 2. requirements
log "2. requirements"
run "requirements coverage and matrix pins" 0 "$PY310" - "$REPO/requirements" <<'PY'
import re, sys
from pathlib import Path
req = Path(sys.argv[1])
norm = lambda n: re.sub(r"[-_.]+", "-", n).lower()
def pins(path):
    out = {}
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._-]*)\s*(===?)\s*(\S+)$", line)
        if m: out[norm(m.group(1))] = m.group(3)
    return out
def names(path):
    out = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or " @ " in line or "==" in line: continue  # URL / inline-pinned (mmcv wheel) lines
        out.append(norm(re.split(r"[\s;<>=!~\[]", line, 1)[0]))
    return out
c121, c128 = pins(req / "constraints-cu121.txt"), pins(req / "constraints-cu128.txt")
for group in ("server.in", "kokoro.in", "legacy-int8.in", "avatar-prep.in"):
    for name in names(req / group):
        assert name in c121, f"{group}: {name} has no cu121 pin"
        if group not in ("legacy-int8.in", "avatar-prep.in"):
            assert name in c128, f"{group}: {name} has no cu128 pin"
assert c121["torch"] == "2.5.1+cu121" and c121["torch-tensorrt"] == "2.5.0" and c121["tensorrt-cu12"] == "10.3.0"
assert c128["torch"] == "2.7.1+cu128" and c128["torchvision"] == "0.22.1+cu128" and c128["torchaudio"] == "2.7.1+cu128"
assert c128["torch-tensorrt"] == "2.7.0+cu128" and c128["tensorrt-cu12"].startswith("10.9.")
assert not any(v.endswith("+cu121") for v in c128.values()), "cu121 pin leaked into cu128"
assert c121["aiortc"] == c128["aiortc"] == "1.14.0" and c121["av"] == "16.1.0" and c121["cffi"] == "2.1.1"
assert "torch_tensorrt===2.5.0" in (req / "constraints-cu121.txt").read_text()
chin = pins(req / "constraints-chin-tools.txt")
for line in (req / "chin-tools.in").read_text().splitlines():
    m = re.match(r"^([A-Za-z0-9._-]+)==(\S+)$", line.strip())
    if m: assert chin[norm(m.group(1))] == m.group(2), line
assert chin["mediapipe"] == "0.10.9" and chin["numpy"] == "2.2.6" and chin["protobuf"] == "3.20.3"
assert "en_core_web_sm @ https://" in (req / "kokoro.in").read_text() and "#sha256=" in (req / "kokoro.in").read_text()
print("requirements OK")
PY

# ------------------------------------------------------------------------- 3. plan / matrix
log "3. --plan matrix and group resolution"
V121="$T/venv121"; make_fake_venv "$V121" cu121
V128="$T/venv128"; make_fake_venv "$V128" cu128
run "plan sm89 -> cu121" 0 env MUSETALK_HOST_FACTS_JSON="$T/facts_89.json" "${INSTALL[@]}" --plan --venv "$T/none"
has "plan sm89 matrix" '^MATRIX=cu121$'
has "plan default groups" '^GROUP_KOKORO=1$'
has "plan pip command uses constraints + pytorch cu121 index" 'PIP_COMMAND=.*server\.in.*kokoro\.in.*constraints-cu121\.txt.*download\.pytorch\.org/whl/cu121.*pypi\.nvidia\.com'
hasnt "plan does not add legacy-int8 by default" 'legacy-int8\.in'
run "plan sm120 -> cu128" 0 env MUSETALK_HOST_FACTS_JSON="$T/facts_120.json" "${INSTALL[@]}" --plan --venv "$T/none"
has "plan sm120 matrix" '^MATRIX=cu128$'
has "plan sm120 index" '^PYTORCH_INDEX=https://download.pytorch.org/whl/cu128$'
run "plan no GPU -> cu121" 0 env MUSETALK_HOST_FACTS_JSON="$T/facts_none.json" "${INSTALL[@]}" --plan --venv "$T/none"
has "plan no GPU matrix" '^MATRIX=cu121$'
has "plan no GPU: self-test off" '^SELFTEST=0$'
run "plan no GPU keeps an installed cu128 venv" 0 env MUSETALK_HOST_FACTS_JSON="$T/facts_none.json" "${INSTALL[@]}" --plan --venv "$V128"
has "plan kept cu128" '^MATRIX=cu128$'
run "plan CUDA_VISIBLE_DEVICES='' hides the GPU" 0 env CUDA_VISIBLE_DEVICES= MUSETALK_HOST_FACTS_JSON="$T/facts_120.json" "${INSTALL[@]}" --plan --venv "$T/none"
has "plan hidden GPU -> cu121" '^MATRIX=cu121$'
run "plan explicit --matrix cu128" 0 env MUSETALK_HOST_FACTS_JSON="$T/facts_89.json" "${INSTALL[@]}" --plan --venv "$T/none" --matrix cu128
has "plan explicit matrix" '^MATRIX=cu128$'
run "plan --without-kokoro --with-legacy-int8" 0 env MUSETALK_HOST_FACTS_JSON="$T/facts_89.json" "${INSTALL[@]}" --plan --venv "$T/none" --without-kokoro --with-legacy-int8
hasnt "plan without kokoro" 'kokoro\.in'
has "plan with legacy-int8" 'legacy-int8\.in'
run "legacy flags: --full-stack --install-modelopt --artifact-dir" 0 env MUSETALK_HOST_FACTS_JSON="$T/facts_89.json" "${INSTALL[@]}" --plan --venv-path "$T/none" --python-bin python3.10 --full-stack --install-modelopt --artifact-dir /x
has "legacy --full-stack -> avatar-prep" '^GROUP_AVATAR_PREP=1$'
has "legacy --install-modelopt -> legacy-int8" '^GROUP_LEGACY_INT8=1$'
run "avatar-prep refused on cu128" 2 env MUSETALK_HOST_FACTS_JSON="$T/facts_120.json" "${INSTALL[@]}" --plan --venv "$T/none" --with-avatar-prep
run "legacy-int8 refused on cu128" 2 env MUSETALK_HOST_FACTS_JSON="$T/facts_89.json" "${INSTALL[@]}" --plan --venv "$T/none" --matrix cu128 --with-legacy-int8
run "bad --matrix value" 2 "${INSTALL[@]}" --plan --matrix cu999
run "unknown option" 2 "${INSTALL[@]}" --frobnicate
run "PYTORCH_INDEX_URL from the image is ignored" 0 env PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu999 MUSETALK_HOST_FACTS_JSON="$T/facts_89.json" "${INSTALL[@]}" --plan --venv "$T/none"
has "plan index not taken from PYTORCH_INDEX_URL" '^PYTORCH_INDEX=https://download.pytorch.org/whl/cu121$'

# ------------------------------------------------------------------------- 4. --check
log "4. --check exit codes (read-only)"
export MUSETALK_HOST_FACTS_JSON="$T/facts_89.json"
touch "$T/marker"; sleep 1
run "check complete pre-existing venv without stamp -> 0" 0 "${INSTALL[@]}" --check --venv "$V121"
has "check warns about the missing stamp" 'WARN stamp: no install stamp'
has "check verdict ok" 'verdict: ok'
has "check auto native VP8 missing is only a warning" 'WARN native_vp8'
run "check wrote nothing" 0 bash -c "! find '$FAKE' '$V121' -newer '$T/marker' -print | grep -q ."
run "check --json is valid JSON" 0 bash -c "\"\${@}\" --check --venv '$V121' --json 2>/dev/null | '$PY310' -c 'import json,sys; d=json.load(sys.stdin); assert d[\"verdict\"]==\"ok\" and d[\"exit_code\"]==0'" _ "${INSTALL[@]}"
run "check --report writes JSON" 0 "${INSTALL[@]}" --check --venv "$V121" --report "$T/report.json"
run "report schema" 0 "$PY310" -c "import json; d=json.load(open('$T/report.json')); assert d['schema']=='musetalk_install_check_v1' and d['groups']['kokoro']['enabled']"
run "check missing venv -> 10" 10 "${INSTALL[@]}" --check --venv "$T/does-not-exist"
run "check GPU needs cu128 but venv is cu121 -> 10" 10 env MUSETALK_HOST_FACTS_JSON="$T/facts_120.json" "${INSTALL[@]}" --check --venv "$V121"
has "check names the matrix problem" 'FAIL matrix: .*needs cu128'
run "check no GPU visible never demands a clean install" 0 env MUSETALK_HOST_FACTS_JSON="$T/facts_none.json" "${INSTALL[@]}" --check --venv "$V121"
has "check no GPU warns" 'WARN matrix: no GPU visible'
PY_OTHER=""
for candidate in python3.12 python3.11 python3.9 python3.8; do
  if command -v "$candidate" >/dev/null 2>&1; then PY_OTHER="$(command -v "$candidate")"; break; fi
done
if [[ -n "$PY_OTHER" ]]; then
  make_fake_venv "$T/venv_wrongpy" cu121 "$PY_OTHER"
  run "check wrong interpreter version -> 10" 10 "${INSTALL[@]}" --check --venv "$T/venv_wrongpy"
else
  log "SKIP wrong-interpreter test (no other python3.x found)"
fi
cp -a "$V121" "$T/venv_missing"; remove_dist "$T/venv_missing" torch_tensorrt; rm -rf "$T/venv_missing"/lib/python3.10/site-packages/torch_tensorrt
run "check missing package -> 11" 11 "${INSTALL[@]}" --check --venv "$T/venv_missing"
has "check lists the missing package" 'missing: torch-tensorrt'
cp -a "$V121" "$T/venv_pin"; set_dist_version "$T/venv_pin" numpy 1.26.4
run "check pin drift -> 11" 11 "${INSTALL[@]}" --check --venv "$T/venv_pin"
has "check names the drifted pin" 'numpy 1.26.4 != ==1.23.5'
cp -a "$V121" "$T/venv_tt"; set_dist_version "$T/venv_tt" torch_tensorrt 2.5.0+cu121
run "check torch_tensorrt local build != PyPI build (=== pin) -> 11" 11 "${INSTALL[@]}" --check --venv "$T/venv_tt"
cp -a "$V121" "$T/venv_nokokoro"; remove_dist "$T/venv_nokokoro" kokoro
run "check kokoro missing (default group on) -> 11" 11 "${INSTALL[@]}" --check --venv "$T/venv_nokokoro"
run "check kokoro missing with --without-kokoro -> 0" 0 "${INSTALL[@]}" --check --venv "$T/venv_nokokoro" --without-kokoro
mv "$FAKE/models/taesd/config.json" "$T/taesd_config.bak"
run "check missing TAESD weights -> 11" 11 "${INSTALL[@]}" --check --venv "$V121"
has "check names the TAESD file" 'models/taesd/config.json'
run "check missing weights with --skip-weights -> 0" 0 "${INSTALL[@]}" --check --venv "$V121" --skip-weights
mv "$T/taesd_config.bak" "$FAKE/models/taesd/config.json"
run "check explicit --with-native-vp8 but not provisioned -> 11" 11 "${INSTALL[@]}" --check --venv "$V121" --with-native-vp8
run "check --with-chin-tools without the venv -> 11" 11 "${INSTALL[@]}" --check --venv "$V121" --with-chin-tools --chin-venv "$T/chin"

# chin venv with the validated pins
"$PY310" -m venv --without-pip "$T/chin" >/dev/null
"$PY310" - "$T/chin" "$FAKE/requirements/constraints-chin-tools.txt" <<'PY'
import glob, os, re, sys
sp = glob.glob(os.path.join(sys.argv[1], "lib", "python3.*", "site-packages"))[0]
for line in open(sys.argv[2]):
    m = re.match(r"^([A-Za-z0-9._-]+)==(\S+)$", line.strip())
    if m:
        d = os.path.join(sp, f"{m.group(1).replace('-', '_')}-{m.group(2)}.dist-info"); os.makedirs(d)
        open(os.path.join(d, "METADATA"), "w").write(f"Name: {m.group(1)}\nVersion: {m.group(2)}\n")
PY
run "check --with-chin-tools with validated pins -> 0" 0 "${INSTALL[@]}" --check --venv "$V121" --with-chin-tools --chin-venv "$T/chin"

# stamp round trip
log "4b. stamp round trip"
STATE="$FAKE/.runtime/install_state.json"
run "stamp writes install_state.json" 0 "${HELPER[@]}" stamp --repo-root "$FAKE" --venv "$V121" --matrix cu121 \
  --groups-json '{"server": true, "kokoro": true, "chin_tools": true}' --chin-venv "$T/chin" --out "$STATE"
run "stamp content" 0 "$PY310" -c "
import json; d=json.load(open('$STATE'))
assert d['schema']=='musetalk_install_state_v1' and d['matrix']=='cu121'
assert d['exports']['MUSETALK_CHIN_TRACKER_PYTHON']=='$T/chin/bin/python', d['exports']
assert d['groups']['chin_tools']['versions']['mediapipe']=='0.10.9'
assert d['constraints_sha256'] and d['server_in_sha256'] and d['versions']['torch']=='2.5.1+cu121'
"
run "check with stamp (chin group from stamp) -> 0" 0 "${INSTALL[@]}" --check --venv "$V121"
has "check reads the stamp" 'ok   stamp: '
has "check verified chin tools from the stamp" 'ok   chin_tools'
hasnt "check prints no spurious ERR-trap line" 'command failed'
set_dist_version "$T/chin" mediapipe 0.10.14
run "check chin mediapipe drift -> 11" 11 "${INSTALL[@]}" --check --venv "$V121"
set_dist_version "$T/chin" mediapipe 0.10.9
"$PY310" -c "import json; p='$STATE'; d=json.load(open(p)); d['matrix']='cu128'; json.dump(d, open(p,'w'))"
run "check stamp says another matrix -> 10" 10 "${INSTALL[@]}" --check --venv "$V121"
"$PY310" -c "import json; p='$STATE'; d=json.load(open(p)); d['matrix']='cu121'; d['venv']='/elsewhere/venv'; json.dump(d, open(p,'w'))"
run "check ignores a stamp written for another venv" 0 "${INSTALL[@]}" --check --venv "$V121"
has "check says the stamp is for another venv" 'describes another venv'
rm -f "$STATE"

# ------------------------------------------------------------------------- 5. install-mode guards
log "5. install-mode guards (no network; must stop before any change)"
run "install refuses a matrix change without --clean -> 10" 10 env MUSETALK_HOST_FACTS_JSON="$T/facts_89.json" "${INSTALL[@]}" --venv "$V128" --skip-apt
has "install explains --clean" 'Re-run with --clean'
run "install cu128 venv untouched" 0 test -f "$V128/pyvenv.cfg"
run "install disk budget enforced" 1 env MUSETALK_INSTALL_MIN_DISK_GB=999999 "${INSTALL[@]}" --venv "$T/fresh" --skip-apt
has "install disk message" 'Not enough disk'
mkdir -p "$T/not_a_venv/keep"; echo precious >"$T/not_a_venv/keep/file"
run "install --clean refuses a non-venv dir" 1 env MUSETALK_INSTALL_MIN_DISK_GB=0 "${INSTALL[@]}" --venv "$T/not_a_venv" --clean --skip-apt
has "install explains the refusal" 'refusing to delete'
has "install phase summary records the failure" '\[FAIL\] Phase 2: Venv'
run "install left the non-venv dir intact" 0 test -f "$T/not_a_venv/keep/file"

# ------------------------------------------------------------------------- 6. download_weights.sh
log "6. download_weights.sh TAESD + Kokoro with stubbed tools"
STUB="$T/stub"; PYSTUB="$T/pystub"; mkdir -p "$STUB" "$PYSTUB/huggingface_hub" "$PYSTUB/kokoro"
: >"$PYSTUB/kokoro/__init__.py"; : >"$PYSTUB/huggingface_hub/__init__.py"
printf 'import os\nHF_HUB_CACHE = os.environ["HF_HUB_CACHE"]\n' >"$PYSTUB/huggingface_hub/constants.py"
cat >"$STUB/huggingface-cli" <<EOF
#!/usr/bin/env bash
echo "\$*" >>"$T/hf.log"
shift  # download
local_dir=""; revision="main"; args=()
while [[ \$# -gt 0 ]]; do
  case "\$1" in
    --local-dir) local_dir="\$2"; shift 2 ;;
    --revision) revision="\$2"; shift 2 ;;
    --max-workers) shift 2 ;;
    *) args+=("\$1"); shift ;;
  esac
done
repo="\${args[0]}"
for f in "\${args[@]:1}"; do
  if [[ -n "\$local_dir" ]]; then dest="\$local_dir/\$f"; else dest="\$HF_HUB_CACHE/models--\${repo//\//--}/snapshots/\$revision/\$f"; fi
  mkdir -p "\$(dirname "\$dest")"; echo stub >"\$dest"
done
EOF
cat >"$STUB/curl" <<'EOF'
#!/usr/bin/env bash
for a in "$@"; do [[ "$a" == -fsSI ]] && { printf 'HTTP/2 200\r\ncontent-length: 46827520\r\n'; exit 0; }; done
exit 0
EOF
printf '#!/usr/bin/env bash\necho "gdown $*" >>"%s/gdown.log"\n' "$T" >"$STUB/gdown"
cat >"$STUB/python" <<EOF
#!/usr/bin/env bash
if [[ "\${1:-}" == "-m" && "\${2:-}" == "pip" ]]; then echo "pip \${*:3}" >>"$T/pip.log"; exit 0; fi
PYTHONPATH="$PYSTUB" exec "$PY310" "\$@"
EOF
chmod +x "$STUB"/*
truncate -s 53289463 "$FAKE/models/face-parse-bisent/79999_iter.pth"
truncate -s 46827520 "$FAKE/models/face-parse-bisent/resnet18-5c106cde.pth"
rm -rf "$FAKE/models/taesd"
DW_ENV=(env PATH="$STUB:$PATH" HF_HUB_CACHE="$T/hfcache" DOWNLOAD_MUSETALK_V1_WEIGHTS=0 DOWNLOAD_AVATAR_PREP_WEIGHTS=0
        DOWNLOAD_RETRIES=1 DOWNLOAD_WAIT_FOR_NETWORK_SECONDS=1)
run "download_weights sequential (TAESD missing, kokoro auto)" 0 bash -c "cd '$FAKE' && \"\$@\" DOWNLOAD_GROUPS_IN_PARALLEL=0 bash ./download_weights.sh" _ "${DW_ENV[@]}"
LAST_OUT="$T/hf.log"
has "TAESD fetched at the pinned revision" 'madebyollin/taesd config\.json diffusion_pytorch_model\.safetensors --revision 614f76814bbe30edbe2e627ace1c2234c81a2c0e'
has "Kokoro cached at the pinned revision with default voices" '--revision f3ff3571791e39611d31c381e3a41a3af07b4987 hexgrad/Kokoro-82M config\.json kokoro-v1_0\.pth voices/af_heart\.pt voices/af_bella\.pt voices/am_michael\.pt'
run "TAESD files landed in models/taesd" 0 test -s "$FAKE/models/taesd/diffusion_pytorch_model.safetensors"
run "Kokoro refs/main written for offline boots" 0 grep -q f3ff3571791e39611d31c381e3a41a3af07b4987 "$T/hfcache/models--hexgrad--Kokoro-82M/refs/main"
LAST_OUT="$T/pip.log"
has "pinned en_core_web_sm installed when missing" 'en_core_web_sm @ https://github\.com/explosion/spacy-models/releases/download/en_core_web_sm-3\.8\.0/en_core_web_sm-3\.8\.0-py3-none-any\.whl#sha256=1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85'
: >"$T/hf.log"
echo other-rev >"$T/hfcache/models--hexgrad--Kokoro-82M/refs/main"
run "download_weights parallel (TAESD present -> skipped)" 0 bash -c "cd '$FAKE' && \"\$@\" DOWNLOAD_GROUPS_IN_PARALLEL=1 bash ./download_weights.sh" _ "${DW_ENV[@]}"
has "download log says TAESD skipped" 'TAESD already present, skipping'
LAST_OUT="$T/hf.log"
hasnt "no TAESD download when present" 'madebyollin/taesd'
run "an existing Kokoro refs/main is never rewritten" 0 grep -qx other-rev "$T/hfcache/models--hexgrad--Kokoro-82M/refs/main"
: >"$T/hf.log"
run "download_weights DOWNLOAD_KOKORO_WEIGHTS=0" 0 bash -c "cd '$FAKE' && \"\$@\" DOWNLOAD_KOKORO_WEIGHTS=0 DOWNLOAD_GROUPS_IN_PARALLEL=0 bash ./download_weights.sh" _ "${DW_ENV[@]}"
LAST_OUT="$T/hf.log"
hasnt "no Kokoro download when disabled" 'Kokoro-82M'
rm -rf "$FAKE/models/taesd"
run "download_weights DOWNLOAD_TAESD_WEIGHTS=0 keeps old behaviour" 0 bash -c "cd '$FAKE' && \"\$@\" DOWNLOAD_TAESD_WEIGHTS=0 DOWNLOAD_KOKORO_WEIGHTS=0 DOWNLOAD_GROUPS_IN_PARALLEL=0 bash ./download_weights.sh" _ "${DW_ENV[@]}"
run "no models/taesd files when TAESD disabled" 0 test ! -e "$FAKE/models/taesd/config.json"

# ------------------------------------------------------------------------- 7. live venv (read-only)
unset MUSETALK_HOST_FACTS_JSON
LIVE_VENV="${TEST_LIVE_VENV_PATH:-/workspace/.venvs/musetalk_trt_stagewise}"
if [[ "${TEST_LIVE_VENV:-1}" == "1" && -x "$LIVE_VENV/bin/python" ]]; then
  log "7. --check against the real server venv (read-only)"
  run "live venv passes --check untouched" 0 bash "$REPO/scripts/install_musetalk.sh" --check --venv "$LIVE_VENV"
else
  log "SKIP live venv check"
fi

printf '\n%d passed, %d failed\n' "$PASS" "$FAIL"
(( FAIL == 0 ))
