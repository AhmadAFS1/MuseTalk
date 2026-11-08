#!/usr/bin/env bash
# Detached validation for one freshly owned stock worker. Rental/expiry/ownership control stays on EC2.
# Inputs under /workspace/new_identity_review: job.env (CODE_SHA, LEASE_DEADLINE), inputs/characters.json,
# inputs/portraits30/ni30_{01..30}.png, inputs/student/{bk_tiny_best.pt,bk_tiny.json},
# inputs/audio/{ai-assistant.mpga,eng.wav}. The EC2 relay checks ownership and the protected instance.
# Start: setsid -f nohup bash /workspace/new_identity_review/new_identity_box_job.sh \
#   >/workspace/new_identity_review/start.log 2>&1 </dev/null
# Reattach through STATUS/job.log; a second start exits while locked. Restart requires identical code/inputs/lease.
# No training, server start, S3, avatar registration or rental mutations. Full-prep bootstrap models must exist.
# Results: source portraits/audio, prepared caches, all 60 AS videos and 60 original S16 videos, metrics/logs/provenance.
# Weights, model deps, checkpoints, auxiliary raw A/N/S videos and npy files are excluded from results.tar.
# Local command preview: CODE_SHA=<40hex> LEASE_DEADLINE=<UTC ISO8601> bash "$0" --dry-run
set -Eeuo pipefail
umask 077
DRY=0
case ${1:-} in --dry-run) DRY=1 ;; "") ;; *) echo "usage: new_identity_box_job.sh [--dry-run]" >&2; exit 2 ;; esac
ROOT=${NEW_IDENTITY_ROOT:-/workspace/new_identity_review}
BOOT=/workspace/MuseTalk
CLONE="$ROOT/MuseTalk"
INPUTS="$ROOT/inputs"
OUT="$ROOT/out"
STATE="$ROOT/state"
LOG="$ROOT/job.log"
RESERVE=1200                  #20minutes left for pull/cleanup after packaging.
PACKAGE_BUDGET=300            #Five minutes for delivery encoding; pull/cleanup reserve stays20minutes.
TEACHER_SHA=7ebf6c98c181e20838e4c0054e96e944ac60d5d692cc01db42839fe11b787007
STUDENT_SHA=a05d2d98f69b431252458d3b25f72e395d557ba693821baee91d11f5fe5a96cd
AUDIO_AI_SHA=98638b2f39c6135098ecbab92122f1eea39a52e622a17c27b9b5f885e0ccf81c
AUDIO_ENG_SHA=654dcbce843d70451d1123f7649a58bee11bb9dec9a7e835c05b1e367efb2078
CURRENT=setup
BOX_PY=""
MP_PY="$ROOT/venv-mp/bin/python"
LOCKED=0
DEADLINE_EPOCH=0
log() { printf '[new_identity] [%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"; }
die() { log "ERROR: $*"; exit 2; }
redact() { sed -u -E 's#https?://[^[:space:]"<>]+#[URL]#g'; }
status() {
  log "STATUS $*"
  if (( ! DRY )); then
    printf '%s %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*" >"$ROOT/STATUS.part"
    mv -f "$ROOT/STATUS.part" "$ROOT/STATUS"
  fi
}
settings() {
  local line key value now
  if [[ -f $ROOT/job.env ]]; then
    while IFS= read -r line || [[ -n $line ]]; do
      [[ -z $line || $line == \#* ]] && continue
      [[ $line =~ ^(CODE_SHA|LEASE_DEADLINE)=([A-Za-z0-9:+.TZ_-]+)$ ]] || die "unsupported job.env key or value"
      key=${BASH_REMATCH[1]} value=${BASH_REMATCH[2]}
      [[ -z ${!key-} || ${!key} == "$value" ]] || die "job.env differs from inherited $key"
      printf -v "$key" '%s' "$value"
    done <"$ROOT/job.env"
  fi
  [[ ${CODE_SHA:-} =~ ^[a-f0-9]{40}$ ]] || die "CODE_SHA must be 40 lowercase hex characters"
  [[ ${LEASE_DEADLINE:-} =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}(Z|\+00:00)$ ]] || die "lease deadline must be UTC ISO8601"
  if (( DRY )); then return 0; fi
  [[ $ROOT == /workspace/new_identity_review ]] || die "runtime root is fixed"
  [[ ${VAST_INSTANCE_ID:-} != 51074906 && ${CONTAINER_ID:-} != 51074906 ]] || die "protected instance refused"
  DEADLINE_EPOCH=$(date -u -d "$LEASE_DEADLINE" +%s) || die "invalid lease deadline"
  now=$(date -u +%s)
  (( DEADLINE_EPOCH > now + RESERVE && DEADLINE_EPOCH <= now + 14400 )) || die "lease has no time or exceeds 4 hours"
  for key in flock timeout git ffmpeg sha256sum python3 tar nvidia-smi; do command -v "$key" >/dev/null || die "missing command $key"; done
}
remaining() {
  if (( DRY )); then printf '7200\n'; return 0; fi
  local n=$(( DEADLINE_EPOCH - $(date -u +%s) - RESERVE ))
  (( n > 0 )) || die "20-minute pull/cleanup reserve reached"
  printf '%s\n' "$n"
}
freeze_request() {
  CURRENT=inputs
  status "verifying frozen inputs and resume signature"
  if (( DRY )); then log "+ stdlib SHA256 audit: exact 30 portraits, frozen 2 audio, student/config, code, wrapper and lease; identical resume only"; return 0; fi
  python3 -I - "$ROOT" "$CODE_SHA" "$LEASE_DEADLINE" "$STUDENT_SHA" "$AUDIO_AI_SHA" "$AUDIO_ENG_SHA" "${BASH_SOURCE[0]}" <<'PY'
import hashlib,json,os,pathlib,sys
root,code,deadline,student,ai,eng,script=sys.argv[1:]; root=pathlib.Path(root); inputs=root/"inputs"
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
ids=["ni30_%02d"%i for i in range(1,31)]
m=json.loads((inputs/"characters.json").read_text())
assert sorted(c["id"] for c in m["characters"])==ids,"exact 30new IDs required"
assert sorted(p.name for p in (inputs/"portraits30").iterdir())==[i+".png" for i in ids],"exact 30portrait filenames required"
paths=[inputs/"characters.json",inputs/"student/bk_tiny_best.pt",inputs/"student/bk_tiny.json",
       inputs/"audio/ai-assistant.mpga",inputs/"audio/eng.wav"]+[inputs/"portraits30"/(i+".png") for i in ids]
assert all(p.is_file() and not p.is_symlink() for p in paths),"inputs must be regular files"
files={str(p.relative_to(root)):{"sha256":sha(p),"bytes":p.stat().st_size} for p in paths}
assert files["inputs/student/bk_tiny_best.pt"]["sha256"]==student,"student checkpoint differs"
assert files["inputs/audio/ai-assistant.mpga"]["sha256"]==ai,"ai-assistant audio differs"
assert files["inputs/audio/eng.wav"]["sha256"]==eng,"eng audio differs"
assert len({files["inputs/portraits30/"+i+".png"]["sha256"] for i in ids})==30,"portrait hashes must be unique"
request={"code_sha":code,"lease_deadline":deadline,"wrapper_sha256":sha(pathlib.Path(script)),"files":files,
         "protocol":"new-static-r5","identities":30,"clips":60,"fps":25,"frames":250,"training":False,
         "historical_D_comparison":False,"pull_cleanup_reserve_seconds":1200}
p=root/"state/request.json"
if p.exists(): assert json.loads(p.read_text())==request,"resume inputs/code/lease differ; existing outputs unchanged"
else:
    t=p.with_suffix(".part"); t.write_text(json.dumps(request,indent=2)+"\n"); os.replace(t,p)
print("frozen request verified; no training or checkpoint selection")
PY
}
wait_boot() {
  CURRENT=bootstrap
  status "waiting for this stock worker's bootstrap lock"
  if (( DRY )); then log "+ hold /tmp/musetalk-onstart.lock until exit; wait<= 40 min, lease-clamped"; return 0; fi
  local seconds
  seconds=$(remaining); (( seconds <= 2400 )) || seconds=2400
  exec 7>/tmp/musetalk-onstart.lock
  flock -w "$seconds" 7 || die "bootstrap did not release its lock"
}
stop_worker() {
  CURRENT=stop
  status "stopping only this fresh worker's API and local TURN processes"
  if (( DRY )); then log "+ TERM/KILL only owned bootstrap API/TURN; refuse unknown GPU jobs/listeners"; return 0; fi
  # No control-plane env sourcing and no API request. Stop only /proc-verified bootstrap processes.
  local sig pid cwd cmd turnpid="" jobs
  if [[ -f /workspace/logs/musetalk/turnserver.pid ]]; then turnpid=$(tr -d '[:space:]' </workspace/logs/musetalk/turnserver.pid); fi
  [[ -z $turnpid || $turnpid =~ ^[0-9]+$ ]] || die "invalid TURN PID file"
  for sig in TERM KILL; do
    for pid in $(pgrep -f 'api_server\.py|(^|/)turnserver([[:space:]]|$)' || true); do
      [[ -r /proc/$pid/cmdline ]] || continue
      cwd=$(readlink "/proc/$pid/cwd" 2>/dev/null || true)
      cmd=$(tr '\0' ' ' <"/proc/$pid/cmdline" 2>/dev/null || true)
      if [[ $cmd == *api_server.py* && ( $cwd == "$BOOT" || $cwd == "$BOOT/"* || $cmd == *"$BOOT/"* ) ]]; then
        log "stopping owned API pid=$pid signal=$sig"; kill -s "$sig" "$pid" 2>/dev/null || true
      elif [[ $cmd =~ (^|/)turnserver([[:space:]]|$) && ( $pid == "$turnpid" || $cwd == "$BOOT" || $cwd == "$BOOT/"* ) ]]; then
        log "stopping owned TURN pid=$pid signal=$sig"; kill -s "$sig" "$pid" 2>/dev/null || true
      fi
    done
    sleep 3
  done
  if (exec 3<>/dev/tcp/127.0.0.1/8000) 2>/dev/null; then die "API listener remains on 8000; refusing shared GPU"; fi
  if pgrep -x turnserver >/dev/null; then die "unverified TURN process remains; it was not killed"; fi
  jobs=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits) || die "cannot inspect GPU ownership"
  [[ -z $jobs ]] || die "GPU process remains; no unrelated process was killed"
}
python_and_clone() {
  CURRENT=code
  status "locating full-prep locked Python and immutable checkout"
  if (( DRY )); then
    BOX_PY=/workspace/.venvs/musetalk_locked_FULLPREP/bin/python
    log "+ locate bootstrap locked venv; require CUDA/full-prep imports"
    log "+ sparse blob-filtered checkout $CODE_SHA -> $CLONE; existing checkout must match"
    return 0
  fi
  local artifact="" contract="$BOOT/configs/dependencies/rtx3090-cp310-cu121-screening.json" stage="$ROOT/clone.stage"
  if [[ -f $contract ]]; then
    artifact=$(python3 -I -c 'import json,re,sys; v=json.load(open(sys.argv[1]))["artifact_id"]; assert re.fullmatch("[a-f0-9]{64}",v); print(v)' "$contract")
  fi
  if [[ -n $artifact && -x /workspace/.venvs/musetalk_locked_$artifact/bin/python ]]; then
    BOX_PY="/workspace/.venvs/musetalk_locked_$artifact/bin/python"
  else
    BOX_PY=$(find /workspace/.venvs -path '*/musetalk_locked_*/bin/python' | sort | head -n1)
  fi
  [[ -n $BOX_PY && -x $BOX_PY ]] || die "no locked worker Python"
  export PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=0 HF_HUB_DISABLE_TELEMETRY=1
  "$BOX_PY" -c 'import torch,diffusers,cv2,transformers,mmcv,mmpose,mmdet,mmengine; assert torch.cuda.is_available(); print("torch",torch.__version__,"diffusers",diffusers.__version__,"GPU",torch.cuda.get_device_name(0))'
  if [[ -d $CLONE/.git ]]; then
    [[ $(git -C "$CLONE" rev-parse HEAD) == "$CODE_SHA" ]] || die "checkout differs from CODE_SHA"
    git -C "$CLONE" diff --quiet && git -C "$CLONE" diff --cached --quiet || die "tracked checkout edits"
  else
    [[ ! -e $CLONE ]] || die "checkout path exists without .git"
    rm -rf "$stage"; git init -q "$stage"
    git -C "$stage" remote add origin https://github.com/AhmadAFS1/MuseTalk.git
    git -C "$stage" config core.sparseCheckout true
    git -C "$stage" config core.sparseCheckoutCone false
    printf '/*\n!/generated/\n!/assets/\n!/character_factory/\n!/third_party_wheels/\n!/uploads/\n' >"$stage/.git/info/sparse-checkout"
    git -C "$stage" -c core.hooksPath=/dev/null fetch --depth 1 --filter=blob:none origin "$CODE_SHA"
    git -C "$stage" -c core.hooksPath=/dev/null checkout --detach "$CODE_SHA"
    mv "$stage" "$CLONE"
  fi
  [[ $(sha256sum "${BASH_SOURCE[0]}" | cut -d' ' -f1) == $(sha256sum "$CLONE/scripts/ios/new_identity_box_job.sh" | cut -d' ' -f1) ]] || die "uploaded wrapper differs from immutable source"
}
models() {
  CURRENT=models
  status "linking existing worker models and verifying frozen teacher/student hashes"
  if (( DRY )); then log "+ existing model symlinks; teacher SHA256=$TEACHER_SHA; student SHA256=$STUDENT_SHA; verify before prep"; return 0; fi
  local name src dst
  mkdir -p "$CLONE/models"
  for name in musetalkV15 sd-vae whisper taesd dwpose face_detection face-parse-bisent; do
    src="$BOOT/models/$name"; dst="$CLONE/models/$name"
    [[ -d $src ]] || die "full-prep model directory missing: $name"
    if [[ -L $dst ]]; then [[ $(readlink "$dst") == "$src" ]] || die "model symlink differs: $name"
    else [[ ! -e $dst ]] || die "model destination not symlink: $name"; ln -s "$src" "$dst"; fi
  done
  printf '%s  %s\n' "$TEACHER_SHA" "$CLONE/models/musetalkV15/unet.pth" "$STUDENT_SHA" "$INPUTS/student/bk_tiny_best.pt" | sha256sum -c --status || die "teacher/student SHA256 mismatch"
  mkdir -p "$OUT/code"
  cp "${BASH_SOURCE[0]}" "$CLONE/scripts/ios/prepare_new_identity_review.py" "$CLONE/scripts/ios/review_distilled_unet.py" "$OUT/code/"
  "$BOX_PY" - "$ROOT" "$CODE_SHA" "$TEACHER_SHA" "$STUDENT_SHA" <<'PY'
import hashlib,json,pathlib,sys,torch,diffusers
r=pathlib.Path(sys.argv[1]); models=r/"MuseTalk/models"
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
paths=[models/p for p in ("musetalkV15/musetalk.json","sd-vae/config.json","whisper/config.json",
"whisper/preprocessor_config.json","taesd/config.json","dwpose/dw-ll_ucoco_384.pth",
"face_detection/s3fd.pth","face-parse-bisent/79999_iter.pth","face-parse-bisent/resnet18-5c106cde.pth")]
for directory, names in (("sd-vae",("diffusion_pytorch_model.safetensors","diffusion_pytorch_model.bin")),
                         ("whisper",("pytorch_model.bin","model.safetensors")),
                         ("taesd",("diffusion_pytorch_model.safetensors","diffusion_pytorch_model.bin"))):
    found=[models/directory/n for n in names if (models/directory/n).is_file()]
    assert found,"required model weights missing: "+directory
    paths+=found
model_inputs={str(p.relative_to(models)):sha(p) for p in paths}
model_inputs["musetalkV15/unet.pth"]=sys.argv[3]  #Verified immediately above, without reading3.4GB twice.
guard=r/"state/model_inputs.json"
if guard.exists(): assert json.loads(guard.read_text())==model_inputs,"model dependencies changed on resume"
else: guard.write_text(json.dumps(model_inputs,indent=2)+"\n")
data={"request":json.loads((r/"state/request.json").read_text()),"model_inputs":model_inputs,
"code_sha":sys.argv[2],"teacher_sha256":sys.argv[3],"student_sha256":sys.argv[4],"torch":torch.__version__,
"diffusers":diffusers.__version__,"gpu":torch.cuda.get_device_name(0),"static_portrait_only":True,
"training_performed":False,"historical_D_comparison":False}
(r/"provenance.json").write_text(json.dumps(data,indent=2)+"\n")
PY
}
facemesh_env() {
  CURRENT=facemesh
  status "verifying isolated MediaPipe 0.10.14 environment"
  if (( DRY )); then log "+ isolated venv: mediapipe 0.10.14, numpy 1.26.4, opencv-contrib-python 4.10.0.84"; return 0; fi
  if [[ -x $MP_PY ]] && "$MP_PY" -c 'import mediapipe,cv2; assert mediapipe.__version__=="0.10.14"' >/dev/null 2>&1; then return 0; fi
  local seconds
  seconds=$(remaining); seconds=$(( seconds - 30 )); (( seconds > 0 )) || die "no time for landmark setup"
  timeout -s INT -k 30 "$seconds" python3 -m venv "$ROOT/venv-mp"
  seconds=$(remaining); seconds=$(( seconds - 30 )); (( seconds > 0 )) || die "no time for landmark install"
  timeout -s INT -k 30 "$seconds" "$MP_PY" -m pip install --disable-pip-version-check -q mediapipe==0.10.14 numpy==1.26.4 opencv-contrib-python==4.10.0.84
  "$MP_PY" -c 'import mediapipe,cv2; assert mediapipe.__version__=="0.10.14"; print("MediaPipe",mediapipe.__version__)'
}
prepare() {
  CURRENT=prepare
  status "preparing 30 static identities with unchanged production prep"
  local args=(scripts/ios/prepare_new_identity_review.py --characters "$INPUTS/characters.json" --portrait-root "$INPUTS/portraits30"
    --audio "$INPUTS/audio/ai-assistant.mpga" "$INPUTS/audio/eng.wav" --checkpoint "$INPUTS/student/bk_tiny_best.pt"
    --checkpoint-sha256 "$STUDENT_SHA" --out "$OUT/prepared")
  if [[ -f $OUT/prepared/prep_inputs.json ]]; then args+=(--resume --rebuild-incomplete); fi
  if (( DRY )); then log "+ (cd $CLONE && lease-clamped timeout $(printf '%q ' "$BOX_PY" "${args[@]}"))"; return 0; fi
  local seconds
  seconds=$(remaining); seconds=$(( seconds - PACKAGE_BUDGET - 30 ))
  (( seconds > 0 )) || die "no preparation budget after packaging/pull reserve"
  (cd "$CLONE" && timeout -s INT -k 30 "$seconds" "$BOX_PY" "${args[@]}") 2>&1 | tee -a "$OUT/prepare.log"
  [[ -s $OUT/prepared/new_avatar_manifest.json ]] || die "preparation manifest missing"
}
review() {
  CURRENT=review
  status "reviewing 60 clips: A/N/S/S16, 250 frames, 25 fps; no historical D"
  local avatars=() path i aid args=() seconds
  if (( DRY )); then
    for ((i=1; i<=30; i++)); do printf -v aid 'ni30_%02d' "$i"; avatars+=("$OUT/prepared/runtime/results/v15/avatars/$aid"); done
  else
    while IFS= read -r path; do avatars+=("$path"); done < <(python3 -I -c 'import json,sys; m=json.load(open(sys.argv[1])); assert len(m["avatars"])==30; print("\n".join(a["cache_dir"] for a in m["avatars"]))' "$OUT/prepared/new_avatar_manifest.json")
    (( ${#avatars[@]} == 30 )) || die "preparation did not supply 30 cache paths"
  fi
  args=(scripts/ios/review_distilled_unet.py --protocol new-static-r5 --new-avatar-manifest "$OUT/prepared/new_avatar_manifest.json"
    --avatars "${avatars[@]}" --audio "$OUT/prepared/audio/ai-assistant.mpga" "$OUT/prepared/audio/eng.wav"
    --student-json "$INPUTS/student/bk_tiny.json" --student-weights "$INPUTS/student/bk_tiny_best.pt"
    --out "$OUT/review" --device cuda --facemesh-python "$MP_PY" --seconds 10 --fps 25 --batch 8)
  if (( DRY )); then log "+ (cd $CLONE && lease-clamped timeout $(printf '%q ' "$BOX_PY" "${args[@]}"))"; return 0; fi
  # Only completed, hash-matched renders may be reused. Otherwise rerun the entire review; never bypass validation.
  if [[ -f $STATE/review_outputs.json ]]; then
    python3 -I - "$ROOT" <<'PY'
import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]); saved=json.loads((r/"state/review_outputs.json").read_text())
for rel,digest in saved.items():
    p=r/rel; h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    assert h.hexdigest()==digest,"completed review output changed"
print("completed review outputs verified; no rerender needed")
PY
    return 0
  fi
  seconds=$(remaining); seconds=$(( seconds - PACKAGE_BUDGET - 60 ))
  (( seconds >= 300 )) || die "under 5 minutes of review budget remains after packaging/pull reserve"
  (cd "$CLONE" && timeout -s INT -k 60 "$seconds" "$BOX_PY" "${args[@]}") 2>&1 | tee -a "$OUT/review.log"
  python3 -I - "$ROOT" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); review=root/"out/review"; s=json.loads((review/"summary.json").read_text())
expected={"ni30_%02d__%s"%(i,a) for i in range(1,31) for a in ("ai-assistant","eng")}
assert s.get("protocol")=="new_static_avatars_r5_v1","wrong protocol"
assert len(s["clips"])==60 and {c["clip"] for c in s["clips"]}==expected,"exact 60 clips required"
paths=[review/"summary.json",review/"summary.md",root/"out/prepared/new_avatar_manifest.json",root/"state/model_inputs.json"]
for c in s["clips"]:
    assert c["S"]["frames"]==250 and c["S"]["fps"]==25,"wrong frame protocol"
    paths += [review/c["clip"]/c["files"]["labelled"],review/c["clip"]/"S16.mp4",review/c["clip"]/"metrics.json"]
hashes={}
for p in paths:
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    hashes[str(p.relative_to(root))]=h.hexdigest()
(root/"state/review_outputs.json").write_text(json.dumps(hashes,indent=2)+"\n")
print("complete 60-clip review recorded; retain reported FAIL/PASS/INCOMPLETE verdicts")
PY
}
package() (
  set -Eeuo pipefail
  local rc=$1 step=$2
  if (( DRY )); then log "+ results.tar: sources/prepcaches/metrics/logs +60 H.264 CRF 26 AS videos +60 original S16 videos; exclude weights/models/checkpoints/raw A/N/S"; return 0; fi
  local delivery="$ROOT/delivery" f rel target seconds count=0 s16count=0
  rm -rf "$delivery"; mkdir -p "$delivery/out/review" "$delivery/inputs" "$delivery/state"
  if (( rc == 0 )); then printf 'ok\n' >"$delivery/DONE"
  else printf 'failed %s rc=%s\n' "$step" "$rc" >"$delivery/DONE"; fi
  for f in "$ROOT/job.env" "$ROOT/provenance.json" "$ROOT/STATUS" "$LOG"; do [[ ! -f $f ]] || cp "$f" "$delivery/"; done
  for f in "$STATE/request.json" "$STATE/model_inputs.json" "$STATE/review_outputs.json"; do [[ ! -f $f ]] || cp "$f" "$delivery/state/"; done
  [[ ! -f $INPUTS/characters.json ]] || cp "$INPUTS/characters.json" "$delivery/inputs/"
  if [[ -f $INPUTS/student/bk_tiny.json ]]; then
    mkdir -p "$delivery/inputs/student"; cp "$INPUTS/student/bk_tiny.json" "$delivery/inputs/student/"
  fi
  [[ ! -d $INPUTS/portraits30 ]] || cp -a "$INPUTS/portraits30" "$delivery/inputs/"
  [[ ! -d $INPUTS/audio ]] || cp -a "$INPUTS/audio" "$delivery/inputs/"
  [[ ! -d $OUT/prepared ]] || cp -a "$OUT/prepared" "$delivery/out/"
  [[ ! -d $OUT/code ]] || cp -a "$OUT/code" "$delivery/out/"
  for f in "$OUT"/*.log; do [[ ! -f $f ]] || cp "$f" "$delivery/out/"; done
  if [[ -d $OUT/review ]]; then
    while IFS= read -r -d '' f; do
      rel=${f#"$OUT/review/"}; target="$delivery/out/review/$rel"; mkdir -p "$(dirname "$target")"; cp "$f" "$target"
    done < <(find "$OUT/review" -type f \( -name '*.json' -o -name '*.md' -o -name '*.log' -o -name '*.jpg' -o -name '*.png' \) -print0)
    while IFS= read -r -d '' f; do
      rel=${f#"$OUT/review/"}; target="$delivery/out/review/$rel"; mkdir -p "$(dirname "$target")"
      cp "$f" "$target"; s16count=$((s16count+1))
    done < <(find "$OUT/review" -type f -name 'S16.mp4' -print0)
    while IFS= read -r -d '' f; do
      rel=${f#"$OUT/review/"}; target="$delivery/out/review/$rel"; mkdir -p "$(dirname "$target")"
      seconds=$(( DEADLINE_EPOCH - $(date -u +%s) - RESERVE - 120 - 15 ))
      (( seconds > 0 )) || { log "encoding reached archive/pull reserve"; return 2; }
      timeout -s INT -k 15 "$seconds" ffmpeg -nostdin -hide_banner -loglevel error -y -i "$f" \
        -c:v libx264 -preset fast -crf 26 -threads 2 -c:a copy "$target.part.mp4" || return 2
      mv "$target.part.mp4" "$target"; count=$((count+1))
    done < <(find "$OUT/review" -type f -name '*_AS.mp4' -print0)
  fi
  python3 -I - "$delivery" "$rc" "$step" "$count" "$s16count" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]); files={}
for p in sorted(root.rglob("*")):
    assert not p.is_symlink(),"archive symlink refused"
    if p.is_file():
        h=hashlib.sha256()
        with p.open("rb") as f:
            for b in iter(lambda:f.read(1<<20),b""): h.update(b)
        files[str(p.relative_to(root))]={"sha256":h.hexdigest(),"bytes":p.stat().st_size}
record={"job_rc":int(sys.argv[2]),"last_step":sys.argv[3],"AS_videos":int(sys.argv[4]),"expected_AS_videos":60,
"S16_videos":int(sys.argv[5]),"expected_S16_videos":60,"S16_video_copy":"Original renderer files copied byte-for-byte; no additional encoding.",
"presentation_video_transcode":{"codec":"h264","crf":26,"resolution_fps_timing":"unchanged","audio":"copied"},
"metrics":"Measured from original raw frames before compression.",
"excludes":"weights, model dependencies, checkpoints, raw A/N/S videos, latent/landmark npy",
"files":files,"bytes_before_archive":sum(v["bytes"] for v in files.values())}
assert int(sys.argv[2]) != 0 or (int(sys.argv[4]) == 60 and int(sys.argv[5]) == 60), "successful job must deliver 60 AS and 60 S16 videos"
(root/"delivery_manifest.json").write_text(json.dumps(record,indent=2)+"\n")
PY
  tar -cf "$ROOT/results.tar.part" -C "$delivery" . || return 2
  mv "$ROOT/results.tar.part" "$ROOT/results.tar"
  (cd "$ROOT" && sha256sum results.tar >results.tar.sha256) || return 2
  log "packaged $count/60 AS and $s16count/60 original S16 videos; archive $(wc -c <"$ROOT/results.tar") bytes (pull limit at most 2 GB)"
)
finish() {
  local rc=$1 step=$CURRENT packed=0
  (( LOCKED )) || return 0
  trap - EXIT HUP INT TERM
  set +e
  status "packaging rc=$rc last_step=$step"
  package "$rc" "$step"; packed=$?
  if (( packed == 0 )); then
    if (( rc == 0 )); then printf 'ok\n' >"$ROOT/DONE"; status "done: results.tar ready; quality verdict retained"
    else printf 'failed %s rc=%s\n' "$step" "$rc" >"$ROOT/DONE"; status "failed: partial results.tar ready"; fi
  else status "failed: packaging rc=$packed; raw outputs remain under out/"; rc=2; fi
  exit "$rc"
}
main() {
  if (( DRY )); then
    settings; freeze_request; wait_boot; stop_worker; python_and_clone; models; facemesh_env; prepare; review; package 0 preview
    return 0
  fi
  [[ $ROOT == /workspace/new_identity_review ]] || die "runtime root is fixed"
  mkdir -p "$ROOT" "$OUT" "$STATE"
  exec 9>"$ROOT/.lock"
  flock -n 9 || { log "job already running; inspect STATUS/job.log"; exit 0; }
  exec > >(redact >>"$LOG") 2>&1
  LOCKED=1
  trap 'finish $?' EXIT
  trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM
  settings; freeze_request
  if [[ -f $ROOT/DONE && $(head -n1 "$ROOT/DONE") == ok ]]; then
    log "matching job already complete; existing results retained"; trap - EXIT; exit 0
  fi
  rm -f "$ROOT/DONE"
  wait_boot; stop_worker; python_and_clone; models; facemesh_env; prepare; review
  CURRENT=complete
}
main "$@"
