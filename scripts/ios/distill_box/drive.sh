#!/usr/bin/env bash
# drive.sh: run the bk_tiny UNet distillation (box_job.sh) on a rented RTX 3090 from this Mac, through EC2 only.
#
# Why: the Mac cannot train the smaller UNet (ios_coreml_implementation.md 5.8), and renting a GPU goes through the
# EC2 controller (musetalk-startup-timing scripts/vast_locked_boot_ec2.py at 4e2b02e; lingua
# docs/musetalk-3090-autoscaling-mission.md sections 2 and 8). The controller holds the Vast key, arms the expiry timer
# before the create request, uses an mtboot- label that no autoscaler prefix matches, and refuses 51074906.
# The box only accepts EC2's SSH key. So every step here is `ssh my-ec2 ...`: the controller for the rental, and a
# nested `sudo -n ssh` (EC2's key, the run's known_hosts, the endpoint from the run's provider-latest.json) for the
# box. Nothing here talks to Vast or the box directly, no credential passes through it, and presigned URLs are
# written to a 0600 file and never printed.
#
# Order:  preflight -> create --yes -> status (until running and the SSH key is attached) -> probe (until the bootstrap
#         is healthy) -> presign -> push -> start -> watch ... -> pull -> cleanup --yes
#   drive.sh [--dry-run] <subcommand> <run> [--yes] [--keys-file F]
#   <run> is the controller's run name, 3-40 chars [a-z0-9-]; the Vast label is mtboot-<run>.
#   presign <run> [--keys-file F]  aws s3 presign --expires-in 14400 for the S3 keys below (or F: "<dest>\t<s3://...>"
#                                  per line) -> $DISTILL_LOCAL_DIR/<run>/urls.tsv, "<dest under inputs/>\t<URL>", 0600
#   preflight <run>                controller preflight: offer, stock template, tickets, identity; no provider write
#   create <run> --yes             controller create, exactly once (refused if the run state dir exists on EC2, or if
#                                  this Mac already sent a create for <run>)
#   status <run>                   controller status: provider state; attaches EC2's SSH key once the box is running
#   probe <run>                    controller probe: bootstrap timeline and /health
#   push <run>                     box_job.sh, job.env, the URL list, the 3 HarnessBundle gate .bin files and the
#                                  training + review clips -> box:/workspace/distill (a tar stream over the nested ssh)
#   start <run>                    box_job.sh detached on the box (setsid nohup); it logs to /workspace/distill/job.log
#   watch <run>                    STATUS, DONE and the tail of job.log (file reads only: no nvidia-smi during the job)
#   pull <run>                     results.tar -> $DISTILL_LOCAL_DIR/<run>/results.tar, size and sha256 checked
#   cleanup <run> --yes            controller cleanup, then check that the expiry timers are inactive and the
#                                  instance absent
#   --dry-run (or DRY_RUN=1)       print the commands instead of running ssh or aws; nothing remote happens
# Hard refusals: 51074906 anywhere in the arguments or as the instance in any output; run names that start with tools
# (the EC2 tools checkout is mtboot-tools-*) or lingua (autoscaler prefixes); a label that does not start with mtboot-.
#
# Settings (environment; defaults):
#   DISTILL_EC2_HOST=my-ec2      DISTILL_EC2_TOOLS=/home/ec2-user/.local/state/mtboot-tools-4e2b02e (the controller
#                                checkout: scripts/vast_locked_boot_ec2.py, or the file at its top level)
#   DISTILL_LIFETIME=14400 (s, 3600-14400)  DISTILL_MAX_PRICE=0.30 ($/h, at most 0.30)  DISTILL_DISK=100 (GB)
#   DISTILL_MACHINE (pin a Vast machine id)  the transfer cap is fixed at $0.0015/GB, --launch stock and
#                                --revision 4e2b02e5cb72cce94f415bce8099222fb25b8156 are fixed, and the box is created
#                                with MUSETALK_SKIP_EAGER_UNET=0 so the bootstrap fetches unet.pth (the teacher)
#   DISTILL_LOCAL_DIR=<repo>/.runtime/mtboot-distill   (gitignored; per run: urls.tsv, create.requested, results.tar)
#   DISTILL_S3_BUCKET=lingua-musetalk-s3-storage, region us-east-1, AWS_PROFILE as set. The URLs live only as long as
#                                the credentials that signed them: presign with long-lived keys, not a short session.
#   TRAIN_AVATARS=roster12       phone bundles s3://<bucket>/avatars/ios/v1/<id>.tar (+ <id>.json sidecar, checked on
#                                the box): "roster12" = the 12 talking-pose roster characters after excluding the six review
#                                identities and every gate identity; or exactly 12 explicit eligible ids. Ten train,
#                                two validate. Only latents_f16.bin (+ avatar.json) is kept.
#   REVIEW_AVATARS=en_01_maya_abe451d3_talking_f6cf8be7fb,en_03_marcus_19eb1ea8_talking_ada35a4a4d,
#                  es_01_valeria_aed74c50_talking_003d585d96,es_03_mateo_884600e1_talking_75d88cda6c,
#                  japanese_realtime_talking_7d94520b7f,latina_guided_20260925_talking_84c5bc80b8
#                                the section 4.2 set, as v15 server avatars s3://<bucket>/avatars/v15/<id>.tar.gz
#   REVIEW_AUDIO=data/audio/ai-assistant.mpga,data/audio/eng.wav   the two held-out clips (paths in this checkout or
#                                absolute)
#   TRAIN_AUDIO                  comma-separated; default every clip in data/audio except the REVIEW_AUDIO ones (9
#                                clips, 151 s)
#   DISTILL_SHA=HEAD of this checkout (the box clones AhmadAFS1/MuseTalk, so it must be pushed)
#   DISTILL_VARIANT=bk_tiny  DISTILL_MAX_MINUTES=150  DISTILL_STEPS=30000  DISTILL_RESERVE_MINUTES=60 (lease kept
#                                for review + pull)  DISTILL_EXTRA_ARGS  REVIEW_SECONDS=10  REVIEW_MAX_MINUTES=50
#                                REVIEW_EXTRA_ARGS  FACEMESH=1   (written to job.env; box_job.sh's header has them)
#   WATCH_LINES=40
set -Eeuo pipefail
umask 077

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BENCH_REPO="$(cd "$HERE/../../.." && pwd)"
EC2_HOST="${DISTILL_EC2_HOST:-my-ec2}"
TOOLS="${DISTILL_EC2_TOOLS:-/home/ec2-user/.local/state/mtboot-tools-4e2b02e}"
EC2_PY=/home/ec2-user/lingua/venv/bin/python
EC2_STATE=/home/ec2-user/.local/state
REVISION=4e2b02e5cb72cce94f415bce8099222fb25b8156
PROTECTED=51074906
LIFETIME="${DISTILL_LIFETIME:-14400}"
MAX_PRICE="${DISTILL_MAX_PRICE:-0.30}"
MAX_TRANSFER=0.0015
DISK="${DISTILL_DISK:-100}"
MACHINE="${DISTILL_MACHINE:-}"
LOCAL="${DISTILL_LOCAL_DIR:-$BENCH_REPO/.runtime/mtboot-distill}"
BUCKET="${DISTILL_S3_BUCKET:-lingua-musetalk-s3-storage}"
REGION=us-east-1
BUNDLE_PREFIX=avatars/ios/v1
AVATAR_PREFIX=avatars/v15
URL_SECONDS=14400
HB="$BENCH_REPO/ios/MuseTalkBench/HarnessBundle"
GATE_FILES="unet_latents_f16.bin:2097152 unet_audio_f16.bin:4915200 unet_ref_f16.bin:1048576"
TRAIN_AVATARS="${TRAIN_AVATARS:-roster12}"
REVIEW_AVATARS="${REVIEW_AVATARS:-en_01_maya_abe451d3_talking_f6cf8be7fb,en_03_marcus_19eb1ea8_talking_ada35a4a4d,es_01_valeria_aed74c50_talking_003d585d96,es_03_mateo_884600e1_talking_75d88cda6c,japanese_realtime_talking_7d94520b7f,latina_guided_20260925_talking_84c5bc80b8}"
REVIEW_AUDIO="${REVIEW_AUDIO:-data/audio/ai-assistant.mpga,data/audio/eng.wav}"
TRAIN_AUDIO="${TRAIN_AUDIO:-}"
DISTILL_VARIANT="${DISTILL_VARIANT:-bk_tiny}"
DISTILL_MAX_MINUTES="${DISTILL_MAX_MINUTES:-150}"
DISTILL_STEPS="${DISTILL_STEPS:-30000}"
DISTILL_RESERVE_MINUTES="${DISTILL_RESERVE_MINUTES:-60}"
DISTILL_EXTRA_ARGS="${DISTILL_EXTRA_ARGS:-}"
REVIEW_SECONDS="${REVIEW_SECONDS:-10}"
REVIEW_MAX_MINUTES="${REVIEW_MAX_MINUTES:-50}"
REVIEW_EXTRA_ARGS="${REVIEW_EXTRA_ARGS:-}"
FACEMESH="${FACEMESH:-1}"
WATCH_LINES="${WATCH_LINES:-40}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=15 -o ServerAliveCountMax=4)
DRY="${DRY_RUN:-0}"
HOP_SHOWN=0

redact() { sed -E 's#https?://[^[:space:]"]+#[URL]#g'; }
say() { printf '[drive] %s\n' "$*"; }
die() { printf '[drive] ERROR: %s\n' "$*" >&2; exit 2; }
usage() { sed -n '2,/^set -Eeuo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'; }
mtime() { stat -f %m "$1" 2>/dev/null || stat -c %Y "$1"; }
csv_items() { tr ',' '\n' <<<"$1" | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' -e '/^$/d'; }

# ------------------------------------------------------------------------------------------------- EC2 and the box

ec2() {  # $1 = command for bash on EC2 (as ec2-user); stdin passes through
  if (( DRY )); then printf "[dry-run] ssh %s %s '%s'\n" "${SSH_OPTS[*]}" "$EC2_HOST" "$1"; return 0; fi
  ssh "${SSH_OPTS[@]}" "$EC2_HOST" "$1" 2> >(redact >&2)
}

ctl_cmd() {  # controller mode + arguments -> the EC2 command that runs it as root
  local q="" a
  for a in "$@"; do q="$q $(printf '%q' "$a")"; done
  printf 'set -eu; T=%q; C="$T/scripts/vast_locked_boot_ec2.py"; sudo -n test -f "$C" || C="$T/vast_locked_boot_ec2.py"; ' "$TOOLS"
  printf 'sudo -n test -f "$C" || { echo "vast_locked_boot_ec2.py not found under $T" >&2; exit 3; }; '
  printf 'exec sudo -n %q -B "$C"%s' "$EC2_PY" "$q"
}

guarded() {  # run an EC2 command; stop if its output names the protected instance as an instance
  local out rc=0
  out=$(ec2 "$1" </dev/null) || rc=$?
  [[ -z $out ]] || printf '%s\n' "$out" | redact
  if grep -qE "\"(id|instance_id)\": *\"?$PROTECTED([^0-9]|\$)" <<<"$out"; then die "output names $PROTECTED (protected RTX 4070 SUPER) as an instance; stopping"; fi
  return "$rc"
}

# Runs on EC2 with $1 = label, $2 = the box script in base64. Reads the run's ledger and provider row as root (the
# state dir is root-only), refuses anything that is not this run's live instance, then hands stdin to the box.
read -r -d '' HOP <<'EOF' || true
set -euo pipefail
label=$1 box=$2
case $label in
  mtboot-tools*) echo "hop: refusing label $label" >&2; exit 3 ;;
  mtboot-?*) ;;
  *) echo "hop: refusing label $label (not mtboot-)" >&2; exit 3 ;;
esac
S=/home/ec2-user/.local/state/$label
ep=$(sudo -n /home/ec2-user/lingua/venv/bin/python -I - "$S" "$label" <<'PY'
import datetime as dt, json, re, sys
root, label = sys.argv[1], sys.argv[2]
ledger = json.load(open(root + "/ledger.json"))
row = json.load(open(root + "/provider-latest.json"))
iid = str(ledger.get("instance_id"))
if ledger.get("label") != label or ledger.get("state") != "created":
    sys.exit("hop: %s is not a live run (ledger state %s)" % (label, ledger.get("state")))
if not iid.isdigit() or iid == "51074906" or str(row.get("id")) != iid or row.get("label") != label:
    sys.exit("hop: provider-latest.json is not this run's instance")
deadline = dt.datetime.fromisoformat(ledger["deadline_utc"])
if dt.datetime.now(dt.timezone.utc) > deadline - dt.timedelta(seconds=120):
    sys.exit("hop: the lease ends in under 2 minutes")
host = row.get("ssh_host") or ""
port = int(row.get("ssh_port") or 0) + (1 if "jupyter" in str(row.get("image_runtype", "")) else 0)
if not re.fullmatch(r"[A-Za-z0-9.-]+", host) or not 0 < port < 65536:
    sys.exit("hop: no SSH endpoint yet (drive.sh status first)")
print(host, port, deadline.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
PY
)
read -r host port deadline <<<"$ep"
exec sudo -n ssh -i /home/ec2-user/.ssh/id_ed25519 -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 \
  -o ServerAliveCountMax=4 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$S/known_hosts" -p "$port" \
  "root@$host" "MTBOOT_DEADLINE_UTC=$deadline bash -c \"\$(printf %s $box | base64 -d)\""
EOF

box() {  # $1 = bash script for the box (runs as root); stdin passes through
  local b64 hop64
  b64=$(printf '%s' "$1" | base64 | tr -d '\n')
  hop64=$(printf '%s' "$HOP" | base64 | tr -d '\n')
  if (( DRY )); then
    printf "[dry-run] ssh %s %s 'bash -c \"\$(printf %%s <hop, %d base64 chars> | base64 -d)\" mtboot-hop %s <box script, %d base64 chars>'\n" \
      "${SSH_OPTS[*]}" "$EC2_HOST" "${#hop64}" "$LABEL" "${#b64}"
    if (( ! HOP_SHOWN )); then HOP_SHOWN=1; printf '%s\n' "$HOP" | sed 's/^/[dry-run]   hop| /'; fi
    printf '%s\n' "$1" | sed 's/^/[dry-run]   box| /'
    return 0
  fi
  ssh "${SSH_OPTS[@]}" "$EC2_HOST" "bash -c \"\$(printf %s $hop64 | base64 -d)\" mtboot-hop $LABEL $b64"
}

read -r -d '' PUSH_BOX <<'EOF' || true
set -euo pipefail
umask 077
R=/workspace/distill I=/workspace/distill/.push
mkdir -p "$R"
if [ -e "$R/.lock" ] && ! flock -n "$R/.lock" true; then echo "push refused: box_job.sh is running" >&2; exit 3; fi
rm -rf "$I"; mkdir -p "$I"
tar -xf - -C "$I" --no-same-owner
(cd "$I" && sha256sum -c --quiet MANIFEST.sha256)
install -m 700 "$I/box_job.sh" "$R/box_job.sh"
install -m 600 "$I/job.env" "$R/job.env"
mkdir -p "$R/inputs/harness" "$R/inputs/audio" "$R/state"
cp -f "$I"/inputs/harness/*.bin "$R/inputs/harness/"
cp -f "$I/inputs/harness/meta.json" "$R/inputs/harness/meta.json"
for d in train review; do rm -rf "$R/inputs/audio/$d"; mv "$I/inputs/audio/$d" "$R/inputs/audio/$d"; done
if [ -f "$I/inputs/urls.tsv" ]; then install -m 600 "$I/inputs/urls.tsv" "$R/inputs/urls.tsv"; rm -f "$R/state/inputs.done"; fi
printf '%s\n' "${MTBOOT_DEADLINE_UTC:-}" > "$R/lease_deadline_utc"
n=$(grep -c . "$I/MANIFEST.sha256")
rm -rf "$I"
echo "box: $n files checked and installed under $R; lease deadline ${MTBOOT_DEADLINE_UTC:-unknown}"
EOF

read -r -d '' START_BOX <<'EOF' || true
set -euo pipefail
R=/workspace/distill
[ -x "$R/box_job.sh" ] && [ -f "$R/job.env" ] || { echo "nothing pushed yet (drive.sh push first)" >&2; exit 3; }
[ -z "${MTBOOT_DEADLINE_UTC:-}" ] || printf '%s\n' "$MTBOOT_DEADLINE_UTC" > "$R/lease_deadline_utc"
if ! flock -n "$R/.lock" true; then echo "box: box_job.sh is already running: $(cat "$R/STATUS" 2>/dev/null)"; exit 0; fi
setsid -f nohup bash "$R/box_job.sh" >>"$R/start.out" 2>&1 </dev/null
sleep 3
echo "box: started box_job.sh; STATUS: $(cat "$R/STATUS" 2>/dev/null || echo 'none yet')"
EOF

read -r -d '' WATCH_BOX <<'EOF' || true
R=/workspace/distill
redact() { sed -E 's#https?://[^[:space:]"]+#[URL]#g'; }
echo "== STATUS: $(cat "$R/STATUS" 2>/dev/null || echo 'none (not started?)')"
[ ! -f "$R/DONE" ] || echo "== DONE: $(cat "$R/DONE")"
if [ -e "$R/.lock" ] && ! flock -n "$R/.lock" true; then echo "== box_job.sh is running"; else echo "== box_job.sh is not running"; fi
echo "== job.log, last __LINES__ lines"
tail -n __LINES__ "$R/job.log" 2>/dev/null | redact
[ ! -f "$R/out/train/log.jsonl" ] || { echo "== last evals (out/train/log.jsonl)"; tail -n 3 "$R/out/train/log.jsonl"; }
[ ! -f "$R/results.tar.sha256" ] || echo "== results.tar: $(wc -c < "$R/results.tar") bytes, sha256 $(cut -c1-16 "$R/results.tar.sha256")"
echo "== /workspace: $(df -h /workspace | awk 'NR == 2 {print $4 " free of " $2}')"
EOF

read -r -d '' PULL_CHECK_BOX <<'EOF' || true
R=/workspace/distill
[ -f "$R/DONE" ] && [ -f "$R/results.tar" ] && [ -f "$R/results.tar.sha256" ] || { echo "results.tar is not final yet (drive.sh watch)" >&2; exit 3; }
printf '%s %s\n' "$(wc -c < "$R/results.tar" | tr -d ' ')" "$(cut -d' ' -f1 "$R/results.tar.sha256")"
cat "$R/DONE"
EOF

PULL_CAT_BOX='cat /workspace/distill/results.tar'

# ------------------------------------------------------------------------------------------------------ subcommands

LAUNCH_ARGS=()
launch_args() {
  [[ $LIFETIME =~ ^[0-9]+$ ]] && (( LIFETIME >= 3600 && LIFETIME <= 14400 )) || die "DISTILL_LIFETIME must be 3600-14400 s"
  awk -v p="$MAX_PRICE" 'BEGIN { exit !(p > 0 && p <= 0.30) }' || die "DISTILL_MAX_PRICE must be above 0 and at most 0.30"
  [[ $DISK =~ ^[0-9]+$ ]] && (( DISK >= 60 && DISK <= 200 )) || die "DISTILL_DISK must be 60-200 GB"
  LAUNCH_ARGS=(--launch stock --revision "$REVISION" --lifetime "$LIFETIME" --max-price "$MAX_PRICE"
               --max-transfer-cost "$MAX_TRANSFER" --disk "$DISK" --set MUSETALK_SKIP_EAGER_UNET=0)
  if [[ -n $MACHINE ]]; then
    [[ $MACHINE =~ ^[0-9]+$ ]] || die "DISTILL_MACHINE must be a number"
    LAUNCH_ARGS+=(--machine "$MACHINE")
  fi
}

cmd_preflight() {
  launch_args
  say "preflight $LABEL: cheapest verified RTX 3090 within \$$MAX_PRICE/h and \$$MAX_TRANSFER/GB, stock launch at ${REVISION:0:7}, lifetime ${LIFETIME}s; no provider write"
  guarded "$(ctl_cmd preflight "$RUN" "${LAUNCH_ARGS[@]}")"
}

cmd_create() {
  launch_args
  (( YES )) || die "create rents a paid RTX 3090 (up to \$$MAX_PRICE/h, lease ${LIFETIME}s); add --yes"
  local mark="$LOCAL/$RUN/create.requested"
  [[ ! -e $mark ]] || die "a create for $LABEL was already sent from this Mac ($(cat "$mark")); reconcile with status, never create twice"
  say "create $LABEL: one create request; the controller arms $LABEL-expiry before it and refuses an existing state dir"
  if (( ! DRY )); then
    mkdir -p "$LOCAL/$RUN"
    (set -o noclobber; date -u +%Y-%m-%dT%H:%M:%SZ > "$mark") || die "create intent already exists; use status only"
  fi
  guarded "if sudo -n test -e $EC2_STATE/$LABEL; then echo 'refusing: $EC2_STATE/$LABEL exists on EC2 (one create per run)' >&2; exit 3; fi; $(ctl_cmd create "$RUN" "${LAUNCH_ARGS[@]}")"
}

cmd_status() {
  say "status $LABEL: provider state (the controller attaches EC2's SSH key once the box runs)"
  guarded "$(ctl_cmd status "$RUN")"
}

cmd_collect() {
  say "collect $LABEL: bootstrap diagnostics retained on EC2 (no job interruption)"
  guarded "$(ctl_cmd collect "$RUN")"
}

cmd_probe() {
  say "probe $LABEL: bootstrap timeline and /health on the box"
  guarded "$(ctl_cmd probe "$RUN")"
}

s3_names() {  # object names (not keys) directly under s3://$BUCKET/$1*
  if (( DRY )); then printf '[dry-run] aws s3 ls s3://%s/%s --region %s\n' "$BUCKET" "$1" "$REGION" >&2; return 0; fi
  { aws s3 ls "s3://$BUCKET/$1" --region "$REGION" || true; } | awk 'NF >= 4 && $1 != "PRE" {print $4}'
}

cmd_presign() {
  command -v aws >/dev/null 2>&1 || die "aws CLI not found"
  local keys=() line id names have dest uri url tmp n=0 re_key='^[A-Za-z0-9][A-Za-z0-9._/-]*'$'\t''s3://[a-z0-9.-]+/[A-Za-z0-9._/+-]+$'
  if [[ -n $KEYS_FILE ]]; then
    [[ -f $KEYS_FILE ]] || die "no keys file $KEYS_FILE"
    while IFS= read -r line || [[ -n $line ]]; do
      [[ -z $line || $line == \#* ]] && continue
      keys+=("$line")
    done < "$KEYS_FILE"
  else
    say "presign $LABEL: training bundles ($TRAIN_AVATARS) from s3://$BUCKET/$BUNDLE_PREFIX/, review avatars from s3://$BUCKET/$AVATAR_PREFIX/"
    names=$(s3_names "$BUNDLE_PREFIX/")
    # Select distinct characters, never another pose of a review character. The source capture metadata
    # names all 14 gate identities; no full model is loaded by this stdlib-only eligibility check.
    local selected
    selected=$(python3 - "$HB/meta.json" "$TRAIN_AVATARS" "$REVIEW_AVATARS" "$DRY" "$names" <<'PY_IDS'
import json, re, sys
from pathlib import Path
meta, requested, review, dry, names = sys.argv[1:]
rows = json.loads(Path(meta).read_text())["unet"]["files"]
gate = {r["avatar"] for r in rows}
if len(gate) != 14 or any(not isinstance(x, str) or not x for x in gate):
    sys.exit("expected complete metadata for the 14 gate avatars")
def identity(s):
    return re.split(r"_(?:idle|smiling|talking)_", s, maxsplit=1)[0]
blocked = {identity(s) for s in gate | set(review.split(","))}
available = sorted(n[:-4] for n in names.splitlines() if n.endswith(".tar"))
ids = ([s for s in available if "_talking_" in s and re.match(r"^(en|es|ur|zh)_\d{2}_", s)
        and identity(s) not in blocked] if requested == "roster12" else requested.split(","))
if dry == "1" and requested == "roster12" and not available:
    # Explain selection without issuing aws calls or inventing a trainable data manifest.
    print("<12-eligible-roster-talking-ids>")
    sys.exit()
if len(ids) != 12 or len(set(map(identity, ids))) != 12:
    sys.exit("need exactly 12 distinct eligible roster characters (10 train + 2 validation)")
for s in ids:
    if not re.fullmatch(r"(?:en|es|ur|zh)_\d{2}_[A-Za-z0-9_]+", s) or identity(s) in blocked:
        sys.exit("ineligible or held-out character in bundle selection: " + s)
    if s not in available and dry != "1":
        sys.exit("missing phone bundle: " + s)
print("\n".join(ids))
PY_IDS
    ) || die "roster selection failed"
    for id in $selected; do
      if (( DRY )) && [[ $id == '<12-eligible-roster-talking-ids>' ]]; then
        keys+=("bundles/<id>.json"$'\t'"s3://$BUCKET/$BUNDLE_PREFIX/<id>.json" "bundles/<id>.tar"$'\t'"s3://$BUCKET/$BUNDLE_PREFIX/<id>.tar")
      else
        if (( DRY )) || grep -qx "$id.json" <<<"$names"; then keys+=("bundles/$id.json"$'\t'"s3://$BUCKET/$BUNDLE_PREFIX/$id.json"); fi
        keys+=("bundles/$id.tar"$'\t'"s3://$BUCKET/$BUNDLE_PREFIX/$id.tar")
      fi
    done
    for id in $(csv_items "$REVIEW_AVATARS"); do
      names=$(s3_names "$AVATAR_PREFIX/$id.tar.gz")
      (( DRY )) || grep -qx "$id.tar.gz" <<<"$names" || die "no v15 avatar s3://$BUCKET/$AVATAR_PREFIX/$id.tar.gz"
      keys+=("avatars/$id.tar.gz"$'\t'"s3://$BUCKET/$AVATAR_PREFIX/$id.tar.gz")
    done
  fi
  (( ${#keys[@]} )) || die "no S3 keys to presign"
  for line in "${keys[@]}"; do
    [[ $line =~ $re_key || ( $DRY == 1 && $line == *"<id>"* ) ]] || die "bad key line (want <dest>\\t<s3://bucket/key>): $line"
    dest=${line%%$'\t'*}
    [[ $dest != *..* ]] || die "bad destination $dest"
  done
  if (( DRY )); then
    for line in "${keys[@]}"; do
      printf '[dry-run] aws s3 presign %s --expires-in %s --region %s  >> %s  (as "%s<TAB><URL>"; URL not shown)\n' \
        "${line#*$'\t'}" "$URL_SECONDS" "$REGION" "$LOCAL/$RUN/urls.tsv" "${line%%$'\t'*}"
    done
    return 0
  fi
  mkdir -p "$LOCAL/$RUN"
  tmp=$(mktemp "$LOCAL/$RUN/.urls.XXXXXX")
  for line in "${keys[@]}"; do
    dest=${line%%$'\t'*} uri=${line#*$'\t'}
    url=$(aws s3 presign "$uri" --expires-in "$URL_SECONDS" --region "$REGION") || { rm -f "$tmp"; die "presign failed for $uri"; }
    [[ $url == https://* ]] || { rm -f "$tmp"; die "presign of $uri did not return an https URL"; }
    printf '%s\t%s\n' "$dest" "$url" >> "$tmp"
    n=$((n + 1))
  done
  url=""
  chmod 600 "$tmp"
  mv -f "$tmp" "$LOCAL/$RUN/urls.tsv"
  say "wrote $n presigned GETs to $LOCAL/$RUN/urls.tsv (0600; URLs not shown), valid until about $(date -u -v+4H +%H:%MZ 2>/dev/null || date -u -d '+4 hours' +%H:%MZ):"
  say "  $(grep -c '^bundles/.*\.tar	' "$LOCAL/$RUN/urls.tsv" || true) phone bundles, $(grep -c '^bundles/.*\.json	' "$LOCAL/$RUN/urls.tsv" || true) sidecars, $(grep -c '^avatars/' "$LOCAL/$RUN/urls.tsv" || true) review avatars"
}

audio_files() {  # comma list -> absolute paths, one per line
  local p
  for p in $(csv_items "$1"); do
    [[ $p == /* ]] || p="$BENCH_REPO/$p"
    printf '%s\n' "$p"
  done
}

write_job_env() {
  local k v re='^[A-Za-z0-9._,:/=+* -]*$'
  for k in RUN_LABEL DISTILL_SHA DISTILL_VARIANT DISTILL_MAX_MINUTES DISTILL_STEPS DISTILL_RESERVE_MINUTES \
           DISTILL_EXTRA_ARGS REVIEW_AVATARS REVIEW_SECONDS REVIEW_MAX_MINUTES REVIEW_EXTRA_ARGS FACEMESH; do
    v=${!k-}
    [[ $v =~ $re ]] || die "$k holds characters box_job.sh refuses: $v"
    printf '%s=%s\n' "$k" "$v"
  done > "$1"
}

cmd_push() {
  local list="$LOCAL/$RUN/urls.tsv" spec name size f base train=() review=() seen=" " entries=0 age bytes stage
  RUN_LABEL=$LABEL
  DISTILL_SHA="${DISTILL_SHA:-$(git -C "$BENCH_REPO" rev-parse HEAD)}"
  [[ $DISTILL_SHA =~ ^[a-f0-9]{40}$ ]] || die "DISTILL_SHA must be a full 40-hex commit"
  if [[ -z "$(git -C "$BENCH_REPO" branch -r --contains "$DISTILL_SHA" 2>/dev/null)" ]]; then
    say "WARNING: no origin branch here contains $DISTILL_SHA; the box clones it from GitHub, so push it first"
  fi
  bash -n "$HERE/box_job.sh" || die "box_job.sh does not parse"
  for spec in $GATE_FILES; do
    name=${spec%%:*} size=${spec##*:}
    [[ -f $HB/$name && "$(wc -c < "$HB/$name" | tr -d ' ')" == "$size" ]] || die "gate file $HB/$name missing or not $size bytes"
  done
  python3 - "$HB/meta.json" <<'PY_META'
import json, sys
m = json.load(open(sys.argv[1]))["unet"]
assert m["frames"] == 128 and len({r["avatar"] for r in m["files"]}) == 14, "incomplete gate metadata"
PY_META
  while IFS= read -r f; do review+=("$f"); done < <(audio_files "$REVIEW_AUDIO")
  if [[ -n $TRAIN_AUDIO ]]; then
    while IFS= read -r f; do train+=("$f"); done < <(audio_files "$TRAIN_AUDIO")
  else
    for f in "$BENCH_REPO"/data/audio/*; do
      case "${f##*/}" in *.wav|*.mpga|*.mp3|*.m4a) ;; *) continue ;; esac
      case ",$REVIEW_AUDIO," in *"/${f##*/},"*|*",${f##*/},"*) continue ;; esac
      train+=("$f")
    done
  fi
  (( ${#train[@]} && ${#review[@]} )) || die "need training and review clips"
  for f in "${train[@]}" "${review[@]}"; do
    base=${f##*/}
    [[ -f $f ]] || die "no audio file $f"
    [[ $base =~ ^[A-Za-z0-9._-]+$ ]] || die "audio file name $base: only [A-Za-z0-9._-]"
    case $seen in *" $base "*) die "two clips named $base (train and review must not share a clip)" ;; esac
    seen="$seen$base "
  done
  if [[ -f $list ]]; then
    age=$(( $(date +%s) - $(mtime "$list") ))
    (( age < URL_SECONDS - 1800 )) || die "the URL list is ${age}s old and expires soon; presign again"
    entries=$(grep -c . "$list" || true)
  else
    say "WARNING: no URL list ($list): the box downloads nothing new (fine only if its inputs are already there)"
  fi
  PUSH_STAGE=$(mktemp -d "${TMPDIR:-/tmp}/distill-push.XXXXXX")
  trap 'rm -rf "$PUSH_STAGE"' EXIT
  stage=$PUSH_STAGE
  mkdir -p "$stage/inputs/harness" "$stage/inputs/audio/train" "$stage/inputs/audio/review"
  ln -s "$HERE/box_job.sh" "$stage/box_job.sh"
  ln -s "$HB/meta.json" "$stage/inputs/harness/meta.json"
  write_job_env "$stage/job.env"
  [[ ! -f $list ]] || ln -s "$list" "$stage/inputs/urls.tsv"
  for spec in $GATE_FILES; do ln -s "$HB/${spec%%:*}" "$stage/inputs/harness/${spec%%:*}"; done
  for f in "${train[@]}"; do ln -s "$f" "$stage/inputs/audio/train/${f##*/}"; done
  for f in "${review[@]}"; do ln -s "$f" "$stage/inputs/audio/review/${f##*/}"; done
  (cd "$stage" && find -L . -type f ! -name MANIFEST.sha256 | sort | while IFS= read -r f; do shasum -a 256 "$f"; done) > "$stage/MANIFEST.sha256"
  bytes=$(cd "$stage" && find -L . -type f -exec cat {} + | wc -c | tr -d ' ')
  say "push $LABEL: $(grep -c . "$stage/MANIFEST.sha256") files, $(( bytes / 1000000 )) MB -> box:/workspace/distill"
  say "  box_job.sh, job.env (sha ${DISTILL_SHA:0:12}, $DISTILL_VARIANT, ${DISTILL_MAX_MINUTES} min), 3 gate files," \
      "${#train[@]} training clips, ${#review[@]} review clips ($(for f in "${review[@]}"; do printf '%s ' "${f##*/}"; done)), URL list with $entries entries"
  sed 's/^/[drive]   job.env: /' "$stage/job.env"
  if (( DRY )); then
    printf '[dry-run] (cd %s && COPYFILE_DISABLE=1 tar -c -h --format ustar --no-mac-metadata --no-xattrs -f - .) | <the box call below>\n' "$stage"
    (cd "$stage" && find -L . -type f | sort | sed 's/^/[dry-run]   stream| /')
    box "$PUSH_BOX"
    return 0
  fi
  (cd "$stage" && COPYFILE_DISABLE=1 tar -c -h --format ustar --no-mac-metadata --no-xattrs -f - .) | box "$PUSH_BOX"
}

cmd_start() {
  say "start $LABEL: box_job.sh detached on the box; it waits for the bootstrap, stops the :8000 server, then trains"
  box "$START_BOX" </dev/null
}

cmd_watch() {
  [[ $WATCH_LINES =~ ^[0-9]+$ ]] || die "WATCH_LINES must be a number"
  say "watch $LABEL: STATUS, DONE, the last $WATCH_LINES lines of job.log (file reads only)"
  box "${WATCH_BOX//__LINES__/$WATCH_LINES}" </dev/null
}

cmd_pull() {
  local dest="$LOCAL/$RUN" info size sha got_size got_sha
  say "pull $LABEL: results.tar from box:/workspace/distill -> $dest/results.tar"
  if (( DRY )); then box "$PULL_CHECK_BOX"; box "$PULL_CAT_BOX"; printf '[dry-run]   > %s/results.tar.part, then size + shasum -a 256 against the box, then mv\n' "$dest"; return 0; fi
  info=$(box "$PULL_CHECK_BOX" </dev/null) || die "results.tar is not ready"
  read -r size sha <<<"$(head -n 1 <<<"$info")"
  [[ $size =~ ^[0-9]+$ && $sha =~ ^[a-f0-9]{64}$ ]] || die "unexpected answer from the box"
  say "box DONE: $(sed -n 2p <<<"$info")"
  mkdir -p "$dest"
  box "$PULL_CAT_BOX" </dev/null > "$dest/results.tar.part"
  got_size=$(wc -c < "$dest/results.tar.part" | tr -d ' ')
  got_sha=$(shasum -a 256 "$dest/results.tar.part" | cut -d' ' -f1)
  [[ $got_size == "$size" && $got_sha == "$sha" ]] || die "results.tar differs from the box's ($got_size bytes vs $size); kept as results.tar.part"
  mv -f "$dest/results.tar.part" "$dest/results.tar"
  say "results.tar: $size bytes, sha256 ${sha:0:16}, $(tar -tf "$dest/results.tar" | grep -vc '/$') files -> $dest/results.tar"
}

read -r -d '' TIMER_CHECK <<'EOF' || true
L=__LABEL__ bad=0
units=$(systemctl list-units --all --plain --no-legend --type=timer "$L-expiry*" 2>/dev/null | awk '{print $1}')
for u in "$L-expiry.timer" $units; do
  s=$(systemctl is-active "$u" 2>/dev/null || true)
  echo "timer $u: ${s:-unknown}"
  [ "$s" = inactive ] || bad=1
done
exit $bad
EOF

cmd_cleanup() {
  (( YES )) || die "cleanup destroys the rented instance; add --yes"
  local rc=0 out
  say "cleanup $LABEL: controller cleanup (owned instance only: destroy, verify absence, stop the expiry timers)"
  guarded "$(ctl_cmd cleanup "$RUN")" || rc=$?
  say "check: no active expiry timer for $LABEL"
  ec2 "${TIMER_CHECK//__LABEL__/$LABEL}" </dev/null || rc=1
  say "check: the instance is absent at the provider"
  out=$(guarded "$(ctl_cmd status "$RUN")") || rc=1
  [[ -z $out ]] || printf '%s\n' "$out"
  if (( DRY )); then return 0; fi
  grep -q '"owned_state": "absent"' <<<"$out" || { say "the controller does not report the instance absent"; rc=1; }
  if (( rc == 0 )); then say "cleanup verified: instance absent, expiry timers inactive"; else die "cleanup NOT verified; check by hand before leaving it"; fi
}

# ------------------------------------------------------------------------------------------------------------- main

YES=0 KEYS_FILE="" ARGS=()
while (( $# )); do
  case $1 in
    --dry-run) DRY=1 ;;
    --yes) YES=1 ;;
    --keys-file) (( $# > 1 )) || die "--keys-file needs a file"; KEYS_FILE=$2; shift ;;
    -h|--help|help) usage; exit 0 ;;
    -*) die "unknown flag $1" ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done
case " ${ARGS[*]-} $MACHINE $KEYS_FILE " in *"$PROTECTED"*) die "refusing: $PROTECTED is the protected production RTX 4070 SUPER" ;; esac
CMD="${ARGS[0]:-}"
[[ -n $CMD ]] || { usage; exit 2; }
RUN="${ARGS[1]:-}"
[[ -n $RUN ]] || die "usage: drive.sh $CMD <run>"
RUN=${RUN#mtboot-}
[[ $RUN =~ ^[a-z0-9][a-z0-9-]{2,39}$ ]] || die "run name: 3-40 chars [a-z0-9-] (the controller's rule)"
case $RUN in tools*|lingua*) die "refusing run name $RUN (EC2 tools checkout or autoscaler prefix)" ;; esac
LABEL="mtboot-$RUN"
case $LABEL in mtboot-?*) ;; *) die "refusing label $LABEL" ;; esac
(( ${#ARGS[@]} == 2 )) || die "unexpected arguments: ${ARGS[*]:2}"
(( DRY )) && say "dry run: commands are printed, nothing runs on EC2, the box or S3"
case $CMD in
  presign) cmd_presign ;;
  preflight) cmd_preflight ;;
  create) cmd_create ;;
  status) cmd_status ;;
  probe) cmd_probe ;;
  collect) cmd_collect ;;
  push) cmd_push ;;
  start) cmd_start ;;
  watch) cmd_watch ;;
  pull) cmd_pull ;;
  cleanup) cmd_cleanup ;;
  *) die "unknown subcommand $CMD (drive.sh --help)" ;;
esac
