#!/usr/bin/env bash
# Create (or print the arguments for) the Vast template that the orchestrator rents workers from.
# Secrets are read from the operator's environment and never written into this repo:
#   CHARACTER_WORKER_TOKEN      bearer token for the worker API (same value as CHARFACTORY_WORKER_TOKEN)
#   MUSETALK_AWS_SECRET_ID      ARN/name of lingua/musetalk-worker-runtime (docs/musetalk_worker_secrets.md)
#   AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY   the musetalk-secret-reader key (NOT the S3 runtime key)
#   H3_WORKFLOWS_GIT_TOKEN      read-only GitHub token for the private minimax-h3-workflows repo
# Put CHARACTER_WORKER_TOKEN and H3_WORKFLOWS_GIT_TOKEN in the AWS runtime secret instead if you prefer
# not to have them in the template; the installer exports every key of that secret.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${TEMPLATE_NAME:-charfactory-h3-worker}"
IMAGE="${TEMPLATE_IMAGE:-vastai/pytorch:cuda-12.1.1-auto}"
DISK="${TEMPLATE_DISK_GB:-200}"
PORT="${CHARACTER_WORKER_PORT:-8010}"
SEARCH="${TEMPLATE_SEARCH:-num_gpus=1 gpu_ram>=12 cpu_ram>=30 disk_space>=200 driver_version>=580.65.6 cuda_vers>=13.0 inet_down>=500 reliability>0.98 compute_cap>=890 compute_cap<1000 rented=False}"
for v in CHARACTER_WORKER_TOKEN MUSETALK_AWS_SECRET_ID AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY; do
  [[ -n "${!v:-}" ]] || { echo "set $v in the environment first" >&2; exit 2; }
done
ENV_FIELD="-p ${PORT}:${PORT} -p 22:22 -e CHARACTER_WORKER_PORT=${PORT} -e CHARACTER_WORKER_TOKEN=${CHARACTER_WORKER_TOKEN} \
-e MUSETALK_AWS_SECRET_ID=${MUSETALK_AWS_SECRET_ID} -e MUSETALK_AWS_SECRET_REGION=${MUSETALK_AWS_SECRET_REGION:-us-east-1} \
-e MUSETALK_SECRETS_STRICT=true -e MUSETALK_SECRETS_VERIFY_S3=1 -e AWS_DEFAULT_REGION=${AWS_DEFAULT_REGION:-us-east-1} \
-e AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID} -e AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}"
[[ -n "${H3_WORKFLOWS_GIT_TOKEN:-}" ]] && ENV_FIELD="$ENV_FIELD -e H3_WORKFLOWS_GIT_TOKEN=${H3_WORKFLOWS_GIT_TOKEN}"
[[ -n "${MUSETALK_REF:-}" ]] && ENV_FIELD="$ENV_FIELD -e MUSETALK_REF=${MUSETALK_REF}"
if [[ "${1:-}" == "--print" ]]; then
  echo "name:      $NAME"; echo "image:     $IMAGE"; echo "disk:      $DISK GB"; echo "search:    $SEARCH"
  echo "env:       $(sed -E 's/(TOKEN|SECRET_ACCESS_KEY|ACCESS_KEY_ID)=[^ ]+/\1=<redacted>/g' <<<"$ENV_FIELD")"
  echo "onstart:   $HERE/onstart_cmd.sh"; exit 0
fi
command -v vastai >/dev/null || { echo "install the Vast CLI: pip install vastai" >&2; exit 2; }
vastai create template --name "$NAME" --image "$IMAGE" --env "$ENV_FIELD" --onstart-cmd "$(cat "$HERE/onstart_cmd.sh")" \
  --search_params "$SEARCH" --disk_space "$DISK" --ssh --direct --raw | tee "$HERE/.last_template.json"
echo
python3 - "$HERE/.last_template.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
def walk(x):
    if isinstance(x, dict):
        for k, v in x.items():
            if 'hash' in k.lower() and isinstance(v, str): print(f'{k}: {v}')
            walk(v)
    elif isinstance(x, list):
        for v in x: walk(v)
walk(d)
print('Set CHARFACTORY_TEMPLATE_HASH to the hash_id above (also visible under Templates -> My Templates).')
PY
