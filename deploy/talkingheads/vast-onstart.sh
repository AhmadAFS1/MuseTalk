#!/usr/bin/env bash
# Paste this entire file into Vast's on-start field.
set -Eeuo pipefail
mkdir -p /workspace
bootstrap_tmp=$(mktemp -d /workspace/.talkingheads-bootstrap.XXXXXX)
trap 'rm -rf "$bootstrap_tmp"' EXIT
# Optional private GitHub access. No credential is stored in a Git remote URL.
if [[ -n "${GITHUB_TOKEN:-}" ]]; then
  cat > "$bootstrap_tmp/askpass" <<'ASKPASS'
#!/usr/bin/env bash
case "$1" in
  *Username*) printf '%s\n' x-access-token ;;
  *Password*) printf '%s\n' "$GITHUB_TOKEN" ;;
  *) exit 1 ;;
esac
ASKPASS
  chmod 700 "$bootstrap_tmp/askpass"
  export GIT_ASKPASS="$bootstrap_tmp/askpass" GIT_TERMINAL_PROMPT=0
fi
# Use Git so the same authentication covers both fetching the installer and sources.
git -C "$bootstrap_tmp" init -q
git -C "$bootstrap_tmp" remote add origin https://github.com/AhmadAFS1/MuseTalk.git
git -C "$bootstrap_tmp" fetch --depth=1 origin deploy/vast-talkingheads-20260915
git -C "$bootstrap_tmp" show FETCH_HEAD:deploy/talkingheads/vast-startup.sh > "$bootstrap_tmp/vast-startup.sh"
cp "$bootstrap_tmp/vast-startup.sh" /workspace/vast-startup.sh
bash /workspace/vast-startup.sh
