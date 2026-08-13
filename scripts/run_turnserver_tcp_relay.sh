#!/usr/bin/env bash
# Start coturn for WebRTC relay transport.
#
# This mode is intended for hosts where we do not want to expose a public UDP
# relay range. Prefer the mapped UDP listener for realtime media and expose the
# mapped TCP listener as a fallback for networks that block UDP.

set -euo pipefail

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  cat <<'EOF'
Usage: scripts/run_turnserver_tcp_relay.sh

Starts coturn in WebRTC relay-only mode using .env.webrtc-turn.local.
Only the mapped TURN listeners need public exposure; the internal relay range
is not exposed because both peers allocate through the same local coturn.
EOF
  exit 0
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-$(cd "$SCRIPT_DIR/.." && pwd)}"
ENV_FILE="${TURN_ENV_FILE:-$REPO_ROOT/.env.webrtc-turn.local}"
CONFIG="${TURN_CONFIG:-/tmp/musetalk-turnserver-tcp-relay.conf}"

if ! command -v turnserver >/dev/null 2>&1; then
  echo "turnserver not found. Install coturn first: apt-get update && apt-get install -y coturn" >&2
  exit 1
fi

if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi

detect_public_ip() {
  if command -v curl >/dev/null 2>&1; then
    curl -fsS --max-time 5 https://api.ipify.org || true
  fi
}

detect_private_ip() {
  hostname -I 2>/dev/null | awk '{print $1}'
}

read_proc1_env() {
  local key="$1"
  if [[ -r /proc/1/environ ]]; then
    tr '\0' '\n' < /proc/1/environ 2>/dev/null | awk -F= -v key="$key" '$1 == key {sub(/^[^=]*=/, ""); print; exit}'
  fi
}

VAST_TCP_PORT_1455="${VAST_TCP_PORT_1455:-$(read_proc1_env VAST_TCP_PORT_1455)}"
VAST_UDP_PORT_3478="${VAST_UDP_PORT_3478:-$(read_proc1_env VAST_UDP_PORT_3478)}"
PUBLIC_IPADDR="${PUBLIC_IPADDR:-$(read_proc1_env PUBLIC_IPADDR)}"

if [[ -z "${TURN_LISTEN_PORT:-}" ]]; then
  if [[ -n "$VAST_UDP_PORT_3478" ]]; then
    TURN_LISTEN_PORT=3478
  elif [[ -n "$VAST_TCP_PORT_1455" ]]; then
    TURN_LISTEN_PORT=1455
  fi
fi
TURN_LISTEN_PORT="${TURN_LISTEN_PORT:-3478}"

if [[ -z "${TURN_PUBLIC_TRANSPORT:-}" ]]; then
  if [[ "$TURN_LISTEN_PORT" == "1455" && -n "$VAST_TCP_PORT_1455" ]]; then
    TURN_PUBLIC_TRANSPORT=tcp
  elif [[ "$TURN_LISTEN_PORT" == "3478" && -n "$VAST_UDP_PORT_3478" ]]; then
    TURN_PUBLIC_TRANSPORT=udp
  else
    TURN_PUBLIC_TRANSPORT=tcp
  fi
fi

if [[ -z "${TURN_PUBLIC_PORT:-}" ]]; then
  if [[ "$TURN_PUBLIC_TRANSPORT" == "tcp" && "$TURN_LISTEN_PORT" == "1455" && -n "$VAST_TCP_PORT_1455" ]]; then
    TURN_PUBLIC_PORT="$VAST_TCP_PORT_1455"
  elif [[ "$TURN_PUBLIC_TRANSPORT" == "udp" && "$TURN_LISTEN_PORT" == "3478" && -n "$VAST_UDP_PORT_3478" ]]; then
    TURN_PUBLIC_PORT="$VAST_UDP_PORT_3478"
  else
    TURN_PUBLIC_PORT="$TURN_LISTEN_PORT"
  fi
fi

TURN_TCP_FALLBACK_LISTEN_PORT="${TURN_TCP_FALLBACK_LISTEN_PORT:-}"
TURN_TCP_FALLBACK_PUBLIC_PORT="${TURN_TCP_FALLBACK_PUBLIC_PORT:-}"
if [[ "$TURN_PUBLIC_TRANSPORT" == "udp" && -n "$VAST_TCP_PORT_1455" ]]; then
  TURN_TCP_FALLBACK_LISTEN_PORT="${TURN_TCP_FALLBACK_LISTEN_PORT:-1455}"
  TURN_TCP_FALLBACK_PUBLIC_PORT="${TURN_TCP_FALLBACK_PUBLIC_PORT:-$VAST_TCP_PORT_1455}"
fi

DETECTED_PUBLIC_IP=""
if [[ ! "${TURN_PUBLIC_IP_PINNED:-0}" =~ ^(1|true|TRUE|yes|YES|on|ON)$ ]]; then
  DETECTED_PUBLIC_IP="$(detect_public_ip)"
fi
if [[ -n "$DETECTED_PUBLIC_IP" ]]; then
  TURN_PUBLIC_IP="$DETECTED_PUBLIC_IP"
else
  TURN_PUBLIC_IP="${TURN_PUBLIC_IP:-${PUBLIC_IP:-${PUBLIC_IPADDR:-}}}"
fi
TURN_PRIVATE_IP="${TURN_PRIVATE_IP:-$(detect_private_ip)}"
TURN_USER="${TURN_USER:-webrtc}"
TURN_PASS="${TURN_PASS:-${WEBRTC_TURN_PASS:-}}"
TURN_REALM="${TURN_REALM:-$TURN_PUBLIC_IP}"

# Coturn still allocates relay endpoints internally for aiortc/browser TURN
# allocations. With relay-only TURN-over-TCP on both peers, this range does not
# need to be opened publicly; only the TURN TCP listener does.
TURN_INTERNAL_RELAY_MIN_PORT="${TURN_INTERNAL_RELAY_MIN_PORT:-49160}"
TURN_INTERNAL_RELAY_MAX_PORT="${TURN_INTERNAL_RELAY_MAX_PORT:-49200}"
TURN_RELAY_THREADS="${TURN_RELAY_THREADS:-4}"

case "$TURN_PUBLIC_TRANSPORT" in
  tcp|udp)
    ;;
  *)
    echo "TURN_PUBLIC_TRANSPORT must be tcp or udp; got: $TURN_PUBLIC_TRANSPORT" >&2
    exit 1
    ;;
esac

if [[ -z "$TURN_PUBLIC_IP" || -z "$TURN_PRIVATE_IP" ]]; then
  echo "Could not determine TURN_PUBLIC_IP or TURN_PRIVATE_IP. Set them in $ENV_FILE." >&2
  exit 1
fi

if [[ -z "$TURN_PASS" ]]; then
  echo "TURN_PASS is empty. Set TURN_PASS in $ENV_FILE." >&2
  exit 1
fi

umask 077
listener_transport_config=""
if [[ "$TURN_PUBLIC_TRANSPORT" == "tcp" ]]; then
  listener_transport_config="no-udp"
fi
auxiliary_listener_config=""
if [[ -n "$TURN_TCP_FALLBACK_LISTEN_PORT" && "$TURN_TCP_FALLBACK_LISTEN_PORT" != "$TURN_LISTEN_PORT" ]]; then
  auxiliary_listener_config="aux-server=0.0.0.0:$TURN_TCP_FALLBACK_LISTEN_PORT"
fi

cat > "$CONFIG" <<EOF
realm=$TURN_REALM
external-ip=$TURN_PUBLIC_IP/$TURN_PRIVATE_IP

listening-ip=0.0.0.0
relay-ip=$TURN_PRIVATE_IP

listening-port=$TURN_LISTEN_PORT
$auxiliary_listener_config
$listener_transport_config
no-tls
no-dtls

min-port=$TURN_INTERNAL_RELAY_MIN_PORT
max-port=$TURN_INTERNAL_RELAY_MAX_PORT
relay-threads=$TURN_RELAY_THREADS

lt-cred-mech
user=$TURN_USER:$TURN_PASS
fingerprint

no-cli
log-file=stdout
simple-log
EOF

cat <<EOF
Starting coturn WebRTC relay mode
  config: $CONFIG
  env: $ENV_FILE
  listen: 0.0.0.0:$TURN_LISTEN_PORT/$TURN_PUBLIC_TRANSPORT
  public URL: turn:$TURN_PUBLIC_IP:$TURN_PUBLIC_PORT?transport=$TURN_PUBLIC_TRANSPORT
  TCP fallback: ${TURN_TCP_FALLBACK_PUBLIC_PORT:+turn:$TURN_PUBLIC_IP:$TURN_TCP_FALLBACK_PUBLIC_PORT?transport=tcp}
  relay policy: expose the mapped TURN listener; do not expose the internal relay range for relay-only tests
EOF

exec turnserver -c "$CONFIG"
