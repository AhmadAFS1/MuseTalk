# TURN Server Setup (Plain Language Guide)

This guide explains what TURN is and how to run a simple TURN server for WebRTC.

---

## What is TURN?
TURN is a relay server used by WebRTC when direct peer-to-peer connections fail (very common on mobile networks). It makes WebRTC reliable across iPhone, Android, and restrictive networks.

Think of it as a fallback path: audio/video goes through TURN only when direct UDP connections do not work.

---

## Do I need it?
- Local dev on the same network: usually no.
- Public internet + mobile users: almost always yes.
- SSH port forwarding: TURN over TCP/TLS is usually required.

---

## Two ways to use TURN

### Option A: Hosted TURN (easiest)
Use a provider like Twilio, Xirsys, or a managed coturn service.
- Pros: fast to set up.
- Cons: costs per GB.

### Option B: Self-host coturn (cheaper, more control)
Run coturn on your own VM or the same server as your API.
- Pros: cheaper at scale.
- Cons: you manage security + bandwidth.

---

## Minimal coturn setup (self-hosted)

### 1) Pick ports
- UDP: 3478 (TURN) and/or a relay range (optional).
- TCP: 3478 or 443 (TURN over TCP/TLS is best for mobile/SSH testing).
- TLS: 5349 (optional) or 443 for TURN over TLS.

### 2) Create credentials
Use a static username/password for testing:
- Username: `webrtc`
- Password: `change_this_password`

### 3) Run coturn (Docker example)
Replace `PUBLIC_IP` and credentials with your values.

```bash
docker run -d --name coturn \
  -p 3478:3478/udp \
  -p 3478:3478/tcp \
  -p 443:443/tcp \
  -e TURN_USERNAME=webrtc \
  -e TURN_PASSWORD=change_this_password \
  -e TURN_REALM=yourdomain.com \
  coturn/coturn:latest \
  -n \
  --log-file=stdout \
  --realm=yourdomain.com \
  --user=webrtc:change_this_password \
  --listening-port=3478 \
  --listening-ip=0.0.0.0 \
  --external-ip=PUBLIC_IP \
  --min-port=40000 \
  --max-port=40100 \
  --no-tls \
  --no-dtls
```

Notes:
- Use `--external-ip` if your server is behind NAT.
- `--min-port/--max-port` are TURN relay ports. Open them in your firewall.
- For production, enable TLS and use valid certs.

---

## Start/Stop coturn (host install)

### Find and stop a running coturn process
```bash
# Find the PID
ps -ef | grep -i 'turnserver\|coturn' | grep -v grep
# or:
pgrep -af turnserver
pgrep -af coturn

# Stop it
sudo kill <PID>
# If it won't stop:
sudo kill -9 <PID>
```

### Start coturn with a config file
```bash
sudo turnserver -c /workspace/turnserver.conf
# or repo helper:
sudo ./scripts/run_turnserver.sh
```

### If coturn runs as a systemd service
```bash
sudo systemctl stop coturn
sudo systemctl start coturn
sudo systemctl restart coturn
```

### If coturn runs in Docker
```bash
docker ps | grep -i coturn
docker stop coturn
docker start coturn
```

---

## Firewall / Security Group ports
Open these on the host (or cloud firewall):
- UDP 3478 (TURN)
- TCP 3478 (TURN over TCP)
- TCP 443 (TURN over TLS, optional but recommended for mobile)
- UDP 40000-40100 (relay ports if you set min/max)

If your provider cannot open ranges, use a hosted TURN service or configure TURN to relay over TCP/TLS only.

---

## TLS/HTTPS requirement
Browsers require HTTPS for WebRTC in production.
- Your API must be HTTPS.
- TURN can be TCP/TLS on 443 for best compatibility.

---

## Example ICE server config (client)
Use this when creating `RTCPeerConnection`:

```js
const pc = new RTCPeerConnection({
  iceServers: [
    { urls: "stun:stun.l.google.com:19302" },
    { urls: "turn:YOUR_TURN_HOST:3478?transport=udp", username: "webrtc", credential: "change_this_password" },
    { urls: "turn:YOUR_TURN_HOST:3478?transport=tcp", username: "webrtc", credential: "change_this_password" },
    { urls: "turns:YOUR_TURN_HOST:443?transport=tcp", username: "webrtc", credential: "change_this_password" }
  ]
});
```

---

## Example ICE server config (aiortc)
Use the same STUN/TURN servers on the server side:

```python
from aiortc import RTCConfiguration, RTCIceServer

config = RTCConfiguration(iceServers=[
    RTCIceServer(urls=["stun:stun.l.google.com:19302"]),
    RTCIceServer(
        urls=[
            "turn:YOUR_TURN_HOST:3478?transport=udp",
            "turn:YOUR_TURN_HOST:3478?transport=tcp",
            "turns:YOUR_TURN_HOST:443?transport=tcp",
        ],
        username="webrtc",
        credential="change_this_password",
    ),
])
```

---

## Forcing TURN-only (reliable testing)
During early testing (especially on SSH tunnels), force TURN relay:
- In client: set `iceTransportPolicy: "relay"`.
- On server: prefer TURN-only if supported by your signaling logic.

This ensures media always flows through TURN.

---

## Troubleshooting checklist
- If WebRTC connects locally but not on iPhone: you likely need TURN over TCP/TLS.
- If ICE fails: check firewall ports and credentials.
- If media is black: verify codecs (H264 + Opus).
- If the connection stalls: force relay and retry.

---

## Cost notes
TURN relays all media traffic, so bandwidth costs apply.
- Expect roughly 0.5 to 1 GB per hour per stream at 720p.
- Hosted TURN charges per GB; self-hosting charges bandwidth from your cloud provider.

### copy and paste below 
cat << 'EOF' > turnserver.conf
realm=37.41.28.10
external-ip=37.41.28.10/172.17.0.4 

listening-ip=0.0.0.0
relay-ip=172.17.0.4 

listening-port=3478
alt-listening-port=443

min-port=40000
max-port=40100

lt-cred-mech
user=webrtc:CHANGE_THIS_PASSWORD

# optional: silence cli warning
cli-password=adminpass

log-file=stdout
simple-log
no-tls
no-dtls
EOF

##install command: 
sudo apt-get update && sudo apt-get install -y coturn
