# Quick start: TURN + API

## 1) Start coturn
Update `turnserver.conf` with your public IPs and password, then run:
```
sudo turnserver -c /workspace/turnserver.conf
```
To make it repeatable, you can also use the helper:
```
sudo ./scripts/run_turnserver.sh
```

## 2) Set WebRTC env vars (match turnserver.conf)
```
export WEBRTC_TURN_URLS="turn:195.142.145.66:12885?transport=udp,turn:195.142.145.66:12964?transport=tcp"
export WEBRTC_TURN_USER="webrtc"
export WEBRTC_TURN_PASS="YOUR_TURN_PASSWORD"
# optional: force relay-only
# export WEBRTC_STUN_URLS=""
```

## 3) Start the API server
```
python api_server.py --host 0.0.0.0 --port 8000
# or use the helper (edit defaults inside once):
bash scripts/run_api_server.sh
```

## 4) Basic flow
- Create a session: `POST /webrtc/sessions/create?avatar_id=test_avatar&user_id=user1`
- Open the player in a browser: `/webrtc/player/{session_id}` (e.g., `http://195.142.145.66:12774/webrtc/player/<id>`)
- The page handles offer/ICE; you should see the idle avatar video.

Notes:
- If you change IP/port mappings or password, update both `turnserver.conf` and the env vars.
- On new machines, reinstall deps: `bash scripts/install_webrtc_deps.sh`.

## kill server like: 
sudo pkill turnserver