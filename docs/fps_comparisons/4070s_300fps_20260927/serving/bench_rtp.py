import os, time, json, asyncio, threading
from aiortc.rtp import RtpPacket, HeaderExtensionsMap
from aiortc.rtcrtpparameters import RTCRtpHeaderExtensionParameters
from pylibsrtp import Policy, Session
res = {}
m = HeaderExtensionsMap()
m.configure(type("P", (), {"headerExtensions": [
    RTCRtpHeaderExtensionParameters(id=1, uri="urn:ietf:params:rtp-hdrext:sdes:mid"),
    RTCRtpHeaderExtensionParameters(id=2, uri="http://www.webrtc.org/experiments/rtp-hdrext/abs-send-time")]})())
srtp = Session(policy=Policy(key=os.urandom(30), ssrc_type=Policy.SSRC_ANY_OUTBOUND))
payload = os.urandom(1200)
def one(i):
    pkt = RtpPacket(payload_type=96, sequence_number=i & 0xFFFF, timestamp=i * 4500)
    pkt.ssrc = 1234; pkt.payload = payload; pkt.marker = 0
    pkt.extensions.abs_send_time = i & 0xFFFFFF; pkt.extensions.mid = "0"
    return srtp.protect(pkt.serialize(m))
N = 20000
t0 = time.perf_counter(); c0 = time.process_time()
for i in range(N): one(i)
res["rtp_build_serialize_srtp_us_per_packet"] = (time.perf_counter() - t0) * 1e6 / N
# asyncio cross-thread handoff
loop = asyncio.new_event_loop(); th = threading.Thread(target=loop.run_forever, daemon=True); th.start()
async def noop(x): return x
N2 = 3000
t0 = time.perf_counter()
for i in range(N2): asyncio.run_coroutine_threadsafe(noop(i), loop).result()
res["run_coroutine_threadsafe_roundtrip_us_idle_loop"] = (time.perf_counter() - t0) * 1e6 / N2
# UDP sendto cost (loopback) per packet
import socket
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); r = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); r.bind(("127.0.0.1", 0)); r.setblocking(False)
addr = r.getsockname(); data = os.urandom(1240)
t0 = time.perf_counter()
for i in range(N):
    s.sendto(data, addr)
    try: r.recv(2048)
    except BlockingIOError: pass
res["udp_sendto_loopback_us_per_packet"] = (time.perf_counter() - t0) * 1e6 / N
print(json.dumps(res, indent=1))
json.dump(res, open("bench_rtp.json", "w"), indent=1)
