#!/usr/bin/env python3
"""Per-frame CPU for colorconv, aiortc RTP+SRTP+UDP send, Opus audio, H264 packetize."""
import asyncio, glob, json, os, socket, statistics, time
import numpy as np, cv2, av

AV = "/workspace/MuseTalk/results/v15/avatars/japanese_realtime_talking_7d94520b7f/full_imgs"
out = {}


def timeit(fn, n, warm=20):
    for _ in range(warm):
        fn()
    t = time.perf_counter(); c = time.process_time()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t) * 1000 / n, (time.process_time() - c) * 1000 / n


# (a) color conversion as in webrtc_tracks: from_ndarray(bgr24).reformat(yuv420p)
img = cv2.imread(sorted(glob.glob(AV + "/*.png"))[0])
for (w, h) in [(512, 832), (384, 672)]:
    im = img if img.shape[1] == w else cv2.resize(img, (w, h))
    wall, cpu = timeit(lambda: av.VideoFrame.from_ndarray(im, format="bgr24").reformat(format="yuv420p"), 500)
    out[f"colorconv_pyav_{w}x{h}_ms"] = round(wall, 4); out[f"colorconv_pyav_{w}x{h}_cpu_ms"] = round(cpu, 4)
    wall, cpu = timeit(lambda: cv2.cvtColor(im, cv2.COLOR_BGR2YUV_I420), 500)
    out[f"colorconv_cv2_{w}x{h}_ms"] = round(wall, 4)
    wall, cpu = timeit(lambda: im.copy(), 500)
    out[f"fullframe_copy_{w}x{h}_ms"] = round(wall, 4)

# (b) aiortc RTP send path per packet: RtpPacket+extensions+serialize, SRTP protect, UDP sendto
from aiortc.rtp import RtpPacket, HeaderExtensionsMap
from aiortc.rtcrtpparameters import RTCRtpParameters, RTCRtpHeaderExtensionParameters
from aiortc import clock
from pylibsrtp import Policy, Session

hmap = HeaderExtensionsMap()
params = RTCRtpParameters(headerExtensions=[
    RTCRtpHeaderExtensionParameters(id=1, uri="urn:ietf:params:rtp-hdrext:sdes:mid"),
    RTCRtpHeaderExtensionParameters(id=3, uri="http://www.webrtc.org/experiments/rtp-hdrext/abs-send-time")])
hmap.configure(params)
key = os.urandom(30)
pol = Policy(key=key, ssrc_type=Policy.SSRC_ANY_OUTBOUND, srtp_profile=Policy.SRTP_PROFILE_AES128_CM_SHA1_80)
tx = Session(policy=pol)
rx_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); rx_sock.bind(("127.0.0.1", 0))
rx_sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 22)
dest = rx_sock.getsockname()
payload = os.urandom(1200)


async def rtp_bench(n_frames, pkts_per_frame):
    loop = asyncio.get_running_loop()
    transport, _ = await loop.create_datagram_endpoint(asyncio.DatagramProtocol, remote_addr=dest)

    async def _send(data):  # stands in for RTCDtlsTransport._send_rtp -> ice -> udp
        transport.sendto(tx.protect(data))
    seq = 0
    t = time.perf_counter(); c = time.process_time()
    for f in range(n_frames):
        for i in range(pkts_per_frame):
            p = RtpPacket(payload_type=96, sequence_number=seq & 0xFFFF, timestamp=f * 4500)
            p.ssrc = 1234; p.payload = payload; p.marker = int(i == pkts_per_frame - 1)
            p.extensions.abs_send_time = (clock.current_ntp_time() >> 14) & 0xFFFFFF
            p.extensions.mid = "0"
            await _send(p.serialize(hmap))
            seq += 1
        if f % 50 == 0:
            try:
                while True:
                    rx_sock.recv(4096, socket.MSG_DONTWAIT)
            except BlockingIOError:
                pass
    wall = (time.perf_counter() - t) * 1000; cpu = (time.process_time() - c) * 1000
    transport.close()
    return wall / (n_frames * pkts_per_frame), cpu / (n_frames * pkts_per_frame)

pk_wall, pk_cpu = asyncio.run(rtp_bench(300, 13))
out["rtp_send_per_packet_ms"] = round(pk_wall, 4); out["rtp_send_per_packet_cpu_ms"] = round(pk_cpu, 4)
out["video_pkts_per_frame_2p5Mbps_20fps"] = round(2_500_000 / 8 / 20 / 1200, 1)
out["rtp_send_per_video_frame_ms_est"] = round(pk_cpu * 2_500_000 / 8 / 20 / 1200, 4)

# (c) Opus (aiortc OpusEncoder: resample + libopus 96 kbps stereo) per 20 ms packet
from aiortc.codecs.opus import OpusEncoder
enc = OpusEncoder()
import fractions
pts = [0]
def opus_one():
    fr = av.AudioFrame.from_ndarray((np.random.randint(-3000, 3000, (1, 960 * 2))).astype(np.int16), format="s16", layout="stereo")
    fr.sample_rate = 48000; fr.pts = pts[0]; fr.time_base = fractions.Fraction(1, 48000); pts[0] += 960
    enc.encode(fr)
wall, cpu = timeit(opus_one, 1000)
out["opus_encode_per_20ms_ms"] = round(wall, 4); out["opus_encode_per_20ms_cpu_ms"] = round(cpu, 4)
out["audio_cpu_ms_per_stream_second_est"] = round((cpu + pk_cpu) * 50, 3)

# (d) aiortc H264 packetize of a ~15.6 KB access unit (Annex-B -> FU-A)
from aiortc.codecs.h264 import H264Encoder
nal = b"\x00\x00\x00\x01\x65" + os.urandom(15600)
wall, cpu = timeit(lambda: H264Encoder._packetize(H264Encoder._split_bitstream(nal)), 2000)
out["h264_packetize_15p6KB_ms"] = round(wall, 4)
print(json.dumps(out, indent=1))
