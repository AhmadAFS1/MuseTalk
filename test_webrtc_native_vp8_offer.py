"""Execute the actual offer endpoint against real SDP engines, with no ICE I/O."""
import ast
import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock

from aiortc import AudioStreamTrack, RTCConfiguration, RTCPeerConnection, RTCRtpSender, VideoStreamTrack
from aiortc.sdp import SessionDescription
from fastapi import HTTPException
from scripts.webrtc_native_vp8 import prefer_native_vp8, validate_native_offer


def endpoint(session, mode):
    namespace = {'HTTPException': HTTPException, 'WebRTCOffer': SimpleNamespace,
                 '_require_webrtc': lambda: None,
                 'webrtc_session_manager': SimpleNamespace(get_session=AsyncMock(return_value=session)),
                 'WEBRTC_VP8_ENCODER_STATUS': {'encoder': mode},
                 'RTCRtpSender': RTCRtpSender, '_h264_caps_logged': False,
                 'prefer_native_vp8': prefer_native_vp8, 'validate_native_offer': validate_native_offer,
                 '_wait_for_ice_gathering': AsyncMock()}
    path = Path(__file__).with_name('api_server.py')
    nodes = [node for node in ast.parse(path.read_text()).body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name in ('webrtc_offer', 'prefer_h264')]
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['webrtc_offer']


class NativeOfferTest(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.peers = []
        self.tracks = []

    def peer(self):
        peer = RTCPeerConnection(RTCConfiguration(iceServers=[]))
        peer._RTCPeerConnection__gather = AsyncMock()
        peer._RTCPeerConnection__connect = AsyncMock()
        self.peers.append(peer)
        return peer

    async def asyncTearDown(self):
        for peer in self.peers:
            await peer.close()
        for track in self.tracks:
            track.stop()

    def session(self):
        video, audio = VideoStreamTrack(), AudioStreamTrack()
        self.tracks.extend([video, audio])
        return SimpleNamespace(pc=self.peer(), idle_sender=None, audio_sender=None,
                               idle_track=video, silence_audio_track=audio)

    async def check_success(self, order=('video', 'audio'), *, mode='native', codec_order='mixed'):
        client = self.peer()
        for kind in order:
            transceiver = client.addTransceiver(kind, direction='recvonly')
            if kind == 'video':
                caps = RTCRtpSender.getCapabilities('video').codecs
                if codec_order == 'mixed':
                    caps = ([c for c in caps if c.mimeType.lower() == 'video/h264']
                            + [c for c in caps if c.mimeType.lower() != 'video/h264'])
                elif codec_order == 'vp8':
                    caps = [c for c in caps if c.mimeType.lower() in ('video/vp8', 'video/rtx')]
                transceiver.setCodecPreferences(caps)
        session = self.session()
        offer_endpoint = endpoint(session, mode)
        identities = None
        for _ in range(2):
            offer = await client.createOffer()
            await client.setLocalDescription(offer)
            with contextlib.redirect_stdout(io.StringIO()):
                answer = await offer_endpoint('session', offer)
            from aiortc import RTCSessionDescription
            await client.setRemoteDescription(RTCSessionDescription(**answer))
            media = SessionDescription.parse(answer['sdp']).media
            self.assertEqual([item.kind for item in media], list(order))
            video = next(item for item in media if item.kind == 'video')
            self.assertEqual(video.rtp.codecs[0].mimeType, 'video/VP8')
            self.assertTrue(any(c.mimeType == 'video/rtx' and c.parameters['apt'] == video.rtp.codecs[0].payloadType
                                for c in video.rtp.codecs))
            self.assertEqual(next(item for item in media if item.kind == 'audio').rtp.codecs[0].mimeType, 'audio/opus')
            self.assertTrue(all(item.direction == 'sendonly' for item in media))
            current = (session.idle_sender, session.audio_sender,
                       tuple((id(t), t.mid) for t in session.pc.getTransceivers()))
            if identities is None:
                identities = current
            self.assertEqual(current, identities)
            self.assertEqual(len(session.pc.getSenders()), 2)
            self.assertEqual(session.pc.signalingState, 'stable')

    async def test_h264_first_mixed_offer_selects_native_vp8_and_repeats(self):
        await self.check_success()

    async def test_audio_first_mixed_offer_selects_native_vp8_and_repeats(self):
        await self.check_success(('audio', 'video'))

    async def test_vp8_only_offer_and_repeat(self):
        await self.check_success(codec_order='vp8')

    async def test_h264_only_offer_fails_before_peer_or_sender_mutation(self):
        client = self.peer()
        video = client.addTransceiver('video', direction='recvonly')
        video.setCodecPreferences([c for c in RTCRtpSender.getCapabilities('video').codecs
                                   if c.mimeType.lower() == 'video/h264'])
        offer = await client.createOffer()
        session = self.session()
        with self.assertRaises(HTTPException) as error:
            await endpoint(session, 'native')('session', offer)
        self.assertEqual(error.exception.status_code, 400)
        self.assertIn('requires client VP8 support', error.exception.detail)
        self.assertEqual(session.pc.getTransceivers(), [])
        self.assertIsNone(session.pc.remoteDescription)
        self.assertIsNone(session.idle_sender)
        self.assertIsNone(session.audio_sender)

    async def test_multiple_video_offer_fails_before_mutation(self):
        client = self.peer()
        client.addTransceiver('video', direction='recvonly')
        client.addTransceiver('video', direction='recvonly')
        session = self.session()
        with self.assertRaises(HTTPException) as error:
            await endpoint(session, 'native')('session', await client.createOffer())
        self.assertEqual(error.exception.status_code, 400)
        self.assertIn('exactly one receiving video', error.exception.detail)
        self.assertEqual(session.pc.getTransceivers(), [])

    async def test_non_offer_description_fails_before_mutation(self):
        client = self.peer()
        client.addTransceiver('video', direction='recvonly')
        offer = await client.createOffer()
        session = self.session()
        with self.assertRaises(HTTPException) as error:
            await endpoint(session, 'native')('session', SimpleNamespace(sdp=offer.sdp, type='answer'))
        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(session.pc.getTransceivers(), [])

    async def test_default_pyav_retains_existing_first_offer_negotiation(self):
        # One default offer establishes the old VP8-first behavior. Its later
        # H264-only preference remains an existing default-profile limitation.
        client = self.peer()
        client.addTransceiver('video', direction='recvonly')
        client.addTransceiver('audio', direction='recvonly')
        offer = await client.createOffer()
        session = self.session()
        with contextlib.redirect_stdout(io.StringIO()):
            answer = await endpoint(session, 'pyav')('session', offer)
        video = next(m for m in SessionDescription.parse(answer['sdp']).media if m.kind == 'video')
        self.assertEqual(video.rtp.codecs[0].mimeType, 'video/VP8')


if __name__ == '__main__':
    unittest.main()
