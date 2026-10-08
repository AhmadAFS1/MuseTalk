#!/usr/bin/env python3
"""Opt-in local live rig adapter: codec evidence and strict missing-data gates.

Uses the canonical client/scorer unchanged. Adds only per-payload counters and
safe codec metadata, never SDP/ICE credentials or encoded media. aiortc1.14.0
internal receive hooks are version-gated. Off-host clocks explicitly INVALID.
This starts clients only when invoked; it never starts/stops a server.
"""
from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import math
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from live_isolated import AUDIO_SHA
import report as strict_report


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def codec_row(codec):
    return {'payload_type': int(codec.payloadType), 'mime_type': str(codec.mimeType).lower(),
            'clock_rate': int(codec.clockRate), 'channels': getattr(codec, 'channels', None)}


def install_codec_hooks(receiver_class, version):
    require(version == '1.14.0', 'codec observer requires reviewed aiortc1.14.0 receive hooks')
    original_receive = receiver_class.receive
    original_packet = receiver_class._handle_rtp_packet
    require(inspect.iscoroutinefunction(original_receive) and inspect.iscoroutinefunction(original_packet),
            'aiortc receive hook shape changed')

    async def receive(self, parameters):
        self._repro_codec_evidence = {'schema': 'aiortc_received_payloads_v1', 'aiortc_version': version,
                                      'negotiated': [codec_row(c) for c in parameters.codecs], 'packets_by_payload': {}}
        return await original_receive(self, parameters)

    async def packet(self, packet, arrival_time_ms):
        payload = int(packet.payload_type)
        result = await original_packet(self, packet, arrival_time_ms)
        evidence = getattr(self, '_repro_codec_evidence', None)
        if evidence is not None:
            counts = evidence['packets_by_payload']
            counts[str(payload)] = counts.get(str(payload), 0) + 1
        return result

    receiver_class.receive = receive
    receiver_class._handle_rtp_packet = packet


def observed_video_codecs(timing):
    codecs = set()
    for evidence in timing.get('received_payloads', []):
        negotiated = {str(row['payload_type']): row['mime_type'] for row in evidence.get('negotiated', [])}
        for pt, count in evidence.get('packets_by_payload', {}).items():
            if count <= 0:
                continue
            require(pt in negotiated, 'received unknown RTP payload type')
            mime = negotiated[pt]
            if mime.startswith('video/') and mime != 'video/rtx':
                codecs.add(mime)
    require(codecs and codecs <= {'video/h264', 'video/vp8'}, 'missing or unreviewed received video codec')
    return sorted(codecs)


def additional_evidence(summary, clock_domain='same_worker', expected_video_codec='video/h264'):
    require(clock_domain == 'same_worker', 'independent monotonic clocks: offhost measurement INVALID')
    require(summary.get('server', {}).get('lifetime_counters') is True, 'lifetime telemetry absent')
    require(not summary.get('unmeasured'), 'canonical live fields unmeasured')
    streams = summary.get('streams') or []
    require(streams and len(streams) == summary['level'], 'missing streams')
    codecs = {}
    for stream in streams:
        require(stream['turns_ok'] > 0 and stream['turns_ok'] == stream['turns_posted'], 'missing/failed accepted speech turn')
        first = stream.get('first_frame') or []
        require(len(first) == stream['turns_ok'] and all(row.get('matched') is True
                and type(row.get('first_frame_s')) in (int, float) and math.isfinite(row['first_frame_s'])
                and row['first_frame_s'] >= 0 for row in first), 'missing per-turn first-frame match')
        ring = stream.get('server_ring') or {}
        require(ring.get('speaking_slots', 0) > 0 and ring.get('ring_gaps') == 0
                and ring.get('ring_missing_entries') == 0, 'missing or incomplete lifetime send ring')
        codecs[str(stream['stream'])] = observed_video_codecs(stream.get('setup_timing') or {})
        require(codecs[str(stream['stream'])] == [expected_video_codec], 'observed wire codec differs from explicit H264 test')
    return {'status': 'EVIDENCE_COMPLETE', 'clock_domain': clock_domain, 'video_codecs_by_stream': codecs,
            'first_frame_measure': 'server first fresh send minus client POST; same worker CLOCK_MONOTONIC only',
            'not_claimed': ['first decoded speech frame latency', 'real-browser/EC2/TURN acceptance', 'NVENC from wire codec alone']}


def install_client_hooks(module):
    original_setup = module.StreamClient.setup

    async def setup(self):
        await original_setup(self)
        self.timing['received_payloads'] = [receiver._repro_codec_evidence for receiver in self.pc.getReceivers()
                                            if hasattr(receiver, '_repro_codec_evidence')]

    module.StreamClient.setup = setup
    original_summary = module.summarize_level

    def summarize(*args, **kwargs):
        result = original_summary(*args, **kwargs)
        result['canonical_verdict_before_strict_evidence'] = result['verdict']
        try:
            result['strict_additional_evidence'] = additional_evidence(result)
        except (ValueError, KeyError, TypeError) as exc:
            result['strict_additional_evidence'] = {'status': 'INVALID', 'reason': str(exc)}
            result['verdict'] = 'INVALID'
        return result

    module.summarize_level = summarize


def safe_encoder_snapshot(server):
    """Config and aggregate observations, explicitly NOT per-call encoder proof."""
    h264 = server.get('h264') or {}
    slots = h264.get('nvenc_sessions') or {}
    return {'h264': {key: h264.get(key) for key in ('impl', 'installed', 'codec', 'max_bitrate')},
            'nvenc_sessions': {key: slots.get(key) for key in
                               ('capacity', 'in_use', 'peak', 'denied', 'opened', 'open_failures')},
            'per_stream_encoder_proven': False,
            'note': 'H264 RTP does not prove NVENC. Review actual encoder-open log and fallback counters.'}


def verify_client_args(args):
    require(args.base_url == 'http://127.0.0.1:8300', 'only same-worker loopback8300 allowed; offhost INVALID')
    require(args.trace_dir and not Path(args.trace_dir).exists(), 'fresh trace directory required')
    require(not Path(args.out_dir).exists(), 'fresh client output directory required')
    require(args.ring >= 256 and math.isfinite(args.poll_interval_s) and 0 < args.poll_interval_s <= 1,
            'full256 send ring and finite poll in (0,1]s required')
    require(args.musetalk_fps == args.playback_fps == 20 and args.batch_size == 16, 'fixed20fps/batch16 required')
    require(not args.record_streams and not args.chain and args.wav_list, 'scored run needs explicit WAV list, no observer/chain')
    require(not args.keep_sessions and args.min_fresh_fraction >= .995 and args.max_held_run <= 2,
            'cleanup and original strict thresholds required')
    require(args.ignore_ice_servers and args.peers_per_shard == 1 and args.client_cpus,
            'explicit isolated local client topology required')
    require(not args.no_pin, 'explicit client CPU affinity required')
    require(len(args.levels) == 1, 'one concurrency level per invocation; gate the next level after PASS')
    require(args.min_steady_s >= 10 and args.settle_s >= 10, 'minimum smoke/settle duration required')
    files = set(args.wav_list.split(','))
    require(len(files) == 1, 'same fixed human recording required on every turn')
    require(strict_report.sha256(next(iter(files))) == AUDIO_SHA, 'human recording SHA mismatch')


def score_trace(args, summary):
    """Run original P1-P3 math and strict acceptance without loosening a bound."""
    spec = importlib.util.spec_from_file_location('canonical_live_trace', ROOT / 'experiments/live15_r5/live_trace_report.py')
    trace = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trace)
    output = Path(args.out_dir) / f'{args.label}_n{summary["level"]:02d}_strict_trace.json'
    previous = sys.argv
    try:
        sys.argv = ['live_trace_report.py', str(Path(args.trace_dir) / f'n{summary["level"]:02d}'),
                    '--settle-s', str(args.settle_s), '--json', str(output)]
        trace.main()
    finally:
        sys.argv = previous
    scored = strict_report.live(strict_report.read(output), summary['level'], args.min_steady_s)
    summary['strict_P1_P2_P3'] = scored
    if scored['status'] != 'PASS' and summary['verdict'] != 'INVALID':
        summary['verdict'] = 'FAIL'


def main(argv=None):
    spec = importlib.util.spec_from_file_location('canonical_live_client', ROOT / 'load_test_webrtc_v2.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Canonical coordinator spawns __file__ for its workers; keep all workers
    # under this same opt-in adapter without editing the original client.
    module.__file__ = str(Path(__file__).resolve())
    args = module.parse_args(argv)
    require(not args.selftest and not args.build_corpus, 'use canonical selftest/corpus tool separately')
    if not args.shard_worker:
        verify_client_args(args)
        require(socket.gethostname() == 'a830e00ce20c', 'not the owned worker hostname: offhost INVALID')
    import aiortc
    install_codec_hooks(aiortc.RTCRtpReceiver, aiortc.__version__)
    install_client_hooks(module)
    original_run_level = module.run_level

    async def run_level(*values, **kwargs):
        result = await original_run_level(*values, **kwargs)
        # One extra read AFTER each timed level, not a new measurement poll loop.
        try:
            import aiohttp
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as http:
                async with http.get(args.base_url + '/webrtc/sessions/stats', params={'view': 'lifetime'}) as response:
                    require(response.status == 200, 'encoder snapshot unavailable')
                    result['server_encoder_after_level'] = safe_encoder_snapshot((await response.json()).get('server') or {})
        except Exception:
            result['server_encoder_after_level'] = {'status': 'UNAVAILABLE', 'per_stream_encoder_proven': False}
        try:
            score_trace(args, result)
        except Exception as exc:
            result['strict_P1_P2_P3'] = {'status': 'INVALID', 'reason': type(exc).__name__}
            result['verdict'] = 'INVALID'
        return result

    module.run_level = run_level
    original_coordinator = module.coordinator

    async def coordinator(parsed):
        # Reject an SSH-forwarded remote server: its reported PID must be an
        # actual local api_server.py process before any sessions are created.
        import aiohttp
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as http:
            started = time.monotonic()
            async with http.get(parsed.base_url + '/webrtc/sessions/stats', params={'view': 'lifetime', 'ring': '256'}) as response:
                require(response.status == 200, 'lifetime endpoint unavailable')
                stats = (await response.json()).get('server') or {}
            ended = time.monotonic()
            require(type(stats.get('monotonic')) in (int, float) and started <= stats['monotonic'] <= ended,
                    'server monotonic timestamp outside local request bracket: offhost INVALID')
            require(stats.get('lifetime_counters') is True, 'lifetime telemetry disabled before client setup')
            pid = stats.get('pid')
            require(type(pid) is int and pid > 0 and b'api_server.py' in Path(f'/proc/{pid}/cmdline').read_bytes(),
                    'server PID is not a local API process: offhost INVALID')
            async with http.get(parsed.base_url + '/worker/state') as response:
                require(response.status == 200 and (await response.json()).get('control_plane_requested') is False,
                        'worker control plane is not isolated')
        return await original_coordinator(parsed)

    module.coordinator = coordinator
    return module.main(argv)


if __name__ == '__main__':
    raise SystemExit(main())
