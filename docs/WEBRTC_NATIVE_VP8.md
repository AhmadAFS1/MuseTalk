# Native VP8 for continuous multipose encoding

The installed aiortc/PyAV VP8 encoder recreates its compression context after sufficiently large bandwidth-estimate changes. That reset caused a visible broad texture change during otherwise continuous avatar playback. The optional native backend changes libvpx's bitrate configuration while retaining its reference frames. It does not change the LTX sources, motion routing, face rendering, audio, RTP packet handling or receiver decoder.

## Install and enable

From the MuseTalk repository, use the tested runtime Python:

```bash
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/install_native_vp8.py
/workspace/.venvs/musetalk_trt_stagewise/bin/python scripts/install_native_vp8.py --verify

WEBRTC_VP8_ENCODER=native \
WEBRTC_NATIVE_VP8_DIR=/workspace/MuseTalk/.runtime/native_vp8 \
MUSETALK_TRT_PROFILE_ENV_FILE=/workspace/MuseTalk/.runtime/musetalk_trt_local_sm89.env \
WEBRTC_MOTION_ATLAS_DIR=/workspace/experiments/realtime_characters \
WEBRTC_MOTION_ALLOW_UNREVIEWED=1 \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
bash scripts/run_trt_stagewise_server.sh --profile baseline --host 127.0.0.1 --port 8000
```

The installer downloads one pinned official PyPI wheel and extracts only the encoder binding, upstream source and provenance/license files. It does **not** install the older aiortc package into the environment. `--wheel /absolute/pinned.whl` supports an offline install; `--directory` selects another artifact directory. Existing valid installs are reused; corrupt or incompatible existing directories are rejected without overwriting them. The model server performs no dependency download at startup.

The default `WEBRTC_VP8_ENCODER=pyav` retains the previous behavior. Select the backend before starting the process; do not change encoder registries during an active call.

## Validated contract

- Linux x86_64, glibc at least 2.17, CPython 3.10; installed aiortc 1.14.0, PyAV 16.1.0 and CFFI 2.1.1. The native binding comes from the pinned aiortc 1.11.0 wheel and embeds libvpx 1.13.1. Startup rejects untested package/platform combinations, changed files and failed encoding/reconfiguration probes.
- The native profile requires one receiving video m-line with VP8 support. It selects VP8 plus RTX before remote-offer negotiation, including when a mixed offer lists H264 first. H264-only, multiple-video and non-offer descriptions fail clearly before peer mutation. Audio-first offers and repeated same-codec offers preserve their senders and MIDs. Codec migration during an active call is not supported.
- The three outbound video tracks supply their actual frame duration. A 20fps transport uses 4,500 ticks at 90kHz even when the LTX source is 24fps. The native encoder requires positive duration, preserves timestamps, serializes encode/close and closes its native context once.
- Bandwidth feedback still changes actual encoded output. Keyframe requests still work; dimensions changing intentionally recreate the context. Bitrate changes preserve the context. Per-encoder allocation, bitrate-change and frame counters make this distinction auditable.

The dependency manifest is [native_vp8_manifest.json](../scripts/native_vp8_manifest.json). It binds the official wheel, executable files and preserved BSD-3-Clause/libvpx license and patent notices by size and SHA256. The native wheel is 1.87MB; extracted runtime files are approximately 2.4MB.

## Evidence and tasks

- [x] Compare actual encoded bytes across 500kbps → 562kbps → 1Mbps → 250kbps, including a same-native fixed-rate control.
- [x] Prove one encoder context and no bitrate-induced keyframes over the 320-frame CPU sequence.
- [x] Check PLI/FIR, resizing, packet decoding, exact timestamps, teardown, dependency failures and negotiation against the real installed implementations.
- [x] Preserve upstream notices and provide an idempotent, hash-checked installer without changing the installed environment.
- [x] Validate received WebRTC talking/smiling and return-to-idle with native encoder telemetry.
- [x] Repeat the multi-character edge-case and barge-in recordings using the native profile.
- [ ] Obtain normal-speed visual acceptance and exercise the integration in the mobile client.

The [isolated experiment](/workspace/experiments/native_vp8_20260925/README.md) records actual final-window rates of 558/941/250kbps for 562/1000/250kbps requests. At the first rate change, adjacent-frame luma difference was 0.419 with native adjustment versus 2.244 with the existing reset. These measurements identify the mechanism and compare this source; they are not a general visual-quality score. The [received Japanese pilot](/workspace/experiments/native_vp8_20260925/live-comparison/README.md) verifies 335 saved frames at exact 20Hz, one native context, 26 actual bitrate reconfigurations and only the initial keyframe. The inspected first/worst bitrate-update windows and final idle return retain clothing, hair and background detail. These event comparisons support the mechanism; normal-speed acceptance is still pending.


The [full native v6 repeat](/workspace/experiments/multipose_validation_20260925_v6/README.md) passes 234 CPU tests and 18 received recordings across Japanese, Indian and Latina. All 4,781 saved frames have exact 20Hz cadence, all 30 returns finish within 0.349–0.449 seconds, and barge-in ownership passes for each identity. The [review gallery](/workspace/experiments/multipose_validation_20260925_v6/review.html) contains the actual received audio/video. These are sequential loopback sessions on one GPU, not a client/device or concurrent-capacity benchmark. The test server was shut down cleanly after capture.


Across the 18 received recordings, native telemetry accounts for all 4,781 frames and 285 bitrate changes: each session creates one encoder context and only its initial keyframe. There are no bitrate-induced keyframes. The audit separately reports 68 trailing sent frames after recording stopped; six complete diagnostic JSON rows interleaved with ordinary stdout were recovered with raw-byte provenance. See [native codec evidence](/workspace/experiments/multipose_validation_20260925_v6/native-codec-evidence.json).
