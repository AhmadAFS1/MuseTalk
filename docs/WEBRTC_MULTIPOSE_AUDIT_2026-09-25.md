# Multipose transport and interruption audit — 2026-09-25

This follow-up found defects after the previous 15-recording checkpoint. Passing source geometry and RTP checks did not establish even saved-video cadence or consistent compression across idle and speech.

## Reproduced problems

1. A new `user_speech_started` event replaced the reserved assistant turn ID. A correctly targeted abort of that assistant then failed with `turn_mismatch`. The manager now preserves the reservation identity; the pose-lab client queries the current assistant ID before aborting and retains the newer local user ID for the next reply.
2. aiortc's default MP4 recorder configured its video encoder for 30Hz while the receiver delivered 20Hz. Clean 50ms RTP increments became alternating approximately 33/67ms saved-video increments. The recorder now explicitly uses the playback frame rate and a 90kHz time base, with a sufficiently fine MP4 movie clock. Audio keeps its independent arrival origin. The installed aiortc recorder exposes no public video-rate setting, so the subclass checks its private container/context API and fails clearly if that API changes.
3. Native LTX source frames retained decoded I-frame metadata. The selected VP8 encoder obeyed it, producing a keyframe for every raw idle frame, while generated/flow frames had no forced picture type. In the live diagnostic, the first raw idle frame after recovery switched back to this behavior and lost detail. The CPU codec reproduction shows unchanged source flags yield VP8 keyframe tags `[0,0,0,0]`; clearing inherited flags yields `[0,1,1,1]`, where zero means keyframe.

The live baseline also confirmed a codec reset at the first receiver bandwidth estimate, roughly three seconds after video started. The estimate increased the target from 500,000 to 866,544 bits/s in this run; it was not evidence of a bandwidth decrease. A codec reset can affect texture independently of pose routing. We retained the existing codec negotiation and adaptive bitrate policy while isolating the inherited-frame-flag defect.

## Reproduction evidence

- [Instrumented baseline recording](/workspace/experiments/multipose_perceptual_audit_20260925/diagnostic-baseline/long-talking-smiling.mp4)
- [Baseline encoder and REMB telemetry](/workspace/experiments/multipose_perceptual_audit_20260925/server-baseline.log)
- [CPU codec proof](/workspace/experiments/multipose_perceptual_audit_20260925/cpu-codec-proof.json)
- [CPU reproduction script](/workspace/experiments/multipose_perceptual_audit_20260925/prove_codec_hypotheses.py)
- [Client abort tests executed in V8](/workspace/experiments/multipose_validation_20260925_v5/pose-lab-barge-in-v8.json)

The diagnostic injection lives outside production code under `multipose_perceptual_audit_20260925/diagnostic_python/`. It logs the negotiated encoder, bitrate changes, source picture type, actual VP8 frame tag, timestamps and current motion metadata without changing media behavior. Source hashes are included in its startup record. All runs use the baseline TensorRT profile with `.runtime/musetalk_trt_local_sm89.env` and OMP/OpenBLAS/MKL thread counts of four. The rejected v2 CPU affinity profile must not be reused.

## Isolated result and remaining codec behavior

The [normalized comparison](/workspace/experiments/multipose_perceptual_audit_20260925/codec-normalization-comparison/comparison.json) verifies exact 50ms saved-video cadence in both diagnostic clips. At the return-to-idle boundary, full-frame/face/body Laplacian variance ratios changed from 0.609/0.689/0.451 before the fix to 1.056/1.062/1.052 after it. Consecutive-frame mean absolute difference fell from 2.670 to 1.123. These are diagnostic comparisons across separately timed runs, not a perceptual score or a controlled identical-pixel replay. Actual VP8 keyframes fell from 87 to 7.

The first receiver estimate no longer caused the earlier three-second detail drop. A later codec recreation at the first sufficiently large bitrate increase still changed image texture. That independent compression behavior remains a known limitation. The [CPU bitrate audit](/workspace/experiments/codec_bitrate_audit_20260925/README.md) tested 320 frames per mode: assigning `bit_rate` to the existing VP8 context produced exactly the same encoded payloads as fixed 500kbps, despite requests for 562kbps, 1Mbps and 250kbps. Stock H264 and fixed-ceiling VBV also failed to implement working in-place adaptation. Those shortcuts were rejected. A supported native bitrate reconfiguration path is the next transport experiment; it must prove actual bandwidth response and preserved decoder continuity before replacing the current adaptive reset behavior.

Two existing codec configuration limitations are recorded separately: this installed aiortc1.14 computes negotiated codecs during `setRemoteDescription`, before this server calls `prefer_h264`; default offers therefore still select VP8. Also, `enable_h264_nvenc` patches a factory that this installed H264 encoder does not call. The startup NVENC preference message does not prove hardware encoding. Neither issue was changed by the frame-metadata fix.

## Validation status

The corrected [v5 recordings](/workspace/experiments/multipose_validation_20260925_v5/review.html) pass all live checks: 18 recordings, 4,779 decoded video frames with exact 50ms cadence, 30 completed returns in 0.349–0.450 seconds, and explicit owner-preserving barge-in on all three identities. The [batch runner](/workspace/experiments/multipose_validation_20260925_v5/run-live-validation.py), [verification runner](/workspace/experiments/multipose_validation_20260925_v5/verify-live-validation.py), individual recordings, receiver evidence and hashes are retained. These are sequential loopback receiver tests, not a mobile-network or concurrent-user benchmark.

The final [CPU regression log](/workspace/experiments/multipose_validation_20260925_v5/final-regressions.log) records **212 passing tests in 9.941 seconds**. A transient extra native task seen by one immediate `/proc` thread-count snapshot is retained in [the failed audit log](/workspace/experiments/multipose_validation_20260925_v5/native-worker-snapshot-failure.log) with the original failure. The exact transient did not reproduce in 2,000 subsequent CPU stop/reset cycles: all retained decoder objects had closed references, garbage collection stayed disabled, and final task counts returned to baseline. The test now distinguishes transient task-directory visibility from a persistent worker using a fixed 100ms deadline; separate tests prove persistent-worker rejection. The focused [lifecycle log](/workspace/experiments/multipose_validation_20260925_v5/decoder-lifecycle-regression.log) retains immediate counts and settling times. No production teardown change was needed, and no live decoder failure was observed.

No source bank is visually approved by this audit. Still-frame comparisons, compression diagnostics and timing tests cannot replace normal-speed perceptual review. The separate bitrate-reset texture change remains open.


A [committed verification snapshot](multipose_validation_2026-09-25_v5.json) retains the evidence hashes, source hashes, implementation hashes and measured results.

## Remaining transport task

- [ ] Prototype a supported native encoder bitrate reconfiguration path in an isolated experiment. Preserve the installed environment and the accepted LTX source masters; this task does not require new LTX generation.
- [ ] Replay the retained 320-frame source with 500kbps → 562kbps → 1Mbps → 250kbps requests. Measure actual encoded bytes and decode the result; assigning a Python property is not proof of adaptation. Compare each boundary to both fixed-rate and current-reset controls.
- [ ] Require preserved encoder/decoder continuity, explicit PLI keyframes, valid behavior after a dimension change, and safe failure on an incompatible native dependency. Do not remove the current adaptive behavior until the replacement passes these checks.
- [ ] Repeat received talking/smiling and barge-in playback with encoder telemetry before replacing the transport used by the final gallery. Record any codec/device/network limits explicitly.
- [ ] Review the source switches, mouth release, source breathing/blinks and compression changes at normal playback speed before recording visual acceptance. The mobile client must also exercise the documented API contract; that application repository is absent from this host.
