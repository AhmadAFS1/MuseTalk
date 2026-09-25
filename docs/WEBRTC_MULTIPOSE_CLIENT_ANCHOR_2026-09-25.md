# Multipose client and actual transition anchor — 2026-09-25

This follow-up fixes a source-frame mismatch in the transition planner and makes the browser lab use the packaged characters. It does not establish that every transition is imperceptible. The approved source trio, prompts, atlas scoring, audio timeline, face masks and review status are unchanged.

## What changed and why

The compositor starts a source transition from its previous completed frame. The planner used the next unrendered source index to select the incoming frame. At 24fps source / 20fps generation these indices differ; a blink can change eye appearance sharply between them. `MotionBank.plan` now selects against the previous planned source index and records `anchor_frame` in each switch diagnostic. The regression first reproduced the old early switch, then verified that an inadmissible actual anchor defers the switch while source/audio time continue normally.

In the inspected historical Latina case, the held frame was idle33, but selection used idle34. Idle33's measured eye aperture is 17.306px; the selected talking96 subsequently reaches 0.900px during the bridge. Correct selection against idle33 chooses talking198, whose six incoming samples stay within 1.427px of the held anchor. This diagnostic explains a concrete mismatch; it is not a perceptual acceptance score. [Candidate reconstruction and limitations](/workspace/experiments/multipose_perceptual_v6_20260925/CODE_DIAGNOSIS_AND_OPTIONS.md).

The browser lab previously used its legacy example manifest even when a character registry was configured. `GET /webrtc/pose-lab?character=<package-directory-id>` now resolves the selected prepared package through the same source-hash registry as live sessions. It validates artifact paths, default idle, physical cache IDs, preparation receipts, frame counts and FPS. Bad requested IDs return an explicit error. An incomplete unrelated package cannot prevent selection of a usable one. With no registry, the old lab remains clearly labelled as unverified legacy behavior.

Before negotiating the peer or enabling speech, the client compares the actual created session's motion bank routing and three source hashes with the selected package. A mismatch closes the session. Character selection is disabled during a call. The speech selector can send direct talking or talking→smiling→talking cues; under-three-second body policy stays on the server.

The client also previously waited for the stream POST response in its event queue, delaying cancellation. A speech request now reserves its reply identity with `assistant_thinking`, orders the stream dispatch after that preflight, and keeps the response promise out of the protocol event queue. An `AbortController` can stop the audio fetch/upload. Newer event sequence numbers fence a delayed multipart upload even if the server has not yet reserved it. User speech retains its own turn ID while cancellation targets the assistant reply. Session/turn epochs invalidate late callbacks, obsolete creates, and sleeping demo steps. No new public API or server cancellation semantics were introduced.

## Evidence and scope

- **262 Python tests passed** in 13.104s, including the shipped JavaScript executed in Node with controlled DOM/fetch/peer objects, actual manager preflight/cancellation cases, catalog validation, and the prior runtime/codec/factory suite. [Exact command](/workspace/experiments/multipose_client_validation_20260925/run-cpu-validation.sh), [log](/workspace/experiments/multipose_client_validation_20260925/final-regressions.log).
- **15 Node tests passed** for the CDP driver's argument handling, bank/received-media assertions, cancellation proof and protocol lifecycle. [Log](/workspace/experiments/multipose_client_validation_20260925/browser-driver-cpu-tests-final.log). These are CPU tests, not a launched browser.
- **Three actual received Latina recordings passed**: short idle, long talking/smiling, and interrupted speech followed by a new reply. All 688 saved frames retain exact 50ms intervals. Four returns completed in 0.3485–0.4139s. [Gallery](/workspace/experiments/multipose_client_validation_20260925/latina/review.html), [source-bound verification](/workspace/experiments/multipose_client_validation_20260925/latina/verification.json). The captures use the native VP8 profile and the corrected planner. They were driven by the aiortc harness, not by the final browser UI.
- **Live HTTP checks passed for all three characters**: served page metadata → session creation → actual bank/source hash parity → default idle → deletion with subsequent 404. Invalid path input returned 400 and an unknown character returned 404. [Evidence](/workspace/experiments/multipose_client_validation_20260925/http-catalog-validation/evidence.json), [reproducible script](/workspace/experiments/multipose_client_validation_20260925/validate-http-catalog.py). This proves actual endpoint/session integration, not peer negotiation or browser playback. All test sessions and the owned server were stopped afterward.
- **Six visual comparison sheets were inspected**. The first idle→talking entry no longer shows the prior doubled-eye contours in either new long/interrupted sample. Terminal return still briefly softens/doubles the eyelids at frames278–279; all four inspected mouth releases close progressively. Later paths use different phases, so they are not controlled pixel-for-pixel comparisons. [Findings and images](/workspace/experiments/multipose_client_validation_20260925/latina-anchor-comparison/README.md). Normal-speed acceptance remains open.
- The previous [18-recording native v6 matrix](/workspace/experiments/multipose_validation_20260925_v6/review.html) covers all three identities and more edge cases, but predates this planner/client change. Do not describe it as 18 reruns of this revision.

The evidence snapshot is `docs/multipose_client_anchor_validation_2026-09-25.json`. Its code hashes are a post-run checkpoint, not a claim that every file was captured at process startup. The client template changed during the earlier aiortc capture; those recordings do not validate the final UI. Current live HTTP checks and their exact scope are recorded separately in the snapshot.

## Run the actual browser test

The configured package directory IDs are `indian_20260925`, `japanese_20260925`, and `latina_guided_20260925`. The first two differ from their manifest `character_id`; the URL and harness take the directory ID.

Start the existing local runtime, without running LTX simultaneously:

```bash
cd /workspace/MuseTalk
WEBRTC_VP8_ENCODER=native \
WEBRTC_NATIVE_VP8_DIR=/workspace/MuseTalk/.runtime/native_vp8 \
MUSETALK_TRT_PROFILE_ENV_FILE=/workspace/MuseTalk/.runtime/musetalk_trt_local_sm89.env \
WEBRTC_MOTION_ATLAS_DIR=/workspace/experiments/realtime_characters \
WEBRTC_MOTION_ALLOW_UNREVIEWED=1 \
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
bash scripts/run_trt_stagewise_server.sh --profile baseline --host 127.0.0.1 --port 8000
```

`WEBRTC_MOTION_ALLOW_UNREVIEWED=1` is for these candidate pilots only. It does not approve them for users. Wait for the server ready message before running clients.

[Provisioning provenance and review outcome](/workspace/experiments/multipose_client_validation_20260925/browser-provisioning/README.md) are retained on disk; that copy of the restoration script still refers to shared-memory downloads and is not self-contained.

The dependency-free Node22 harness uses actual UI clicks, file input, RTCPeerConnection, inbound VP8/audio statistics, video presentation callbacks and MediaRecorder. It saves a browser re-encode as `received.webm`, not an RTP-exact recording. With `--barge-in`, it holds a stream HTTP response, clicks user speech and abort, waits for at least one second of this generation’s live talking/smiling plus receiver presentation advancement, then proves server cancellation and browser fetch cancellation before releasing the response. It requires a new completed return of at most 0.5 seconds bound to the exact emitted anchor, and verifies a distinct following reply. A queued-only cancellation or a previous return cannot satisfy this proof. It cleans its owned browser process group, temporary profile and session on failure as well as success.

**Execution is still pending explicit approval.** Automatic approval review rejected the original combined download/extraction/browser launch because the root filesystem was nearly full and the downloaded browser would run without a sandbox. A narrower dependency-only operation was approved after size/hash/signature checks. No browser launch occurred. Do not infer launch permission from that narrower approval or bypass the restriction with another wrapper.

Verified provisioning materials remain under `/dev/shm/musetalk-browser-pilot-20260925/`; this is volatile. The executable staging directory was cleaned. `restore-tooling.py` refuses restoration unless 20MiB remains free. Check its provenance and current disk availability before any approved restoration. Do not remove unrelated assets to make room. Profile, cache, logs and recordings should use owned shared-memory paths. Stop and clean verified task-owned tooling before copying results to the overlay.

After approval and verified restoration, the concrete UI test is:

```bash
cd /workspace/MuseTalk
LD_LIBRARY_PATH=/workspace/.tools/musetalk-browser-pilot/deps \
/opt/nvm/versions/node/v22.15.0/bin/node scripts/test_webrtc_pose_lab_browser.mjs \
  --browser /workspace/.tools/musetalk-browser-pilot/chrome-headless-shell \
  --base-url http://127.0.0.1:8000 \
  --character japanese_20260925 \
  --audio /workspace/experiments/japanese_multipose_20260925/audio/long.wav \
  --output /dev/shm/musetalk-pose-lab-japanese-review-new \
  --barge-in --no-sandbox
```

Use a new output directory. This root container requires the explicit sandbox exception, hence the pending approval. The harness accepts loopback origins only; it does not download tooling or install packages. Its pass would prove exercised desktop browser behavior, not mobile integration or visual approval.

## Tasks remaining

- [x] Reproduce and fix the previous-source/next-source anchor mismatch.
- [x] Record three targeted live cases with that correction and verify transport/returns.
- [x] Connect the browser lab to actual prepared character packages and reject mismatched active banks.
- [x] Implement upload/preflight cancellation with separate assistant/user ownership and stale callback guards.
- [x] Exercise served manifests and actual session bank selection/cleanup over HTTP for all three characters.
- [x] Inspect the new entry and return contact sheets and retain remaining defects explicitly.
- [x] Run the full 262-test regression suite and 15 CDP-driver CPU tests.
- [ ] Execute real Chromium against the final lab after the pending approval; save actual media and cancellation evidence.
- [ ] Review transition recordings at normal speed before marking any package reviewed.
- [ ] Resolve any remaining face-blend artifacts using isolated experiments and received-media evidence.
- [ ] Test actual mobile playback, reconnect and interruption when the mobile repository is available.

The next quality work should target the remaining concrete defects. The v6 audit found brief Latina eye/face softness inside blends and Indian beard-texture change at the first MuseTalk composite. The new anchor-only comparison narrows which remain. Do not reroll approved LTX prompts to hide a runtime defect. A blanket six-frame blink gate would reduce mandatory return coverage from 723/723 to 622/723, so it is not safe to apply globally. If needed, evaluate candidate reselection/deferral only on optional expressive entries while audio keeps advancing. For beard texture, compare the saved mask and raw/generated same-source frame before changing mask/composition; preserve the current articulating mouth from the first phoneme. A whole-mouth alpha ramp would weaken speech and is not an accepted fix.
