# MuseTalk on-device timing harness

Measures the Core ML MuseTalk models on a real iPhone (or this Mac) and scores the
UNet with the repo's own gate, so a phone run is directly comparable to the Mac
numbers in `docs/ios_unet_bench_2026-10-30.md` and to the server TensorRT engines.
How the whole Core ML pipeline works (models, chunking, audio windows, blend, pacing)
and where every component stands is in `docs/ios_coreml_implementation.md`.

What it reports per UNet model and compute-unit setting:

- load time (first, and cached), first-prediction latency
- gate statistics on the production captures (`calibration/unet_portable_bs8`,
  fp16 PyTorch reference, per-file mae ≤ 0.01 and max_abs ≤ 0.5)
- per-frame timing: mean, p50/p90/p99, max, frames over 40/50 ms
- compute-plan placement: how many ops (and what share of the estimated cost)
  the Neural Engine, GPU and CPU each take
- a sustained run (back-to-back frames for N seconds, bucketed by 5 s, with the
  thermal state), the UNet → TAESD chain per frame, and the Whisper front end
  (log-mel + encoder, with the feature error against fp32 torch)

Layout:

```
ios/MuseTalkBench/
  Package.swift                 HarnessCore (shared) + bench-mac (CLI)
  Sources/HarnessCore/          loading, timing, metrics, compute plan, report
  Sources/bench-mac/            command-line runner for this Mac's Neural Engine
  App/project.yml               xcodegen spec for the iOS app
  App/MuseTalkBench.xcodeproj   generated (committed so Xcode opens it without xcodegen)
  App/Sources/                  SwiftUI app: options, run/stop, log, share the JSON
  HarnessBundle/                GENERATED, gitignored: models + inputs + meta.json
```

## 1. Build the resource bundle (once per model change)

From the repo root, in `.venv-ct` (torch 2.7 + coremltools 9, see the bench doc §10):

```bash
.venv-ct/bin/python scripts/ios/prepare_harness_bundle.py
```

Default models: the two chunks of the Apple-layout int8 UNet cut in PyTorch at the mid/up
boundary with fp16 boundaries (`unet_anec0_int8w_b1_chunk1/2`, 330 + 486 MB, made by
`scripts/ios/export_coreml_ane_chunks.py --cut 0 --check`; coremltools' `bisect_model`, in
`scripts/ios/chunk_coreml.py`, cuts with fp32 boundaries, which the iPhone 12 Pro's Neural
Engine compiler rejected: those chunks ran on the CPU/GPU at 388–870 ms/frame),
`taesd_fp16_b1`, `whisper_encoder_fp16`, `whisper_logmel_fp32`; `--models` picks others
from `models/ios_bench/coreml/manifest.json`. The chunks are what a phone needs: a single
850M-parameter program is compiled inside the app's process (2.46 GB peak measured on an
iPhone 12 Pro against a 3.07 GB allowance) and writes a 1.6 GB cache, which is what killed
the unchunked runs; the harness runs chunk1 → chunk2 per frame, outputs feeding inputs by
name, and reports them as one UNet named `<parent>_chunked`.
It also writes the 128 gate frames (positional encoding removed, since the Core ML
graph adds it) and a 30 s Whisper segment with fp32 reference features. Total ≈ 1.66 GB.

## 2. Sanity run on this Mac

```bash
cd ios/MuseTalkBench && swift build -c release
.build/release/bench-mac --resources HarnessBundle --out ../../docs/ios_unet_bench/harness_mac_m5pro.json
```

The Mac run should reproduce the Python bench (≈19 ms/frame for `unet_ane_int8w_b1`
on the M5 Pro Neural Engine, gate mae ≈ 0.013); if it does not, the harness is
wrong, not the phone.

## 3. iPhone

Phone side: Developer Mode on (Settings → Privacy & Security → Developer Mode),
plugged in over USB, unlocked, "Trust this computer" accepted. Mac side: an Apple
ID in Xcode (Settings → Accounts); the project signs automatically with team
`563AETFKT9` (change `DEVELOPMENT_TEAM` in `App/project.yml` and regenerate for
another team).

Either open `App/MuseTalkBench.xcodeproj` in Xcode, pick the phone, Run, and tap
*Run benchmark*; or from the command line:

```bash
cd ios/MuseTalkBench/App
xcodebuild -project MuseTalkBench.xcodeproj -scheme MuseTalkBench -configuration Release \
  -destination 'id=<device udid>' -derivedDataPath DerivedData -allowProvisioningUpdates build
APP=DerivedData/Build/Products/Release-iphoneos/MuseTalkBench.app
xcrun devicectl device install app --device <device udid> "$APP"
xcrun devicectl device process launch --device <device udid> --console --terminate-existing \
  com.lingua.musetalkbench --autorun --frames 200 --sustained 30 --exit
```

Flags the phone runs used: `--units ALL,CPU_AND_NE --no-b4 --no-gpu --sustained 30 --frames 200
--accuracy-frames 128 --no-reload --taesd-units ALL,CPU_ONLY --settle 10` (`--settle` pauses between
model loads; `--no-reload` skips the second load; `--models <name>` and `--no-unet` select phases;
`--chain N`, `--no-plan` too). The console shows every phase with the process footprint, the memory
headroom iOS grants, free disk, and memory-pressure / thermal events.

`--autorun` runs with the given options as soon as the app is up, prints the JSON
report between `BENCH_JSON_BEGIN` / `BENCH_JSON_END` on the console, saves it in
the app's Documents folder (visible in Files → On My iPhone → MuseTalk Bench, or
`xcrun devicectl device copy from --device <udid> --domain-type appDataContainer
--domain-identifier com.lingua.musetalkbench --source Documents/<file> --destination .`),
and `--exit` quits when done. Other flags: `--no-b4`, `--no-gpu`, `--no-whisper`.

Expect the first load of each UNet on the phone to be slow (the ANE compiles the
850M-parameter program once and caches it) and the whole default run to take a
few minutes on an A14. Keep the app in the foreground; it keeps the screen on.

### Playing a downloaded avatar bundle with your own audio (what the Lingua module does)

```bash
# export + pack a phone bundle from a v15 avatar cache, upload it
.venv-ct/bin/python scripts/ios/prepare_avatar_bundle.py --avatar <v15 dir>/<avatar_id> --phone-only --out /tmp/pb
python3 scripts/ios/pack_phone_bundle.py /tmp/pb/avatar --out /tmp/pb/dist      # <avatar_id>.tar + .json
aws s3 cp /tmp/pb/dist/<avatar_id>.tar  s3://lingua-musetalk-s3-storage/avatars/ios/v1/
aws s3 cp /tmp/pb/dist/<avatar_id>.json s3://lingua-musetalk-s3-storage/avatars/ios/v1/
# Mac check: a whole turn from that bundle and a TTS WAV (any sample rate), silent
swift run -c release bench-mac --resources HarnessBundle --lipsync-session /tmp/pb/avatar --wav clip.wav --preroll 0.4
# phone: the app downloads the bundle (SHA-256 checked) and the WAV, then plays
xcrun devicectl device process launch --device <udid> --console --terminate-existing com.lingua.musetalkbench \
  --lipsync --exit --avatar-url "$(aws s3 presign s3://lingua-musetalk-s3-storage/avatars/ios/v1/<avatar_id>.tar)" \
  --avatar-sha256 "$(jq -r .sha256 /tmp/pb/dist/<avatar_id>.json)" --avatar-id <avatar_id> \
  --wav "$(aws s3 presign s3://<bucket>/<clip>.wav)" --preroll 0.4
```

The models the phone needs are packed the same way by `scripts/ios/pack_model_pack.py --out /tmp/mp` (one tar per
`.mlmodelc`, a trimmed `meta.json`, `manifest.json`) and live at `s3://lingua-musetalk-s3-storage/musetalk/ios/v1/`;
`RemoteBundle.fetchModelPack` downloads a manifest into a folder that loads exactly like `HarnessBundle`. The
harness app still ships its models in the bundle; the Lingua module downloads them.

## Lipsync tab (streaming view, first version)

`scripts/ios/prepare_avatar_bundle.py --avatar <avatar dir> --audio <clip>` exports one avatar cache
(unique frames as JPEG, masks, latents, the clip as 16 kHz float32 plus an m4a for playback, the repo's
per-frame Whisper windows and a few Python-composed check frames) into `HarnessBundle/avatar/`. The app's
Lipsync tab loads the chunked UNet, TAESD and the Whisper models on the Neural Engine, builds the per-frame
audio windows (`HarnessCore/Lipsync.swift`: the repo's gather and fixed-point blend ported to Swift), renders
frames ahead of an `AVAudioPlayer` clock, skips frames when generation falls behind the audio (what the server
does when it drops; the skip aims at the clock plus the pipeline's own latency in frames, so the frame it
produces is still in the future when it comes out), and displays through a `CADisplayLink`. The frame loop is a two-stage pipeline
(`LipsyncPipeline`): the Neural Engine stage (UNet chunks + TAESD) and the CPU stage (JPEG decode, resize, blend,
CGImage) run on two threads joined by a short queue, so a frame costs the slower stage, not the sum. Launch with
`--lipsync --exit` to autoplay and print a `LIPSYNC_RESULT` JSON line (generated fps, stage times, skipped, stalls,
and the audio route, system volume and whether `play()` started; the same appears in the status line under the
picture, so a silent run can be told from a muted phone or a Bluetooth route).
`bench-mac --lipsync-check HarnessBundle/avatar` runs the same loop on the Mac against the Python references:
windows within the ANE encoder's error, composed frames within 0.17/255; sequential 23.6 ms/frame
(UNet 18.9 + TAESD 1.7 + compose 3.0), pipelined 20.8 ms/frame wall = the Neural Engine stage alone.
On the iPhone 12 Pro (2026-10-31, `docs/ios_unet_bench/lipsync_iphone12pro_pipelined_run2.json`): the 426-frame
Latina clip plays end to end with **291 faces generated and 290 shown in 17.7 s = 16.4 fps, 0 stalls, 0 late
drops**, 135 frames skipped by the clock; Neural Engine stage 65.7 ms, CPU stage 10.4 ms in parallel; 25 frames
pre-rolled; first launch after an install compiles the chunks (28 s) and the Whisper models (4 s). Two bugs the Mac
path never reached were found on the phone and fixed: the frame buffer read its own `count` inside a locked
closure (NSLock is not re-entrant; the view hung on its title with frame 0 stored), and the skip-ahead aimed at
the clock itself, so a 64 ms inference landed 1.5 frames late and 130 of 298 faces were dropped (9.4 fps shown);
it now aims `clock + lead`, lead = pipeline latency in frames + 1 (3 on the A14).

## Notes

- The iOS Simulator has no Neural Engine: Core ML runs on the Mac CPU there, so
  simulator numbers only prove the app runs.
- `MLComputePlan` (placement) needs iOS 17.4+; older devices get timings without it.
- The GPU "accuracy anchor" run (CPU_AND_GPU on 64 frames) is there because the
  same model was exact on the Mac GPU; it is slow on a phone GPU.
