# Handoff: MuseTalk on the iPhone, toward 25 fps (state at 2026-11-03 ~21:30 CST)

This is the full context for continuing the on-device MuseTalk work in another session or with another assistant.
Read it top to bottom once. The detailed engineering record is `docs/ios_coreml_implementation.md` in this repo
(the "implementation doc"); this file explains where things stand, what was decided, what is half-built, and the
rules that must not be broken.

## 1. The goal, in the user's words

- MuseTalk lipsync runs **on the phone** (the Lingua / LumaTalk app's "device renderer"), first target the
  **iPhone 12 Pro (A14)**: **25+ fps**, with **lipsync quality unchanged on video**.
- It must work for **ALL avatars, old and new**, including avatars created later through Lingua's
  `/createCharacter`, with **no per-avatar training step**.
- The user runs SoulX FlashHead tests on the same Mac and the same phone.

## 2. Machines, repos, branches

| What | Where |
|---|---|
| Mac | M5 Pro, 24 GB RAM. **Memory-starved: ~30 GB swap in use**, so don't train or run long GPU jobs on it |
| Phone | iPhone 12 Pro, A14, 6 GB, iOS 26.2, UDID `00008101-00123DA43A51003A`, driven with `xcrun devicectl` |
| MuseTalk fork | `AhmadAFS1/MuseTalk`, local checkout `/Users/ahmadsmacair/code/musetalk-ios-bench` |
| Lingua app/backend | `AhmadAFS1/lingua`, local `/Users/ahmadsmacair/code/lingua/lingua` (shared checkout: commit only your own paths; untracked `work/`, `docs/benchmarks/` and the modified `.claude/worktrees/lumatalk-catalog` are not ours) |
| Vast test-rental controller | `/Users/ahmadsmacair/code/musetalk-startup-timing/scripts/vast_locked_boot_ec2.py`, runs **on the EC2 control host** (`ssh my-ec2`), never on the Mac |
| iOS harness app | `ios/MuseTalkBench/` in the MuseTalk repo (Xcode project `App/MuseTalkBench.xcodeproj`; HarnessCore Swift package; `bench-mac` CLI) |

MuseTalk branches:
- `main` = `fc3c92d`, the user's merge of PR #7 (`claude/ios-25fps`, everything through `9ffbb88`).
- **`25fps-training-lip-network`**, the user-named branch for the current work. Its last reviewed commit is
  `05329ba`. The work-in-progress commit on top of it holds the half-built distillation tooling (§6).
- `claude/ios-smaller-unet` = `05329ba` (same as above before the WIP; can be deleted).

Lingua:
- **Draft PR AhmadAFS1/lingua#14** (`claude/ios-two-workers`, `2209a4f7`, based on `ui-rework`) is still open.
  It sets `inferWorkers = 2` by default, which gives +20 % on the phone with no new model pack. Its phone call
  check is still open.
- Local branch `claude/ios-25fps` = `809cf462` (the same change).
- PRs #11 and #13 are merged to master, #12 to ui-rework.
- Before any EAS build or update, `runtimeVersion` should go past 1.0.2. ExpoModulesCore checks AsyncFunction
  argument counts strictly, so mismatched JS and native builds reject each other. That decision is the user's.

## 3. What was done, in order

1. **2026-10-30, feasibility.** Core ML export of the MuseTalk UNet.
   - The Neural Engine (ANE) runs it, but its fp16 arithmetic gives latent mae ~0.013 against the server
     reference, which fails the repo gate of 0.01.
   - The video review accepted it: 5 of 6 characters pass r5; Japanese fails lip aperture.
   - The UNet ships as **two chunks with fp16 boundaries**: a single program gets the app killed for memory.
2. **2026-10-31 to 11-01, Lingua Phase 0-2.** Avatar device bundles on S3: all 48 roster bundles,
   `s3://lingua-musetalk-s3-storage/avatars/ios/v1/`. The device renderer is in the app. Phone calls ran at
   16.5-20.9 fps shown with 0 stalls. The pose protocol was built. Implementation doc §2-§5.5.
3. **2026-11-02, round 1** (implementation doc §5.6):
   - Levers: SPLIT_EINSUM_V2 attention, the constant time embedding folded out, TAESD on the GPU on the Mac
     only, and two ANE workers.
   - Mac: 1.35x.
   - iPhone: 14.8 → 18.6 fps hot; 19.1 fps when cool on 11-03.
4. **2026-11-03**:
   - **Cache cleanup:** 266 GB of old Core ML compile caches deleted on the Mac.
   - **GPU worker** (§5.7): whole frames on the GPU next to the ANE.
     - Mac: 1.58-1.87x, with video quality equal or better.
     - **iPhone: 0.80x, a regression.** The A14 GPU takes ~500 ms per frame and slows the ANE. It stays off by default.
   - **Three ANE workers on the phone:** 19.0 fps against 19.1 with two. The ANE is saturated with two.
   - **Thermal level recorded in every lipsync result:** `thermal_start/end/changes` in `LipsyncSession`.
     - New reusable phone A/B script `scripts/ios/run_phone_ab.sh`: no fixed cooldown, a Neural Engine check on
       every step, and a SoulX watchdog.
     - Review fixes are in.
   - **Phone storage:** an iOS 27.2 update downloading in the background filled the phone (0.5 GB free).
     - With little free space Core ML silently runs the UNet on the CPU (~1 s/frame).
     - The user deleted the update; the phone now has ~12-20 GB free.
     - Keep several GB free and Automatic iOS update downloads off.
   - **Network-side levers, all rejected on quality or speed** (§3.5):
     - frame interpolation;
     - the UNet on a cropped lower face;
     - token downsampling;
     - `fastPrediction` (4 % slower on the phone).
   - **Smaller UNet speed sweep** (§5.8), untrained, BK-SDM shapes. The result that matters:
     - **bk_tiny renders every frame of the 24 fps clip on the iPhone 12 Pro: 24.0 fps, 0 skipped, even at
       thermal "serious".** Capacity is about 26-28 fps.
     - Today's model gives 17.3 fps in the same run.
     - Untrained, its output is garbage, so it needs **distillation**. (The one untrained score taken was bk_small's:
       mae 0.73. bk_tiny was not scored; it keeps fewer of the full model's weights.)
   - Started building the distillation round-1 tooling (§6). It was stopped by the user's request before review.

## 4. Key numbers (all measured; sources in the implementation doc)

| | iPhone 12 Pro | Mac M5 Pro |
|---|---|---|
| Shipped chunks, 1 worker | 14.8 fps hot, 15.7 cool | 21.0 ms/frame |
| Folded V2 chunks, 2 ANE workers (today's best) | 18.6 hot, 19.1 cool (17.3 at "fair→serious") | 15.5 ms |
| + 1 GPU worker | 15.1 fps (0.80x, rejected) | 9.8 ms (1.58x) |
| 3 ANE workers | 19.0 fps (no gain) | no gain |
| UNet alone, today's chunks | 52.0-53.9 ms | 16.1-16.4 ms |
| bk_small (untrained, plain layout, one program) | UNet 37.5-37.9 ms; pipeline 18.2-21.5 fps | 14.2 ms |
| **bk_tiny (untrained, plain layout, one program)** | **UNet 32.5-33.0 ms; pipeline 24.0 fps, all 426 frames** | 13.1 ms |
| bk_base / full in the plain layout as one program | CPU fallback / memory kill (need two Apple-layout chunks) | 15.2 / 23.8 ms |

Quality yardsticks, on the 128 gate frames against the server fp16 reference:
- full UNet fp32: mae 0.0004;
- Neural Engine: 0.013, the accepted on-device error;
- repo gate: 0.01.

Video review: r5 thresholds on 12 clips (6 avatars × 2 audios, §5.7). Today's phone chain (arm D) passes 10 of 12.

## 5. Rejected levers: don't retry (details in implementation doc §3.5)

- int8 activations
- 4/6-bit palettization
- fp32 islands
- batch 4: the A14 is compute-bound, only −3.8 %
- deep-feature reuse across frames
- frame interpolation: mae 0.107-0.134
- the UNet on the lower 24 latent rows: 0.16-0.20
- token downsampling: 0.015 with max_abs 3.8, for only 5-8 % of the time
- `fastPrediction`: −4 % on the phone
- TAESD off the ANE stage on the A14: GPU 67 ms, CPU slows the ANE
- a GPU worker on the A14
- a third ANE worker
- `MLComputeUnits.all` on the phone
- `bisect_model` chunks: fp32 boundaries fall to CPU/GPU

The A14's Neural Engine time is real compute. A Mac profile shows conv 36 %, linear 24 %, softmax 7 % and
reshape/transpose 13 %. **The only path to 25 fps on this phone is a smaller network.** Newer iPhones are unmeasured.

## 6. Current work: distil a smaller UNet (round 1), and where it stopped

**What distillation means here.** Train the smaller network (bk_tiny: 318M parameters, −39 % compute) to copy the
full UNet's output. It is done **once, for the shared network, not per avatar**. Avatars are only inputs: face
latents and audio. After training, old and new avatars use it exactly as they use today's UNet. It ships only if it
passes the lipsync video review on avatars it never saw. The student starts from the full model's weights for every
block it keeps; that is BK-SDM's recipe, so the data needs are small.

**User decisions, 2026-11-03:**
- Rental: a **stock launch** with MuseTalk code at `32b7349` (the commit the Vast character workers pin), no
  production template hash.
- Test avatars: kept **on the test GPU only**, never published to the app.
- **Spending cap $10 in total.** A 3090 at ≤$0.30/h makes that ~30 h; round 1 is ~4 h, about $1.20.
- The user did not want to be surprised by spending: say what will be rented and its cost first.

**Data plan:**
- **Training:** latents of 10 roster characters from the phone bundles `avatars/ios/v1/<id>.tar`. Each tar's
  `latents_f16.bin` is `[N,8,32,32]` fp16, the UNet input per unique frame. All 48 bundles' latents total ~171 MB.
- **Validation:** 2 more roster characters, used only for checkpoint selection.
- **Test, never in training:**
  - the 14 gate-frame avatars (`HarnessBundle/unet_*_f16.bin`, not in git; 16 gate frames overlap Japanese/Latina);
  - the 6 review avatars: `en_01_maya`, `en_03_marcus`, `es_01_valeria`, `es_03_mateo`, `japanese_realtime`,
    `latina_guided`. Their v15 caches `avatars/v15/<id>.tar.gz` are needed for rendering;
  - the review audio `data/audio/ai-assistant.mpga` and `eng.wav`.
- **Audio for training:** the other 9 clips in `data/audio` (~150 s). That is thin; more speech would help (§8).
- **Box access to S3:** presigned GET URLs made on the Mac (`aws s3 presign`, default profile, 4 h). No
  long-lived credentials on Vast, and the URLs are never printed.

**Scripts** (on `25fps-training-lip-network`):

| File | State |
|---|---|
| `scripts/ios/export_small_unets.py` | Reviewed and committed (05329ba). WIP adds `build(..., processor="train")`, `--weights` for trained exports, and the bk_tiny up-block weight remap |
| `scripts/ios/distill_small_unet.py` | **WIP, built, tiny-tested on the Mac CPU** (4 steps + resume + strict load), **not run on CUDA, not reviewed**. Features below |
| `scripts/ios/review_distilled_unet.py` | **WIP, built, tiny-tested on the Mac** (12 frames; untrained student → FAIL as expected), **not reviewed**. Details below |
| `scripts/ios/distill_box/box_job.sh`, `drive.sh` | **WIP, incomplete**: the builder was stopped mid-work. They pass `bash -n`; their dry-run output was not checked and they were never reviewed. **Review and test them before any rental** |
| `docs/ios_unet_bench/small_unets_2026-11-03/run_phone_small_unets.sh` | Committed. Phone speed sweep of single-program UNets |

`distill_small_unet.py` features:
- fp16 teacher, bf16-autocast student, SDPA;
- `latents.pt` and `latents_f16.bin` input, de-duplicated;
- a validation split for checkpoint selection; the gate frames are test-only;
- an audio-window cache;
- warmup + cosine LR, `--max-minutes`, and atomic resume checkpoints;
- `<variant>.json` for the stock UNet wrapper.

`review_distilled_unet.py` details:
- Arms: A = full fp32 with TF32 off, N = full fp16, S = student fp32, S16 = student fp16.
- Outputs: videos, side-by-sides and r5 metrics.
- **Verdict:** every clip passes r5 against A **and** the student is non-inferior to today's phone chain (arm D in
  `docs/ios_unet_bench/review_mixed_units/metrics.json`). The margins are constants at the top of the script.

**Rental plan** (from the read-only mapping of the controller):
- Run on `my-ec2` as root:
  `sudo -n /home/ec2-user/lingua/venv/bin/python -B /home/ec2-user/.local/state/mtboot-tools-4e2b02e/scripts/vast_locked_boot_ec2.py <mode> <run>`
- Modes: `preflight`, `create` (**exactly once**; it installs the expiry timer before the PUT), `status`, `probe`,
  `collect`, `cleanup`, `expire`.
- Flags: **`--launch stock` is mandatory** (the default sends the production template hash),
  `--revision 4e2b02e5cb72cce94f415bce8099222fb25b8156`, `--lifetime` ≤ 14400 s (extend cap 4 h in total),
  `--max-price 0.30`.
- The label is forced to `mtboot-<run>`. Names used before: c1-a … c8-a, pe-h1, hc-a, hc-c, hc-e.
- The controller **cannot boot `32b7349`**: that commit lacks the locked bootstrap. So boot `4e2b02e`, stop its
  :8000 server, and check our branch out beside it in `/workspace/distill/MuseTalk`. Leave `/workspace/MuseTalk`
  alone.
- SSH reaches the box only through EC2:
  `ssh my-ec2 'sudo -n ssh -i /home/ec2-user/.ssh/id_ed25519 -o UserKnownHostsFile=/home/ec2-user/.local/state/mtboot-<run>/known_hosts -p <port> root@<host>'`
  The host and port are in `provider-latest.json`.
- The box needs `models/musetalkV15`, `models/whisper` and `models/taesd`, the three gate `.bin` files from the
  Mac, the audio clips, and mediapipe 0.10.14 in a small venv for the FaceMesh landmarks.
- Server libraries are diffusers 0.30.2 with torch cu121. Building `mid_block_type=None` there is unverified, so
  check it first on the box.
- After the run: `collect`, pull the results, then `cleanup`. Verify the instance is absent and
  `systemctl is-active mtboot-<run>-expiry.timer` says `inactive`.

**Round 2, not started: brand-new avatars.** The `/createCharacter` pipeline lives in
`character_factory/character_worker`:
- MiniMax H3 runs locally in ComfyUI from public HF weights (61 GB); there is no paid API.
- It is accepted only on Ada GPUs (4070S/4090) and refuses a 3090 by default. It needs driver ≥ 580.65 and
  ≥ 150 GB of disk; a job takes ~15 min on a 4090, after a 19-55 min first boot.
- Keep the outputs local with "Route B" (no S3): `batch_three_pose.py` plan/generate, then `video_qc.py`, then a
  private MuseTalk server with S3 off, then `/avatars/prepare` and `/generate`.
- Portraits come from the wizard's 12 presets or paid OpenAI image generation. No image model runs on any box today.
- Do this only after round 1 passes, with the user present.

**Phone steps after a pass:**
1. Export the trained student: `export_small_unets.py --weights`, ideally as two Apple-layout chunks. That needs
   Apple's `ml-stable-diffusion` code, whose `/tmp` clone is gone; re-downloading it needs the user's OK.
2. Time it with `run_phone_ab.sh`.
3. Score its on-device (ANE) quality on the Mac with the `review_mixed_units.py` approach.
4. Then a Lingua model pack and the renderer switch, behind a flag.

## 7. Rules that must not be broken

- **Never touch Vast instance 51074906** (the production RTX 4070 SUPER). Read-only listing is fine.
- **Never rent with the production template hash `4dc0868e…`.** When the autoscaler is on, it destroys unregistered
  instances with that hash.
- Test rental labels must not start with `lingua-` (the controller enforces `mtboot-`).
- Every rental gets an expiry timer *before* create. Price ≤ $0.30/h. Destroy and verify after every run.
- Never change autoscaler settings.
- **Docker/GHCR for MuseTalk workers is retired:** don't revive it. Don't push `codex/rtx3090-r5-delivery` or
  `codex/rtx3090-syncnet-startup-repair`.
- Ask before merging to MuseTalk `main` or Lingua `master`. Open draft PRs. Keep Lingua's
  `VAST_CHARACTER_MUSETALK_REF=32b7349`.
- Never print credentials or presigned URLs. GitHub token: `printf 'protocol=https\nhost=github.com\n\n' | git credential fill`;
  `gh` is not logged in, so use the REST API with `curl --data-binary @file.json`.
  The Mac has AWS (default profile) but no Vast key; Vast is reached only through EC2.
- Never enter the user's sign-in credentials anywhere.
- SoulX FlashHead tests share the Mac and the phone: never kill or renice their processes, and never take the phone
  while a SoulX app runs. The phone scripts watch for one. Interleave A/B runs.
- Don't restart the user's phone. Never empty the Trash or delete permanently without the user's explicit ask.
- Commits end with a `Co-Authored-By` line naming the assistant. PR bodies end with the tool's attribution line.
- **Spend:** say what you'll rent and what it costs before renting; the cap is $10 in total.

## 8. How to run things (copy-paste)

Phone A/B (lipsync tab, thermal-gated, mirrored order):
```bash
APP=/Users/ahmadsmacair/code/musetalk-ios-bench/ios/MuseTalkBench/App/DerivedData/Build/Products/Release-iphoneos/MuseTalkBench.app \
  /Users/ahmadsmacair/code/musetalk-ios-bench/scripts/ios/run_phone_ab.sh <out dir> "D T T D" \
  "D=--unet unet_anev2fc0_int8w_b1_chunked --infer-workers 2" "T=--unet unet_bk_tiny_int8w_b1 --infer-workers 2"
```
The app bundle (`ios/MuseTalkBench/HarnessBundle`, gitignored) now holds:
- the shipped chunks `unet_anec0`;
- the folded V2 chunks `unet_anev2fc0`;
- the four plain sweep UNets (`unet_int8w_b1`, `unet_bk_{base,small,tiny}_int8w_b1`).

The batch-4 chunks were moved to `models/ios_bench/harness_unused/`; there's a meta backup in the old session scratch.

Build the harness app:
```bash
cd /Users/ahmadsmacair/code/musetalk-ios-bench/ios/MuseTalkBench/App
xcodebuild -project MuseTalkBench.xcodeproj -scheme MuseTalkBench -configuration Release -destination 'generic/platform=iOS' -derivedDataPath DerivedData -allowProvisioningUpdates build
```

Read the phone's free storage (no model load):
```bash
xcrun devicectl device process launch --device 00008101-00123DA43A51003A --console --terminate-existing com.lingua.musetalkbench --autorun --no-unet --no-whisper --no-gpu --no-b4 --chain 0 --sustained 0 --no-plan --exit
```

Speed of a UNet variant on the Mac Neural Engine:
```bash
cd /Users/ahmadsmacair/code/musetalk-ios-bench && .venv-ct/bin/python scripts/ios/export_small_unets.py --variants bk_tiny --time
```

Distillation and review (on a CUDA box; see each script's docstring for every option):
```bash
python scripts/ios/distill_small_unet.py --variant bk_tiny --device cuda --latents '<train glob>' --val-count 2 --audio '<clips glob>' --max-minutes 150 --out <dir>
python scripts/ios/review_distilled_unet.py --avatars <6 review avatar dirs> --audio data/audio/ai-assistant.mpga,data/audio/eng.wav --student-json <dir>/bk_tiny.json --student-weights <dir>/bk_tiny_best.pt --out <dir>/review --device cuda --facemesh-python <mediapipe venv python>
```

## 9. Open items, in order

1. **Review and finish `scripts/ios/distill_box/`**, and review `distill_small_unet.py` and
   `review_distilled_unet.py`. The planned reviewer lenses:
   - training and eval correctness on CUDA with diffusers 0.30.2;
   - rental safety: one box, no production hash, no 51074906, nothing left running, no secrets printed, and the
     nested-ssh quoting.
2. Round 1 rental (§6), within the $10 cap. Put the verdict in implementation doc §5.8 and tracker row 24.
3. If it passes: round 2 (new avatars on a 4090, local only), then the phone export (Apple-layout chunks) and an
   on-device quality check.
4. If it fails: one more round with one documented change (more steps, more speech data, bk_small, or another LR).
   Possible extra speech: public datasets would need the user's OK to download.
5. Lingua: the lingua#14 phone call check, and the `runtimeVersion` decision.
6. Unmeasured: newer iPhones (A15+) with today's network; a real 4 GB phone.

## 10. The overnight `/goal` prompt drafted for the user (needs §9 item 1 first)

```text
/goal Overnight, unattended: train and quality-test the smaller MuseTalk UNet (bk_tiny) for 25 fps on the iPhone 12 Pro, then stop. Repo ~/code/musetalk-ios-bench, branch 25fps-training-lip-network; context in docs/ios_coreml_implementation.md §5.8 and docs/HANDOFF_ios_25fps_2026-11-03.md.
1. Finish the build: review and fix distill_small_unet.py, review_distilled_unet.py and scripts/ios/distill_box/ (incomplete), rerun the tiny local tests (keep them tiny: the Mac is low on memory), commit and push.
2. Rent exactly ONE RTX 3090 through scripts/ios/distill_box/drive.sh and the EC2 controller: --launch stock (never the production template hash 4dc0868e), revision 4e2b02e, label mtboot-<new run name>, max $0.30/h, lifetime at most 4 h with the expiry timer installed before create. Run create exactly once; on an uncertain result use status, never create again. Never touch Vast instance 51074906, never change autoscaler settings, never print credentials or presigned URLs.
3. Data: train on 10 roster characters' phone-bundle latents (avatars/ios/v1) and hold 2 more out for checkpoint selection. Never train on the review set (Maya, Marcus, Valeria, Mateo, Japanese, Latina), the 14 gate-frame avatars, or the review audio (ai-assistant.mpga, eng.wav).
4. Train bk_tiny on the box for up to 150 minutes, resuming after any dropped connection. Report the gate-frame score at every eval.
5. Review: render full fp32, full fp16 and student lipsync for the 6 review avatars x 2 held-out clips with review_distilled_unet.py. Pass = every clip passes the r5 thresholds against the full model AND the student is no worse than today's phone chain (the D numbers in review_mixed_units).
6. Pull results to the Mac: docs/ios_unet_bench/small_unets_2026-11-03/round1/ (logs, metrics, summary, a few side-by-side videos under ~50 MB in total). Put the student weights in models/ios_bench/distilled/ (not in git). Then destroy the rental with drive.sh cleanup and verify the instance is gone and its expiry timer inactive.
7. If round 1 fails and less than $5 has been spent, one second rental is allowed with one documented change, same rules. Never exceed $10 in total.
8. Record the verdict in §5.8 and the tracker, commit and push the branch. Do not merge, do not open a PR, do not touch the phone or Lingua.
Done when: a pass/fail verdict with numbers is committed and pushed, every rental is destroyed and verified gone, and total spend is reported (at most $10).
```

## 11. State at handoff

- Nothing is running from this work: no rental, no training, no phone job.
- The build workflow and the Lingua Metro dev server on port 8083 were stopped. Two older Expo servers (ports 8082
  and default, started about a week earlier from other sessions) were left alone.
- The phone is the user's; the harness app on it holds the bk_* sweep models.
- Uncommitted work at handoff: none. The WIP tooling is committed on `25fps-training-lip-network` as an explicitly
  unreviewed WIP commit.
