# Round 2: trained bk_tiny — FAIL

**The student fails all 12 held-out clips in both fp32 and fp16; the full fp16 control passes r5 on 12/12.**
Every student clip also fails the no-worse comparison against today's phone chain D. More training slightly
improves validation and some average video metrics, but does not recover acceptable lip motion.

## Controlled second attempt

The only changed training setting is the step cap: **30,000 → 100,000**. Code remains
`d8826442c5d1bd8772185a7fc564eddd98faf19f`; bk_tiny, teacher weights, seed, data hashes, train/validation split,
batch 8, gradient checkpointing, LR 1e-4, 500-step warmup, feature weight 1, evaluation every 250 steps and the
150-minute limit are unchanged. This is a fresh run from the same teacher initialization, not an extension of
round 1's checkpoint. [experiment_comparison.json](experiment_comparison.json) verifies the manifest comparison.
The higher cap lets the existing wall-time cosine schedule determine the stopping point.

Both runs use ten training characters (2,400 frames), two validation characters (256 fixed selection pairs)
and nine training audio clips (3,256 windows). All 14 gate identities and six review identities, including
their other poses, remain excluded from training and selection. The two review audios remain held out.
See [data_manifest.json](data_manifest.json). Both runs use an RTX 3090, torch 2.5.1+cu121 and diffusers 0.30.2.

| Training result | Round 1 | Round 2 |
|---|---:|---:|
| Actual last step | 30,000 | 37,807 |
| Best step, selected only by validation | 28,250 | 35,750 |
| Best validation MAE | 0.07585432 | **0.07568650** |
| Gate MAE at that checkpoint, informational | 0.09673169 | 0.09698398 |
| Gate max absolute error at that checkpoint | 4.85269 | 5.01862 |
| Last validation MAE | 0.07745910 | 0.07680055 |
| Last gate MAE | 0.09764411 | 0.09712794 |
| Evaluations, including step 0 | 121 | 153 |

The round-2 update loop stopped at the **150.0-minute budget**; the final evaluation records 150.0421 minutes,
and the final console line rounds to **150.1 minutes** after evaluation and saving. The timer includes periodic
evaluations and checkpoints. Round 1's final console line was 116.7 minutes. The selected round-2 checkpoint
was evaluated at 141.8205 minutes. Every evaluation reports the 128-frame gate score in
[train/log.jsonl](train/log.jsonl); [train/train.log](train/train.log) includes the stopping record.

Round 2's best validation MAE is about 0.22% lower than round 1's, while its corresponding gate MAE is slightly
higher. **Round 2 is selected for the phone speed benchmark solely because validation MAE is lower.** Neither
gate scores nor review clips were used to choose that checkpoint or the phone candidate.

Best checkpoint SHA256: `a05d2d98f69b431252458d3b25f72e395d557ba693821baee91d11f5fe5a96cd`.
The untracked local files are `models/ios_bench/distilled/round2_bk_tiny_{best,last}.pt` and
`models/ios_bench/distilled/round2_bk_tiny.json`.

## Held-out video review

Six avatars × two clips, 250 frames per clip at 25 fps, with the same 10-second protocol and regions as D.
A is the full fp32 reference with TF32 off; N is full fp16; S and S16 are the student in fp32 and fp16.
**N passes r5 on 12/12; S and S16 pass on 0/12.** Both student arms fail the D comparison on 12/12 as well.

The r5 gates require aperture correlation ≥0.97, zero lag, mean aperture error ≤0.5 px, mouth/face flicker
ratio ≤1.05 and mouth sharpness ratio ≥0.95. Every round-2 student clip fails aperture correlation and aperture
error; several fail sharpness, and the Japanese clips lag by 1–2 frames. Rejection therefore does not depend
on the additional zero-margin D comparisons. N's D comparisons are informational: even the near-identical
control can miss a ratio bound where D is slightly below the reference's 1.0.

| Student fp32 metric across 12 clips | Round 1 range (mean) | Round 2 range (mean) |
|---|---:|---:|
| Lip aperture correlation | 0.53769–0.96255 (0.85306) | 0.54558–0.95953 (0.85832) |
| Mean lip aperture error, px | 1.34410–3.29190 (2.00797) | 1.38090–3.28210 (1.96773) |
| Mouth PSNR, dB | 26.62–31.35 (28.70417) | 26.67–31.27 (28.82833) |
| Mouth sharpness ratio | 0.75420–1.04270 (0.92087) | 0.76160–1.04030 (0.90911) |
| Mouth error flicker | 2.36890–4.51680 (3.42081) | 2.36280–4.44890 (3.39898) |

The changes are mixed: average aperture error and PSNR improve slightly, while average sharpness worsens.
S16 reaches similar numbers and the same rejection. Full per-clip metrics, r5 failures and D comparisons are
in [review/summary.json](review/summary.json), [review/summary.md](review/summary.md) and each clip's metrics file.

## Selected comparisons

- [Maya / eng](review/en_01_maya_abe451d3_talking_f6cf8be7fb__eng/en_01_maya_abe451d3_talking_f6cf8be7fb__eng_AS.mp4)
- [Marcus / ai-assistant](review/en_03_marcus_19eb1ea8_talking_ada35a4a4d__ai-assistant/en_03_marcus_19eb1ea8_talking_ada35a4a4d__ai-assistant_AS.mp4)
- [Japanese / ai-assistant](review/japanese_realtime_talking_7d94520b7f__ai-assistant/japanese_realtime_talking_7d94520b7f__ai-assistant_AS.mp4)
- [Mateo / ai-assistant](review/es_03_mateo_884600e1_talking_75d88cda6c__ai-assistant/es_03_mateo_884600e1_talking_75d88cda6c__ai-assistant_AS.mp4)

These A | S presentation videos use H.264 CRF 26 compression at unchanged resolution, frame count and timing.
Numerical metrics were computed from the original raw frames before compression. The complete source archive
remains local at `.runtime/mtboot-distill/dt-r2-20261104a/results.tar`;
[curation_manifest.json](curation_manifest.json) records source/delivery hashes and the unchanged checkpoint hash.

## Phone inference: 25-fps generation achieved on the test clip; quality FAIL

The selected trained checkpoint was exported to plain Core ML with int8 weights and tested locally on the
physical **iPhone 12 Pro (iPhone13,3), iOS 26.2**, after both GPU rentals were destroyed. D is today's folded
V2 two-chunk UNet; S is `unet_bk_tiny_r2_selected_int8w_b1`. Both use two inference workers and zero GPU workers.
Each workload runs **D S S D**, with the same Latina avatar and 17.76-second waveform (284,160 samples at
16 kHz): 426 frames at 24 fps, 444 frames at 25 fps. [Execution provenance](phone/execution.json) and
[student provenance](phone/student_provenance.json) identify the checkpoint and workload.

| Workload / model | Mean generated fps | Mean shown fps | Generated frames per run | Skipped per run |
|---|---:|---:|---:|---:|
| 24 fps / D | 18.20493 | 18.12025 | 322, 323 of 426 | 104, 103 |
| 24 fps / S | **24.04221** | **23.92933** | **426, 426 of 426** | **0, 0** |
| 25 fps / D | 18.45007 | 18.39365 | 315, 339 of 444 | 129, 105 |
| 25 fps / S | **25.04537** | **24.96075** | **444, 444 of 444** | **0, 0** |

The trained student improves generated throughput by **32.06% at 24 fps** and **35.75% at 25 fps** against
these matched D runs. It generates every frame in both workloads, with zero skips, late drops or stalls.
At 25 fps it shows 442 and 443 of the 444 generated frames, so the displayed average remains slightly below
25 fps. All D runs also have zero late drops/stalls; their missing frames are recorded as skipped.
Full accepted records are in [phone24/results.jsonl](phone/phone24/results.jsonl) and
[phone25/results.jsonl](phone/phone25/results.jsonl); [phone_comparison.json](phone/phone_comparison.json)
contains the recomputable aggregates.

The two student 24-fps runs stay thermally nominal; both 25-fps runs stay fair. The 24-fps D runs are nominal
and fair respectively. At 25 fps, the first D run changes nominal → fair; the accepted final D retry stays
nominal. Its first attempt emitted a result but was **rejected by the SoulX watchdog**, and is excluded from
aggregates. SoulX was closed under the user's explicit authorization; the accepted retry is `4_D_a2.log`.
Both the rejected log and accepted retry are retained, with the interruption in [the run summary](phone/phone25/summary.txt).

Initial single-worker UNet checks under `CPU_AND_NE` measure D → S at **53.19 → 33.18 ms** (24-fps session)
and **54.33 → 33.31 ms** (25-fps session), consistent with the healthy Neural Engine path. Compute-plan
inspection was disabled, so these checks do not establish that every operation ran on the Neural Engine.
Available phone storage at the checks was **26,695 MB** and **25,594 MB**; storage did not block the test.

This is a short-clip throughput result, not a prolonged hot-session guarantee. The 25-fps run regenerates audio
windows at 25 fps; bundled reference/check artifacts remain from 24 fps and are **not 25-fps quality evidence**.
Quality is determined by the separate 12-clip review above, which fails. The usable-25-fps verdict is therefore
**FAIL**, despite the generation-speed improvement. See [the speed-only protocol note](phone/phone25_speed_only.md).
The Mac's original avatar JSON is restored byte for byte to 24 fps; the last installed benchmark app retains
the 25-fps workload. Existing model variants remain available. Lingua is unchanged. The tooling and results
are submitted for PR review; the rejected student has not been integrated and no merge has been performed.

## Rental closed; no further rental

The final permitted rental was **59067440**, label `mtboot-dt-r2-20261104a`: one RTX 3090, stock launch,
worker revision 4e2b02e, $0.2747743056/h and a pre-create four-hour expiry timer. Create requests: **1**.
Results and weights were pulled before cleanup. The receipt verifies **provider absence** and an
**inactive expiry timer** after cleanup; see [rental_receipt.json](rental_receipt.json).

Estimated round-2 spend is **$0.9597** (`estimated_usd=0.9597327527075665`); the combined round-1 + round-2
estimate is **$1.7227** (`1.7226798452716725`), below the $10 limit. These are time-plus-transfer estimates,
not invoice-confirmed charges. The receipt uses the higher ledger hourly quote through verified provider
absence and container byte counters at the quoted transfer rates; those counters can omit host/image traffic.
Both rentals are destroyed and verified gone. **No third rental is authorized.**
The [final combined verification](cleanup_verification.json) rechecks provider absence and inactive timers
for both instance IDs after the phone test. The two report folders together contain about 25 MB, including
eight compressed side-by-side videos; weights and complete source archives remain local and untracked.
