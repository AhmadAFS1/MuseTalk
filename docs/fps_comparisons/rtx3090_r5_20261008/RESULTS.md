# RTX 3090 r5 execution — in progress

This is a live evidence index, not a completed release claim. The protected RTX
4070 SUPER worker and production EC2 service have not been replaced or restarted.

## Current measured outcomes

| Measurement | Current result |
|---|---|
| GPU-path FPS | Native v1: 385.75668080919866 / 388.64419629013895; earlier portable: 278.288586320095 / 277.4908139113453. Separate ≥180 s windows; diagnostic, not full-recipe FPS |
| Full-recipe aggregate FPS | Latest A2 native overlap T: 356.59394751135187 / 355.3897959226974 over 96.917 / 97.245 s; valid FAIL against unchanged 400. Earlier A1 native T: 252.59611538153948 / 230.39816210925395; SUST: 231.089–246.552. Recovered portable control T: 253.59753865499397 / 240.87736112161846; original failures retained |
| Native quality | Original v1 rejected: 236 / 698 frozen bounds fail; TAESD max 7 LSB versus reference 5. Separate typed-v3 decoder also rejected: max 6 LSB and 36 / 88 TAESD bounds fail; not a new full-698 assessment. No bounds widened |
| Live delivery / capacity | Both software-H264 profiles N1/N3 strictPASS/N5 validFAIL. NVENC N1FAIL,0hardware opens/18open failures; PyAV/FFmpeg standalone probes alsoFAIL. No accepted capacity or EC2 release claim |
| Request-to-usable-call startup | Still unmeasured through EC2/TURN. Request-to-verified-health was 1,046.45 s (17m26s), not usable-call readiness |

### 00:28 UTC explicit gate/capture routing implemented, still not GPU-tested

The additive [single-child launcher](../../../scripts/repro_3090/taesd_fp32_candidate_child.py)
and [quality adapter](../../../scripts/repro_3090/taesd_fp32_candidate_quality.py)
now route the isolated candidate through the unchanged full-corpus gate and
six-avatar capture. The parent preserves original UNet/metric children and
guard/watch execution. Deferred decoder import preserves CPU-worker-first
spawn. Separate receipts require actual successful candidate invocations and
matching manifest/key/plan hashes; original strict failures stay failures.

Root and peer review caught and fixed abbreviated-flag and launcher
check-to-execution gaps. Exact quality flags are required, and each launcher
executes the exact bytes whose SHA is checked at execution time. Real CPU spawn
tests also verify that workers do not import GPU code or re-import unchecked
launcher source. Focused child22/parent19 tests passed; the
[final operator CPU run](harnesses/candidate_routing_cpu_0031.json) passed246/257
with eleven explicit dependency skips. The exact `2d8623e` revision subsequently
[passed Linux CI](harnesses/candidate_routing_linux_ci_0034.json) at 00:33:49 UTC:
246 passes, eleven explicit dependency skips, and zero failures among257 tests.
CPU contracts and mock GPU calls do not establish actual precision, quality or FPS.
No original decoder/gate/default/698-bound changes occurred.

The [metadata recovery check](harnesses/original_metadata_recovery_check_0016.json)
found zero exact matches among the twelve original download-metadata files on
the protected 4070 (ten differ, two are absent). Known off-host archive inventories
do not prove an exact A1 metadata copy. This is not proof changed weights, nor
global unrecoverability, and no original878-input PASS is claimed. Exact
restoration or an explicit preregistered successor comparison lineage remains
necessary; preserve every original quality bound and actual payload/source hash.

The [corrected fresh offer search](provisioning/a3_offer_discovery_0015.json)
found five eligible advertised 3090 offers without unnecessary 350W/12CPU/32GB
hard filters. Best advertised fit was Florida machine143947, $0.294444/hour
including150GB storage, $0 transfer, 2.1Gbps download. This stale receipt must
not be used to purchase; actual current power, guest resources and speed remain
unmeasured. [Read-only EC2 refresh](provisioning/ec2_resume_preflight_0022.json)
found production revision970fa989 and no owned GPU experiments, with unchanged
$12.7175 reservations—not billed spending. No production mutation or A3 rental
occurred. New local expiry checks bind the timer deadline before credentials or
provider calls; they are not yet deployed.

### 00:05 UTC isolated decoder candidate APIs prepared, not GPU-tested

The separate [FP32 candidate runtime](../../../scripts/repro_3090/taesd_fp32_candidate_runtime.py)
and [CPU review](harnesses/taesd_fp32_candidate_runtime_cpu_0005.json) are prepared:
24 mock/stdlib contracts passed and one real-ONNX dependency test explicitly
skipped locally. Root and independent peer review found no concrete blocker at
the recorded hashes. Imports are stdlib-only/default-off. The builder requires
fresh artifacts, strongly typed batch8/opt3/native-none configuration and TF32
clear/readback; the loader has explicit manifest SHA, regenerated graph/proof,
plan hashes and exact double probes, without automatic build or fallback.
Verified source bytes execute directly without a second read or bytecode cache.

Canonical decoder/gate/defaults remain byte-identical. No actual original graph
has been transformed, engine built, FP32 execution observed, quality gate run,
or FPS measured for this candidate. The explicit scoped gate/capture launcher
was implemented in the later section above; fresh owned GPU preflight and actual
graph/build/quality/performance work are still needed.

The exact candidate commit `2cdfe99` [passed Linux CI](harnesses/taesd_fp32_candidate_linux_ci_0008.json)
at 00:07:57 UTC: 190 tests, 179 passes, eleven explicit dependency skips, zero
failures/errors. The new runtime suite passed 24 with one real-ONNX skip. This
validates mocked CPU contracts, not actual TensorRT precision or quality.

### 23:46 UTC A2 expiry verified; no rented test GPU remains

The [scheduled-expiry observation](provisioning/a2_expiry_observation_2346.json)
confirms non-forced destruction of the exact owned instance `54909897` at
23:45:02.396453 UTC. The expiry service finished successfully, and an independent
provider GET at 23:45:31.937145 UTC found that ID absent. Cleared transient units
alone were not treated as completion proof. The protected 4070 was not targeted.
Do not reconnect to the retired A2 SSH alias or reuse its resource descriptor.

All five experiment archives were verified in private storage before expiry;
the experiment results remain recoverable without retaining the rental. The
shared reservation remains $12.717528 of the $30 cap, not finalized billing.
The final selected eleven-suite operator CPU run reports 164 tests: 157 passed,
seven real-ONNX tests explicitly skipped because ONNX is absent locally, and
zero failures. Those fourteen final-Conv tests separately passed with actual
ONNX on A2.
Quality, 400 FPS, publication and startup acceptance remain unmet.

The exact checkpoint's [Linux CI failed](harnesses/a2_candidate_linux_ci_failure_2355.json):
144 passed, three NumPy skips and three preservation-test errors; the final
TAESD suite was not reached. The synthetic tests omitted the fixed operator-root
mock, so the production Mac-only safety guard correctly refused the Linux path.
Only test fixtures were repaired. All thirteen preservation tests now pass both
normally and with an independently simulated nonoperator/Linux root, including
new rejection-before-payload/cloud coverage. The production helper is byte-identical;
the new Linux result is pending, not inferred from local passes.

The corrected exact commit `e6d38eaf` subsequently [passed Linux CI](harnesses/a2_candidate_linux_ci_success_2357.json)
at 23:57:33 UTC: 165 tests, 155 passes, ten explicit optional-dependency skips,
zero failures/errors. All thirteen preservation tests passed. NumPy/ONNX skips
are not numerical execution evidence; actual A2 ONNX CPU validation is separate.
No Docker build, GPU, publication, or resource mutation occurred in this CI run.

### 23:39 UTC geometry-controlled preparation repeats exactly

The separately named [geometry-v2 smoke](avatars/a2_latent_geometry_2335/comparison.json)
ran 23:36:07.756870–23:38:35.892049 UTC and finished
**PASS_REPEAT_EXACT_DIAGNOSTIC_ONLY**, both children exit 0. Both runs have
exactly 240 successful pre-DWPose flag receipts and the original post-detector
reset. All latent/audio/box/cropbox values and all 240 masks match exactly.
The native FP16 encoder, seed123 placement, source, math and precision are unchanged.

Audio and both geometries also match the historical cache exactly. Historical
latents still differ (max 0.04974365234375, mean 0.000185378117748769), and 41 mask
arrays differ: this is a distinct new preparation candidate, not a latents-only
change or a quality/FPS acceptance. Do not replace frozen inputs or original caches.
All-six expansion and rendered-output validation remain required.

All five private archives have passed conditional upload, version-bound fresh
download and every payload SHA check. The [geometry-v2 proof](release/a2_latent_geometry_private_persistence_2342.json)
finished 23:43:14 UTC: 79,108,045 bytes and all 24 payload hashes verified, before
the unchanged 23:45 UTC expiry. All 12 small local geometry files also match their
remote SHA-256 values. Preservation never upgrades
the experiments' original outcomes. The earlier ten-suite CPU run passed all 150
tests on operator Python3.12.14/NumPy2.3.5, and geometry-v2's 25 tests passed on A2.
The expanded eleven-suite result and actual expiry are recorded above.

An isolated final-convolution FP32-island ONNX proposal has 14 actual CPU tests
passing on A2, including real ONNX checker/shape-inference and mutation proofs.
It has not transformed the actual TAESD graph, built an engine, or run a quality
gate; canonical loaders/defaults are unchanged. A reviewed strongly typed builder
with TF32 disabled and the full unchanged quality gate are still needed.

### 23:30 UTC fixed-cuDNN repeat failed; host refused power changes

The [fixed-cuDNN one-avatar smoke](avatars/a2_latent_fixed_cudnn_2318/comparison.json)
ran 23:25:48–23:28:09 UTC and finished **FAIL_REPEAT_CHANGED**, with both children
exiting 0. Audio now matches exactly across repeats and the historical tensor
(`0784da8560b3a51dd26eb0c37739c96cfb2dcd68f139592f83389849c4c2ecf3`).
This is improved audio repeatability, not an all-component pass: one face-box
value and three cropbox values differ by 2 px; 7,809 latent values differ
(max 2.48291015625, mean 0.0005282640837322106). The only changed mask is frame123,
whose shape is 530×530 versus 528×528. Both runtime receipts show the intended
post-detector `benchmark=True → False` reset. Historical versus repeat1 still
changes geometry and 43 masks; it is not a latents-only experiment. No all-six
expansion, quality, FPS or startup acceptance follows.

All ten small plan/comparison/runtime/preparation/log/watch files were copied
read-only and their SHA-256 values matched remote readback (353,670 bytes total).
No cache tensors, NPZs, PNGs or media were copied in this update.

The [bounded 350 W power attempt](native/a2_power350_2320_power/power-receipt.json)
is **INVALID: power_set_refused** at 23:30:20 UTC. The 350 W setter returned 4,
so the canonical benchmark never started and no 350 W FPS result exists. The
300 W restore setter also returned 4: `restore.verified=false` remains intact.
Readback in that receipt nevertheless shows the exact GPU still at 300 W,
46°C, 0% utilization and 1 MiB used. This is unchanged-limit readback, not a
successful restoration operation. The 5,850-byte receipt was copied with matching
remote/local SHA-256; no permission bypass or new power attempt was made here.

Local source inspection identifies a narrower remaining geometry hypothesis:
S3FD sets benchmarking true between DWPose calls, while the v1 reset occurs only
after all 240 detector iterations. DWPose landmarks are truncated to integers
before box construction; a small landmark boundary change can affect geometry.
Neither smoke log reports a fallback detector box. A distinct default-off v2
adapter now additionally resets benchmarking immediately before each original
`inference_topdown` call and requires 240 successful flag receipts. Its 25 CPU
tests pass; root review/GPU evaluation remains separate. No canonical math,
model, seed placement or precision setting changed. Its actual repeat result is
recorded above. Preserve failed evidence and
honor 23:45 UTC expiry; do not infer 400 FPS or quality acceptance.

### Typed-v3 decoder rejection and lightweight workflow

The [typed-v3 assessment](native/a2_taesd_typed_v3/assessment.json) is terminal
**FAIL**. The [gate](native/a2_taesd_typed_v3/gate/gate_taesd_trt.json) evaluated
448 captures / 3,584 frames: full-frame max/mean error 6 LSB /
0.06763140644345965, and 104-row max/mean 3 / 0.0631663267475023. Fused-post and
rerun mismatched-byte counts are both zero. Exact reruns do not imply accuracy:
36/88 frozen TAESD bounds fail, versus 32/88 for the earlier rejected native
decoder. Only those 88 statistics were assessed; UNet and the complete 698-bound
quality comparison were not rerun. Original failures and quality bars remain.

The expanded lightweight CPU workflow passes 145 tests locally on Python3.12.14
with NumPy. The power wrapper's 19 tests, credential bridge's 22 tests and
preservation helper's 12 tests pass independently. Their hardware/cloud actions
are mocked; this is not a power experiment, private upload or new Linux CI result.
The workflow adds only test paths/invocations, with no dependency downloads,
Docker build, credentials or remote resource actions.

### 23:05 UTC canonical preparation repeat failed; no six-avatar expansion

The [one-avatar seed123 smoke](avatars/a2_latent_smoke_2303/comparison.json)
finished **FAIL_REPEAT_CHANGED** at 23:04:53 UTC. Both canonical preparation
children exited 0 with identical recorded host, GPU and runtime versions.
Their image latents, face boxes and cropboxes are exact, but audio conditioning
differs (max 0.03125, mean 0.00030931543439833653), and 26/240 mask arrays differ
(maximum 6 intensity levels). The second audio tensor's hash equals the original
historical tensor. This does not establish why the first differs or make it safe
to ignore the repeat failure.

Historical versus new preparation is also not a latents-only change: latent
max/mean differences are 2.48291015625 / 0.0007145401040967651; one face-box value
and three cropbox values change by 2 px. Forty-two masks differ, including one
shape difference. Existing caches and frozen quality bars remain untouched.
All-six preparation is held until repeatability is understood; no visual,
quality, throughput or startup improvement is inferred.

The small plan/comparison/runtime/preparation/log/watch evidence is retained
under [the smoke directory](avatars/a2_latent_smoke_2303/plan.json): 10 files,
353,471 bytes, remote/local SHA-256 matched; no `.pt`, `.npz`, PNG or media copied.
The separate typed-decoder diagnostic started at 23:05 UTC and is now terminal
FAIL, as recorded above. It is not a running session to resume.

At this earlier checkpoint, lightweight workflow coverage passed 103 CPU tests on Python
3.12.14 with NumPy, including all 15 latent-preparation tests. The stdlib-only
latent test invocation reports 12 passes and 3 explicit NumPy skips. The workflow
included this test file without dependency/model downloads or Docker work.
The expanded local result is above; the new Linux CI outcome is not yet observed.

### 23:03 UTC actual A2 scheduler parity and aggregate result

The [2202 serial/overlap pair](native/a2_tracking_pair_2202_tracking-parity/report.json)
finished **PASS** at 22:03:56 UTC: all six identities' generated face pixels, raw
refined frames, generated landmarks and chin deltas match exactly between the
two scheduler modes. This is same-engine scheduler parity only, not quality
parity with the frozen 4070/portable references or release acceptance.

The subsequent [2205 overlap aggregate](native/a2_overlap_T_2205_aggregate/report.json)
finished **valid FAIL** at 22:11:19 UTC against 400 FPS. Each repeat completed 34,560
full-recipe frames: 356.59394751135187 FPS over 96.91695622203406 s, then
355.3897959226974 FPS over 97.24533567507751 s. Both exceed the unchanged 60 s minimum;
neither reaches 400. [Detailed T telemetry](native/a2_overlap_T_2205_aggregate/a2_overlap_T_2205_T.json)
records median GPU event-busy fraction 0.9959730218492162 (about 99.6%), GPU power
medians 299.46/299.44 W at a 300 W limit, and 79°C median temperature in both repeats.
The six-stream, full-height, full-strength refined-chin workload excludes live
encoding and RTP. Cross-host comparison with older A1 numbers is not a controlled
measurement of scheduler speedup. Native-v1's 236/698 frozen quality-bound failures
and TAESD 7 LSB versus reference 5 remain unchanged; no threshold was widened.

Only the aggregate report, detailed T report and [environment](native/a2_overlap_T_2205_aggregate/environment.json)
were retrieved from owned A2 for this update (three JSON files, 185,717 bytes total); remote/local
SHA-256 values match. No media or GPU work was copied or run by this documentation
update. At the 23:03 UTC checkpoint, the separate seed123 canonical latent
preparation was running; its terminal repeat failure is recorded above.

### 22:02 UTC new3090 preparation, explicit lineage and updated priority

The updated user priority is native3090 full-recipe400+FPS at the highest passing
4070-level quality, controlled new latents following that preparation path, then
the complete downloadable Docker and approximately60-second create-to-health.
Health and EC2-routed usable-call timing remain separate; neither has a new SLA
pass. The full prior quality/FPS requirements have not been lowered.

[A2 preparation](startup/a2_health_preparation_2202.json): owned54909897 reached
portable TensorRT backend/health verification21:03:58UTC but was unregistered,
with0cachedavatars. Its API/TURN are stopped and GPU isolated. Canonical FaceMesh
installation,6fixture300files,39audiofiles and449calibrationfiles restored PASS.
[All16native payload hashes](native/a2_native_restore_integrity_2153.json)
match the privately persisted rejected-v1 archive; this is not GPU acceptance.

The [original878input check](harnesses/a2_original_frozen_inputs_audit_2154.json)
failed honestly: 862 matched, 11 fresh HF metadata files and 1 intentionally changed
worker differed, and 4 SyncNet paths were absent (16 failures total). SyncNet was subsequently restored
with its original payload SHA. A strict [successor lineage](harnesses/tracking-a2-lineage-v1.json)
keeps all 878 paths and original runtime bytes, allowing 12 validated cache metadata
changes plus 1 exact reviewed worker revision (13 changes total). No old manifest or 698
quality bound was rewritten; native-v1 remains rejected. The first pair passed
runtime/input preflight then stopped INVALID before rendering because a fixed
diagnostic stage was not allowlisted. That narrow bug now has an actual CPU
subprocess regression test;14parity tests pass on both operator and Linux3.10.
The fresh2202pair subsequently passed same-engine scheduler parity; the actual
overlap T result above remains below400FPS and is not native quality approval.

[Bounded-bridge CI](native/bounded_credentials_linux_ci_2106.json) is terminal
success with69Linux CPU tests and an actual harmless owned-worker command. The
existing [a57dependency build](release/dependency_ci_a57f2de/assessment.json) is
also terminal success; all11reports were fetched with the matching archive SHA.
It built only `Dockerfile.dependencies`,12,335,002,076uncompressed bytes, with
unchanged apt/pip pins. No full serving image/publication/GPU/startup claim.
Budget remains12.7175reserved/30, not final billing; ownedA2 expiry23:45UTC.

### 20:20 UTC exact scheduler gate and verified dependency CI

The new `25_tracking_parity.sh` runs two isolated, same-engine six-avatar
captures, serial then overlap, under the existing preflight/GPU watchdog.
It checks GPU UUID before/after each child, rechecks frozen inputs, and validates
actual face pixels, finite FP32landmarks/FP64chin arrays and completed raw refined
hashes. The long overlap aggregate suite now requires a successful pair receipt
and recomputes its evidence, bound to this GPU, engine/decoder/input/profile and
current harness. Output differences remain FAIL; missing, corrupt, wrong-mode,
changed-source or stale evidence is INVALID. It does not approve a rejected
engine or change the400FPS/quality bars.

[CPU gate evidence](native/tracking_parity_gate_cpu_2020.json):13synthetic tests
pass, plus3strict archive tests,78existing harness tests and9overlap tests. The
new parser also read all6existing native capture array files with their actual
canonical shapes and finite payloads. That is parsing integration, not an actual
overlap comparison or quality acceptance. Real GPU/parity results remain pending.

[CI37830969052](https://github.com/AhmadAFS1/MuseTalk/actions/runs/37830969052)
finished success at20:10:24UTC. All11small reports/logs were fetched with the exact
28002-byte ZIP SHA matching GitHub's declared digest. [Provenance](release/dependency_ci_7a78e4a/provenance.json)
and [assessment](release/dependency_ci_7a78e4a/assessment.json) retain scope:
128installer checks passed,0failed, with wrong-interpreter/live-venv checks
explicitly skipped. Sequential/parallel SyncNet opt-out/opt-in and required
DWPose/S3FD tests pass. The dependency-only image is12,334,998,487uncompressed
bytes, just7777above the preceding build, with identical apt pins and pip freeze.
No full serving image, GPU, publication, compressed pull or startup saving follows.
The already queueda57f2de dependency build is now running; it was not restarted.

### 19:54 UTC default-off tracking overlap and quota check

The scheduling experiment is now implemented behind `--tracking-overlap`, never
selected by default. It calls the unchanged canonical tracker on one helper
thread, with exactly one outstanding call/shared-memory writer, while the main
worker composes the preceding frame. Frame order, copied landmarks, three-tap
chin filtering, reset boundaries and all GPU/quality/FPS gates are retained.
Ignored or mixed worker modes fail report validation. Tracking call service time
and blocking wait are separate; overlapping durations are not summed.

[CPU evidence](native/tracking_overlap_cpu_1954.json): nine overlap tests pass,
including the actual worker loop with synthetic inputs over two clips, exact
synthetic refined/face hashes and filter state, event-proven concurrency, and
timeout cleanup of a real owned subprocess pipe. Existing78harness tests pass;
76Docker tests are OK with1Linux-only skip. These operator tests ran on Python
3.14.7, not production3.10. A separate lightweight Linux3.10 CI is prepared with
no model/dependency downloads or GPU; it does not restart the long Docker build.
Actual FaceMesh/pixel/landmark parity and any speedup remain unmeasured. Native
quality rejection and every400FPS failure remain unchanged; no default, template,
production service or paid GPU changed.

Subsequently [Linux3.10 CI37835666432](https://github.com/AhmadAFS1/MuseTalk/actions/runs/37835666432)
completed **success** on exact7699542f44fd146c20e9b09b6a734792124ade1f at19:57:22UTC.
Run/job/log readback confirms all9overlap tests and28existing report/harness tests
are OK, including the real owned-pipe timeout test and399.96FPS rejection.
[Linux evidence](native/tracking_overlap_linux_ci_1959.json) is CPU-only; it does
not establish real FaceMesh or native pixel/landmark parity or a GPU speedup.

Account allowance checked19:45:36UTC: **10% used /90% remaining** in the shared
weekly window, ordinary usage allowed, purchased credits0, one unused free full
reset. It resetsOctober15at11:17:04UTC/06:17:04Chicago. Exact subscription tokens,
model-specific missing limits and full-task consumption are unavailable. Quota
is not currently the blocker, but this is no completion guarantee; bounded
experiments and milestone checks remain the policy. No reset/purchase was made.

### 19:35 UTC CPU implementation and throughput diagnosis

The full Dockerfile now supports an explicitly manifest-selected fresh final
stage. Its optional smaller CUDA/cuDNN base is restricted to the exact pair used
by the dependency experiment; omission retains the original development base.
Both stages keep all required apt pins and the complete `/opt/musetalk` payload,
and the final stage repeats full source/model/import/MMCV checks. Runtime-base
overrides fail before model staging. No TensorRT resource is pruned from the
full image. Local76Docker contract tests pass with1Linux-only skip. This is code
preparation, not an actual full-image build, GPU pass, publication, compressed
transfer size or measured cold-start saving.

[Aggregate scheduling diagnosis](native/aggregate_occupancy_1935.json) binds the
complete nativeT2/SUST5 reports and inspected source hashes. Fixed16-row UNet
calls had only63.6–69.6% useful row occupancy; partial8-row jobs were padded, not
counted as valid completed frames. CPU workers were nearly never idle, and
tracking/composition ran serially. FaceMesh service time is contained in tracking
IPC time and must not be added to it. These observations support testing bounded
tracking/composition overlap with original frame order/filter/output parity;
they do not prove a speedup, sole cause or accepted400FPS capacity. Two focused
CPU accounting tests pass, including rejection of padding/frame/denominator
inconsistency. Native numerical rejection and every400FPS failure remain unchanged.

CI37830969052 at exact7a78e4a has passed Linux installer/startup contracts and
is still executing the dependency-only build. It cannot validate this subsequent
full-Dockerfile change. No paid GPU is retained and no production/template/default
selection has changed.

### Latest isolated diagnostics, 18:43 UTC

At19:00:03UTC the owned development rental54798270 was destroyed non-forced,
and provider absence was verified. [Exact cleanup receipt](provisioning/a1_cleanup_1900.json)
was read back from EC2 and the systemd journal. The protected4070 was not targeted;
no paid GPU is currently retained. Quota at19:02UTC is8%used/92%remaining weekly,
credits0 and one unused reset. Independent CPU-only contract/release work remains;
the full quality/performance/image/fresh-instance goal is not complete.

[Full-file tracing](startup/fullfile_model_access_1843_v2.json) validates the known
UNet positive-control open and observes startup, a canonical human-WAV stream,
and a fresh canonical preparation. The original preparation client expected the
wrong metadata spelling and failed; separate read-only verification confirms its
completed480-frame/480-mask cache without repeating preparation. No SyncNet
checkpoint path access was observed. This is scoped dependency evidence, not a
missing-model, TTS, installer or cold-start pass. [Private raw trace preservation](release/private_fullfile_trace_v2_1825.json)
passes exact-version freshGET and all8 payload hashes (128MB raw trace, 5.28MB archive).

[Same-precision TAESDopt5](native/taesd_opt5_v2_assessment_1842.json) is rejected:
all448 captures/3584 frames were evaluated; full-frame maximum remains7LSB versus
portable reference5. The decoder plan changes, post plan is unchanged, and exactness
checks pass. No speedup was measured or default changed. [Private plan/evidence preservation](release/private_taesd_opt5_v2_1838.json)
passes conditionalPUT, freshGET and all8 payload hashes.

[Actual canonical still inspection](quality/native_v1_canonical_partial_visual_review_1828.json)
covers18 native-resolution assets acrossall6 identities. No obvious large new
geometry defect is apparent in these samples, but numerical rejection remains;
motion/audio/silence/transitions and full production48 visual validation are incomplete.

Quota at18:16UTC:7%used/93%remaining weekly, purchased credits0, one unused free
reset. Account percentages are not an exact remaining token balance or a completion
guarantee. No reset, purchase or new agent was used. The owned19:00UTC expiry timer
was read back active at18:42UTC and18:46UTC. [Missing-SyncNet capability v2](startup/syncnet_dependency_assessment_1855.json)
now passes direct canonical startup, human-WAV speech and fresh480-frame/480-mask
preparation. The checkpoint was restored with its original SHA and the API stopped;
0OOM/foreign GPU was observed. This supports a separately tested training-only
download contract, which is not yet implemented, not a measured boot saving.
The subsequent installer change is now implemented locally, pending actual Linux
regression CI: normal API installs (including avatar preparation) omit SyncNet;
explicit training-checkpoint opt-in and the standalone downloader's historical
defaults remain supported. Full API model checks still require13assets, including
DWPose/S3FD and TAESD. Local76unittest cases are OK with1Linux-only skip; Bash
syntax and diff checks pass. New Linux tests cover sequential/parallel downloads,
missing opted-in SyncNet failures, and missing preparation-model failures. No
cold-boot saving is measured or accepted, and license/release gates are unchanged.
The change is pushed at`7a78e4abf09ffcc640e2979d5b08887fa70bdd4f`;
[exact-revision Linux CI](https://github.com/AhmadAFS1/MuseTalk/actions/runs/37830969052)
has now passed the CPU contracts and Linux startup/installer regression step;
the dependency-only Docker build remains in progress. Full job/artifact verification
is pending. The19:18UTC account read remains9%weekly used/91%remaining, with
normal usage available, credits0, and1unused reset. The weekly reset is
October15at11:17:04UTC (06:17:04Chicago). No remaining-token balance or guarantee
of whole-task completion is available; check major milestones, keep experiments bounded.

All1930 diagnostic-cache file hashes were inventoried;8core tensor/metadata files
were bound to that inventory. [Private evidence preservation](release/private_missing_syncnet_v2_1847.json)
passes conditionalPUT/freshGET/all16payload hashes. Complete diagnostic PNG payload
was not archived: a slow partial recursive copy was cancelled and retained locally;
canonical source and complete production48 review evidence have separate preservation.
The first attempt failed only because its harness nested two GPU leases; the checkpoint
was restored with its original SHA. No startup savings are accepted from these diagnostics.

Instance A is Vast **54798270**, label `musetalk-r5-3090-dev-20261008`, created once
through the existing EC2 API. Source revision is
`dae1e88ad3587fddbebe6d41f85faa569001f6c1`, with explicit control-plane registration
disabled for isolation. The production worker **51074906** remains protected.

Observed timeline (UTC, October 8):

- 07:15:25.547: EC2 create request started.
- 07:15:25.986: provider acceptance returned (0.439 s client-observed interval).
- 07:18:52: provider log endpoint still reported no container.
- 07:19:32: provider reported running; this is not application readiness.
- 07:19:36: source bootstrap began.
- 07:20:34: first successful operator SSH; hardware verified.
- 07:24:17: checkout revision verified; dependency installation still downloading TensorRT 10.3 libraries.
- 07:29:04: model download phase started.
- 07:31:21–07:32:33: portable engine bundle restore (72 s).
- 07:32:52: canonical startup completed after 631 s; final server health verification took 19 s.
- 07:36:29: one production talking-pose warm request began; response after 23.82 s, including 21.77 s S3 restore and 1.43 s cache load.
- 07:43: owned API/TURN drained and stopped before isolated GPU work.
- 07:56:50: two three-minute portable GPU-path runs completed; no foreign GPU workload observed.
- 08:05:08: portable full-recipe T completed; 34,560 frames per shared window of 134.64657760900445 / 133.51156995497877 seconds.
- 08:20:46 / 08:33:24: two portable quality runs completed. Exact source-prefix, batching, fused-post and repeat checks pass; original UNet max-error, TAESD max-LSB and landmark failures remain FAIL.
- 08:35:40.744: measured reference envelope frozen before native candidate evaluation.
- 08:38:54: native-build preflight failed before any engine build. NVML queries were observed temporarily blocked for about a minute; bounded runtime diagnosis is in progress, not a successful native build.
- 08:41:54: corrected bounded diagnostic reproduced CUDA driver initialization failure. No OOM; physical cause not established.
- 08:48: a 14GB repository backup completed on the rented host. Subsequent SSH routes became unreachable; planned recovery guard was not applied and no reboot was sent.
- 08:52:18: Vast reported machine153039/instance54798270 **offline**. Replacement offers were checked but no additional rental was purchased.
- 11:17:36: Vast again reported **running**; SSH and a CUDA tensor allocation succeeded on the original hostname and GPU UUID. Automatic startup had replaced the experiment checkout with the initial source revision and restarted API/TURN. The preserved experiment directory survived.
- 11:18–11:21: owned API/TURN drained/stopped; preserved checkout restored; builder dependencies restored on the same pinned matrix. A hostname-specific guard now suspends this development worker's automatic reinstall/autostart. No operator reboot was sent.
- 11:23: native preflight passed against the exact frozen 878-file manifest. Native sm86 v1 engine build started on revision `d4e78b79105e0c1f5b732fb651e5dcd7fcb5dca3`.
- 11:29: native FP16 `down0rest,up3,tail` build completed successfully after 324.1 seconds under its GPU lease. INT8 recipe blocks are building. Complete-chain, native TAESD, quality, and throughput remain unverified.
- 11:45: native INT8 `down1,down2,down3,mid` plan files exist; the builder remains active on subsequent blocks. This is partial build progress, not a complete or accepted engine set.
- 11:52:57: complete native UNet manifest finalized with all 11 blocks, graph/direct equality and repeat determinism. Probe output SHA is `23950ef8e4117aef4488a7e14e032450875c808837ef844151af30958b79908c`. Most exported ONNX hashes differ from the portable reference (tail matches); numerical parity is not inferred from successful compilation.
- 11:53:52: native TAESD completed, actual key `1e967e6e715c9f1a8375`, opt3, full-height bs8, hardware compatibility `none`; fused/repository post probe has zero mismatched bytes. Total observed native build interval was 1,823 seconds; this is development compilation, which must not recur at normal image boot.
- Native suite preflight then passed against all 878 frozen files, followed by sequential quality/envelope, paired GPU diagnostics and full-recipe T/SUST evaluation.
- Native v1 quality completed: original strict gates remain **FAIL**, and frozen-reference parity is **FAIL** with 236 of 698 bounds missed. TAESD full-frame max rose to **7 LSB** from portable **5 LSB**, so the candidate is rejected; no bounds were widened. Source-prefix and fused/repeat byte invariants pass. Actual native visual inspection and production-pose validation remain incomplete.
- Two native GPU-path runs completed with 69,440 / 69,968 valid frames over 180.00984416999927 / 180.0309915029993 seconds: **385.75668080919866 / 388.64419629013895 FPS**. Diagnostic validity passes, but this is below 400 and excludes the full composition/live workload.
- Native full-recipe T completed as valid **FAIL**: **252.59611538153948 / 230.39816210925395 FPS**. SUST5 completed as valid **FAIL**: **246.552153419512 / 231.08910512636766 / 233.40398117752895 / 231.26400993583042 / 231.96003877267862 FPS**. All windows contain 34,560 completed valid frames over their own shared wall times; no padding is counted.
- T had 1,888 / 2,470 partial jobs of 3,104 / 3,395 total; GPU credit waits consumed 11.8% / 14.2% of wall time. A software-thermal slowdown was separately observed during SUST. These are diagnostic signals, not proof of a single cause. Thread profiles differ only in hardware compatibility.
- Recovered portable control completed: 34,560 valid frames over 136.27892519499983 / 143.47550072400009 seconds, **253.59753865499397 / 240.87736112161846 FPS**, valid **FAIL** against 300. Native and portable full-pipeline ranges are close despite faster native GPU diagnostics. This is the same recovered host and frozen inputs, but not a fully interleaved full-pipeline experiment; elapsed thermal/host conditions remain confounders.
- Native serving-only diagnostic archive copied off-host, SHA256 `1f766487cf9272929988d17f9a4d6f76ee9c8c8fdde9b149174e1e02dbcafc5e`, 983,926,034 bytes. Fresh operator CPU restore verified all 16 payload files. It is explicitly rejected diagnostic evidence, not the active r5 bundle or a releasable artifact.
- The same archive was conditionally uploaded to a new private checksum-keyed S3 object, followed by an independent fresh GET and fresh CPU restore verifying its archive SHA and all 16 files. No existing object or active artifact mapping was overwritten.
- All 32 native quality capture/report files were separately archived, conditionally persisted in private S3, freshly downloaded and CPU-restore verified: SHA256 `079f79938d45d62dfe5316e84be984cd86d7c648ea324023bd34b55d87090bb9`, 274,657,275 bytes. Numerical rejection and partial visual-review limitations remain unchanged.
- Operator fallback conditionally uploaded all five private model objects after checking their exact frozen hashes. Worker-side fresh content verification then timed out at 300 seconds on its first file; delivery remains **incomplete**, not a boot-speed pass. The ongoing 48-pose audit had completed 21 objects without reported failures at the last checkpoint. Concurrent CPU/network work is disclosed; no single cause or steady-state transfer speed is established.
- At 14:19, version-specific private reads failed with HTTP 403 despite a successful ordinary HEAD. At 14:32, ordinary full/range prefixes and a four-thread whole-file download passed for the first pinned 89,843,225-byte object. Its complete SHA matched in 23.2537 seconds; before/after HEAD version was unchanged. These paired diagnostics establish different access behavior, not the cause of the earlier unversioned stream timeout or a cold-start speedup. The actual image fetch now supports bounded parallel reads with full SHA/size and stable-version checks, preserving explicit version requests without a permission fallback. A read-only all-five test is running; no IAM or bucket settings changed.
- The production-cache CPU audit had reached **45/48 PASS** without reported failures at 14:42; it remains incomplete until its terminal report. GPU/live timings remain isolated from this download/hash work.
- A second dependency-only Docker experiment uses a fresh final stage on the **same pinned devel base and apt versions**, with only the checksum-pinned 1,397,061,088-byte Windows TensorRT build resource eligible for removal. Linux resources remain. All 71 CPU contracts ran: 70 passed and one platform-specific check skipped. New actual Docker size/import evidence is still pending; neither a release image nor a public registry digest is claimed.
- The CPU cache audit completed **48/48 PASS** at 14:42:56 UTC. This is integrity/shape/decoding evidence, not visual or encoder-provenance approval. At 14:44:55, all five private model objects also passed full SHA verification via the **actual image-fetch code** in a read-only trial lasting about175 seconds. The largest file took138.53 seconds. Per-object timings include concurrent audit traffic and are not boot savings; no IAM/bucket/public ACL changes occurred.
- All20 required human-speech pose code/Whisper/audio files match the original878-file frozen manifest. First real pose attempt is retained as INVALID because the adapter dereferenced the venv interpreter to system Python; the corrected talking-pose attempt passes240-frame recipe checks. Six original-resolution source/standard/refined samples were directly inspected, with shared lower-face softness and no gross new jaw seam in this partial view. Full motion/audio/all48 visual review remains outstanding.
- Full48 rendering is in progress. Short smiling source videos trigger the adapter's initial240-source-frame requirement despite a sufficient canonical forward/reverse cache (first example158sourceframes/316cacheframes at24fps). A local, separately tested correction verifies cache length equals2xsourcecount and renders existing saved indices unchanged; it does not resize, reencode latents, or change numerical quality bounds. The active run is preserved and its remote adapter is not changed mid-run. [Docker size experiment CI](https://github.com/AhmadAFS1/MuseTalk/actions/runs/37794887252) has passed Linux contracts and is building.
- The first full48 attempt finished **INVALID**, with32 render-recipe checks PASS and16 smiling poses rejected by that adapter gate. Its raw5.48MB report was copied off-worker and fingerprinted; a [compact index](avatars/native_v1_all48_1500_invalid_index.json) retains the failure and lineage. After the GPU child ended, the separately tested canonical-cycle correction was copied and hash-verified. A new full48 attempt is running in a distinct output. This does not waive native numerical rejection or establish visual acceptance.
- Account quota at15:19UTC reports3% used/97% remaining in its weekly window, normal usage allowed, purchased credits0 and one unused free full reset. This is account-wide allowance, not an exact remaining token count or a guarantee that every outstanding technical gate fits. No reset or credits were consumed/purchased by this task.
- Independent API-cache staging completed **48/48 PASS** in71.99seconds:47new exclusive copies and1verified existing independent copy. All required fixed-file and PNG inventory hashes match the completed CPU audit; source hashes remain unchanged and target inodes are separate. This is a local development handoff, not S3 download timing or fresh-instance readiness. [Report](avatars/api_cache_stage_1535.json).
- Docker CI37794887252 completedsuccess onff8b77b. All10small reports were fetched read-only without copying authentication or writing on the protected worker; complete archive SHA matches GitHub's declared digest. Actual uncompressed size is **18,176,397,997bytes**, down **1,483,163,666bytes /7.544%** from19,659,561,663; apt pins and pip freeze are identical. Final-stage CPU imports and compiled MMCV CUDA12.1 checks passed. This is not a registry digest, compressed transfer size, GPU acceptance, or boot-speed measurement. [Verified evidence](release/dependency_ci_ff8b77b/provenance.json), [size comparison](release/dependency_size_delta_ff8b77b.json).
- A separately labelled follow-up keeps the pinned devel builder, unchanged apt packages, private/model exclusions and all Linux TensorRT resources, but uses the matching pinned CUDA12.1.1/cuDNN8 **runtime** final base. Its actual size/import build is pending; no extra reduction is claimed yet. Isolated live plan/preflight passed without server launch, preserving warnings about missing diagnostic engine-store records. Actual backend proof and fresh call tests remain required.

## Current bottleneck interpretation

[NVENC acceptance failed on this A1 host](native/nvenc_acceptance_failure_1750.json): fresh N1 has0hardware encoder opens and18open failures, software fallbacks and strictP1/P2/P3FAIL. After the API stopped, independent PyAV16.1 and systemFFmpeg4.4 single-black-frame probes both failed `OpenEncodeSessionEx: unsupported device (2)`. PyAV loaded the595.91.07 encode library, initialized API13 and recognized RTX3090sm86, so missing library or too-old API initialization is not established as the cause. Existing container capability environment omits `video`; [NVIDIA documents that capability for the Video Codec SDK](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/docker-specialized.html), but loaded libraries mean this configuration observation alone does not explain the failure. No host driver/container/device-permission changes or higher live levels were attempted.

[Private raw live evidence](release/private_live_baseline_tuned_1740.json) now passes conditional checksum-keyed S3PUT, exact-version freshGET and all152 payload hashes: seven stages across the3codec profiles plus API logs, compressed3,915,581bytes, archiveSHAd56d04d0.... Raw traces/logs may contain session credentials and are excluded from publicGit. This preserves failures; it does not establish acceptance.

Latest terminal live result: [tuned N3](native/isolated_live_x264tuned_v1_s0_n3_1716.json) strictPASS3/3, but [tuned N5](native/isolated_live_x264tuned_v1_ramp_n5_1719.json) validFAIL over330.56seconds: all5P1/P3FAIL and oneP2FAIL,18minimum anchored FPS,166.9ms worst gap,27gaps over100ms and6held frames. Actual encoder-open log confirms libx264/veryfast/thread1. Relative to the earlier sequential baseline, maxthreads fell360→214 and server loop-lag p99max68.864→30.781ms, but worst gap and held-frame outcome worsened; this is not a causal or net PASS claim. NoN10/N15/soak advanced. [Assessment](native/isolated_live_x264tuned_v1_n5_analysis_1735.json). The owned idle tuned API was stopped gracefully before a separate NVENC-only plan. That plan changes only encoder implementation/output root, restarts scoring atN1 and refuses progression after missing hardware-slot or fallback evidence. Production remains untouched.

[Graph-lineage inspection](native/native_portable_graph_lineage_1735.json) verifies identical recipe/precision and calibration filenames, but10of11 UNet ONNX hashes differ; onlytail matches. Build-device FP16 timestep embedding baked into non-tail wrappers is a concrete source lead, **not a proven cause**. TAESD has the exact same ONNX graph yet maxerror7versus5LSB, so it needs a separate tactic/precision diagnosis. Frozen bounds remain unchanged. [Actual Maya3-pose still review](avatars/native_v1_maya_3pose_still_review_1722.json) covers18source/standard/refined samples at original dimensions from the verified private restore: shared lower-face softness, no obvious gross new chin seam in these stills. It does not include portableA/B, continuous motion/audio/transition review or all48 subjective acceptance.

The [bounded graph diagnostic](native/time_embedding_graph_diagnostic_1750.json) then reproduced the exact native down0rest ONNX hash. CPU16half and GPU1 constants each differ from GPU16 by2embedding elements (max0.000030517578125), changing only the `emb` initializer; node structure is unchanged. RoundedCPU32 computation changes302elements but still only that initializer. None reproduces the portable graph. This strengthens reproducibility/provenance evidence, not a quality-fix or cause-of-all-failures claim. No new engine was built and the17.5s leased probe caused0OOM events.

Corrected native production rendering is **48/48 compatibility PASS**, not numerical or visual acceptance. The [terminal compact index](avatars/native_v1_all48_cycle_v2_1523_index.json) binds the8.2MB raw report by SHA. All48 muxed review videos independently decode to their240 original frame hashes and byte-identical first10s human PCM. The initial archive failure remains preserved; [all3 private archive fresh GETs and clean CPU restores](release/native48_private_persistence_and_restore_1712.json) now pass, with293unique files and all48 pose hashes reverified.

Account quota at17:36UTC is6%used/94%remaining in the shared weekly allowance, ordinary usage allowed, purchased credits0 and one unused free full reset. No exact subscription token balance or completion guarantee is available. Quota is not the current blocker; engineering acceptance and existing registry/access/rights gates remain outstanding. No reset or purchase was made.

The isolated API is healthy on127.0.0.1:8300 with registration disabled and both native TensorRT backends active. All16 talking process caches are resident in3512.87MiB with0evictions. Actual idle clips consume157.5MiB each: the2400MiB limit holds15, so warming16 evicts1. Active-stage residency is checked before timing; no all16/all48 idle-warm claim is made. The scheduler also raises compose/encode workers from requested4 to10 using visible host CPU count instead of the18.43199 container quota; this is a concrete configuration issue, not yet a proven throughput cause.

[First new S0 evidence](native/isolated_live_v1_s0_n1_1605.json) passes strict P1/P2/P3:20anchored frames/s,66.1ms worst gap,100%fresh,0held frames,100%PTS joins. Overall result remains **INVALID** because RTP payload97 provesVP8, not the intendedH264. A separately tested H264-only client-offer hook retries in a new output without changing server or scorer. One stream cannot establish multi-stream capacity, native quality parity, NVENC, or EC2/browser readiness.

The separate [H264-only retry](native/isolated_live_h264_v2_s0_n1_1613.json) finished strict **PASS**: actual video/H264 packets received,20anchored frames/s,74.7ms worst gap,100%fresh,0held frames,100%PTS joins over116.1seconds. Two matched first fresh-frame latencies are0.6696/0.7908seconds on the same worker clock, not first decoded speech-frame latencies. This is the software-aiortc/libx264 diagnostic profile, not a claim of NVENC or deployment acceptance. Three streams are gated on this exact strict result after media I/O finishes.

The matching-runtime-base follow-up is building in [CI37803474546](https://github.com/AhmadAFS1/MuseTalk/actions/runs/37803474546) at exactead7e0116577a7b32da341af514c8371c187cc9c. No further size or startup saving is counted yet. Account quota at15:58UTC is4%used/96%remaining, normal usage allowed, purchased credits0, one unused free reset; exact remaining subscription tokens and full-task completion are not guaranteed. The OpenAI Docs skill informed interpretation of allowance percentages, not engineering acceptance.

That CI subsequently finished success at16:26:35UTC. Its10reports were retrieved read-only and the complete ZIP SHA matches GitHub's declared0ceddda0... digest. The dependency-only image is now **12,334,990,710uncompressed bytes**, **7,324,570,953bytes /37.257% smaller** than the original19,659,561,663-byte build. Apt pins and pip freeze remain identical to the preceding pruned build; final-stage CPU imports and compiled MMCV CUDA12.1 check pass. [Verified evidence](release/dependency_ci_ead7e01/provenance.json), [size comparison](release/dependency_size_delta_ead7e01.json). No registry digest, compressed pull size, GPU/container acceptance, full model/Kokoro capability or boot savings are inferred. The full release Dockerfile still retains its original build stage pending validation.

All48 native production review media are archived in3parts totaling6,217,118,253bytes, with101/100/100manifest entries; the repeated global evidence is intentional. All240frame hashes perpose and byte-exact first10s human PCM were verified before omitting duplicate silent videos. Copies offA1 are running outside scored load. A tiny private decoded-proof conditional PUT from the existing runtime credential identity returned403; its failure is preserved, no IAM/bucket policy was changed and no blind retry occurred. Operator-only conditional persistence and fresh verification remain pending. An extra$2transfer contingency was reserved within the same$30authorization, total reservations9.584739583333334USD—not confirmed spend and not another rental.

The source bootstrap itself took 631 seconds and portable engine restore 72 seconds;
one avatar warm took 23.82 seconds, including 21.77 seconds in S3 restore. Building
native engines took 1,823 seconds in development and must never occur at normal
boot. These observations support an immutable prebuilt runtime plus checksum-pinned
engine/private-model delivery, not a blind snapshot containing credentials and caches.
The actual dependency-only Docker build now passes, but its 19.66 GB uncompressed
size excludes model weights and plans. Compressed pull size, registry digest,
three fresh/restart distributions, and request-to-usable-call remain unmeasured.

For throughput, native T filled about 69.6% / 63.6% of padded batch slots. Its
per-job GPU service time was 37.87 / 36.56 ms; CPU FaceMesh and composition work,
batch formation and GPU credit waits are measured leads. Reported nested IPC and
FaceMesh wall times must not be added as independent costs. The thermal flag is
an observation, not a proven explanation; GPU memory temperature is unavailable.
No power/clock/fan changes were made. A rejected numerical candidate is not promoted
because its GPU kernels are faster. Existing releases and templates remain unchanged.

This separates several minutes of provider/image startup from source cloning,
installation, and later model/avatar warmup. Exact image-pull boundaries and
provider cache state are not known. The original health milestone was observed
without manual repair. Cross-host UTC clock skew has not been independently
calibrated; stage durations from individual processes are reported separately.

## Hardware and spending controls

- Actual GPU: NVIDIA GeForce RTX 3090, sm86, 24,576 MiB visible VRAM, driver
  595.91.07, 350 W configured power limit.
- Vast machine 153039; 150 GB instance disk, about 149 GB initially free;
  `/dev/shm` about 31 GB. Cgroup CPU quota is **18.432 CPUs**, not the 96 visible
  host CPU IDs. See the exact memory/CPU evidence below.
- User-approved total: **$30**, including experiment costs. Vast download and
  upload rates must each be no more than $1.50/TB.
- Selected provider rate: about $0.241111/hour including selected storage;
  conservative adjusted reservation uses $0.249249/hour. Provider reports
  $1.333333/TB transfer each way. First resource reservation is $5.584740.
- A separate $2 AWS audit/transfer contingency was added under the same $30
  authorization before the 48-pose content download, retaining the original A1
  reservation. Combined reservation is $7.584740, not an invoice. The shared EC2
  ledger was amended under its lock with an exact previous-SHA guard and a backup.
- An EC2-owned experimental-only expiry timer is active for **19:00 UTC**,
  with bounded retries. It is best-effort software cleanup, not a provider-side
  hard cap. No production instance is an eligible cleanup target.
- AWS S3 request/storage/egress costs are separate from Vast traffic rates and
  must remain within the same $30 total. Do not assume AWS's account-wide free
  transfer allowance is available.

## Verified preparation

- CPU tests cover isolation, startup, harness reports, timing-cache provenance,
  Docker lifecycle and private-model delivery. Individual reports distinguish
  actual passes from platform/dependency skips; these do not establish GPU/image acceptance.
- 64 expected S3 objects passed fresh HEAD checks: 16 portraits and 48 pose
  caches. This proves availability/size/metadata, not archive or latent integrity.
- The exact pinned portable r5 and load-test audio archives were absent at their
  configured S3 keys. Their original known-good archives were recovered read-only
  from the protected worker, checksum-verified, and uploaded to those missing
  content-addressed keys. Fresh S3 downloads matched both hashes. The portable
  archive also passed a clean CPU restore with 17 verified files.
- Quality policy and measured reference envelope are frozen separately. The two
  portable captures have identical canonical output pixels and all 594 per-avatar
  metrics; 104 explicitly registered UNet/TAESD metrics are also bounded, with
  per-metric observed repeat spread only. Historical references were checked for
  matching inputs, metric implementations and supporting artifact hashes. All
  original strict failures remain failures. Native parity and visual acceptance
  are still incomplete.
- Docker lifecycle/private-delivery changes passed independent review and CPU
  tests. The first dependency CI failed a root-only startup-test assumption;
  that test was corrected for non-root runners. Its successor passed Linux CPU
  and startup tests, then exposed a real MMEngine 0.10.4/PyTorch 2.5.1 Adafactor
  registration collision in `mmcv.ops`. That failure was reproduced on the GPU
  worker. A narrow upstream registry-name backport preserves the package pins;
  actual avatar-prep imports are now included in the installer smoke. The next
  [dependency CI run](https://github.com/AhmadAFS1/MuseTalk/actions/runs/37769890795)
  passed, with its nine audit reports preserved. No serving image has been built or published; no Vast template
  has been edited or promoted.
- The isolated live adapter requires full lifetime telemetry, matched first
  frames, complete send rings, and observed H264 payloads. Each load level has a
  separate invocation so a failed level stops escalation. These CPU contracts
  passed; native GPU/live validation remains outstanding.
- Provider finalized-charge access returned HTTP 401 at 08:21:58 UTC. Actual
  invoiced spending remains unavailable, not zero; reservation and expiry controls
  remain in force. No permissions were widened or billing request repeatedly retried.

## Evidence and unresolved work

- [Run state](run_state.json)
- [EC2 provisioning contract](provisioning/EC2_CONTRACT.md)
- [Instance A initial observations](provisioning/instance-a-initial-observations.json)
- [Original artifact recovery](provisioning/baseline_artifact_recovery.json)
- [Live S3 availability audit](avatars/s3_head_audit.json)
- [Quality acceptance policy](quality/quality_acceptance.json)
- [Startup optimization targets](startup/acceptance.json)
- [Source-install startup breakdown](startup/source_install_baseline.json)
- [Local smoke assessment](portable/source_install_smoke_assessment.json)
- [Sustained portable GPU-path evidence](portable/portable_v1_gpu/report.json)
- [Portable full-recipe T](portable/portable_v1_aggregate/report.json)
- [First portable quality reference](quality/portable_ref1_quality/report.json)
- [Second portable quality reference](quality/portable_ref2_quality/report.json)
- [Frozen measured quality envelope](quality/reference-envelope-v2.json)
- [Budget checkpoint](provisioning/budget-checkpoint-0748.json)
- [Finalized-charge availability check](provisioning/budget-checkpoint-0820.json)
- [Instance A CUDA failure and host outage](provisioning/instance-a-host-outage.json)
- [Actual portable still-frame review, with limitations](quality/portable_reference_visual_inspection.json)
- [Recovered pinned-input native preflight](native/native_v1_preflight_recovered_v2_check/report.json)
- [Complete native UNet manifest](native/native_v1_build/engine_manifest.json)
- [Native decoder metadata](native/native_v1_build/taesd_trt_1e967e6e715c9f1a8375.json)
- [Native candidate preflight](native/native_v1_check/report.json)
- [Native v1 rejection decision](quality/native_v1_quality_decision.json)
- [Native full-recipe T/SUST evidence](native/native_v1_aggregate/report.json)
- [Recovered portable control T](portable/portable_recovered_v1_aggregate/report.json)
- [Native partial direct still-frame inspection](quality/native_v1_partial_visual_inspection.json)
- [Private native diagnostic persistence and fresh restore](release/native_v1_diagnostic_persistence.json)
- [Private native capture persistence and fresh restore](release/native_v1_media_persistence.json)
- [Private-model fallback upload/read assessment](release/private_model_operator_fallback_assessment.json)

Registry publishing access remains unresolved. A read-only native x64 GitHub
runner probe succeeded with about 86 GiB free disk, 4 CPUs, 16 GB RAM, and Docker/
Buildx available; see [builder inventory](release/builder_inventory_summary.json).
The full image build and audited artifact delivery remain unvalidated.
The renewed [dependency CI](https://github.com/AhmadAFS1/MuseTalk/actions/runs/37769890795)
uses revision `b00cc20fa5b77c051e468d993c6522a4ae921f9e`, with the narrowly scoped
MMEngine optimizer-registration backport and an actual `mmcv.ops` import check.
It passed; the [retrieved evidence](release/dependency_ci_b00cc20/provenance.json)
has a matching 25,022-byte artifact archive digest. The dependency image is
19,659,561,663 bytes uncompressed before weights/native plans; compressed registry
transfer size and boot latency are unmeasured. This dependency-only image contains
no models, excludes Kokoro, was not GPU-tested or published, and is not promotable.
The earlier failed run remains evidence.
Accepted native engines, ≥400 FPS evidence, quality parity, production-pose render audit,
image publication, browser template edit, fresh Instance B, real EC2/TURN media,
repeated startup trials, and live soak/scale-in tests remain required. No overall
completion or release approval is implied by the CPU preparation.

## 16:52 UTC checkpoint

The runtime-base dependency experiment [passed Linux CI](https://github.com/AhmadAFS1/MuseTalk/actions/runs/37803474546).
The actual uncompressed image is 12,334,990,710 bytes, 37.257% smaller than the
original 19,659,561,663-byte dependency image. Apt pins and pip freeze are equal;
final-stage CPU imports and compiled MMCV CUDA12.1 checks passed. This is not a
compressed registry size, full serving image, GPU validation, or measured cold
startup saving. The [size comparison](release/dependency_size_delta_ead7e01.json)
and complete matching-digest CI artifacts preserve those distinctions.

The H264-only client offer passed isolated S0N1 with the unchanged strict scorer.
The first VP8 attempt remains INVALID. S0N3 is running after all heavy worker
archive transfers ended. The same-plan API was restarted for a separate startup
trace; that trace failed its known-UNet positive control and cannot establish
that SyncNet is unused. The exact detached tracer was stopped alone, and all14
API threads had TracerPid0 before scoring. No SyncNet prerequisite was removed.

All48 original-resolution audio-bearing review videos passed decoded frame-hash
and byte-exact human-PCM integrity checks. Three private review archives total
6,217,118,253 bytes; all copied off-worker and their complete hashes match.
Mac-only conditional S3 persistence and fresh GET verification are in progress.
Fresh CPU restore and full visual/audio review remain separate, unfinished gates.
The runtime's tiny write probe failed403; no IAM/bucket policy was widened.

The protected shared budget reservation is now $9.584740 of the $30 cap, including
an additional AWS traffic contingency; this is not finalized billing. Account
quota last checked15:58UTC was4% used/96% remaining, with no reset or purchase.
The experimental RTX3090 still expires19:00UTC.

At16:58UTC, isolated S0N3 passed all3 streams with the unchanged strict scorer:
115.5s steady, anchored minimum20fps, worst gap88.4ms, no gaps over100ms,
100%fresh frames, no held frames, and100%PTS joins. The five-stream stage is
running behind the exact previous-result gate. Archive1's conditional PUT,
version-specific fresh S3 GET and clean CPU restore passed, with101payload
files verified; archives2/3 remain pending. A fresh account quota check reports
5%used/95%remaining, no reset consumed and no purchased credits.

At17:12UTC, all3 private review archives passed exact-version fresh GET and
clean CPU restore. A combined reread verifies293unique payload files spanning
all48poses; see [private persistence descriptor](release/native48_private_persistence_and_restore_1712.json).
These are archival integrity passes, not subjective visual/audio or native
quality approval. Large media remain private and excluded from Git.

The baseline five-stream stage is a validFAIL, preserved with all original
thresholds: all5streams fail strict P1/P2/P3, minimum anchored15fps, worst
gap143.2ms,47gaps over100ms, despite100%fresh/no held frames. No10/15/soak
escalation occurred. Its [diagnosis](native/isolated_live_h264_v2_n5_analysis_1708.json)
records measured68.9ms server-loop p99 lag,360threads, and the actual default
aiortc medium/automatic-thread encoder; causation is not established.

A separate SHA-pinned trial enables only the existing x264tuned implementation,
activating veryfast/one-thread settings. Engines, avatar inputs, CPU allocation
and strict thresholds are unchanged; a new output root preserves the baseline.
The isolated API432033 confirmed actual backend selection, warmed the same16
talking process caches and is running fresh S0N1. It cannot inherit the baseline's
passes. Production, native quality rejection and full-recipe400FPS failures
remain unchanged. The [startup assessment](startup/optimization_assessment_1701.json)
separates measured preparation from still-unmeasured cold-call readiness.

The tuned trial's fresh S0N1 subsequently passed:114.4s steady, anchored20fps,
worst gap70.5ms, no gaps over100ms, actual H264 and installed x264tuned/thread1.
Its new three-stream stage is running under the exact trial-specific PASS gate.

## 20:36 UTC checkpoint

The development instance expired and was destroyed at19:00UTC; provider absence
was verified. Nativev1 remains rejected for numerical quality and below400FPS.
The complete subsequent live/archive/SyncNet evidence is indexed in run_state.json;
earlier in-progress checkpoints above are historical, not current status.

The exact scheduler pair gate passed53Linux/Python3.10 CPU tests at commit0cb45fd;
see [CI readback](native/tracking_parity_linux_ci_2036.json). It requires all6avatars'
saved faces, refined-frame hashes, landmarks and chin deltas to match the serial
control on the same GPU/engine/input/source before a long overlap run. Actual
FaceMesh/GPU parity and speedup remain unmeasured. Production defaults are unchanged.

Account usage reports11%weekly used/89%remaining, zero purchased credits and one
unused free full reset. The weekly window resets2026-10-15T11:17:04UTC. These are
shared-account limits, not an exact task token budget or a guarantee of completion.
No reset or credits were consumed. Execution continues with one agent and bounded
experiments; the $30 infrastructure cap is separate from Codex quota.

## October 9, 01:14 UTC checkpoint

A3 (54939993) was created once through EC2, with a bound 02:45 UTC expiry and
$2 reservation. Its actual RTX3090 has a 420W default limit; no power setting
was changed. Source-install health was observed after about 537 seconds, not
the requested 60-second image boot or usable-call readiness. The owned API and
TURN were stopped for isolated experiments; production and the 4070 are intact.

All18 canonical reference videos match the saved 4070 hashes. CPU restoration
now proves all866 nonmetadata inputs exact, including SyncNet, the historical
worker, and the eleven MediaPipe files from the canonical 0.10.9 wheel. Twelve
genuine Hugging Face download metadata records have new bytes. A separate
metadata-only lineage was preregistered at01:11:52UTC; all698 numerical bounds
and original reference files remain unchanged. This does not accept a candidate.

[Linux CI for 2bd9126](https://github.com/AhmadAFS1/MuseTalk/actions/runs/37867412893)
passed272 tests with11 explicit dependency skips. ActualA3 CPU tests also passed
the14 real-ONNX transformer tests and all33 runtime contracts; TensorRT behavior
is not established by those CPU tests.

The first isolated FP32-final-convolution build was stopped by the unchanged
watchdog before export: NVML reports hostPID714396, which is outside this
container's visible PID namespace. The attempt is INVALID, not a quality result.
Logs and partial fresh output are retained. A precise driver-provided process
mapping is being investigated; foreign-workload protection is not disabled.
No new400FPS, quality, image, publication, or startup-SLA pass is claimed.

The exact metadata-lineage verifier independently re-read all878 files at
01:13:28UTC and passed. The direct driver identity query then failed closed
(status74, unchanged PID sentinel, own client freed). A fresh alternate-offer
search found seven budget-eligible advertisements, but no matching VM offers
and no established host-PID option; none were purchased. Older driver versions
alone do not prove namespace compatibility.

An explicit single-process ownership watchdog is now being implemented for
review: a retained real native CUDA context must be present in a successful
complete NVML enumeration on the same physical GPU; a sole PID therefore
identifies it. MPS exclusion, exact UUID binding, parent acknowledgment, child
lifetime binding, and continuous foreign-process rejection are required. It
will not replace the original watchdog or initialize CUDA in CPU-spawn workers.
This design is not yet an actual GPU ownership, quality, or performance pass.
