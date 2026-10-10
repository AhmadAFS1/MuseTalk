# Private GHCR publication — implementation findings

This journal records each publication change and verified result. It complements
[the Docker fast-startup plan](DOCKER_FAST_STARTUP_PLAN.md). No entry is a claim
of full-image GPU acceptance or faster cold startup unless explicitly measured.

## 2026-10-10 04:25 UTC — starting state reconciled

- Task branch: `codex/rtx3090-r5-delivery`; starting commit
  `29fd462e7a15d7cb81fbadde7946bd096c152007`.
- Run [38016153305](https://github.com/AhmadAFS1/MuseTalk/actions/runs/38016153305)
  passed private placeholder publication/pull, Linux CPU/startup contracts and
  dependency-image build/import checks. Its combined audit/publication step
  failed; independent dependency pull was skipped.
- Saved inspect receipt identifies actual `linux/amd64` dependency image config
  `sha256:9a927db536edaafe7a4e4a68a52ea61f412d02ecb81327ce700f04fa67078ef8`,
  source `0e18b145f920e3cf548e639c5d507944402128b1`, 12,335,016,514
  uncompressed bytes and the correct nonpromotable dependency label.
- Saved artifact contains build reports but no completed layer-scan/publication
  receipt. The outer exception handler suppressed the precise Python failure.
  A scanner false positive is a hypothesis, not a confirmed cause.
- Actual full serving image, reviewed release metadata, secure private build
  inputs and a separate Vast pull-only credential remain outstanding.
- Preserve unrelated local `scripts/repro_3090/prepare_ec2_development.py` edits
  and private/untracked evidence. No new rental, AWS resource, production secret
  or permission change is part of this diagnostic step.

## Execution sequence

1. Add credential-safe stage/rule/type diagnostics and persistent failure receipts.
2. Reproduce against small pinned base-image/format diagnostics before another
   expensive CUDA dependency compilation; retain fail-closed publication.
3. Fix the evidenced cause, with narrowly targeted regression tests and no broad
   credential-scan bypass. Rerun dependency publication and independent pull.
4. Assemble reviewed exact full-image inputs, build/privately publish the candidate
   and independently pull/CPU-check it. Do not invent redistribution or quality
   decisions, copy interactive login tokens into workers, or activate production.
5. Only after a runnable image and approved pull/runtime credentials exist, rent
   a fresh owned RTX 3090 with exact expiry and measure request-to-usable output.

Progress and changes are appended below immediately after each implementation
increment and test/run result.

## 2026-10-10 — change 1: safe failure diagnostics

- Added fixed stage names around registry login/privacy, image identity, layer
  export/layout/scan, push and independent pull.
- Audit findings now identify the rule, sanitized installed path (or only its
  fingerprint), layer identifier, complete file SHA-256 and file size. Matched
  content, tokens, URLs and arbitrary exception messages are never reported.
- CLI failures preserve an exclusive-create `failure.json`, including stage and
  a bounded exception-type name. Existing receipts cannot be overwritten.
- Credential patterns and fail-closed behavior remain unchanged. Added tests
  for secret-safe diagnostics, full-file hashing and receipt persistence.
- Test execution and CI reproduction results will be recorded separately below.

## 2026-10-10 — change 1 validation / change 2: inexpensive reproduction

- Local Docker suite: 100 tests, successful with the one Linux-only pidfd test
  skipped on macOS. `git diff --check` passes. Credential rejection rules were
  not relaxed. Corrected the history-stage marker to precede its subprocess.
- Added a CPU-only audit reproduction workflow for the exact public CUDA runtime
  base. It has only `contents:read`, no registry login/publisher token, no model
  inputs and no rental. Its 20-minute ceiling and small JSON-only artifacts keep
  the diagnostic bounded; no image layers/file contents are uploaded.
- If the audit rejects an exported layer, the diagnostic additionally records
  Docker export member sizes and compression classifications. This tests parser
  compatibility without assuming the failure is a credential match.
- The first non-secret request selects diagnostic iteration 1. The live run and
  its findings remain to be recorded after push/observation.
- Validation after adding the workflow: Docker suite remains 100 tests with one
  macOS skip; diagnostic YAML parsing and whitespace checks pass. No dependency
  rebuild or registry push is triggered by this diagnostic request.

## 2026-10-10 04:32 UTC — diagnostic dispatched

- Pushed commit `d0762c0a4cfd314c8c45d412254671f470a4e4bc` through the
  existing protected CLI login, without copying its token or changing the
  working runtime checkout.
- Run [38024375746](https://github.com/AhmadAFS1/MuseTalk/actions/runs/38024375746)
  is executing the pinned-runtime-base audit. No dependency rebuild, GHCR push,
  AWS change or GPU rental has been triggered.
- Independently reviewed full-release readiness: the existing ten-file model
  assessment identifies eligible exact bytes, but specifically retains TensorRT
  runtime distribution-scope and multimedia source/notice gates. Full metadata
  must not relabel those pending decisions as completed review. Existing candidate
  quality/FPS failures remain explicit and do not prevent diagnostic transport.

## 2026-10-10 04:35 UTC — first reproduced blocker / change 3

- Diagnostic run `38024375746` rejected the **public pinned runtime base**, before
  any registry login/push. The failure is `AuditFinding`, stage `audit-layer`,
  rule `private-key-marker`, path
  `usr/lib/x86_64-linux-gnu/libgnutls.so.30.31.0`, size 2,000,320 bytes, SHA-256
  `31890d4c10c55e8756cd7f721fdf787065107a31ffd81de5efe3b3b0eb7431c2`.
  This is a concrete reproducible scanner blocker, not a publishing-permission
  problem. The original full dependency run could have additional blockers.
- Changed **image-layer** PEM detection to require header plus key material,
  rather than treating a parser's literal header string as a credential.
  No binary/library path exemption was added. Raw, JSON-escaped, RSA/EC/DSA,
  OpenSSH, encrypted and legacy encrypted PEM are covered; read overlap is
  expanded to retain encryption headers across chunk boundaries.
- GitHub/AWS credential rules, forbidden credential paths, image config/history
  checks, private visibility and anonymous-denial requirements remain enforced.
  The stricter release-source marker check is unchanged.
- Added regression tests for harmless marker literals **and** embedded key
  material in the same synthetic binary path. Iteration 2 will test the classifier
  against the actual public base; no successful actual-image audit is claimed yet.
- Additional full-image capability check: the protected 4070's local API was not
  reachable on port 8000. This is not proof that Kokoro is absent/disabled and
  does not authorize silently dropping TTS from a full serving image.
- Change 3 local validation: 102 Docker tests complete successfully with one
  macOS-only skip; whitespace checks pass. The saved export diagnostic identifies
  tar layers, not a gzip parser failure, so no archive-format workaround was added.

## 2026-10-10 — user confirms full-image TTS capability

- The user explicitly confirms local **Kokoro TTS is not required** in this Docker
  image. Audio is generated by a separate OmniVoice-TTS worker. Combining workers
  is a possible future change, not part of this image.
- Full-image metadata may therefore honestly select `kokoro:false`, retain the
  explicit local-TTS-disabled runtime policy and omit Kokoro voices/model cache
  and its optional speech dependencies. This is now an approved capability choice,
  not an inference from missing captured files or the 4070's unavailable endpoint.
- This removes the local-TTS input ambiguity; it does not clear unrelated model,
  native, TensorRT, multimedia, image-pull or GPU/cold-start gates.

## 2026-10-10 — iteration 2 result: further public-base evidence needed

- Run [38024853291](https://github.com/AhmadAFS1/MuseTalk/actions/runs/38024853291)
  passed CPU contracts but again rejected the exact same public GnuTLS library
  SHA-256, now under `private-key-material`. Its embedded content is not merely
  a standalone header; the first classifier repair alone is insufficient.
- Retain fail-closed publication. Next verify public upstream test-key provenance
  and the exact installed binary before considering any narrowly scoped
  path/hash/rule allowance. No dependency rebuild, publication or rental occurred.

## 2026-10-10 — change 4: verified public fixtures, exact-file allowances

- Independently fetched the public pinned NVIDIA registry manifest and first
  compressed layer, verified both registry SHA-256 identities, and read only the
  implicated library. Its whole-file hash equals both failed CI receipts.
- All six embedded PEM blocks exactly match constants in the
  [GnuTLS 3.7.3 self-test source](https://raw.githubusercontent.com/gnutls/gnutls/3.7.3/lib/crypto-selftests-pk.c),
  source SHA-256 `c3d79122e072177f55dc30a98f68b949c0f36f6159a363aae9512a22a9280ed0`.
  These are published known-answer fixtures, not developer private keys. No key
  text is stored in diagnostics or this journal.
- Independently checked the actual pinned boto3/botocore **1.42.97** PyPI wheels,
  verified their official PyPI SHA-256 values, and fingerprinted three unchanged
  AWS example documentation files containing example-format identifiers.
- Added four **exact path + complete file SHA-256 + specific rule** allowances
  with public provenance. Changed bytes, a different path, other token categories
  and credential directories still fail. Scanner now collects every rule in each
  file rather than stopping detection after the first match. Every applied
  allowance appears in the layer receipt, together with the exception-list hash.
- Hardened diagnostic operation reporting to validate its fixed format even for
  directly constructed exceptions. No arbitrary exception text is reflected.
- This is a credential-audit repair, not a license review, serving acceptance or
  relaxation of private registry requirements. Validation/results follow below.
- Change 4 validation: 104 Docker tests successful with one macOS-only skip;
  `git diff --check` passes. The new reusable read-only provenance verifier
  independently refetched the public inputs and passed exact path/hash/rule-set
  equality with all four reviewed allowances. CI iteration 3 now repeats that
  verification before auditing every pinned-base layer. No successful actual
  base audit or dependency publication is claimed until its run completes.

## 2026-10-10 — iteration 3 dispatched

- Pushed `88ff504a18a2fe8e4f4bb758a70d64f9001fb4bf`, including the approved
  no-Kokoro capability decision and exact public-fixture scanner repair.
- Run [38025387316](https://github.com/AhmadAFS1/MuseTalk/actions/runs/38025387316)
  is executing provenance verification and the actual public-base audit.
- Publication/privacy/token requirements remain unchanged. The dependency
  rebuild will start only after this inexpensive gate passes.

## 2026-10-10 — change 5: separate-TTS capability regression

- Verified the existing full-image implementation selects `--without-kokoro`
  from `kokoro:false`, sets `SETUP_KOKORO=0` and
  `MUSETALK_DISABLE_LOCAL_TTS=1`, disables automatic installation, and forces
  Hugging Face/Transformers offline. No new installer rewrite is needed.
- Added a regression contract for that exact external-speech-worker profile.
  105 Docker tests complete successfully, with one macOS-only skip. No Kokoro
  model/voice/cache bytes are added to the full-image input set.
- CI iteration 3 has passed independent public-fixture provenance verification
  and CPU contracts; the actual base-image layer audit is still running.

## 2026-10-10 — change 6: actual installed dependency evidence

- Added a read-only inventory collector to both dependency and full-image CPU
  build reports. It runs offline inside the built image, not on a live GPU worker.
- Captures installed pip names/versions, declared-license metadata, complete
  native-library/notice file SHA-256s, missing RECORD-owned native files, dpkg
  binary/source package versions and installed OS copyright file hashes.
- The collector does not report direct-URL/auth metadata, follow files outside
  bounded package/doc roots, run GPU inference, mutate an image or assert legal
  acceptance. It explicitly retains `publication_review_accepted:false`.
- This supplies concrete final-byte evidence for the existing runtime review
  gates; it does not satisfy corresponding-source delivery by itself. Source
  builds and exact package scope still need reconciliation before full release.
- Inventory validation: initially caught a Python-version difference where
  `importlib.metadata.files` omits missing RECORD entries. The collector now reads
  raw RECORD ownership explicitly, preserving the pruned-library evidence.
  All 107 Docker tests now complete successfully, with one macOS-only skip.
- Local startup-shell suite cannot execute beyond its initial syntax checks with
  macOS's bundled Bash 3 (`declare -g` unsupported); Homebrew Bash is not installed.
  No shell was installed or production worker touched. The existing native Linux
  dependency workflow runs this suite before building and remains the actual
  startup-shell validation gate.

## 2026-10-10 — actual base audit PASS / dependency retry request

- Run [38025387316](https://github.com/AhmadAFS1/MuseTalk/actions/runs/38025387316)
  completed successfully. Native Linux CPU contracts, independent public-fixture
  verification and **all exported layers of the pinned runtime base** passed.
  The exact GnuTLS allowance is reported; no model/runtime serving claim follows.
- Updated the publication workflow to preserve sanitized failure receipts even
  when bootstrap or independent pull fails, not only successful result receipts.
  The dependency job already preserves its failure receipt.
- This workflow change requests a fresh dependency rebuild/publication on the
  next clean pushed commit. Current local Docker suite: 107 tests, successful
  with one macOS-only skip. Full release/runtime review gates remain separate.

## 2026-10-10 — dependency retry dispatched / remaining evidence question

- Pushed `795bb4c625295fe0ac647fab83c9732e8eef8f80` without modifying the
  protected worker's runtime checkout. Run
  [38025750175](https://github.com/AhmadAFS1/MuseTalk/actions/runs/38025750175)
  is executing the bounded dependency publication sequence. Its preceding
  build took 48m48s; the new run is not a serving image or cold-start measurement.
- Asked for any existing exact-package TensorRT/FFmpeg review and
  corresponding-source delivery record. Those saved dossier gates cannot be
  changed to PASS just because Kokoro is now excluded. Continue the in-scope
  publication diagnostics and installed-inventory collection while awaiting
  evidence; do not trigger an unreviewed full-image publication or rental.
- Downloaded actual base receipt: config
  `sha256:02f0c5f1a54bd88a5242a21ef690ab6826c7d36eb2b8134b32860a258427d97e`,
  3,379,534,165 uncompressed bytes, **ten layers scanned**. Exactly the reviewed
  GnuTLS fixture allowance was applied; all other base payloads passed unchanged.
- Dependency retry has passed private bootstrap and independent placeholder
  pull on fresh runners. Native Linux dependency/startup validation and compilation
  are next; dependency/full-image publication and serving readiness remain pending.
- Latest readback: **native Linux CPU contracts and startup-shell regression
  passed** in the dependency job. Credential-free compilation is now in progress.
  This resolves the local Bash 3 test limitation without installing a local shell.

## 2026-10-10 — repository evidence and responsibility clarified

- The user has no separate TensorRT/FFmpeg review or source-delivery records.
  Rechecked the repository: root `LICENSE` contains MuseTalk's MIT license and
  references several model dependencies; `README.md` explicitly distinguishes
  MuseTalk from other models' own license requirements.
- Our supplemental dossier already contains the exact TensorRT 10.3 packaged
  license texts, TensorRT OSS notices, PyAV license, pinned FFmpeg vendor build
  script/patch and source URLs/SHA-256s in `supplemental/sources.json`.
- Those records are available engineering inputs, not missing paperwork the user
  must supply. Their `built_image_binding:NOT_YET_CAPTURED` entries and the
  forthcoming installed-dependency inventory distinguish captured upstream terms
  from a verified finished-image payload/source-delivery record.
- Asking the user for preexisting records was premature as a prerequisite for
  completing that engineering reconciliation. Continue checking repository and
  official exact-package sources under the approved private GHCR/worker scope;
  escalate only a concrete remaining ambiguity requiring an owner decision or
  qualified interpretation. Do not silently turn pending review flags into PASS
  or treat private registry visibility as a blanket license exemption.
- Read-only CI check: dependency run `38025750175` is still in credential-free
  compilation; previous bootstrap, independent pull and Linux startup gates passed.
  This clarification changes no validator, package selection or production state.

## 2026-10-10 — verified the user's fork directly / retry failure diagnosed

- Confirmed via authenticated GitHub API that the working repository is
  **AhmadAFS1/MuseTalk**, not only upstream TMElyralab/MuseTalk. Its default branch
  is `main`; our implementation records are committed on
  `codex/rtx3090-r5-delivery`, remote SHA
  `2fa55ddafc4d3a5ab63ab0b31461af9c8371161e` at inspection.
- Remote blob SHA for `supplemental/sources.json` is
  `904dafdd7716da11b1c67363d260d1495aa298dc`; the model-file assessment blob is
  `7bb4a32be937d2232a0ea046221a871dc6234f8f`. Both match the local committed
  checkout. These notices/build references are present in the user's own fork.
- Retrieved run `38025750175` logs and its 25,028-byte build-report artifact.
  The image **compiled and passed final-stage CPU/import/MMCV CUDA 12.1 checks**,
  producing config
  `sha256:2e8600e9f5499a1191fd9e1e0a826b17009be6c2f26e6024c296760d400a2cc1`,
  12,335,019,078 uncompressed bytes. The overall build/report step then failed;
  registry audit/publication and independent dependency pull were skipped.
- Actual failure: the new offline inventory subprocess exited 2 because
  `/opt/musetalk/app/docker/musetalk/dependency_inventory.py` was absent.
  `context.py`'s explicit tracked-source allowlist omitted the helper. The saved
  `installed-dependencies.json` is an error message, **not a valid inventory**.
  This is my source-packaging defect, not missing information from the user or a
  compilation failure. No dependency image reached GHCR in this run.

## 2026-10-10 — change 7: include and precheck the inventory helper

- Added only `dependency_inventory.py` to the existing tracked-source allowlist;
  no credential, model/media, notice-dossier or broad directory bypass was added.
- Dependency preflight now rejects a missing helper before Docker pulls and
  compilation. Dependency and full Dockerfiles also check its presence before
  installation, preventing a late missing-file error after a long compilation.
- Added a tracked-context regression test that verifies helper inclusion and
  byte hash while retaining private-path exclusions. Updated the dependency
  workflow for a clean retry; actual results are recorded separately below.
- Change 7 local validation: **108 Docker tests successful**, one macOS-only skip;
  whitespace checks pass. The new tracked-context test reproduces the omitted
  helper condition that the earlier inventory-only synthetic tests missed.
- Pushed fix `8d4c8e2fd68882e04a95abb3e7dd5a3e921d6778` to the user's fork.
  Retry [38030025665](https://github.com/AhmadAFS1/MuseTalk/actions/runs/38030025665)
  is running at that exact source revision. No model-bearing publication,
  worker launch, rental, AWS resource or permission change has occurred.

## 2026-10-10 — dependency inventory succeeds / change 8: PyAV fixture

- Run [38030025665](https://github.com/AhmadAFS1/MuseTalk/actions/runs/38030025665)
  compiled and passed CPU/import/startup checks, including the repaired offline
  inventory: **156 pip distributions and 419 dpkg packages**. Publication failed
  in `audit-layer`; independent dependency pull was skipped.
- Its sanitized receipt identifies only
  `av.libs/libgnutls-b786e1df.so.30.41.0`, 2,309,529 bytes, SHA-256
  `a372925fcc5697819476f4e67b22147669cbdff58cb398b01ce0c540d1c5a92e`.
  Downloaded the official exact PyAV 16.1.0 CPython 3.10 amd64 wheel, verified
  its whole SHA-256 against PyPI metadata, and scanned **every wheel member**.
  This library was its only scanner finding. All eight embedded PEM blocks
  exactly match public GnuTLS self-test constants in the hash-pinned source.
- Added one exact path/hash/rule allowance and extended the public provenance
  verifier to reproduce the whole-wheel check. No broad library exemption,
  credential output or license-acceptance flag change. FFmpeg/TensorRT payload
  and source review remain separate from this false-positive repair.
- User requested completion followed by a fresh Vast RTX 3090 cold-boot test
  and a separate Docker-specific startup bash script. Continue that preparation;
  do not rent a non-serving dependency diagnostic or change production defaults.

## 2026-10-10 — change 9: dedicated Vast Docker startup script

- Added `scripts/vast_docker_onstart.sh` and selected it under `tini` in the full
  Dockerfile. It runs inside the selected, digest-pinned headless Vast image;
  it is not a nested-Docker launcher or source installer. Fixed baked-path checks
  fail closed, then it execs the existing supervised immutable lifecycle.
- Its timestamped `VAST_DOCKER ENTRYPOINT` marker measures post-pull execution,
  explicitly not readiness. Provider request/allocation/pull timestamps must be
  combined with canonical model/GPU/health/usable-call evidence for cold start.
- Added behavioral tests for modes, missing files, symlinked metadata, argument
  rejection without echoing inputs and Dockerfile/context selection. No live
  launch template, running worker or production recipe was changed.
- Change 8/9 validation: **114 local Docker tests passed**, with one Linux pidfd
  test skipped on macOS. The read-only network provenance verifier passed all
  five exact fixture entries (whole boto3/botocore/PyAV wheels and pinned base).
  Dependency CI now reruns this cheap check before lengthy compilation.
- Pushed `ca2c34dc514e07a259f16c00604a47119644656e`; dependency publication
  retry [38049282097](https://github.com/AhmadAFS1/MuseTalk/actions/runs/38049282097)
  is running. Bootstrap and independent private placeholder pull already passed.

## 2026-10-10 — change 10: bind cold-start observer to the requested image

- Reused the existing EC2 startup observer rather than introducing another
  rental/timing implementation. Optional `init --image-digest` binds a full
  private GHCR digest into the fresh ledger; Docker creates reject image drift,
  source-template IDs and SSH-mode replacement before budget reservation/POST.
- Reports carry that requested digest, explicitly **not** claiming it proves
  actual running-container identity. Provider/image readback remains required.
  Existing secret nonpersistence, no ambiguous-create retries, expiry intent,
  cache-state labeling and first usable media gates are unchanged.
- Standalone GPU/model readiness is not EC2 routability or live-call acceptance.
  Unknown provider cache stays `unknown`; a new instance does not prove a cold
  host or absence of cached Docker layers. No measured latency is claimed yet.
- Change 10 validation: **30 startup-observer tests pass** (synthetic requests,
  no cloud writes); the 114-test Docker suite remains successful with one macOS
  skip. The unrelated `prepare_ec2_development.py` worktree edits are preserved
  and excluded from our commits.

## 2026-10-10 — full-image input staging and actual notice binding

- Rehashed the existing no-Kokoro ten-file model archive: 3,951,486,311 bytes,
  SHA-256 `acff525e22a9ee80917e2ae394ed0746dbaa8cf13b0762b6a9de2104265d2d19`.
  Its destination in the existing regional S3 bucket returned authenticated
  404 (absent). All four public-access-block settings are true; there is no
  bucket policy. Began a checksum-verified AES256 `PutObject` with expected-owner
  check and `If-None-Match:*` at
  `docker-build-inputs/sha256/<archive SHA-256>/weights.tar.gz`. This cannot
  overwrite a previous object. Upload completion/readback is recorded separately.
- Run 38030025665's actual dependency image is
  `sha256:2cd9327f0363906750234dabfec39ad62bc7082fb3a60c294f93fc5076d6600e`,
  12,335,024,109 uncompressed bytes. Its installed inventory SHA-256 is
  `b5bc5fa1534a04a7516983551b13084f0f1630c6c1ed138b5a04dd54255a45cd`.
- Exact **installed** TensorRT 10.3 libs/bindings notice hashes now both match
  the fork's captured packaged terms:
  `64bd290f0251405f783ba1d2e155c500542be69795e51147a1d9f11a57bda8cc`.
  Four Linux libraries and the binding are present; the deliberately pruned
  Windows resource remains visible as missing in raw RECORD inventory.
- Installed PyAV 16.1.0 wrapper notice SHA-256 matches its captured upstream
  BSD notice (`76af0461ffb92e19f1c14449e95557d83a2dfaa1baf202d49e5f1d8746c0da19`).
  This is not a blanket license for its 87 native files or bundled codecs.
  OpenCV retains its 151,157-byte third-party notice; imageio-ffmpeg retains its
  wrapper notice. Do not confuse notice byte binding with a qualified resolution
  of the previously documented TensorRT distribution/FFmpeg source questions.
- Asked only for the secure location of a separate expiring `read:packages`
  credential (or human-assisted setup), not for repository license paperwork.
  No publisher/OAuth token was extracted or sent to a rented host. Full metadata,
  full private candidate publication, runtime access and the GPU test are pending.
