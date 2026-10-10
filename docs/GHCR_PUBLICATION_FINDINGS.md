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
- Upload **completed successfully**. Version-pinned authenticated `HeadObject`
  confirms 3,951,486,311 bytes, SHA-256 checksum
  `rP9SXiKp7oCRfirjlO0HRtuqjPE7B2K2qd4hBCZdLRk=`, AES256 encryption and version
  `eOxBtMevZGxnA395QPPt9Lep2Z18W8k3`. Anonymous HEAD of the now-existing object
  returns **403**. No full object re-download is claimed by this readback.
- This is one temporary build-input object in the existing bucket, not an AWS
  registry or worker-ready image. Retain until the full-image build verifies it;
  then reconcile/remove only this task-owned staging version if no longer needed.
  No bucket policy/lifecycle, IAM, production secret or permanent compute change.
- Observer/doc changes are pushed at `81dabfb`; the unrelated tracked worktree
  edit is still excluded. Production EC2 readback is active with 2.8 GB free:
  it remains unsuitable for image compilation. Latest retry readback has bootstrap
  and independent placeholder pull PASS; credential-free dependency compilation
  in progress. A full serving digest and measured boot saving remain unavailable.

## 2026-10-10 — EC2 pull credential verified / change 11: public SSL fixtures

- The operator reported replacing the screenshot-exposed pull token and saving
  the expected username/token fields in the existing `lingua/api-keys` secret,
  region `us-east-1`. A read-only EC2 diagnostic using its IAM role confirmed
  successful secret retrieval, a classic token, exactly `read:packages`, an
  expiration header, and GitHub package visibility `private` (HTTP 200).
- GHCR pull authorization and HEAD of the existing non-serving placeholder
  returned HTTP 200 and the expected manifest digest. This proves credential
  transport only: no full serving pull, running container, cold-start timing,
  old-token revocation history or deployed autoscaler acceptance is claimed.
  No token values were displayed, written locally or sent to a rented worker.
  No AWS/IAM mutation, backend restart or GPU rental was performed.
- Latest delivery run remains
  [38049282097](https://github.com/AhmadAFS1/MuseTalk/actions/runs/38049282097),
  completed `failure` at source `ca2c34dc514e07a259f16c00604a47119644656e`.
  Bootstrap and its independent private pull passed; dependencies failed in
  the all-layer publication audit, and independent dependency pull was skipped.
- Hash-verified artifact `11669780644` (archive SHA-256
  `2e5f4db07642f203e7b544b47d9c4812690b5d989be456a9744bae37f12c8354`)
  identifies `future/backports/test/badcert.pem`, 1,928 bytes, SHA-256
  `262a107916641c7f211ac5898c0177535cd0bdc5aa872cc6e883842694d8f521`,
  as the exact `private-key-material` finding in stage `audit-layer`.
- Verified the complete official future 1.0.0 wheel against PyPI's declared
  SHA-256 `929292d34f5872e70396626ef385ec22355a1fae8ad29e1a734c3e43f9fbc216`
  and scanned every member without printing key material. The failed file
  exactly matches its public SSL test fixture. Six public test-key files were
  found; each now has a separate exact-path/full-hash/private-key-rule allowance.
  No wildcard, GitHub-token exception or blanket library allowance was added.
- Extended the reproducible public provenance verifier to the whole future
  wheel and added a synthetic SSL-fixture test that rejects changed bytes,
  moved paths and co-located GitHub credentials. Validation and publication
  retry results are recorded separately; the full serving image and GPU boot
  experiment remain pending.
- Change 11 validation: the **115-test Docker suite passes**, with one expected
  Linux-pidfd test skipped on macOS. The read-only public provenance verifier
  reproduces **all 11 exact-file exceptions** successfully, including the six
  future fixtures. `git diff --check` passes. Unrelated tracked/untracked
  preparation and media evidence remain excluded from this change.
- Repair committed and pushed as `e2349882620d6f26e82b613c5dc38e8d7cc31d88`.
  Explicitly dispatched private delivery retry
  [38078158793](https://github.com/AhmadAFS1/MuseTalk/actions/runs/38078158793);
  GitHub readback confirms `in_progress` at that exact source revision.
  Its eventual success/failure is not yet known. The protected RTX 4070 checkout
  remains at `5cc706e90e50e93da1310628c025a84199cd8042`; no checkout, restart or
  GPU operation was performed. Task-owned temporary Git-bundle copies were
  removed after the push; committed source remains recoverable in Git.

## 2026-10-10 — Planned `main`-merge image builds documented

- At the operator's request, added the image-update lifecycle and future
  build-on-merge plan to
  [Docker fast-startup plan](DOCKER_FAST_STARTUP_PLAN.md#planned-automated-image-builds-after-merging-to-main).
  Existing images/workers do not change when GitHub source changes; runtime
  changes require a new image, while documentation-only changes can skip it.
- Read back both local GHCR workflow definitions: their automatic push triggers
  still target only `codex/rtx3090-r5-delivery` with explicit path filters, not
  `main`. Recorded the intended full-image build, private publication, exact
  digest verification and separately approved promotion sequence, including
  layer-cache limits and unchanged existing workers.
- Documentation only: no workflow edit/dispatch, production digest change,
  credential mutation, GPU rental, AWS resource or external push. Main-merge
  automation remains planned, not implemented. No fresh CI status or cold-start
  measurement is claimed by this entry.

## 2026-10-10 19:45 UTC — Vast template readiness check

- Read live GitHub run/job state for
  [38078158793](https://github.com/AhmadAFS1/MuseTalk/actions/runs/38078158793)
  at source `e2349882620d6f26e82b613c5dc38e8d7cc31d88`: bootstrap and
  independent placeholder pull succeeded; dependencies remain `in_progress`
  in the credential-free native amd64 build/CPU-check step. No successful
  dependency publication or independent dependency pull is reported yet.
- Rechecked the publisher contract: dependency receipts explicitly set
  `serving_image:false` and `promotion_eligible:false`, with no model weights
  or native serving bundle. Even a successful dependency publication is not
  an image usable as a MuseTalk worker in a new Vast template.
- GitHub's workflow inventory does not yet list the full-candidate workflow;
  the latest ten delivery-branch runs contain no full-candidate run. Local
  full-candidate request metadata is absent, and the documented full-image
  assembly/publication remains pending. No runnable full-image digest or
  fresh-container RTX 3090/cold-start acceptance evidence is available.
- Verdict: **not ready for a serving Vast template**. Updated the main plan's
  stale registry/credential checkpoint. This check made no workflow dispatch,
  registry/credential change, rental or production/template mutation; only
  local documentation was updated. Existing source-install startup remains
  the usable fallback while the full image is completed and tested.

## 2026-10-10 20:05 UTC — Dependency publication and independent pull PASS

- Run [38078158793](https://github.com/AhmadAFS1/MuseTalk/actions/runs/38078158793)
  completed `success` at `20:04:55Z` for source
  `e2349882620d6f26e82b613c5dc38e8d7cc31d88`. All four jobs passed:
  bootstrap, verify-bootstrap, dependencies and verify-dependencies.
- Retrieved the publication and independent-pull artifacts in memory; both
  ZIP sizes/SHA-256s match GitHub's artifact metadata. The exact published
  manifest digest is
  `sha256:7cd567bd77d421565c6813e5372561be31a8b3cab73f307430abd6b95d0067f6`.
  The receipt records **6,860,592,912 compressed layer bytes**, plus 16,470
  config bytes. This is registry payload size, not measured transfer time or
  a full serving-image size. All-layer audit, private visibility, anonymous
  denial and independent fresh-runner pull passed.
- Saved selected verified non-secret fields and artifact provenance in
  [publication readback](fps_comparisons/rtx3090_r5_20261008/release/ghcr_dependency_publication_e234988_verified.json).
  It explicitly preserves `serving_image:false`, `promotion_eligible:false`,
  no GPU acceptance and no Docker cold-start measurement. Do not use this
  dependency-only digest as the new serving Vast template.
- Next-stage read-only EC2 preflight confirms its IAM role can HEAD the exact
  previously staged weight/native archive **versions**, with matching sizes
  and SHA metadata. No multi-GB re-download or fresh full-content hash proof
  is claimed. Its saved pull credential authenticates to GHCR, and HEAD of
  the new dependency digest returns HTTP 200 with matching manifest identity.
- A concrete runtime handoff gap remains: `linguaEc2role` receives
  `AccessDeniedException` for `GetSecretValue` on the existing worker-runtime
  secret ARN ending `Dof4b8`. The existing local AWS identity is denied too.
  Therefore this preflight did not obtain worker credentials or test private
  model access; it does not prove that the worker's separate identity is denied.
  Requested operator approval for **only** this role/action/secret read grant;
  no IAM/secret changes have been made. This access gap affects startup
  credential transport, not the already-passed dependency publication.
- Full-image metadata/notice and native provenance assembly, private candidate
  build/publication/independent pull, and the fresh RTX 3090 usable-output boot
  test remain next. Preserve documented TensorRT/multimedia review obligations
  and historical quality/FPS verdicts rather than manufacturing a release pass.
  No GPU rental, production/template update, new AWS resource or background
  full-candidate dispatch was performed in this progress check.

## 2026-10-10 — Approved runtime-secret grant, CLI denied / console review

- Operator authorized the exact existing worker-runtime secret read grant,
  followed by a full private image build/publication and fresh RTX 3090 startup
  test. `iam:GetRole` confirms the actual role spelling is `linguaEc2role`,
  ARN `arn:aws:iam::211125449207:role/linguaEc2role`; do not target a guessed
  case variant or change its trust relationship.
- Existing CLI identity `lingua-backend-user` cannot list/read inline policies
  or perform `iam:PutRolePolicy`. The attempted new, uniquely named policy
  `MuseTalkWorkerRuntimeSecretRead-20261010-c51cb4d1` returned `AccessDenied`:
  **no CLI grant was applied**. No escalation of the local user's permissions
  or other IAM edit was attempted.
- Signed-in AWS console successfully shows the exact role and its eight
  existing policies. Prepared a separate new inline policy, leaving them
  untouched: sole action `secretsmanager:GetSecretValue`, sole resource the
  existing secret ARN ending `Dof4b8`, no wildcard or KMS/IAM/S3 grant. The
  policy is at Review/Create, **not saved**; browser policy requires final
  action-time confirmation, requested separately from the earlier CLI approval.
- A reproducible non-secret JSON policy is saved in Lingua's prepared rollout
  worktree at `backend/docs/iam/musetalk-worker-runtime-secret-read.json`.
  Full-image source/model/native assembly continues independently; successful
  permission readback, full publication and fresh GPU timing are not claimed.

## 2026-10-10 20:26 UTC — Runtime-secret grant saved and model access verified

- After the operator reported completion, AWS console readback showed success
  for `MuseTalkWorkerRuntimeSecretRead-20261010-c51cb4d1` on `linguaEc2role`,
  with nine policies rather than the previous eight. EC2's `iam-role`
  credential provider now successfully reads the existing worker-runtime secret.
  Values were consumed in process memory only and never printed or saved.
- The stored key pair authenticates as
  `arn:aws:iam::211125449207:user/musetalk-s3-runtime`. Initial combined probes
  incorrectly suggested model access was denied: the denied operation was an
  explicit-version object read, not the ordinary current-object startup path.
  Separated HEAD/current Range-GET checks pass for all four required face-parse
  and S3FD objects, with exact size/SHA metadata and unchanged observed versions.
  Only four bytes were fetched; this is access verification, not a fresh full
  content hash or GPU/runtime acceptance test.
- The existing `release.fetch_private_model` already uses ordinary `GetObject`
  when no `version_id` is requested, checks full downloaded SHA-256, and checks
  HEAD versions before/after. Use this existing checked path; do not request
  `s3:GetObjectVersion`, broaden IAM or insert current observed versions as new
  manifest version requests. No further policy or secret changes were made.
- Full-image reviewed input assembly/publication and a fresh GPU startup test
  remain pending. Dependency publication remains successful; no rental,
  production/template update, full-candidate dispatch or boot-time claim was
  made during this readback.

## 2026-10-10 20:31 UTC — Actual full-image input assembly checkpoint

- Extended the fixed read-only CI evidence collector for the successful
  `e234988` GHCR artifact layout. Verified the run/commit, exact artifact ID,
  GitHub-declared ZIP SHA/size and all 14 expected members before writing them
  locally. Build and publication `result.json` files remain distinct; no remote
  files, authentication or protected worker checkout were copied/changed.
  [Verified dependency evidence](fps_comparisons/rtx3090_r5_20261008/release/dependency_ci_e234988/provenance.json)
  now supplies actual OS pins and installed dependency fingerprints: 156 pip
  distributions, 419 dpkg packages, 415 OS copyright-file records. These are
  dependency-image facts, not the unbuilt full image's final SBOM or license pass.
- Added a reproducible, network-free assembly inventory helper. Its actual run
  verifies 211 allowlisted runtime source files against the committed `e234988`
  bytes, rehashes both preserved archives (3,951,486,311 and 983,926,034 bytes),
  streams/rechecks all 16 native payload hashes and the two canonical sidecars,
  and checks all 12 retained notices for the ten unchanged eligible weight files.
  Four required private runtime models remain external; unused SyncNet and
  optional Kokoro are not selected. No credentials, media or readiness stamps
  are assembled. Individual weight members were not rehashed again in this
  run; the hashed prior clean-restore receipt supplies that earlier evidence.
- [Full candidate assembly inventory](fps_comparisons/rtx3090_r5_20261008/release/full_candidate_assembly_e234988.json)
  is explicitly **not** `release.json`, not publication authorization and not a
  quality/FPS/GPU acceptance verdict. Preserved native build metadata does not
  supply a complete input-hash trace; the record retains that limitation.
  Remaining TensorRT/multimedia redistribution/source findings remain visible.
- The full-image loader still requires `redistribution_reviewed:true` for all
  included assets/packages, even for the selected private nonpromotable candidate.
  Requested an explicit operator choice for a restricted owner-only test-image
  path with unresolved public-redistribution findings retained, versus keeping
  the existing gate. No review flag, gate, workflow or package visibility was
  changed. No legal/public-release clearance is inferred from private storage.
- Seven focused synthetic collector/assembly tests pass (four collector, three
  assembly), including changed native bytes, duplicate/traversal members,
  symlink inputs and distinct build/publication receipt names. A first local
  assembly attempt failed on the sidecar's actual `size` field; the reader was
  corrected to the existing schema and the real assembly then passed. No
  partial deployable metadata was emitted by the failed attempt.
- Full manifest/metadata upload, ephemeral CI inputs, full build/publication,
  independent pull and fresh RTX 3090 measurement are **not started**. No GPU
  rental or production/template mutation has occurred.

### 20:33 UTC — Checkpoint committed and pushed

- Committed the selected non-secret documents, verified CI receipts, assembly
  inventory and helper/tests as `b8431cf2cf4e6fe3cf3ffce1bdf7e7858bbabb8d` and
  pushed only `codex/rtx3090-r5-delivery`. GitHub branch readback matches the
  exact commit. The unrelated development-preparation edit and private/untracked
  media remained excluded. All 23 selected files passed the source credential
  pattern scan and staged diff whitespace checks before commit.
- The small task-owned Git bundle was removed locally and remotely after
  successful push; it is recoverable from the pushed Git commit. The protected
  RTX 4070 checkout HEAD remained `5cc706e90e50e93da1310628c025a84199cd8042`;
  no checkout, restart or GPU operation was performed there.
- No candidate request/dispatch or serving-image publication was triggered by
  this checkpoint. Restricted owner-only test-image approval remains pending;
  no new permission, secret value, rental or production setting changed.

### 20:55 UTC — Docker bootstrap and private candidate policy correction

- The user clarified that private GHCR was already selected and authorized the
  updated Vast script plus startup/app-boot test. No redundant privacy approval
  is needed. The temporary standalone candidate stage prevents production
  registration during validation; it is not an owner-only distribution demand.
- Added `scripts/vast_3090_docker_boot.sh` and selected it as the full image
  ENTRYPOINT. It delegates to the existing supervisor, removes clone/install
  work, defaults to EC2-injected worker runtime values and does not embed keys.
  Vast image selection and `image_login` happen before bash runs. The exposed
  AWS keys in the old pasted script should be rotated; rotation is not verified.
- Corrected the metadata validator's public-only packaging requirement for an
  explicitly private, nonpromotable candidate. It now requires hashed findings
  bound to source/native hashes, retained notices, honest public review status
  and no blanket use-rights clearance. Public/validated gates stay unchanged;
  public GitHub release transport rejects private metadata. Original quality
  and aggregate FAIL verdicts are retained. This is not legal/license clearance.
- Added a metadata assembly helper from the exact preserved inputs and a
  [script handoff](VAST_RTX3090_DOCKER_BOOT.md). No full-image dispatch, rental,
  production/template change or startup measurement has happened in this step.
