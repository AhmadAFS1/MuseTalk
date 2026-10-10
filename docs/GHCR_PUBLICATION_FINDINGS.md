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
