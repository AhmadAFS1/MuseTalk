# Dependency optimization evidence

Execution log: [progress and results](../dependency_install_optimization_progress.md).

- `acquisition-manifest.json`: sanitized capture of the six complete Linux pip
  reports, bootstrap index metadata, original package sources, requirements and
  hashes. This is an **acquisition candidate**, not yet a deployable wheelhouse.
  Four source archives still need controlled Linux builds; most byte sizes still
  need measurement. Two missing historical hashes are explicitly identified in
  each artifact's `hash_provenance` rather than presented as historical evidence.

Stable upstream package URLs contain no credentials or temporary query tickets.
Raw pip/debug logs and launch credentials remain in protected EC2 storage and
must not be copied into this folder.

- `metadata-closure.json`: analysis of the captured requirements against explicit
  Linux CPython 3.10 markers/extras. Zero missing/version-conflicting active edges
  in all three groups. Metadata-only evidence, not native/install validation.
- `wheelhouse-manifest-a2.json`: all 194 verified wheels, including the four
  controlled Linux builds. Used by the corrected local pip/uv screening pair.
- `results.json`: sanitized interim timing receipts, quote/lease, failures, tool
  hashes and evidence inventory. `goal_complete` deliberately remains false.
- `timing-summary.json`: reproducible first-host medians, all retained samples
  and observer-clean sensitivity. This input alone does not prove later
  different-host/canary completion.
- `canary-results.json`: different verified-US-host canonical install/output
  acceptance, 511.097s package phase, failed second-host S3/recovery receipts,
  and verified cleanup of 56211950. The cleaned ledger is retained; do not mix
  these measurements into first-host medians.
- `resources.jsonl.gz`: preserved first-host raw sampler capture; 3,138 resource
  samples plus one header. Capture limits are documented in the report.

- `final-screen-results.json`: full serial/concurrent acquisition, actual
  copy/hardlink native checks, supported cache clean, verified teardown and
  final owned-resource/timer inventory. Screening gains are not full-boot gains.
- `evidence-audit.json`: reproducible consistency/coverage checks of captured
  inventories, native exits, videos, timing calculations and cleanup, bound to
  input SHA256 values. This is not an additional native run or rollout approval.

See the final [coverage audit](../DEPENDENCY_INSTALL_OPTIMIZATION_COVERAGE_20261016.md)
for exact requirements and conditional options not promoted to execution.
Historical snapshot `goal_complete=false` flags are retained capture metadata;
the final coverage/report carry the current gated-experiment completion decision.
