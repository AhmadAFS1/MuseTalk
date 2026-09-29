"""Multi-stream chin render harness (see scripts/chin_multistream_render.py).

Modules:
  paths    - fixed locations (worktree, accepted identities, FaceMesh venv) and code-hash checks
  arena    - per-identity shared, read-only precompute ("lean" chin.prepare_refined) in /dev/shm
  worker   - one ordered process per stream: Tracker + 3-tap filter + chin.corrected_refined
  gpu      - backend setup from the worktree and the single GPU issue loop
  serial   - render_stage.py's timed loop, verbatim, for the single-stream serial baseline
  telemetry- /proc CPU accounting, nvidia-smi and MemAvailable samplers
  check_prep - CPU-only proof that the lean arena composes bit-exactly (accepted faces -> render.json)
"""
