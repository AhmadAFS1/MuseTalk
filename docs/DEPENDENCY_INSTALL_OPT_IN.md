# Experimental locked dependency installation

Status: implemented as an **opt-in**, not approved for the production default.
See [test progress](dependency_install_optimization_progress.md) and the
[full plan](DEPENDENCY_INSTALL_OPTIMIZATION_PLAN_20261015.md). No Docker is used.
The [replacement Vast bootstrap and rollout plan](VAST_LOCKED_STARTUP_ROLLOUT_PLAN_20261016.md)
separate the template, repo installer and EC2 ticket-injection changes. These
local deployment files are not published or production-enabled yet.

## What changes

The source-pip default keeps its existing package pins, indexes and capability
groups. A locked mode obtains the reviewed exact wheels with bounded parallelism
and installs offline using pip or pinned uv. It does not rebuild TensorRT engines,
change the inference recipe, remove Kokoro/avatar preparation, or change models.

For cu121/Linux x86_64/CPython 3.10/glibc >=2.35 only:

```bash
export MUSETALK_DEPENDENCY_INSTALL_MODE=locked-wheelhouse-uv
export MUSETALK_DEPENDENCY_CONTRACT=/workspace/MuseTalk/configs/dependencies/rtx3090-cp310-cu121-screening.json
export MUSETALK_DEPENDENCY_CACHE=/workspace/.cache/musetalk-dependencies
export MUSETALK_DEPENDENCY_WORKERS=4

# PRIVATE_GET_TICKET_FD is a protected, already-open descriptor supplied by EC2.
# Its JSON contains {manifest: <read ticket>, wheels: {sha256: <read ticket>}}.
# Do not paste tickets/credentials into the template, command line, or Git.
export MUSETALK_DEPENDENCY_TICKET_FD="$PRIVATE_GET_TICKET_FD"

bash scripts/install_musetalk.sh \
  --venv /workspace/.venvs/musetalk_locked_candidate \
  --matrix cu121 --with-kokoro --with-avatar-prep
```

This is a usage example, **not a complete launch command**: EC2 must issue the
short-lived private read envelope first, using the existing approved role.
The published objects remain private. Large CUDA wheels come from their exact
reviewed original CDNs; only the manifest and four small built wheels use S3.

The manifest digest is pinned independently in reviewed configuration. Every
wheel's size/hash is verified before offline installation. The uv binary is
version/hash-pinned too. Requests/auth fields are not saved in locks or receipts.

## Timing and correctness boundaries

Locked acquisition, common venv/bootstrap setup, tool delivery, bytecode,
installation and inventory/package-consistency checks all belong to the measured
dependency phase. The canonical import/native/GPU self-test, weights and API
readiness remain visible later; skipping them does not qualify as a speedup.

Full-stack is the equivalent baseline. `server` and `server+kokoro` locks exist,
but unsupported combinations (avatar-prep without Kokoro, legacy INT8, cu128)
must retain source-pip or get their own reviewed contract. No broad package sync,
system-site-package inheritance or version relaxation is used.

## Reuse, failure and rollback

- Keep the venv/cache **outside** the replaceable source checkout. `--clean` is
  permitted only when the venv path is absent; it refuses an existing directory
  or dangling symlink. Never use a boot script that wipes those retained paths.
  When integrating `vast_onstart.sh`, use `SETUP_CLEAN=0` for normal operation:
  the install-state check automatically adds `--clean` for an absent venv. The
  old template's unconditional `SETUP_CLEAN=1` is suitable only for this fresh
  canary, not a reusable locked worker. No production template was edited here.
- An unchanged accepted environment is checked against dependency-input hashes,
  full installed inventory and the repo's read-only state checker, before any
  private ticket is read or package network request occurs. Source-only changes
  do not alter dependency identity. Canonical locked checks also require the
  existing trusted manifest cache and accepted owner/group marker; they never
  create a cache or treat missing locked configuration as a valid source check.
- A missing/broken/foreign venv or changed contract is not a cache hit. An install
  marker alone cannot override conflicting package/module facts.
- The standalone helper supports explicit `--repair` for **missing packages** in
  an owned, inactive matching environment. It refuses version/ABI changes and
  running environments; a new path preserves the old environment for rollback.
  The canonical wrapper does not silently enable repair or source fallback.
- A per-venv lock prevents simultaneous installers. Partial downloads are not
  published; auth/hash failures do not silently switch to another package source.
- Rollback: select `MUSETALK_DEPENDENCY_INSTALL_MODE=source-pip` and the retained
  known-good source environment/artifact, rather than deleting the only working
  venv. No production template or autoscaler default has been switched here.

Same-host application-output/performance, equal cold-JIT comparison, Linux failure
tests and canonical/warm-onstart passed. Fresh-host confirmation of the final
hardening and the one-worker rollout gate are documented separately in progress.
A same-host local-wheel or restart result is not a fresh-scale-out measurement.

Official background: [pip secure installs](https://pip.pypa.io/en/stable/topics/secure-installs/)
and [uv's pip compatibility differences](https://docs.astral.sh/uv/pip/compatibility/).
