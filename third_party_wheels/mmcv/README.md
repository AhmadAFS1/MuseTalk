# Repo-Local `mmcv` Wheels

The full-stack installer checks this directory before trying the OpenMMLab
wheel index or falling back to a local source build.

Current behavior:

- `scripts/setup_trt_experiment_env.sh` looks here first for
  `mmcv-${MMCV_VERSION}-*.whl`
- if a matching wheel exists here, the installer uses it directly
- if no local wheel exists and the current environment has to source-build
  `mmcv`, the installer now attempts to save the built wheel here for future
  reuse

Expected current wheel shape for the validated full-stack env:

- `mmcv-2.1.0-cp310-cp310-linux_x86_64.whl`

Important compatibility note:

- only reuse a wheel when the environment still matches closely enough
- Python: `3.10`
- torch runtime family: `2.5.1+cu121`
- CUDA toolkit/runtime family: `12.1`
- platform: Linux x86_64

If you want to point the installer at a different wheel location, set either:

- `MMCV_LOCAL_WHEEL_DIR=/path/to/wheels`
- `MMCV_LOCAL_WHEEL_PATH=/path/to/mmcv-2.1.0-....whl`
