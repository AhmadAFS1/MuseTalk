# MuseTalk requirements

`scripts/install_musetalk.sh` installs the server venv from these files. The repo-root
`requirements.txt` is the upstream MuseTalk list (it pulls tensorflow/gradio). It is **not** the
validated server stack, so never install it into the server venv.

| file | what | installed by |
|---|---|---|
| `server.in` | top-level server requirements (fast / fast300 / legacy recipes) | always |
| `kokoro.in` | local Kokoro TTS (+ pinned `en_core_web_sm` URL) | `--with-kokoro` (default) |
| `legacy-int8.in` | nvidia-modelopt ONNX/QDQ INT8 VAE stack (`MUSETALK_RECIPE=legacy_int8`) | `--with-legacy-int8` (cu121 only) |
| `avatar-prep.in` | mmcv/mmdet/mmpose names (installed step by step, see the file) | `--with-avatar-prep` (cu121 only) |
| `chin-tools.in` | MediaPipe FaceMesh for the offline chin tracker, in its **own** venv | `--with-chin-tools` |
| `constraints-cu121.txt` | exact pins, CUDA 12.1 matrix (sm_50..sm_90), from the validated live venv | matrix `cu121` |
| `constraints-cu128.txt` | exact pins, CUDA 12.8 matrix (Blackwell, cc >= 10.0). Resolves; **untested on hardware** | matrix `cu128` |
| `constraints-chin-tools.txt` | closure of the validated chin-tracker env | chin-tools venv |

The installer runs one pip resolve for all selected groups:

```bash
<venv>/bin/python -m pip install -r requirements/server.in [-r requirements/kokoro.in] [-r requirements/legacy-int8.in] \
  -c requirements/constraints-<matrix>.txt \
  --extra-index-url https://download.pytorch.org/whl/<matrix> --extra-index-url https://pypi.nvidia.com
```

`--matrix auto` picks `cu128` only when the selected GPU has compute capability >= 10.0. With no GPU
it picks `cu121`, or keeps the matrix of an existing venv. The installer ignores
`PYTORCH_INDEX_URL` from the base image on purpose, because the matrix chooses the index.
`bash scripts/install_musetalk.sh --plan` prints the exact command without running it.

## Pinning rules worth knowing

- **`torch_tensorrt===2.5.0` (cu121).** Here `===` is arbitrary equality, not a typo. Under PEP 440
  `==2.5.0` also matches `2.5.0+cu121`, and pip then prefers that local build from
  download.pytorch.org. That is a different binary from the PyPI 2.5.0 build that the validated venv
  and its engines use. `--check` compares with the same semantics.
- **`setuptools==60.2.0`.** This is the validated venv's version (openxlab pins `~=60.2.0`), and
  `triton/runtime/build.py` imports it. `pip freeze` omits setuptools, so the pin is added by hand.
- **Native VP8** needs exactly `aiortc==1.14.0`, `av==16.1.0` and `cffi==2.1.1`. All three
  matrices keep them.
- **cu128.** The torch/torchvision/torchaudio/torch_tensorrt `+cu128` pins come from the contract.
  tensorrt* 10.9.0.34, triton 3.3.1, sympy, networkx and nvidia-* are the exact versions pip chose in
  the recorded dry-run. They are pinned so that later installs reproduce that run.
- **Kokoro.** Weights are not pip packages. `download_weights.sh` (`DOWNLOAD_KOKORO_WEIGHTS=auto|1|0`)
  pre-caches hexgrad/Kokoro-82M at a pinned revision, together with the default voices.

## Regenerating

1. **cu121 from a validated venv.** Freeze the venv, then drop its URL/file lines (`en_core_web_sm`
   and `mmcv`). Re-add the header comments and the `setuptools==` / `torch_tensorrt===` lines:

   ```bash
   /workspace/.venvs/musetalk_trt_stagewise/bin/python -m pip freeze > /tmp/freeze.txt
   grep -v -E '^(en_core_web_sm|mmcv) @ ' /tmp/freeze.txt   # body of constraints-cu121.txt
   ```

2. **cu128.** Start from the cu121 body without the
   torch/torchvision/torchaudio/triton/torch_tensorrt/tensorrt*/nvidia-*/sympy/networkx lines. Add the
   cu128 pins, run the dry-run below, and pin what it selects for tensorrt*/triton/sympy/networkx/nvidia-*.

3. **Chin tools.** Read the dist-info names of the validated tracker env (read only, never modify
   it). Take the closure of `mediapipe`, `numpy` and `opencv-contrib-python`.

4. **Verify** each set in a scratch venv. Never use the live one:

   ```bash
   python3.10 -m venv /tmp/resolve && /tmp/resolve/bin/python -m pip install -q pip==26.2.1
   /tmp/resolve/bin/python -m pip install --dry-run --ignore-installed --no-cache-dir --report /tmp/cu121.json \
     -r requirements/server.in -r requirements/kokoro.in -r requirements/legacy-int8.in \
     -c requirements/constraints-cu121.txt \
     --extra-index-url https://download.pytorch.org/whl/cu121 --extra-index-url https://pypi.nvidia.com
   /tmp/resolve/bin/python -m pip install --dry-run --ignore-installed --no-cache-dir --report /tmp/cu128.json \
     -r requirements/server.in -r requirements/kokoro.in -c requirements/constraints-cu128.txt \
     --extra-index-url https://download.pytorch.org/whl/cu128 --extra-index-url https://pypi.nvidia.com
   /tmp/resolve/bin/python -m pip install --dry-run --ignore-installed --report /tmp/chin.json \
     -r requirements/chin-tools.in -c requirements/constraints-chin-tools.txt
   rm -rf /tmp/resolve
   ```

   A dry-run still downloads every wheel whose index has no PEP 658 metadata. Expect about 5-7 GB
   of transient disk and a large burst of page cache. On a shared box, run it when the box is quiet.

5. Record the results in `docs/startup_rework_20260928/impl/installer/resolve_*.json`, then run
   `bash scripts/test_install_musetalk.sh`. It checks, among other things, that every top-level name
   has a pin in both matrices.

The results from 2026-09-28 are in `docs/startup_rework_20260928/impl/installer/resolve_{cu121,cu128,chin_tools}.json`.
All three sets resolve. cu128 is not tested on hardware.
