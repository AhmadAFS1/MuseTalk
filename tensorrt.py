"""
Compatibility shim for environments where NVIDIA's wheels provide the
`tensorrt_bindings` module but not the legacy top-level `tensorrt` module.

Torch-TensorRT 1.4 still imports `tensorrt`, so re-exporting the bindings
module here lets the current MuseTalk backend scripts run inside the existing
/content/py310 environment without changing the wider torch stack.
"""

from tensorrt_bindings import *  # noqa: F401,F403
from tensorrt_bindings import __version__  # noqa: F401
