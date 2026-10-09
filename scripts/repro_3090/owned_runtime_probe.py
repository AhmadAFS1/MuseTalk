"""Actual runtime/device facts; explicit owned guard/watch required by caller."""
import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--enable', action='store_true')
    a = p.parse_args(argv)
    lease = os.environ.get('BOX_GUARD_LEASE_FILE')
    if not a.enable or not lease or not Path(lease + '.holder').is_file():
        raise ValueError('explicit canonical guard required')
    import torch
    import tensorrt
    import torch_tensorrt
    print('OWNED_RUNTIME ' + json.dumps(dict(torch=torch.__version__, tensorrt=tensorrt.__version__,
        torch_tensorrt=torch_tensorrt.__version__, cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(), visible_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
        gpu=torch.cuda.get_device_name(0), compute_capability='.'.join(map(str,torch.cuda.get_device_capability(0))))), flush=True)


if __name__ == '__main__': main()
