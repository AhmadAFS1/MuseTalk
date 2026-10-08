"""Re-export one FP16 block with alternative timestep constants; never build/promote an engine."""
import argparse
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import socket
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path('/workspace/MuseTalk')
sys.path.insert(0, str(ROOT))
from scripts import build_unet_stagewise as builder
from scripts import unet_stagewise_trt as sw
import torch

NATIVE_ONNX = 'f299bc5eb705decaf23852c231426db15fbf4fd114879cce42c4a4c6c9304729'
PORTABLE_ONNX = '3a46428a09320be5b615d26e1204bd280f589c957439d1411b197e4d3df589f0'


def tensor_sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def graph_record(raw):
    import onnx
    model = onnx.load_model_from_string(raw)
    initializers = {value.name: (value, hashlib.sha256(value.SerializeToString()).hexdigest())
                    for value in model.graph.initializer}
    structure = [(node.op_type, list(node.input), list(node.output),
                  [a.SerializeToString().hex() for a in node.attribute]) for node in model.graph.node]
    return {'onnx_sha256': hashlib.sha256(raw).hexdigest(), 'onnx_bytes': len(raw),
            'node_count': len(model.graph.node), 'initializer_count': len(initializers),
            'node_structure_sha256': hashlib.sha256(json.dumps(structure).encode()).hexdigest()}, initializers


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert socket.gethostname() == 'a830e00ce20c'
    assert dt.datetime.now(dt.timezone.utc) < dt.datetime(2026, 10, 8, 19, tzinfo=dt.timezone.utc)
    assert args.out.is_absolute() and args.out.parent == ROOT / 'docs/fps_comparisons/rtx3090_r5_20261008/native'
    assert not args.out.exists() and not args.out.is_symlink()
    with socket.socket() as check:
        assert check.connect_ex(('127.0.0.1', 8300)) != 0, 'API still listening'
    assert subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                                   text=True, timeout=10).strip() == '', 'foreign GPU work'
    assert subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'],
             text=True, timeout=10).strip() == 'GPU-5640f670-debe-ec22-1cfb-4b1f63bc1d53'
    for name, expected in (('unet.pth', '7ebf6c98c181e20838e4c0054e96e944ac60d5d692cc01db42839fe11b787007'),
                           ('musetalk.json', '5b6923aee04d71692e0e9846c471e0a4ea07a4f686d39545e472bd4ba17e1b47')):
        assert sw.sha256_file(ROOT / 'models/musetalkV15' / name) == expected, 'frozen model input changed'
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    device = torch.device('cuda:0')
    model = builder.load_eager_unet(device)
    gpu16 = sw.time_embedding_t0(model, 16, device)
    gpu_repeat = sw.time_embedding_t0(model, 16, device)
    gpu1 = sw.time_embedding_t0(model, 1, device)
    cpu_model = SimpleNamespace(time_proj=copy.deepcopy(model.time_proj).cpu(),
                               time_embedding=copy.deepcopy(model.time_embedding).cpu(), dtype=torch.float16)
    cpu16 = sw.time_embedding_t0(cpu_model, 16, torch.device('cpu')).to(device)
    cpu_model.time_embedding.float()
    cpu_model.dtype = torch.float32
    cpu32rounded = sw.time_embedding_t0(cpu_model, 16, torch.device('cpu')).half().to(device)
    variants = [('native_gpu16', gpu16), ('cpu_fp16_16', cpu16),
                ('gpu_fp16_1', gpu1), ('cpu_fp32_with_fp16_weights_rounded', cpu32rounded)]
    data = {'schema': 'owned3090_time_embedding_export_diagnostic_v1', 'status': 'IN_PROGRESS',
            'started_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
            'native_gpu16_repeat_exact': torch.equal(gpu16, gpu_repeat),
            'torch_version': torch.__version__, 'block': 'down0rest', 'precision': 'fp16',
            'recipe': 'srccache batch16; no quantization, learned recipe edits or engine building',
            'variants': [], 'native_quality_status': 'REJECTED_UNCHANGED', 'release_ready': False}
    spec = sw.chain_spec(model, 'srccache')
    wrappers = sw.make_block_wrappers(model, gpu16)
    latent, audio = sw.probe_inputs(16)
    tensors, _ = sw.trace_block_inputs(model, spec, wrappers, latent.to(device), audio.to(device))
    block = next(row for row in spec if row['name'] == 'down0rest')
    arguments = [tensors[key] for key in block['inputs']]
    baseline_initializers = None
    import onnx
    import numpy as np
    for label, embedding in variants:
        wrappers = sw.make_block_wrappers(model, embedding)
        raw = sw.export_block_onnx(wrappers['down0rest'], arguments)
        record, initializers = graph_record(raw)
        diff = (embedding.float() - gpu16.float()).abs()
        record.update(label=label, embedding_sha256=tensor_sha(embedding),
                      embedding_diff_count=int((embedding != gpu16).sum()),
                      embedding_max_abs_diff=float(diff.max()), embedding_mean_abs_diff=float(diff.mean()),
                      matches_frozen_native_onnx=record['onnx_sha256'] == NATIVE_ONNX,
                      matches_frozen_portable_onnx=record['onnx_sha256'] == PORTABLE_ONNX)
        if baseline_initializers is None:
            baseline_initializers = initializers
        else:
            changed = []
            for name in sorted(set(baseline_initializers) | set(initializers)):
                before, after = baseline_initializers.get(name), initializers.get(name)
                if before is None or after is None:
                    changed.append({'name': name, 'status': 'ADDED_OR_REMOVED'})
                elif before[1] != after[1]:
                    a, b = onnx.numpy_helper.to_array(before[0]), onnx.numpy_helper.to_array(after[0])
                    row = {'name': name, 'before_sha256': before[1], 'after_sha256': after[1],
                           'shape_equal': a.shape == b.shape, 'dtype_equal': a.dtype == b.dtype}
                    if a.shape == b.shape and a.dtype == b.dtype:
                        row['different_elements'] = int(np.count_nonzero(a != b))
                        row['max_abs_diff'] = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
                    changed.append(row)
            record['initializer_changes_count'] = len(changed)
            record['initializer_changes'] = changed[:24]
            record['initializer_changes_truncated'] = len(changed) > 24
        data['variants'].append(record)
        print(json.dumps({k: record[k] for k in ('label', 'onnx_sha256', 'embedding_diff_count',
                          'embedding_max_abs_diff', 'matches_frozen_native_onnx', 'matches_frozen_portable_onnx')}), flush=True)
        del raw
    data['status'] = ('PASS_DIAGNOSTIC_NATIVE_GRAPH_REPRODUCED' if data['variants'][0]['matches_frozen_native_onnx']
                      else 'DIAGNOSTIC_ONLY_NATIVE_GRAPH_NOT_REPRODUCED')
    data['finished_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    with args.out.open('x') as output:
        json.dump(data, output, indent=2, allow_nan=False)
        output.write('\n')
    print(json.dumps({'status': data['status'], 'release_ready': False}), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
