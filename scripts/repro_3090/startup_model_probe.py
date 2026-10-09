"""Default-off fresh-process model-startup diagnostic, not EC2 readiness.

Compare 0/1/1/0 only with the same original native-v1 artifacts. Exercise a
canonical real bs8 capture padded to16 and the retained avatar VAE encoder.
No HTTP server, registration, cloud access or production preparation writes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path('/workspace/MuseTalk')
PINS = {
    'scripts/avatar_manager_parallel.py': '202ba063d3a1e089243f0199f37f889f059c81e2dbdda3fe06f4e5b7dbc3dc03',
    'scripts/trt_runtime.py': '29841be8cc47f2cc7b442315fd447f3eeed6d50f717ef83c41e4e0efc6b6e5af',
    'musetalk/models/vae.py': '831dcbd8f389e9a77d8169674cb966ef52725ccc6bdacbceed81c0ee0d46966c',
    'models/tensorrt_unet_stagewise_sm86_r5_v1/bs16/manifest.json': 'f66b46ca38d0e34af69ee5c01be52d93cc3d2f1ba3ac7b8a68ae0426c3658316',
    'calibration/unet_multi_avatar_20260928/unet_io_000001_bs8_pid3537045.pt': 'e8d12ebc968722726820e5d79c0f223de76b4cf8b54ad665e3cabee1a95cb41c',
}
CAPTURE = ROOT / 'calibration/unet_multi_avatar_20260928/unet_io_000001_bs8_pid3537045.pt'


def require(value, reason):
    if not value:
        raise ValueError(reason)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flags(mode):
    require(mode in ('0', '1'), 'exact startup mode required')
    values = {}
    for line in (ROOT / 'scripts/repro_3090/profiles/native.env').read_text().splitlines():
        if line and not line.startswith('#'):
            key, value = line.split('=', 1); values[key] = value
    values.update(MUSETALK_SKIP_EAGER_UNET=mode, MUSETALK_UNET_STAGEWISE_PROBE_CHECK='1',
        MUSETALK_UNET_STAGEWISE_PROBE_TOL='0', MUSETALK_FREE_EAGER_UNET='1', MUSETALK_COMPILE='0',
        MUSETALK_UNET_STAGEWISE_CACHE_DIR=str(ROOT/'models/tensorrt_unet_stagewise_sm86_r5_v1'),
        MUSETALK_TAESD_TRT_DIR=str(ROOT/'models/taesd/trt_native_sm86_r5_v1'),
        AVATAR_S3_ENABLED='0', LINGUA_CONTROL_PLANE_ENABLED='0', LINGUA_WORKER_CALLBACK_REQUIRED='0',
        OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--enable', action='store_true')
    parser.add_argument('--skip-eager', choices=('0', '1'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    a = parser.parse_args(argv)
    lease = os.environ.get('BOX_GUARD_LEASE_FILE')
    require(a.enable and lease == '/workspace/.gpu_lease' and Path(lease+'.holder').is_file(), 'explicit canonical guard required')
    require(a.out.parent == ROOT/'docs/fps_comparisons/rtx3090_r5_20261008/startup'
        and not a.out.exists() and not any(p.is_symlink() for p in (a.out,*a.out.parents)), 'fresh scoped output required')
    for name, digest in PINS.items():
        require(sha(ROOT/name) == digest, 'fixed source or input changed: '+name)
    os.environ.update(flags(a.skip_eager))
    sys.path.insert(0,str(ROOT))
    started = time.monotonic()
    import torch
    from scripts.avatar_manager_parallel import ParallelAvatarManager
    imports_s = time.monotonic()-started
    args = SimpleNamespace(version='v15',gpu_id=0,vae_type='sd-vae',
        unet_config='./models/musetalkV15/musetalk.json', unet_model_path='./models/musetalkV15/unet.pth',
        whisper_dir='./models/whisper',left_cheek_width=90,right_cheek_width=90)
    init_start = time.monotonic()
    manager = ParallelAvatarManager(args,max_concurrent_inferences=1)
    try:
        torch.cuda.synchronize(); init_s = time.monotonic()-init_start
        require(manager.skip_eager_unet == (a.skip_eager=='1')
            and manager.unet_backend_name == 'tensorrt_unet_stagewise'
            and manager.vae_decode_backend_name != 'pytorch', 'expected actual backends required')
        capture = torch.load(CAPTURE,map_location='cpu',weights_only=False)
        def padded(tensor):
            require(tensor.shape[0]==8,'fixed real bs8 capture required')
            return torch.cat((tensor,tensor),dim=0).to('cuda',torch.float16)
        latent,audio = padded(capture['latent_batch']),padded(capture['audio_feature_batch'])
        with torch.inference_mode():
            predicted = manager.unet.model(latent,manager.timesteps,encoder_hidden_states=audio).sample
            decoded = manager.vae.decode_latents_tensor(predicted)
            # Fixed diagnostic tensor, not a claim of full production prepare.
            torch.manual_seed(20261009)
            image = torch.linspace(-1,1,3*256*256,device='cuda',dtype=torch.float32).to(torch.float16).reshape(1,3,256,256)
            require(bool(torch.isfinite(image).all()), 'diagnostic encoder input nonfinite')
            encoded = manager.vae.encode_latents(image)
            torch.cuda.synchronize()
        def tensor_evidence(value):
            cpu = value.detach().cpu().contiguous()
            return dict(shape=list(cpu.shape),dtype=str(cpu.dtype),
                nonfinite_count=int((~torch.isfinite(cpu)).sum()),sha256=hashlib.sha256(cpu.numpy().tobytes()).hexdigest())
        report = dict(schema='owned_fresh_process_startup_model_probe_v1',status='PASS_DIAGNOSTIC',
            skip_eager_unet=a.skip_eager,imports_s=imports_s,manager_init_s=init_s,
            import_init_wall_s=imports_s+init_s,model_startup_profile=manager.model_startup_profile,
            unet=tensor_evidence(predicted),decoded=tensor_evidence(decoded),retained_vae_encoder=tensor_evidence(encoded),
            input_capture_sha256=PINS[str(CAPTURE.relative_to(ROOT))],source_pins=PINS,
            peak_allocated_vram_bytes=torch.cuda.max_memory_allocated(),
            preparation_scope='seeded fixed diagnostic encoder tensor only; NOT full avatars/prepare validation',
            startup_acceptance=False,quality_accepted=False,release_ready=False,
            measurement_scope='Fresh local process; excludes image pull/provider/secrets/HTTP/EC2/avatar readiness/live call')
        if any(report[k]['nonfinite_count'] for k in ('unet','decoded','retained_vae_encoder')):
            report['status'] = 'FAIL_NONFINITE_DIAGNOSTIC_OUTPUT'
        with a.out.open('x') as output:
            json.dump(report,output,indent=2,allow_nan=False);output.write('\n')
        print(json.dumps({k:report[k] for k in ('status','skip_eager_unet','imports_s','manager_init_s')}),flush=True)
        return 0 if report['status']=='PASS_DIAGNOSTIC' else 2
    finally:
        manager.shutdown()


if __name__=='__main__': raise SystemExit(main())
