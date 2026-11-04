"""Tiny CPU topology smoke: exact BK mappings, four training steps and a dropped-process resume."""
import gc
import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from diffusers import UNet2DConditionModel

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'scripts/ios'))
import export_small_unets as e
import distill_small_unet as d

torch.set_num_threads(1)
torch.manual_seed(12)
with tempfile.TemporaryDirectory(prefix='musetalk-distill-tiny-') as directory:
    root = Path(directory)
    ud = root / 'models/musetalkV15'
    ud.mkdir(parents=True)
    cfg = dict(sample_size=32, in_channels=8, out_channels=4, cross_attention_dim=384,
               layers_per_block=2, block_out_channels=[8, 16, 32, 32], norm_num_groups=4, attention_head_dim=4,
               down_block_types=['CrossAttnDownBlock2D']*3+['DownBlock2D'],
               up_block_types=['UpBlock2D']+['CrossAttnUpBlock2D']*3)
    (ud/'musetalk.json').write_text(json.dumps(cfg))
    full = UNet2DConditionModel(**cfg)
    torch.save(full.state_dict(), ud/'unet.pth')
    e.VARIANTS['bk_tiny']['block_out_channels'] = [8, 16, 32]
    for variant in ('bk_base', 'bk_small', 'bk_tiny'):
        student, _ = e.build(variant, ud, processor='train', full=full)
        shift = len(full.up_blocks)-len(student.unet.up_blocks)
        for key, value in student.unet.state_dict().items():
            assert torch.equal(value, full.state_dict()[e.teacher_key(key, shift)]), key
        del student
    del full
    gc.collect()
    hb = root/'harness'
    hb.mkdir()
    rng = np.random.default_rng(1)
    for name,shape in [('latents',(128,8,32,32)),('audio',(128,50,384)),('ref',(128,4,32,32))]:
        rng.standard_normal(shape).astype(np.float16).tofile(hb/f'unet_{name}_f16.bin')
    (hb/'meta.json').write_text(json.dumps({'unet':{'files':[{'avatar':'gate_only'}]}}))
    avatarpaths=[]
    for aid in ('roster_a','roster_b','roster_c'):
        p=root/aid/'latents_f16.bin'
        p.parent.mkdir()
        rng.standard_normal((2,8,32,32)).astype(np.float16).tofile(p)
        avatarpaths.append(str(p))
    # Guard avatar identity and gate alias before touching any model.
    glat=torch.randn(1,8,32,32).half()
    for aid in ('gate_only',next(iter(d.REVIEW_AVATARS))):
        try:
            d.validate_data({aid:torch.ones(1,8,32,32)},glat,hb/'meta.json')
        except SystemExit:
            pass
        else:
            raise AssertionError('forbidden identity accepted')
    try:
        d.validate_data({'alias':glat},glat,hb/'meta.json')
    except SystemExit:
        pass
    else:
        raise AssertionError('gate alias accepted')
    audio=root/'train.wav'
    audio.write_bytes(b'tiny deterministic audio cache fixture')
    out=root/'run'
    out.mkdir()
    torch.save({'files':[[audio.name,d.file_digest(audio)]],'windows':torch.randn(4,50,384).half()}, out/'audio_windows.pt')
    argv=['distill_small_unet.py','--latents',','.join(avatarpaths),'--audio',str(audio),'--out',str(out),'--variant','bk_tiny','--val-count','1','--expected-train-avatars','2','--expected-val-avatars','1','--val-pairs','2','--gate-frames','1','--batch','1','--steps','4','--eval-every','2','--log-every','1','--warmup','1','--ckpt-minutes','0.00000001','--device','cpu']
    real_save=d.save
    def interrupt_after_checkpoint(obj,path):
        real_save(obj,path)
        if path.name=='ckpt.pt' and obj['step']==1:
            raise InterruptedError('simulate dropped process after atomic checkpoint')
    with patch.object(d,'REPO_ROOT',root),patch.object(d,'HB',hb),patch.object(sys,'argv',argv),patch.object(d,'save',interrupt_after_checkpoint):
        try:
            d.main()
        except InterruptedError:
            pass
        else:
            raise AssertionError('checkpoint did not interrupt')
    gc.collect()
    with patch.object(d,'REPO_ROOT',root),patch.object(d,'HB',hb),patch.object(sys,'argv',argv+['--resume']):
        d.main()
    ck=torch.load(out/'ckpt.pt',map_location='cpu',weights_only=True)
    assert ck['step']==4
    assert ck['manifest']['train_ids'] and len(ck['manifest']['train_ids'])==2
    e.build('bk_tiny',ud,processor='train',weights=out/'bk_tiny_best.pt')
    # Resume must recover optimizer/RNG/scheduler state, not just the weights.
    uninterrupted=root/'uninterrupted'
    uninterrupted.mkdir()
    (uninterrupted/'audio_windows.pt').write_bytes((out/'audio_windows.pt').read_bytes())
    control_argv=argv.copy()
    control_argv[control_argv.index('--out')+1]=str(uninterrupted)
    with patch.object(d,'REPO_ROOT',root),patch.object(d,'HB',hb),patch.object(sys,'argv',control_argv):
        d.main()
    actual=torch.load(out/'bk_tiny_last.pt',map_location='cpu',weights_only=True)
    expected=torch.load(uninterrupted/'bk_tiny_last.pt',map_location='cpu',weights_only=True)
    assert all(torch.equal(actual[k],expected[k]) for k in actual), 'resume diverged from uninterrupted control'
    # Trained exports must not allocate/profile the full teacher on the Mac.
    export_argv=['export_small_unets.py','--variants','bk_tiny','--weights',str(out/'bk_tiny_best.pt'),
                 '--tag','test','--unet-dir',str(ud),'--out-dir',str(out),'--results',str(out/'export.json')]
    with (patch.object(sys,'argv',export_argv),patch.object(e,'load_full',side_effect=AssertionError('loaded full')),
         patch.object(e,'gflops',side_effect=AssertionError('profiled full')),
         patch.object(e,'export',return_value=out/'unet_bk_tiny_test_int8w_b1.mlpackage')):
        e.main()
    evals=[v for v in map(json.loads,(out/'log.jsonl').read_text().splitlines()) if 'gate_mae' in v]
    assert [v['step'] for v in evals]==[0,2,4]
    assert all(np.isfinite(v['gate_mae']) for v in evals)
    print('PASS: exact BK initialization; held-out guards; 4 CPU steps; bit-identical interrupted resume; finite gate at eval0/2/4; strict best load; student-only trained export')
