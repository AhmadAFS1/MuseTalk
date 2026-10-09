"""CPU-only paired startup/probe report; no full boot or release acceptance."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

SEQUENCE = ['0','1','1','0']
ACKS = {'QUIET_INITIALIZATION_ACK','LIVE_CONTEXT_MEMBERSHIP_BOUND','TARGET_FINISHED_ACK'}
SHAPES = {'unet':[16,4,32,32], 'decoded':[16,3,256,256], 'retained_vae_encoder':[1,4,32,32]}


def require(ok,reason):
    if not ok: raise ValueError(reason)


def summarize(trials,watches):
    require(len(trials)==len(watches)==4 and [r['skip_eager_unet'] for r in trials]==SEQUENCE,'complete preregistered0/1/1/0 required')
    for row,watch in zip(trials,watches):
        require(row['status']=='PASS_DIAGNOSTIC' and row['schema']=='owned_fresh_process_startup_model_probe_v1','failed/incomplete trial')
        require(row['source_pins']==trials[0]['source_pins'] and row['input_capture_sha256']==trials[0]['input_capture_sha256'],'different source or input')
        require(row['model_startup_profile']['skip_eager_unet']==(row['skip_eager_unet']=='1')
                and row['model_startup_profile']['avatar_vae_encoder_retained'] is True,'wrong actual treatment or encoder')
        require(watch[-1]['status']=='COMPLETE' and watch[-1]['returncode']==0
                and ACKS <= {r['status'] for r in watch},'authenticated ownership incomplete')
        require(watch[0]['status']=='ENROLLMENT_PENDING' and watch[0]['target_sha256']==watches[0][0]['target_sha256'],'different owned child')
        for field in ('imports_s','manager_init_s','import_init_wall_s'):
            require(type(row[field]) in (float,int) and math.isfinite(row[field]) and row[field]>0,'invalid timing')
        for field,shape in SHAPES.items():
            require(row[field]==trials[0][field] and row[field]['shape']==shape
                    and row[field]['dtype']=='torch.float16' and row[field]['nonfinite_count']==0,'nonfinite or changed output')
    baseline=[r['manager_init_s'] for r in trials if r['skip_eager_unet']=='0']
    treatment=[r['manager_init_s'] for r in trials if r['skip_eager_unet']=='1']
    old,new=statistics.mean(baseline),statistics.mean(treatment)
    return dict(schema='fixed_startup_model_pair_summary_v1',status='PASS_LOCAL_MODEL_PROBE_PARITY',
        sequence=SEQUENCE,manager_init_s=[r['manager_init_s'] for r in trials],
        import_init_wall_s=[r['import_init_wall_s'] for r in trials],
        baseline_mean_manager_init_s=old,treatment_mean_manager_init_s=new,
        mean_manager_init_saving_s=old-new,mean_manager_init_reduction_percent=(old-new)/old*100,
        all_three_output_hashes_equal=True,retained_vae_encoder=True,
        peak_torch_allocated_vram_bytes=[r['peak_allocated_vram_bytes'] for r in trials],
        ownership='All four exact same child enrolled, live-context-bound and COMPLETE0',
        scope='Fresh local processes on same owned A5; warm filesystem cache, fixed one real bs8 capture padded16 and seeded diagnostic encoder input',
        limitations=['NOT full avatars/prepare or local WebRTC validation','NOT cold image pull, provider/secrets/registration/avatar-ready or request-to-usable-call',
                     'Original failed pre-retry diagnostic remains preserved and is not part of this four-trial pair',
                     'No selected quality-approved400FPS release; tested native-v1 artifacts remain quality-rejected diagnostics'],
        startup_acceptance=False,production_or_image_enabled=False,release_ready=False)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    p.add_argument('--trial',action='append',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(argv)
    require(not a.out.exists(),'fresh summary required')
    trials,watches,evidence=[],[],[]
    for path in a.trial:
        require(path.is_absolute() and not path.is_symlink(),'absolute nonsymlink trial required')
        watch=path.with_suffix('.watch.jsonl')
        trials.append(json.loads(path.read_text()))
        watches.append([json.loads(line) for line in watch.read_text().splitlines()])
        evidence.extend(dict(path=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in (path,watch))
    result=summarize(trials,watches);result['evidence']=evidence
    with a.out.open('x') as output:
        json.dump(result,output,indent=2,allow_nan=False);output.write('\n')
    print(json.dumps({k:result[k] for k in ('status','mean_manager_init_saving_s','release_ready')}))


if __name__=='__main__': main()
