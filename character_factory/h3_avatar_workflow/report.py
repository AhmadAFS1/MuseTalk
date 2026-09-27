"""Build a resumable batch gallery and measured findings without hiding failures."""
from pathlib import Path
import argparse,html,json,os
from common import HERE,dump,read,sha,stamp

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();root=a.output.resolve()
    batch=read(root/'batch.json');rows=[];cards=[];details=[];tests=[];counts={s:0 for s in ('portraits','sources','renders','validated')};audit={}
    review=read(root/'visual_review.json') if (root/'visual_review.json').exists() else {}
    for specpath in batch['specs']:
        spec=read(specpath);folder=Path(spec['output']);key=spec['id'];label=spec['label'];counts['portraits']+=1
        records={name:read(folder/f'{name}.json') if (folder/f'{name}.json').exists() else {} for name in ('h3','render','validation','metrics')}
        h3,render,valid,metrics=(records[n] for n in ('h3','render','validation','metrics'))
        for name,r in [('sources',h3),('renders',render),('validated',valid)]:counts[name]+=r.get('status')=='complete'
        state='Portrait ready'
        if h3:state='H3 '+h3.get('status','pending')
        if render:state='Rendered; validation pending'
        if valid.get('status')=='complete':state='Technical checks passed; visual review pending' if valid['technical_pass'] else 'Technical checks failed; inspect evidence'
        observation=review.get('subjects',{}).get(key,{})
        if observation:state+=' — '+observation['status']
        def link(name,text):return f'[{text}]({key}/{name})' if (folder/name).exists() else 'Pending'
        rows.append(f'| {label} | {link("portrait.png","Portrait")} | {link("source.mp4","H3 base")} | {link("avatar.mp4","Refined avatar")} | {link("comparison.mp4","Compare")} | {state} |')
        body=f'<section><h2>{html.escape(label)}</h2><p>{html.escape(state)}</p><img loading="lazy" src="{key}/portrait.png" alt="{html.escape(label)} base portrait">'
        for name,title in [('source.mp4','H3 base — native audio'),('avatar.mp4','Refined TAESD override — controlled speech'),('comparison.mp4','Source / standard / refined'),('face_comparison.mp4','Magnified face comparison')]:
            if (folder/name).exists():body+=f'<details><summary>{title}</summary><video controls preload="none" src="{key}/{name}"></video><p><a href="{key}/{name}">Open video</a></p></details>'
        for name,title in [('contact.jpg','Sampled faces'),('consecutive.jpg','Consecutive frames'),('mask_contact.jpg','Masks and jaws')]:
            if (folder/name).exists():body+=f'<a href="{key}/{name}">{title}</a> · '
        body+='</section>';cards.append(body)
        if valid.get('status')=='complete':
            g=metrics['geometry'];d=dict(id=key,label=label,h3_seconds=h3['generation_seconds'],warm_render_fps=render['warm_render_fps'],
                technical_pass=valid['technical_pass'],frames=240,source_endpoint_matches_avatar=valid['endpoint_matches_source'],
                generated_chin_error_standard=g['standard']['target_chin_abs_error_px']['mean'],
                generated_chin_error_refined=g['refined']['target_chin_abs_error_px']['mean'],
                jaw_step_standard=g['standard']['source_relative_jaw_step_percent_eye_span'],
                jaw_step_refined=g['refined']['source_relative_jaw_step_percent_eye_span'],
                texture_ratio_standard=metrics['texture_retention_ratio']['standard'],texture_ratio_refined=metrics['texture_retention_ratio']['refined'])
            details.append(d)
            if observation:d['visual_review']=observation
            pixels=read(folder/'pixel_checks.json');prep=read(folder/'preparation.json')
            tests.append(dict(id=key,passed=valid['technical_pass'],frames=pixels['frames'],
                protected_lip_max_rgb_difference=pixels['protected_lip_max_rgb_difference'],
                reference_max_rgb_difference=pixels['reference_max_rgb_difference'],
                minimum_jacobian=pixels['minimum_jacobian'],decoded_media=len(valid['media']),
                shared_exact_endpoints=valid['endpoint_matches_source'],native_prepared_frames=prep['frames']))
        for name in ('spec.json','portrait.png','h3.json','audio.json','preparation.json','tracking.json','render.json','validation.json','metrics.json'):
            if (folder/name).exists():audit[f'{key}/{name}']=sha(folder/name)
    complete=counts['validated']==counts['portraits']
    title='Avatar diversity validation — accepted H3 / TAESD workflow'
    doc=[f'# {title}','',f'Updated {stamp()}. Batch status: **{"complete" if complete else "in progress"}**. '+', '.join(f'{n} {k}' for k,n in counts.items())+'.','',
        f"{counts['portraits']} fictional adult identities are included in this batch. The creation briefs vary appearance; their labels are not demographic classifications inferred from images. This small batch is a practical stress test, not a representative study of demographic groups.",'',
        '[Open video gallery](index.html). The initial Japanese and Latina clips remain the user-reviewed baseline. New identities require their own visual judgment.','',
        (review.get('summary','')+' [Visual findings](VISUAL_REVIEW.md).' if review else 'Visual observations will be recorded after inspection.'),'',
        '| Character | Base image | H3 source | New avatar | Three-way comparison | Status |','| --- | --- | --- | --- | --- | --- |',*rows,'',
        '## What is held constant','',
        'Built-in imagegen portraits; navy/home-office framing, relaxed subtly lowered head and closed resting lips. The accepted expressive H3 motion prompt is retained, with identity descriptors adapted. H3: seed42, 8 steps, CFG1, 512×896, native243→240 frames at24FPS, the same first/last portrait and exact shared-anchor packaging. MuseTalk: fresh native FP16 encoding and source-specific masks, TensorRT FP16 UNet, compiled TAESD decoder, full chin alignment and the accepted refined mask. Female and male tests use the same continuous speech text with corresponding Kokoro voices; comparisons within each avatar use exactly the same audio and generated predictions.','',
        'Three-way videos show **H3 source / standard TAESD blend / refined TAESD chin alignment**. Replacement speech plays for all panels, so the independently generated H3 mouth is not expected to match that track. Open the source separately for native H3 audio. The standard/refined comparison panels use the raw render before endpoint packaging; open the separate refined avatar for the shared-anchor delivery.','',
        '## Measurements','',
        'Interior frames 24–215. Chin error is absolute vertical displacement from the separately tracked generated chin. Jaw step measures changes in source-relative jaw landmarks. These are diagnostic proxies, not perceptual quality or realism scores. High-pass texture ratios include motion/geometry effects and do not count beard hairs.','',
        '| Character | Chin error px, standard → refined | Jaw step % eye spacing, standard → refined | Texture/source ratio, standard → refined | Warm render FPS |',
        '| --- | ---: | ---: | ---: | ---: |']
    for d in details:doc.append(f"| {d['label']} | {d['generated_chin_error_standard']:.2f} → {d['generated_chin_error_refined']:.2f} | {d['jaw_step_standard']:.3f} → {d['jaw_step_refined']:.3f} | {d['texture_ratio_standard']:.2f} → {d['texture_ratio_refined']:.2f} | {d['warm_render_fps']:.1f} |")
    doc += ['',
        'Warm FPS is one full run per identity, including fresh model outputs, transfer, finite-output checks, per-frame tracking and correction. Loading, source preparation, TTS/audio features, validation, encoding and delivery are excluded. This is not the earlier three-repeat controlled speed benchmark or measured WebRTC FPS.','',
        '## Validation and visual review','',
        'Each completed validation record contains full FFmpeg decode checks on six media files, 240 frames at24FPS, exact first/last RGB equality and the same anchor across source/override. Render checks compare the optimized warp with its full reference, test protected lip pixels and require positive warp Jacobians. The reusable chin module separately reproduced all480 accepted Japanese/Latina raw frames exactly.','',
        'Inspect `contact.jpg`, `consecutive.jpg` and `mask_contact.jpg` in each avatar directory. Mask views use cyan for the original source jaw, magenta for the generated target, yellow for the warped 50% blend boundary and grayscale for alpha. A blend boundary is not an anatomical contour and should not be expected to coincide with either jaw.','',
        'The full beard, short beard and moustache/goatee cases explicitly test whether MuseTalk changes facial hair near the mouth or cheeks. Retaining the source chin can help that region, but it cannot guarantee hair inside the generated speech area. Technical passes alone do not establish beard preservation, seamlessness or universal avatar quality.','',
        '## Reuse','',
        f'[Workflow guide]({os.path.relpath(HERE/"WORKFLOW.md",root)}), [runner]({os.path.relpath(HERE/"create_avatar.py",root)}), [exact batch prompts/config]({os.path.relpath(Path(batch["config"]),root)}), [baseline pixel parity]({os.path.relpath(HERE/"baseline_parity.json",root)}).','',
        'Portrait originals were generated through the built-in imagegen tool and copied into the workspace. Each avatar stores its complete portrait prompt, H3 prompt, source hash, model graph, speech record, prepared cache and stage logs. Completed stages verify artifact hashes before reuse; changed inputs require a new output directory. No existing avatar bank, model checkpoint or earlier experiment was deleted or replaced.','']
    if (root/'musetalk_review_reel.mp4').exists():doc+=['[Watch all six MuseTalk tests in one review reel](musetalk_review_reel.mp4). [Per-avatar test videos and masks](MUSETALK_TESTS.md).','']
    if (root/'VISUAL_REVIEW.md').exists():doc+=['[Frame-level visual observations](VISUAL_REVIEW.md).','']
    if tests:
        doc+=['## Recorded technical test results','',
            '| Avatar | Frames | Protected lip max difference | Reference max difference | Minimum Jacobian | Full media decodes | Shared exact endpoints | Result |',
            '| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |']
        for t in tests:doc.append(f"| {t['id']} | {t['frames']} | {t['protected_lip_max_rgb_difference']} | {t['reference_max_rgb_difference']} | {t['minimum_jacobian']:.3f} | {t['decoded_media']} | {t['shared_exact_endpoints']} | {'Pass' if t['passed'] else 'Fail'} |")
        doc+=['','Pixel differences are measured before encoding and endpoint packaging. A minimum Jacobian above 0.25 is required to avoid the rejected compressed/folded mapping. Technical passes do not constitute user visual acceptance.','']
    (root/'README.md').write_text('\n'.join(doc))
    reel=(root/'musetalk_review_reel.mp4').exists()
    (root/'index.html').write_text('<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{title}</title><style>body{{font:16px system-ui;background:#151719;color:#eee;max-width:1300px;margin:auto;padding:24px}}a{{color:#91caff}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:24px}}section{{background:#22262a;padding:18px;border-radius:12px}}img{{width:100%;max-height:430px;object-fit:contain}}video{{width:100%;background:black}}summary{{cursor:pointer;padding:12px 0}}p{{line-height:1.5}}</style>'
        f'<h1>{title}</h1><p>{html.escape(", ".join(f"{n} {k}" for k,n in counts.items()))}. <a href="README.md">Findings and tests</a></p>'
        +(f'<p>{html.escape(review["summary"])} <a href="VISUAL_REVIEW.md">Visual findings</a></p>' if review else '')
        +('<p><a href="MUSETALK_TESTS.md">All MuseTalk tests and masks</a></p><video controls preload="none" src="musetalk_review_reel.mp4" style="max-width:100%"></video>' if reel else '')
        +'<div class="grid">'+''.join(cards)+'</div>'
        '<script>document.addEventListener("play",e=>{if(e.target.tagName==="VIDEO")document.querySelectorAll("video").forEach(v=>{if(v!==e.target)v.pause()})},true)</script>')
    dump(root/'summary.json',dict(status='complete' if complete else 'in_progress',counts=counts,subjects=details,tests=tests,updated_utc=stamp()))
    for name in ('visual_review.json','VISUAL_REVIEW.md','temporal_review.jpg','MUSETALK_TESTS.md','musetalk_review_reel.mp4','musetalk_review_reel_test.json'):
        if (root/name).exists():audit[name]=sha(root/name)
    dump(root/'audit.json',dict(status='snapshot',updated_utc=stamp(),files=audit))
    print('REPORT',root/'index.html',counts,flush=True)

if __name__=='__main__':main()
