"""Full media, endpoint, geometry and facial-hair review evidence."""
from pathlib import Path
import sys,subprocess
import cv2,numpy as np
from common import HERE,check_video,completed,digest,dump,finish,read,sha,spec_arg
from track_stage import track

def stats(values):
    a=np.asarray(values,float)
    return dict(mean=float(a.mean()),median=float(np.median(a)),p95=float(np.percentile(a,95)),max=float(a.max()))

def comparison(target,paths,labels,audio,closeup=False):
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';graph=[]
    for i,label in enumerate(labels):
        crop='crop=312:350:100:260,scale=468:526:flags=lanczos,' if closeup else ''
        graph.append(f'[{i}:v]trim=end_frame=240,setpts=PTS-STARTPTS,{crop}pad=iw:ih+48:0:48:color=0x101010,'
            f"drawtext=fontfile={font}:text='{label}':expansion=none:fontsize=20:fontcolor=white:x=(w-tw)/2:y=12[v{i}]")
    graph.append(''.join(f'[v{i}]' for i in range(len(paths)))+f'hstack=inputs={len(paths)}:shortest=1[v]')
    graph.append(f'[{len(paths)}:a]atrim=duration=10,asetpts=PTS-STARTPTS[a]')
    args=['ffmpeg','-v','error','-xerror','-y']
    for p in paths:args.extend(['-i',str(p)])
    args.extend(['-i',str(audio),'-filter_complex',';'.join(graph),'-map','[v]','-map','[a]',
        '-c:v','libx264','-threads','2','-crf','16','-preset','fast','-pix_fmt','yuv420p','-c:a','aac','-r','24','-t','10','-movflags','+faststart',str(target)])
    subprocess.run(args,check=True)

def tile(f,label):
    cropped=cv2.resize(f[240:650,65:447],(306,328),interpolation=cv2.INTER_CUBIC)
    head=np.full((30,306,3),15,np.uint8);cv2.putText(head,label,(5,21),cv2.FONT_HERSHEY_SIMPLEX,.48,(245,245,245),1,cv2.LINE_AA)
    return np.vstack([head,cropped])

def main():
    spec=spec_arg();out=Path(spec['output']);w=Path(spec['workspace']);cv2.setNumThreads(2)
    sys.path[:0]=[str(w/'MuseTalk'),str(w/'minimax-h3')]
    import chin
    from package_spoken_transition_bank import package
    from certify_endpoints import decoded_endpoints
    signature=digest(dict(spec=spec['signature'],code=sha(__file__),render=sha(out/'render.json'),source=sha(out/'source.mp4')))
    if completed(out/'validation.json',signature):return
    delivery=out/'avatar.mp4'
    if delivery.exists():raise RuntimeError(f'Unrecorded delivery exists: {delivery}; inspect the partial validation before resuming')
    packaging=package(out/'refined_raw.mp4',delivery,out/'anchor.png');dump(out/'packaging.json',packaging)
    proof=decoded_endpoints(delivery);sourceproof=decoded_endpoints(out/'source.mp4')
    exact=proof['first_last_exact'] and sourceproof['first_last_exact'] and proof['first_rgb_sha256']==sourceproof['first_rgb_sha256']
    assert exact,'Shared source/override endpoints differ'
    paths=[out/'source.mp4',out/'standard_raw.mp4',out/'refined_raw.mp4']
    labels=['H3 source','TAESD standard blend','TAESD refined chin 100%']
    comparison(out/'comparison.mp4',paths,labels,out/'speech.wav')
    comparison(out/'face_comparison.mp4',paths,labels,out/'speech.wav',True)
    source=np.load(out/'source_landmarks.npy');generated=np.load(out/'generated_landmarks.npy')
    finals={}
    for name in ('standard','refined'):
        finals[name]=track(out/f'{name}_raw.mp4');np.save(out/f'{name}_landmarks.npy',finals[name])
    geometry={};source_motion=[];roll=[]
    for p in source:
        _,horizontal,_,span=chin.axes(p);source_motion.append(float(np.linalg.norm(p[13]-p[14])/span));roll.append(float(np.degrees(np.arctan2(horizontal[1],horizontal[0]))))
    for name,points in finals.items():
        error=[];length=[];residual=[];source_error=[]
        for p,g,q in zip(source,generated,points):
            center,right,down,span=chin.axes(p)
            error.append(float((q[152]-g[152])@down));length.append(float((q[152]-q[17])@down))
            source_error.append(float(np.linalg.norm(q[152]-p[152])/span*100))
            residual.append((q[chin.JAW]-p[chin.JAW])@np.stack([right,down],axis=1)/span)
        geometry[name]=dict(target_chin_abs_error_px=stats(np.abs(error)[24:216]),
            positive_excess_chin_length_px=stats(np.maximum(error,0)[24:216]),
            lower_lip_to_chin_px=stats(np.asarray(length)[24:216]),source_chin_error_percent_eye_span=stats(np.asarray(source_error)[24:216]),
            source_relative_jaw_step_percent_eye_span=float(np.linalg.norm(np.diff(np.asarray(residual)[24:216],axis=0),axis=-1).mean()*100))
    caps=[cv2.VideoCapture(str(p)) for p in paths];contacts=[];texture={'source':[],'standard':[],'refined':[]};continuous=[];mask_rows=[]
    mask_samples=np.load(out/'mask_samples.npz')
    for i in range(240):
        frames=[]
        for cap in caps:
            ok,f=cap.read();assert ok;frames.append(f)
        if i in (24,48,96,144,192,215):contacts.append(np.hstack([tile(f,f'{label} f{i}') for f,label in zip(frames,('Source','Standard','Refined'))]))
        if str(i) in mask_samples:
            alpha=mask_samples[str(i)];overlay=frames[2].copy()
            contours,_=cv2.findContours((alpha>=128).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay,contours,-1,(0,255,255),1)
            for pts,color in [(source[i],(255,220,0)),(generated[i],(220,0,255))]:
                cv2.polylines(overlay,[np.rint(pts[chin.JAW]).astype(np.int32)],False,color,1,cv2.LINE_AA)
            mask_rows.append(np.hstack([tile(frames[2],f'Refined f{i}'),tile(overlay,'Source / target / alpha50'),tile(np.repeat(alpha[:,:,None],3,2),'Actual warped mask')]))
        if 136<=i<148:continuous.append(np.hstack([tile(f,f'{label} f{i}') for f,label in zip(frames,('Source','Standard','Refined'))]))
        if 24<=i<216:
            p=source[i];g=generated[i];_,_,_,span=chin.axes(p);mask=np.zeros((896,512),np.uint8)
            cv2.fillConvexPoly(mask,cv2.convexHull(np.rint(p[chin.JAW]).astype(np.int32)),255)
            radius=max(2,int(round(.07*span)));kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1))
            mask=cv2.erode(mask,kernel);mask[:int(p[2,1])]=0
            lip=np.zeros_like(mask)
            for q in (p,g):cv2.fillConvexPoly(lip,cv2.convexHull(np.rint(q[chin.LIPS]).astype(np.int32)),255)
            lip=cv2.dilate(lip,kernel);mask[lip>0]=0;region=mask>0
            if not region.any():raise ValueError('No lower-face texture diagnostic region')
            for name,f in zip(texture,frames):
                gray=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY).astype(np.float32)
                high=np.abs(gray-cv2.GaussianBlur(gray,(0,0),1.2));texture[name].append(float(high[region].mean()))
    for cap in caps:cap.release()
    cv2.imwrite(str(out/'contact.jpg'),np.vstack(contacts),[cv2.IMWRITE_JPEG_QUALITY,93])
    cv2.imwrite(str(out/'consecutive.jpg'),np.vstack(continuous),[cv2.IMWRITE_JPEG_QUALITY,93])
    cv2.imwrite(str(out/'mask_contact.jpg'),np.vstack(mask_rows),[cv2.IMWRITE_JPEG_QUALITY,93])
    metrics=dict(window=[24,216],geometry=geometry,source_mouth_opening_eye_spans=stats(source_motion[24:216]),
        source_roll_p95_minus_p05_degrees=float(np.percentile(roll[24:216],95)-np.percentile(roll[24:216],5)),
        lower_face_highpass={k:stats(v) for k,v in texture.items()},
        texture_retention_ratio={name:float(np.mean(texture[name])/max(np.mean(texture['source']),1e-6)) for name in ('standard','refined')},
        caveat='Landmark error is not perceptual quality. High-pass ratios measure local grayscale texture, not beard identity; mask support, motion and the changed chin geometry affect this diagnostic. Inspect actual facial hair in the videos.')
    dump(out/'metrics.json',metrics)
    media=[check_video(p) for p in paths+[delivery]]
    media.append(check_video(out/'comparison.mp4',width=1536,height=944))
    media.append(check_video(out/'face_comparison.mp4',width=1404,height=574))
    render=read(out/'render.json')
    artifacts=[delivery,out/'comparison.mp4',out/'face_comparison.mp4',out/'metrics.json',out/'contact.jpg',out/'consecutive.jpg',out/'mask_contact.jpg',out/'packaging.json',out/'standard_landmarks.npy',out/'refined_landmarks.npy']
    finish(out/'validation.json',signature,artifacts,media=media,endpoint_proof=proof,endpoint_matches_source=exact,
        technical_pass=render['quality_checks_pass'] and exact,visual_acceptance='Pending human review for this identity',
        comparison_audio='Same controlled replacement speech in all panels. H3 source mouth has independently generated phonemes.')
    print('VALIDATED',spec['id'],flush=True)

if __name__=='__main__':main()
