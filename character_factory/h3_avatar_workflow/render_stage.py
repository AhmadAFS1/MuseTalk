"""Fresh TAESD inference with the approved, full-strength refined chin blend."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import gc,hashlib,sys,time
import cv2,numpy as np,torch
from common import HERE,completed,digest,dump,finish,read,sha,spec_arg
from backend import Tracker,setup,encode

def hash_frames(frames):
    h=hashlib.sha256()
    for f in frames:h.update(f.tobytes())
    return h.hexdigest()

@torch.inference_mode()
def main():
    spec=spec_arg();out=Path(spec['output']);w=Path(spec['workspace'])
    sys.path.insert(0,str(w/'MuseTalk'))
    import chin
    torch.set_num_threads(4);cv2.setNumThreads(2)
    model_paths=[w/'MuseTalk/models/taesd/diffusion_pytorch_model.safetensors',w/'MuseTalk/models/taesd/config.json',
        w/'MuseTalk/models/tensorrt_unet_sm89_bs8_local/unet_trt.ts',w/'MuseTalk/musetalk/utils/blending.py']
    model_hashes={str(p.relative_to(w)):sha(p) for p in model_paths}
    signature=digest(dict(spec=spec['signature'],code=sha(__file__),chin=sha(HERE/'chin.py'),backend=sha(HERE/'backend.py'),models=model_hashes,
        cache=sha(out/'cache.pt'),masks=sha(out/'masks.npz'),source=sha(out/'source.mp4'),landmarks=sha(out/'source_landmarks.npy')))
    if completed(out/'render.json',signature):return
    cap=cv2.VideoCapture(str(out/'source.mp4'));frames=[]
    while True:
        ok,f=cap.read()
        if not ok:break
        frames.append(f)
    cap.release();assert len(frames)==240
    d=dict(cache=torch.load(out/'cache.pt',map_location='cpu',weights_only=False),masks=np.load(out/'masks.npz'),
        p=np.load(out/'source_landmarks.npy'),g=np.zeros((240,478,2),np.float32))
    chin.prepare_refined(d,frames)
    load_start=time.perf_counter();unet,decoder,sf=setup(w);load_seconds=time.perf_counter()-load_start
    stamp=torch.tensor([0],device='cuda')
    for _ in range(4):
        z=unet(d['cache']['latents'][:8].cuda(),stamp,encoder_hidden_states=d['cache']['audio'][:8].cuda()).sample
        decoder.decode(z,sf,torch.float16)
    torch.cuda.synchronize();tracker=Tracker(w,out/'generated_tracking.log');tracker.reset()
    outputs=[];faces=[];queue=[];previous=None;timing=dict(tracking_ipc_ms=0.,compose_ms=0.)
    def emit(record,next_delta):
        nonlocal previous
        i,face,g,delta=record;d['g'][i]=g;old=delta if previous is None else previous
        d['chin_delta'][i]=np.clip(.25*old+.5*delta+.25*next_delta,0,.18)
        start=time.perf_counter();outputs.append(chin.corrected_refined(frames[i],d,i,face));timing['compose_ms']+=(time.perf_counter()-start)*1000
        previous=delta
    @torch.inference_mode()
    def generate(i):
        z=unet(d['cache']['latents'][i:i+8].cuda(),stamp,encoder_hidden_states=d['cache']['audio'][i:i+8].cuda()).sample
        pixels=decoder.decode(z,sf,torch.float16)
        if not torch.isfinite(pixels).all():raise RuntimeError('Non-finite generated pixels')
        return pixels.float().mul(255).round().clamp(0,255).to(torch.uint8).flip(1).permute(0,2,3,1).contiguous().cpu().numpy()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            start=time.perf_counter();future=pool.submit(generate,0)
            for base in range(0,240,8):
                pixels=future.result()
                if base+8<240:future=pool.submit(generate,base+8)
                faces.extend(pixels)
                for j,face in enumerate(pixels):
                    i=base+j;t=time.perf_counter();g,_=tracker.track(frames[i],face,d['cache']['boxes'][i]);timing['tracking_ipc_ms']+=(time.perf_counter()-t)*1000
                    source,target=chin.curves(d['p'][i],g);delta=source-target;queue.append((i,face,g,delta))
                    if len(queue)==2:emit(queue.pop(0),delta)
            if queue:emit(queue[0],queue[0][3])
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
    finally:tracker.close()
    assert len(outputs)==240
    np.save(out/'generated_landmarks.npy',d['g']);np.save(out/'chin_delta.npy',d['chin_delta'])
    np.savez_compressed(out/'faces.npz',faces=np.asarray(faces))
    standard=[];rows=[];max_lip=0;min_jac=1.;max_reference=0;mask_samples={}
    # Offline diagnostics are excluded from the timed render path.
    for i,(frame,face) in enumerate(zip(frames,faces)):
        plain=chin.standard(frame,d,i,face);standard.append(plain)
        mask=d['refined_masks'][i].current(d['g'][i],d['chin_delta'][i]);b=d['cache']['boxes'][i];cb=d['cache']['cropboxes'][i]
        x,y,x1,y1=map(int,b);retained=chin.get_image_blending(frame.copy(),cv2.resize(face,(x1-x,y1-y)),b,mask,cb)
        reference,meta=chin.aligned(retained,d,i,1.)
        max_reference=max(max_reference,int(np.abs(reference.astype(np.int16)-outputs[i].astype(np.int16)).max()))
        _,_,_,span=chin.axes(d['p'][i]);lip=np.zeros(frame.shape[:2],np.uint8)
        cv2.fillConvexPoly(lip,cv2.convexHull(np.rint(d['g'][i][chin.LIPS]).astype(np.int32)),255)
        radius=max(4,int(round(.06*span)));lip=cv2.dilate(lip,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2*radius+1,2*radius+1)))>0
        lip_diff=int(np.abs(plain.astype(np.int16)-outputs[i].astype(np.int16))[lip].max())
        max_lip=max(max_lip,lip_diff);min_jac=min(min_jac,meta['min_map_jacobian'])
        rows.append(dict(frame=i,protected_lip_difference=lip_diff,**meta))
        if i in (24,48,96,144,192,215):
            alpha=np.zeros(frame.shape[:2],np.uint8);cx,cy=cb[:2]
            alpha[y:y1,x:x1]=mask[y-cy:y1-cy,x-cx:x1-cx]
            mask_samples[str(i)]=chin.warp_roi(np.repeat(alpha[:,:,None],3,2),d,i,1.)[:,:,0]
    checks=dict(frames=240,protected_lip_max_rgb_difference=max_lip,reference_max_rgb_difference=max_reference,
        minimum_jacobian=min_jac,passes=max_lip==0 and max_reference==0 and min_jac>.25,rows=rows)
    dump(out/'pixel_checks.json',checks)
    np.savez_compressed(out/'mask_samples.npz',**mask_samples)
    encode(out/'standard_raw.mp4',standard,out/'speech.wav');encode(out/'refined_raw.mp4',outputs,out/'speech.wav')
    finish(out/'render.json',signature,[out/'standard_raw.mp4',out/'refined_raw.mp4',out/'faces.npz',out/'generated_landmarks.npy',out/'chin_delta.npy',out/'pixel_checks.json',out/'mask_samples.npz'],
        raw_refined_sha256=hash_frames(outputs),generated_faces_sha256=hash_frames(faces),
        load_seconds=load_seconds,render_seconds=elapsed,warm_render_fps=240/elapsed,timing=timing,
        timing_scope='One warm run; fresh inference, transfers, finite-pixel checks, per-frame tracking IPC and refined composition. Excludes load, preparation, audio features, offline validation and encoding.',
        decoder='compiled TAESD',unet='TensorRT FP16 batch8',encoder='native FP16',chin_strength=1.,quality_checks_pass=checks['passes'],
        model_and_blending_code_sha256=model_hashes,
        code_sha256={p.name:sha(p) for p in [HERE/'chin.py',HERE/'backend.py',HERE/'render_stage.py',HERE/'tracker_worker.py']},
        torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__)
    print('RENDER READY',spec['id'],checks['passes'],240/elapsed,flush=True)

if __name__=='__main__':main()
