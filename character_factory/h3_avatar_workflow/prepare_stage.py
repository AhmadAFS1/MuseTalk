"""Fresh native FP16 avatar encoding, with source-bound face masks and boxes."""
from pathlib import Path
import gc,hashlib,os,sys,tempfile,time
import cv2,numpy as np,torch
from common import completed,digest,dump,finish,read,sha,spec_arg

@torch.inference_mode()
def main():
    spec=spec_arg();out=Path(spec['output']);muse=Path(spec['workspace'])/'MuseTalk';os.chdir(muse)
    sys.path[:0]=[str(muse),str(muse/'scripts'),str(muse/'musetalk/utils')]
    from musetalk.models.vae import VAE
    from musetalk.models.unet import PositionalEncoding
    from musetalk.utils.preprocessing import get_landmark_and_bbox,coord_placeholder
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.blending import get_image_prepare_material
    from musetalk.utils.audio_processor import AudioProcessor
    from transformers import WhisperModel
    torch.set_num_threads(4);cv2.setNumThreads(2)
    models={str(p.relative_to(muse)):sha(p) for p in [muse/'models/sd-vae/diffusion_pytorch_model.bin',
        muse/'models/sd-vae/config.json',muse/'models/whisper/pytorch_model.bin',muse/'models/whisper/config.json',
        muse/'models/whisper/preprocessor_config.json',muse/'musetalk/models/unet.py',muse/'musetalk/utils/audio_processor.py']}
    signature=digest(dict(spec=spec['signature'],code=sha(__file__),source=sha(out/'source.mp4'),audio=sha(out/'speech.wav'),models=models))
    record=out/'preparation.json'
    if completed(record,signature):return
    start=time.perf_counter();cap=cv2.VideoCapture(str(out/'source.mp4'));frames=[]
    while True:
        ok,f=cap.read()
        if not ok:break
        frames.append(f)
    cap.release();assert len(frames)==240 and frames[0].shape==(896,512,3)
    v=VAE(model_path=str(muse/'models/sd-vae'),use_float16=True);v.vae.eval()
    parser=FaceParsing(left_cheek_width=90,right_cheek_width=90)
    with tempfile.TemporaryDirectory(prefix='face_detect_',dir=out) as td:
        files=[]
        for i,f in enumerate(frames):
            p=Path(td)/f'{i:06d}.png';assert cv2.imwrite(str(p),f);files.append(str(p))
        boxes,detected=get_landmark_and_bbox(files,0)
        if len(boxes)!=240 or any(tuple(b)==coord_placeholder for b in boxes):raise RuntimeError('Face detection failed; review this source before resuming')
        del detected
    boxes=np.asarray(boxes,np.int32);boxes[:,3]=np.minimum(boxes[:,3]+10,896)
    # Audio conditioning is shareable; image latents and masks are source-specific.
    audio_key=digest(dict(audio=sha(out/'speech.wav'),models=models,fps=24))[:24]
    audio_cache=out.parent/'_audio'/f'features_{audio_key}.pt';audio_cache.parent.mkdir(exist_ok=True)
    if audio_cache.exists():audio=torch.load(audio_cache,map_location='cpu',weights_only=True)
    else:
        whisper=WhisperModel.from_pretrained(str(muse/'models/whisper')).half().cuda().eval()
        pe=PositionalEncoding(d_model=384).cuda().half().eval();processor=AudioProcessor(str(muse/'models/whisper'))
        features,n=processor.get_audio_feature(str(out/'speech.wav'))
        chunks=processor.get_whisper_chunk(features,'cuda',torch.float16,whisper,n,fps=24,audio_padding_length_left=2,audio_padding_length_right=2)
        chunks=chunks[:240] if isinstance(chunks,torch.Tensor) else torch.stack(chunks[:240])
        if len(chunks)!=240:raise ValueError('Expected exactly 240 audio conditioning chunks')
        audio=pe(chunks.cuda().half()).cpu();torch.save(audio,audio_cache)
        del whisper,pe,processor;gc.collect();torch.cuda.empty_cache()
    latents=[];masks={};cropboxes=[];torch.manual_seed(123)
    for i,(f,box) in enumerate(zip(frames,boxes)):
        x,y,x1,y1=map(int,box)
        if not 0<=x<x1<=512 or not 0<=y<y1<=896:raise ValueError(('invalid face box',i,box))
        crop=cv2.resize(f[y:y1,x:x1],(256,256),interpolation=cv2.INTER_LANCZOS4)
        latents.append(v.get_latents_for_unet(crop).cpu())
        mask,cb=get_image_prepare_material(f,tuple(map(int,box)),fp=parser,mode='jaw');masks[str(i)]=mask;cropboxes.append(cb)
        if i%60==0:print('PREPARED',i,flush=True)
    latent=torch.cat(latents);assert torch.isfinite(latent).all() and torch.isfinite(audio).all()
    torch.save(dict(latents=latent,audio=audio,boxes=boxes,cropboxes=cropboxes),out/'cache.pt')
    np.savez_compressed(out/'masks.npz',**masks)
    finish(record,signature,[out/'cache.pt',out/'masks.npz'],seconds=time.perf_counter()-start,frames=240,
        encoder='Native MuseTalk FP16 SD-VAE',seed=123,mask='jaw, cheeks90, bbox lower margin10',
        source_sha256=sha(out/'source.mp4'),audio_sha256=sha(out/'speech.wav'),
        audio_features_sha256=hashlib.sha256(audio.numpy().tobytes()).hexdigest(),
        model_and_audio_code_sha256=models,
        torch=torch.__version__,numpy=np.__version__,opencv=cv2.__version__)

if __name__=='__main__':main()
