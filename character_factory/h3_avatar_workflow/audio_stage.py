"""Continuous replacement speech, preserving every word and short natural gaps."""
from pathlib import Path
import io,shutil,subprocess,sys
import numpy as np,soundfile as sf
from common import completed,digest,dump,finish,read,sha,spec_arg

TEXT="Hi I'm glad you're here and I'd like to show you how this works so follow along with me as we choose a look speak a few words and watch the avatar respond in real time"

def main():
    spec=spec_arg();out=Path(spec['output']);voice=spec['voice'];w=Path(spec['workspace'])
    audio_source=Path(spec['audio']) if spec.get('audio') else None
    signature=digest(dict(spec=spec['signature'],text=TEXT,code=sha(__file__),audio_sha256=sha(audio_source) if audio_source else None))
    record=out/'audio.json'
    if completed(record,signature):return
    target=out/'speech.wav';details={}
    if audio_source:
        a,sr=sf.read(audio_source)
        if a.ndim!=1 or abs(len(a)/sr-10)>.002:raise ValueError('Custom replacement audio must be mono and exactly ten seconds')
        subprocess.run(['ffmpeg','-v','error','-y','-i',str(audio_source),'-ar','24000','-ac','1','-c:a','pcm_s16le',str(target)],check=True)
        details=dict(custom_source=str(audio_source),source_sha256=sha(audio_source))
    else:
        cache=out.parent/'_audio'/voice;cache.mkdir(parents=True,exist_ok=True)
        key=digest(dict(voice=voice,text=TEXT,code=sha(__file__)));cached=cache/'speech.wav'
        if not completed(cache/'record.json',key):
            sys.path.insert(0,str(w/'SoulX-FlashHead'))
            from soulx_rtc.tts import KokoroService
            raw,headers=KokoroService().synthesize(TEXT,voice,voice[0],1.)
            (cache/'speech_raw.wav').write_bytes(raw);a,sr=sf.read(io.BytesIO(raw));window=int(sr*.02)
            rms=np.sqrt(np.mean(a[:len(a)//window*window].reshape(-1,window)**2,axis=1));voiced=np.where(rms>=.002)[0]
            if not len(voiced):raise ValueError('TTS contains no detected speech')
            begin=int(max(0,(voiced[0]-2)*window));end=int(min(len(a),(voiced[-1]+3)*window))
            a=a[begin:end];rms=np.sqrt(np.mean(a[:len(a)//window*window].reshape(-1,window)**2,axis=1))
            silent=rms<.002;cuts=[];start=None
            for i,quiet in enumerate(np.r_[silent,False]):
                if quiet and start is None:start=i
                if not quiet and start is not None:
                    if start>0 and i<len(silent) and (i-start)*.02>=.28:
                        remove=(i-start-6)*window;left=(start*window+i*window-remove)//2;cuts.append((left,left+remove))
                    start=None
            parts=[];cursor=0
            for left,right in cuts:parts.append(a[cursor:left]);cursor=right
            parts.append(a[cursor:]);dense=np.concatenate(parts)
            sf.write(cache/'speech_short_gaps.wav',dense,sr,subtype='PCM_16')
            tempo=len(dense)/sr/9.6
            if not .5<=tempo<=2:raise ValueError(f'Unexpected TTS duration: tempo={tempo}')
            subprocess.run(['ffmpeg','-v','error','-xerror','-y','-i',str(cache/'speech_short_gaps.wav'),'-af',
                f'atempo={tempo:.10f},adelay=150,apad,atrim=duration=10,asetpts=PTS-STARTPTS',
                '-ar','24000','-ac','1','-c:a','pcm_s16le',str(cached)],check=True)
            finish(cache/'record.json',key,[cached,cache/'speech_raw.wav',cache/'speech_short_gaps.wav'],
                text=TEXT,voice=voice,headers=headers,tempo=tempo,trimmed_source_samples=[begin,end],
                removed_gap_samples=cuts,lead_seconds=.15,content_target_seconds=9.6)
        shutil.copy2(cached,target);details=read(cache/'record.json')
        details={k:v for k,v in details.items() if k not in ('status','signature','artifacts','finished_utc')}
    a,sr=sf.read(target);assert sr==24000 and a.ndim==1 and len(a)==240000 and np.isfinite(a).all()
    finish(record,signature,[target],sample_rate=sr,samples=len(a),**details)
    print('AUDIO READY',target,flush=True)

if __name__=='__main__':main()
