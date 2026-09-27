import av, fractions, sys, subprocess
ctxs=[]
for i in range(int(sys.argv[1])):
    try:
        c=av.CodecContext.create("h264_nvenc","w"); c.width,c.height=512,832; c.pix_fmt="yuv420p"
        c.bit_rate=2500000; c.framerate=fractions.Fraction(20,1); c.time_base=fractions.Fraction(1,20)
        c.options={"preset":"p2","tune":"ll","bf":"0","rc":"cbr_ld_hq"}; c.open()
        f=av.VideoFrame(512,832,"yuv420p"); f.pts=0; list(c.encode(f)); ctxs.append(c)
    except Exception as e:
        print(f"session {i+1} FAILED: {type(e).__name__}: {e}"); break
print("opened sessions:", len(ctxs))
print(subprocess.run(["nvidia-smi","--query-gpu=memory.used,encoder.stats.sessionCount","--format=csv,noheader"],capture_output=True,text=True).stdout.strip())
