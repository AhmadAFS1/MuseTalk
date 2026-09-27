"""CPU-only microbenchmarks of MuseTalk post-GPU compose: standard vs refined chin path.

Read-only on /workspace. Uses saved TAESD faces + saved generated landmarks/chin_delta
from the avatar_diversity batch so no GPU is touched. Run with PYTHONDONTWRITEBYTECODE=1.
"""
import sys, os, time, json, threading, multiprocessing as mproc
sys.dont_write_bytecode = True
sys.path[:0] = ['/workspace/MuseTalk', '/workspace/MuseTalk/character_factory/h3_avatar_workflow']
import numpy as np, cv2, torch
import chin
from musetalk.utils.blending import get_image_blending_with_plan

WHO = sys.argv[1] if len(sys.argv) > 1 else 'black_woman'
N = int(os.environ.get('NFRAMES', '96'))
CVT = int(os.environ.get('CVTHREADS', '1'))
cv2.setNumThreads(CVT); torch.set_num_threads(1)
root = f'/workspace/experiments/avatar_diversity_20260927/{WHO}'

cap = cv2.VideoCapture(f'{root}/source.mp4'); frames = []
for _ in range(N):
    ok, f = cap.read(); assert ok; frames.append(f)
cap.release()
d = dict(cache=torch.load(f'{root}/cache.pt', map_location='cpu', weights_only=False),
         masks=np.load(f'{root}/masks.npz'), p=np.load(f'{root}/source_landmarks.npy'))
faces = np.load(f'{root}/faces.npz')['faces'][:N]
t = time.perf_counter(); chin.prepare_refined(d, frames); prep_s = time.perf_counter() - t
d['g'] = np.load(f'{root}/generated_landmarks.npy'); d['chin_delta'] = np.load(f'{root}/chin_delta.npy')

def timeit(fn, reps=3):
    ts = []
    for _ in range(reps):
        for i in range(N):
            t0 = time.perf_counter_ns(); fn(i); ts.append((time.perf_counter_ns() - t0) / 1e6)
    a = np.asarray(ts); return dict(mean=round(float(a.mean()), 4), p50=round(float(np.median(a)), 4), p95=round(float(np.percentile(a, 95)), 4))

def box(i): return tuple(map(int, d['cache']['boxes'][i]))
def resized(i):
    x, y, x1, y1 = box(i); return cv2.resize(faces[i], (x1 - x, y1 - y))
R = [resized(i) for i in range(N)]
masks_now = [d['refined_masks'][i].current(d['g'][i], d['chin_delta'][i]) for i in range(N)]
def plan_with(i, mask):
    plan = d['plans'][i].copy(); ys, xs = plan['clip_slice']; cx, cy = d['cache']['cropboxes'][i][:2]
    alpha = mask[ys.start - cy:ys.stop - cy, xs.start - cx:xs.stop - cx]
    plan['alpha_u8'] = alpha[:, :, None]; plan['alpha'] = (alpha.astype(np.float32) / 255.)[:, :, None]
    return plan
def plan_u8only(i, mask):
    plan = d['plans'][i].copy(); ys, xs = plan['clip_slice']; cx, cy = d['cache']['cropboxes'][i][:2]
    plan['alpha_u8'] = mask[ys.start - cy:ys.stop - cy, xs.start - cx:xs.stop - cx][:, :, None]; return plan
retained = [get_image_blending_with_plan(frames[i].copy(), R[i], plan_with(i, masks_now[i])) for i in range(N)]
yuv_buf = {}
try:
    import av
except Exception:
    av = None

stages = {
    'frame_copy_512x896': lambda i: frames[i].copy(),
    'resize_256_to_bbox': lambda i: resized(i),
    'std_blend_fixedpoint(plan)': lambda i: get_image_blending_with_plan(frames[i].copy(), R[i], d['plans'][i]),
    'STANDARD_total(copy+resize+blend)': lambda i: chin.standard(frames[i], d, i, faces[i]),
    'geometry_curves+filter': lambda i: np.clip(.25 * (lambda s, g: s - g)(*chin.curves(d['p'][i], d['g'][i])) * 4 / 4, 0, .18),
    'refined_mask_current(g,delta)': lambda i: d['refined_masks'][i].current(d['g'][i], d['chin_delta'][i]),
    'sourcemask_current(g)[chin100 v1]': lambda i: d['source_masks'][i].current(d['g'][i]),
    'plan_alpha_rebuild(float+u8)': lambda i: plan_with(i, masks_now[i]),
    'plan_alpha_rebuild(u8 only)': lambda i: plan_u8only(i, masks_now[i]),
    'blend_with_dynamic_alpha': lambda i: get_image_blending_with_plan(frames[i].copy(), R[i], plan_with(i, masks_now[i])),
    'warp_roi(100%)': lambda i: chin.warp_roi(retained[i], d, i, 1.),
    'REFINED_total(corrected_refined)': lambda i: chin.corrected_refined(frames[i], d, i, faces[i]),
    'CHIN_v1_total(corrected SourceMask)': lambda i: chin.corrected(frames[i], d, i, faces[i]),
    'tracker_input_prep(frame->shm+resize paste)': lambda i: (lambda buf: (buf.__setitem__(slice(None), frames[i]), buf.__setitem__((slice(box(i)[1], box(i)[3]), slice(box(i)[0], box(i)[2])), resized(i))))(np.empty_like(frames[i])),
    'cv2_BGR2RGB_512x896': lambda i: cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB),
    'cv2_BGR2YUV_I420_512x896': lambda i: cv2.cvtColor(frames[i], cv2.COLOR_BGR2YUV_I420),
}
if av is not None:
    stages['pyav_from_ndarray+reformat_yuv420p'] = lambda i: av.VideoFrame.from_ndarray(frames[i], format='bgr24').reformat(format='yuv420p')

def scaling(fn, workers, seconds=2.0):
    stop = time.perf_counter() + seconds; counts = [0] * workers
    def run(k):
        j = k
        while time.perf_counter() < stop:
            fn(j % N); j += 1; counts[k] += 1
    th = [threading.Thread(target=run, args=(k,)) for k in range(workers)]
    t0 = time.perf_counter(); [x.start() for x in th]; [x.join() for x in th]
    return round(sum(counts) / (time.perf_counter() - t0), 1)

def _proc(fn_name, seconds, q, k):
    cv2.setNumThreads(1); fn = stages[fn_name]; stop = time.perf_counter() + seconds; c = 0; j = k
    while time.perf_counter() < stop:
        fn(j % N); j += 1; c += 1
    q.put(c / seconds)

def pscaling(fn_name, workers, seconds=2.0):
    ctx = mproc.get_context('fork'); q = ctx.Queue()
    ps = [ctx.Process(target=_proc, args=(fn_name, seconds, q, k)) for k in range(workers)]
    [p.start() for p in ps]; r = [q.get() for _ in ps]; [p.join() for p in ps]
    return round(sum(r), 1)

if __name__ == '__main__':
    out = dict(who=WHO, frames=N, cv2_threads=CVT, numpy=np.__version__, opencv=cv2.__version__,
               frame_shape=list(frames[0].shape), bbox_example=list(box(0)), cropbox_example=list(d['cache']['cropboxes'][0]),
               plan_clip_shape=[s.stop - s.start for s in d['plans'][0]['clip_slice']],
               refined_region_shape=[s.stop - s.start for s in d['refined_masks'][0].region],
               prepare_refined_seconds_for_N=round(prep_s, 3), stages_ms={})
    for k, fn in stages.items():
        fn(0); out['stages_ms'][k] = timeit(fn)
        print(k, out['stages_ms'][k], flush=True)
    if os.environ.get('SCALING', '1') == '1':
        out['thread_scaling_fps'] = {}; out['process_scaling_fps'] = {}
        for name in ('STANDARD_total(copy+resize+blend)', 'REFINED_total(corrected_refined)'):
            out['thread_scaling_fps'][name] = {w: scaling(stages[name], w) for w in (1, 2, 4, 8, 12, 16)}
            print('threads', name, out['thread_scaling_fps'][name], flush=True)
            out['process_scaling_fps'][name] = {w: pscaling(name, w) for w in (1, 4, 8, 16)}
            print('procs', name, out['process_scaling_fps'][name], flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'compose_{WHO}_cv{CVT}.json'), 'w'), indent=1)
