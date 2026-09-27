"""FaceMesh (refined, tracking mode) per-frame CPU cost + multi-process scaling.
Runs in /workspace/SoulX-FlashHead/.venv. Read-only on /workspace."""
import sys, os, time, json, resource
sys.dont_write_bytecode = True
import numpy as np, cv2, mediapipe as mp
import multiprocessing as mproc
WHO = sys.argv[1] if len(sys.argv) > 1 else 'black_woman'
N = int(os.environ.get('NFRAMES', '96'))
root = f'/workspace/experiments/avatar_diversity_20260927/{WHO}'
cv2.setNumThreads(1)

def load():
    cap = cv2.VideoCapture(f'{root}/source.mp4'); frames = []
    for _ in range(N):
        ok, f = cap.read(); assert ok; frames.append(f)
    cap.release()
    faces = np.load(f'{root}/faces.npz')['faces'][:N]
    import torch  # not available here; boxes from saved cache via numpy fallback
    return frames, faces

def load_np():
    cap = cv2.VideoCapture(f'{root}/source.mp4'); frames = []
    for _ in range(N):
        ok, f = cap.read(); assert ok; frames.append(f)
    cap.release()
    faces = np.load(f'{root}/faces.npz')['faces'][:N]
    boxes = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'boxes_{WHO}.npy'))[:N]
    composite = []
    for f, face, b in zip(frames, faces, boxes):
        x, y, x1, y1 = map(int, b); g = f.copy(); g[y:y1, x:x1] = cv2.resize(face, (x1 - x, y1 - y)); composite.append(g)
    return composite, boxes

def mesh_new(refine=True):
    return mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=refine,
                                           min_detection_confidence=.5, min_tracking_confidence=.5)

def run_seq(imgs, refine=True, reps=2, crop=None):
    mesh = mesh_new(refine); t_proc = []; t_extract = []; t_cvt = []
    H, W = imgs[0].shape[:2]
    for r in range(reps):
        for i, im in enumerate(imgs):
            t0 = time.perf_counter(); rgb = cv2.cvtColor(im if crop is None else im[crop[1]:crop[3], crop[0]:crop[2]], cv2.COLOR_BGR2RGB)
            t1 = time.perf_counter(); res = mesh.process(rgb)
            t2 = time.perf_counter()
            h, w = rgb.shape[:2]
            pts = np.array([(p.x * w, p.y * h) for p in res.multi_face_landmarks[0].landmark], np.float32)
            t3 = time.perf_counter()
            if r > 0 or i >= 8:
                t_cvt.append(t1 - t0); t_proc.append(t2 - t1); t_extract.append(t3 - t2)
    mesh.close()
    f = lambda a: round(1000 * float(np.mean(a)), 4)
    return dict(cvt_ms=f(t_cvt), process_ms=f(t_proc), extract_ms=f(t_extract), total_ms=f(np.add(np.add(t_cvt, t_proc), t_extract)))

def fast_extract_test(imgs):
    mesh = mesh_new(True); res = mesh.process(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB)); lm = res.multi_face_landmarks[0]
    t = time.perf_counter()
    for _ in range(200): np.array([(p.x * 512, p.y * 896) for p in lm.landmark], np.float32)
    a = (time.perf_counter() - t) / 200
    t = time.perf_counter()
    for _ in range(200):
        np.fromiter((v for p in lm.landmark for v in (p.x, p.y)), np.float32, count=956)
    b = (time.perf_counter() - t) / 200
    mesh.close(); return dict(listcomp_ms=round(a * 1000, 4), fromiter_ms=round(b * 1000, 4))

def _worker(k, seconds, q):
    imgs, _ = load_np(); mesh = mesh_new(True)
    for im in imgs[:8]: mesh.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    r0 = resource.getrusage(resource.RUSAGE_SELF); c = 0; stop = time.perf_counter() + seconds; t0 = time.perf_counter(); j = 8
    while time.perf_counter() < stop:
        im = imgs[j % len(imgs)]; res = mesh.process(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        np.array([(p.x * 512, p.y * 896) for p in res.multi_face_landmarks[0].landmark], np.float32); c += 1; j += 1
    wall = time.perf_counter() - t0; r1 = resource.getrusage(resource.RUSAGE_SELF)
    q.put((c / wall, (r1.ru_utime + r1.ru_stime - r0.ru_utime - r0.ru_stime) / wall))

def pscale(n, seconds=4.0):
    ctx = mproc.get_context('spawn'); q = ctx.Queue()
    ps = [ctx.Process(target=_worker, args=(k, seconds, q)) for k in range(n)]
    [p.start() for p in ps]; r = [q.get() for _ in ps]; [p.join() for p in ps]
    return dict(procs=n, agg_fps=round(sum(x[0] for x in r), 1), per_proc_fps=round(float(np.mean([x[0] for x in r])), 1),
                cpu_cores_per_proc=round(float(np.mean([x[1] for x in r])), 2))

if __name__ == '__main__':
    imgs, boxes = load_np(); out = dict(who=WHO, mediapipe=mp.__version__, numpy=np.__version__, opencv=cv2.__version__)
    b = boxes[0].astype(int); crop = [int(v) for v in (max(0, b[0] - 64), max(0, b[1] - 96), min(512, b[2] + 64), min(896, b[3] + 64))]
    out['full_frame_refined'] = run_seq(imgs, True); print('full refined', out['full_frame_refined'], flush=True)
    out['full_frame_unrefined'] = run_seq(imgs, False); print('full unrefined', out['full_frame_unrefined'], flush=True)
    out['crop_refined'] = dict(crop=crop, **run_seq(imgs, True, crop=crop)); print('crop refined', out['crop_refined'], flush=True)
    out['extract'] = fast_extract_test(imgs); print('extract', out['extract'], flush=True)
    if os.environ.get('SCALING', '1') == '1':
        out['process_scaling'] = []
        for n in [int(x) for x in os.environ.get('PROCS', '1,2,4,8,16').split(',')]:
            r = pscale(n); out['process_scaling'].append(r); print('scale', r, flush=True)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f'facemesh_{WHO}.json'), 'w'), indent=1)
