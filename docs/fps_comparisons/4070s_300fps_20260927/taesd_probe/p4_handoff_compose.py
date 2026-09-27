"""Probe 4: GPU->CPU handoff of decoded faces, and CPU vs GPU resize-to-bbox + alpha blend.

Uses real avatar cache (japanese_realtime_talking_7d94520b7f: 832x512 frames, real bboxes/masks)
and a synthetic 170x210 bbox on the same 832x512 frame, per the task spec.
"""
import sys, time, glob, pickle, os
sys.path.insert(0, os.path.dirname(__file__))
from common import *  # noqa
import cv2
import torch.nn.functional as F
sys.path.insert(0, str(ROOT))
from musetalk.utils.blending import prepare_image_blending_plan, get_image_blending_with_plan, get_crop_box

torch.backends.cudnn.benchmark = True
BS = 16
AV = ROOT / "results/v15/avatars/japanese_realtime_talking_7d94520b7f"
res = {"gpu_start": gpu_state("start"), "bs": BS, "avatar": str(AV)}

# ---------------------------------------------------------------- decoded faces (real)
model = load_taesd()
raw = repo_raw_decode(model)
Z = real_latents(BS).to(DEV)
with torch.inference_mode():
    img = raw(Z).to(torch.float16).contiguous()  # [16,3,256,256] fp16 in [0,1], RGB (as backend returns)
torch.cuda.synchronize()

# ---------------------------------------------------------------- 4a. handoff
def repo_post(image):
    return (image.detach().float().mul(255).round().clamp_(0, 255).to(torch.uint8)
            .flip(1).permute(0, 2, 3, 1).contiguous().cpu().numpy())


def gpu_convert(image):
    return (image.float().mul(255).round().clamp_(0, 255).to(torch.uint8)
            .flip(1).permute(0, 2, 3, 1).contiguous())

gpu_convert_c = torch.compile(gpu_convert, mode="max-autotune-no-cudagraphs", dynamic=False)
pinned = torch.empty((BS, 256, 256, 3), dtype=torch.uint8, pin_memory=True)


def walls(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); ts = []
    for _ in range(iters):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return {"median_ms": float(np.median(ts)), "p90_ms": float(np.percentile(ts, 90)), "n": iters}

H = {}
with torch.inference_mode():
    ref_u8 = repo_post(img)
    assert np.array_equal(ref_u8, gpu_convert_c(img).cpu().numpy())
    st = gpu_state("handoff")
    H["repo_path_total_wall(convert+pageable .cpu())"] = walls(lambda: repo_post(img))
    H["gpu_convert_eager_events"] = bench_events(lambda: gpu_convert(img), warmup=10, iters=50)
    H["gpu_convert_compiled_events"] = bench_events(lambda: gpu_convert_c(img), warmup=10, iters=50)
    u8 = gpu_convert_c(img)
    H["d2h_pageable_3.1MB_wall"] = walls(lambda: u8.cpu())
    H["d2h_pinned_3.1MB_events"] = bench_events(lambda: pinned.copy_(u8, non_blocking=True), warmup=10, iters=50)

    def opt_path():
        pinned.copy_(gpu_convert_c(img), non_blocking=True)
        return pinned.numpy()
    H["optimized_total_wall(compiled convert + pinned D2H)"] = walls(opt_path)
    H["gpu_state"] = st
res["handoff_bs16"] = H
for k, v in H.items():
    if isinstance(v, dict) and "median_ms" in v:
        print(f"handoff {k:55s}: {v['median_ms']:.3f} ms/batch ({v['median_ms']/BS*1e3:.1f} us/frame)", flush=True)

# ---------------------------------------------------------------- load real avatar
frames = [cv2.imread(p) for p in sorted(glob.glob(str(AV / "full_imgs/*.png")))[:BS]]
coords = pickle.load(open(AV / "coords.pkl", "rb"))[:BS]
mcoords = pickle.load(open(AV / "mask_coords.pkl", "rb"))[:BS]
masks = [cv2.imread(p) for p in sorted(glob.glob(str(AV / "mask/*.png")))[:BS]]
faces_u8 = ref_u8  # [16,256,256,3] BGR uint8 as the CPU compose receives it


def make_cases():
    real = []
    for i in range(BS):
        real.append(dict(frame=frames[i], bbox=[int(v) for v in coords[i]], mask=masks[i],
                         mcrop=[int(v) for v in mcoords[i]]))
    synth = []
    for i in range(BS):
        x1, y1, x2, y2 = [int(v) for v in coords[i]]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bb = [cx - 85, cy - 105, cx + 85, cy + 105]  # 170 x 210
        crop, s = get_crop_box(bb, 1.5)
        m = cv2.resize(masks[i], (crop[2] - crop[0], crop[3] - crop[1]), interpolation=cv2.INTER_LINEAR)
        synth.append(dict(frame=frames[i], bbox=bb, mask=m, mcrop=crop))
    return {"real_japanese_bbox~224x308": real, "synthetic_bbox_170x210": synth}

CASES = make_cases()
res["compose"] = {}

# ---------------------------------------------------------------- 4b. CPU compose path
def cpu_compose(case, face, plan):
    x1, y1, x2, y2 = case["bbox"]
    ori = case["frame"].copy()
    rf = cv2.resize(face, (x2 - x1, y2 - y1))
    return get_image_blending_with_plan(ori, rf, plan)


for cname, cases in CASES.items():
    plans = [prepare_image_blending_plan(c["frame"].shape, c["bbox"], c["mask"], c["mcrop"]) for c in cases]
    R = {"frame_hw": list(cases[0]["frame"].shape[:2]),
         "bbox_wh": [cases[0]["bbox"][2] - cases[0]["bbox"][0], cases[0]["bbox"][3] - cases[0]["bbox"][1]],
         "clip_roi_hw": [plans[0]["clip_slice"][0].stop - plans[0]["clip_slice"][0].start,
                         plans[0]["clip_slice"][1].stop - plans[0]["clip_slice"][1].start]}
    cpu_out = [cpu_compose(cases[i], faces_u8[i], plans[i]) for i in range(BS)]
    for nthreads in (1, 0):
        cv2.setNumThreads(nthreads if nthreads else os.cpu_count())
        ts = []
        for rep in range(8):
            t0 = time.perf_counter()
            for i in range(BS):
                cpu_compose(cases[i], faces_u8[i], plans[i])
            ts.append((time.perf_counter() - t0) * 1e3 / BS)
        R[f"cpu_ms_per_frame_cv2threads={'1' if nthreads else 'all'}"] = float(np.median(ts[2:]))
    # component split (single thread)
    cv2.setNumThreads(1)
    def tm(f, n=200):
        t0 = time.perf_counter()
        for k in range(n): f(k % BS)
        return (time.perf_counter() - t0) * 1e3 / n
    R["cpu_split_ms"] = {
        "frame_copy": tm(lambda i: cases[i]["frame"].copy()),
        "cv2_resize": tm(lambda i: cv2.resize(faces_u8[i], (cases[i]["bbox"][2] - cases[i]["bbox"][0],
                                                            cases[i]["bbox"][3] - cases[i]["bbox"][1]))),
        "blend_with_plan(incl roi copy)": tm(lambda i: get_image_blending_with_plan(
            cases[i]["frame"].copy(), cv2.resize(faces_u8[i], (cases[i]["bbox"][2] - cases[i]["bbox"][0],
                                                               cases[i]["bbox"][3] - cases[i]["bbox"][1])), plans[i])),
    }
    R["cpu_split_ms"]["blend_only"] = (R["cpu_split_ms"]["blend_with_plan(incl roi copy)"]
                                       - R["cpu_split_ms"]["frame_copy"] - R["cpu_split_ms"]["cv2_resize"])
    cv2.setNumThreads(os.cpu_count())

    # ------------------------------------------------------------ 4c. GPU compose (batched grid_sample)
    # Precompute (once per avatar cycle position, cacheable like the CPU plan):
    #   sampling grid mapping each clip-ROI pixel to the 256x256 face, alpha (zeroed where no face),
    #   base ROI pixels. All padded to a common canvas so a batch of mixed bboxes is ONE kernel chain.
    HC = max(p["clip_slice"][0].stop - p["clip_slice"][0].start for p in plans)
    WC = max(p["clip_slice"][1].stop - p["clip_slice"][1].start for p in plans)
    grids = torch.zeros((BS, HC, WC, 2), dtype=torch.float32)
    alphas = torch.zeros((BS, 1, HC, WC), dtype=torch.float32)
    bases = torch.zeros((BS, 3, HC, WC), dtype=torch.float32)
    for i, (c, p) in enumerate(zip(cases, plans)):
        cy, cx = p["clip_slice"]
        h, w = cy.stop - cy.start, cx.stop - cx.start
        x1, y1, x2, y2 = c["bbox"]
        fw, fh = x2 - x1, y2 - y1
        # full-frame pixel coords of the ROI
        yy, xx = torch.meshgrid(torch.arange(cy.start, cy.stop, dtype=torch.float32),
                                torch.arange(cx.start, cx.stop, dtype=torch.float32), indexing="ij")
        # resized-face pixel coords (dst of cv2.resize) -> source 256 coords (cv2 INTER_LINEAR / align_corners=False)
        u = (xx - x1 + 0.5) * (256.0 / fw) - 0.5
        v = (yy - y1 + 0.5) * (256.0 / fh) - 0.5
        grids[i, :h, :w, 0] = (2 * u + 1) / 256 - 1
        grids[i, :h, :w, 1] = (2 * v + 1) / 256 - 1
        a = torch.from_numpy(p["alpha_u8"][:, :, 0].astype(np.float32))
        inside = ((xx >= x1) & (xx < x2) & (yy >= y1) & (yy < y2)).float()
        alphas[i, 0, :h, :w] = a * inside  # outside the pasted face the overlay == base, so alpha is moot
        bases[i, :, :h, :w] = torch.from_numpy(c["frame"][cy, cx].astype(np.float32)).permute(2, 0, 1)
    grids, alphas = grids.to(DEV), alphas.to(DEV)
    bases_u8 = bases.to(torch.uint8).to(DEV)
    roi_pinned = torch.empty((BS, HC, WC, 3), dtype=torch.uint8, pin_memory=True)

    def gpu_compose(face01_rgb):
        # face01_rgb [B,3,256,256] fp16 RGB [0,1] straight from the decoder (no uint8 round trip)
        f = face01_rgb.float().mul(255).round().clamp(0, 255).flip(1)  # BGR 0..255 (matches CPU input)
        rs = F.grid_sample(f, grids, mode="bilinear", padding_mode="border", align_corners=False)
        b = bases_u8.float()
        out = torch.floor((rs.round() * alphas + b * (255 - alphas)) / 255)
        return out.clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).contiguous()

    gpu_compose_c = torch.compile(gpu_compose, mode="max-autotune-no-cudagraphs", dynamic=False)
    with torch.inference_mode():
        g = gpu_compose_c(img).cpu().numpy()
        # correctness vs CPU path over the clip ROI
        diffs = []
        for i, p in enumerate(plans):
            cy, cx = p["clip_slice"]
            h, w = cy.stop - cy.start, cx.stop - cx.start
            d = np.abs(g[i, :h, :w].astype(int) - cpu_out[i][cy, cx].astype(int))
            diffs.append((d.max(), (d > 1).mean(), (d > 0).mean()))
        R["gpu_vs_cpu_u8"] = {"max_lsb": int(max(d[0] for d in diffs)),
                              "frac_gt1lsb": float(np.mean([d[1] for d in diffs])),
                              "frac_differ": float(np.mean([d[2] for d in diffs]))}
        st = gpu_state(f"gpu compose {cname}")
        R["gpu_compose_eager_events_bs16"] = bench_events(lambda: gpu_compose(img), warmup=10, iters=50)
        R["gpu_compose_compiled_events_bs16"] = bench_events(lambda: gpu_compose_c(img), warmup=10, iters=50)
        out_roi = gpu_compose_c(img)
        R["roi_bytes_per_frame"] = int(HC * WC * 3)
        R["d2h_roi_pinned_events_bs16"] = bench_events(lambda: roi_pinned.copy_(out_roi, non_blocking=True),
                                                       warmup=10, iters=50)
        # full composed frame on GPU (frames resident) -> D2H full frames
        full = torch.zeros((BS, *cases[0]["frame"].shape), dtype=torch.uint8, device=DEV)
        full_pinned = torch.empty(full.shape, dtype=torch.uint8, pin_memory=True)
        R["full_frame_bytes"] = int(full[0].numel())
        R["d2h_fullframe_pinned_events_bs16"] = bench_events(lambda: full_pinned.copy_(full, non_blocking=True),
                                                             warmup=10, iters=50)
        R["gpu_state"] = st
    # CPU side after GPU ROI compose: frame copy + paste ROI
    cv2.setNumThreads(1)
    roi_np = roi_pinned.numpy()

    def paste(i):
        ori = cases[i]["frame"].copy()
        cy, cx = plans[i]["clip_slice"]
        ori[cy, cx] = roi_np[i, :cy.stop - cy.start, :cx.stop - cx.start]
        return ori
    R["cpu_after_gpu_roi_ms_per_frame(copy+paste)"] = tm(paste)

    def paste_inplace(i, buf=[cases[0]["frame"].copy()]):
        cy, cx = plans[i]["clip_slice"]
        buf[0][cy, cx] = roi_np[i, :cy.stop - cy.start, :cx.stop - cx.start]
    R["cpu_after_gpu_roi_ms_per_frame(paste only, reused buffer)"] = tm(paste_inplace)
    cv2.setNumThreads(os.cpu_count())
    res["compose"][cname] = R
    print(cname, json.dumps({k: (round(v, 4) if isinstance(v, float) else
                                 (round(v['median_ms'], 4) if isinstance(v, dict) and 'median_ms' in v else v))
                             for k, v in R.items() if k != "gpu_state"}, default=str), flush=True)
    dump("p4_handoff_compose.json", res)

res["gpu_end"] = gpu_state("end")
dump("p4_handoff_compose.json", res)
