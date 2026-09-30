#!/usr/bin/env python3
"""Unit tests for scripts/musetalk_host_profile.py (startup rework Component A).

CPU-only and stdlib-only: fake repo trees, fake venv trees (site-packages/*.dist-info names),
injected host facts (MUSETALK_HOST_FACTS_JSON), a fake nvidia-smi, and store entries written with
the engine-store component's own scripts/musetalk_engine_keys.py. Never imports torch, never touches
the GPU, never writes outside a TemporaryDirectory.

Run: python3 -m unittest -v test_musetalk_host_profile
"""
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parent
TOOL = REPO / "scripts" / "musetalk_host_profile.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


hp = _load("musetalk_host_profile", TOOL)
ek = _load("musetalk_engine_keys", REPO / "scripts" / "musetalk_engine_keys.py")

GPU_4070S = {"index": 0, "name": "NVIDIA GeForce RTX 4070 SUPER", "compute_capability": "8.9",
             "memory_total_mib": 12282, "memory_used_mib": 500, "power_limit_w": 220.0,
             "power_default_limit_w": 220.0, "driver_version": "595.84", "uuid": "GPU-aaaa-1111"}
GPU_4070TI = dict(GPU_4070S, index=1, name="NVIDIA GeForce RTX 4070 Ti", uuid="GPU-bbbb-2222")
GPU_3090 = dict(GPU_4070S, name="NVIDIA GeForce RTX 3090", compute_capability="8.6", memory_total_mib=24576,
                power_limit_w=350.0, power_default_limit_w=350.0, uuid="GPU-cccc-3333")
GPU_A100 = dict(GPU_4070S, name="NVIDIA A100-SXM4-40GB", compute_capability="8.0", memory_total_mib=40960,
                uuid="GPU-dddd-4444")
GPU_H100 = dict(GPU_4070S, name="NVIDIA H100 80GB HBM3", compute_capability="9.0", memory_total_mib=81559,
                uuid="GPU-eeee-5555")
GPU_T4 = dict(GPU_4070S, name="Tesla T4", compute_capability="7.5", memory_total_mib=15360, uuid="GPU-ffff-6666")
GPU_3070 = dict(GPU_3090, name="NVIDIA GeForce RTX 3070", memory_total_mib=6144, uuid="GPU-9999-7777")
CU121 = {"torch": "2.5.1+cu121", "tensorrt_cu12_bindings": "10.3.0", "tensorrt_cu12": "10.3.0",
         "torch_tensorrt": "2.5.0", "triton": "3.1.0", "aiortc": "1.14.0", "av": "16.1.0", "cffi": "2.1.1",
         "nvidia_cuda_runtime_cu12": "12.1.105"}
CU128 = dict(CU121, torch="2.7.1+cu128", tensorrt_cu12_bindings="10.9.0.34", tensorrt_cu12="10.9.0.34",
             torch_tensorrt="2.7.0", triton="3.3.1", nvidia_cuda_runtime_cu12="12.8.90")

# Exact strings printed by scripts/avatar_manager_parallel.py (see test_log_strings_match_source).
LOG_VAE_TAESD = "✅ VAE decode backend active: taesd\n"
LOG_VAE_TAESD_TRT = "✅ VAE decode backend active: taesd_trt\n"
LOG_VAE_PYTORCH = "ℹ️  VAE decode backend: PyTorch\n"
LOG_UNET_MULTI = "✅ UNet backend active: tensorrt_unet_multi\n"
LOG_UNET_STAGEWISE = "✅ UNet backend active: tensorrt_unet_stagewise\n"
LOG_UNET_PYTORCH = "ℹ️  UNet backend: PyTorch\n"


def make_venv(root: Path, packages: dict, python_version="3.10.12", extra=()):
    sp = root / "lib" / f"python{python_version.rsplit('.', 1)[0]}" / "site-packages"
    sp.mkdir(parents=True)
    (root / "bin").mkdir(parents=True, exist_ok=True)
    (root / "bin" / "python").write_text("")
    (root / "pyvenv.cfg").write_text(f"home = /usr/bin\ninclude-system-site-packages = false\n"
                                     f"version = {python_version}\n")
    for name, version in dict(packages, **dict(extra)).items():
        (sp / f"{name}-{version}.dist-info").mkdir()
    return root


class Tree:
    """A fake repo root + venv + workspace under one TemporaryDirectory."""

    def __init__(self, tmp: Path, packages=CU121, gpus=(GPU_4070S,), ram_total=30000, ram_avail=20000,
                 cpu_effective=31, machine="x86_64", python_version="3.10.12"):
        self.tmp = tmp
        self.repo = tmp / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / ".runtime").mkdir()
        self.ws = tmp / "ws"
        self.ws.mkdir()
        self.venv = make_venv(tmp / "venv", packages, python_version)
        self.facts = {
            "gpus": list(gpus),
            "cpu": {"nproc": 32, "affinity": 32, "cgroup_quota_cpus": None, "effective": cpu_effective},
            "ram": {"mem_total_mb": ram_total, "mem_available_mb": ram_avail, "cgroup_limit_mb": None,
                    "cgroup_available_mb": None, "effective_total_mb": ram_total,
                    "effective_available_mb": ram_avail},
            "disk_free_gb": 100.0, "machine": machine, "os": "Linux",
        }
        self.write_facts()

    def write_facts(self):
        self.facts_path = self.tmp / "facts.json"
        self.facts_path.write_text(json.dumps(self.facts))

    def host_facts(self, cvd=None):
        """Facts in the shape the resolver passes to musetalk_engine_keys."""
        env = {} if cvd is None else {"CUDA_VISIBLE_DEVICES": cvd}
        old = dict(os.environ)
        try:
            os.environ.pop("MUSETALK_HOST_FACTS_JSON", None)
            os.environ["MUSETALK_HOST_FACTS_JSON"] = str(self.facts_path)
            return hp.detect(self.repo, self.venv, environ=env)
        finally:
            os.environ.clear()
            os.environ.update(old)

    # -- store entries (written with the engine-store component's own helpers) --------------
    def _validated(self, kind, facts, batch, directory, **fields):
        fp = ek.make_fingerprint(kind, facts, batch, "built", **fields)
        fp["validation"] = {"passed": True, "status": "passed", "engine_key": fp["engine_key"],
                            "validated_utc": "2026-09-28T00:00:00Z", "mae_max": 0.002, "max_abs_max": 0.3,
                            "capture_dir": "calibration/unet_portable_bs8", "files": 16}
        ek.write_fingerprint(directory, fp)
        return fp

    def add_unet_ts(self, facts=None, key=None):
        facts = facts or self.host_facts()
        key = key or ek.engine_key("unet_ts", facts)
        d = self.repo / "models" / "tensorrt_unet" / key / "bs8"
        d.mkdir(parents=True)
        (d / "unet_trt.ts").write_bytes(b"x" * 100)
        (d / "unet_trt_meta.json").write_text(json.dumps({"type": "unet", "batch_range": [8, 8]}))
        self._validated("unet_ts", facts, 8, d, engine_bytes=100)
        return d

    def add_ts_mode_record(self, d, mode="manual", passed=True):
        """What `unet_engine_store.py validate --kind unet_ts --cudagraphs <mode>` records."""
        fp = ek.read_fingerprint(d)
        record = {"passed": passed, "status": "passed" if passed else "failed", "engine_key": fp["engine_key"]}
        fp["validation"].setdefault("modes", {})[f"cudagraphs_{mode}"] = record
        ek.write_fingerprint(d, fp)

    def add_stagewise(self, batch=16, facts=None):
        facts = facts or self.host_facts()
        key = ek.engine_key("unet_stagewise", facts)
        d = self.repo / "models" / "tensorrt_unet_stagewise" / key / f"bs{batch}"
        d.mkdir(parents=True)
        files = {}
        for block in ek.STAGEWISE_BLOCKS:
            (d / f"{block}.plan").write_bytes(b"p" * 10)
            files[f"{block}.plan"] = 10
        manifest = {"schema": ek.STAGEWISE_MANIFEST_SCHEMA, "batch": batch, "complete": True,
                    "tensorrt_version": "10.3.0", "compute_capability": [8, 9], "gpu": GPU_4070S["name"],
                    "blocks": {b: {"engine_file": f"{b}.plan"} for b in ek.STAGEWISE_BLOCKS}}
        (d / "manifest.json").write_text(json.dumps(manifest))
        self._validated("unet_stagewise", facts, batch, d, files=files,
                        manifest_sha256=ek.sha256_file(d / "manifest.json"))
        return d

    def add_legacy_stagewise(self, batch=16):
        d = self.repo / "models" / "tensorrt_unet_stagewise_sm89" / f"bs{batch}"
        d.mkdir(parents=True)
        manifest = {"schema": ek.STAGEWISE_MANIFEST_SCHEMA, "batch": batch, "complete": True,
                    "tensorrt_version": "10.3.0", "compute_capability": [8, 9], "gpu": GPU_4070S["name"],
                    "blocks": {b: {"engine_file": f"{b}.plan"} for b in ek.STAGEWISE_BLOCKS}}
        (d / "manifest.json").write_text(json.dumps(manifest))
        return d

    def add_taesd_trt(self, batch=8, facts=None, verdict="PASS"):
        facts = facts or self.host_facts()
        key = ek.engine_key("taesd_trt", facts)
        d = self.repo / "models" / "taesd" / "trt" / key / f"bs{batch}"
        d.mkdir(parents=True)
        meta_name = "taesd_trt_0123456789abcdef0123.json"
        files = {}
        for name in ("taesd_trt_0123456789abcdef0123.decoder.plan", "taesd_trt_0123456789abcdef0123.post_bgr_u8.plan"):
            (d / name).write_bytes(b"t" * 8)
            files[name] = 8
        meta = {"schema": "taesd_trt_engine_v1", "key": "0123456789abcdef0123"}
        if verdict:
            meta["gate"] = {"verdict": verdict, "G_TAESD_full_max": 2 if verdict == "PASS" else 5}
        (d / meta_name).write_text(json.dumps(meta))
        files[meta_name] = (d / meta_name).stat().st_size
        self._validated("taesd_trt", facts, batch, d, engine_file=meta_name, files=files,
                        runtime_key="0123456789abcdef0123",
                        runtime_fingerprint={"opt_level": 3, "strongly_typed": False, "batch": batch})
        return d

    def fake_python(self, rc, message=""):
        """An executable venv python stand-in: exits rc (after printing message) whatever it runs."""
        py = self.venv / "bin" / "python"
        py.write_text(f"#!/bin/sh\necho '{message}' >&2\nexit {int(rc)}\n")
        py.chmod(py.stat().st_mode | stat.S_IEXEC)
        return py

    # -- fake serving code ----------------------------------------------------------------
    def add_reader(self, *names):
        body = "import os\n" + "".join(f"V{i} = os.getenv({n!r})\n" for i, n in enumerate(names))
        (self.repo / "scripts" / "fake_reader.py").write_text(body)

    def add_native_vp8(self, h264_marker=False):
        manifest = {"schema_version": 1, "extracted": [{"path": "aiortc/codecs/_vpx.abi3.so", "bytes": 4}],
                    "bundled_notices": []}
        (self.repo / "scripts" / "native_vp8_manifest.json").write_text(json.dumps(manifest))
        d = self.repo / ".runtime" / "native_vp8"
        (d / "aiortc" / "codecs").mkdir(parents=True)
        (d / "aiortc" / "codecs" / "_vpx.abi3.so").write_bytes(b"1234")
        (d / "installation.json").write_text("{}")
        src = ["def validate_native_offer(sdp, description_type='offer'):\n"]
        if h264_marker:
            src.insert(0, "NATIVE_VP8_H264_FALLBACK = True\n")
        else:
            src.append('    raise ValueError("Native VP8 profile requires client VP8 support; '
                       'H264-only offers are unsupported")\n')
        (self.repo / "scripts" / "webrtc_native_vp8.py").write_text("".join(src))

    def recipe(self, text, name="fast300"):
        d = self.repo / "configs" / "recipes"
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{name}.env").write_text(textwrap.dedent(text))

    # -- running the CLI -------------------------------------------------------------------
    def env(self, **extra):
        env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "HOME": str(self.tmp), "LANG": "C.UTF-8",
               "WORKSPACE": str(self.ws), "MUSETALK_HOST_FACTS_JSON": str(self.facts_path)}
        env.update({k: str(v) for k, v in extra.items()})
        return env

    def run(self, *args, **extra):
        return subprocess.run([sys.executable, "-B", str(TOOL)] + list(args), env=self.env(**extra),
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, timeout=60)

    def resolve(self, recipe="fast", **extra):
        out, rep = self.tmp / "resolved.env", self.tmp / "resolved.json"
        args = ["resolve", "--repo-root", str(self.repo), "--venv", str(self.venv), "--out", str(out),
                "--report", str(rep)]
        if recipe:
            args += ["--recipe", recipe]
        proc = self.run(*args, **extra)
        env = {k: v for k, v, _ in hp.parse_env_file(out)} if out.exists() else {}
        report = json.loads(rep.read_text()) if rep.exists() else {}
        return proc, env, report


class Base(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="mt_hostprofile_")
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def tree(self, **kw):
        return Tree(self.tmp, **kw)

    def assertOk(self, proc):
        self.assertEqual(proc.returncode, 0, proc.stderr)


# --------------------------------------------------------------------------- detect
class DetectTests(Base):
    def test_venv_versions_from_dist_info_names(self):
        t = self.tree()
        v = hp.detect_venv(t.venv)
        self.assertEqual(v["python_version"], "3.10.12")
        self.assertEqual(v["torch"], "2.5.1+cu121")
        self.assertEqual(v["torch_cuda_tag"], "cu121")
        self.assertEqual(v["tensorrt"], "10.3.0")  # from tensorrt_cu12_bindings (no plain tensorrt dist)
        self.assertEqual(v["torch_tensorrt"], "2.5.0")
        self.assertEqual((v["aiortc"], v["av"], v["cffi"]), ("1.14.0", "16.1.0", "2.1.1"))

    def test_cuda_tag_from_runtime_when_torch_has_no_local_tag(self):
        venv = make_venv(self.tmp / "v2", {"torch": "2.7.1", "nvidia_cuda_runtime_cu12": "12.8.90"})
        self.assertEqual(hp.detect_venv(venv)["torch_cuda_tag"], "cu128")

    def test_missing_venv(self):
        v = hp.detect_venv(self.tmp / "nope")
        self.assertFalse(v["exists"])
        self.assertIsNone(v["torch"])

    def test_gpu_selection_honours_cuda_visible_devices(self):
        gpus = [GPU_4070S, GPU_4070TI]
        self.assertEqual(hp.select_gpu(gpus, None)[0]["index"], 0)
        self.assertEqual(hp.select_gpu(gpus, "1,0")[0]["index"], 1)
        self.assertEqual(hp.select_gpu(gpus, "GPU-bbbb")[0]["index"], 1)
        self.assertIsNone(hp.select_gpu(gpus, "")[0])
        self.assertIsNone(hp.select_gpu(gpus, "-1")[0])
        self.assertIsNone(hp.select_gpu(gpus, "7")[0])
        self.assertIsNone(hp.select_gpu([], None)[0])

    def test_detect_cli_with_injected_facts(self):
        t = self.tree(gpus=(GPU_4070S, GPU_4070TI))
        proc = t.run("detect", "--repo-root", str(t.repo), "--venv", str(t.venv), CUDA_VISIBLE_DEVICES="1")
        self.assertOk(proc)
        facts = json.loads(proc.stdout)
        self.assertEqual(facts["gpu"]["name"], GPU_4070TI["name"])
        self.assertEqual(facts["engine_facts"]["tensorrt_version"], "10.3.0")
        self.assertEqual(facts["cpu"]["effective"], 31)
        self.assertEqual(facts["schema"], "musetalk_host_facts_v1")

    def test_fake_nvidia_smi_parsing_and_compute_cap_retry(self):
        smi = self.tmp / "nvidia-smi"
        smi.write_text(textwrap.dedent("""\
            #!/bin/sh
            case "$1" in
              *compute_cap*) echo 'Field "compute_cap" is not a valid field to query.' >&2; exit 2;;
            esac
            echo '0, NVIDIA GeForce RTX 3090, 24576, 1200, [N/A], 350.00, 535.1, GPU-cccc'
            """))
        smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
        gpus, info = hp.query_gpus(str(smi))
        self.assertTrue(info["ok"], info)
        self.assertEqual(gpus[0]["name"], "NVIDIA GeForce RTX 3090")
        self.assertIsNone(gpus[0]["compute_capability"])  # retried without the field
        self.assertEqual(gpus[0]["memory_total_mib"], 24576)
        self.assertIsNone(gpus[0]["power_limit_w"])
        self.assertEqual(gpus[0]["power_default_limit_w"], 350.0)

    def test_fake_nvidia_smi_full(self):
        smi = self.tmp / "nvidia-smi"
        smi.write_text("#!/bin/sh\necho '0, NVIDIA GeForce RTX 4070 SUPER, 8.9, 12282, 843, 180.00, 220.00, "
                       "595.84, GPU-dddd'\n")
        smi.chmod(smi.stat().st_mode | stat.S_IEXEC)
        gpus, info = hp.query_gpus(str(smi))
        self.assertEqual(gpus[0]["compute_capability"], "8.9")
        self.assertEqual(gpus[0]["power_limit_w"], 180.0)
        missing, info = hp.query_gpus(str(self.tmp / "absent-smi"))
        self.assertEqual(missing, [])
        self.assertIn("not found", info["error"])

    def test_cgroup_v2_cpu_and_memory(self):
        cg = self.tmp / "cg"
        cg.mkdir()
        (cg / "cpu.max").write_text("3071999 100000\n")
        (cg / "memory.max").write_text(str(29 * 1024 ** 3))
        (cg / "memory.current").write_text(str(25 * 1024 ** 3))
        (cg / "memory.stat").write_text(f"anon 1\nactive_file {1024 ** 3}\ninactive_file {1024 ** 3}\n"
                                        f"slab_reclaimable {512 * 1024 ** 2}\nshmem 999\n")
        self.assertAlmostEqual(hp.cgroup_cpu_quota(cg), 30.72, places=2)
        limit, avail = hp.cgroup_memory(cg)
        self.assertEqual(limit, 29 * 1024 ** 3)
        self.assertEqual(avail, (4 + 2.5) * 1024 ** 3)
        meminfo = self.tmp / "meminfo"
        meminfo.write_text("MemTotal: 31695000 kB\nMemFree: 1 kB\nMemAvailable: 9000000 kB\n")
        ram = hp.detect_ram(meminfo, cg)
        self.assertEqual(ram["effective_total_mb"], 29 * 1024)
        self.assertEqual(ram["effective_available_mb"], min(9000000 // 1024, int(6.5 * 1024)))
        cpu = hp.detect_cpu(cg)
        self.assertLessEqual(cpu["effective"], 31)

    def test_cgroup_unlimited_and_v1(self):
        cg = self.tmp / "cg2"
        cg.mkdir()
        (cg / "cpu.max").write_text("max 100000\n")
        (cg / "memory.max").write_text("max\n")
        self.assertIsNone(hp.cgroup_cpu_quota(cg))
        self.assertEqual(hp.cgroup_memory(cg), (None, None))
        v1 = self.tmp / "cg1"
        (v1 / "cpu").mkdir(parents=True)
        (v1 / "memory").mkdir()
        (v1 / "cpu" / "cpu.cfs_quota_us").write_text("400000\n")
        (v1 / "cpu" / "cpu.cfs_period_us").write_text("100000\n")
        (v1 / "memory" / "memory.limit_in_bytes").write_text(str(16 * 1024 ** 3))
        (v1 / "memory" / "memory.usage_in_bytes").write_text(str(10 * 1024 ** 3))
        (v1 / "memory" / "memory.stat").write_text(f"total_active_file {1024 ** 3}\ntotal_inactive_file 0\n")
        self.assertEqual(hp.cgroup_cpu_quota(v1), 4.0)
        self.assertEqual(hp.cgroup_memory(v1), (16 * 1024 ** 3, 7 * 1024 ** 3))


# --------------------------------------------------------------------------- env files
class EnvFileTests(Base):
    def test_quote_round_trip_and_bash_compat(self):
        value = "it's a \"path\" with $HOME and # hash"
        path = self.tmp / "x.env"
        path.write_text(f"# comment\n\nexport A={hp.shell_single_quote(value)}\nB=plain # trailing\nC=\"dq\"\n")
        parsed = {k: v for k, v, _ in hp.parse_env_file(path)}
        self.assertEqual(parsed, {"A": value, "B": "plain", "C": "dq"})
        out = subprocess.run(["bash", "-c", f"set -a; . {path}; printf '%s' \"$A\""], stdout=subprocess.PIPE,
                             universal_newlines=True, env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")})
        self.assertEqual(out.stdout, value)

    def test_parse_buckets(self):
        self.assertEqual(hp.parse_buckets("16, 8,8"), [8, 16])
        for bad in ("", "a", "0", "8,-8"):
            with self.assertRaises(ValueError):
                hp.parse_buckets(bad)


# --------------------------------------------------------------------------- resolve: fast
class FastRecipeTests(Base):
    def test_fast_recipe_is_eager_without_engine(self):
        t = self.tree()
        proc, env, rep = t.resolve(recipe="fast")  # the default recipe is r5 (R5RecipeTests)
        self.assertOk(proc)
        self.assertEqual(env["MUSETALK_RECIPE"], "fast")
        self.assertEqual(env["MUSETALK_VAE_BACKEND"], "taesd")
        self.assertEqual(env["MUSETALK_TAESD_COMPILE"], "1")
        self.assertEqual(env["MUSETALK_TRT_FALLBACK"], "0")
        self.assertEqual(env["MUSETALK_TRT_ENABLED"], "0")
        self.assertEqual(env["MUSETALK_COMPILE"], "0")
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "eager")
        self.assertEqual(env["MUSETALK_TRT_UNET_ENABLED"], "0")
        self.assertNotIn("MUSETALK_TRT_UNET_PATHS", env)
        for knob in ("HLS_SCHEDULER_FIXED_BATCH_SIZES", "MUSETALK_TAESD_WARMUP_BATCHES",
                     "MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES", "HLS_SCHEDULER_MAX_BATCH",
                     "HLS_SCHEDULER_STARTUP_SLICE_SIZE"):
            self.assertEqual(env[knob], "8", knob)
        self.assertEqual(env["WEBRTC_VP8_ENCODER"], "pyav")
        self.assertEqual(env["WEBRTC_NATIVE_VP8_DIR"], str(t.repo.resolve() / ".runtime" / "native_vp8"))
        self.assertEqual(env["GPU_TOTAL_MEMORY_GB"], "12.0")
        self.assertEqual(env["AVATAR_CACHE_MAX_MEMORY_MB"], "6000")  # 0.2 x 30000
        self.assertEqual(env["HLS_PREP_WORKERS"], "8")
        self.assertEqual(env["PROFILE"], "throughput_record")
        self.assertEqual(env["MUSETALK_BLEND_FIXED_POINT"], "1")
        self.assertEqual(rep["expect"], {"vae": "taesd_compiled", "unet": "eager"})
        self.assertIn("no usable unet_ts engine", rep["unet"]["reason"])
        self.assertNotIn("MUSETALK_CHIN_TRACKER_PYTHON", env)
        # contract report shape
        for key in ("facts", "decisions", "warnings", "errors", "emitted", "levers", "summary", "estimate"):
            self.assertIn(key, rep)

    def test_cpu_and_ram_scaling(self):
        t = self.tree(cpu_effective=6, ram_total=8000)
        proc, env, rep = t.resolve()
        self.assertOk(proc)
        self.assertEqual(env["HLS_COMPOSE_WORKERS"], "3")
        self.assertEqual(env["MUSETALK_AVATAR_LOAD_WORKERS"], "6")
        self.assertEqual(env["AVATAR_CACHE_MAX_MEMORY_MB"], "2048")  # clamp floor
        self.assertTrue(any("< 16 GB total" in w for w in rep["warnings"]))
        (self.tmp / "big").mkdir()
        t2 = Tree(self.tmp / "big", ram_total=200000, cpu_effective=3)
        proc, env, _ = t2.resolve()
        self.assertEqual(env["AVATAR_CACHE_MAX_MEMORY_MB"], "16384")  # clamp ceiling
        self.assertEqual(env["HLS_ENCODE_WORKERS"], "2")

    def test_trt_selected_with_validated_engine(self):
        t = self.tree()
        d = t.add_unet_ts()
        proc, env, rep = t.resolve()
        self.assertOk(proc)
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "trt")
        self.assertEqual(env["MUSETALK_TRT_UNET_ENABLED"], "1")
        self.assertEqual(env["MUSETALK_TRT_UNET_PATHS"], f"8:{d.resolve() / 'unet_trt.ts'}")
        self.assertEqual(rep["expect"]["unet"], "trt")
        self.assertEqual(rep["unet"]["engine"]["match"], "exact")

    def test_unvalidated_engine_is_not_used(self):
        t = self.tree()
        d = t.add_unet_ts()
        fp = json.loads((d / "fingerprint.json").read_text())
        fp["validation"]["passed"] = False
        (d / "fingerprint.json").write_text(json.dumps(fp))
        proc, env, rep = t.resolve()
        self.assertOk(proc)
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "eager")
        proc, env, rep = t.resolve(MUSETALK_UNET_MODE="trt")
        self.assertEqual(proc.returncode, 2)
        self.assertTrue(any("no usable unet_ts engine" in e for e in rep["errors"]))
        self.assertIn("RESOLUTION FAILED", (self.tmp / "resolved.env").read_text())
        self.assertEqual(env, {})

    def test_same_cc_engine_from_other_gpu_model_warns(self):
        t = self.tree()
        other = dict(t.host_facts(), gpu=GPU_4070TI)
        other["engine_facts"] = hp.engine_facts_from(other)
        t.add_unet_ts(facts=other)
        proc, env, rep = t.resolve()
        self.assertOk(proc)
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "trt")
        self.assertEqual(rep["unet"]["engine"]["match"], "same_cc")
        self.assertTrue(any("same-sm engine" in w for w in rep["warnings"]))

    def test_low_ram_auto_falls_back_trt_mode_warns(self):
        t = self.tree(ram_avail=6000)
        t.add_unet_ts()
        proc, env, rep = t.resolve()
        self.assertOk(proc)
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "eager")
        self.assertIn("MemAvailable", rep["unet"]["reason"])
        proc, env, rep = t.resolve(MUSETALK_UNET_MODE="trt")
        self.assertOk(proc)
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "trt")
        self.assertTrue(any("MemAvailable" in w for w in rep["warnings"]))
        proc, env, rep = t.resolve(MUSETALK_TRT_UNET_MIN_MEM_AVAILABLE_GB="4")
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "trt")

    def test_low_vram_auto_eager(self):
        t = self.tree(gpus=(dict(GPU_4070S, memory_total_mib=6144),))
        t.add_unet_ts()
        proc, env, rep = t.resolve()
        self.assertOk(proc)
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "eager")
        self.assertTrue(any("VRAM" in w for w in rep["warnings"]))

    def test_eight_gb_card_passes_nominal_threshold(self):
        t = self.tree(gpus=(dict(GPU_4070S, memory_total_mib=8188),))
        t.add_unet_ts()
        proc, env, _ = t.resolve()
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "trt")

    def test_caller_buckets_couple_dependents(self):
        t = self.tree()
        t.add_unet_ts()
        proc, env, rep = t.resolve(HLS_SCHEDULER_FIXED_BATCH_SIZES="16")
        self.assertOk(proc)
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "trt")  # 16 = 2 x the bs8 engine
        self.assertEqual(env["HLS_SCHEDULER_FIXED_BATCH_SIZES"], "16")
        self.assertEqual(env["MUSETALK_TAESD_WARMUP_BATCHES"], "16")
        self.assertEqual(env["MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES"], "16")
        self.assertEqual(env["HLS_SCHEDULER_MAX_BATCH"], "16")
        self.assertEqual(env["HLS_SCHEDULER_STARTUP_SLICE_SIZE"], "8")
        dec = {d["knob"]: d for d in rep["decisions"]}
        self.assertEqual(dec["HLS_SCHEDULER_FIXED_BATCH_SIZES"]["source"], "caller")

    def test_non_multiple_buckets(self):
        t = self.tree()
        t.add_unet_ts()
        proc, env, rep = t.resolve(HLS_SCHEDULER_FIXED_BATCH_SIZES="4,8")
        self.assertOk(proc)
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "eager")
        self.assertIn("not multiples", rep["unet"]["reason"])
        self.assertEqual(env["MUSETALK_TAESD_WARMUP_BATCHES"], "4,8")
        proc, env, rep = t.resolve(HLS_SCHEDULER_FIXED_BATCH_SIZES="4,8", MUSETALK_UNET_MODE="trt")
        self.assertEqual(proc.returncode, 2)
        proc, env, rep = t.resolve(HLS_SCHEDULER_FIXED_BATCH_SIZES="8", HLS_SCHEDULER_MAX_BATCH="12",
                                   MUSETALK_UNET_MODE="trt")
        self.assertEqual(proc.returncode, 2)
        proc, env, rep = t.resolve(HLS_SCHEDULER_FIXED_BATCH_SIZES="8,x")
        self.assertEqual(proc.returncode, 2)

    def test_mismatched_caller_warmup_warns(self):
        t = self.tree()
        proc, env, rep = t.resolve(MUSETALK_TAESD_WARMUP_BATCHES="8,16")
        self.assertOk(proc)
        self.assertEqual(env["MUSETALK_TAESD_WARMUP_BATCHES"], "8,16")
        self.assertTrue(any("MUSETALK_TAESD_WARMUP_BATCHES=8,16 differs" in w for w in rep["warnings"]))

    def test_layering_caller_over_overrides_over_resolved(self):
        t = self.tree()
        t.add_reader("WEBRTC_IDLE_FRAME_CACHE")
        o1, o2 = self.tmp / "o1.env", self.tmp / "o2.env"
        o1.write_text("export HLS_SCHEDULER_FIXED_BATCH_SIZES=16\nAVATAR_CACHE_MAX_MEMORY_MB=4000\n"
                      "WEBRTC_IDLE_FRAME_CACHE=1\n")
        o2.write_text("HLS_SCHEDULER_FIXED_BATCH_SIZES=24\nHLS_MAX_PENDING_JOBS=30\n")
        proc, env, rep = t.resolve(MUSETALK_ENV_OVERRIDES_FILE=f"{o1}:{self.tmp / 'missing.env'}:{o2}",
                                   AVATAR_CACHE_MAX_MEMORY_MB="1234")
        self.assertOk(proc)
        self.assertEqual(env["AVATAR_CACHE_MAX_MEMORY_MB"], "1234")  # caller > overrides
        self.assertEqual(env["HLS_SCHEDULER_FIXED_BATCH_SIZES"], "16")  # first overrides file wins
        self.assertEqual(env["MUSETALK_TAESD_WARMUP_BATCHES"], "16")  # dependents from the overrides value
        self.assertEqual(env["HLS_MAX_PENDING_JOBS"], "30")
        self.assertNotIn("WEBRTC_IDLE_FRAME_CACHE", env)  # pass-through levers are never emitted
        dec = {d["knob"]: d for d in rep["decisions"]}
        self.assertEqual(dec["AVATAR_CACHE_MAX_MEMORY_MB"]["source"], "caller")
        self.assertTrue(dec["HLS_SCHEDULER_FIXED_BATCH_SIZES"]["source"].startswith("overrides:"))
        self.assertEqual(dec["HLS_PREP_WORKERS"]["source"], "resolver")
        lev = {l["name"]: l for l in rep["levers"]}
        self.assertEqual(lev["WEBRTC_IDLE_FRAME_CACHE"]["effective"], "1")
        self.assertTrue(lev["WEBRTC_IDLE_FRAME_CACHE"]["source"].startswith("overrides:"))
        self.assertEqual(rep["layers"]["overrides_files_used"], [str(o1), str(o2)])

    def test_overrides_preloaded_by_launcher_are_attributed(self):
        t = self.tree()
        o = t.repo / ".runtime" / "musetalk_overrides.env"
        o.write_text("HLS_MAX_PENDING_JOBS=40\n")
        proc, env, rep = t.resolve(HLS_MAX_PENDING_JOBS="40")  # launcher exported it already
        dec = {d["knob"]: d for d in rep["decisions"]}
        self.assertTrue(dec["HLS_MAX_PENDING_JOBS"]["source"].startswith("overrides:"))
        # scripts/run_musetalk_server.sh passes the layer of every key it exported
        proc, env, rep = t.resolve(HLS_MAX_PENDING_JOBS="40", MUSETALK_ENV_OVERRIDE_KEYS="",
                                   MUSETALK_ENV_CALLER_KEYS="HLS_MAX_PENDING_JOBS")
        dec = {d["knob"]: d for d in rep["decisions"]}
        self.assertEqual(dec["HLS_MAX_PENDING_JOBS"]["source"], "caller")
        proc, env, rep = t.resolve(HLS_MAX_PENDING_JOBS="41", MUSETALK_ENV_OVERRIDE_KEYS="HLS_MAX_PENDING_JOBS")
        dec = {d["knob"]: d for d in rep["decisions"]}
        self.assertTrue(dec["HLS_MAX_PENDING_JOBS"]["source"].startswith("overrides:"))

    def test_empty_caller_value_warns(self):
        t = self.tree()
        proc, env, rep = t.resolve(MUSETALK_VAE_BACKEND="")
        self.assertOk(proc)
        self.assertEqual(env["MUSETALK_VAE_BACKEND"], "")
        self.assertEqual(rep["expect"]["vae"], "pytorch")
        self.assertTrue(any("EMPTY" in w for w in rep["warnings"]))

    def test_caller_stagewise_respected_without_ts_paths(self):
        t = self.tree()
        t.add_unet_ts()
        d = t.add_stagewise(batch=16)
        proc, env, rep = t.resolve(MUSETALK_UNET_BACKEND="trt_stagewise")
        self.assertOk(proc)
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "trt_stagewise")
        self.assertEqual(env["MUSETALK_TRT_UNET_ENABLED"], "0")
        self.assertNotIn("MUSETALK_TRT_UNET_PATHS", env)
        self.assertEqual(env["MUSETALK_UNET_STAGEWISE_CACHE_DIR"], str(d.parent.resolve()))
        self.assertEqual(env["MUSETALK_UNET_STAGEWISE_BATCH"], "16")
        self.assertEqual(env["HLS_SCHEDULER_FIXED_BATCH_SIZES"], "16")
        self.assertEqual(rep["expect"]["unet"], "trt_stagewise")

    def test_caller_stagewise_without_any_engine_is_an_error(self):
        t = self.tree()
        proc, env, rep = t.resolve(MUSETALK_UNET_BACKEND="trt_stagewise")
        self.assertEqual(proc.returncode, 2)
        t.add_legacy_stagewise(16)
        proc, env, rep = t.resolve(MUSETALK_UNET_BACKEND="trt_stagewise")
        self.assertOk(proc)  # the unvalidated legacy set loads, with a warning
        self.assertTrue(any("no validation record" in w for w in rep["warnings"]))

    def test_caller_trt_backend_and_paths(self):
        t = self.tree()
        proc, env, rep = t.resolve(MUSETALK_UNET_BACKEND="trt")
        self.assertEqual(proc.returncode, 2)  # explicit TRT without an engine
        ts = self.tmp / "custom.ts"
        ts.write_bytes(b"x")
        proc, env, rep = t.resolve(MUSETALK_UNET_BACKEND="trt", MUSETALK_TRT_UNET_PATHS=f"8:{ts}")
        self.assertOk(proc)
        self.assertEqual(env["MUSETALK_TRT_UNET_PATHS"], f"8:{ts}")
        self.assertTrue(any("not checked against the engine store" in w for w in rep["warnings"]))
        proc, env, rep = t.resolve(MUSETALK_UNET_BACKEND="trt", MUSETALK_TRT_UNET_PATHS=f"8:{ts}.missing")
        self.assertEqual(proc.returncode, 2)

    def test_eager_mode(self):
        t = self.tree()
        t.add_unet_ts()
        proc, env, rep = t.resolve(MUSETALK_UNET_MODE="eager")
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "eager")
        proc, env, rep = t.resolve(MUSETALK_UNET_MODE="bogus")
        self.assertEqual(proc.returncode, 2)

    def test_blackwell_needs_cu128(self):
        gpu = dict(GPU_4070S, name="NVIDIA GeForce RTX 5090", compute_capability="12.0", memory_total_mib=32607)
        t = self.tree(gpus=(gpu,))
        proc, env, rep = t.resolve()
        self.assertEqual(proc.returncode, 2)
        self.assertTrue(any("cu128" in e for e in rep["errors"]))
        t2dir = self.tmp / "bw"
        t2dir.mkdir()
        t2 = Tree(t2dir, packages=CU128, gpus=(gpu,))
        proc, env, rep = t2.resolve()
        self.assertOk(proc)
        self.assertEqual(rep["engines"]["unet_ts"]["key"], "sm120-nvidia-geforce-rtx-5090-trt10.9.0.34-tt2.7.0")

    def test_no_gpu(self):
        t = self.tree(gpus=())
        proc, env, rep = t.resolve()
        self.assertEqual(proc.returncode, 2)
        proc, env, rep = t.resolve(MUSETALK_ALLOW_NO_GPU="1")
        self.assertOk(proc)
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "eager")
        self.assertNotIn("GPU_TOTAL_MEMORY_GB", env)

    def test_cuda_visible_devices_hides_gpu(self):
        t = self.tree()
        proc, env, rep = t.resolve(CUDA_VISIBLE_DEVICES="")
        self.assertEqual(proc.returncode, 2)
        self.assertTrue(any("hides every GPU" in e for e in rep["errors"]))

    def test_power_cap_warning(self):
        t = self.tree(gpus=(dict(GPU_4070S, power_limit_w=150.0),))
        proc, env, rep = t.resolve()
        self.assertTrue(any("power-capped" in w for w in rep["warnings"]))

    def test_invalid_encoder_values_are_errors(self):
        t = self.tree()
        self.assertEqual(t.resolve(WEBRTC_VP8_ENCODER="vp9")[0].returncode, 2)
        self.assertEqual(t.resolve(WEBRTC_H264_IMPL="qsv")[0].returncode, 2)
        proc, env, rep = t.resolve(WEBRTC_H264_IMPL="nvenc")
        self.assertOk(proc)
        lev = {l["name"]: l for l in rep["levers"]}
        self.assertEqual((lev["WEBRTC_H264_IMPL"]["effective"], lev["WEBRTC_H264_IMPL"]["source"]),
                         ("nvenc", "caller"))

    def test_caller_native_vp8_warns_about_h264_clients(self):
        t = self.tree()
        t.add_native_vp8()
        proc, env, rep = t.resolve(WEBRTC_VP8_ENCODER="native")
        self.assertOk(proc)
        self.assertEqual(env["WEBRTC_VP8_ENCODER"], "native")
        self.assertTrue(any("H264-only WebRTC offers get HTTP 400" in w for w in rep["warnings"]))

    def test_selftest_compile_failure_and_estimate(self):
        t = self.tree()
        t.add_unet_ts()
        st = {"schema": "musetalk_gpu_selftest_v1", "gpu": {"name": GPU_4070S["name"], "compute_capability": "8.9"},
              "torch_version": "2.5.1+cu121", "cuda_ok": True,
              "taesd": {"compile_ok": False, "compile_mode": "max-autotune", "warmup_s": None, "ms_bs8": None,
                        "eager_ms_bs8": 12.4, "error": "InductorError"},
              "unet_eager": {"ok": True, "ms_bs8": 37.0, "error": None}, "trt_import_ok": True}
        (t.repo / ".runtime" / "gpu_selftest.json").write_text(json.dumps(st))
        proc, env, rep = t.resolve()
        self.assertOk(proc)
        self.assertEqual(env["MUSETALK_TAESD_COMPILE"], "0")
        self.assertTrue(rep["estimate"]["available"])
        self.assertIn("GPU-path", rep["estimate"]["line"])
        self.assertAlmostEqual(rep["estimate"]["gpu_path_fps"], 8000 / (37.0 * hp.TS_VS_EAGER_UNET_RATIO + 12.4), 0)
        st["torch_version"] = "2.7.1+cu128"  # another venv: ignored
        (t.repo / ".runtime" / "gpu_selftest.json").write_text(json.dumps(st))
        proc, env, rep = t.resolve()
        self.assertEqual(env["MUSETALK_TAESD_COMPILE"], "1")
        self.assertFalse(rep["estimate"]["available"])
        # MUSETALK_GPU_SELFTEST_FILE (the installer's --selftest-out variable) points elsewhere.
        st["torch_version"] = "2.5.1+cu121"
        st["taesd"].update(compile_ok=True, ms_bs8=6.8)
        other = t.tmp / "elsewhere_selftest.json"
        other.write_text(json.dumps(st))
        proc, env, rep = t.resolve(MUSETALK_GPU_SELFTEST_FILE=other)
        self.assertOk(proc)
        self.assertEqual(env["MUSETALK_TAESD_COMPILE"], "1")
        self.assertTrue(rep["estimate"]["available"])
        self.assertAlmostEqual(rep["estimate"]["gpu_path_fps"], 8000 / (37.0 * hp.TS_VS_EAGER_UNET_RATIO + 6.8), 0)

    def test_chin_tracker_python(self):
        t = self.tree()
        proc, env, rep = t.resolve()
        self.assertNotIn("MUSETALK_CHIN_TRACKER_PYTHON", env)
        chin = make_venv(t.ws / ".venvs" / "musetalk_chin_tools",
                         {"mediapipe": "0.10.9", "numpy": "1.26.4", "opencv_python": "4.9.0.80"})
        proc, env, rep = t.resolve()
        self.assertEqual(env["MUSETALK_CHIN_TRACKER_PYTHON"], str(chin / "bin" / "python"))
        proc, env, rep = t.resolve(MUSETALK_CHIN_TRACKER_PYTHON="/workspace/SoulX-FlashHead/.venv/bin/python")
        self.assertEqual(env["MUSETALK_CHIN_TRACKER_PYTHON"], "/workspace/SoulX-FlashHead/.venv/bin/python")
        self.assertTrue(any("another project's venv" in w for w in rep["warnings"]))

    def test_legacy_recipe_emits_only_recipe(self):
        t = self.tree(gpus=())
        proc, env, rep = t.resolve(recipe="legacy_int8")
        self.assertOk(proc)
        self.assertEqual(env, {"MUSETALK_RECIPE": "legacy_int8"})
        proc, env, rep = t.resolve(recipe=None, MUSETALK_RECIPE="legacy_int8")
        self.assertEqual(env, {"MUSETALK_RECIPE": "legacy_int8"})

    def test_recipe_from_overrides_file(self):
        t = self.tree()
        (t.repo / ".runtime" / "musetalk_overrides.env").write_text("MUSETALK_RECIPE=legacy_int8\n")
        proc, env, rep = t.resolve(recipe=None)
        self.assertEqual(env["MUSETALK_RECIPE"], "legacy_int8")


# --------------------------------------------------------------------------- resolve: fast300
class Fast300Tests(Base):
    ALL_LEVERS = """\
        # 300 fps levers
        MUSETALK_UNET_BACKEND=trt_stagewise
        MUSETALK_UNET_STAGEWISE_BATCH=16
        MUSETALK_TAESD_BACKEND=trt
        MUSETALK_TRT_UNET_CUDAGRAPHS=manual
        MUSETALK_FREE_EAGER_UNET=1
        WEBRTC_VP8_ENCODER=native
        WEBRTC_NATIVE_VP8_THREADS=1
        WEBRTC_H264_IMPL=x264tuned
        WEBRTC_NONBLOCKING_HANDOFF=1
        HLS_SCHEDULER_POLICY=edf
        HLS_GPU_PIPELINE_DEPTH=2
        MUSETALK_VAE_BACKEND=pytorch
        WEBRTC_HANDOFF_VERIFY=1
        NOT_READ_ANYWHERE=1
        """

    def levers(self, rep):
        return {i["name"]: i for i in rep["recipe_levers"]}

    def test_commented_recipe_equals_fast(self):
        t = self.tree()
        t.add_unet_ts()
        t.recipe("# MUSETALK_UNET_BACKEND=trt_stagewise\n# MUSETALK_TAESD_BACKEND=trt\n")
        _, fast, _ = t.resolve("fast")
        proc, f300, rep = t.resolve("fast300")
        self.assertOk(proc)
        self.assertEqual(f300.pop("MUSETALK_RECIPE"), "fast300")
        fast.pop("MUSETALK_RECIPE")
        self.assertEqual(fast, f300)
        self.assertTrue(any("all commented out" in n for n in rep["notes"]))

    def test_missing_recipe_file_warns(self):
        t = self.tree()
        proc, env, rep = t.resolve("fast300")
        self.assertOk(proc)
        self.assertTrue(any("fast300 == fast" in w for w in rep["warnings"]))

    def test_levers_dropped_without_prerequisites(self):
        t = self.tree()
        t.add_unet_ts()
        t.add_native_vp8()
        t.add_reader("WEBRTC_NONBLOCKING_HANDOFF", "HLS_SCHEDULER_POLICY", "WEBRTC_H264_IMPL",
                     "MUSETALK_TRT_UNET_CUDAGRAPHS", "MUSETALK_FREE_EAGER_UNET")
        (t.repo / "scripts" / "webrtc_h264_override.py").write_text("")
        t.recipe(self.ALL_LEVERS)
        proc, env, rep = t.resolve("fast300")
        self.assertOk(proc)
        lv = self.levers(rep)
        self.assertEqual(lv["MUSETALK_UNET_BACKEND"]["status"], "dropped")
        self.assertIn("no validated unet_stagewise engine", lv["MUSETALK_UNET_BACKEND"]["reason"])
        self.assertEqual(lv["MUSETALK_TAESD_BACKEND"]["status"], "dropped")
        self.assertEqual(lv["MUSETALK_TRT_UNET_CUDAGRAPHS"]["status"], "enabled")  # .ts TRT UNet active
        self.assertEqual(lv["MUSETALK_FREE_EAGER_UNET"]["status"], "enabled")
        self.assertEqual(lv["WEBRTC_VP8_ENCODER"]["status"], "dropped")
        self.assertIn("H264-only", lv["WEBRTC_VP8_ENCODER"]["reason"])
        self.assertEqual(lv["WEBRTC_NATIVE_VP8_THREADS"]["status"], "dropped")
        self.assertEqual(lv["WEBRTC_H264_IMPL"]["status"], "enabled")
        self.assertEqual(lv["WEBRTC_NONBLOCKING_HANDOFF"]["status"], "enabled")
        self.assertEqual(lv["HLS_SCHEDULER_POLICY"]["status"], "enabled")
        self.assertEqual(lv["HLS_GPU_PIPELINE_DEPTH"]["status"], "dropped")  # no code reads it here
        self.assertEqual(lv["MUSETALK_VAE_BACKEND"]["status"], "dropped")  # protected
        self.assertEqual(lv["WEBRTC_HANDOFF_VERIFY"]["status"], "dropped")  # debug-only
        self.assertEqual(lv["NOT_READ_ANYWHERE"]["status"], "dropped")
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "trt")
        self.assertEqual(env["MUSETALK_TRT_UNET_CUDAGRAPHS"], "manual")
        self.assertEqual(env["WEBRTC_VP8_ENCODER"], "pyav")
        self.assertEqual(env["WEBRTC_NONBLOCKING_HANDOFF"], "1")
        self.assertEqual(env["MUSETALK_VAE_BACKEND"], "taesd")
        self.assertNotIn("NOT_READ_ANYWHERE", env)
        self.assertNotIn("WEBRTC_HANDOFF_VERIFY", env)
        for item in rep["recipe_levers"]:
            self.assertIn(item["status"], ("enabled", "dropped", "overridden"), item)
            self.assertTrue(item["reason"], item)

    def test_engine_levers_enabled_when_validated(self):
        t = self.tree()
        t.add_unet_ts()
        sw = t.add_stagewise(16)
        taesd = t.add_taesd_trt(8)
        t.add_reader("MUSETALK_FREE_EAGER_UNET")
        t.recipe(self.ALL_LEVERS)
        proc, env, rep = t.resolve("fast300")
        self.assertOk(proc)
        lv = self.levers(rep)
        self.assertEqual(lv["MUSETALK_UNET_BACKEND"]["status"], "enabled")
        self.assertEqual(lv["MUSETALK_UNET_STAGEWISE_BATCH"]["status"], "enabled")
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "trt_stagewise")
        self.assertEqual(env["MUSETALK_TRT_UNET_ENABLED"], "0")
        self.assertNotIn("MUSETALK_TRT_UNET_PATHS", env)
        self.assertEqual(env["MUSETALK_UNET_STAGEWISE_CACHE_DIR"], str(sw.parent.resolve()))
        self.assertEqual(env["HLS_SCHEDULER_FIXED_BATCH_SIZES"], "16")
        self.assertEqual(env["MUSETALK_TAESD_WARMUP_BATCHES"], "16")
        self.assertEqual(env["HLS_SCHEDULER_MAX_BATCH"], "16")
        self.assertEqual(env["MUSETALK_TAESD_BACKEND"], "trt")
        self.assertEqual(env["MUSETALK_TAESD_TRT_DIR"], str(taesd.resolve()))
        self.assertEqual(env["MUSETALK_TAESD_TRT_BUILD"], "0")
        self.assertEqual(env["MUSETALK_TAESD_TRT_BATCH"], "8")
        self.assertEqual(env["MUSETALK_TAESD_TRT_OPT_LEVEL"], "3")
        self.assertEqual(lv["MUSETALK_TRT_UNET_CUDAGRAPHS"]["status"], "dropped")  # stagewise, not .ts
        self.assertEqual(lv["MUSETALK_FREE_EAGER_UNET"]["status"], "enabled")
        self.assertEqual(rep["expect"], {"vae": "taesd_trt", "unet": "trt_stagewise"})

    def test_caller_wins_over_recipe_lever(self):
        t = self.tree()
        t.add_stagewise(16)
        t.recipe(self.ALL_LEVERS)
        proc, env, rep = t.resolve("fast300", MUSETALK_UNET_MODE="eager", MUSETALK_TAESD_BACKEND="compiled")
        self.assertOk(proc)
        lv = self.levers(rep)
        self.assertEqual(lv["MUSETALK_UNET_BACKEND"]["status"], "dropped")
        self.assertIn("MUSETALK_UNET_MODE=eager", lv["MUSETALK_UNET_BACKEND"]["reason"])
        self.assertEqual(lv["MUSETALK_TAESD_BACKEND"]["status"], "overridden")
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "eager")
        self.assertEqual(rep["expect"]["vae"], "taesd_compiled")

    def test_recipe_bucket_line_couples(self):
        t = self.tree()
        t.add_unet_ts()
        t.recipe("HLS_SCHEDULER_FIXED_BATCH_SIZES=16\nHLS_SCHEDULER_MAX_BATCH=16\n")
        proc, env, rep = t.resolve("fast300")
        self.assertOk(proc)
        self.assertEqual(env["HLS_SCHEDULER_FIXED_BATCH_SIZES"], "16")
        self.assertEqual(env["MUSETALK_TAESD_WARMUP_BATCHES"], "16")
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "trt")
        t.recipe("HLS_SCHEDULER_FIXED_BATCH_SIZES=8,oops\n")
        proc, env, rep = t.resolve("fast300")
        self.assertOk(proc)
        self.assertEqual(self.levers(rep)["HLS_SCHEDULER_FIXED_BATCH_SIZES"]["status"], "dropped")
        self.assertEqual(env["HLS_SCHEDULER_FIXED_BATCH_SIZES"], "8")

    def test_native_vp8_enabled_only_with_h264_path_and_preflight(self):
        t = self.tree()
        t.fake_python(0)
        t.add_native_vp8(h264_marker=True)
        t.recipe("WEBRTC_VP8_ENCODER=native\nWEBRTC_NATIVE_VP8_THREADS=1\n")
        t.add_reader("WEBRTC_NATIVE_VP8_THREADS")
        proc, env, rep = t.resolve("fast300")
        self.assertOk(proc)
        self.assertEqual(env["WEBRTC_VP8_ENCODER"], "native")
        self.assertEqual(env["WEBRTC_NATIVE_VP8_THREADS"], "1")
        # same tree but the venv pins drift: dropped
        t2dir = self.tmp / "drift"
        t2dir.mkdir()
        t2 = Tree(t2dir, packages=dict(CU121, av="14.0.0"))
        t2.fake_python(0)
        t2.add_native_vp8(h264_marker=True)
        t2.recipe("WEBRTC_VP8_ENCODER=native\n")
        proc, env, rep = t2.resolve("fast300")
        self.assertEqual(env["WEBRTC_VP8_ENCODER"], "pyav")
        self.assertIn("av 14.0.0", self.levers(rep)["WEBRTC_VP8_ENCODER"]["reason"])
        # static prerequisites fine but the real preflight (venv python) fails
        t3dir = self.tmp / "pf"
        t3dir.mkdir()
        t3 = Tree(t3dir)
        t3.fake_python(1, "RuntimeError: Native VP8 hash mismatch")
        t3.add_native_vp8(h264_marker=True)
        t3.recipe("WEBRTC_VP8_ENCODER=native\n")
        proc, env, rep = t3.resolve("fast300")
        self.assertEqual(env["WEBRTC_VP8_ENCODER"], "pyav")
        self.assertIn("hash mismatch", self.levers(rep)["WEBRTC_VP8_ENCODER"]["reason"])

    def test_legacy_stagewise_set_needs_adoption(self):
        t = self.tree()
        t.add_legacy_stagewise(16)
        t.recipe("MUSETALK_UNET_BACKEND=trt_stagewise\n")
        proc, env, rep = t.resolve("fast300")
        self.assertOk(proc)
        reason = self.levers(rep)["MUSETALK_UNET_BACKEND"]["reason"]
        self.assertIn("adopt", reason)
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "eager")


REAL_RECIPE = REPO / "configs" / "recipes" / "fast300.env"


def uncomment_groups(text, *groups):
    """Switch on every '#KEY=VALUE' line of the named @lever groups (the orchestrator's action)."""
    out, active = [], False
    for line in text.splitlines():
        match = hp._LEVER_GROUP_RE.match(line)
        if match:
            active = match.group(1) in groups
        elif not line.strip():
            active = False
        elif active and len(line) > 1 and line[0] == "#" and line[1].isupper():
            line = line[1:]
        out.append(line)
    return "\n".join(out) + "\n"


@unittest.skipUnless(REAL_RECIPE.is_file(), "configs/recipes/fast300.env not present")
class LeverGroupTests(Base):
    """The tracked recipe file's '# @lever <group> requires=...' groups are atomic."""

    def tree_with_code(self, **kw):
        t = self.tree(**kw)
        for rel, key in (("scripts/unet_stagewise_trt.py", "MUSETALK_UNET_BACKEND"),
                         ("scripts/vae_fast_decoder.py", "MUSETALK_TAESD_BACKEND"),
                         ("scripts/trt_runtime.py", "MUSETALK_TRT_UNET_CUDAGRAPHS"),
                         ("scripts/avatar_manager_parallel.py", "MUSETALK_FREE_EAGER_UNET"),
                         ("scripts/api_avatar.py", "MUSETALK_AVATAR_MASK_CHANNELS MUSETALK_AVATAR_FRAME_STORE"),
                         ("scripts/webrtc_media_flags.py", "WEBRTC_NONBLOCKING_HANDOFF MUSETALK_THREAD_CAPS"),
                         ("scripts/hls_gpu_scheduler.py", "HLS_GPU_PIPELINE_DEPTH HLS_SCHEDULER_POLICY"),
                         ("scripts/webrtc_h264_override.py", "WEBRTC_H264_IMPL")):
            (t.repo / rel).write_text(f"# reads {key}\n")
        return t

    def use(self, t, *groups):
        t.recipe(uncomment_groups(REAL_RECIPE.read_text(), *groups))

    def groups(self, rep):
        return {g["name"]: g for g in rep["recipe_groups"]}

    def test_tracked_file_is_all_off_and_equals_fast(self):
        t = self.tree_with_code()
        t.add_unet_ts()
        self.use(t)
        _, fast, _ = t.resolve("fast")
        proc, f300, rep = t.resolve("fast300")
        self.assertOk(proc)
        fast.pop("MUSETALK_RECIPE"), f300.pop("MUSETALK_RECIPE")
        self.assertEqual(fast, f300)
        self.assertTrue(rep["recipe_groups"])
        self.assertEqual({g["status"] for g in rep["recipe_groups"]}, {"off"})

    def test_stagewise_group_is_atomic(self):
        t = self.tree_with_code()
        self.use(t, "stagewise_unet", "free_eager_unet")
        proc, env, rep = t.resolve("fast300")
        self.assertOk(proc)
        g = self.groups(rep)
        self.assertEqual(g["stagewise_unet"]["status"], "dropped")
        self.assertIn("engine:unet_stagewise", g["stagewise_unet"]["reason"])
        self.assertEqual(env["HLS_SCHEDULER_FIXED_BATCH_SIZES"], "8")  # the group's bucket line went with it
        self.assertEqual(g["free_eager_unet"]["status"], "dropped")  # unet:trt_any fails with eager
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "eager")
        self.assertNotIn("MUSETALK_FREE_EAGER_UNET", env)
        sw = t.add_stagewise(16)
        proc, env, rep = t.resolve("fast300")
        g = self.groups(rep)
        self.assertEqual(g["stagewise_unet"]["status"], "enabled")
        self.assertEqual(g["free_eager_unet"]["status"], "enabled")
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "trt_stagewise")
        self.assertEqual(env["MUSETALK_UNET_STAGEWISE_CACHE_DIR"], str(sw.parent.resolve()))
        self.assertEqual(env["HLS_SCHEDULER_FIXED_BATCH_SIZES"], "16")
        self.assertEqual(env["MUSETALK_TAESD_WARMUP_BATCHES"], "16")
        self.assertEqual(env["MUSETALK_FREE_EAGER_UNET"], "1")

    def test_state_prerequisite_needs_a_second_pass(self):
        t = self.tree_with_code()
        self.use(t, "ts_unet_cudagraphs")
        proc, env, rep = t.resolve("fast300")  # no .ts engine: UNet eager -> engine:unet_ts fails
        self.assertOk(proc)
        self.assertEqual(self.groups(rep)["ts_unet_cudagraphs"]["status"], "dropped")
        self.assertNotIn("MUSETALK_TRT_UNET_CUDAGRAPHS", env)
        d = t.add_unet_ts()
        proc, env, rep = t.resolve("fast300")  # .ts validated, but not in cudagraphs mode
        self.assertOk(proc)
        g = self.groups(rep)["ts_unet_cudagraphs"]
        self.assertEqual(g["status"], "dropped")
        self.assertIn("cudagraphs_manual", g["reason"])
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "trt")
        self.assertNotIn("MUSETALK_TRT_UNET_CUDAGRAPHS", env)
        t.add_ts_mode_record(d, "manual", passed=False)
        proc, env, rep = t.resolve("fast300")
        self.assertEqual(self.groups(rep)["ts_unet_cudagraphs"]["status"], "dropped")
        t.add_ts_mode_record(d, "manual", passed=True)
        proc, env, rep = t.resolve("fast300")
        self.assertEqual(self.groups(rep)["ts_unet_cudagraphs"]["status"], "enabled")
        self.assertEqual(env["MUSETALK_TRT_UNET_CUDAGRAPHS"], "manual")

    def test_taesd_group_with_engine(self):
        t = self.tree_with_code()
        d = t.add_taesd_trt()
        self.use(t, "taesd_trt")
        proc, env, rep = t.resolve("fast300")
        self.assertOk(proc)
        self.assertEqual(self.groups(rep)["taesd_trt"]["status"], "enabled", rep["recipe_groups"])
        self.assertEqual(env["MUSETALK_TAESD_BACKEND"], "trt")
        self.assertEqual(env["MUSETALK_TAESD_TRT_BUILD"], "0")
        self.assertEqual(env["MUSETALK_TAESD_TRT_STRICT"], "1")
        self.assertEqual(env["MUSETALK_TAESD_TRT_DIR"], str(d.resolve()))
        self.assertEqual(rep["expect"]["vae"], "taesd_trt")

    def test_taesd_group_needs_quality_gate_pass(self):
        t = self.tree_with_code()
        t.add_taesd_trt(verdict="FAIL")
        self.use(t, "taesd_trt")
        proc, env, rep = t.resolve("fast300")
        self.assertOk(proc)
        g = self.groups(rep)["taesd_trt"]
        self.assertEqual(g["status"], "dropped")
        self.assertIn("G-TAESD quality gate is FAIL", g["reason"])
        self.assertNotIn("MUSETALK_TAESD_BACKEND", env)
        self.assertEqual(rep["expect"]["vae"], "taesd_compiled")
        proc, env, rep = t.resolve("fast", MUSETALK_TAESD_BACKEND="trt")  # caller forces it: respected + warned
        self.assertOk(proc)
        self.assertEqual(rep["expect"]["vae"], "taesd_trt")
        self.assertTrue(any("quality gate is FAIL" in w for w in rep["warnings"]))

    def test_native_vp8_group_dropped_before_running_the_preflight(self):
        t = self.tree_with_code()
        t.add_native_vp8()
        marker = self.tmp / "preflight_ran"
        py = t.venv / "bin" / "python"
        py.write_text(f"#!/bin/sh\ntouch {marker}\nexit 0\n")
        py.chmod(py.stat().st_mode | stat.S_IEXEC)
        self.use(t, "native_vp8")
        proc, env, rep = t.resolve("fast300")
        self.assertOk(proc)
        g = self.groups(rep)["native_vp8"]
        self.assertEqual(g["status"], "dropped")
        self.assertIn("h264:native_fallback", g["reason"])
        self.assertNotIn("vp8:native_preflight", g["prereqs"])
        self.assertFalse(marker.exists(), "the expensive preflight must not run once a cheap prerequisite failed")
        self.assertEqual(env["WEBRTC_VP8_ENCODER"], "pyav")

    def test_prereq_free_group_and_invalid_line(self):
        t = self.tree_with_code()
        text = uncomment_groups(REAL_RECIPE.read_text(), "malloc_arena", "edf_scheduler", "h264_x264tuned")
        t.recipe(text.replace("HLS_SCHEDULER_POLICY=edf", "HLS_SCHEDULER_POLICY=bogus"))
        proc, env, rep = t.resolve("fast300")
        self.assertOk(proc)
        g = self.groups(rep)
        self.assertEqual(g["malloc_arena"]["status"], "enabled")
        self.assertEqual(env["MALLOC_ARENA_MAX"], "4")
        self.assertEqual(g["h264_x264tuned"]["status"], "enabled")
        self.assertEqual(env["WEBRTC_H264_IMPL"], "x264tuned")
        self.assertEqual(g["edf_scheduler"]["status"], "dropped")
        self.assertNotIn("HLS_SCHEDULER_POLICY", env)

    def test_code_prerequisite(self):
        t = self.tree()  # no fake code files: every code:<path> prerequisite fails
        self.use(t, "nonblocking_handoff")
        proc, env, rep = t.resolve("fast300")
        g = self.groups(rep)["nonblocking_handoff"]
        self.assertEqual(g["status"], "dropped")
        self.assertIn("code:scripts/webrtc_media_flags.py", g["reason"])

    def test_caller_overrides_a_group_key(self):
        t = self.tree_with_code()
        t.add_stagewise(16)
        self.use(t, "stagewise_unet")
        proc, env, rep = t.resolve("fast300", HLS_SCHEDULER_FIXED_BATCH_SIZES="16,32")
        self.assertOk(proc)
        self.assertEqual(self.groups(rep)["stagewise_unet"]["status"], "enabled")
        self.assertEqual(env["HLS_SCHEDULER_FIXED_BATCH_SIZES"], "16,32")
        self.assertEqual(env["MUSETALK_TAESD_WARMUP_BATCHES"], "16,32")
        items = {i["name"]: i for i in rep["recipe_levers"]}
        self.assertEqual(items["HLS_SCHEDULER_FIXED_BATCH_SIZES"]["status"], "overridden")


# --------------------------------------------------------------------------- verify-log
REAL_R5 = REPO / "configs" / "recipes" / "r5.env"
R5_BUNDLE = REPO / "configs" / "trt_bundles" / "ampere-plus-r5-srcg50-int8.json"  # the portable candidate (last)
R5_NATIVE = REPO / "configs" / "trt_bundles" / "rtx4070super-r5-srcg50-int8.json"  # the RTX 4070 SUPER one (first)


@unittest.skipUnless(REAL_R5.is_file() and R5_BUNDLE.is_file(), "r5 recipe / bundle descriptor not present")
class R5RecipeTests(Base):
    """Recipe r5: the tracked configs/recipes/r5.env + the pinned bundle (bundle:<name> prerequisite)."""

    CODE = (("scripts/unet_stagewise_trt.py", "MUSETALK_UNET_BACKEND"),
            ("scripts/webrtc_media_flags.py", "WEBRTC_DEADLINE_PACING WEBRTC_NONBLOCKING_HANDOFF "
             "MUSETALK_OFFLOOP_DIAGNOSTICS MUSETALK_THREAD_CAPS MUSETALK_DISABLE_LOCAL_TTS WEBRTC_LIFETIME_COUNTERS"),
            ("scripts/webrtc_idle_frame_cache.py", "WEBRTC_IDLE_FRAME_CACHE"),
            ("scripts/gc_tuning.py", "MUSETALK_GC_FREEZE"),
            ("scripts/hls_gpu_scheduler.py", "HLS_SKIP_CROSSFADE_COPY"),
            ("scripts/api_avatar.py", "MUSETALK_AVATAR_MASK_CHANNELS"))

    def r5_tree(self, restored=True, **kw):
        t = Tree(Path(tempfile.mkdtemp(dir=str(self.tmp))), **kw)  # several trees per test
        for rel, keys in self.CODE:
            (t.repo / rel).write_text(f"# reads {keys}\n")
        t.recipe(REAL_R5.read_text(), name="r5")
        (t.repo / "configs" / "trt_bundles").mkdir(parents=True)
        for desc in (REPO / "configs" / "trt_bundles").glob("*.json"):
            (t.repo / "configs" / "trt_bundles" / desc.name).write_text(desc.read_text())
        if restored:
            self.restore(t)
        return t

    def restore(self, t, stamp_sha=None, descriptor=None):
        """What trt_artifact_bundle.py restore --sidecar-dir leaves behind (tiny stand-in files)."""
        desc = descriptor or json.loads(R5_BUNDLE.read_text())
        unet = t.repo / desc["engines"]["unet_stagewise"]["cache_dir"] / "bs16"
        unet.mkdir(parents=True, exist_ok=True)
        (unet / "manifest.json").write_text(json.dumps({"batch": 16, "complete": True}))
        (unet / "prefix.plan").write_bytes(b"p" * 32)
        taesd = t.repo / desc["engines"]["taesd_trt"]["dir"]
        taesd.mkdir(parents=True, exist_ok=True)
        meta = taesd / f"taesd_trt_{desc['engines']['taesd_trt']['key']}.json"
        meta.write_text("{}")
        side = t.repo / desc["sidecar_dir"]
        side.mkdir(parents=True, exist_ok=True)
        files = [{"path": str(f.relative_to(t.repo)), "sha256": "x", "size": f.stat().st_size}
                 for f in (unet / "manifest.json", unet / "prefix.plan", meta)]
        (side / hp.TRT_BUNDLE_MANIFEST).write_text(json.dumps({"files": files}))
        (side / hp.TRT_BUNDLE_STAMP).write_text(json.dumps({"archive_sha256": stamp_sha or desc["sha256"],
                                                             "mode": "restored"}))
        return unet

    def groups(self, rep):
        return {g["name"]: g for g in rep["recipe_groups"]}

    def assertEnginesDropped(self, env, rep, reason):
        self.assertNotEqual(env.get("MUSETALK_UNET_BACKEND"), "trt_stagewise")
        self.assertNotIn("MUSETALK_TAESD_BACKEND", env)
        self.assertNotIn("MUSETALK_UNET_STAGEWISE_CACHE_DIR", env)
        group = self.groups(rep)["r5_engines"]
        self.assertEqual(group["status"], "dropped")
        self.assertIn(reason, group["reason"])
        # the serving groups do not depend on the engines
        self.assertEqual(env["WEBRTC_DEADLINE_PACING"], "1")
        self.assertEqual(env["MUSETALK_GC_FREEZE"], "1")

    def test_descriptor_is_hardware_compatible(self):
        desc = json.loads(R5_BUNDLE.read_text())
        host = desc["host"]
        self.assertNotIn("engine_key", host)  # not tied to one GPU model
        self.assertEqual((host["hardware_compatibility"], host["min_compute_capability"], host["tensorrt_version"]),
                         ("ampere_plus", "8.0", "10.3.0"))
        self.assertEqual(desc["engines"]["taesd_trt"]["hardware_compatibility"], "ampere_plus")
        self.assertTrue(desc["s3_key"].startswith("trt-artifacts/") and desc["sha256"] in desc["s3_key"])
        self.assertTrue(desc["sidecar_dir"].startswith(".runtime/"))
        candidates = re.search(r"requires=bundle:([^,\s]+)", REAL_R5.read_text()).group(1).split("|")
        self.assertEqual(candidates, [json.loads(R5_NATIVE.read_text())["name"], desc["name"]])  # portable last

    def test_restored_bundle_selects_its_engines(self):
        t = self.r5_tree()
        proc, env, rep = t.resolve("r5")
        self.assertOk(proc)
        self.assertEqual(env["MUSETALK_RECIPE"], "r5")
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "trt_stagewise")
        self.assertEqual(env["MUSETALK_TRT_UNET_ENABLED"], "0")
        desc = json.loads(R5_BUNDLE.read_text())
        self.assertEqual(env["MUSETALK_UNET_STAGEWISE_CACHE_DIR"],
                         str(t.repo / desc["engines"]["unet_stagewise"]["cache_dir"]))
        self.assertEqual(env["MUSETALK_UNET_STAGEWISE_BATCH"], "16")
        self.assertEqual(env["MUSETALK_TAESD_BACKEND"], "trt")
        self.assertEqual(env["MUSETALK_TAESD_TRT_DIR"], str(t.repo / "models" / "taesd" / "trt"))
        self.assertEqual(env["MUSETALK_TAESD_TRT_BATCH"], "8")
        self.assertEqual(env["MUSETALK_TAESD_TRT_BUILD"], "0")
        self.assertEqual(env["MUSETALK_TAESD_TRT_STRICT"], "1")
        self.assertEqual(env["MUSETALK_TAESD_TRT_HW_COMPAT"], "ampere_plus")
        for knob in ("HLS_SCHEDULER_FIXED_BATCH_SIZES", "HLS_SCHEDULER_MAX_BATCH",
                     "MUSETALK_TAESD_WARMUP_BATCHES", "MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES"):
            self.assertEqual(env[knob], "16", knob)
        self.assertEqual(env["MUSETALK_TRT_FALLBACK"], "0")
        self.assertEqual(rep["expect"], {"vae": "taesd_trt", "unet": "trt_stagewise"})
        self.assertEqual(rep["warnings"], [])
        self.assertEqual(rep["unet"]["engine"]["match"], f"bundle:{desc['name']}")
        groups = self.groups(rep)
        self.assertEqual({n for n, g in groups.items() if g["status"] == "enabled"},
                         {"r5_engines", "r5_deadline_pacing", "r5_handoff", "r5_idle_frame_cache", "r5_gc",
                          "r5_offloop", "r5_thread_caps", "r5_scheduler", "r5_avatar_layout"})
        self.assertEqual(groups["r5_no_local_tts"]["status"], "off")
        self.assertEqual(groups["r5_telemetry"]["status"], "off")
        self.assertNotIn("MUSETALK_DISABLE_LOCAL_TTS", env)
        self.assertEqual(env["WEBRTC_QUEUE_PACKED_I420"], "1")
        self.assertEqual(env["MUSETALK_GC_THRESHOLDS"], "700,10,100")
        self.assertEqual(env["WEBRTC_IDLE_FRAME_CACHE_MAX_MB"], "2400")

    def test_not_restored_or_other_archive_drops_engines_only(self):
        proc, env, rep = self.r5_tree(restored=False).resolve("r5")
        self.assertOk(proc)
        self.assertEnginesDropped(env, rep, "not restored")
        t = self.r5_tree(restored=False)
        self.restore(t, stamp_sha="0" * 64)
        proc, env, rep = t.resolve("r5")
        self.assertOk(proc)
        self.assertEnginesDropped(env, rep, "not restored")

    def test_gpu_specific_bundle_is_preferred_on_its_gpu(self):
        native = json.loads(R5_NATIVE.read_text())
        t = self.r5_tree()  # the portable bundle restored ...
        self.restore(t, descriptor=native)  # ... and the RTX 4070 SUPER one
        proc, env, rep = t.resolve("r5")
        self.assertOk(proc)
        self.assertEqual(env["MUSETALK_UNET_STAGEWISE_CACHE_DIR"],
                         str(t.repo / native["engines"]["unet_stagewise"]["cache_dir"]))
        self.assertNotIn("MUSETALK_TAESD_TRT_HW_COMPAT", env)
        self.assertEqual(rep["unet"]["engine"]["match"], f"bundle:{native['name']}")
        for gpu in (GPU_3090, GPU_4070TI):  # other GPUs: the portable one, even with both restored
            with self.subTest(gpu=gpu["name"]):
                other = Tree(Path(tempfile.mkdtemp(dir=str(self.tmp))), gpus=(gpu,))
                shutil.rmtree(other.repo)
                shutil.copytree(t.repo, other.repo, symlinks=True)
                proc, env, rep = other.resolve("r5")
                self.assertOk(proc)
                self.assertEqual(env["MUSETALK_TAESD_TRT_HW_COMPAT"], "ampere_plus")
                self.assertEqual(rep["unet"]["engine"]["match"], f"bundle:{json.loads(R5_BUNDLE.read_text())['name']}")

    def test_any_ampere_or_newer_gpu_uses_the_bundle(self):
        for gpu in (GPU_3090, GPU_4070TI, GPU_A100, GPU_H100):
            with self.subTest(gpu=gpu["name"]):
                proc, env, rep = self.r5_tree(gpus=(gpu,)).resolve("r5")
                self.assertOk(proc)
                self.assertEqual(env["MUSETALK_UNET_BACKEND"], "trt_stagewise")
                self.assertEqual(env["MUSETALK_TAESD_BACKEND"], "trt")
                self.assertEqual(self.groups(rep)["r5_engines"]["status"], "enabled")

    def test_pre_ampere_other_tensorrt_or_small_vram_drops_engines(self):
        proc, env, rep = self.r5_tree(gpus=(GPU_T4,)).resolve("r5")
        self.assertOk(proc)
        self.assertEnginesDropped(env, rep, "Tesla T4 is sm7.5")
        proc, env, rep = self.r5_tree(packages=CU128).resolve("r5")
        self.assertOk(proc)
        self.assertEnginesDropped(env, rep, "bundle engines need TensorRT 10.3.0; the venv has 10.9.0.34")
        proc, env, rep = self.r5_tree(gpus=(GPU_3070,)).resolve("r5")
        self.assertOk(proc)
        self.assertEnginesDropped(env, rep, "VRAM 6.0 GB < 8 GB")

    def test_exact_engine_key_descriptor_still_supported(self):
        t = self.r5_tree(restored=False)
        desc = dict(json.loads(R5_BUNDLE.read_text()), name="gpu-specific-test")
        desc["host"] = {"engine_key": ek.engine_key("unet_stagewise", t.host_facts())}
        desc["sidecar_dir"] = ".runtime/trt_artifacts/gpu-specific-test"
        (t.repo / "configs" / "trt_bundles" / "gpu-specific-test.json").write_text(json.dumps(desc))
        t.recipe(re.sub(r"bundle:[^,\s]+,", "bundle:gpu-specific-test,", REAL_R5.read_text()), name="r5")
        self.restore(t, descriptor=desc)
        proc, env, rep = t.resolve("r5")
        self.assertOk(proc)
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "trt_stagewise")
        other = Tree(Path(tempfile.mkdtemp(dir=str(self.tmp))), gpus=(GPU_4070TI,))  # same sm_89, other model
        shutil.rmtree(other.repo)
        shutil.copytree(t.repo, other.repo, symlinks=True)
        proc, env, rep = other.resolve("r5")
        self.assertOk(proc)
        self.assertEnginesDropped(env, rep, "these plans load only on the exact GPU model")

    def test_default_recipe_is_r5(self):
        t = self.r5_tree()
        proc, env, rep = t.resolve(None)
        self.assertOk(proc)
        self.assertEqual((rep["recipe"], env["MUSETALK_RECIPE"]), ("r5", "r5"))
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "trt_stagewise")

    def test_bundle_check_cli(self):
        native, portable = json.loads(R5_NATIVE.read_text())["name"], json.loads(R5_BUNDLE.read_text())["name"]
        both = f"{native}|{portable}"

        def states(proc):
            return [tuple(line.split("\t")[:2]) for line in proc.stdout.splitlines() if line]

        for gpus, want, fit in (((GPU_4070S,), 0, [(native, "ok"), (portable, "ok")]),
                                ((GPU_3090,), 0, [(native, "no"), (portable, "ok")]),
                                ((GPU_T4,), 3, [(native, "no"), (portable, "no")])):
            with self.subTest(gpu=gpus[0]["name"]):
                t = self.r5_tree(restored=False, gpus=gpus)
                proc = t.run("bundle-check", "--bundle", both, "--host-only", "--repo-root", str(t.repo),
                             "--venv", str(t.venv))
                self.assertEqual(proc.returncode, want, proc.stdout + proc.stderr)
                self.assertEqual(states(proc), fit)
        t = self.r5_tree(restored=False)
        proc = t.run("bundle-check", "--bundle", portable, "--repo-root", str(t.repo), "--venv", str(t.venv))
        self.assertEqual(proc.returncode, 3, proc.stdout)  # host fits, but nothing is restored
        self.assertIn("not restored", proc.stdout)
        proc = t.run("bundle-check", "--bundle", "nope", "--repo-root", str(t.repo), "--venv", str(t.venv))
        self.assertEqual((proc.returncode, states(proc)), (3, [("nope", "no")]))
        proc = t.run("bundle-check", "--bundle", "|", "--repo-root", str(t.repo), "--venv", str(t.venv))
        self.assertEqual(proc.returncode, 2)

    def test_changed_or_missing_file_drops_engines(self):
        t = self.r5_tree()
        (self.restore(t) / "prefix.plan").write_bytes(b"q" * 33)
        proc, env, rep = t.resolve("r5")
        self.assertOk(proc)
        self.assertEnginesDropped(env, rep, "missing or changed since the restore")

    def test_caller_pinning_another_set_drops_the_whole_group(self):
        t = self.r5_tree()
        other = t.repo / "models" / "other_stagewise"
        (other / "bs16").mkdir(parents=True)
        proc, env, rep = t.resolve("r5", MUSETALK_UNET_STAGEWISE_CACHE_DIR=str(other))
        self.assertOk(proc)
        self.assertNotEqual(env.get("MUSETALK_UNET_BACKEND"), "trt_stagewise")
        self.assertNotIn("MUSETALK_TAESD_BACKEND", env)  # atomic: the TAESD half goes too
        self.assertEqual(self.groups(rep)["r5_engines"]["status"], "dropped")

    def test_other_recipes_ignore_r5(self):
        t = self.r5_tree()
        proc, env, rep = t.resolve("fast")
        self.assertOk(proc)
        self.assertNotIn("WEBRTC_DEADLINE_PACING", env)
        self.assertEqual(rep["recipe_groups"], [])
        self.assertNotEqual(env.get("MUSETALK_UNET_BACKEND"), "trt_stagewise")


class VerifyLogTests(Base):
    def write(self, text, name="api.log"):
        path = self.tmp / name
        path.write_text(text, encoding="utf-8")
        return path

    def test_match_mismatch_offset_and_timeout(self):
        old = LOG_VAE_PYTORCH + LOG_UNET_PYTORCH
        log = self.write("boot 1\n" + old + "boot 2\n" + LOG_VAE_TAESD + "✅ Models loaded!\n" + LOG_UNET_MULTI)
        offset = len(("boot 1\n" + old).encode("utf-8"))
        rc, res = hp.verify_log(log, offset, "taesd", "trt", timeout=0)
        self.assertEqual(rc, 0, res)
        self.assertEqual(res["found"], {"vae": "taesd", "unet": "tensorrt_unet_multi"})
        rc, res = hp.verify_log(log, 0, "taesd", "trt", timeout=0)  # the old boot's PyTorch lines come first
        self.assertEqual(rc, 1)
        self.assertIn("pytorch", " ".join(res["mismatches"]))
        rc, res = hp.verify_log(log, offset, "taesd_trt", "any", timeout=0)
        self.assertEqual(rc, 1)
        rc, res = hp.verify_log(log, offset, "taesd", "eager", timeout=0)
        self.assertEqual(rc, 1)
        log2 = self.write(LOG_VAE_TAESD, "partial.log")
        started = time.time()
        rc, res = hp.verify_log(log2, 0, "taesd", "trt", timeout=0.3, poll=0.1)
        self.assertEqual(rc, 3)
        self.assertEqual(res["missing"], ["unet"])
        self.assertLess(time.time() - started, 3)
        rc, res = hp.verify_log(log2, 0, "taesd", "any", timeout=0)
        self.assertEqual(rc, 0)
        rc, res = hp.verify_log(self.tmp / "absent.log", 0, "taesd", "any", timeout=0)
        self.assertEqual(rc, 3)

    def test_stagewise_trt_taesd_and_fallback_extras(self):
        log = self.write("⚠️  TAESD TRT unavailable (FileNotFoundError: x); using compiled TAESD\n" + LOG_VAE_TAESD +
                         LOG_UNET_STAGEWISE + "ℹ️  Eager UNet released (MUSETALK_FREE_EAGER_UNET=1)\n")
        rc, res = hp.verify_log(log, 0, "taesd_trt", "trt_stagewise", timeout=0)
        self.assertEqual(rc, 1)
        self.assertIn("taesd_trt_fallback", res["extras"])
        self.assertTrue(res["extras"]["eager_unet_released"])
        rc, res = hp.verify_log(log, 0, "taesd", "trt_stagewise", timeout=0)
        self.assertEqual(rc, 0)
        log = self.write(LOG_VAE_TAESD_TRT + LOG_UNET_STAGEWISE, "ok.log")
        self.assertEqual(hp.verify_log(log, 0, "taesd_trt", "tensorrt_any", timeout=0)[0], 0)

    def test_offset_beyond_size_rescans(self):
        log = self.write(LOG_VAE_TAESD + LOG_UNET_PYTORCH)
        rc, res = hp.verify_log(log, 10 ** 9, "taesd", "eager", timeout=0)
        self.assertEqual(rc, 0)
        self.assertTrue(res["notes"])

    def test_cli_auto_from_resolved_report(self):
        log = self.write(LOG_VAE_TAESD + LOG_UNET_MULTI)
        rep = self.tmp / "r.json"
        rep.write_text(json.dumps({"expect": {"vae": "taesd_compiled", "unet": "trt"}}))
        t = self.tree()
        proc = t.run("verify-log", "--log", str(log), "--expect-vae", "auto", "--expect-unet", "auto",
                     "--resolved", str(rep), "--timeout", "0")
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertEqual(json.loads(proc.stdout)["status"], "match")
        rep.write_text(json.dumps({"expect": {"vae": "taesd_compiled", "unet": "eager"}}))
        proc = t.run("verify-log", "--log", str(log), "--expect-vae", "auto", "--expect-unet", "auto",
                     "--resolved", str(rep), "--timeout", "0")
        self.assertEqual(proc.returncode, 1)
        proc = t.run("verify-log", "--log", str(log), "--expect-unet", "auto", "--timeout", "0")
        self.assertEqual(proc.returncode, 2)  # auto without a resolved report

    def test_log_strings_match_source(self):
        """The patterns and backend names must exist verbatim in the serving code (drift guard)."""
        amp = (REPO / "scripts" / "avatar_manager_parallel.py").read_text(encoding="utf-8")
        for needle in ('VAE decode backend active: {self.vae_decode_backend_name}', 'VAE decode backend: PyTorch',
                       'UNet backend active: {self.unet_backend_name}', 'UNet backend: PyTorch'):
            self.assertIn(needle, amp)
        vfd = (REPO / "scripts" / "vae_fast_decoder.py").read_text(encoding="utf-8")
        self.assertIn('name = "taesd"', vfd)
        self.assertIn('name = "taesd_trt"', vfd)
        self.assertIn("TAESD TRT unavailable (", vfd)
        trt = (REPO / "scripts" / "trt_runtime.py").read_text(encoding="utf-8")
        self.assertIn('name = "tensorrt_unet"', trt)
        self.assertIn('name = "tensorrt_unet_multi"', trt)
        sw = (REPO / "scripts" / "unet_stagewise_trt.py").read_text(encoding="utf-8")
        self.assertIn('name = "tensorrt_unet_stagewise"', sw)
        for names in hp.VAE_EXPECT.values():
            for name in names or ():
                if name != "pytorch":
                    self.assertTrue(f'"{name}"' in vfd + trt, name)
        for names in hp.UNET_EXPECT.values():
            for name in names or ():
                if name != "pytorch":
                    self.assertTrue(f'"{name}"' in trt + sw, name)


# --------------------------------------------------------------------------- engine CLIs / safety
class EngineCliTests(Base):
    def test_engine_key_and_find_engine(self):
        t = self.tree()
        proc = t.run("engine-key", "--repo-root", str(t.repo), "--venv", str(t.venv), "--kind", "unet_ts")
        self.assertOk(proc)
        self.assertEqual(proc.stdout.strip(), "sm89-nvidia-geforce-rtx-4070-super-trt10.3.0-tt2.5.0")
        proc = t.run("engine-key", "--repo-root", str(t.repo), "--venv", str(t.venv), "--kind", "unet_stagewise")
        self.assertEqual(proc.stdout.strip(), "sm89-nvidia-geforce-rtx-4070-super-trt10.3.0")
        proc = t.run("find-engine", "--repo-root", str(t.repo), "--venv", str(t.venv), "--kind", "unet_ts")
        self.assertEqual(proc.returncode, 3)
        d = t.add_unet_ts()
        proc = t.run("find-engine", "--repo-root", str(t.repo), "--venv", str(t.venv), "--kind", "unet_ts")
        self.assertOk(proc)
        out = json.loads(proc.stdout)
        self.assertEqual(out["engine_path"], str(d / "unet_trt.ts"))
        self.assertIn("MUSETALK_TRT_UNET_PATHS", out["env"])
        t.add_taesd_trt()
        proc = t.run("find-engine", "--repo-root", str(t.repo), "--venv", str(t.venv), "--kind", "taesd_trt")
        self.assertOk(proc)
        self.assertEqual(json.loads(proc.stdout)["env"]["MUSETALK_TAESD_TRT_BUILD"], "0")

    def test_engine_key_without_tensorrt(self):
        t = self.tree(packages={"torch": "2.5.1+cu121"})
        proc = t.run("engine-key", "--repo-root", str(t.repo), "--venv", str(t.venv))
        self.assertEqual(proc.returncode, 2)

    def test_never_imports_torch(self):
        """resolve/detect/verify-log with torch, tensorrt and aiortc made unimportable."""
        t = self.tree()
        t.add_unet_ts()
        code = textwrap.dedent(f"""
            import sys, runpy
            for name in ("torch", "tensorrt", "torch_tensorrt", "aiortc", "av"):
                sys.modules[name] = None
            sys.argv = ["musetalk_host_profile.py", "resolve", "--repo-root", {str(t.repo)!r},
                        "--venv", {str(t.venv)!r}, "--out", {str(self.tmp / 'n.env')!r},
                        "--report", {str(self.tmp / 'n.json')!r}]
            try:
                runpy.run_path({str(TOOL)!r}, run_name="__main__")
            except SystemExit as exc:
                assert exc.code in (0, None), exc.code
            assert all(sys.modules.get(n) is None for n in ("torch", "tensorrt", "aiortc"))
            print("OK")
            """)
        proc = subprocess.run([sys.executable, "-B", "-c", code], env=t.env(), stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn("OK", proc.stdout)

    def test_flag_registry_parser_never_executes(self):
        t = self.tree()
        (t.repo / "scripts" / "webrtc_media_flags.py").write_text(
            "import torch\nraise SystemExit('executed!')\nFLAGS = {'WEBRTC_X': ('0', 'doc'), 'WEBRTC_Y': ('a', 'b')}\n")
        reg = hp.load_flag_registries(t.repo)
        self.assertEqual(reg["WEBRTC_X"]["default"], "0")
        self.assertEqual(reg["WEBRTC_Y"]["default"], "a")

    def test_real_repo_registries_parse(self):
        reg = hp.load_flag_registries(REPO)
        self.assertIn("WEBRTC_NONBLOCKING_HANDOFF", reg)
        self.assertIn("MUSETALK_TAESD_BACKEND", reg)


if __name__ == "__main__":
    unittest.main()
