"""CPU-only tests for scripts/musetalk_engine_keys.py and scripts/unet_engine_store.py.

No torch, no GPU: every GPU tool the store launches (validate_unet_backend.py, tensorrt_export.py,
build_unet_stagewise.py, vae_fast_decoder.py) is replaced by a stdlib fake inside a temporary repo,
and host facts are injected through MUSETALK_HOST_FACTS_JSON.

Run: python -m unittest test_unet_engine_store -v
"""
import contextlib
import hashlib
import io
import json
import os
import pickle
import subprocess
import sys
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts import musetalk_engine_keys as ek  # noqa: E402
from scripts import unet_engine_store as store  # noqa: E402

GPU_4070S = "NVIDIA GeForce RTX 4070 SUPER"
KEY_TS = "sm89-nvidia-geforce-rtx-4070-super-trt10.3.0-tt2.5.0"
KEY_PLANS = "sm89-nvidia-geforce-rtx-4070-super-trt10.3.0"


def framed_device(device: str) -> bytes:
    raw = device.encode()
    return b"X" + len(raw).to_bytes(4, "little") + raw + b"q\n"


def fake_ts_bytes(device: str, pad: int = 6000) -> bytes:
    return os.urandom(pad) + b"c__torch__.torch.classes.tensorrt\nEngine\n" + framed_device(device) + os.urandom(pad)


FAKE_VALIDATOR = r'''
import argparse, json, os, sys
p = argparse.ArgumentParser()
for flag in ("--capture-dir", "--padded-batch-size", "--fail-mae", "--fail-max-abs", "--warmup", "--iters",
             "--report-path", "--backend", "--trt-path"):
    p.add_argument(flag, default="")
p.add_argument("--limit", type=int, default=8)
p.add_argument("--group-captures", type=int, default=1)
a = p.parse_args()
env = {k: v for k, v in os.environ.items() if k.startswith("MUSETALK_")}
with open(os.environ["FAKE_CALLS"], "a") as fh:
    fh.write(json.dumps({"tool": "validate_unet_backend", "argv": sys.argv[1:], "env": env}) + "\n")
if os.environ.get("FAKE_VALIDATOR_CRASH") == "1":
    sys.exit(3)
if a.backend == "trt" and not os.path.exists(a.trt_path):
    sys.exit(4)
mae = float(os.environ.get("FAKE_MAE", "0.002"))
files = len([f for f in os.listdir(a.capture_dir) if f.startswith("unet_io_")][: a.limit]) // a.group_captures
summary = {"files": files, "mae_max": mae, "max_abs_max": 0.12}
with open(a.report_path, "w") as fh:
    json.dump({"backend": a.backend, "summary": summary, "files": []}, fh)
sys.exit(1 if mae > float(a.fail_mae) else 0)
'''

FAKE_EXPORT = r'''
import argparse, json, os, sys
p = argparse.ArgumentParser()
p.add_argument("--output-dir")
p.add_argument("--validate-unet-report-path")
p.add_argument("--batch-sizes")
a, _ = p.parse_known_args()
with open(os.environ["FAKE_CALLS"], "a") as fh:
    fh.write(json.dumps({"tool": "tensorrt_export", "argv": sys.argv[1:]}) + "\n")
os.makedirs(a.output_dir, exist_ok=True)
device = os.environ.get("FAKE_DEVICE", "0%8%9%0%NVIDIA GeForce RTX 4070 SUPER").encode()
blob = os.urandom(3000) + b"X" + len(device).to_bytes(4, "little") + device + b"q\n" + os.urandom(3000)
with open(os.path.join(a.output_dir, "unet_trt.ts"), "wb") as fh:
    fh.write(blob)
if os.environ.get("FAKE_EXPORT_FAIL") == "1":
    sys.exit(1)
b = int(a.batch_sizes)
meta = {"type": "unet", "batch_range": [b, b], "opt_batch": b, "dtype": "float16", "save_format": "torchscript",
        "validation": {"passed": True}}
json.dump(meta, open(os.path.join(a.output_dir, "unet_trt_meta.json"), "w"))
json.dump({"passed": True, "summary": {"files": 16, "mae_max": 0.002, "max_abs_max": 0.15}},
          open(a.validate_unet_report_path, "w"))
'''

FAKE_STAGEWISE_BUILD = r'''
import argparse, hashlib, json, os, sys
p = argparse.ArgumentParser()
p.add_argument("--batch", type=int)
p.add_argument("--root")
p.add_argument("--opt-level")
p.add_argument("--timing-cache")
p.add_argument("--report")
p.add_argument("--max-minutes")
a = p.parse_args()
with open(os.environ["FAKE_CALLS"], "a") as fh:
    fh.write(json.dumps({"tool": "build_unet_stagewise", "argv": sys.argv[1:]}) + "\n")
d = os.path.join(a.root, "bs%d" % a.batch)
os.makedirs(d, exist_ok=True)
blocks = {}
for name in ["head", "down0", "down1", "down2", "down3", "mid", "up0", "up1", "up2", "up3", "tail"]:
    data = (name * 50).encode()
    open(os.path.join(d, name + ".plan"), "wb").write(data)
    blocks[name] = {"engine_file": name + ".plan", "engine_sha256": hashlib.sha256(data).hexdigest()}
open(os.path.join(d, "probe_output.pt"), "wb").write(b"probe")
open(a.timing_cache, "wb").write(b"cache")
manifest = {"schema": "musetalk_unet_stagewise_trt_v1", "batch": a.batch, "tensorrt_version": "10.3.0",
            "torch_version": "2.5.1+cu121", "gpu": "NVIDIA GeForce RTX 4070 SUPER", "compute_capability": [8, 9],
            "blocks": blocks, "complete": os.environ.get("FAKE_STAGEWISE_INCOMPLETE") != "1",
            "probe": {"output_sha256": "ab" * 32, "output_file": "probe_output.pt"}}
json.dump(manifest, open(os.path.join(d, "manifest.json"), "w"))
'''

FAKE_TAESD = r'''
import argparse, hashlib, json, os, sys
p = argparse.ArgumentParser()
p.add_argument("command")
p.add_argument("--batch", type=int, default=8)
p.add_argument("--force", action="store_true")
a = p.parse_args()
d = os.environ["MUSETALK_TAESD_TRT_DIR"]
env = {k: v for k, v in os.environ.items() if k.startswith("MUSETALK_")}
with open(os.environ["FAKE_CALLS"], "a") as fh:
    fh.write(json.dumps({"tool": "vae_fast_decoder", "argv": sys.argv[1:], "env": env}) + "\n")
key = "abcdef0123456789abcd"
meta_path = os.path.join(d, "taesd_trt_%s.json" % key)
probe = {"fp16_sha256": "11" * 32, "u8_fused_sha256": "22" * 32, "fused_vs_repo_post_mismatched_bytes": 0}
if a.command == "build":
    os.makedirs(d, exist_ok=True)
    dec, post = b"decoder-plan" * 10, b"post-plan" * 10
    open(os.path.join(d, "taesd_trt_%s.decoder.plan" % key), "wb").write(dec)
    open(os.path.join(d, "taesd_trt_%s.post_bgr_u8.plan" % key), "wb").write(post)
    meta = {"schema": "taesd_trt_engine_v1", "key": key,
            "fingerprint": {"gpu": "NVIDIA GeForce RTX 4070 SUPER", "compute_capability": "8.9", "tensorrt": "10.3.0",
                            "batch": a.batch, "opt_level": int(os.environ.get("MUSETALK_TAESD_TRT_OPT_LEVEL", "3")),
                            "strongly_typed": False},
            "decoder_plan": "taesd_trt_%s.decoder.plan" % key, "decoder_plan_sha256": hashlib.sha256(dec).hexdigest(),
            "post_plan": "taesd_trt_%s.post_bgr_u8.plan" % key, "post_plan_sha256": hashlib.sha256(post).hexdigest(),
            "probe": probe}
    json.dump(meta, open(meta_path, "w"))
    print(json.dumps({"key": key, "fingerprint": meta["fingerprint"], "probe": probe, "build": {}}, indent=1))
if not os.path.exists(meta_path) or os.environ.get("FAKE_TAESD_VERIFY_FAIL") == "1":
    print("FileNotFoundError: missing", file=sys.stderr)
    sys.exit(1)
print("TAESD TRT backend: key=%s probe=ok" % key)
print(json.dumps({"key": key, "probe": probe}, indent=1))
'''


def make_capture(path: Path) -> None:
    payload = {"schema_version": 1, "kind": "unet_io_batch", "latent_batch": 0, "audio_feature_batch": 0,
               "pred_latents": 0, "timesteps": 0, "actual_batch": 8, "padded_batch": 8}
    stem = path.stem
    with zipfile.ZipFile(str(path), "w") as archive:
        archive.writestr(f"{stem}/data.pkl", pickle.dumps(payload, protocol=2))
        archive.writestr(f"{stem}/version", "3\n")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class StoreTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="engine_store_test_")
        self.base = Path(self.tmp.name)
        self.repo = self.base / "repo"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "models/musetalkV15").mkdir(parents=True)
        (self.repo / "models/musetalkV15/unet.pth").write_bytes(b"w")
        (self.repo / "models/musetalkV15/musetalk.json").write_text("{}")
        for name, body in (("validate_unet_backend.py", FAKE_VALIDATOR), ("tensorrt_export.py", FAKE_EXPORT),
                           ("build_unet_stagewise.py", FAKE_STAGEWISE_BUILD), ("vae_fast_decoder.py", FAKE_TAESD)):
            (self.repo / "scripts" / name).write_text(body)
        corpus = self.repo / "calibration/unet_portable_bs8"
        corpus.mkdir(parents=True)
        rows = []
        for i in range(1, 17):
            capture = corpus / f"unet_io_{i:06d}_bs8_pid1.pt"
            make_capture(capture)
            rows.append({"file": capture.name, "bytes": capture.stat().st_size, "sha256": sha(capture)})
        (corpus / "manifest.json").write_text(json.dumps({"schema": store.CORPUS_SCHEMA, "files": rows}))
        self.calls = self.base / "calls.jsonl"
        self.facts_path = self.base / "facts.json"
        self.set_facts()
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("MUSETALK_", "TRT_ARTIFACT_", "FAKE_")) and k != "CUDA_VISIBLE_DEVICES"}
        env.update({"MUSETALK_HOST_FACTS_JSON": str(self.facts_path), "FAKE_CALLS": str(self.calls),
                    "MUSETALK_ENGINE_LOG_DIR": str(self.base / "logs"), "PYTHONDONTWRITEBYTECODE": "1"})
        self.env_patch = mock.patch.dict(os.environ, env, clear=True)
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        self.tmp.cleanup()

    def set_facts(self, name=GPU_4070S, cc="8.9", trt="10.3.0", tt="2.5.0", mem_mb=20000.0, gpus=None, **extra):
        facts = {
            "gpus": gpus if gpus is not None else [{"index": 0, "name": name, "compute_capability": cc,
                                                    "memory_total_mib": 12282, "memory_used_mib": 300}],
            "venv": {"torch": "2.5.1+cu121", "tensorrt": trt, "torch_tensorrt": tt},
            "ram": {"effective_available_mb": mem_mb}, "disk_free_gb": 100.0,
        }
        facts.update(extra)
        self.facts_path.write_text(json.dumps(facts))
        return facts

    def cli(self, *argv, expect=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = store.main(list(argv) + ["--repo-root", str(self.repo), "--venv-python", sys.executable])
        text = out.getvalue()
        try:
            payload = json.loads(text) if text.strip() else {}
        except json.JSONDecodeError:
            self.fail(f"non-JSON stdout for {argv}: {text!r}\nstderr: {err.getvalue()}")
        if expect is not None:
            self.assertEqual(rc, expect, f"{argv} -> rc {rc}\nstdout: {text}\nstderr: {err.getvalue()}")
        return rc, payload

    def calls_for(self, tool):
        if not self.calls.exists():
            return []
        return [json.loads(line) for line in self.calls.read_text().splitlines() if json.loads(line)["tool"] == tool]

    def legacy_ts(self, device=f"0%8%9%0%{GPU_4070S}", rel="models/tensorrt_unet_sm89_bs8_local"):
        directory = self.repo / rel
        directory.mkdir(parents=True, exist_ok=True)
        ts = directory / "unet_trt.ts"
        ts.write_bytes(fake_ts_bytes(device))
        (directory / "unet_trt_meta.json").write_text(json.dumps(
            {"type": "unet", "batch_range": [8, 8], "opt_batch": 8, "dtype": "float16", "save_format": "torchscript",
             "validation": {"passed": True, "capture_dir": "/elsewhere"}}))
        return ts

    def legacy_stagewise(self, batch=16, gpu=GPU_4070S, cc=(8, 9)):
        directory = self.repo / "models/tensorrt_unet_stagewise_sm89" / f"bs{batch}"
        directory.mkdir(parents=True)
        blocks = {}
        for name in ek.STAGEWISE_BLOCKS:
            (directory / f"{name}.plan").write_bytes(name.encode() * 20)
            blocks[name] = {"engine_file": f"{name}.plan", "engine_sha256": sha(directory / f"{name}.plan")}
        (directory / "probe_output.pt").write_bytes(b"probe")
        manifest = {"schema": ek.STAGEWISE_MANIFEST_SCHEMA, "batch": batch, "tensorrt_version": "10.3.0",
                    "torch_version": "2.5.1+cu121", "gpu": gpu, "compute_capability": list(cc), "blocks": blocks,
                    "complete": True, "probe": {"output_sha256": "cd" * 32, "output_file": "probe_output.pt"}}
        (directory / "manifest.json").write_text(json.dumps(manifest))
        return directory


class KeyAndFactsTest(unittest.TestCase):
    def test_slug_and_keys(self):
        self.assertEqual(ek.slug(GPU_4070S), "nvidia-geforce-rtx-4070-super")
        self.assertEqual(ek.slug("  NVIDIA A100-SXM4-80GB "), "nvidia-a100-sxm4-80gb")
        facts = {"gpus": [{"index": 0, "name": GPU_4070S, "compute_capability": "8.9"}],
                 "venv": {"tensorrt": "10.3.0", "torch_tensorrt": "2.5.0", "torch": "2.5.1+cu121"}}
        self.assertEqual(ek.engine_key("unet_ts", facts), KEY_TS)
        self.assertEqual(ek.engine_key("unet_stagewise", facts), KEY_PLANS)
        self.assertEqual(ek.engine_key("taesd_trt", facts), KEY_PLANS)
        parts = ek.parse_engine_key(KEY_TS)
        self.assertEqual(parts, {"cc": (8, 9), "gpu_slug": "nvidia-geforce-rtx-4070-super", "tensorrt": "10.3.0",
                                 "torch_tensorrt": "2.5.0"})
        blackwell = ek.engine_key("unet_ts", {"gpu_name": "NVIDIA GeForce RTX 5090", "compute_capability": "12.0",
                                              "tensorrt_version": "10.9.0.34", "torch_tensorrt_version": "2.7.0+cu128"})
        self.assertEqual(blackwell, "sm120-nvidia-geforce-rtx-5090-trt10.9.0.34-tt2.7.0")
        self.assertEqual(ek.parse_engine_key(blackwell)["cc"], (12, 0))
        self.assertTrue(ek.same_cc_key(KEY_TS, "sm89-nvidia-geforce-rtx-4090-trt10.3.0-tt2.5.0"))
        self.assertFalse(ek.same_cc_key(KEY_TS, "sm86-nvidia-geforce-rtx-3090-trt10.3.0-tt2.5.0"))
        with self.assertRaises(ek.EngineKeyError):
            ek.engine_key("unet_ts", {"gpus": [], "venv": {"tensorrt": "10.3.0", "torch_tensorrt": "2.5.0"}})
        with self.assertRaises(ek.EngineKeyError):
            ek.engine_key("unet_ts", {"gpu_name": GPU_4070S, "compute_capability": "8.9", "tensorrt_version": "10.3.0"})

    def test_cuda_visible_devices_selection(self):
        gpus = [{"index": 0, "name": "NVIDIA GeForce RTX 3090", "compute_capability": "8.6"},
                {"index": 1, "name": GPU_4070S, "compute_capability": "8.9"}]
        self.assertEqual(ek.normalize_facts({"gpus": gpus}, environ={})["gpu_name"], "NVIDIA GeForce RTX 3090")
        self.assertEqual(ek.normalize_facts({"gpus": gpus}, environ={"CUDA_VISIBLE_DEVICES": "1,0"})["gpu_name"],
                         GPU_4070S)
        self.assertFalse(ek.normalize_facts({"gpus": gpus}, environ={"CUDA_VISIBLE_DEVICES": ""})["has_gpu"])
        self.assertEqual(ek.normalize_facts({"gpus": gpus, "selected_gpu": 1})["cc"], (8, 9))
        self.assertEqual(ek.normalize_facts({"gpu": gpus[0]})["compute_capability"], "8.6")
        # the resolver's detect shape: an explicit gpu=None means "no GPU selected", never re-selected
        self.assertFalse(ek.normalize_facts({"gpus": gpus, "gpu": None, "selected_gpu_index": None},
                                            environ={})["has_gpu"])
        resolver_facts = {"gpus": gpus, "gpu": gpus[1], "selected_gpu_index": 1,
                          "venv": {"tensorrt": "10.3.0", "torch_tensorrt": "2.5.0", "torch": "2.5.1+cu121"}}
        self.assertEqual(ek.engine_key("unet_ts", resolver_facts), KEY_TS)
        self.assertEqual(ek.parse_cc([8, 9]), (8, 9))
        self.assertEqual(ek.parse_cc("sm_120"), (12, 0))
        self.assertEqual(ek.parse_cc(8.6), (8, 6))

    def test_venv_versions_from_dist_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            sp = Path(tmp) / "lib/python3.10/site-packages"
            for name in ("torch-2.5.1+cu121.dist-info", "tensorrt_cu12-10.3.0.dist-info",
                         "torch_tensorrt-2.5.0.dist-info", "triton-3.1.0.dist-info"):
                (sp / name).mkdir(parents=True)
            versions = ek.venv_versions(Path(tmp))
        self.assertEqual((versions["torch"], versions["tensorrt"], versions["torch_tensorrt"], versions["triton"]),
                         ("2.5.1+cu121", "10.3.0", "2.5.0", "3.1.0"))
        self.assertEqual(versions["torch_cuda_tag"], "cu121")
        self.assertEqual(versions["python_version"], "3.10")


class ScanTest(unittest.TestCase):
    def test_scan_across_chunk_boundaries_and_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "engine.ts"
            data = bytearray(os.urandom(9000))
            framed = framed_device(f"0%8%9%0%{GPU_4070S}")
            unframed = b"\x0110%8%6%0%NVIDIA GeForce RTX 3090\x00"
            for offset in (250, 2040, 4095):
                data[offset:offset + len(framed)] = framed
            data[7000:7000 + len(unframed)] = unframed
            path.write_bytes(bytes(data))
            before = sha(path)
            for chunk in (256, 300, 1000, 1 << 20):
                result = ek.scan_embedded_devices(path, chunk_bytes=chunk, overlap=256, use_cache=False)
                self.assertEqual(result["matches"], 4, chunk)
                strings = sorted(d["device_string"] for d in result["devices"])
                self.assertEqual(strings, sorted([f"0%8%9%0%{GPU_4070S}", "10%8%6%0%NVIDIA GeForce RTX 3090"]))
                framed_entry = [d for d in result["devices"] if d["major"] == 8 and d["minor"] == 9][0]
                self.assertTrue(framed_entry["framed"])
                self.assertEqual(framed_entry["name"], GPU_4070S)  # no trailing pickle opcode
            cache = Path(tmp) / "cache"
            first = ek.scan_embedded_devices(path, cache_dir=cache)
            second = ek.scan_embedded_devices(path, cache_dir=cache)
            self.assertFalse(first["cached"])
            self.assertTrue(second["cached"])
            os.utime(path, ns=(time.time_ns(), time.time_ns() + 5_000_000_000))
            self.assertFalse(ek.scan_embedded_devices(path, cache_dir=cache)["cached"])
            self.assertEqual(sha(path), before)  # the original is never written
            facts = {"gpu_name": GPU_4070S, "compute_capability": "8.9"}
            self.assertTrue(any("sm8.6" in p for p in ek.embedded_device_problems(first, facts)))


class FingerprintTest(StoreTestBase):
    def test_fingerprint_roundtrip_and_problems(self):
        facts = json.loads(self.facts_path.read_text())
        fp = ek.make_fingerprint("unet_ts", facts, 8, "adopted", engine_bytes=10)
        self.assertEqual(fp["schema"], "musetalk_unet_engine_v1")
        self.assertEqual(fp["engine_key"], KEY_TS)
        self.assertFalse(fp["validation"]["passed"])
        entry = self.base / "entry"
        ek.write_fingerprint(entry, fp)
        self.assertEqual(ek.read_fingerprint(entry), json.loads((entry / "fingerprint.json").read_text()))
        self.assertIsNone(ek.read_fingerprint(self.base / "missing"))
        (entry / "fingerprint.json").write_text("{broken")
        with self.assertRaises(ValueError):
            ek.read_fingerprint(entry)
        bad = dict(fp, schema="nope")
        self.assertTrue(ek.fingerprint_problems(bad))
        with self.assertRaises(ValueError):
            ek.write_fingerprint(entry, bad)
        self.assertIn("missing torch_tensorrt_version", ek.fingerprint_problems(dict(fp, torch_tensorrt_version=None)))

    def test_usable_rules(self):
        facts = json.loads(self.facts_path.read_text())
        entry = self.base / "store" / KEY_TS / "bs8"
        entry.mkdir(parents=True)
        (entry / "unet_trt.ts").write_bytes(b"x" * 100)
        (entry / "unet_trt_meta.json").write_text("{}")
        fp = ek.make_fingerprint("unet_ts", facts, 8, "built", engine_bytes=100)
        item = {"dir": str(entry), "fingerprint": fp}
        verdict = ek.usable(item, facts)
        self.assertFalse(verdict)
        self.assertIn("not validated", verdict.reason)
        fp["validation"] = dict(fp["validation"], passed=True, status="passed", engine_key=KEY_TS)
        ok, reason = ek.usable(item, facts)
        self.assertTrue(ok, reason)
        self.assertEqual(ek.usable(item, facts).match, "exact")
        self.assertFalse(ek.usable(item, dict(facts, gpus=[{"index": 0, "name": "NVIDIA GeForce RTX 3090",
                                                            "compute_capability": "8.6"}])))
        other_trt = dict(facts, venv=dict(facts["venv"], tensorrt="10.9.0.34"))
        self.assertFalse(ek.usable(item, other_trt))
        other_tt = dict(facts, venv=dict(facts["venv"], torch_tensorrt="2.7.0"))
        self.assertFalse(ek.usable(item, other_tt))
        sibling = dict(facts, gpus=[{"index": 0, "name": "NVIDIA GeForce RTX 4090", "compute_capability": "8.9"}])
        self.assertFalse(ek.usable(item, sibling))
        self.assertEqual(ek.usable(item, sibling, allow_same_cc=True).match, "same_cc")
        fp["validation"]["engine_key"] = "sm89-other-trt10.3.0-tt2.5.0"
        self.assertIn("validation was recorded for key", ek.usable(item, facts).reason)
        fp["validation"]["engine_key"] = KEY_TS
        (entry / "unet_trt.ts").write_bytes(b"x" * 99)
        self.assertIn("size changed", ek.usable(item, facts).reason)
        (entry / "unet_trt.ts").unlink()
        os.symlink(str(self.base / "gone.ts"), str(entry / "unet_trt.ts"))
        self.assertIn("dangling symlink", ek.usable(item, facts).reason)


class AdoptTest(StoreTestBase):
    def test_adopt_refuses_device_mismatch(self):
        ts = self.legacy_ts(device="0%8%6%0%NVIDIA GeForce RTX 3090")
        before = sha(ts)
        rc, out = self.cli("adopt", "--ts", str(ts), expect=store.EXIT_REFUSED)
        self.assertEqual(out["status"], "refused")
        self.assertTrue(any("sm8.6" in p for p in out["problems"]))
        root = self.repo / "models/tensorrt_unet"
        self.assertFalse((root / KEY_TS).exists())
        self.assertEqual([p for p in root.iterdir() if ".partial-" in p.name], [])
        self.assertEqual(sha(ts), before)
        self.assertEqual(self.calls_for("validate_unet_backend"), [])

    def test_adopt_no_validate_then_validate_then_list(self):
        ts = self.legacy_ts()
        before = sha(ts)
        rc, out = self.cli("adopt", "--ts", str(ts), "--no-validate", expect=store.EXIT_NONE)
        entry = self.repo / "models/tensorrt_unet" / KEY_TS / "bs8"
        self.assertTrue((entry / "unet_trt.ts").is_symlink())
        self.assertEqual(os.readlink(str(entry / "unet_trt.ts")), os.path.realpath(str(ts)))
        self.assertFalse((entry / "unet_trt_meta.json").is_symlink())
        fp = ek.read_fingerprint(entry)
        self.assertEqual((fp["source"], fp["embedded_device"]), ("adopted", f"0%8%9%0%{GPU_4070S}"))
        self.assertIsNone(fp["engine_sha256"])
        self.assertFalse(out["usable"])
        rc, out = self.cli("validate", "--kind", "unet_ts", expect=store.EXIT_OK)
        self.assertTrue(out["usable"])
        call = self.calls_for("validate_unet_backend")[-1]
        argv = call["argv"]
        self.assertEqual(argv[argv.index("--backend") + 1], "trt")
        self.assertEqual(argv[argv.index("--trt-path") + 1], str(entry / "unet_trt.ts"))
        self.assertEqual(argv[argv.index("--limit") + 1], "16")
        self.assertEqual(argv[argv.index("--padded-batch-size") + 1], "8")
        self.assertNotIn("--group-captures", argv)
        meta = json.loads((entry / "unet_trt_meta.json").read_text())
        self.assertTrue(meta["validation"]["passed"])
        self.assertEqual(meta["validation_original"]["capture_dir"], "/elsewhere")
        self.assertEqual(json.loads((ts.parent / "unet_trt_meta.json").read_text())["validation"]["capture_dir"],
                         "/elsewhere")  # original meta untouched
        rc, out = self.cli("list", "--kind", "unet_ts", expect=store.EXIT_OK)
        info = out["kinds"]["unet_ts"]
        self.assertEqual(info["best"], str(entry))
        self.assertEqual(info["entries"][0]["env"], {"MUSETALK_TRT_UNET_PATHS": f"8:{entry / 'unet_trt.ts'}"})
        found = ek.find_engine("unet_ts", json.loads(self.facts_path.read_text()), self.repo)
        self.assertEqual(found["dir"], str(entry))
        # a failing gate demotes the entry and marks the runtime meta as failed
        with mock.patch.dict(os.environ, {"FAKE_MAE": "0.5"}):
            self.cli("validate", "--kind", "unet_ts", expect=store.EXIT_FAIL)
        self.assertFalse(ek.usable(entry, json.loads(self.facts_path.read_text())))
        self.assertIs(json.loads((entry / "unet_trt_meta.json").read_text())["validation"]["passed"], False)
        self.assertEqual(sha(ts), before)

    def test_adopt_is_idempotent_and_guards_replacement(self):
        ts = self.legacy_ts()
        self.cli("adopt", "--ts", str(ts), expect=store.EXIT_OK)
        calls = len(self.calls_for("validate_unet_backend"))
        rc, out = self.cli("adopt", "--ts", str(ts), expect=store.EXIT_OK)
        self.assertEqual(out["status"], "already_adopted")
        self.assertEqual(len(self.calls_for("validate_unet_backend")), calls)
        other = self.legacy_ts(rel="models/tensorrt_unet_other")
        rc, out = self.cli("adopt", "--ts", str(other), expect=store.EXIT_REFUSED)
        self.assertIn("already holds another engine", out["problems"][0])
        rc, out = self.cli("adopt", "--ts", str(other), "--force", expect=store.EXIT_OK)
        entry = self.repo / "models/tensorrt_unet" / KEY_TS / "bs8"
        self.assertEqual(os.readlink(str(entry / "unet_trt.ts")), os.path.realpath(str(other)))
        self.assertTrue(ts.exists())  # replacing an adopted entry never deletes the original engine

    def test_adopt_validation_failure_registers_nothing(self):
        ts = self.legacy_ts()
        with mock.patch.dict(os.environ, {"FAKE_MAE": "0.2"}):
            rc, out = self.cli("adopt", "--ts", str(ts), expect=store.EXIT_FAIL)
        self.assertEqual(out["status"], "validation_failed")
        self.assertFalse((self.repo / "models/tensorrt_unet" / KEY_TS / "bs8").exists())

    def test_validate_cudagraphs_mode_is_recorded_separately(self):
        ts = self.legacy_ts()
        self.cli("adopt", "--ts", str(ts), expect=store.EXIT_OK)
        with mock.patch.dict(os.environ, {"MUSETALK_TRT_UNET_CUDAGRAPHS": "runtime", "MUSETALK_UNET_BACKEND": "eager"}):
            rc, out = self.cli("validate", "--kind", "unet_ts", "--cudagraphs", "manual", expect=store.EXIT_OK)
        call = self.calls_for("validate_unet_backend")[-1]
        entry = self.repo / "models/tensorrt_unet" / KEY_TS / "bs8"
        self.assertEqual(call["argv"][call["argv"].index("--backend") + 1], "runtime")
        self.assertEqual(call["env"]["MUSETALK_TRT_UNET_CUDAGRAPHS"], "manual")  # caller's value scrubbed
        self.assertEqual(call["env"]["MUSETALK_UNET_BACKEND"], "trt")
        self.assertEqual(call["env"]["MUSETALK_TRT_UNET_PATHS"], f"8:{entry / 'unet_trt.ts'}")
        self.assertEqual(call["env"]["MUSETALK_TRT_FALLBACK"], "0")
        fp = ek.read_fingerprint(entry)
        self.assertTrue(fp["validation"]["passed"])
        self.assertTrue(fp["validation"]["modes"]["cudagraphs_manual"]["passed"])
        self.assertTrue((entry / "validation_cudagraphs_manual.json").exists())


class StagewiseTest(StoreTestBase):
    def test_legacy_stagewise_recognised_adopted_and_validated(self):
        legacy = self.legacy_stagewise()
        rc, out = self.cli("list", "--kind", "unet_stagewise", expect=store.EXIT_OK)
        candidates = out["kinds"]["unet_stagewise"]["legacy_candidates"]
        self.assertEqual([c["path"] for c in candidates], [str(legacy)])
        self.assertEqual(candidates[0]["compatible_problems"], [])
        with mock.patch.dict(os.environ, {"MUSETALK_UNET_STAGEWISE_PROBE_TOL": "1"}):
            rc, out = self.cli("adopt", "--dir", str(legacy), expect=store.EXIT_OK)
        entry = self.repo / "models/tensorrt_unet_stagewise" / KEY_PLANS / "bs16"
        for name in ["manifest.json", "probe_output.pt"] + [f"{b}.plan" for b in ek.STAGEWISE_BLOCKS]:
            self.assertTrue((entry / name).is_symlink(), name)
        call = self.calls_for("validate_unet_backend")[-1]
        argv, env = call["argv"], call["env"]
        self.assertEqual(argv[argv.index("--backend") + 1], "runtime")
        self.assertEqual(argv[argv.index("--group-captures") + 1], "2")
        self.assertEqual(env["MUSETALK_UNET_BACKEND"], "trt_stagewise")
        cache_dir = Path(env["MUSETALK_UNET_STAGEWISE_CACHE_DIR"])  # validated in the partial, before promotion
        self.assertTrue(cache_dir.name.startswith(f".{KEY_PLANS}.partial-bs16-"), cache_dir)
        self.assertEqual(cache_dir.parent, entry.parent.parent)
        self.assertEqual(env["MUSETALK_UNET_STAGEWISE_BATCH"], "16")
        self.assertNotIn("MUSETALK_UNET_STAGEWISE_PROBE_TOL", env)  # caller knobs never leak into validation
        found = ek.find_engine("unet_stagewise", json.loads(self.facts_path.read_text()), self.repo)
        self.assertEqual(found["env"], {"MUSETALK_UNET_STAGEWISE_CACHE_DIR": str(entry.parent),
                                        "MUSETALK_UNET_STAGEWISE_BATCH": "16"})
        self.assertEqual(found["quality_gate"]["verdict"], "PASS")
        # a rebuilt legacy manifest invalidates the adopted entry until it is re-validated
        manifest = json.loads((legacy / "manifest.json").read_text())
        manifest["probe"]["output_sha256"] = "ef" * 32
        (legacy / "manifest.json").write_text(json.dumps(manifest))
        verdict = ek.usable(entry, json.loads(self.facts_path.read_text()))
        self.assertFalse(verdict)
        self.assertIn("manifest.json changed", verdict.reason)

    def test_stagewise_refuses_other_gpu_and_incomplete(self):
        legacy = self.legacy_stagewise(gpu="NVIDIA GeForce RTX 3090", cc=(8, 6))
        rc, out = self.cli("adopt", "--dir", str(legacy), expect=store.EXIT_REFUSED)
        self.assertTrue(any("sm8.6" in p for p in out["problems"]))

    def test_stagewise_build_and_incomplete_resume(self):
        with mock.patch.dict(os.environ, {"FAKE_STAGEWISE_INCOMPLETE": "1"}):
            rc, out = self.cli("build", "--kind", "unet_stagewise", "--batch", "16", expect=store.EXIT_FAIL)
        root = self.repo / "models/tensorrt_unet_stagewise"
        resume = [p for p in root.iterdir() if p.name.endswith(".partial-bs16-resume")]
        self.assertEqual(len(resume), 1)  # kept so the next build resumes block by block
        self.assertFalse((root / KEY_PLANS / "bs16").exists())
        rc, out = self.cli("build", "--kind", "unet_stagewise", "--batch", "16", expect=store.EXIT_OK)
        entry = root / KEY_PLANS / "bs16"
        self.assertTrue(ek.usable(entry, json.loads(self.facts_path.read_text())))
        self.assertFalse(any(".partial-" in p.name for p in root.iterdir()))
        build_call = self.calls_for("build_unet_stagewise")[-1]["argv"]
        self.assertEqual(build_call[build_call.index("--timing-cache") + 1], str(root / KEY_PLANS / "timing_cache.bin"))
        self.assertEqual(ek.read_fingerprint(entry)["source"], "built")


class BuildTest(StoreTestBase):
    def test_build_unet_ts_success(self):
        rc, out = self.cli("build", "--kind", "unet_ts", expect=store.EXIT_OK)
        entry = self.repo / "models/tensorrt_unet" / KEY_TS / "bs8"
        fp = ek.read_fingerprint(entry)
        self.assertEqual(fp["source"], "built")
        self.assertEqual(fp["engine_sha256"], sha(entry / "unet_trt.ts"))
        self.assertEqual(fp["embedded_device"], f"0%8%9%0%{GPU_4070S}")
        self.assertTrue(ek.usable(entry, json.loads(self.facts_path.read_text())))
        argv = self.calls_for("tensorrt_export")[-1]["argv"]
        for flag, value in (("--components", "unet"), ("--batch-sizes", "8"), ("--save-format", "exported_program"),
                            ("--min-block-size", "1"), ("--validate-unet-limit", "16"),
                            ("--validate-unet-padded-batch-size", "8")):
            self.assertEqual(argv[argv.index(flag) + 1], value, flag)
        self.assertIn("--require-valid-unet", argv)
        self.assertTrue(argv[argv.index("--output-dir") + 1].endswith("/bs8"))
        self.assertIn(".partial-bs8-", argv[argv.index("--output-dir") + 1])
        rc, out = self.cli("build", "--kind", "unet_ts", expect=store.EXIT_OK)
        self.assertEqual(out["status"], "already_usable")
        self.assertTrue((self.base / "logs" / f"unet_engine_build_{KEY_TS}.log").exists())

    def test_build_failure_leaves_no_entry_or_partial(self):
        with mock.patch.dict(os.environ, {"FAKE_EXPORT_FAIL": "1"}):
            rc, out = self.cli("build", "--kind", "unet_ts", expect=store.EXIT_FAIL)
        root = self.repo / "models/tensorrt_unet"
        self.assertFalse((root / KEY_TS / "bs8").exists())
        self.assertEqual([p.name for p in root.iterdir() if ".partial-" in p.name], [])

    def test_build_preflight_refuses_low_ram_and_no_gpu(self):
        self.set_facts(mem_mb=4000)
        rc, out = self.cli("build", "--kind", "unet_ts", expect=store.EXIT_NONE)
        self.assertEqual(out["status"], "preflight_refused")
        self.assertTrue(any("MemAvailable" in p for p in out["preflight"]["problems"]))
        self.assertEqual(self.calls_for("tensorrt_export"), [])
        self.set_facts(gpus=[])
        rc, out = self.cli("build", "--kind", "unet_ts", expect=store.EXIT_NONE)
        self.assertIn("no engine key", out["error"])

    def test_taesd_build_and_flat_adopt(self):
        with mock.patch.dict(os.environ, {"MUSETALK_TAESD_TRT_OPT_LEVEL": "4", "MUSETALK_TAESD_TRT_STRICT": "0"}):
            rc, out = self.cli("build", "--kind", "taesd_trt", expect=store.EXIT_OK)
        call = self.calls_for("vae_fast_decoder")[-1]
        self.assertEqual(call["env"]["MUSETALK_TAESD_TRT_OPT_LEVEL"], "4")  # recipe knob kept
        self.assertNotIn("MUSETALK_TAESD_TRT_STRICT", call["env"])
        entry = self.repo / "models/taesd/trt" / KEY_PLANS / "bs8"
        found = ek.find_engine("taesd_trt", json.loads(self.facts_path.read_text()), self.repo)
        self.assertEqual(found["dir"], str(entry))
        self.assertEqual(found["env"]["MUSETALK_TAESD_TRT_DIR"], str(entry))
        self.assertEqual(found["env"]["MUSETALK_TAESD_TRT_BUILD"], "0")
        self.assertEqual(found["env"]["MUSETALK_TAESD_TRT_OPT_LEVEL"], "4")
        rc, out = self.cli("validate", "--kind", "taesd_trt", expect=store.EXIT_OK)
        verify = self.calls_for("vae_fast_decoder")[-1]
        self.assertEqual(verify["argv"][0], "verify")
        self.assertEqual(verify["env"]["MUSETALK_TAESD_TRT_BUILD"], "0")
        # flat legacy files written by `vae_fast_decoder.py build` into models/taesd/trt are adoptable
        flat = self.repo / "models/taesd/trt"
        for path in entry.iterdir():
            if path.name.startswith("taesd_trt_") and not path.name.endswith(".cache"):
                (flat / path.name).write_bytes(path.read_bytes())
        rc, out = self.cli("list", "--kind", "taesd_trt", expect=store.EXIT_OK)
        legacy = out["kinds"]["taesd_trt"]["legacy_candidates"]
        self.assertEqual(len(legacy), 1)
        self.assertEqual(legacy[0]["compatible_problems"], [])
        self.assertIsNone(out["kinds"]["taesd_trt"]["entries"][0]["quality_gate"]["verdict"])

    def test_taesd_gate_rewrite_keeps_entry_usable_but_rebuild_does_not(self):
        rc, out = self.cli("build", "--kind", "taesd_trt", expect=store.EXIT_OK)
        entry = self.repo / "models/taesd/trt" / KEY_PLANS / "bs8"
        facts = json.loads(self.facts_path.read_text())
        meta_path = [p for p in entry.glob("taesd_trt_*.json")][0]
        meta = json.loads(meta_path.read_text())
        meta["gate"] = {"verdict": "FAIL", "G_TAESD_full_max": 5, "report": "gate.json"}  # what the gate script does
        meta_path.write_text(json.dumps(meta, indent=2))
        described = ek.describe_entry(ek.list_entries("taesd_trt", self.repo)[0], facts)
        self.assertTrue(described["usable"], described["reasons"])
        self.assertEqual(described["quality_gate"]["verdict"], "FAIL")
        self.assertEqual(described["quality_gate"]["full_max_lsb"], 5)
        meta["decoder_plan_sha256"] = "00" * 32  # plans rebuilt under the same runtime key
        meta_path.write_text(json.dumps(meta))
        verdict = ek.usable(entry, facts)
        self.assertFalse(verdict)
        self.assertIn("decoder_plan_sha256 changed", verdict.reason)


class RemoteTest(StoreTestBase):
    def test_file_remote_round_trip_and_tamper(self):
        ts = self.legacy_ts()
        remote = self.base / "remote"
        rc, out = self.cli("publish", "--kind", "unet_ts", "--remote", f"file://{remote}", expect=store.EXIT_NONE)
        self.cli("adopt", "--ts", str(ts), "--no-validate", expect=store.EXIT_NONE)
        rc, out = self.cli("publish", "--kind", "unet_ts", "--remote", f"file://{remote}", expect=store.EXIT_REFUSED)
        self.cli("validate", "--kind", "unet_ts", expect=store.EXIT_OK)
        rc, out = self.cli("publish", "--kind", "unet_ts", "--remote", f"file://{remote}", expect=store.EXIT_OK)
        self.assertEqual(out["status"], "published")
        obj = remote / KEY_TS / "bs8"
        self.assertTrue((obj / "unet_trt.tar").exists() and (obj / "fingerprint.json").exists())
        remote_fp = json.loads((obj / "fingerprint.json").read_text())
        self.assertEqual(remote_fp["archive"]["sha256"], sha(obj / "unet_trt.tar"))
        self.assertEqual(remote_fp["engine_sha256"], sha(ts))
        rc, out = self.cli("publish", "--kind", "unet_ts", "--remote", f"file://{remote}", expect=store.EXIT_OK)
        self.assertEqual(out["status"], "already_published")
        self.assertFalse(any(p.name.startswith(".publish-") for p in (self.repo / "models/tensorrt_unet").iterdir()))
        # restore into a second store
        other_store = self.base / "store_b"
        rc, out = self.cli("restore", "--kind", "unet_ts", "--store", str(other_store), "--remote", f"file://{remote}",
                           expect=store.EXIT_OK)
        entry = other_store / KEY_TS / "bs8"
        self.assertFalse((entry / "unet_trt.ts").is_symlink())
        self.assertEqual(sha(entry / "unet_trt.ts"), sha(ts))
        fp = ek.read_fingerprint(entry)
        self.assertEqual(fp["source"], "restored")
        self.assertTrue(fp["validation"]["passed"])
        self.assertTrue(ek.usable(entry, json.loads(self.facts_path.read_text())))
        # tampered archive: refused, nothing registered, no partial left
        third = self.base / "store_c"
        data = bytearray((obj / "unet_trt.tar").read_bytes())
        data[600] ^= 0xFF
        (obj / "unet_trt.tar").write_bytes(bytes(data))
        rc, out = self.cli("restore", "--kind", "unet_ts", "--store", str(third), "--remote", f"file://{remote}",
                           expect=store.EXIT_REFUSED)
        self.assertIn("checksum", out["error"])
        self.assertFalse((third / KEY_TS / "bs8").exists())
        self.assertEqual([p.name for p in third.iterdir() if ".partial-" in p.name], [])
        # another GPU asks the same remote: nothing for its key
        self.set_facts(name="NVIDIA GeForce RTX 3090", cc="8.6")
        rc, out = self.cli("restore", "--kind", "unet_ts", "--store", str(third), "--remote", f"file://{remote}",
                           expect=store.EXIT_NONE)
        self.assertEqual(out["status"], "not_found")

    def test_stagewise_publish_restore_round_trip(self):
        legacy = self.legacy_stagewise()
        self.cli("adopt", "--dir", str(legacy), expect=store.EXIT_OK)
        remote = self.base / "remote_sw"
        rc, out = self.cli("publish", "--kind", "unet_stagewise", "--batch", "16", "--remote", f"file://{remote}",
                           expect=store.EXIT_OK)
        self.assertEqual(len(out["archive"]["members"]), 13)  # 11 plans + manifest + probe output
        other = self.base / "store_sw_b"
        rc, out = self.cli("restore", "--kind", "unet_stagewise", "--batch", "16", "--store", str(other),
                           "--remote", f"file://{remote}", expect=store.EXIT_OK)
        entry = other / KEY_PLANS / "bs16"
        self.assertFalse((entry / "head.plan").is_symlink())
        self.assertEqual(sha(entry / "manifest.json"), sha(legacy / "manifest.json"))
        self.assertTrue(ek.usable(entry, json.loads(self.facts_path.read_text())))
        # a plan whose bytes do not match the manifest hash is refused
        third = self.base / "store_sw_c"
        archive = remote / KEY_PLANS / "bs16" / "engine.tar"
        import tarfile
        work = self.base / "tamper"
        work.mkdir()
        with tarfile.open(str(archive)) as tar:
            tar.extractall(str(work))
        (work / "mid.plan").write_bytes(b"x" * (work / "mid.plan").stat().st_size)
        with tarfile.open(str(archive), "w") as tar:
            for name in sorted(os.listdir(str(work))):
                tar.add(str(work / name), arcname=name)
        fp_path = remote / KEY_PLANS / "bs16" / "fingerprint.json"
        remote_fp = json.loads(fp_path.read_text())
        remote_fp["archive"].update(sha256=sha(archive), bytes=archive.stat().st_size)
        fp_path.write_text(json.dumps(remote_fp))
        rc, out = self.cli("restore", "--kind", "unet_stagewise", "--batch", "16", "--store", str(third),
                           "--remote", f"file://{remote}", expect=store.EXIT_REFUSED)
        self.assertIn("mid.plan sha256 mismatch", out["error"])
        self.assertFalse((third / KEY_PLANS / "bs16").exists())

    def test_build_does_not_publish_unless_asked(self):
        remote = self.base / "remote"
        with mock.patch.dict(os.environ, {"MUSETALK_UNET_ENGINE_REMOTE": f"file://{remote}"}):
            self.cli("build", "--kind", "taesd_trt", expect=store.EXIT_OK)
            self.assertFalse(remote.exists())
            self.cli("build", "--kind", "taesd_trt", "--force", "--publish", expect=store.EXIT_OK)
        self.assertFalse(remote.exists())  # MUSETALK_UNET_ENGINE_REMOTE is the unet_ts remote only
        with mock.patch.dict(os.environ, {"MUSETALK_TAESD_TRT_ENGINE_REMOTE": f"file://{remote}"}):
            rc, out = self.cli("build", "--kind", "taesd_trt", "--force", "--publish", expect=store.EXIT_OK)
        self.assertEqual(out["publish"]["status"], "published")
        self.assertTrue((remote / KEY_PLANS / "bs8" / "engine.tar").exists())


class PartialAndEnsureTest(StoreTestBase):
    def test_partials_and_promote(self):
        root = self.repo / "models/tensorrt_unet"
        dead = subprocess.Popen([sys.executable, "-c", "pass"])
        dead.wait()
        stale = root / f".{KEY_TS}.partial-bs8-{dead.pid}-1"
        live = root / f".{KEY_TS}.partial-bs8-{os.getppid()}-2"
        resume = root / f".{KEY_PLANS}.partial-bs16-resume"
        for path in (stale, live, resume):
            (path / "bs8").mkdir(parents=True)
        target = self.base / "outside_engine.ts"
        target.write_bytes(b"keep me")
        os.symlink(str(target), str(stale / "bs8" / "unet_trt.ts"))
        self.assertEqual(ek.list_entries("unet_ts", self.repo), [])
        rc, out = self.cli("clean", "--kind", "unet_ts", expect=store.EXIT_OK)
        self.assertEqual(out["removed"]["unet_ts"], [str(stale)])
        self.assertTrue(live.exists() and resume.exists())
        self.assertEqual(target.read_bytes(), b"keep me")  # rmtree never follows engine symlinks
        rc, out = self.cli("clean", "--kind", "unet_ts", "--resume", expect=store.EXIT_OK)
        self.assertFalse(resume.exists())
        # promote replaces an existing entry and removes the old copy
        final = root / KEY_TS / "bs8"
        final.mkdir(parents=True)
        (final / "marker").write_text("old")
        partial = store.new_partial(root, KEY_TS, 8)
        (partial / "bs8" / "marker").write_text("new")
        with contextlib.redirect_stderr(io.StringIO()):
            store.promote(partial, 8, root, KEY_TS)
        self.assertEqual((final / "marker").read_text(), "new")
        self.assertFalse(partial.exists())
        self.assertFalse(any(".old-" in p.name for p in root.iterdir()))
        with self.assertRaises(store.StoreError):
            store.safe_rmtree(self.base / "logs", root)

    def test_ensure_paths(self):
        rc, out = self.cli("ensure", "--kind", "unet_ts", "--provision", "off", expect=store.EXIT_NONE)
        self.assertEqual(out["result"], "none")
        rc, out = self.cli("ensure", "--kind", "unet_ts", "--provision", "adopt", "--require", expect=store.EXIT_USAGE)
        self.legacy_ts(device="0%8%6%0%NVIDIA GeForce RTX 3090", rel="models/trt_backup/tensorrt_unet_static_bs8")
        self.legacy_ts()
        rc, out = self.cli("ensure", "--kind", "unet_ts", "--provision", "auto", expect=store.EXIT_OK)
        self.assertEqual(out["result"], "usable")
        self.assertEqual([s["step"] for s in out["steps"]], ["check", "adopt"])
        self.assertEqual(out["entry"]["env"]["MUSETALK_TRT_UNET_PATHS"].split(":", 1)[0], "8")
        self.assertEqual(self.calls_for("tensorrt_export"), [])
        calls = len(self.calls_for("validate_unet_backend"))
        rc, out = self.cli("ensure", "--kind", "unet_ts", expect=store.EXIT_OK)
        self.assertEqual([s["step"] for s in out["steps"]], ["check"])
        self.assertEqual(len(self.calls_for("validate_unet_backend")), calls)

    def test_ensure_revalidates_unvalidated_entry_and_skips_failed(self):
        ts = self.legacy_ts()
        self.cli("adopt", "--ts", str(ts), "--no-validate", expect=store.EXIT_NONE)
        rc, out = self.cli("ensure", "--kind", "unet_ts", "--provision", "adopt", expect=store.EXIT_OK)
        self.assertEqual([s["step"] for s in out["steps"]], ["check", "revalidate"])
        entry = self.repo / "models/tensorrt_unet" / KEY_TS / "bs8"
        with mock.patch.dict(os.environ, {"FAKE_MAE": "0.9"}):
            self.cli("validate", "--kind", "unet_ts", expect=store.EXIT_FAIL)
            calls = len(self.calls_for("validate_unet_backend"))
            rc, out = self.cli("ensure", "--kind", "unet_ts", "--provision", "adopt", expect=store.EXIT_NONE)
        self.assertEqual(len(self.calls_for("validate_unet_backend")), calls)  # a FAILED engine is not retried
        self.assertTrue(any("FAILED validation" in (s.get("reason") or "") for s in out["steps"]))
        self.assertTrue(entry.exists())

    def test_ensure_low_ram_keeps_unvalidated_entry_then_validates_later(self):
        ts = self.legacy_ts()
        self.set_facts(mem_mb=8000)  # below the 10 GB validation floor and the 14 GB build floor
        rc, out = self.cli("ensure", "--kind", "unet_ts", expect=store.EXIT_NONE)
        entry = self.repo / "models/tensorrt_unet" / KEY_TS / "bs8"
        self.assertEqual(ek.read_fingerprint(entry)["validation"]["status"], "not_run")
        self.assertEqual(self.calls_for("validate_unet_backend"), [])
        rc, out = self.cli("ensure", "--kind", "unet_ts", expect=store.EXIT_NONE)
        steps = {s["step"]: s["status"] for s in out["steps"]}
        self.assertEqual(steps["revalidate"], "not_run")
        self.assertEqual(steps["adopt"], "skipped")
        self.assertEqual(steps["build"], "preflight_refused")
        self.set_facts()
        rc, out = self.cli("ensure", "--kind", "unet_ts", expect=store.EXIT_OK)
        self.assertEqual([s["step"] for s in out["steps"]], ["check", "revalidate"])
        self.assertEqual(os.readlink(str(entry / "unet_trt.ts")), os.path.realpath(str(ts)))

    def test_ensure_readopts_stale_entry(self):
        ts = self.legacy_ts()
        self.cli("ensure", "--kind", "unet_ts", expect=store.EXIT_OK)
        ts.write_bytes(fake_ts_bytes(f"0%8%9%0%{GPU_4070S}", pad=7000))  # engine replaced in place (other size)
        rc, out = self.cli("ensure", "--kind", "unet_ts", "--provision", "adopt", expect=store.EXIT_OK)
        self.assertEqual([s["status"] for s in out["steps"]][:3], ["none", "stale", "adopted"])
        entry = self.repo / "models/tensorrt_unet" / KEY_TS / "bs8"
        self.assertEqual(ek.read_fingerprint(entry)["engine_bytes"], ts.stat().st_size)
        ts.write_bytes(fake_ts_bytes("0%8%6%0%NVIDIA GeForce RTX 3090"))  # e.g. the 3090 bundle restored over it
        rc, out = self.cli("ensure", "--kind", "unet_ts", "--provision", "adopt", expect=store.EXIT_NONE)
        self.assertFalse(ek.usable(entry, json.loads(self.facts_path.read_text())))

    def test_last_json_object_in_mixed_log(self):
        text = 'INFO x\n{"key": "a", "probe": 1}\nnoise {"x"\n{\n "key": "b",\n "probe": {"p": 2}\n}\ntrailing log\n'
        self.assertEqual(store.last_json_object(text, ("key", "probe"))["key"], "b")
        self.assertIsNone(store.last_json_object("no json here", ("key",)))

    def test_ensure_builds_when_nothing_to_adopt_and_skips_on_low_ram(self):
        self.set_facts(mem_mb=6000)
        rc, out = self.cli("ensure", "--kind", "unet_ts", expect=store.EXIT_NONE)
        build = [s for s in out["steps"] if s["step"] == "build"][0]
        self.assertEqual(build["status"], "preflight_refused")
        self.set_facts()
        rc, out = self.cli("ensure", "--kind", "unet_ts", expect=store.EXIT_OK)
        self.assertEqual(out["steps"][-1]["step"], "build")
        self.assertEqual(len(self.calls_for("tensorrt_export")), 1)


class CorpusAndImportTest(unittest.TestCase):
    def test_tracked_portable_corpus(self):
        corpus = REPO / "calibration/unet_portable_bs8"
        if not corpus.exists():
            self.skipTest("portable corpus not present in this checkout")
        result = store.check_corpus(corpus)
        self.assertTrue(result["ok"], result["problems"])
        self.assertEqual(result["files"], 16)
        manifest = json.loads((corpus / "manifest.json").read_text())
        self.assertGreaterEqual(len(manifest["avatars"]), 14)
        self.assertEqual({r["split"] for r in manifest["files"]}, {"main", "holdout"})
        self.assertTrue(all("_bs8_" in r["file"] for r in manifest["files"]))

    def test_cli_never_imports_torch(self):
        code = (
            "import sys; sys.modules['torch'] = None; sys.modules['tensorrt'] = None; "
            "sys.modules['torch_tensorrt'] = None; sys.path.insert(0, %r); "
            "from scripts import unet_engine_store as s; "
            "rc = s.main(['list', '--repo-root', %r]); "
            "assert 'torch' not in [m for m in sys.modules if sys.modules[m] is not None]; raise SystemExit(rc)"
        ) % (str(REPO), tempfile.gettempdir())
        env = dict(os.environ)
        with tempfile.TemporaryDirectory() as tmp:
            facts = Path(tmp) / "facts.json"
            facts.write_text(json.dumps({"gpus": [], "venv": {}}))
            env.update({"MUSETALK_HOST_FACTS_JSON": str(facts)})
            proc = subprocess.run([sys.executable, "-c", code], env=env, stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE, universal_newlines=True, timeout=60)
        self.assertEqual(proc.returncode, 0, proc.stderr)
        self.assertIn('"kinds"', proc.stdout)


if __name__ == "__main__":
    unittest.main()
