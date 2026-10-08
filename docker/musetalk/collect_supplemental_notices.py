#!/usr/bin/env python3
"""Capture a fixed, primary-source notice dossier; never install or fetch models.

This is review evidence, not a release manifest or a license-approval engine.
Version-tagged/dynamic references are hashed at capture and must subsequently be
compared with notices in the exact built image. No claim of binary identity is
made for an upstream source-license file.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

SOURCES = {
    "compvis_autoencoder_mit": "https://raw.githubusercontent.com/CompVis/latent-diffusion/a506df5756472e2ebaf9078affdde2c4f1502cd4/LICENSE",
    "compvis_autoencoder_readme": "https://raw.githubusercontent.com/CompVis/latent-diffusion/a506df5756472e2ebaf9078affdde2c4f1502cd4/README.md",
    "nvidia_container_license": "https://gitlab.com/nvidia/container-images/cuda/-/raw/master/NGC-DL-CONTAINER-LICENSE",
    "cuda_12_1_1_base_recipe": "https://gitlab.com/nvidia/container-images/cuda/-/raw/master/dist/12.1.1/ubuntu2204/base/Dockerfile",
    "cuda_12_1_1_eula": "https://docs.nvidia.com/cuda/archive/12.1.1/eula/index.html",
    "cudnn_9_1_0_eula": "https://docs.nvidia.com/deeplearning/cudnn/backend/v9.1.0/reference/eula.html",
    "tensorrt_10_3_libs_license": "https://raw.githubusercontent.com/NVIDIA/TensorRT/c5b9de37f7ef9034e2efc621c664145c7c12436e/python/packaging/libs_wheel/LICENSE.txt",
    "tensorrt_10_3_bindings_license": "https://raw.githubusercontent.com/NVIDIA/TensorRT/c5b9de37f7ef9034e2efc621c664145c7c12436e/python/packaging/bindings_wheel/LICENSE.txt",
    "tensorrt_10_3_oss_license": "https://raw.githubusercontent.com/NVIDIA/TensorRT/c5b9de37f7ef9034e2efc621c664145c7c12436e/LICENSE",
    "tensorrt_10_3_oss_notice": "https://raw.githubusercontent.com/NVIDIA/TensorRT/c5b9de37f7ef9034e2efc621c664145c7c12436e/NOTICE",
    "torch_tensorrt_2_5_license": "https://raw.githubusercontent.com/pytorch/TensorRT/v2.5.0/LICENSE",
    "pytorch_2_5_1_license": "https://raw.githubusercontent.com/pytorch/pytorch/v2.5.1/LICENSE",
    "mmcv_2_1_license": "https://raw.githubusercontent.com/open-mmlab/mmcv/v2.1.0/LICENSE",
    "pyav_16_1_license": "https://raw.githubusercontent.com/PyAV-Org/PyAV/v16.1.0/LICENSE.txt",
    "pyav_16_1_vendor_reference": "https://raw.githubusercontent.com/PyAV-Org/PyAV/v16.1.0/scripts/ffmpeg-latest.json",
    "pyav_ffmpeg_8_0_1_3_build": "https://raw.githubusercontent.com/PyAV-Org/pyav-ffmpeg/8.0.1-3/scripts/build-ffmpeg.py",
    "pyav_ffmpeg_8_0_1_3_patch": "https://raw.githubusercontent.com/PyAV-Org/pyav-ffmpeg/8.0.1-3/patches/ffmpeg.patch",
    "imageio_ffmpeg_0_6_license": "https://raw.githubusercontent.com/imageio/imageio-ffmpeg/v0.6.0/LICENSE",
    "ffmpeg_legal": "https://ffmpeg.org/legal.html",
}
PACKAGES = {"mmengine": "0.10.4", "mmdet": "3.2.0", "mmpose": "1.3.1",
            "chumpy": "0.70", "soundfile": "0.12.1", "soxr": "1.1.0", "certifi": "2026.7.22",
            "kokoro": "0.9.4", "misaki": "0.9.4", "espeakng-loader": "0.2.4", "phonemizer-fork": "3.3.2"}


def package_metadata(output):
    if output.exists():
        raise ValueError("Refusing metadata evidence overwrite")
    def one(item):
        name, version = item
        url = f"https://pypi.org/pypi/{name}/{version}/json"
        data = subprocess.check_output(["curl", "--fail", "--silent", "--show-error", "--max-time", "30",
                                        "--max-filesize", "1048576", "--proto", "=https", url])
        if len(data) > 1024 ** 2:
            raise ValueError("Package metadata too large")
        info = json.loads(data)["info"]
        return {"name": name, "version": version, "url": url,
                "response_sha256": hashlib.sha256(data).hexdigest(),
                "license": info.get("license"), "license_expression": info.get("license_expression"),
                "license_classifiers": [x for x in info.get("classifiers", []) if x.startswith("License")],
                "project_urls": info.get("project_urls"), "built_image_binding": "NOT_YET_CAPTURED"}
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(one, sorted(PACKAGES.items())))
    with output.open("x") as stream:
        json.dump({"schema": "musetalk_package_license_metadata_evidence_v1", "review_date_utc": "2026-10-08",
                   "scope": "Uploader-declared primary package metadata, not a final SBOM or blanket grant for bundled libraries",
                   "packages": result}, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"output": str(output), "package_metadata_records": len(result)}))


def capture(output):
    output.mkdir(parents=True, exist_ok=False)
    def one(item):
        name, url = item
        data = subprocess.check_output(["curl", "--fail", "--silent", "--show-error", "--location",
                                        "--max-time", "30", "--max-filesize", "1048576",
                                        "--proto", "=https", "--proto-redir", "=https", url])
        if len(data) > 1024 ** 2:
            raise ValueError("Notice/reference exceeds size limit")
        data.decode("utf-8")
        path = name + (".html" if "<html" in data[:2000].decode().lower() else ".txt")
        (output / path).write_bytes(data)
        return name, {"url": url, "path": path, "sha256": hashlib.sha256(data).hexdigest(),
                      "size_bytes": len(data), "built_image_binding": "NOT_YET_CAPTURED"}
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = dict(pool.map(one, sorted(SOURCES.items())))
    (output / "sources.json").write_text(json.dumps({"schema": "musetalk_primary_notice_sources_v1",
        "capture_date_utc": "2026-10-08", "scope": "Primary upstream license/reference text only; no model or image download",
        "sources": result}, indent=2) + "\n")
    print(json.dumps({"output": str(output), "sources": len(result)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--output", type=Path)
    group.add_argument("--package-metadata-output", type=Path)
    args = parser.parse_args()
    if args.output:
        capture(args.output)
    else:
        package_metadata(args.package_metadata_output)
