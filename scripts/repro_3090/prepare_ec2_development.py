#!/usr/bin/env python3
"""Run on the existing EC2 host: read-only cloud preflight + private local payload.

This does NOT rent, update a template, or alter any service. Output files contain
the existing template's bootstrap credentials and must stay outside Git, mode
0600, in a 0700 directory. Never print the raw files or commit them.
"""
import argparse
import datetime as dt
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


class PreparationInvalid(RuntimeError):
    pass


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def sanitize_script(script):
    result = []
    for line in script.splitlines():
        match = re.match(r"(\s*(?:export\s+)?)([A-Z][A-Z0-9_]*)=(.*)", line)
        if match and any(word in match[2] for word in ("ACCESS_KEY", "SECRET_ACCESS", "TOKEN", "PASSWORD")):
            line = f"{match[1]}{match[2]}='[REDACTED]'"
        result.append(line)
    return "\n".join(result) + "\n"


def transform(script, revision):
    if not re.fullmatch(r"[a-f0-9]{40}", revision):
        raise PreparationInvalid("Exact 40-character revision required")
    replacements = {
        'REPO_BRANCH="main"': 'REPO_BRANCH="codex/rtx3090-r5-delivery"\nREPO_REV="' + revision + '"',
        'echo "[bootstrap] publishing git checkout"':
            'git -C "$STAGE_DIR" fetch --depth 1 origin "$REPO_REV"\n'
            'git -C "$STAGE_DIR" checkout --detach "$REPO_REV"\n'
            'test "$(git -C "$STAGE_DIR" rev-parse HEAD)" = "$REPO_REV"\n\n'
            'echo "[bootstrap] publishing git checkout"',
        'SETUP_CLEAN=1 \\\n':
            '# Isolated development worker: callback opt-out is code-supported at the pinned revision.\n'
            '# The runtime secret was checked below and contains no overrides of these flags.\n'
            'export LINGUA_CONTROL_PLANE_ENABLED=0\n'
            'export LINGUA_WORKER_CALLBACK_REQUIRED=0\n\nSETUP_CLEAN=1 \\\n',
    }
    for old, new in replacements.items():
        if script.count(old) != 1:
            raise PreparationInvalid("Template startup no longer matches the reviewed baseline")
        script = script.replace(old, new)
    return script


def save_private(root, name, data):
    target = root / name
    if target.exists():
        raise PreparationInvalid("Private evidence target already exists; use a new directory")
    fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(data)
        handle.flush()
        os.fsync(handle.fileno())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--label", required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--offer-id", type=int)
    selection.add_argument("--machine-id", type=int, help="Resolve its fresh, possibly changing bundle offer ID")
    parser.add_argument("--min-download-mbps", type=float, default=2000)
    args = parser.parse_args()
    root = Path(args.output_dir).resolve()
    allowed = Path("/home/ec2-user/.local/state").resolve()
    if not root.is_relative_to(allowed) or root == allowed:
        raise PreparationInvalid("Protected evidence must be below /home/ec2-user/.local/state")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{3,119}", args.label):
        raise PreparationInvalid("Invalid unique label")
    logging.disable(logging.CRITICAL)
    pid = subprocess.check_output(["systemctl", "show", "lingua-api", "-p", "MainPID", "--value"], text=True).strip()
    env = dict(x.split("=", 1) for x in Path(f"/proc/{pid}/environ").read_text().split("\0") if "=" in x)
    os.environ.update(env)
    os.chdir("/home/ec2-user/lingua/backend")
    sys.path.insert(0, os.getcwd())
    from services.secrets_loader import bootstrap_secrets
    bootstrap_secrets()
    import requests
    import boto3
    from services.vast_client import VastClient, offer_cost_violation
    response = requests.get("https://console.vast.ai/api/v0/template/", params={
        "select_cols": json.dumps(["*"]), "select_filters": json.dumps({"creator_id": {"eq": 356317}})},
        headers={"Authorization": "Bearer " + os.environ["VAST_API_KEY"]}, timeout=30)
    if response.status_code != 200:
        raise PreparationInvalid("Template read failed; response omitted")
    templates = response.json()["templates"]
    baseline = next(t for t in templates if t["id"] == 408714)
    target = next(t for t in templates if t["id"] == 756005)
    if target["name"] != "-(NEEDS UPDATE )PyTorch (Vast) - Musetalk w/ NVENC enabled & 12.1 CUDA":
        raise PreparationInvalid("Target template identity changed")
    if sha(baseline["onstart"]) != "14ccb8011f7b5b2cb10e77551ea32826fd729f24eef2570cea9493a086eb8c39":
        raise PreparationInvalid("Baseline template startup changed")
    values = {}
    for line in baseline["onstart"].splitlines():
        if line.strip().startswith("export "):
            for token in shlex.split(line.strip()[7:]):
                if "=" in token:
                    k, v = token.split("=", 1)
                    values[k] = v
    secret = boto3.client("secretsmanager", region_name="us-east-1",
                          aws_access_key_id=values["AWS_ACCESS_KEY_ID"],
                          aws_secret_access_key=values["AWS_SECRET_ACCESS_KEY"])
    payload = json.loads(secret.get_secret_value(SecretId=values["MUSETALK_AWS_SECRET_ID"])["SecretString"])
    if any(k.startswith("LINGUA_") for k in payload):
        raise PreparationInvalid("Runtime secret can override callback isolation; re-review required")
    s3 = boto3.client("s3", region_name=payload.get("AWS_DEFAULT_REGION", "us-east-1"),
                      aws_access_key_id=payload["AWS_ACCESS_KEY_ID"],
                      aws_secret_access_key=payload["AWS_SECRET_ACCESS_KEY"],
                      aws_session_token=payload.get("AWS_SESSION_TOKEN"))
    s3.head_bucket(Bucket=payload["AVATAR_S3_BUCKET"])
    script = transform(baseline["onstart"], args.revision)
    parsed = subprocess.run(["bash", "-n"], input=script, text=True, capture_output=True)
    if parsed.returncode:
        raise PreparationInvalid("Transformed startup shell did not parse")
    filters = {"limit": 100, "type": "ondemand", "rentable": {"eq": True}, "verified": {"eq": True},
               "num_gpus": {"eq": 1}, "gpu_name": {"eq": "RTX 3090"}, "cpu_cores_effective": {"gte": 8},
               "cpu_ram": {"gte": 32000}, "cpu_ghz": {"gte": 2}, "disk_space": {"gte": 150},
               "allocated_storage": 150, "inet_down": {"gte": args.min_download_mbps}, "inet_down_cost": {"lte": 1.50 / 1024},
               "inet_up_cost": {"lte": 1.50 / 1024}, "dph_total": {"lte": .30}, "storage_cost": {"lte": .40},
               "reliability": {"gte": .90}, "duration": {"gte": 86400}, "order": [["dph_total", "asc"]]}
    offers = VastClient().search_offers(filters)
    offer = next((o for o in offers if (o.get("id") == args.offer_id if args.offer_id is not None
                                      else o.get("machine_id") == args.machine_id)
                  and not offer_cost_violation(o, filters)), None)
    if offer is None or offer_cost_violation(offer, filters):
        print(json.dumps({"currently_matching_offers": [{k: o.get(k) for k in (
            "id", "machine_id", "gpu_name", "dph_total", "dph_total_adj", "cpu_name", "cpu_cores_effective",
            "cpu_ram", "inet_down", "inet_up", "inet_down_cost", "inet_up_cost", "storage_cost", "disk_bw")}
            for o in offers]}, sort_keys=True))
        raise PreparationInvalid("Selected offer disappeared or violates current cost caps")
    request = {"count": 1, "offer_id": offer["id"], "create_request": {
        "template_hash_id": baseline["hash_id"], "label": args.label, "disk": 150,
        "onstart": script, "env": {"LINGUA_CONTROL_PLANE_ENABLED": "0", "LINGUA_WORKER_CALLBACK_REQUIRED": "0"}}}
    observed = dt.datetime.now(dt.timezone.utc).isoformat()
    root.mkdir(parents=True, mode=0o700)
    os.chmod(root, 0o700)
    for item in (baseline, target):
        save_private(root, f"template-{item['id']}.before.json", json.dumps(item, indent=2, sort_keys=True) + "\n")
    save_private(root, "development-create.json", json.dumps(request, indent=2, sort_keys=True) + "\n")
    save_private(root, "development-onstart.sh", script)
    save_private(root, "offer.json", json.dumps({"observed_at_utc": observed, "allocated_storage_gb": 150,
                                                "offer": offer}, indent=2, sort_keys=True) + "\n")
    summary = {"observed_at_utc": observed, "private_directory": str(root), "cloud_writes": False,
               "files_mode": "0600", "directory_mode": "0700", "revision": args.revision,
               "label": args.label, "offer_id": offer["id"], "machine_id": offer["machine_id"],
               "dph_total": offer["dph_total"], "inet_down_cost": offer["inet_down_cost"],
               "inet_up_cost": offer["inet_up_cost"], "bootstrap_secret_access": True,
               "cpu_name": offer.get("cpu_name"), "cpu_cores_effective": offer.get("cpu_cores_effective"),
               "cpu_ram": offer.get("cpu_ram"), "inet_down": offer.get("inet_down"),
               "inet_up": offer.get("inet_up"), "disk_bw": offer.get("disk_bw"),
               "dph_total_adj": offer.get("dph_total_adj"), "storage_cost": offer.get("storage_cost"),
               "reliability": offer.get("reliability"), "geolocation": offer.get("geolocation"),
               "runtime_s3_bucket_head": True, "runtime_secret_key_names": sorted(payload),
               "runtime_secret_has_lingua_overrides": False, "baseline_template_id": baseline["id"],
               "target_template_id": target["id"], "target_template_hash": target["hash_id"],
               "baseline_onstart_sha256": sha(baseline["onstart"]), "development_onstart_sha256": sha(script),
               "target_template_sha256": sha(json.dumps(target, sort_keys=True)),
               "development_onstart_sanitized": sanitize_script(script)}
    save_private(root, "preparation-summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        detail = str(exc) if isinstance(exc, PreparationInvalid) else type(exc).__name__
        print(f"Preparation failed: {detail}; no API writes issued", file=sys.stderr)
        raise SystemExit(1)
