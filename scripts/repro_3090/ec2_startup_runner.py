#!/usr/bin/env python3
"""EC2 credential bootstrap and narrowly scoped experimental-worker expiry.

Copies credential values only in process memory from the deployed service's
configuration/approved Secrets Manager path. Never prints credentials. Run with
the deployed /home/ec2-user/lingua/venv/bin/python and permission to read the
lingua-api service environment. Only an explicitly installed experimental timer
should invoke `expire`; no timer is installed by this program.
"""
from __future__ import annotations
import argparse
import datetime as dt
import importlib.util
import json
import logging
import os
from pathlib import Path
import re
import subprocess
import sys


def load_observer():
    spec = importlib.util.spec_from_file_location("startup_observer", Path(__file__).with_name("50_startup.py"))
    observer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(observer)
    return observer


def load_credentials():
    logging.disable(logging.CRITICAL)
    pid = subprocess.check_output(["systemctl", "show", "lingua-api", "-p", "MainPID", "--value"], text=True).strip()
    if not pid.isdigit() or pid == "0":
        raise RuntimeError("Production service environment is unavailable")
    os.environ.update(dict(x.split("=", 1) for x in Path(f"/proc/{pid}/environ").read_text().split("\0") if "=" in x))
    sys.path.insert(0, "/home/ec2-user/lingua/backend")
    from services.secrets_loader import bootstrap_secrets
    bootstrap_secrets()
    if not os.getenv("RUNTIME_CONFIG_TOKEN") or not os.getenv("VAST_API_KEY"):
        raise RuntimeError("Required admin/provider credentials are unavailable")


def validate_expiry_deadline(ledger, deadline, observer):
    """Validate the ledger binding before credentials or provider operations.

    Legacy ledgers without resource_deadline_utc retain their explicit CLI
    deadline. A present binding, including a malformed/null value, is never
    treated as legacy or silently replaced by the invocation's deadline.
    """
    requested = observer.utc_parse(deadline)
    if "resource_deadline_utc" in ledger:
        bound = ledger["resource_deadline_utc"]
        if not isinstance(bound, str):
            raise observer.Invalid("Bound resource deadline must be a UTC timestamp")
        if observer.utc_parse(bound) != requested:
            raise observer.Invalid("Expiry deadline differs from the bound resource deadline")
    return requested


def expire_owned(ledger, deadline, instances, api_fn, token, observer, now=None):
    """One reconciled, non-forced destroy attempt. Safe to rerun from a timer."""
    requested = validate_expiry_deadline(ledger, deadline, observer)
    now = now or dt.datetime.now(dt.timezone.utc)
    if now < requested:
        return {"status": "not_due", "cloud_writes": False}
    label = ledger.get("label", "")
    if not re.fullmatch(r"musetalk-r5-3090-(dev|release)-[A-Za-z0-9_.-]+", label):
        raise observer.Invalid("Expiry refuses an unrecognized experimental label")
    if ledger.get("state") == "initialized":
        return {"status": "never_submitted", "cloud_writes": False}
    if not any(e.get("name") == "request_started" for e in ledger.get("events", [])):
        raise observer.Invalid("Expiry requires evidence of this run's create attempt")
    matches = [row for row in instances if row.get("label") == label]
    if not matches:
        # A missing label is only proof of deletion when the recorded ID is also
        # absent; a relabelled instance is ambiguous, not cleanup success.
        if any(str(row.get("id")) == str(ledger.get("instance_id")) for row in instances):
            raise observer.Invalid("Recorded instance has been relabelled; expiry refuses it")
        return {"status": "absent", "cloud_writes": False, "instance_id": ledger.get("instance_id")}
    if len(matches) != 1:
        raise observer.Invalid("Duplicate experimental labels; expiry refuses ambiguous targets")
    instance = matches[0]
    instance_id = str(instance.get("id", ""))
    if instance_id == "51074906" or not instance_id.isdigit():
        raise observer.Invalid("Expiry refuses the protected production worker or an invalid ID")
    if ledger.get("instance_id") not in (None, instance_id):
        raise observer.Invalid("Expiry instance ID differs from this run's ledger")
    if instance.get("gpu_name") != "RTX 3090" or instance.get("num_gpus") != 1:
        raise observer.Invalid("Expiry target does not match the owned single-3090 experiment")
    response = api_fn(ledger["control_plane_url"], "/api/runtime/workers/musetalk/destroy", token,
                      method="POST", payload={"instance_id": instance_id, "force": False})
    result = response.get("result", {})
    if response.get("success") is not True or result.get("action") not in {"destroy", "drain"}:
        raise observer.Invalid("Expiry did not receive a successful destroy/drain response")
    return {"status": result["action"], "instance_id": instance_id, "label": label,
            "cloud_writes": True, "at_utc": now.isoformat(), "force": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    run = commands.add_parser("observer")
    run.add_argument("arguments", nargs=argparse.REMAINDER)
    expiry = commands.add_parser("expire")
    expiry.add_argument("--run-dir", required=True)
    expiry.add_argument("--not-before-utc", required=True)
    args = parser.parse_args(argv)
    observer = load_observer()
    try:
        if args.mode == "observer":
            load_credentials()
            return observer.main(args.arguments)
        path = Path(args.run_dir) / "startup-ledger.json"
        with observer.locked(path.with_suffix(".lock")):
            ledger = observer.load_json(path)
            validate_expiry_deadline(ledger, args.not_before_utc, observer)
            load_credentials()
            from services.vast_client import VastClient
            client = VastClient()
            outcome = expire_owned(ledger, args.not_before_utc, client.list_instances(), observer.api,
                                   os.environ["RUNTIME_CONFIG_TOKEN"], observer)
            if outcome["status"] == "destroy":
                # A HTTP destroy response alone is not proof storage charges ended.
                remains = any(str(row.get("id")) == outcome["instance_id"] for row in client.list_instances())
                outcome["provider_absence_verified"] = not remains
            observer.atomic_json(Path(args.run_dir) / "resource-expiry.json", outcome)
            print(json.dumps(outcome, sort_keys=True))
            return 0 if outcome["status"] in {"absent", "never_submitted", "not_due"} or outcome.get("provider_absence_verified") else 3
    except Exception as exc:
        print(f"EC2 observer operation failed ({type(exc).__name__}); inspect protected ledger", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
