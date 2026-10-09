#!/usr/bin/env python3
"""Auditable EC2/Vast startup observer. Stdlib only; create is explicitly opt-in.

The EC2 API is not idempotent. A durable submitting marker is written BEFORE
the one POST. Every exception leaves an ambiguous request that can only be
reconciled, never retried automatically. This program never runs autoscaler/run,
drains a worker, changes a template, or repairs a new instance.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

SCHEMA = "musetalk_startup_observer_v1"
EVENTS = (
    "request_started", "provider_accepted", "image_pull_started", "image_pull_finished",
    "container_started", "ssh_ready", "bootstrap_started", "secrets_ready",
    "artifacts_verified", "models_loaded", "gpu_warm", "health_ready",
    "registered", "avatar_ready", "routable", "first_usable_frame", "steady_call_ready",
)
REQUIRED_READY = {"request_started", "provider_accepted", "artifacts_verified",
                  "health_ready", "registered", "avatar_ready", "routable",
                  "first_usable_frame", "steady_call_ready"}
WORKER_EVENTS = set(EVENTS) - {"request_started", "provider_accepted", "registered",
                             "routable", "first_usable_frame", "steady_call_ready"}
PROCESS_CLOCK = f"{socket.gethostname()}:{uuid.uuid4()}"


class Invalid(RuntimeError):
    pass


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="microseconds")


def utc_parse(value):
    try:
        result = dt.datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (ValueError, TypeError) as exc:
        raise Invalid("A valid timezone-aware UTC timestamp is required") from exc
    if result.tzinfo is None or result.utcoffset() != dt.timedelta(0):
        raise Invalid("Timestamps must explicitly use UTC")
    return result


def number(value, name, minimum=0):
    if isinstance(value, bool):
        raise Invalid(f"{name} must be numeric")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise Invalid(f"{name} is missing or nonnumeric") from exc
    if not math.isfinite(result) or result < minimum:
        raise Invalid(f"{name} is nonfinite or below {minimum}")
    return result


def load_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


@contextlib.contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        os.chmod(path, 0o600)
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


def safe_base_url(value):
    parsed = urllib.parse.urlsplit(value)
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise Invalid("Base URL must not contain credentials, query, or fragment")
    if parsed.scheme != "https" and not (
        parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}
    ):
        raise Invalid("Use HTTPS, or an SSH tunnel to a loopback HTTP endpoint")
    if not parsed.hostname:
        raise Invalid("Base URL has no hostname")
    return value.rstrip("/")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Never forward admin credentials or an ambiguous creation to another URL.
        raise Invalid(f"API redirect refused (HTTP {code})")


def api(base, path, token, method="GET", payload=None, timeout=30):
    if not token:
        raise Invalid("Admin token environment variable is empty")
    request = urllib.request.Request(
        safe_base_url(base) + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"X-Admin-Token": token, "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
            value = json.load(response)
    except urllib.error.HTTPError as exc:
        # Provider/backend error bodies can echo credentials. Never emit them.
        raise Invalid(f"API HTTP {exc.code}; response body deliberately omitted") from None
    except (urllib.error.URLError, TimeoutError) as exc:
        raise Invalid(f"API transport failure: {type(exc).__name__}") from None
    if not isinstance(value, dict):
        raise Invalid("API returned a non-object JSON response")
    return value


def add_event(ledger, name, *, source, utc=None, evidence_ref=None,
              uncertainty_seconds=None, local=True):
    if name not in EVENTS:
        raise Invalid(f"Unknown startup event: {name}")
    for existing in ledger["events"]:
        if existing["name"] == name:
            if utc is not None and utc_parse(utc) != utc_parse(existing["utc"]):
                raise Invalid(f"Conflicting immutable event timestamp: {name}")
            return
    stamp = utc or utc_now()
    utc_parse(stamp)
    event = {"name": name, "utc": stamp, "source": source,
             "observed_at_utc": utc_now(), "evidence_ref": evidence_ref,
             "uncertainty_seconds": uncertainty_seconds}
    if local and utc is None:
        event.update(clock_domain=PROCESS_CLOCK, monotonic_ns=time.monotonic_ns())
    # Foreign monotonic readings are intentionally never imported.
    ledger["events"].append(event)


def initialize(run_dir, label, control_plane, cache_state="unknown", sla_seconds=None,
               resource_deadline_utc=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{3,119}", label):
        raise Invalid("Use a unique 4-120 character alphanumeric run label")
    path = Path(run_dir) / "startup-ledger.json"
    with locked(path.with_suffix(".lock")):
        if path.exists():
            raise Invalid("Run ledger already exists; observe or reconcile it")
        created_at = utc_now()
        deadline = None
        if resource_deadline_utc is not None:
            if not isinstance(resource_deadline_utc, str):
                raise Invalid("Resource deadline must be a UTC timestamp")
            deadline = utc_parse(resource_deadline_utc)
            remaining = (deadline - utc_parse(created_at)).total_seconds()
            if not 120 < remaining <= 24 * 3600:
                raise Invalid("Resource deadline must be more than 120 seconds and at most 24 hours ahead")
        ledger = {"schema": SCHEMA, "label": label, "created_at_utc": created_at,
                  "control_plane_url": safe_base_url(control_plane),
                  "cache_state": cache_state, "startup_sla_seconds": sla_seconds,
                  "state": "initialized", "instance_id": None, "events": [],
                  "observations": [], "errors": [], "media_verified": False}
        if deadline is not None:
            # An immutable intent binding, not timer installation or permission
            # to create a resource. Legacy ledgers omit this field entirely.
            ledger["resource_deadline_utc"] = deadline.isoformat()
        atomic_json(path, ledger)
    return ledger


def validate_budget(budget, offer_document, request):
    """Conservative reservation, not a claim that metadata authorizes spending."""
    for key in ("approval_reference", "lifetime_enforcer_reference"):
        if not isinstance(budget.get(key), str) or not budget[key].strip():
            raise Invalid(f"Budget requires {key}; no inferred authorization")
    total = number(budget.get("total_usd"), "total_usd", .000001)
    hourly_limit = number(budget.get("max_hourly_usd"), "max_hourly_usd", .000001)
    hours = number(budget.get("planned_lifetime_hours"), "planned_lifetime_hours", .000001)
    lifetime_limit = number(budget.get("max_lifetime_hours"), "max_lifetime_hours", .000001)
    if hours > lifetime_limit:
        raise Invalid("Planned lifetime exceeds authorized lifetime ceiling")
    observed = utc_parse(offer_document.get("observed_at_utc"))
    age = (dt.datetime.now(dt.timezone.utc) - observed).total_seconds()
    if age < -5 or age > 300:
        raise Invalid("Offer must have been rechecked within the last five minutes")
    offer = offer_document.get("offer", {})
    if str(offer.get("id")) != str(request.get("offer_id")):
        raise Invalid("Create offer_id must match the inspected offer")
    if offer.get("gpu_name") != "RTX 3090" or offer.get("num_gpus") != 1:
        raise Invalid("Expected one exact RTX 3090")
    if offer.get("verification") != "verified":
        raise Invalid("Offer is not verified")
    prices = [number(offer[k], k) for k in ("discounted_total_per_hour", "dph_total", "dph_total_adj")
              if offer.get(k) is not None]
    if not prices:
        raise Invalid("Offer has no allocated-storage hourly price")
    hourly = max(prices)  # conservative if an adjusted price is higher
    if hourly > hourly_limit:
        raise Invalid("Offer exceeds hourly ceiling")
    disk = number(request.get("create_request", {}).get("disk"), "explicit disk", 1)
    if disk != number(offer_document.get("allocated_storage_gb"), "allocated_storage_gb", 1):
        raise Invalid("Offer pricing must use the same explicit disk allocation")
    if number(offer.get("disk_space"), "disk_space") < disk:
        raise Invalid("Offer cannot supply the requested disk")
    transfer = 0.0
    for direction in ("down", "up"):
        price = number(offer.get(f"inet_{direction}_cost"), f"inet_{direction}_cost")
        limit = number(budget.get(f"max_inet_{direction}_usd_per_gb"), f"max_inet_{direction}_usd_per_gb")
        if price > limit:
            raise Invalid(f"Offer exceeds {direction} transfer ceiling")
        transfer += price * number(budget.get(f"reserved_inet_{direction}_gb"), f"reserved_inet_{direction}_gb")
    # dph_total is expected to include allocated disk. Add storage again as a
    # conservative reserve; this avoids underbudgeting across provider schemas.
    storage = number(offer.get("storage_cost"), "storage_cost") * disk * hours / (24 * 30)
    margin = number(budget.get("additional_reserve_usd"), "additional_reserve_usd")
    reserved = hourly * hours + transfer + storage + margin
    if reserved > total:
        raise Invalid("Worst-case reservation exceeds total experiment budget")
    return {"total_usd": total, "reserved_usd": reserved, "hourly_usd": hourly,
            "planned_lifetime_hours": hours, "transfer_reserve_usd": transfer,
            "storage_reserve_usd": storage, "additional_reserve_usd": margin,
            "approval_reference": budget["approval_reference"],
            "lifetime_enforcer_reference": budget["lifetime_enforcer_reference"],
            "lifetime_enforcement": "external; observer does not terminate instances"}


def create_once(run_dir, request, budget, offer_document, budget_ledger, token, api_fn=api):
    path = Path(run_dir) / "startup-ledger.json"
    if path.resolve() == Path(budget_ledger).resolve():
        raise Invalid("Budget ledger and startup ledger must be separate files")
    with locked(path.with_suffix(".lock")), locked(Path(budget_ledger).with_suffix(".lock")):
        ledger = load_json(path)
        if ledger["state"] != "initialized":
            raise Invalid("Creation already attempted or instance attached: reconcile; NEVER retry POST")
        if request.get("count") != 1 or not request.get("offer_id"):
            raise Invalid("Exactly count=1 and a preflighted explicit offer_id are required")
        if request.get("create_request", {}).get("label") != ledger["label"]:
            raise Invalid("Creation label must equal the ledger's unique label")
        if not token:
            raise Invalid("Admin token environment variable is empty")
        reservation = validate_budget(budget, offer_document, request)
        book_path = Path(budget_ledger)
        book = load_json(book_path) if book_path.exists() else {
            "schema": "musetalk_experiment_budget_v1", "total_usd": reservation["total_usd"], "reservations": {}}
        if number(book.get("total_usd"), "budget ledger total") != reservation["total_usd"]:
            raise Invalid("Shared budget ledger total does not match authorization")
        reservations = book["reservations"]
        if ledger["label"] in reservations:
            raise Invalid("Label already reserves budget; reconcile the original attempt")
        committed = sum(number(v["reserved_usd"], "existing reservation") for v in reservations.values())
        if committed + reservation["reserved_usd"] > reservation["total_usd"]:
            raise Invalid("Shared experiment budget cannot cover another reservation")
        reservations[ledger["label"]] = reservation
        atomic_json(book_path, book)
        ledger.update(state="submitting", request_sha256=digest(request),
                      request_keys=sorted(request), create_override_keys=sorted(request["create_request"]),
                      offer_id=str(request["offer_id"]), budget=reservation,
                      budget_ledger=str(book_path.resolve()))
        add_event(ledger, "request_started", source="ec2_create_client")
        atomic_json(path, ledger)
        try:
            response = api_fn(ledger["control_plane_url"], "/api/runtime/workers/musetalk/create",
                              token, method="POST", payload=request)
            actions = response.get("result", {}).get("actions", [])
            if response.get("success") is not True or len(actions) != 1:
                raise Invalid("Creation response did not prove exactly one created instance")
            action = actions[0]
            if (action.get("action") != "create" or action.get("label") != ledger["label"]
                    or not str(action.get("instance_id", "")).isdigit()):
                raise Invalid("Creation response identity mismatch")
            ledger.update(state="created", instance_id=str(action["instance_id"]))
            add_event(ledger, "provider_accepted", source="ec2_create_response")
        except Exception as exc:
            ledger["state"] = "ambiguous"
            # No arbitrary exception text, response body, or request secrets.
            ledger["errors"].append({"at_utc": utc_now(), "stage": "create", "type": type(exc).__name__})
            atomic_json(path, ledger)
            raise Invalid("Create outcome ambiguous. Budget remains reserved; reconcile by label, never repost") from None
        atomic_json(path, ledger)
        return ledger


def reconcile(ledger, inventory):
    rows = inventory.get("instances")
    if not isinstance(rows, list):
        raise Invalid("Reconciliation requires a fresh provider instances list, not a cached EC2 list")
    observed = utc_parse(inventory.get("observed_at_utc"))
    age = (dt.datetime.now(dt.timezone.utc) - observed).total_seconds()
    if age < -5 or age > 300:
        raise Invalid("Provider reconciliation inventory is stale")
    rows = [r for r in rows if r.get("label") == ledger["label"]]
    if len(rows) != 1:
        raise Invalid(f"Found {len(rows)} matching provider instances; no automatic retry or selection")
    instance = rows[0]
    if not str(instance.get("id", "")).isdigit():
        raise Invalid("Provider instance has no valid id")
    if ledger.get("instance_id") not in (None, str(instance["id"])):
        raise Invalid("Reconciliation conflicts with the already-recorded instance")
    ledger.update(instance_id=str(instance["id"]), state="reconciled")
    if not any(e["name"] == "provider_accepted" for e in ledger["events"]):
        # This is an upper bound, not a fabricated original acceptance time.
        add_event(ledger, "provider_accepted", source="provider_reconciliation_upper_bound",
                  utc=inventory["observed_at_utc"], local=False, uncertainty_seconds=None)
    return ledger


def import_worker_events(ledger, path):
    document = load_json(path)
    if str(document.get("instance_id")) != ledger.get("instance_id"):
        raise Invalid("Worker events belong to a different instance")
    for event in document.get("events", []):
        if event.get("name") not in WORKER_EVENTS or not event.get("evidence_ref"):
            raise Invalid("Worker events require an allowed name and inspectable evidence reference")
        add_event(ledger, event["name"], source="worker_utc", utc=event.get("utc"),
                  evidence_ref=event["evidence_ref"], local=False,
                  uncertainty_seconds=event.get("clock_uncertainty_seconds"))


def validate_media(document, instance_id, evidence_dir):
    if document.get("schema") != "musetalk_startup_media_v1":
        raise Invalid("Unknown media evidence schema")
    if str(document.get("instance_id")) != str(instance_id):
        raise Invalid("Media evidence belongs to a different instance")
    if document.get("network_path") != "ec2_routed_webrtc" or document.get("client_kind") != "real_browser":
        raise Invalid("Deployment readiness requires a real browser through EC2, not a local health probe")
    for field in ("audio_response_verified", "avatar_verified", "backend_verified", "no_deferred_build_or_download"):
        if document.get(field) is not True:
            raise Invalid(f"Media proof is missing {field}")
    if not re.fullmatch(r"[a-f0-9]{64}", str(document.get("test_audio_sha256", ""))):
        raise Invalid("Test audio identity is missing")
    first = utc_parse(document.get("first_usable_frame_utc"))
    validated = utc_parse(document.get("validated_at_utc"))
    if validated <= first:
        raise Invalid("Readiness validation must finish after the first usable frame")
    duration = number(document.get("measurement_seconds"), "measurement_seconds", 10)
    if (validated - first).total_seconds() + .001 < duration:
        raise Invalid("Media duration exceeds its recorded UTC interval")
    streams = document.get("streams")
    if not isinstance(streams, list) or not streams:
        raise Invalid("Missing per-stream cadence/freshness results")
    for row in streams:
        minimums = {"minimum_anchored_1s_decoded_fps": 20, "speaking_fresh_fraction": .995,
                    "content_fps": 18, "decoded_valid_frames": 1}
        maximums = {"maximum_held_run": 2, "maximum_gap_ms": 120,
                    "gaps_over_100ms": duration / 600, "send_cadence_caused_gaps": 0}
        for key, limit in minimums.items():
            number(row.get(key), key, limit)
        if number(row.get("speaking_fresh_fraction"), "speaking_fresh_fraction") > 1:
            raise Invalid("Fresh-frame fraction exceeds one")
        for key, limit in maximums.items():
            if number(row.get(key), key) > limit:
                raise Invalid(f"Media gate failed: {key}")
        if not row.get("negotiated_codec"):
            raise Invalid("Negotiated codec is missing")
    artifacts = document.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise Invalid("Media assertions need hashed raw trace/video artifacts")
    base = Path(evidence_dir).resolve()
    for artifact in artifacts:
        target = (base / artifact.get("path", "")).resolve()
        if not target.is_relative_to(base) or not target.is_file():
            raise Invalid("Media artifact missing or outside evidence directory")
        h = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != artifact.get("sha256"):
            raise Invalid("Media artifact SHA-256 mismatch")
    return True


def import_media(ledger, path):
    document = load_json(path)
    validate_media(document, ledger.get("instance_id"), Path(path).parent)
    add_event(ledger, "first_usable_frame", source="verified_browser_media", local=False,
              utc=document["first_usable_frame_utc"], evidence_ref=str(Path(path).resolve()),
              uncertainty_seconds=document.get("clock_uncertainty_seconds"))
    add_event(ledger, "steady_call_ready", source="verified_browser_media", local=False,
              utc=document["validated_at_utc"], evidence_ref=str(Path(path).resolve()),
              uncertainty_seconds=document.get("clock_uncertainty_seconds"))
    ledger.update(media_verified=True, media_report_sha256=digest(document))


def observe_once(ledger, token, poll_seconds, api_fn=api):
    if not ledger.get("instance_id"):
        raise Invalid("Observe requires a created or reconciled instance")
    response = api_fn(ledger["control_plane_url"], "/api/runtime/workers/musetalk", token)
    if response.get("success") is not True:
        raise Invalid("Worker inventory request was not successful")
    matches = [w for w in response.get("workers", [])
               if str(w.get("instance_id")) == ledger["instance_id"]]
    if len(matches) > 1:
        raise Invalid("Duplicate worker registrations for the instance")
    observation = {"utc": utc_now(), "registered_count": len(matches), "poll_interval_seconds": poll_seconds}
    if matches:
        worker = matches[0]
        observation.update(worker_id=worker.get("worker_id"), status=worker.get("status"),
                           capacity=worker.get("capacity"), active_count=worker.get("active_count"))
        add_event(ledger, "registered", source="ec2_worker_inventory", uncertainty_seconds=poll_seconds)
        servers = response.get("pool_status", {}).get("servers", [])
        for server in servers:
            same = server.get("worker_id") == worker.get("worker_id") and bool(worker.get("worker_id"))
            if same and server.get("assignable") is True and server.get("status") == "healthy":
                add_event(ledger, "routable", source="ec2_assignable_pool_row", uncertainty_seconds=poll_seconds)
    ledger["observations"].append(observation)


def report(ledger):
    indexed = {e["name"]: e for e in ledger["events"]}
    issues = list(ledger.get("invalid_evidence", []))
    for before, after in (("request_started", "provider_accepted"), ("image_pull_started", "image_pull_finished"),
                          ("request_started", "first_usable_frame"), ("first_usable_frame", "steady_call_ready")):
        if before in indexed and after in indexed and utc_parse(indexed[after]["utc"]) < utc_parse(indexed[before]["utc"]):
            issues.append(f"Timestamp order invalid: {after} before {before}")
    start = indexed.get("request_started")
    durations = {}
    for name, event in indexed.items():
        if start:
            local = (event.get("clock_domain") == start.get("clock_domain") and event.get("clock_domain")
                     and "monotonic_ns" in event and "monotonic_ns" in start)
            seconds = ((event["monotonic_ns"] - start["monotonic_ns"]) / 1e9 if local else
                       (utc_parse(event["utc"]) - utc_parse(start["utc"])).total_seconds())
            if seconds < 0:
                issues.append(f"{name} precedes the request")
            durations[name] = {"seconds_from_request": seconds,
                               "method": "same_process_monotonic" if local else "UTC_correlation",
                               "clock_uncertainty_seconds": event.get("uncertainty_seconds")}
    missing = sorted(REQUIRED_READY - indexed.keys())
    if not ledger.get("media_verified"):
        missing.append("validated_browser_media_evidence")
    verdict = "INVALID" if issues else "INCOMPLETE" if missing else "PASS"
    sla = ledger.get("startup_sla_seconds")
    if verdict == "PASS" and sla is not None and durations["steady_call_ready"]["seconds_from_request"] > sla:
        verdict = "FAIL"
        issues.append("Steady-call readiness exceeded the frozen startup SLA")
    return {"schema": SCHEMA, "verdict": verdict, "instance_id": ledger.get("instance_id"),
            "label": ledger["label"], "cache_state": ledger["cache_state"], "durations": durations,
            "missing_readiness_evidence": missing, "issues": issues,
            "unavailable_stages": [name for name in EVENTS if name not in indexed],
            "interval_policy": "Parallel intervals are not summed; missing stages remain unavailable",
            "health_alone_is_readiness": False, "startup_sla_seconds": sla}


def write_report(run_dir, ledger, result):
    atomic_json(Path(run_dir) / "startup-report.json", result)
    indexed = {e["name"]: e for e in ledger["events"]}
    lines = ["# Startup timeline", "", f"Verdict: **{result['verdict']}**. Instance: `{ledger.get('instance_id')}`.",
             f"Cache state: `{ledger['cache_state']}`. HTTP health alone is not readiness.", "",
             "| Event | UTC | Seconds from request | Clock / observation uncertainty |",
             "| --- | --- | ---: | --- |"]
    for name in EVENTS:
        event = indexed.get(name)
        duration = result["durations"].get(name)
        if event is None:
            lines.append(f"| {name} | unavailable | — | unavailable; not inferred |")
        else:
            seconds = f"{duration['seconds_from_request']:.6f}" if duration else "unavailable"
            method = duration["method"] if duration else "UTC observation only"
            uncertainty = event.get("uncertainty_seconds")
            note = "clock uncertainty unknown" if uncertainty is None else f"uncertainty ≤ {uncertainty} s"
            lines.append(f"| {name} | {event['utc']} | {seconds} | {method}; {note} |")
    lines.extend(["", "Parallel intervals are not summed. First usable frame and validation completion are distinct.",
                  "Polled event timestamps are observation upper bounds, not exact underlying transition times."])
    if result["missing_readiness_evidence"]:
        lines.extend(["", "Missing readiness evidence: " + ", ".join(result["missing_readiness_evidence"]) + "."])
    if result["issues"]:
        lines.extend(["", "Issues: " + "; ".join(result["issues"]) + "."])
    atomic_text(Path(run_dir) / "startup-timeline.md", "\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init")
    init.add_argument("--run-dir", required=True); init.add_argument("--label", required=True)
    init.add_argument("--control-plane-url", required=True)
    init.add_argument("--cache-state", choices=("cold_proven", "provider_cached_proven", "restart", "unknown"), default="unknown")
    init.add_argument("--startup-sla-seconds", type=float)
    init.add_argument("--resource-deadline-utc", help="Optional UTC deadline bound into a fresh ledger; does not install an expiry timer")
    create = sub.add_parser("create", help="One real paid EC2 create request; never auto-retries")
    create.add_argument("--request-json", required=True); create.add_argument("--budget-json", required=True)
    create.add_argument("--offer-json", required=True); create.add_argument("--budget-ledger", required=True)
    rec = sub.add_parser("reconcile")
    rec.add_argument("--provider-inventory-json", required=True)
    obs = sub.add_parser("observe")
    obs.add_argument("--timeout-seconds", type=float, default=900)
    obs.add_argument("--poll-seconds", type=float, default=5)
    obs.add_argument("--worker-events-json"); obs.add_argument("--media-evidence-json")
    rep = sub.add_parser("report")
    for child in (create, rec, obs, rep):
        child.add_argument("--run-dir", required=True)
    for child in (create, obs):
        child.add_argument("--admin-token-env", default="RUNTIME_CONFIG_TOKEN")
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            if args.startup_sla_seconds is not None:
                number(args.startup_sla_seconds, "startup SLA", .001)
            initialize(args.run_dir, args.label, args.control_plane_url, args.cache_state,
                       args.startup_sla_seconds, args.resource_deadline_utc)
            return 0
        path = Path(args.run_dir) / "startup-ledger.json"
        if args.command == "create":
            ledger = create_once(args.run_dir, load_json(args.request_json), load_json(args.budget_json),
                                 load_json(args.offer_json), args.budget_ledger, os.getenv(args.admin_token_env, ""))
            print(json.dumps({"state": ledger["state"], "instance_id": ledger["instance_id"]}))
            return 0
        if args.command == "reconcile":
            with locked(path.with_suffix(".lock")):
                ledger = reconcile(load_json(path), load_json(args.provider_inventory_json))
                atomic_json(path, ledger)
            print(json.dumps({"state": ledger["state"], "instance_id": ledger["instance_id"]}))
            return 0
        if args.command == "observe":
            poll = number(args.poll_seconds, "poll_seconds", 1)
            timeout = number(args.timeout_seconds, "timeout_seconds", .001)
            deadline = time.monotonic() + timeout
            while True:
                with locked(path.with_suffix(".lock")):
                    ledger = load_json(path)
                    try:
                        observe_once(ledger, os.getenv(args.admin_token_env, ""), poll)
                    except Exception as exc:
                        ledger["errors"].append({"at_utc": utc_now(), "stage": "observe", "type": type(exc).__name__})
                    for evidence_path, importer in ((args.worker_events_json, import_worker_events),
                                                     (args.media_evidence_json, import_media)):
                        if evidence_path and Path(evidence_path).exists():
                            try:
                                importer(ledger, evidence_path)
                            except Exception as exc:
                                ledger.setdefault("invalid_evidence", []).append(f"Evidence validation failed: {type(exc).__name__}")
                    atomic_json(path, ledger)
                    result = report(ledger)
                    write_report(args.run_dir, ledger, result)
                if result["verdict"] in {"PASS", "FAIL", "INVALID"} or time.monotonic() >= deadline:
                    print(json.dumps(result))
                    return 0 if result["verdict"] == "PASS" else 1
                time.sleep(min(poll, 30, max(0, deadline - time.monotonic())))
        ledger = load_json(path)
        result = report(ledger)
        write_report(args.run_dir, ledger, result)
        print(json.dumps(result))
        return 0 if result["verdict"] == "PASS" else 1
    except (Invalid, OSError, ValueError, KeyError) as exc:
        # Invalid messages are authored by this program; arbitrary exception text
        # can expose request/env data and is intentionally not displayed.
        print(str(exc) if isinstance(exc, Invalid) else f"Invalid input: {type(exc).__name__}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
