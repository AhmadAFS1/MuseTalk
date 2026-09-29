#!/usr/bin/env python
"""PASS/FAIL for the N=3 loopback smoke (gpu_sequence.sh step S0).

Reads the load_test_webrtc_v2.py level JSON and checks, on the live candidate
server: every turn accepted and completed, no client errors, output_fps 20,
fresh fraction >= 0.99 (N=3 is far below capacity), 0 stall seconds, every
stream cache-backed (WEBRTC_IDLE_FRAME_CACHE), non-blocking handoff active with
WEBRTC_HANDOFF_VERIFY counters at zero mismatches, the scheduler callback no
longer blocking (< 2 ms per batch), and the expected encoder in use.

  check_smoke.py LEVEL_JSON [--expect-vp8 native|pyav] [--out result.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("level_json")
    ap.add_argument("--expect-vp8", default="native")
    ap.add_argument("--max-callback-ms", type=float, default=2.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    s = json.loads(Path(args.level_json).read_text())
    streams = s["streams"]
    handoffs = [st.get("live_handoff_last") or {} for st in streams]
    checks = {
        "turns_all_ok": all(st["turns_ok"] == st["turns_posted"] and st["turns_ok"] > 0 for st in streams),
        "no_client_errors": not s["errors"],
        "output_fps_20": s["checks"]["output_fps_20"],
        "fresh_fraction_ge_0.99": (s["aggregate"]["fresh_fraction_min"] or 0) >= 0.99,
        "no_stalls": (s["aggregate"]["stall_seconds_max"] or 0) == 0,
        "idle_cache_backed": all(st.get("idle_cache_backed") is True for st in streams),
        "handoff_nonblocking": s["server"].get("live_handoff_mode") == "nonblocking"
                               and all(h.get("mode") == "nonblocking" for h in handoffs),
        "handoff_verify_zero_mismatch": all(
            h.get("verify_frames", 0) > 0 and h.get("verify_content_mismatch") == 0
            and h.get("verify_order_errors") == 0 and h.get("verify_i420_mismatch", 0) == 0
            for h in handoffs),
        "callback_ms_per_batch": (s["aggregate"].get("callback_ms_per_batch") or 99) <= args.max_callback_ms,
        "vp8_encoder": (s["server"].get("vp8") or {}).get("encoder") == args.expect_vp8,
        "valid_run": all(s["validity"].values()),
    }
    passed = all(checks.values())
    detail = {
        "fresh_fraction_min": s["aggregate"]["fresh_fraction_min"],
        "first_frame_p95_s": s["aggregate"].get("first_frame_p95_s"),
        "callback_ms_per_batch": s["aggregate"].get("callback_ms_per_batch"),
        "generated_fps": s["aggregate"].get("generated_fps_server"),
        "verify": [{k: h.get(k) for k in ("verify_frames", "verify_content_mismatch", "verify_order_errors",
                                          "verify_i420_checked", "verify_i420_mismatch", "max_pending_frames_seen",
                                          "overflow_waits", "max_handoff_latency_ms")} for h in handoffs],
        "client_lag_p99_ms": s["client"]["loop_lag_p99_ms_max"],
        "server_rss_max_mb": s["server"].get("server_rss_max_mb"),
        "server_vram_max_mb": s["server"].get("server_vram_max_mb"),
    }
    out = {"passed": passed, "checks": checks, "detail": detail, "level_json": args.level_json}
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"{'PASS' if passed else 'FAIL'} smoke N={s['level']}: fresh_min={detail['fresh_fraction_min']} "
          f"first_frame_p95={detail['first_frame_p95_s']}s callback={detail['callback_ms_per_batch']}ms/batch "
          f"verify_frames={[v['verify_frames'] for v in detail['verify']]} "
          f"failed={[k for k, v in checks.items() if not v]}")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
