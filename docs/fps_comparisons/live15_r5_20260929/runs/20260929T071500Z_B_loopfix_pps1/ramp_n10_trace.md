# Live trace report: tmp/live15_r5/20260929T071500Z_B_loopfix_pps1/traces/ramp/n10

```
{
 "level_dir": "tmp/live15_r5/20260929T071500Z_B_loopfix_pps1/traces/ramp/n10",
 "streams": 10,
 "streams_pass": 6,
 "all_pass": false,
 "min_anchored_1s": 19,
 "worst_gap_ms": 108.7,
 "gaps_over_100ms_total": 4,
 "min_server_fresh_fraction": 0.99725,
 "max_server_held_run": 1,
 "min_pts_join": 1.0,
 "client_loop_spikes": 10,
 "client_loop_spike_max_ms": 78.7,
 "server_cadence_min_anchored_1s": 20,
 "server_send_interval_max_ms": 92.1,
 "gap_causes": {
  "server_late": 3,
  "transport": 1
 }
}
```

| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | chinese_bob_pink_bedroom_talking_3373c10448 | 301.4 | 19.999 | 20 | 20 | 96.7 | 0  | 0.99787 | 1/1 | 19 | 54.4/62.1 | 1.0 | PASS | PASS | PASS |
| 1 | indian_realtime_talking_20f9845543 | 296.8 | 19.999 | 19 | 19 | 106.2 | 1 {'server_late': 1} | 0.99737 | 1/1 | 19 | 54.2/62.1 | 1.0 | FAIL | PASS | FAIL |
| 2 | japanese_realtime_talking_7d94520b7f | 302.8 | 19.999 | 19 | 19 | 104.8 | 1 {'server_late': 1} | 0.99743 | 1/1 | 19 | 54.7/62.3 | 1.0 | FAIL | PASS | FAIL |
| 3 | latina_guided_20260925_talking_84c5bc80b8 | 315.9 | 20.0 | 19 | 19 | 108.7 | 1 {'server_late': 1} | 0.99812 | 1/1 | 19 | 54.6/62.3 | 1.0 | FAIL | PASS | FAIL |
| 4 | codex_smoke | 300.8 | 19.999 | 19 | 19 | 100.6 | 1 {'transport': 1} | 0.99725 | 1/1 | 19 | 53.6/61.6 | 1.0 | FAIL | PASS | FAIL |
| 5 | japanese_baddie_ltx23_talking_v1 | 297.8 | 19.999 | 20 | 19 | 97.0 | 0  | 0.99739 | 1/1 | 19 | 54.3/62.7 | 1.0 | PASS | PASS | PASS |
| 6 | latina_baddie_ltx23_talking_v1 | 303.8 | 20.0 | 20 | 19 | 95.2 | 0  | 0.99763 | 1/1 | 19 | 54.4/62.0 | 1.0 | PASS | PASS | PASS |
| 7 | ltx23_talking_endpoint_override_pilot_20260922 | 294.8 | 19.999 | 20 | 19 | 96.9 | 0  | 0.9978 | 1/1 | 19 | 54.5/62.6 | 1.0 | PASS | PASS | PASS |
| 8 | japanese_relay64_talking_preview_20260925 | 311.8 | 19.998 | 20 | 19 | 87.2 | 0  | 0.99768 | 1/1 | 19 | 54.6/62.4 | 1.0 | PASS | PASS | PASS |
| 9 | chinese_bob_pink_bedroom_idle_d4b06da317 | 292.8 | 19.999 | 20 | 19 | 84.4 | 0  | 0.99753 | 1/1 | 19 | 54.3/61.3 | 1.0 | PASS | PASS | PASS |
