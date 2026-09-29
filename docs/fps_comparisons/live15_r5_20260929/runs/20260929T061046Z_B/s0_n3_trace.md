# Live trace report: tmp/live15_r5/20260929T061046Z_B/traces/s0/n03

```
{
 "level_dir": "tmp/live15_r5/20260929T061046Z_B/traces/s0/n03",
 "streams": 3,
 "streams_pass": 1,
 "all_pass": false,
 "min_anchored_1s": 19,
 "worst_gap_ms": 101.4,
 "gaps_over_100ms_total": 1,
 "min_server_fresh_fraction": 0.99242,
 "max_server_held_run": 1,
 "min_pts_join": 1.0,
 "client_loop_spikes": 1,
 "client_loop_spike_max_ms": 24.9,
 "server_cadence_min_anchored_1s": 19,
 "server_send_interval_max_ms": 100.9,
 "gap_causes": {
  "server_send": 1
 }
}
```

| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | chinese_bob_pink_bedroom_talking_3373c10448 | 13.0 | 20.002 | 20 | 19 | 92.7 | 0  | 0.99242 | 1/1 | 19 | 53.4/57.8 | 1.0 | PASS | FAIL | PASS |
| 1 | indian_realtime_talking_20f9845543 | 13.0 | 19.999 | 19 | 19 | 101.4 | 1 {'server_send': 1} | 1.0 | 0/0 | 19 | 53.1/60.4 | 1.0 | FAIL | PASS | FAIL |
| 2 | japanese_realtime_talking_7d94520b7f | 13.7 | 19.998 | 20 | 19 | 92.4 | 0  | 1.0 | 0/0 | 20 | 53.4/61.6 | 1.0 | PASS | PASS | PASS |
