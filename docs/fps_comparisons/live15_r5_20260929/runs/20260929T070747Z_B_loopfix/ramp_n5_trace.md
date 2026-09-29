# Live trace report: tmp/live15_r5/20260929T070747Z_B_loopfix/traces/ramp/n05

```
{
 "level_dir": "tmp/live15_r5/20260929T070747Z_B_loopfix/traces/ramp/n05",
 "streams": 5,
 "streams_pass": 1,
 "all_pass": false,
 "min_anchored_1s": 19,
 "worst_gap_ms": 131.7,
 "gaps_over_100ms_total": 4,
 "min_server_fresh_fraction": 0.99721,
 "max_server_held_run": 1,
 "min_pts_join": 1.0,
 "client_loop_spikes": 2,
 "client_loop_spike_max_ms": 84.6,
 "server_cadence_min_anchored_1s": 20,
 "server_send_interval_max_ms": 87.8,
 "gap_causes": {
  "client_loop": 4
 }
}
```

| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | chinese_bob_pink_bedroom_talking_3373c10448 | 320.6 | 19.999 | 19 | 19 | 105.3 | 1 {'client_loop': 1} | 0.99721 | 1/1 | 19 | 54.1/57.7 | 1.0 | FAIL | PASS | FAIL |
| 1 | indian_realtime_talking_20f9845543 | 313.9 | 19.999 | 19 | 19 | 128.0 | 1 {'client_loop': 1} | 0.99721 | 1/1 | 19 | 53.9/57.6 | 1.0 | FAIL | PASS | FAIL |
| 2 | japanese_realtime_talking_7d94520b7f | 309.7 | 19.999 | 19 | 19 | 131.2 | 1 {'client_loop': 1} | 0.99782 | 1/1 | 19 | 54.3/57.6 | 1.0 | FAIL | PASS | FAIL |
| 3 | latina_guided_20260925_talking_84c5bc80b8 | 304.7 | 19.999 | 19 | 19 | 131.7 | 1 {'client_loop': 1} | 0.99784 | 1/1 | 19 | 54.2/57.8 | 1.0 | FAIL | PASS | FAIL |
| 4 | codex_smoke | 299.8 | 19.999 | 20 | 19 | 84.8 | 0  | 0.99783 | 1/1 | 19 | 53.3/56.7 | 1.0 | PASS | PASS | PASS |
