# Live trace report: tmp/live15_r5/20260929T081939Z_B_garbage/traces/ramp/n05

```
{
 "level_dir": "tmp/live15_r5/20260929T081939Z_B_garbage/traces/ramp/n05",
 "streams": 5,
 "streams_pass": 0,
 "all_pass": false,
 "min_anchored_1s": 19,
 "worst_gap_ms": 146.4,
 "gaps_over_100ms_total": 14,
 "min_server_fresh_fraction": 0.99721,
 "max_server_held_run": 1,
 "min_pts_join": 1.0,
 "client_loop_spikes": 5,
 "client_loop_spike_max_ms": 62.8,
 "server_cadence_min_anchored_1s": 19,
 "server_send_interval_max_ms": 125.1,
 "gap_causes": {
  "server_send": 6,
  "server_late": 4,
  "transport": 4
 }
}
```

| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | chinese_bob_pink_bedroom_talking_3373c10448 | 300.6 | 19.999 | 19 | 19 | 135.2 | 1 {'server_send': 1} | 0.99721 | 1/1 | 19 | 54.2/57.7 | 1.0 | FAIL | PASS | FAIL |
| 1 | indian_realtime_talking_20f9845543 | 294.8 | 19.999 | 19 | 19 | 126.0 | 3 {'server_late': 1, 'server_send': 1, 'transport': 1} | 0.99721 | 1/1 | 19 | 53.9/57.7 | 1.0 | FAIL | PASS | FAIL |
| 2 | japanese_realtime_talking_7d94520b7f | 300.8 | 19.998 | 19 | 19 | 126.5 | 2 {'server_late': 1, 'server_send': 1} | 0.99782 | 1/1 | 19 | 54.3/57.8 | 1.0 | FAIL | PASS | FAIL |
| 3 | latina_guided_20260925_talking_84c5bc80b8 | 299.8 | 19.999 | 19 | 20 | 129.1 | 4 {'transport': 3, 'server_late': 1} | 0.99784 | 1/1 | 19 | 54.3/58.1 | 1.0 | FAIL | PASS | FAIL |
| 4 | codex_smoke | 299.8 | 19.999 | 19 | 19 | 146.4 | 4 {'server_send': 3, 'server_late': 1} | 0.99783 | 1/1 | 19 | 53.2/56.8 | 1.0 | FAIL | PASS | FAIL |
