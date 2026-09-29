# Live trace report: tmp/live15_r5/20260929T062327Z_B_gclog/traces/ramp/n05

```
{
 "level_dir": "tmp/live15_r5/20260929T062327Z_B_gclog/traces/ramp/n05",
 "streams": 5,
 "streams_pass": 0,
 "all_pass": false,
 "min_anchored_1s": 18,
 "worst_gap_ms": 200.6,
 "gaps_over_100ms_total": 77,
 "min_server_fresh_fraction": 0.99721,
 "max_server_held_run": 1,
 "min_pts_join": 1.0,
 "client_loop_spikes": 1,
 "client_loop_spike_max_ms": 29.7,
 "server_cadence_min_anchored_1s": 18,
 "server_send_interval_max_ms": 194.2,
 "gap_causes": {
  "server_send": 57,
  "transport": 9,
  "server_late": 11
 }
}
```

| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | chinese_bob_pink_bedroom_talking_3373c10448 | 322.7 | 19.993 | 18 | 17 | 177.7 | 19 {'server_send': 14, 'transport': 4, 'server_late': 1} | 0.99721 | 1/1 | 18 | 54.1/82.1 | 1.0 | FAIL | PASS | FAIL |
| 1 | indian_realtime_talking_20f9845543 | 315.9 | 20.003 | 18 | 19 | 193.8 | 14 {'server_send': 11, 'server_late': 2, 'transport': 1} | 0.99738 | 1/1 | 18 | 53.8/81.3 | 1.0 | FAIL | PASS | FAIL |
| 2 | japanese_realtime_talking_7d94520b7f | 310.9 | 19.995 | 18 | 17 | 185.9 | 15 {'server_send': 10, 'transport': 2, 'server_late': 3} | 0.99782 | 1/1 | 18 | 54.2/80.0 | 1.0 | FAIL | PASS | FAIL |
| 3 | latina_guided_20260925_talking_84c5bc80b8 | 305.9 | 19.999 | 18 | 17 | 200.6 | 14 {'server_send': 11, 'server_late': 2, 'transport': 1} | 0.99775 | 1/1 | 18 | 54.2/82.4 | 1.0 | FAIL | PASS | FAIL |
| 4 | codex_smoke | 300.8 | 19.998 | 18 | 17 | 194.5 | 15 {'server_late': 3, 'server_send': 11, 'transport': 1} | 0.99783 | 1/1 | 18 | 53.3/77.1 | 1.0 | FAIL | PASS | FAIL |
