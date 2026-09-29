# Live trace report: tmp/live15_r5/20260929T061046Z_B/traces/ramp/n05

```
{
 "level_dir": "tmp/live15_r5/20260929T061046Z_B/traces/ramp/n05",
 "streams": 5,
 "streams_pass": 0,
 "all_pass": false,
 "min_anchored_1s": 18,
 "worst_gap_ms": 199.5,
 "gaps_over_100ms_total": 71,
 "min_server_fresh_fraction": 0.99721,
 "max_server_held_run": 1,
 "min_pts_join": 1.0,
 "client_loop_spikes": 1,
 "client_loop_spike_max_ms": 24.4,
 "server_cadence_min_anchored_1s": 18,
 "server_send_interval_max_ms": 195.5,
 "gap_causes": {
  "server_send": 49,
  "transport": 13,
  "server_late": 9
 }
}
```

| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | chinese_bob_pink_bedroom_talking_3373c10448 | 322.5 | 19.999 | 18 | 18 | 190.4 | 18 {'server_send': 11, 'transport': 5, 'server_late': 2} | 0.9973 | 1/1 | 18 | 54.1/86.9 | 1.0 | FAIL | PASS | FAIL |
| 1 | indian_realtime_talking_20f9845543 | 315.9 | 19.991 | 18 | 18 | 194.0 | 14 {'server_send': 11, 'server_late': 1, 'transport': 2} | 0.99721 | 1/1 | 18 | 53.8/80.9 | 1.0 | FAIL | PASS | FAIL |
| 2 | japanese_realtime_talking_7d94520b7f | 310.9 | 20.007 | 18 | 17 | 199.5 | 12 {'server_send': 8, 'transport': 2, 'server_late': 2} | 0.99782 | 1/1 | 18 | 54.2/80.6 | 1.0 | FAIL | PASS | FAIL |
| 3 | latina_guided_20260925_talking_84c5bc80b8 | 306.5 | 19.999 | 18 | 18 | 192.3 | 13 {'server_send': 9, 'transport': 2, 'server_late': 2} | 0.99784 | 1/1 | 18 | 54.2/85.2 | 1.0 | FAIL | PASS | FAIL |
| 4 | codex_smoke | 301.3 | 19.998 | 18 | 18 | 186.3 | 14 {'server_late': 2, 'server_send': 10, 'transport': 2} | 0.99783 | 1/1 | 18 | 53.1/84.4 | 1.0 | FAIL | PASS | FAIL |
