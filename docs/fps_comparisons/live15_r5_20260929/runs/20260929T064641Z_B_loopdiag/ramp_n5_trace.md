# Live trace report: tmp/live15_r5/20260929T064641Z_B_loopdiag/traces/ramp/n05

```
{
 "level_dir": "tmp/live15_r5/20260929T064641Z_B_loopdiag/traces/ramp/n05",
 "streams": 5,
 "streams_pass": 0,
 "all_pass": false,
 "min_anchored_1s": 19,
 "worst_gap_ms": 142.0,
 "gaps_over_100ms_total": 59,
 "min_server_fresh_fraction": 0.99721,
 "max_server_held_run": 1,
 "min_pts_join": 1.0,
 "client_loop_spikes": 1,
 "client_loop_spike_max_ms": 23.9,
 "server_cadence_min_anchored_1s": 19,
 "server_send_interval_max_ms": 136.5,
 "gap_causes": {
  "server_send": 35,
  "transport": 15,
  "server_late": 9
 }
}
```

| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | chinese_bob_pink_bedroom_talking_3373c10448 | 321.6 | 19.998 | 19 | 19 | 142.0 | 15 {'server_send': 7, 'transport': 4, 'server_late': 4} | 0.99721 | 1/1 | 19 | 54.2/88.1 | 1.0 | FAIL | PASS | FAIL |
| 1 | indian_realtime_talking_20f9845543 | 314.8 | 19.999 | 19 | 19 | 118.6 | 11 {'server_late': 1, 'server_send': 6, 'transport': 4} | 0.99738 | 1/1 | 18 | 53.9/84.3 | 1.0 | FAIL | PASS | FAIL |
| 2 | japanese_realtime_talking_7d94520b7f | 309.8 | 19.998 | 19 | 19 | 117.5 | 11 {'server_late': 1, 'server_send': 8, 'transport': 2} | 0.99782 | 1/1 | 19 | 54.5/84.4 | 1.0 | FAIL | PASS | FAIL |
| 3 | latina_guided_20260925_talking_84c5bc80b8 | 305.4 | 19.998 | 19 | 19 | 117.5 | 14 {'server_late': 1, 'server_send': 10, 'transport': 3} | 0.99776 | 1/1 | 19 | 54.3/84.6 | 1.0 | FAIL | PASS | FAIL |
| 4 | codex_smoke | 300.4 | 19.996 | 19 | 19 | 115.4 | 8 {'server_late': 2, 'server_send': 4, 'transport': 2} | 0.99783 | 1/1 | 19 | 53.4/84.4 | 1.0 | FAIL | PASS | FAIL |
