# Live trace report: tmp/live15_r5/20260929T062958Z_B_gcfreeze/traces/ramp/n05

```
{
 "level_dir": "tmp/live15_r5/20260929T062958Z_B_gcfreeze/traces/ramp/n05",
 "streams": 5,
 "streams_pass": 0,
 "all_pass": false,
 "min_anchored_1s": 18,
 "worst_gap_ms": 137.3,
 "gaps_over_100ms_total": 40,
 "min_server_fresh_fraction": 0.99721,
 "max_server_held_run": 1,
 "min_pts_join": 1.0,
 "client_loop_spikes": 1,
 "client_loop_spike_max_ms": 45.7,
 "server_cadence_min_anchored_1s": 19,
 "server_send_interval_max_ms": 131.6,
 "gap_causes": {
  "transport": 10,
  "server_late": 7,
  "server_send": 23
 }
}
```

| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | chinese_bob_pink_bedroom_talking_3373c10448 | 321.6 | 19.997 | 18 | 18 | 131.7 | 9 {'transport': 4, 'server_late': 3, 'server_send': 2} | 0.99721 | 1/1 | 18 | 54.0/79.5 | 1.0 | FAIL | PASS | FAIL |
| 1 | indian_realtime_talking_20f9845543 | 315.8 | 19.995 | 19 | 18 | 127.2 | 4 {'server_send': 4} | 0.99721 | 1/1 | 19 | 53.6/77.0 | 1.0 | FAIL | PASS | FAIL |
| 2 | japanese_realtime_talking_7d94520b7f | 310.9 | 19.996 | 19 | 19 | 136.1 | 11 {'server_late': 3, 'transport': 2, 'server_send': 6} | 0.99782 | 1/1 | 19 | 54.2/83.1 | 1.0 | FAIL | PASS | FAIL |
| 3 | latina_guided_20260925_talking_84c5bc80b8 | 305.8 | 19.999 | 19 | 19 | 137.3 | 8 {'server_send': 6, 'transport': 2} | 0.99776 | 1/1 | 19 | 54.2/82.4 | 1.0 | FAIL | PASS | FAIL |
| 4 | codex_smoke | 300.8 | 19.999 | 19 | 19 | 127.7 | 8 {'server_send': 5, 'transport': 2, 'server_late': 1} | 0.99783 | 1/1 | 19 | 53.3/81.2 | 1.0 | FAIL | PASS | FAIL |
