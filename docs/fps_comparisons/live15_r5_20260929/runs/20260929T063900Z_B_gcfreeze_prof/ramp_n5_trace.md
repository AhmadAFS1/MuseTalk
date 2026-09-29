# Live trace report: tmp/live15_r5/20260929T063900Z_B_gcfreeze_prof/traces/ramp/n05

```
{
 "level_dir": "tmp/live15_r5/20260929T063900Z_B_gcfreeze_prof/traces/ramp/n05",
 "streams": 5,
 "streams_pass": 0,
 "all_pass": false,
 "min_anchored_1s": 18,
 "worst_gap_ms": 162.0,
 "gaps_over_100ms_total": 57,
 "min_server_fresh_fraction": 0.99721,
 "max_server_held_run": 1,
 "min_pts_join": 1.0,
 "client_loop_spikes": 1,
 "client_loop_spike_max_ms": 24.2,
 "server_cadence_min_anchored_1s": 19,
 "server_send_interval_max_ms": 128.2,
 "gap_causes": {
  "server_send": 36,
  "server_late": 15,
  "transport": 6
 }
}
```

| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | chinese_bob_pink_bedroom_talking_3373c10448 | 321.6 | 19.999 | 19 | 19 | 134.7 | 12 {'server_send': 8, 'server_late': 3, 'transport': 1} | 0.99721 | 1/1 | 19 | 54.0/78.5 | 1.0 | FAIL | PASS | FAIL |
| 1 | indian_realtime_talking_20f9845543 | 315.9 | 20.005 | 18 | 19 | 160.7 | 11 {'server_send': 7, 'server_late': 4} | 0.99738 | 1/1 | 18 | 53.6/85.2 | 1.0 | FAIL | PASS | FAIL |
| 2 | japanese_realtime_talking_7d94520b7f | 310.8 | 19.995 | 19 | 19 | 137.5 | 13 {'server_late': 3, 'transport': 3, 'server_send': 7} | 0.99782 | 1/1 | 19 | 54.1/81.8 | 1.0 | FAIL | PASS | FAIL |
| 3 | latina_guided_20260925_talking_84c5bc80b8 | 305.9 | 19.999 | 18 | 19 | 162.0 | 11 {'server_send': 7, 'transport': 1, 'server_late': 3} | 0.99775 | 1/1 | 18 | 54.1/83.3 | 1.0 | FAIL | PASS | FAIL |
| 4 | codex_smoke | 300.9 | 19.999 | 18 | 19 | 160.5 | 10 {'server_send': 7, 'transport': 1, 'server_late': 2} | 0.99783 | 1/1 | 18 | 53.2/81.0 | 1.0 | FAIL | PASS | FAIL |
