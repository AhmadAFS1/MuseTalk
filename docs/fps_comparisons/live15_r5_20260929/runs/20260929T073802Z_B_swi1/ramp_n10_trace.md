# Live trace report: tmp/live15_r5/20260929T073802Z_B_swi1/traces/ramp/n10

```
{
 "level_dir": "tmp/live15_r5/20260929T073802Z_B_swi1/traces/ramp/n10",
 "streams": 10,
 "streams_pass": 3,
 "all_pass": false,
 "min_anchored_1s": 13,
 "worst_gap_ms": 412.7,
 "gaps_over_100ms_total": 16,
 "min_server_fresh_fraction": 0.99725,
 "max_server_held_run": 1,
 "min_pts_join": 1.0,
 "client_loop_spikes": 10,
 "client_loop_spike_max_ms": 77.9,
 "server_cadence_min_anchored_1s": 13,
 "server_send_interval_max_ms": 385.0,
 "gap_causes": {
  "client_loop": 5,
  "server_send": 9,
  "server_late": 2
 }
}
```

| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | chinese_bob_pink_bedroom_talking_3373c10448 | 304.6 | 19.977 | 13 | 14 | 402.0 | 4 {'client_loop': 2, 'server_send': 2} | 0.99788 | 1/1 | 13 | 54.7/63.0 | 1.0 | FAIL | FAIL | FAIL |
| 1 | indian_realtime_talking_20f9845543 | 298.8 | 19.977 | 13 | 16 | 412.7 | 3 {'client_loop': 1, 'server_late': 1, 'server_send': 1} | 0.99737 | 1/1 | 13 | 54.0/62.5 | 1.0 | FAIL | FAIL | FAIL |
| 2 | japanese_realtime_talking_7d94520b7f | 300.8 | 19.978 | 14 | 15 | 391.4 | 3 {'client_loop': 1, 'server_send': 2} | 0.99743 | 1/1 | 14 | 54.7/62.9 | 1.0 | FAIL | FAIL | FAIL |
| 3 | latina_guided_20260925_talking_84c5bc80b8 | 314.6 | 19.998 | 19 | 19 | 128.7 | 3 {'server_send': 1, 'client_loop': 1, 'server_late': 1} | 0.99812 | 1/1 | 19 | 54.3/61.0 | 1.0 | FAIL | PASS | FAIL |
| 4 | codex_smoke | 299.7 | 20.004 | 19 | 20 | 120.5 | 1 {'server_send': 1} | 0.99725 | 1/1 | 19 | 53.7/61.7 | 1.0 | FAIL | PASS | FAIL |
| 5 | japanese_baddie_ltx23_talking_v1 | 299.4 | 19.999 | 19 | 19 | 117.7 | 1 {'server_send': 1} | 0.99739 | 1/1 | 19 | 54.7/62.8 | 1.0 | FAIL | PASS | FAIL |
| 6 | latina_baddie_ltx23_talking_v1 | 301.8 | 19.997 | 20 | 19 | 98.0 | 0  | 0.99763 | 1/1 | 19 | 54.4/62.8 | 1.0 | PASS | PASS | PASS |
| 7 | ltx23_talking_endpoint_override_pilot_20260922 | 293.8 | 19.999 | 20 | 19 | 117.8 | 1 {'server_send': 1} | 0.9978 | 1/1 | 19 | 54.8/62.9 | 1.0 | PASS | PASS | FAIL |
| 8 | japanese_relay64_talking_preview_20260925 | 309.8 | 19.997 | 20 | 19 | 87.3 | 0  | 0.99768 | 1/1 | 19 | 54.6/62.1 | 1.0 | PASS | PASS | PASS |
| 9 | chinese_bob_pink_bedroom_idle_d4b06da317 | 302.8 | 19.999 | 20 | 19 | 77.3 | 0  | 0.99762 | 1/1 | 19 | 54.5/62.1 | 1.0 | PASS | PASS | PASS |
