# Live trace report: tmp/live15_r5/20260929T072948Z_B_lagdiag/traces/ramp/n10

```
{
 "level_dir": "tmp/live15_r5/20260929T072948Z_B_lagdiag/traces/ramp/n10",
 "streams": 10,
 "streams_pass": 0,
 "all_pass": false,
 "min_anchored_1s": 18,
 "worst_gap_ms": 138.0,
 "gaps_over_100ms_total": 84,
 "min_server_fresh_fraction": 0.99716,
 "max_server_held_run": 1,
 "min_pts_join": 1.0,
 "client_loop_spikes": 10,
 "client_loop_spike_max_ms": 78.4,
 "server_cadence_min_anchored_1s": 19,
 "server_send_interval_max_ms": 133.7,
 "gap_causes": {
  "server_late": 51,
  "server_send": 22,
  "transport": 11
 }
}
```

| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | chinese_bob_pink_bedroom_talking_3373c10448 | 304.6 | 19.999 | 19 | 19 | 136.7 | 11 {'server_late': 7, 'server_send': 4} | 0.99788 | 1/1 | 18 | 54.5/64.4 | 1.0 | FAIL | PASS | FAIL |
| 1 | indian_realtime_talking_20f9845543 | 297.7 | 19.997 | 18 | 19 | 121.0 | 13 {'server_late': 7, 'server_send': 4, 'transport': 2} | 0.99737 | 1/1 | 18 | 54.3/65.3 | 1.0 | FAIL | PASS | FAIL |
| 2 | japanese_realtime_talking_7d94520b7f | 303.7 | 19.997 | 19 | 19 | 138.0 | 9 {'server_late': 5, 'server_send': 4} | 0.99743 | 1/1 | 19 | 54.6/63.8 | 1.0 | FAIL | PASS | FAIL |
| 3 | latina_guided_20260925_talking_84c5bc80b8 | 313.8 | 20.0 | 19 | 19 | 132.9 | 12 {'server_late': 7, 'server_send': 4, 'transport': 1} | 0.99812 | 1/1 | 19 | 54.4/63.4 | 1.0 | FAIL | PASS | FAIL |
| 4 | codex_smoke | 291.7 | 20.0 | 19 | 19 | 118.0 | 5 {'server_late': 5} | 0.99716 | 1/1 | 18 | 53.8/63.1 | 1.0 | FAIL | PASS | FAIL |
| 5 | japanese_baddie_ltx23_talking_v1 | 301.8 | 19.999 | 19 | 19 | 118.1 | 9 {'server_late': 6, 'transport': 3} | 0.9974 | 1/1 | 18 | 54.5/64.0 | 1.0 | FAIL | PASS | FAIL |
| 6 | latina_baddie_ltx23_talking_v1 | 305.8 | 19.999 | 19 | 19 | 114.2 | 6 {'server_late': 3, 'server_send': 1, 'transport': 2} | 0.99763 | 1/1 | 19 | 54.4/62.8 | 1.0 | FAIL | PASS | FAIL |
| 7 | ltx23_talking_endpoint_override_pilot_20260922 | 293.8 | 20.0 | 19 | 19 | 119.4 | 6 {'server_late': 4, 'transport': 2} | 0.9978 | 1/1 | 19 | 54.7/63.9 | 1.0 | FAIL | PASS | FAIL |
| 8 | japanese_relay64_talking_preview_20260925 | 311.8 | 19.997 | 19 | 19 | 115.9 | 7 {'server_send': 4, 'server_late': 3} | 0.99768 | 1/1 | 19 | 54.4/62.5 | 1.0 | FAIL | PASS | FAIL |
| 9 | chinese_bob_pink_bedroom_idle_d4b06da317 | 292.7 | 19.997 | 19 | 19 | 131.6 | 6 {'transport': 1, 'server_send': 1, 'server_late': 4} | 0.99753 | 1/1 | 19 | 54.3/61.3 | 1.0 | FAIL | PASS | FAIL |
