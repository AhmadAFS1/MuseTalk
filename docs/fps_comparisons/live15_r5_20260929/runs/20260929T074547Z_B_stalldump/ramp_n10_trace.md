# Live trace report: tmp/live15_r5/20260929T074547Z_B_stalldump/traces/ramp/n10

```
{
 "level_dir": "tmp/live15_r5/20260929T074547Z_B_stalldump/traces/ramp/n10",
 "streams": 10,
 "streams_pass": 0,
 "all_pass": false,
 "min_anchored_1s": 16,
 "worst_gap_ms": 216.4,
 "gaps_over_100ms_total": 92,
 "min_server_fresh_fraction": 0.99725,
 "max_server_held_run": 1,
 "min_pts_join": 1.0,
 "client_loop_spikes": 11,
 "client_loop_spike_max_ms": 76.6,
 "server_cadence_min_anchored_1s": 16,
 "server_send_interval_max_ms": 204.4,
 "gap_causes": {
  "server_send": 26,
  "client_loop": 6,
  "server_late": 42,
  "transport": 18
 }
}
```

| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | chinese_bob_pink_bedroom_talking_3373c10448 | 304.6 | 19.998 | 17 | 19 | 206.4 | 10 {'server_send': 3, 'client_loop': 1, 'server_late': 5, 'transport': 1} | 0.99788 | 1/1 | 17 | 54.6/63.9 | 1.0 | FAIL | FAIL | FAIL |
| 1 | indian_realtime_talking_20f9845543 | 298.8 | 19.988 | 16 | 18 | 216.4 | 12 {'server_send': 4, 'client_loop': 1, 'transport': 3, 'server_late': 4} | 0.99737 | 1/1 | 16 | 54.4/63.9 | 1.0 | FAIL | FAIL | FAIL |
| 2 | japanese_realtime_talking_7d94520b7f | 299.7 | 20.001 | 17 | 17 | 204.2 | 11 {'server_late': 4, 'transport': 1, 'client_loop': 1, 'server_send': 5} | 0.99743 | 1/1 | 17 | 54.7/63.7 | 1.0 | FAIL | FAIL | FAIL |
| 3 | latina_guided_20260925_talking_84c5bc80b8 | 314.8 | 19.989 | 16 | 17 | 208.4 | 10 {'server_send': 3, 'transport': 2, 'client_loop': 1, 'server_late': 4} | 0.99812 | 1/1 | 16 | 54.9/65.0 | 1.0 | FAIL | FAIL | FAIL |
| 4 | codex_smoke | 300.8 | 19.99 | 16 | 18 | 205.6 | 11 {'client_loop': 1, 'server_send': 3, 'transport': 2, 'server_late': 5} | 0.99725 | 1/1 | 16 | 53.9/63.8 | 1.0 | FAIL | FAIL | FAIL |
| 5 | japanese_baddie_ltx23_talking_v1 | 301.7 | 20.008 | 18 | 19 | 162.5 | 10 {'server_late': 5, 'server_send': 2, 'transport': 3} | 0.9974 | 1/1 | 18 | 54.3/63.1 | 1.0 | FAIL | PASS | FAIL |
| 6 | latina_baddie_ltx23_talking_v1 | 302.8 | 20.001 | 19 | 19 | 158.8 | 10 {'server_send': 2, 'transport': 3, 'server_late': 4, 'client_loop': 1} | 0.99763 | 1/1 | 19 | 54.7/64.2 | 1.0 | FAIL | PASS | FAIL |
| 7 | ltx23_talking_endpoint_override_pilot_20260922 | 294.7 | 20.001 | 19 | 20 | 131.1 | 7 {'server_late': 5, 'server_send': 2} | 0.9978 | 1/1 | 19 | 54.5/62.4 | 1.0 | FAIL | PASS | FAIL |
| 8 | japanese_relay64_talking_preview_20260925 | 309.8 | 19.998 | 19 | 19 | 112.0 | 6 {'server_send': 1, 'transport': 2, 'server_late': 3} | 0.99768 | 1/1 | 19 | 54.8/63.6 | 1.0 | FAIL | PASS | FAIL |
| 9 | chinese_bob_pink_bedroom_idle_d4b06da317 | 291.8 | 19.998 | 19 | 19 | 109.4 | 5 {'server_send': 1, 'server_late': 3, 'transport': 1} | 0.99752 | 1/1 | 18 | 54.4/61.9 | 1.0 | FAIL | PASS | FAIL |
