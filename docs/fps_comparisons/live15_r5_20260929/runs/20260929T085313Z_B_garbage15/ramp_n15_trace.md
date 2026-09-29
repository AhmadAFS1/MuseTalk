# Live trace report: tmp/live15_r5/20260929T085313Z_B_garbage15/traces/ramp/n15

```
{
 "level_dir": "tmp/live15_r5/20260929T085313Z_B_garbage15/traces/ramp/n15",
 "streams": 15,
 "streams_pass": 0,
 "all_pass": false,
 "min_anchored_1s": 17,
 "worst_gap_ms": 205.1,
 "gaps_over_100ms_total": 355,
 "min_server_fresh_fraction": 0.9961,
 "max_server_held_run": 1,
 "min_pts_join": 1.0,
 "client_loop_spikes": 15,
 "client_loop_spike_max_ms": 89.0,
 "server_cadence_min_anchored_1s": 18,
 "server_send_interval_max_ms": 188.8,
 "gap_causes": {
  "server_send": 161,
  "server_late": 157,
  "transport": 36,
  "client_loop": 1
 }
}
```

| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | chinese_bob_pink_bedroom_talking_3373c10448 | 303.5 | 19.999 | 17 | 19 | 194.1 | 24 {'server_send': 10, 'server_late': 13, 'transport': 1} | 0.99744 | 1/1 | 17 | 55.2/70.3 | 1.0 | FAIL | FAIL | FAIL |
| 1 | indian_realtime_talking_20f9845543 | 306.8 | 19.998 | 17 | 18 | 204.9 | 24 {'server_late': 12, 'client_loop': 1, 'transport': 4, 'server_send': 7} | 0.99683 | 1/1 | 17 | 54.8/71.5 | 1.0 | FAIL | FAIL | FAIL |
| 2 | japanese_realtime_talking_7d94520b7f | 305.8 | 19.999 | 17 | 17 | 191.7 | 23 {'server_send': 11, 'server_late': 11, 'transport': 1} | 0.99861 | 1/1 | 17 | 55.3/70.2 | 1.0 | FAIL | FAIL | FAIL |
| 3 | latina_guided_20260925_talking_84c5bc80b8 | 296.8 | 19.996 | 17 | 17 | 201.8 | 24 {'server_late': 12, 'server_send': 11, 'transport': 1} | 0.99738 | 1/1 | 17 | 55.3/71.6 | 1.0 | FAIL | FAIL | FAIL |
| 4 | codex_smoke | 302.8 | 19.999 | 17 | 17 | 193.6 | 23 {'server_send': 11, 'server_late': 11, 'transport': 1} | 0.9965 | 1/1 | 17 | 54.2/68.3 | 1.0 | FAIL | FAIL | FAIL |
| 5 | japanese_baddie_ltx23_talking_v1 | 295.8 | 19.999 | 17 | 19 | 201.7 | 25 {'transport': 3, 'server_late': 13, 'server_send': 9} | 0.99856 | 1/1 | 16 | 55.2/70.0 | 1.0 | FAIL | FAIL | FAIL |
| 6 | latina_baddie_ltx23_talking_v1 | 293.9 | 19.999 | 17 | 18 | 187.1 | 24 {'server_send': 11, 'server_late': 11, 'transport': 2} | 0.99737 | 1/1 | 17 | 55.5/70.9 | 1.0 | FAIL | FAIL | FAIL |
| 7 | ltx23_talking_endpoint_override_pilot_20260922 | 302.8 | 19.998 | 17 | 19 | 204.3 | 25 {'server_send': 12, 'server_late': 9, 'transport': 4} | 0.9961 | 1/1 | 17 | 55.2/72.9 | 1.0 | FAIL | FAIL | FAIL |
| 8 | japanese_relay64_talking_preview_20260925 | 291.9 | 20.0 | 17 | 17 | 197.3 | 24 {'server_send': 12, 'server_late': 12} | 0.99875 | 1/1 | 17 | 55.7/70.6 | 1.0 | FAIL | FAIL | FAIL |
| 9 | chinese_bob_pink_bedroom_idle_d4b06da317 | 294.9 | 19.998 | 17 | 17 | 201.6 | 24 {'server_send': 12, 'server_late': 9, 'transport': 3} | 0.99757 | 1/1 | 17 | 54.8/71.0 | 1.0 | FAIL | FAIL | FAIL |
| 10 | indian_realtime_idle_099877cef2 | 298.8 | 19.998 | 17 | 19 | 199.4 | 21 {'server_late': 10, 'server_send': 9, 'transport': 2} | 0.9963 | 1/1 | 17 | 54.6/71.2 | 1.0 | FAIL | FAIL | FAIL |
| 11 | japanese_realtime_idle_41f91fd081 | 299.9 | 19.997 | 18 | 17 | 196.0 | 22 {'transport': 3, 'server_late': 6, 'server_send': 13} | 0.99878 | 1/1 | 18 | 54.9/70.0 | 1.0 | FAIL | PASS | FAIL |
| 12 | latina_guided_20260925_idle_ee675cb4fd | 308.9 | 19.999 | 18 | 19 | 186.1 | 25 {'server_send': 9, 'server_late': 12, 'transport': 4} | 0.99769 | 1/1 | 18 | 54.8/71.5 | 1.0 | FAIL | PASS | FAIL |
| 13 | latina_baddie_ltx23_idle_v1 | 302.8 | 19.998 | 17 | 18 | 205.1 | 23 {'server_send': 14, 'server_late': 7, 'transport': 2} | 0.99659 | 1/1 | 17 | 54.6/67.0 | 1.0 | FAIL | FAIL | FAIL |
| 14 | chinese_bob_pink_bedroom_smiling_dae31eb56e | 296.9 | 19.999 | 17 | 19 | 193.8 | 24 {'server_late': 9, 'transport': 5, 'server_send': 10} | 0.99856 | 1/1 | 17 | 54.6/69.4 | 1.0 | FAIL | FAIL | FAIL |
