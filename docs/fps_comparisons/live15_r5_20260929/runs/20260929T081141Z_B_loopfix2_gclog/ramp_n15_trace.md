# Live trace report: tmp/live15_r5/20260929T081141Z_B_loopfix2_gclog/traces/ramp/n15

```
{
 "level_dir": "tmp/live15_r5/20260929T081141Z_B_loopfix2_gclog/traces/ramp/n15",
 "streams": 15,
 "streams_pass": 0,
 "all_pass": false,
 "min_anchored_1s": 18,
 "worst_gap_ms": 163.8,
 "gaps_over_100ms_total": 326,
 "min_server_fresh_fraction": 0.9961,
 "max_server_held_run": 1,
 "min_pts_join": 1.0,
 "client_loop_spikes": 15,
 "client_loop_spike_max_ms": 78.1,
 "server_cadence_min_anchored_1s": 19,
 "server_send_interval_max_ms": 134.2,
 "gap_causes": {
  "server_late": 172,
  "transport": 46,
  "server_send": 108
 }
}
```

| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | chinese_bob_pink_bedroom_talking_3373c10448 | 302.5 | 19.998 | 18 | 19 | 155.4 | 20 {'server_late': 10, 'transport': 1, 'server_send': 9} | 0.99744 | 1/1 | 18 | 54.8/76.8 | 1.0 | FAIL | PASS | FAIL |
| 1 | indian_realtime_talking_20f9845543 | 307.8 | 19.996 | 18 | 19 | 156.8 | 21 {'server_late': 13, 'transport': 3, 'server_send': 5} | 0.99683 | 1/1 | 18 | 54.7/79.1 | 1.0 | FAIL | PASS | FAIL |
| 2 | japanese_realtime_talking_7d94520b7f | 304.8 | 19.997 | 18 | 19 | 145.3 | 24 {'server_send': 10, 'server_late': 10, 'transport': 4} | 0.99861 | 1/1 | 18 | 55.2/78.4 | 1.0 | FAIL | PASS | FAIL |
| 3 | latina_guided_20260925_talking_84c5bc80b8 | 297.8 | 19.999 | 18 | 19 | 154.1 | 17 {'server_late': 9, 'server_send': 7, 'transport': 1} | 0.99738 | 1/1 | 18 | 55.0/77.3 | 1.0 | FAIL | PASS | FAIL |
| 4 | codex_smoke | 302.8 | 19.998 | 18 | 19 | 156.3 | 24 {'server_late': 13, 'server_send': 8, 'transport': 3} | 0.9965 | 1/1 | 18 | 54.1/78.7 | 1.0 | FAIL | PASS | FAIL |
| 5 | japanese_baddie_ltx23_talking_v1 | 293.8 | 19.997 | 18 | 19 | 150.2 | 22 {'server_late': 13, 'transport': 3, 'server_send': 6} | 0.99856 | 1/1 | 18 | 55.1/79.1 | 1.0 | FAIL | PASS | FAIL |
| 6 | latina_baddie_ltx23_talking_v1 | 297.8 | 19.994 | 19 | 19 | 145.4 | 25 {'server_late': 12, 'transport': 6, 'server_send': 7} | 0.99737 | 1/1 | 18 | 55.2/80.2 | 1.0 | FAIL | PASS | FAIL |
| 7 | ltx23_talking_endpoint_override_pilot_20260922 | 298.9 | 19.992 | 18 | 19 | 163.8 | 22 {'server_late': 10, 'server_send': 8, 'transport': 4} | 0.9961 | 1/1 | 18 | 54.9/78.0 | 1.0 | FAIL | PASS | FAIL |
| 8 | japanese_relay64_talking_preview_20260925 | 292.8 | 19.998 | 18 | 19 | 151.6 | 19 {'transport': 3, 'server_late': 7, 'server_send': 9} | 0.99875 | 1/1 | 18 | 55.4/80.0 | 1.0 | FAIL | PASS | FAIL |
| 9 | chinese_bob_pink_bedroom_idle_d4b06da317 | 295.8 | 19.999 | 18 | 19 | 149.7 | 22 {'server_late': 12, 'server_send': 8, 'transport': 2} | 0.99757 | 1/1 | 18 | 55.0/80.9 | 1.0 | FAIL | PASS | FAIL |
| 10 | indian_realtime_idle_099877cef2 | 300.9 | 19.998 | 18 | 19 | 147.4 | 24 {'transport': 5, 'server_late': 13, 'server_send': 6} | 0.9963 | 1/1 | 18 | 54.6/78.7 | 1.0 | FAIL | PASS | FAIL |
| 11 | japanese_realtime_idle_41f91fd081 | 292.9 | 19.996 | 18 | 19 | 155.7 | 20 {'server_late': 13, 'server_send': 6, 'transport': 1} | 0.99874 | 1/1 | 18 | 55.1/79.8 | 1.0 | FAIL | PASS | FAIL |
| 12 | latina_guided_20260925_idle_ee675cb4fd | 309.9 | 19.996 | 18 | 19 | 159.3 | 19 {'server_late': 10, 'server_send': 5, 'transport': 4} | 0.99769 | 1/1 | 18 | 54.6/78.2 | 1.0 | FAIL | PASS | FAIL |
| 13 | latina_baddie_ltx23_idle_v1 | 292.8 | 19.997 | 19 | 19 | 136.1 | 24 {'server_late': 14, 'transport': 4, 'server_send': 6} | 0.99666 | 1/1 | 19 | 54.7/78.1 | 1.0 | FAIL | PASS | FAIL |
| 14 | chinese_bob_pink_bedroom_smiling_dae31eb56e | 295.9 | 19.996 | 18 | 19 | 146.3 | 23 {'server_send': 8, 'server_late': 13, 'transport': 2} | 0.99856 | 1/1 | 18 | 54.8/78.5 | 1.0 | FAIL | PASS | FAIL |
