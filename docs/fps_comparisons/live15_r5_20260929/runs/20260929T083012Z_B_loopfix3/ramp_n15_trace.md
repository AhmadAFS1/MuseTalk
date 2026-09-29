# Live trace report: tmp/live15_r5/20260929T083012Z_B_loopfix3/traces/ramp/n15

```
{
 "level_dir": "tmp/live15_r5/20260929T083012Z_B_loopfix3/traces/ramp/n15",
 "streams": 15,
 "streams_pass": 0,
 "all_pass": false,
 "min_anchored_1s": 18,
 "worst_gap_ms": 153.5,
 "gaps_over_100ms_total": 40,
 "min_server_fresh_fraction": 0.9961,
 "max_server_held_run": 1,
 "min_pts_join": 1.0,
 "client_loop_spikes": 15,
 "client_loop_spike_max_ms": 95.8,
 "server_cadence_min_anchored_1s": 19,
 "server_send_interval_max_ms": 125.1,
 "gap_causes": {
  "server_late": 18,
  "server_send": 10,
  "transport": 12
 }
}
```

| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | chinese_bob_pink_bedroom_talking_3373c10448 | 301.5 | 19.998 | 19 | 19 | 130.4 | 5 {'server_late': 2, 'server_send': 1, 'transport': 2} | 0.99744 | 1/1 | 19 | 55.1/77.7 | 1.0 | FAIL | PASS | FAIL |
| 1 | indian_realtime_talking_20f9845543 | 306.8 | 19.997 | 19 | 19 | 115.0 | 3 {'server_late': 3} | 0.99683 | 1/1 | 19 | 54.6/76.9 | 1.0 | FAIL | PASS | FAIL |
| 2 | japanese_realtime_talking_7d94520b7f | 304.8 | 19.995 | 19 | 19 | 144.8 | 4 {'server_send': 1, 'server_late': 1, 'transport': 2} | 0.99861 | 1/1 | 19 | 55.2/76.5 | 1.0 | FAIL | PASS | FAIL |
| 3 | latina_guided_20260925_talking_84c5bc80b8 | 297.8 | 19.998 | 19 | 19 | 120.1 | 3 {'server_late': 2, 'transport': 1} | 0.99738 | 1/1 | 19 | 54.8/76.9 | 1.0 | FAIL | PASS | FAIL |
| 4 | codex_smoke | 302.8 | 19.997 | 19 | 19 | 120.5 | 3 {'transport': 1, 'server_send': 1, 'server_late': 1} | 0.9965 | 1/1 | 19 | 54.2/76.6 | 1.0 | FAIL | PASS | FAIL |
| 5 | japanese_baddie_ltx23_talking_v1 | 292.8 | 19.995 | 18 | 19 | 151.0 | 2 {'server_send': 1, 'server_late': 1} | 0.99856 | 1/1 | 18 | 55.2/76.6 | 1.0 | FAIL | PASS | FAIL |
| 6 | latina_baddie_ltx23_talking_v1 | 297.8 | 19.997 | 19 | 19 | 118.8 | 2 {'transport': 1, 'server_late': 1} | 0.99737 | 1/1 | 19 | 55.0/77.4 | 1.0 | FAIL | PASS | FAIL |
| 7 | ltx23_talking_endpoint_override_pilot_20260922 | 300.8 | 19.998 | 18 | 19 | 153.5 | 2 {'server_send': 1, 'server_late': 1} | 0.9961 | 1/1 | 18 | 55.0/76.1 | 1.0 | FAIL | PASS | FAIL |
| 8 | japanese_relay64_talking_preview_20260925 | 300.8 | 19.998 | 19 | 18 | 122.7 | 3 {'transport': 2, 'server_late': 1} | 0.99858 | 1/1 | 19 | 55.1/77.0 | 1.0 | FAIL | PASS | FAIL |
| 9 | chinese_bob_pink_bedroom_idle_d4b06da317 | 293.8 | 19.999 | 19 | 19 | 146.2 | 3 {'server_send': 1, 'transport': 1, 'server_late': 1} | 0.99758 | 1/1 | 19 | 55.3/77.2 | 1.0 | FAIL | PASS | FAIL |
| 10 | indian_realtime_idle_099877cef2 | 299.9 | 19.998 | 19 | 19 | 144.7 | 3 {'server_send': 1, 'server_late': 1, 'transport': 1} | 0.99631 | 1/1 | 19 | 54.6/77.8 | 1.0 | FAIL | PASS | FAIL |
| 11 | japanese_realtime_idle_41f91fd081 | 299.8 | 19.998 | 19 | 19 | 147.6 | 2 {'server_send': 1, 'transport': 1} | 0.99878 | 1/1 | 19 | 55.2/77.6 | 1.0 | FAIL | PASS | FAIL |
| 12 | latina_guided_20260925_idle_ee675cb4fd | 307.8 | 19.998 | 18 | 19 | 151.1 | 1 {'server_send': 1} | 0.99769 | 1/1 | 18 | 54.9/75.7 | 1.0 | FAIL | PASS | FAIL |
| 13 | latina_baddie_ltx23_idle_v1 | 292.8 | 19.998 | 19 | 19 | 138.0 | 2 {'server_send': 1, 'server_late': 1} | 0.99666 | 1/1 | 19 | 54.8/75.5 | 1.0 | FAIL | PASS | FAIL |
| 14 | chinese_bob_pink_bedroom_smiling_dae31eb56e | 294.9 | 19.998 | 19 | 19 | 115.6 | 2 {'server_late': 2} | 0.99856 | 1/1 | 19 | 54.8/75.4 | 1.0 | FAIL | PASS | FAIL |
