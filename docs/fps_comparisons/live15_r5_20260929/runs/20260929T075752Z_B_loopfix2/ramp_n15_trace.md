# Live trace report: tmp/live15_r5/20260929T075752Z_B_loopfix2/traces/ramp/n15

```
{
 "level_dir": "tmp/live15_r5/20260929T075752Z_B_loopfix2/traces/ramp/n15",
 "streams": 15,
 "streams_pass": 0,
 "all_pass": false,
 "min_anchored_1s": 19,
 "worst_gap_ms": 115.9,
 "gaps_over_100ms_total": 31,
 "min_server_fresh_fraction": 0.9961,
 "max_server_held_run": 1,
 "min_pts_join": 1.0,
 "client_loop_spikes": 15,
 "client_loop_spike_max_ms": 79.6,
 "server_cadence_min_anchored_1s": 19,
 "server_send_interval_max_ms": 107.5,
 "gap_causes": {
  "server_late": 17,
  "transport": 9,
  "server_send": 5
 }
}
```

| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | chinese_bob_pink_bedroom_talking_3373c10448 | 300.4 | 19.998 | 19 | 20 | 110.1 | 1 {'server_late': 1} | 0.99743 | 1/1 | 19 | 55.1/72.7 | 1.0 | FAIL | PASS | FAIL |
| 1 | indian_realtime_talking_20f9845543 | 310.8 | 19.998 | 19 | 19 | 106.5 | 2 {'server_late': 1, 'transport': 1} | 0.99684 | 1/1 | 19 | 54.5/70.4 | 1.0 | FAIL | PASS | FAIL |
| 2 | japanese_realtime_talking_7d94520b7f | 303.8 | 19.999 | 19 | 19 | 102.6 | 4 {'server_late': 3, 'transport': 1} | 0.99861 | 1/1 | 19 | 55.0/72.3 | 1.0 | FAIL | PASS | FAIL |
| 3 | latina_guided_20260925_talking_84c5bc80b8 | 294.8 | 19.999 | 19 | 19 | 108.1 | 3 {'server_late': 3} | 0.99738 | 1/1 | 19 | 55.1/71.8 | 1.0 | FAIL | PASS | FAIL |
| 4 | codex_smoke | 291.7 | 19.998 | 19 | 20 | 107.2 | 2 {'transport': 1, 'server_late': 1} | 0.99635 | 1/1 | 19 | 54.1/71.5 | 1.0 | FAIL | PASS | FAIL |
| 5 | japanese_baddie_ltx23_talking_v1 | 293.8 | 19.998 | 19 | 19 | 115.4 | 2 {'server_send': 1, 'transport': 1} | 0.99856 | 1/1 | 19 | 55.0/73.0 | 1.0 | FAIL | PASS | FAIL |
| 6 | latina_baddie_ltx23_talking_v1 | 294.7 | 19.997 | 19 | 19 | 112.0 | 2 {'server_late': 2} | 0.99737 | 1/1 | 19 | 55.1/72.2 | 1.0 | FAIL | PASS | FAIL |
| 7 | ltx23_talking_endpoint_override_pilot_20260922 | 298.8 | 19.998 | 19 | 19 | 112.6 | 2 {'server_send': 1, 'transport': 1} | 0.9961 | 1/1 | 19 | 55.1/73.6 | 1.0 | FAIL | PASS | FAIL |
| 8 | japanese_relay64_talking_preview_20260925 | 297.8 | 19.996 | 19 | 19 | 110.2 | 2 {'transport': 1, 'server_late': 1} | 0.99858 | 1/1 | 19 | 54.8/72.1 | 1.0 | FAIL | PASS | FAIL |
| 9 | chinese_bob_pink_bedroom_idle_d4b06da317 | 293.7 | 19.997 | 19 | 19 | 108.2 | 3 {'server_late': 3} | 0.99757 | 1/1 | 19 | 55.0/73.5 | 1.0 | FAIL | PASS | FAIL |
| 10 | indian_realtime_idle_099877cef2 | 298.8 | 19.998 | 19 | 19 | 113.7 | 2 {'server_send': 1, 'transport': 1} | 0.9963 | 1/1 | 19 | 54.4/72.1 | 1.0 | FAIL | PASS | FAIL |
| 11 | japanese_realtime_idle_41f91fd081 | 297.8 | 19.998 | 19 | 19 | 111.0 | 1 {'server_send': 1} | 0.99878 | 1/1 | 19 | 54.9/72.2 | 1.0 | FAIL | PASS | FAIL |
| 12 | latina_guided_20260925_idle_ee675cb4fd | 307.8 | 19.998 | 19 | 19 | 108.9 | 1 {'server_late': 1} | 0.99769 | 1/1 | 18 | 54.7/70.9 | 1.0 | FAIL | PASS | FAIL |
| 13 | latina_baddie_ltx23_idle_v1 | 291.7 | 19.997 | 19 | 19 | 104.2 | 1 {'transport': 1} | 0.99666 | 1/1 | 19 | 54.7/71.8 | 1.0 | FAIL | PASS | FAIL |
| 14 | chinese_bob_pink_bedroom_smiling_dae31eb56e | 293.8 | 19.998 | 19 | 19 | 115.9 | 3 {'server_send': 1, 'transport': 1, 'server_late': 1} | 0.99856 | 1/1 | 19 | 54.7/71.9 | 1.0 | FAIL | PASS | FAIL |
