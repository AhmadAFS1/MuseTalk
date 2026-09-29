# Live trace report: tmp/live15_r5/20260929T084351Z_B_loopfix3_lightpoll/traces/ramp/n15

```
{
 "level_dir": "tmp/live15_r5/20260929T084351Z_B_loopfix3_lightpoll/traces/ramp/n15",
 "streams": 15,
 "streams_pass": 10,
 "all_pass": false,
 "min_anchored_1s": 19,
 "worst_gap_ms": 103.0,
 "gaps_over_100ms_total": 6,
 "min_server_fresh_fraction": 0.9961,
 "max_server_held_run": 1,
 "min_pts_join": 1.0,
 "client_loop_spikes": 15,
 "client_loop_spike_max_ms": 82.3,
 "server_cadence_min_anchored_1s": 20,
 "server_send_interval_max_ms": 86.3,
 "gap_causes": {
  "server_late": 2,
  "transport": 4
 }
}
```

| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | chinese_bob_pink_bedroom_talking_3373c10448 | 300.5 | 19.996 | 19 | 19 | 101.1 | 1 {'server_late': 1} | 0.99744 | 1/1 | 19 | 55.0/68.1 | 1.0 | FAIL | PASS | FAIL |
| 1 | indian_realtime_talking_20f9845543 | 304.8 | 19.999 | 19 | 19 | 101.8 | 2 {'transport': 2} | 0.99683 | 1/1 | 19 | 54.8/67.1 | 1.0 | FAIL | PASS | FAIL |
| 2 | japanese_realtime_talking_7d94520b7f | 302.8 | 19.998 | 20 | 19 | 95.6 | 0  | 0.99861 | 1/1 | 19 | 55.2/68.9 | 1.0 | PASS | PASS | PASS |
| 3 | latina_guided_20260925_talking_84c5bc80b8 | 295.8 | 19.998 | 20 | 19 | 96.0 | 0  | 0.99738 | 1/1 | 19 | 55.3/67.4 | 1.0 | PASS | PASS | PASS |
| 4 | codex_smoke | 298.8 | 20.0 | 19 | 19 | 103.0 | 1 {'transport': 1} | 0.9965 | 1/1 | 19 | 54.2/66.4 | 1.0 | FAIL | PASS | FAIL |
| 5 | japanese_baddie_ltx23_talking_v1 | 292.8 | 19.998 | 19 | 19 | 102.7 | 1 {'transport': 1} | 0.99856 | 1/1 | 19 | 55.2/67.5 | 1.0 | FAIL | PASS | FAIL |
| 6 | latina_baddie_ltx23_talking_v1 | 293.8 | 19.999 | 19 | 19 | 101.0 | 1 {'server_late': 1} | 0.99737 | 1/1 | 19 | 54.8/67.7 | 1.0 | FAIL | PASS | FAIL |
| 7 | ltx23_talking_endpoint_override_pilot_20260922 | 301.8 | 19.997 | 20 | 19 | 92.7 | 0  | 0.9961 | 1/1 | 19 | 55.1/69.5 | 1.0 | PASS | PASS | PASS |
| 8 | japanese_relay64_talking_preview_20260925 | 298.9 | 19.999 | 20 | 19 | 98.7 | 0  | 0.99858 | 1/1 | 19 | 55.1/68.6 | 1.0 | PASS | PASS | PASS |
| 9 | chinese_bob_pink_bedroom_idle_d4b06da317 | 294.8 | 19.999 | 20 | 19 | 95.5 | 0  | 0.99757 | 1/1 | 19 | 55.2/68.5 | 1.0 | PASS | PASS | PASS |
| 10 | indian_realtime_idle_099877cef2 | 298.8 | 19.998 | 20 | 19 | 94.7 | 0  | 0.9963 | 1/1 | 19 | 55.0/68.7 | 1.0 | PASS | PASS | PASS |
| 11 | japanese_realtime_idle_41f91fd081 | 291.9 | 19.998 | 20 | 20 | 92.9 | 0  | 0.99874 | 1/1 | 19 | 55.5/69.9 | 1.0 | PASS | PASS | PASS |
| 12 | latina_guided_20260925_idle_ee675cb4fd | 305.9 | 19.999 | 20 | 19 | 94.4 | 0  | 0.99769 | 1/1 | 19 | 54.8/67.8 | 1.0 | PASS | PASS | PASS |
| 13 | latina_baddie_ltx23_idle_v1 | 302.9 | 19.998 | 20 | 19 | 96.2 | 0  | 0.99659 | 1/1 | 19 | 54.5/65.7 | 1.0 | PASS | PASS | PASS |
| 14 | chinese_bob_pink_bedroom_smiling_dae31eb56e | 294.9 | 19.999 | 20 | 19 | 95.7 | 0  | 0.99856 | 1/1 | 19 | 55.1/67.8 | 1.0 | PASS | PASS | PASS |
