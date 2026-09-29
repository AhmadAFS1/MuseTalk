# Live trace report: tmp/live15_r5/20260929T091234Z_B_final/traces/ramp/n15

```
{
 "level_dir": "tmp/live15_r5/20260929T091234Z_B_final/traces/ramp/n15",
 "streams": 15,
 "streams_pass": 7,
 "all_pass": false,
 "min_anchored_1s": 19,
 "worst_gap_ms": 101.6,
 "gaps_over_100ms_total": 3,
 "min_server_fresh_fraction": 0.9961,
 "max_server_held_run": 1,
 "min_pts_join": 1.0,
 "client_loop_spikes": 15,
 "client_loop_spike_max_ms": 80.1,
 "server_cadence_min_anchored_1s": 20,
 "server_send_interval_max_ms": 85.0,
 "gap_causes": {
  "server_late": 3
 }
}
```

| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | chinese_bob_pink_bedroom_talking_3373c10448 | 299.5 | 19.999 | 20 | 19 | 97.8 | 0  | 0.99743 | 1/1 | 19 | 54.8/66.0 | 1.0 | PASS | PASS | PASS |
| 1 | indian_realtime_talking_20f9845543 | 307.8 | 19.999 | 20 | 19 | 91.9 | 0  | 0.99684 | 1/1 | 19 | 54.6/65.4 | 1.0 | PASS | PASS | PASS |
| 2 | japanese_realtime_talking_7d94520b7f | 302.8 | 19.999 | 20 | 19 | 92.2 | 0  | 0.99861 | 1/1 | 19 | 55.1/66.5 | 1.0 | PASS | PASS | PASS |
| 3 | latina_guided_20260925_talking_84c5bc80b8 | 292.8 | 19.997 | 19 | 19 | 100.7 | 1 {'server_late': 1} | 0.99738 | 1/1 | 19 | 55.2/68.0 | 1.0 | FAIL | PASS | FAIL |
| 4 | codex_smoke | 298.8 | 19.998 | 19 | 19 | 100.0 | 0  | 0.9965 | 1/1 | 19 | 54.2/66.6 | 1.0 | FAIL | PASS | PASS |
| 5 | japanese_baddie_ltx23_talking_v1 | 293.9 | 19.999 | 19 | 19 | 96.9 | 0  | 0.99856 | 1/1 | 19 | 54.9/67.6 | 1.0 | FAIL | PASS | PASS |
| 6 | latina_baddie_ltx23_talking_v1 | 295.9 | 19.999 | 19 | 19 | 101.6 | 1 {'server_late': 1} | 0.99737 | 1/1 | 19 | 55.0/67.7 | 1.0 | FAIL | PASS | FAIL |
| 7 | ltx23_talking_endpoint_override_pilot_20260922 | 299.9 | 19.998 | 19 | 19 | 100.7 | 1 {'server_late': 1} | 0.9961 | 1/1 | 19 | 55.3/69.3 | 1.0 | FAIL | PASS | FAIL |
| 8 | japanese_relay64_talking_preview_20260925 | 295.8 | 19.997 | 20 | 19 | 97.3 | 0  | 0.99858 | 1/1 | 19 | 55.4/67.9 | 1.0 | PASS | PASS | PASS |
| 9 | chinese_bob_pink_bedroom_idle_d4b06da317 | 295.8 | 19.999 | 19 | 19 | 93.2 | 0  | 0.99757 | 1/1 | 19 | 55.1/68.9 | 1.0 | FAIL | PASS | PASS |
| 10 | indian_realtime_idle_099877cef2 | 296.8 | 19.998 | 19 | 19 | 98.6 | 0  | 0.9963 | 1/1 | 19 | 54.8/68.1 | 1.0 | FAIL | PASS | PASS |
| 11 | japanese_realtime_idle_41f91fd081 | 297.9 | 19.998 | 20 | 19 | 95.8 | 0  | 0.99878 | 1/1 | 19 | 54.8/67.2 | 1.0 | PASS | PASS | PASS |
| 12 | latina_guided_20260925_idle_ee675cb4fd | 304.9 | 19.998 | 20 | 19 | 96.5 | 0  | 0.99769 | 1/1 | 19 | 54.9/67.7 | 1.0 | PASS | PASS | PASS |
| 13 | latina_baddie_ltx23_idle_v1 | 293.9 | 19.998 | 20 | 20 | 96.5 | 0  | 0.99666 | 1/1 | 19 | 54.8/67.6 | 1.0 | PASS | PASS | PASS |
| 14 | chinese_bob_pink_bedroom_smiling_dae31eb56e | 291.9 | 19.999 | 19 | 19 | 97.6 | 0  | 0.99856 | 1/1 | 19 | 55.0/68.2 | 1.0 | FAIL | PASS | PASS |
