# Live trace report: /workspace/MuseTalk-perf300/tmp/live15_r5/20260929T093219Z_B_final_soak/traces/soak/n15

```
{
 "level_dir": "/workspace/MuseTalk-perf300/tmp/live15_r5/20260929T093219Z_B_final_soak/traces/soak/n15",
 "streams": 15,
 "streams_pass": 0,
 "all_pass": false,
 "min_anchored_1s": 4,
 "worst_gap_ms": 872.7,
 "gaps_over_100ms_total": 49,
 "min_server_fresh_fraction": 0.99652,
 "max_server_held_run": 1,
 "min_pts_join": 1.0,
 "client_loop_spikes": 20,
 "client_loop_spike_max_ms": 822.3,
 "server_cadence_min_anchored_1s": 19,
 "server_send_interval_max_ms": 135.6,
 "gap_causes": {
  "transport": 4,
  "client_loop": 3,
  "server_late": 24,
  "server_send": 18
 }
}
```

| stream | avatar | window s | mean fps | anchored 1 s min | fixed-bin min | gap max ms | gaps>100 ms (causes) | server fresh | held run srv/cli | content 1 s min | latency p50/p99 ms | pts join | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | chinese_bob_pink_bedroom_talking_3373c10448 | 3316.4 | 19.998 | 15 | 18 | 279.2 | 4 {'transport': 1, 'client_loop': 1, 'server_late': 1, 'server_send': 1} | 0.99758 | 1/1 | 15 | 55.1/69.0 | 1.0 | FAIL | FAIL | FAIL |
| 1 | indian_realtime_talking_20f9845543 | 3320.7 | 19.998 | 18 | 19 | 133.0 | 3 {'server_send': 2, 'server_late': 1} | 0.99654 | 1/1 | 18 | 54.8/68.5 | 1.0 | FAIL | PASS | FAIL |
| 2 | japanese_realtime_talking_7d94520b7f | 3311.7 | 19.998 | 18 | 19 | 130.5 | 2 {'server_late': 1, 'server_send': 1} | 0.99866 | 1/1 | 18 | 55.1/68.8 | 1.0 | FAIL | PASS | FAIL |
| 3 | latina_guided_20260925_talking_84c5bc80b8 | 3320.7 | 19.998 | 18 | 19 | 184.2 | 3 {'server_send': 2, 'client_loop': 1} | 0.99758 | 1/1 | 18 | 55.3/68.9 | 1.0 | FAIL | PASS | FAIL |
| 4 | codex_smoke | 3314.8 | 19.998 | 18 | 19 | 117.6 | 4 {'server_late': 3, 'transport': 1} | 0.99654 | 1/1 | 18 | 54.1/68.3 | 1.0 | FAIL | PASS | PASS |
| 5 | japanese_baddie_ltx23_talking_v1 | 3315.8 | 19.998 | 18 | 19 | 142.4 | 2 {'server_late': 1, 'server_send': 1} | 0.99865 | 1/1 | 18 | 55.1/68.5 | 1.0 | FAIL | PASS | FAIL |
| 6 | latina_baddie_ltx23_talking_v1 | 3317.8 | 19.998 | 18 | 19 | 128.6 | 3 {'transport': 2, 'server_send': 1} | 0.99757 | 1/1 | 18 | 54.9/68.7 | 1.0 | FAIL | PASS | FAIL |
| 7 | ltx23_talking_endpoint_override_pilot_20260922 | 3317.9 | 19.998 | 18 | 18 | 130.8 | 4 {'server_send': 2, 'server_late': 2} | 0.99652 | 1/1 | 18 | 55.4/69.0 | 1.0 | FAIL | PASS | FAIL |
| 8 | japanese_relay64_talking_preview_20260925 | 3317.9 | 19.998 | 18 | 19 | 142.2 | 4 {'server_late': 3, 'server_send': 1} | 0.99864 | 1/1 | 18 | 55.3/69.1 | 1.0 | FAIL | PASS | FAIL |
| 9 | chinese_bob_pink_bedroom_idle_d4b06da317 | 3331.9 | 19.998 | 18 | 18 | 136.0 | 4 {'server_late': 2, 'server_send': 2} | 0.99757 | 1/1 | 18 | 55.0/68.7 | 1.0 | FAIL | PASS | FAIL |
| 10 | indian_realtime_idle_099877cef2 | 3316.9 | 19.998 | 18 | 18 | 134.6 | 3 {'server_late': 2, 'server_send': 1} | 0.99652 | 1/1 | 18 | 54.6/68.2 | 1.0 | FAIL | PASS | FAIL |
| 11 | japanese_realtime_idle_41f91fd081 | 3316.0 | 19.998 | 18 | 19 | 134.1 | 5 {'server_late': 4, 'server_send': 1} | 0.99864 | 1/1 | 18 | 55.2/69.0 | 1.0 | FAIL | PASS | FAIL |
| 12 | latina_guided_20260925_idle_ee675cb4fd | 3319.9 | 19.998 | 18 | 19 | 127.5 | 2 {'server_late': 2} | 0.99754 | 1/1 | 18 | 54.8/68.1 | 1.0 | FAIL | PASS | FAIL |
| 13 | latina_baddie_ltx23_idle_v1 | 3318.0 | 19.998 | 18 | 19 | 132.1 | 2 {'server_late': 1, 'server_send': 1} | 0.99652 | 1/1 | 18 | 54.9/68.4 | 1.0 | FAIL | PASS | FAIL |
| 14 | chinese_bob_pink_bedroom_smiling_dae31eb56e | 3329.0 | 19.998 | 4 | 13 | 872.7 | 4 {'server_send': 2, 'client_loop': 1, 'server_late': 1} | 0.99865 | 1/1 | 4 | 54.8/68.5 | 1.0 | FAIL | FAIL | FAIL |
