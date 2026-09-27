import sys, json
for line in open("webrtc_transport_load_results.jsonl"):
    if not line.startswith('{'): print(line.strip()); continue
    d = json.loads(line); c = d['config']; s = d['sender'] or {}
    print(c['mode'], c['enc'], c['streams'], '| recv agg %.1f min %.1f p99gap %s maxgap %s | proc %.2f cores loop %.2f lag p99 %.1f max %.1f push p50 %s p99 %s thr %s und %s' % (
      d['recv_aggregate_fps'] or 0, d['recv_fps_min'] or 0, round(d['recv_gap_ms_p99_worst'] or 0), round(d['recv_gap_ms_max_worst'] or 0),
      s.get('process_cores',0), s.get('loop_thread_core_frac',0), s.get('loop_lag_ms_p99',0), s.get('loop_lag_ms_max',0),
      None if s.get('push_batch_ms_p50') is None else round(s['push_batch_ms_p50'],1), None if s.get('push_batch_ms_p99') is None else round(s['push_batch_ms_p99'],1), s.get('threads'), s.get('underruns_total')))
