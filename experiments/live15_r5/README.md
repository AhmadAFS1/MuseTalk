# Live WebRTC test rig: 15 concurrent streams on the r5 engines

Findings, results and recommendations: `docs/fps_comparisons/live15_r5_20260929/README.md`.
This file covers how to run the rig and what it measures.

## What it does

`run_live15.sh` starts a private MuseTalk API server from this worktree on `127.0.0.1:8300`, with the r5 engines.
It then drives real WebRTC viewers against it with `load_test_webrtc_v2.py`: aiortc peers over loopback that decode
the video, one client process per viewer. Each viewer uses a different avatar (`avatars.txt`) and speaks
back-to-back turns from `experiments/throughput300_candidate/audio_corpus`, with a 1 s gap between turns.

The rig is isolated from everything else on the box:
- port 8300 only, and a private runtime dir under the run dir;
- no Lingua control plane, TURN, STUN or S3;
- it never binds 8000/8200 and never uses `/workspace/logs/musetalk`, which holds the user's coturn pid;
- the server is a child of the script, so `box_guard` covers it, and the EXIT trap always stops it;
- `sample_box.sh` stops the server at once if any other GPU compute app appears.

```bash
cd /workspace/MuseTalk-perf300
E=$PWD/experiments/live15_r5; RUN=$PWD/tmp/live15_r5/$(date -u +%Y%m%dT%H%M%SZ)_B_final; mkdir -p $RUN
bash scripts/box_guard.sh run --min-avail-gb 14 --need-disk-gb 2 --wait-min 60 --kill-below-gb 3.0 --label live15 -- \
  bash $E/run_live15.sh B_final $RUN "$E/loopfix.env:$E/serve.env:$E/common.env" ramp,soak "5 10 15"
python experiments/live15_r5/summarize_runs.py tmp/live15_r5      # one table over every run
```

Stages:
- **s0**: N=1 then N=3, 2 turns each, as a smoke test.
- **ramp**: at each level, real joins 5 s apart, then 300 s of speech after each join. The ramp stops at the
  first level that fails.
- **soak**: when run together with the ramp, only after the ramp has passed the soak size. `LIVE15_SOAK_N`
  streams (default 15) join 20 s apart and each gets 3320 s of speech. Observers record 90 s from 1800 s on the
  streams in `LIVE15_SOAK_RECORD` (default: the first and last). Setting it empty turns them off; an observer
  stalls its own client process when it starts (see the docs page, §6).

## Override files

The server parses these, and never sources them. In a colon list the first file wins.

| file | what it adds |
|---|---|
| `common.env` | Arm A: the r5 engines (bs16, srcg50 INT8 UNet, TAESD TRT), scheduler shape 16/[16], lifetime counters, deadline pacing, the lean avatar layout, a 9 GB avatar cache |
| `serve.env` | Arm B serving levers: exact by design (non-blocking handoff, 2 converter threads, idle frame cache, thread caps, crossfade copy skip, stage-sync timing off) |
| `loopfix.env` | the event-loop fixes found by this test (the production configuration; see the docs page) |
| `pipe.env` | Arm C: pipeline depth 2 and EDF scheduling. Not run live. |
| `gcthr.env` | the last candidate before it was folded into `loopfix.env` |
| `gclog.env`, `gcfreeze.env`, `loopdiag.env`, `loopfixdiag.env`, `lagdiag.env`, `stalldump.env`, `garbage*.env`, `loopfix_gclog.env`, `swi1.env`, `enc_x264faster.env` | diagnostics and rejected candidates, kept so each run dir can be reproduced |

Diagnostics change the timing they measure. The GC log callback runs on every collection, and the lag and stall
samplers wake the loop every 5 ms. So score pass/fail only from runs without them.

## Pass criteria (defined before the first run)

Scored per stream by `live_trace_report.py` from client arrivals joined to the server's send ring (both are
CLOCK_MONOTONIC on this box):
- **P1 rate:** every 1 s window that starts at a frame arrival holds at least 20 frames.
- **P2 fresh:** server fresh fraction ≥ 0.995 over speaking slots, and max held run ≤ 2. Client content frames
  ≥ 18 per anchored 1 s window.
- **P3 buffering:** max client gap ≤ 120 ms. Gaps over 100 ms occur at most once per stream per 10 min, and none
  is caused by the server's own send cadence.

The report also prints the **server cadence**: the same anchored count over the server's send times. This is what a
browser sees before the network, because a browser completes a frame on its last packet. aiortc's receiver instead
releases frame N when frame N+1's first packet arrives. So it shows a server send that is 25–50 ms late as a
client gap over 100 ms, which the report labels `server_late`.

## Files

- `run_live15.sh`: the driver (stages, pre-warm of all 15 avatars, verification of backends and isolation, one
  client process per viewer).
- `live_trace_report.py`: per-stream P1–P3, gap attribution, server cadence.
- `summarize_runs.py`: a markdown table over every run dir.
- `sample_box.sh`: box.csv, gpu.csv and gpu_apps.csv at 1 Hz, sched.jsonl every 10 s, the foreign-GPU stop.
- `cpu_sampler.py`: cpu_hi.csv, 20 Hz busy fraction of the server and client CPU sets.
- `stop_server.sh`: stops only our own :8300 server.
- `avatars.txt`: the 15 avatars, all complete, none fixed-face-height.
