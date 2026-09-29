# Live WebRTC test: 15 concurrent streams on the r5 engines (2026-09-29)

**Question.** Can the r5 build of MuseTalk serve 15 concurrent live WebRTC calls, each on a different avatar,
without buffering? The platform's definition of buffering is any stream dropping below 20 fps.

**Answer.**

| Streams | Result |
|---|---|
| 5 | Passes every criterion on every stream, with margin |
| 10 | Passes every criterion on every stream, with margin (5-minute runs; not soaked for an hour) |
| 15 | Not yet buffer-free for a full hour |

- **The live server as it was failed at 5 streams.** All the fixes are in `experiments/live15_r5/loopfix.env`.
  Each is behind a flag that defaults off and leaves every output frame unchanged.
- **At 15 streams, 5-minute runs sit exactly at the edge.** One run passed on all 15 streams; the final run had
  7 of 15 pass, and the other 8 had one or two single frames arriving 92–102 ms after the previous one. The
  server always sent at least 20 frames per second per stream.
- **The 60-minute soak at 15 streams failed the strict bar.** The first 15 minutes were clean. Then two
  server-wide stalls of about 80 ms, at 30 and 50 minutes, gave every stream one 114–142 ms hitch, and some
  1 s windows dropped to 18 frames (§6).

**The GPU is not what limits live serving.** At 15 streams it generates 214 fps of a ~400 fps capacity, averages
about 52% busy, and uses 4.4 GB of VRAM. The limit is the single Python event loop that paces every stream. It
stalls whenever something holds the GIL: garbage collection, blocking calls, or GIL-heavy work on other threads.

## 1. Setup

**Server.** `api_server.py` from this worktree on `127.0.0.1:8300`, launched through `scripts/run_musetalk_server.sh`
by `experiments/live15_r5/run_live15.sh`, under `box_guard`. It uses the r5 engines:
- UNet `models/tensorrt_unet_stagewise_sm89_srcg50`: bs16, layer-selective INT8 recipe `gmac_0.50`.
- TAESD TRT engine `6111388248264a4ef2ae`: bs8.
- Scheduler shape `FIXED=16 / MAX_BATCH=16`, `WEBRTC_DEADLINE_PACING=1`.
- The lean avatar layout, with all 15 avatars pre-warmed into a 9 GB cache and no evictions.

All runs use the Arm B serving levers (`serve.env`). These were CPU-tested as exact before this test:
- non-blocking handoff with 2 converter threads;
- idle-frame cache;
- thread caps;
- crossfade copy skip;
- stage-sync timing off.

Arm A (no serving levers) and Arm C (pipeline depth 2 + EDF scheduling) were not run live (§8).

**Viewers.** `load_test_webrtc_v2.py` runs real aiortc 1.14 WebRTC peers over loopback:
- They send recvonly offers, decode the video, and record every frame's arrival time and RTP pts.
- There is one client process per viewer, pinned to CPUs 12–15 and 28–31. The server is pinned to 0–11 and 16–27.
- Each viewer uses a different avatar and speaks back-to-back turns from the audio corpus with 1 s gaps, so about
  70% of the time is speech.
- Viewers join 5 s apart in the ramp and 20 s apart in the soak.

**Measurements.**
- Client arrival of every decoded frame.
- The server's per-track send ring: for every frame handed to the encoder, its time and whether it was fresh,
  held, prebuffer or idle, polled at 1 Hz and joined to client frames by pts.
- RAM, GPU and CPU samples at 1–20 Hz.

**Pass criteria, fixed before the first run.** They are scored per stream by `experiments/live15_r5/live_trace_report.py`:
- **P1 rate:** every 1 s window starting at a frame arrival holds ≥ 20 frames. This is "never below 20 fps"
  measured at the viewer.
- **P2 fresh:** server fresh fraction ≥ 0.995, max held run ≤ 2, and client content ≥ 18 frames per 1 s window.
- **P3 buffering:** max gap ≤ 120 ms, and gaps over 100 ms occur at most once per stream per 10 min, none of them
  caused by the server's send cadence.

The report also prints the server cadence: the P1 count over the server's own send times.

**Isolation.**
- The server used a private runtime dir, with no Lingua control plane, TURN, STUN or S3.
- Ports 8000 and 8200 and the user's coturn pid file were never touched; coturn was alive before and after every run.
- A sampler stops the test server at once if any other GPU process appears. None appeared.

## 2. Results

Every run is listed in [`runs_table.md`](runs_table.md), and the per-run artefacts are in [`runs/`](runs/).
The final configuration is `loopfix.env` + `serve.env` + `common.env`. Its runs, and the run that passed on
all 15 streams:

| run | N | streams passing P1–P3 | worst 1 s window (client) | worst client gap | gaps > 100 ms | server: worst 1 s window / largest send interval | server fresh fraction, max held run |
|---|---|---|---|---|---|---|---|
| `B_final` | 5 | **5/5** | 20 | 70.0 ms | 0 | 20 / 69 ms | 0.9972, 1 |
| `B_final` | 10 | **10/10** | 20 | 88.9 ms | 0 | 20 / 72 ms | 0.9973, 1 |
| `B_gcthr` | 15 | **15/15** | 20 | 90.1 ms | 0 | 20 / 82 ms | 0.9961, 1 |
| `B_final` | 15 | 7/15 | 19 (8 streams, 1–2 windows each) | 101.6 ms | 3 | 20 / 85 ms | 0.9961, 1 |
| `B_final_soak` | 15, 60 min | 0/15 | 18 (4 on the two observer streams) | 142 ms (873 ms observer) | 49 | 19 / 136 ms | 0.9965, 1 |

- **Frame rate.** Every stream averaged 19.996–19.999 fps in every run.
- **Latency.** Frame latency from server send to decoded frame is 54 ms p50 at every N (a one-frame artefact of the
  aiortc receiver, §4.8). The p99 is 58 ms at N=5, 61 ms at N=10 and 69 ms at N=15.
- **Turn start.** From `POST /stream` to the first lip-synced frame sent: 0.68 / 0.74 / 0.98 s p95 at N = 5 / 10 / 15.
- **Resources at N=15.** Server RSS peaks at 10.4 GB and MemAvailable bottoms at 4.9 GB (on this shared 30 GB box).
  The server runs up to 615 threads. The server CPU set is 9% busy at p50 and 33% at p99.

How each fix moved the result. Each row adds to the one above; "gaps" means client gaps over 100 ms.

| step | N | result |
|---|---|---|
| Arm B as it was | 5 | 0/5 streams; 71 gaps, worst 199.5 ms; server send interval up to 195 ms |
| + `gc.freeze()` (§4.1) | 5 | 0/5; 40 gaps, worst 137 ms |
| + `nvidia-smi` off the loop (§4.2) | 5 | server passes (send ≤ 71 ms); client 1/5 |
| + one client process per viewer (§4.3, test rig) | 5 / 10 | **5/5** / 6/10 |
| + idle clips built at warm (§4.4) | 10 / 15 | **10/10** / 0/15; 31 gaps |
| + light test polling (§4.5, test rig) and packed-I420 FIFO (§4.6) | 15 | 10/15; 6 gaps, worst 103 ms |
| + rarer full GC (§4.7) | 15 | **15/15** in `B_gcthr`; 7/15 in `B_final` |

## 3. Why the offline 400 fps did not carry over directly

The offline stability test (`4070s_400fps_20260928/stability_n15.md`) ran 15 streams at 26.6 fps or more each.
That was a GPU-bound harness with no WebRTC.

The live server adds a per-stream media path in Python for every one of those streams: pacing, VP8 encode, RTP,
SRTP and audio. All of it is timed by one asyncio event loop. That loop must wake every 50 ms for each of 15 video
tracks and every 20 ms for each of 15 audio tracks, and it shares the GIL with a dozen or more busy worker threads.

A frame is late whenever the loop cannot run for about 40 ms. Nothing in the offline harness measures that.

## 4. What stalled the loop, and the fixes

Each fix is behind a flag. The default is today's behaviour.

### 4.1 Full garbage collections: 140 ms every ~70 s (`MUSETALK_GC_FREEZE=1`)

The largest stalls hit every stream at once, about every 70 s, as client gaps of about 200 ms. The GC log
(`MUSETALK_GC_LOG=1`) showed generation-2 collections of 138–147 ms at exactly those moments (35.5, 105.4, 175.2,
239.1 and 306.2 s at N=5; `evidence/01_gc_pauses_n5.txt`). Those account for 26 of the 85 client gaps over
100 ms in that run. Most of the rest came from §4.2.

A full collection scans every tracked object in the process while holding the GIL. With models, 15 avatars and
their compose plans loaded, that is about 470,000 objects.

The fix, `scripts/gc_tuning.py`, calls `gc.collect()` and then `gc.freeze()` after startup and after each avatar
load. This moves the long-lived heap into the permanent generation, and full collections then take about 20 ms.

### 4.2 `nvidia-smi` on the event loop: every turn and every health check (`MUSETALK_OFFLOOP_DIAGNOSTICS=1`)

With asyncio's slow-callback log at 40 ms, 74 of 78 loop blocks were HTTP handlers. The cause is
`_sample_live_gpu_stats()`, which runs `subprocess.run(["nvidia-smi", ...])` for 40–80 ms. It is reached from four
places that run on the event loop:
- the `🎬 WebRTC stream request … snapshot=` log line, at the start of every turn;
- `GET /stats`;
- `GET /health`;
- `GET /worker/state`, through `worker_control_plane.state_snapshot()` and then `_get_worker_metrics()`.

Each call freezes every stream. In production that means every turn start of every call, and every load-balancer
or control-plane health check.

With the flag on, all four run on a worker thread. The log line is printed when the snapshot is ready. Loop blocks
of 40 ms or more fell from 78 to 7, and gaps from 59 to 9 (N=5).

The control-plane heartbeat already sampled from its own thread.

### 4.3 Test rig: five viewers in one client process

With 5 viewers per client process, one viewer's join (peer-connection and DTLS setup) stalled the other viewers'
receive loop by about 85 ms. The server's send ring showed no gap at those moments.

`run_live15.sh` now runs one client process per viewer (`--peers-per-shard 1`). That models independent viewers
and costs 79 MB RSS per process.

### 4.4 An avatar's first session decodes its idle clip while others stream (`WEBRTC_IDLE_FRAME_CACHE_WARM=1`)

With the loop-lag logger and the 20 Hz CPU sampler on, every client gap over 100 ms at N=10 followed a server
loop stall within 200 ms (102 of 102). Neither CPU set was saturated: the client set peaked at 34% busy and the
server set at 42%.

A stall sampler (`MUSETALK_LOOP_STALL_DUMP_MS`) records every thread's stack when the loop is overdue. It showed
the loop in `select`, ready to run but waiting for the GIL, while 6–10 other threads were running Python. Over
22 stalls, the active threads were:

| thread | stalls where it was active |
|---|---|
| avatar PNG decode | 15/22 |
| GPU scheduler | 14/22 |
| encoders | 14/22 |
| compose | 13/22 |
| yuv convert | 10/22 |
| idle-frame-cache decode | 5/22 |

The bursts clustered at joins. An avatar's idle clip is decoded into the idle-frame cache only when that avatar's
first session is created, which means a GIL-heavy decode (240–700 ms of worker time) while the other calls
stream. Until the clip is ready, that session also decodes idle frames itself on the loop.

There was also a size problem. The cache budget (1536 MB) is smaller than 15 avatars' clips (2.17 GB). At 15
calls, some sessions would never be cached and would decode on the loop for the whole call.

The fix: `POST /avatars/{id}/cache/warm` now also builds that avatar's idle and pose clips, and waits for them when
`wait=true`. The budget is raised to 2400 MB. With this, N=10 went from 6/10 to 10/10.

### 4.5 Test rig: the tester's own polling was ~12% of the loop

Each client process polled `GET /webrtc/sessions/stats?view=lifetime` (every session's counters) twice a second.
At 15 viewers that is about 25 requests/s, whose replies are built on the event loop. It came to 10,594 requests
and 49 s of handler time in 7 minutes.

Each client now polls only its own session's `GET /webrtc/sessions/{id}/status`. That was 383 all-session requests
in the next run. The measurement's own server polling is now small, and no real browser client polls like this.

### 4.6 Every queued `av.VideoFrame` carries a reference cycle (`WEBRTC_QUEUE_PACKED_I420=1`)

At N=15, generation-2 collections came every ~10 s at 27–31 ms each. 307 of 329 client gaps over 100 ms started
within 200 ms after one; that run had the GC log on, which perturbs timing, so it is only used for attribution.

`MUSETALK_GC_GARBAGE_TYPES` samples what a collection frees. During streaming it was 60%
`av.video.format.VideoFormatComponent`, 20% `VideoFormat` and 20% tuples, about 3,000 formats per collection
at N=5.

PyAV gives every `VideoFrame` its own `VideoFormat`, whose components point back at it. That forms a reference
cycle that only the cyclic GC can free. Live frames wait in each session's FIFO for up to 20 s, which is 400 frames
because generation runs ahead of playback. That is long enough for the cycles to be promoted to generation 2.

The fix queues the packed I420 bytes instead. The handoff converter threads already produce them with the same
PyAV call. `recv()` builds the `av.VideoFrame` when it pops the frame.

Verification:
- **Real track, played frames:** `test_live_handoff_fifo.py` T10, and T7 run with the flag, show the frames
  `recv()` plays are SHA-identical and in order for the blocking, non-blocking and converter-thread handoffs.
- **Metadata and encoder output:** colour metadata is identical, and H.264 packets are byte-identical. VP8 packets
  cannot be compared, because libvpx realtime mode is not deterministic even for identical frames, run to run.
- **GC behaviour:** in a CPU FIFO simulation, generation-2 collections dropped from one per 8,000 frames (20,790
  objects) to none. Live at N=15, `VideoFormat` cycles reaching generation 2 fell from about 2,000–3,000 to
  about 180–370 per interval. The rest come from the frame in the encoder and the last frame each track holds.

This fix was not measured live on its own, because it landed together with the lighter polling of §4.5.

### 4.7 Remaining full collections still scan the live per-session heap (`MUSETALK_GC_THRESHOLDS=700,10,100`)

With cyclic garbage production now low, each generation-2 collection costs about 30 ms at 15 streams, almost all
of it scanning live objects. Making them 10 times rarer (the CPython default is 700, 10, 10) turned N=15 from
10/15 into 15/15, with a worst gap of 90 ms and no gaps over 100 ms.

Memory stayed flat: RSS peaked at 10.1 GB, against 10.0 GB without the change. The cyclic garbage no longer
holds frame buffers, since the `VideoFrame` itself is freed by reference counting.

### 4.8 How to read the client gaps (measurement)

aiortc's receiver emits frame N only when the first packet of frame N+1 arrives. So loopback latency is a constant
~54 ms, and a client gap reflects the send delay of the *next* frame. A server send 30–35 ms late, an 80–86 ms
send interval, shows up at the client as a 92–102 ms gap.

A browser completes a frame on its last packet, so it would see the server cadence plus network jitter. The report
labels these gaps `server_late`, and prints the server cadence beside P1–P3. It does not replace them.

### 4.9 Tried and rejected

| lever | result |
|---|---|
| GIL switch interval 1 ms (`MUSETALK_SWITCH_INTERVAL_MS=1`) | Worse: a 385 ms server send stall and a 413 ms client gap at N=10 |
| All diagnostics (GC log, loop-lag logger, stall sampler, garbage sampler) | Each one makes the stalls worse; for example 326 gaps instead of 31 at N=15 with the GC log. They were used only to attribute causes, and every pass/fail above comes from runs without them |

## 5. Other findings

- **The negotiated codec is VP8, not H.264.** `prefer_h264()` (`api_server.py`) runs after
  `setRemoteDescription()`. aiortc 1.14 negotiates codecs inside `setRemoteDescription`, so the preference never
  applies, and the answer takes the first codec in the offer. For aiortc and Chrome that is VP8.
  - The "WebRTC H.264 impl=aiortc libx264" startup line therefore does not describe what is sent.
  - Every live number here is for aiortc's VP8 encoder. Neither the repo's browser player nor the client-validation
    pages set codec preferences.
  - Fixing the order (prefer before `setRemoteDescription`) would switch production to H.264. That is a behaviour
    change, so it is left for a decision.
- **Memory per call is about 270 MB.** Round-robin generation runs ahead of playback until the strict FIFO holds
  400 frames (20 s) of up to 688 KB each.
  - At 15 calls that is about 4 GB, on top of 4.3 GB after warm-up and 2.2 GB of idle clips.
  - The EDF policy (`HLS_SCHEDULER_POLICY=edf`, 5 s run-ahead cap) would cut the queues about 4×, but it has
    never run live.
- **Diagnostics that are safe to leave in production:** none of the new diagnostic flags. They all perturb timing.
  The server's built-in 1 Hz loop-lag monitor underestimates short stalls, because it saw a maximum of 23 ms while
  sends were 36 ms late.

## 6. 60-minute soak at 15 streams

Run `20260929T093219Z_B_final_soak`, final configuration:
- 15 viewers join 20 s apart, then each speaks continuously for 55 minutes.
- The level held 3,596 s of steady state, generating 230.6 fps.
- Server fresh fraction was 0.9965 or higher, with no held run longer than 1 frame.
- The timeline is in [`soak_timeline_n15.md`](soak_timeline_n15.md), in 5-minute windows.

| window (s after start) | client gaps > 100 ms | worst client gap | worst 1 s window (client) | server largest send interval | server RSS |
|---|---|---|---|---|---|
| 0–900 | 0 | 99.4 ms | 19 | 85 ms | 9.6–10.4 GB |
| 900–1500 | 5 | 114.9 ms | 19 | 95 ms | 10.4 GB |
| 1500–1800 | 11 | 279 ms (observer, see below) | 15 | **130.5 ms** | 10.4 GB |
| 1800–2700 | 11 | 873 ms (observer) | 4 (observer) | 89 ms | 10.5 GB |
| 2700–3000 | 20 | 142.4 ms | 18 | **135.6 ms** | 10.6 GB |
| 3000–3600 | 2 | 102.3 ms | 19 | 86 ms | 10.6 GB |

**Server-wide stalls.** There were two, at about 1800 s and 2985 s. The server's own loop-lag monitor recorded
83 ms and 80 ms. Every stream's send interval was 114–136 ms at those moments, which is where the server cadence
drops to 19. Their cause is not established, because this run had no diagnostics (they perturb timing, §4.9).
Candidates:
- A full GC collection over a live heap that grows during the hour. Server RSS rose steadily from 9.6 to 10.6 GB,
  about 17 MB/min.
- Something coinciding with the observer start at 1800 s.

**Measurement artefacts.**
- The two observer recorders (streams 0 and 14) stalled their own client processes for 265 ms and 822 ms when
  recording started at 1800 s and 2080 s. That caused the 279 ms and 873 ms gaps and the 4-frame window on those
  two streams.
- Client 3 stalled for 150 ms at 2217 s.

None of these is server-side.

**Observer videos.** `tmp/live15_r5/20260929T093219Z_B_final_soak/traces/soak/n15/s{00,14}_observer.mp4` are
90 s each at 512×896, with pts equal to arrival time. They show what those viewers saw from 30 minutes in,
including the recorder-start glitch.

**Memory.** Server RSS grew by 1 GB over the hour and MemAvailable bottomed at 4.4 GB. For a longer call, or more
calls, this growth needs a cause before it can be ruled out as a leak.

**Verdict.** 15 concurrent streams are not yet buffer-free for an hour by the platform's own definition. Each
stall is a single ~130 ms hitch on every stream, and there were two in 60 minutes.

## 7. Recommendations

1. **Enable the `loopfix.env` flags in production.** One caveat: `MUSETALK_GC_FREEZE` also freezes each loaded
   avatar. An evicted avatar's memory is still freed by reference counting, but any reference cycles in it are not,
   and this test never evicted one. The first five fix real production stalls regardless of
   stream count:
   - `MUSETALK_GC_FREEZE=1`
   - `MUSETALK_OFFLOOP_DIAGNOSTICS=1`
   - `WEBRTC_IDLE_FRAME_CACHE_WARM=1`, together with an idle-frame-cache budget that holds every served avatar's clips
   - `WEBRTC_QUEUE_PACKED_I420=1`
   - `MUSETALK_GC_THRESHOLDS=700,10,100`

   Each keeps frames identical. Rollback is deleting the line.
2. **Keep calling `POST /avatars/{id}/cache/warm` before routing a call to a worker**, as the control plane already
   does. It now also builds the idle clips.
3. **Plan 10 calls per RTX 4070 SUPER worker** until the 15-stream soak passes. At 15, 5-minute runs are at the
   edge, and the hour-long soak had two ~130 ms hitches (§6). Before raising the number, find the cause of the two
   ~80 ms stalls and of the 1 GB/h RSS growth.
4. **Decide on the codec (§5).** H.264 would change encoder cost and client compatibility, and it needs its own
   live run.
5. **To gain headroom beyond 15**, move per-stream media work (encode, RTP, pacing) out of the single Python
   process, for example into per-call media workers. Every stall found here is one process-wide GIL or GC pause
   hitting all calls at once.

## 8. Not done

- **Arm A and Arm C live runs.** All runs used Arm B. The attribution above comes from one fix at a time on Arm B.
  Arm C (pipeline depth 2 + EDF) would also cap per-call memory.
- **A soak at 10 streams, and a diagnostic soak at 15** to name the two ~80 ms stalls and the RSS growth.
- **A real-browser client.** The tester is aiortc. §4.8 explains why a browser should see slightly fewer late
  frames, but this was not measured.
- **H.264 live.** See §5.

## 9. Files

**Server changes.** All are flag-gated and default off:

| file | change |
|---|---|
| `scripts/gc_tuning.py` (new) | `MUSETALK_GC_FREEZE`, `MUSETALK_GC_THRESHOLDS`, `MUSETALK_GC_LOG`; diagnostics `MUSETALK_GC_GARBAGE_TYPES`, `MUSETALK_ASYNCIO_SLOW_MS`, `MUSETALK_LOOP_LAG_LOG_MS`, `MUSETALK_LOOP_STALL_DUMP_MS`, `MUSETALK_SWITCH_INTERVAL_MS` |
| `api_server.py` | GC hooks; off-loop snapshot, `/stats`, `/health` and `/worker/state`; idle clips built at avatar warm |
| `scripts/avatar_cache.py` | freeze after each avatar load |
| `scripts/webrtc_media_flags.py` | registers `MUSETALK_OFFLOOP_DIAGNOSTICS`, `WEBRTC_IDLE_FRAME_CACHE_WARM` and `WEBRTC_QUEUE_PACKED_I420` |
| `scripts/webrtc_live_handoff.py`, `scripts/webrtc_tracks.py` | the packed-I420 FIFO |

**Test rig.**

| file | change |
|---|---|
| `load_test_webrtc_v2.py` | staggered real joins, duration-bound turns, per-frame traces, the send ring dump, observers, POST retries, per-session shard polling |
| `experiments/live15_r5/` | the driver, override files, trace report, samplers, `soak_timeline.py`, `summarize_runs.py`, `extract_evidence.py`, `archive_runs.sh` and the runbook (`experiments/live15_r5/README.md`) |
| `docs/fps_comparisons/4070s_300fps_impl_20260928/webrtc/test_live_handoff_fifo.py` | new case T10 |
| `…/test_load_harness_e2e.py` | the fake server gains `/status` |
| `api_server.py` (test support) | `GET /webrtc/sessions/{id}/status?light=1` returns only the turn state; the default reply is unchanged |

**Checks run.**

| check | result |
|---|---|
| resolver unit tests | 66 OK |
| `load_test_webrtc_v2.py --selftest` | 6/6 |
| harness end-to-end against the fake server | 6/6 |
| `test_live_handoff_fifo.py` with T10 | 11/11 |
