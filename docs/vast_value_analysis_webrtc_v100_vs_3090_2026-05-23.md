# Vast.ai Value Analysis: V100 32GB vs RTX 3090 for WebRTC

Date: 2026-05-23

## Summary

For the current MuseTalk WebRTC workload, the best bang-for-buck on Vast.ai is
still the RTX 3090 when it is available around `$0.16-$0.18/hr`.

The V100 32GB batch-24 profile is technically strong: it completed both 5 and 8
concurrent streams and beat the RTX 3090 reference on average frame cadence. But
the current V100 32GB on-demand price is higher, and the throughput gain is not
large enough to offset the price gap.

Practical recommendation:

- Use RTX 3090 for cost-efficient WebRTC serving.
- Use V100 32GB only when the price drops near the break-even points below, or
  when the extra 32GB VRAM is valuable for experiments.
- Do not use V100 batch 28 as the preferred production profile; batch 24 is
  better on both throughput and VRAM.

## Pricing Snapshot

Vast.ai is a marketplace, so prices fluctuate with host supply, demand,
reliability, geography, storage, and bandwidth. Vast recommends checking current
rates in the dashboard or querying with the CLI/API.

Sources:

- Vast pricing docs: `https://docs.vast.ai/guides/instances/pricing`
- Vast search offers API: `https://docs.vast.ai/api-reference/search/search-offers`

Live CLI queries used:

```bash
vastai search offers \
  'gpu_name=RTX_3090 num_gpus=1 rentable=true verified=true reliability>0.98' \
  --raw --limit 20 -o 'dph_total'

vastai search offers \
  'gpu_name in ["Tesla_V100", "V100"] num_gpus=1 rentable=true verified=true reliability>0.98' \
  --raw --limit 20 -o 'dph_total'

vastai search offers \
  'gpu_name=RTX_4090 num_gpus=1 rentable=true verified=true reliability>0.98' \
  --raw --limit 10 -o 'dph_total'

vastai search offers \
  'gpu_name in ["RTX_A6000", "RTX_6000_Ada", "A40", "L40S"] num_gpus=1 rentable=true verified=true reliability>0.98' \
  --raw --limit 20 -o 'dph_total'
```

Current relevant on-demand ranges observed:

| GPU | Current good on-demand range | Notes |
| --- | ---: | --- |
| RTX 3090 24GB | `~$0.161-$0.20/hr` | Many offers; cheapest good observed offer was about `$0.161/hr` |
| Tesla V100 32GB | `~$0.208-$0.221/hr` | Fewer offers; 32GB VRAM; same class as tested server |
| RTX 4090 24GB | `~$0.28/hr` cheapest; more commonly `~$0.40/hr+` | Untested for this stack |
| A40 46GB | `~$0.321/hr+` | More VRAM; likely not best unless VRAM-bound |
| L40S 46GB | `~$1.20/hr+` | Too expensive for this workload unless performance is far better |

Example live offers:

| GPU | Offer price | Reliability | Notes |
| --- | ---: | ---: | --- |
| RTX 3090 | `$0.161/hr` | `0.9979` | Nevada, US; 24GB VRAM |
| RTX 3090 | `$0.166/hr` | `0.9822` | California, US; 24GB VRAM |
| RTX 3090 | `$0.171/hr` | `0.9959` | Czechia; 24GB VRAM |
| Tesla V100 32GB | `$0.208/hr` | `0.9912` | Utah, US; 32GB VRAM |
| Tesla V100 32GB | `$0.221/hr` | `0.9994+` | Minnesota, US; 32GB VRAM |
| RTX 4090 | `$0.280/hr` | `0.9829` | Cheapest observed 4090; PCIe bandwidth looked weak |
| A40 | `$0.321/hr` | `0.9986` | 46GB VRAM |

## Performance Inputs

All performance numbers below come from the saved local WebRTC load-test reports.

V100 batch 24:

- 5 streams: `load_test_webrtc_report_v100_20_20_5streams_batch24only_libx264_20260523.json`
- 8 streams: `load_test_webrtc_report_v100_20_20_8streams_batch24only_test_avatar_2_libx264_20260523.json`

V100 batch 28:

- 5 streams: `load_test_webrtc_batch28_only_5.json`
- 8 streams: `load_test_webrtc_v100_batch28_8streams_test_avatar2_20260523.json`

RTX 3090 reference:

- Clean diagnostic run: `load_test_webrtc_report_20_20_4_5_6_8streams_8_16_diagnostics_libx264.json`

Important test-shape caveats:

- All comparisons are `20/20 fps` WebRTC using `libx264`.
- V100 batch 24 used a sparse `HLS_SCHEDULER_MAX_BATCH=24`,
  `HLS_SCHEDULER_FIXED_BATCH_SIZES=24`, and
  `MUSETALK_TRT_STAGEWISE_WARMUP_BATCHES=24` profile.
- V100 batch 28 used a batch-28-only profile.
- RTX 3090 reference used `throughput_record`, TRT warmups `8,16`, and request
  `batch_size=8`.
- V100 8-stream batch-24 used `test_avatar_2`; V100 5-stream batch-24 and RTX
  reference used `test_avatar`.

## Cost Model

Definitions:

```text
approx_fps_per_stream = 1 / avg_frame_interval_s
aggregate_fps = completed_streams * approx_fps_per_stream
cost_per_stream_hour = hourly_price / completed_streams
cost_per_aggregate_fps_hour = hourly_price / aggregate_fps
```

Prices used for the primary comparison:

- RTX 3090: `$0.161/hr`
- V100 32GB: `$0.208/hr`

These use the cheapest good observed on-demand offers from the live Vast
snapshot. If prices move, recompute the final two cost columns.

## 5 Concurrent Streams

| GPU/Profile | Completed | Avg FPS/stream | Aggregate FPS | Price/hr | Cost per stream-hr | Cost per aggregate FPS-hr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V100 batch 24 | `5/5` | `11.36` | `56.8` | `$0.208` | `$0.0416` | `$0.00366` |
| V100 batch 28 | `5/5` | `9.43` | `47.2` | `$0.208` | `$0.0416` | `$0.00441` |
| RTX 3090 | `5/5` | `9.09` | `45.5` | `$0.161` | `$0.0323` | `$0.00355` |

Detailed performance:

| GPU/Profile | Avg live-ready | Avg frame interval | Max frame interval | Wall time | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| V100 batch 24 | `10.226s` | `0.088s` | `1.874s` | `43.8s` | `25724 MB` |
| V100 batch 28 | `10.898s` | `0.106s` | `2.668s` | `52.6s` | `29439 MB` |
| RTX 3090 | `5.534s` | `0.110s` | `1.688s` | `46.3s` | `24145 MB` |

Read:

- V100 batch 24 has the best raw cadence at 5 streams.
- RTX 3090 reaches live much faster and has the lowest cost per stream-hour.
- RTX 3090 slightly wins cost per aggregate FPS-hour because it is cheaper.
- V100 batch 28 is dominated by V100 batch 24: slower, more VRAM, worse tail.

## 8 Concurrent Streams

| GPU/Profile | Completed | Avg FPS/stream | Aggregate FPS | Price/hr | Cost per stream-hr | Cost per aggregate FPS-hr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| V100 batch 24 | `8/8` | `6.06` | `48.5` | `$0.208` | `$0.0260` | `$0.00429` |
| V100 batch 28 | `8/8` | `5.41` | `43.2` | `$0.208` | `$0.0260` | `$0.00481` |
| RTX 3090 | `8/8` | `5.71` | `45.7` | `$0.161` | `$0.0202` | `$0.00353` |

Detailed performance:

| GPU/Profile | Avg live-ready | Avg frame interval | Max frame interval | Wall time | Peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| V100 batch 24 | `9.230s` | `0.165s` | `5.640s` | `70.9s` | `25892 MB` |
| V100 batch 28 | `8.956s` | `0.185s` | `9.192s` | `79.2s` | `29623 MB` |
| RTX 3090 | `10.407s` | `0.175s` | `3.698s` | `75.1s` | `23901 MB` |

Read:

- V100 batch 24 has the best average cadence at 8 streams.
- RTX 3090 has better worst-case jitter and much better price efficiency.
- V100 batch 28 completes, but it is worse than both V100 batch 24 and RTX 3090.
- None of these profiles is true smooth `20 fps` at 8 streams. The target frame
  interval for 20 fps is `0.050s`; the best observed 8-stream profile averaged
  `0.165s`.

## Break-Even Pricing

At current RTX 3090 price of about `$0.161/hr`, V100 must be at or below these
prices to match RTX 3090 cost efficiency:

| Scenario | V100 break-even price |
| --- | ---: |
| 5 streams, batch 24 | `~$0.202/hr` |
| 8 streams, batch 24 | `~$0.171/hr` |
| 5 streams, batch 28 | `~$0.167/hr` |
| 8 streams, batch 28 | `~$0.153/hr` |

Current V100 32GB listings around `$0.208-$0.221/hr` miss the 8-stream
break-even by a meaningful margin.

## Recommendation

Best bang-for-buck today:

```text
RTX 3090 at about $0.16-$0.18/hr
```

Best tested V100 profile:

```text
V100 32GB batch 24 only
```

Avoid for production:

```text
V100 batch 28
```

Decision rules:

- Rent RTX 3090 when you want the best cost per completed stream and cost per
  delivered frame.
- Rent V100 32GB if the price drops to roughly `$0.17/hr` or below for 8-stream
  work, or if the extra 32GB VRAM is needed for further batch/engine experiments.
- Prefer V100 batch 24 over V100 batch 28. Batch 24 used less VRAM and performed
  better in both 5-stream and 8-stream tests.
- Consider RTX 4090 only after direct testing. The cheapest observed 4090 was
  around `$0.28/hr`, so it would need to deliver roughly `1.74x` the RTX 3090
  throughput to match RTX 3090 value at `$0.161/hr`.
- A40/A6000/L40S should not be first choices for this workload unless future
  testing proves the extra VRAM unlocks much higher concurrency.

## Operational Notes

Instance selection should also consider:

- Reliability score: prefer `>0.98`, ideally `>0.99`.
- Network: WebRTC needs decent upload, not just download.
- CPU and RAM: libx264 encode and WebRTC session management can become CPU-heavy.
- PCIe bandwidth: avoid very weak PCIe bandwidth listings when possible.
- Port availability: wall testing needs public ports.
- Bandwidth fees: Vast charges bandwidth separately and rates vary by host.
- Storage fees: stopped instances can still accrue storage costs until deleted.
