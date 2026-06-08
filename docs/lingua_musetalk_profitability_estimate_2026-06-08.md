# Lingua MuseTalk Profitability Estimate - 2026-06-08

This document updates the Lingua cost estimate using the measured MuseTalk
WebRTC throughput from the current RTX 6000 Ada server.

The product assumption is a CallAnnie-like language companion: paying users can
have realtime spoken conversations with an AI companion and optionally see a
live talking avatar.

## Bottom Line

The current measured MuseTalk production-quality capacity is:

```text
4 smooth concurrent WebRTC sessions per RTX 6000 Ada at 20 fps
```

That is lower than the older Lingua planning estimate of roughly `8 stable
concurrent live streams per RTX 3090`. The older number should not be used for
strict 20 fps subscription economics.

At current Vast on-demand pricing, the app can be profitable at `19.99/month`,
but only if live video minutes are budgeted and admission-controlled. The main
business risk is not fixed GPU cost at small scale; it is offering unlimited
live video minutes to heavy users while GPU utilization stays low.

Recommended starting commercial shape:

- free plan with `2` live-avatar minutes/day, no rollover
- `$19.99/month` core plan
- include a capped live-avatar allowance of `600` minutes/month on Core
- sell a higher unlimited/fair-use tier around `$39.99/month`
- meter or upsell additional live-avatar minutes
- keep text/audio-only conversation available after the video allowance is used
- scale MuseTalk from peak concurrent live sessions, not total paid users

## Measured MuseTalk Capacity

Use the optimized RTX 6000 Ada report as the source of truth:

- Report: `load_test_webrtc_rtx6000ada_int8_trt_unet_split8_20fps_20260608.md`
- GPU: RTX 6000 Ada, 48 GB
- VAE: five-stage INT8 TensorRT
- UNet: FP16 TensorRT split8
- Recommended buckets: `8,16`

Measured strict 20 fps result:

| Streams | Completed | Avg interval | Aggregate FPS | Smooth 20 fps |
| ---: | ---: | ---: | ---: | :--- |
| 4 | `4/4` | `0.051s` | `78.4` | Yes |
| 8 | `8/8` | `0.070s` | `114.3` | No |
| 12 | `12/12` | `0.105s` | `114.3` | No |
| 16 | `16/16` | `0.143s` | `111.9` | No |
| 20 | `20/20` | `0.179s` | `111.7` | No |

Practical read:

- Production-quality live avatar capacity: `4` smooth 20 fps calls per RTX
  6000 Ada.
- Planning-safe capacity for a hard SLA: `3` calls per GPU.
- Degraded/non-premium mode could allow more sessions at lower frame cadence,
  but that should not be advertised as smooth FaceTime-style video.

The `4,8,16` bucket follow-up did not materially increase FPS and raised VRAM,
so `8,16` remains the recommended serving profile.

## Current Vast Price Snapshot

Queried live through the Vast `search offers` API on 2026-06-08 with:

```text
type=on-demand
verified=true
rentable=true
rented=false
num_gpus=1
reliability2>=0.98
allocated_storage=100 GB
order=dph_total asc
```

Vast is a realtime marketplace, so these are a snapshot, not a durable quote.

| GPU | Low current offer | Top-5 avg current offers | 24/7 monthly at top-5 avg | Notes |
| --- | ---: | ---: | ---: | --- |
| RTX 6000 Ada | `$0.689/hr` | `$0.731/hr` | `$534/mo` | Measured MuseTalk target |
| RTX 3090 | `$0.188/hr` | `$0.205/hr` | `$150/mo` | TTS proxy and scale-out candidate |
| RTX 4090 | `$0.428/hr` | `$0.465/hr` | `$340/mo` | 24 GB; not this measured path |
| RTX A6000 | `$0.428/hr` | `$0.501/hr` | `$366/mo` | 48 GB candidate, must benchmark |
| L40S | `$0.619/hr` | `$0.756/hr` | `$552/mo` | 48 GB candidate, must benchmark |

For this estimate, use:

```text
MuseTalk GPU: RTX 6000 Ada at $0.731/hr = $534/month
TTS GPU: RTX 3090 at $0.205/hr = $150/month
AWS control plane: $30/month
```

Current conservative fixed stack:

| Component | Monthly cost |
| --- | ---: |
| 1 RTX 6000 Ada MuseTalk worker | `$534` |
| 1 RTX 3090 TTS worker | `$150` |
| AWS control plane | `$30` |
| Total | `$714/mo` |

Live-video-only lower bound:

| Component | Monthly cost |
| --- | ---: |
| 1 RTX 6000 Ada MuseTalk worker | `$534` |
| AWS control plane | `$30` |
| Total | `$564/mo` |

## RTX 3090 Scale-Out Note

The RTX 3090 is much cheaper than the RTX 6000 Ada in the current Vast snapshot,
but the older Lingua assumption of `8` stable concurrent RTX 3090 streams should
not be used for smooth `20 fps` pricing.

The strongest local RTX 3090 MuseTalk markdown result used five-stage VAE INT8
plus TRT UNet split8 on a 300W RTX 3090 host. It reached roughly `71-72`
aggregate FPS. That is good for an estimated `3` smooth `20 fps` sessions per
healthy RTX 3090, not `4+` strict sessions, because C4 needs `80` aggregate FPS.

At the current top-5 Vast planning prices:

- `1 x RTX 6000 Ada`: `$0.731/hr`, `4` measured smooth sessions,
  about `$0.183` per smooth slot-hour.
- `1 x RTX 3090`: `$0.205/hr`, `3` planning smooth sessions,
  about `$0.068` per smooth slot-hour.
- `3 x RTX 3090`: `$0.615/hr`, about `9` planning smooth sessions,
  cheaper than `1 x RTX 6000 Ada` while offering more total smooth capacity.

So the best production shape is likely one RTX 6000 Ada as the always-on
known-good baseline, then RTX 3090 workers for scale-out after the same current
optimized code path is rebenchmarked on a healthy uncapped RTX 3090 host. The
detailed 3090-vs-6000 Ada planning table is in
`docs/lingua_pricing_tier_recommendation_2026-06-08.md`.

## Revenue And Variable AI Cost

Subscription:

```text
price = $19.99/month
```

Net revenue after app store fee:

| Store fee | Net revenue/user |
| ---: | ---: |
| 30% | `$13.99` |
| 15% | `$16.99` |

Current repo variable AI cost from the Lingua estimate remains a reasonable
planning band:

| User type | Variable AI cost/month |
| --- | ---: |
| Text-first active user | `$0.50-$2.00` |
| Voice-active current user | `$1.00-$4.00` |

For a spoken companion app, use the voice-active band until the runtime stack is
fully local.

Approximate contribution per user before fixed GPU costs:

| Store fee | Variable AI cost | Contribution/user |
| ---: | ---: | ---: |
| 30% | `$1` | `$12.99` |
| 30% | `$4` | `$9.99` |
| 15% | `$1` | `$15.99` |
| 15% | `$4` | `$12.99` |

## Break-Even Users

Using the current conservative fixed stack of `$714/month`:

| Case | Break-even users |
| --- | ---: |
| 30% store fee, `$4` variable AI | `72` |
| 30% store fee, `$1` variable AI | `55` |
| 15% store fee, `$4` variable AI | `55` |
| 15% store fee, `$1` variable AI | `45` |

Using the live-video-only lower bound of `$564/month`:

| Case | Break-even users |
| --- | ---: |
| 30% store fee, `$4` variable AI | `57` |
| 30% store fee, `$1` variable AI | `44` |
| 15% store fee, `$4` variable AI | `44` |
| 15% store fee, `$1` variable AI | `36` |

Practical read: the app does not need a huge subscriber base to cover one
always-on MuseTalk worker. The business becomes fragile only when heavy live
video usage forces many additional GPUs or when users are allowed unlimited
video minutes without a price tier.

## Video Minute Economics

For the measured RTX 6000 Ada path:

```text
cost_per_live_video_minute =
  gpu_hourly_cost / (smooth_sessions_per_gpu * 60 * utilization)
```

At `4` smooth sessions/GPU:

| GPU utilization | Video GPU cost/min |
| ---: | ---: |
| 100% | `$0.0030` |
| 50% | `$0.0061` |
| 25% | `$0.0122` |
| 10% | `$0.0305` |

At a safer `3` sessions/GPU:

| GPU utilization | Video GPU cost/min |
| ---: | ---: |
| 100% | `$0.0041` |
| 50% | `$0.0081` |
| 25% | `$0.0162` |
| 10% | `$0.0406` |

Per-user video GPU cost at `4` sessions/GPU:

| Live video minutes/user/month | 50% utilization | 25% utilization |
| ---: | ---: | ---: |
| 100 | `$0.61` | `$1.22` |
| 180 | `$1.10` | `$2.19` |
| 300 | `$1.83` | `$3.65` |
| 600 | `$3.65` | `$7.31` |
| 1000 | `$6.09` | `$12.18` |
| 1500 | `$9.14` | `$18.27` |

This is why `600` included video minutes is a reasonable but aggressive
`$19.99` allowance. It is materially better than `300` minutes while still
leaving contribution margin at normal utilization. `1000+` minutes should move
to the higher unlimited/fair-use tier.

Free users at `2` minutes/day can use up to `60` live-avatar minutes/month. At
current GPU prices, that is about `$0.37-$0.73` in video GPU cost per fully
active free user at `50%-25%` utilization before STT/LLM/TTS cost. This is fine
as a conversion engine, but it needs no rollover, abuse controls, and lower
priority than paid traffic.

## Peak Concurrency Model

Do not size servers from total subscribers. Size from peak concurrent live
sessions.

Use:

```text
peak_live_concurrency =
  paid_users
  * peak_day_live_usage_rate
  * avg_live_minutes_on_peak_day
  * burst_factor
  / peak_window_minutes
```

For a `5 hour` peak window:

```text
peak_window_minutes = 300
```

Measured GPU requirement:

```text
musetalk_gpus = ceil(peak_live_concurrency / 4)
```

Hard-SLA planning:

```text
musetalk_gpus = ceil(peak_live_concurrency / 3)
```

## Autoscaled Scenario Model

Assume:

- 1 base MuseTalk RTX 6000 Ada always on
- 1 base TTS RTX 3090 always on
- AWS core always on
- extra MuseTalk GPUs run for `5 peak hours/day`
- off-peak pool averages `30%` of peak size

Extra MuseTalk GPU monthly average:

```text
$534 * (5/24 + 19/24 * 0.30) = about $238/month
```

If every extra GPU is kept on 24/7, use `$534/month` per extra GPU instead.

### Moderate Live Usage

Assumptions:

- `20%` of paying users use live video on a peak day
- average peak-day live use = `30 min`
- burst factor = `2`
- peak window = `5 hours`

This gives:

```text
peak_live_concurrency = paid_users * 0.04
```

Profit range:

- low end: 30% store fee plus `$4/user/month` variable AI
- high end: 15% store fee plus `$1/user/month` variable AI

| Paying users | Peak live concurrency | MuseTalk GPUs | Fixed monthly cost | Profit range |
| ---: | ---: | ---: | ---: | ---: |
| 50 | `2` | `1` | `$714` | `-$214` to `$86` |
| 100 | `4` | `1` | `$714` | `$286` to `$886` |
| 250 | `10` | `3` | `$1,189` | `$1,309` to `$2,809` |
| 500 | `20` | `5` | `$1,665` | `$3,331` to `$6,331` |
| 1000 | `40` | `10` | `$2,855` | `$7,138` to `$13,137` |

### Heavy Live Usage

Assumptions:

- `30%` of paying users use live video on a peak day
- average peak-day live use = `60 min`
- burst factor = `2`
- peak window = `5 hours`

This gives:

```text
peak_live_concurrency = paid_users * 0.12
```

| Paying users | Peak live concurrency | MuseTalk GPUs | Fixed monthly cost | Profit range |
| ---: | ---: | ---: | ---: | ---: |
| 50 | `6` | `2` | `$951` | `-$452` to `-$152` |
| 100 | `12` | `3` | `$1,189` | `-$190` to `$410` |
| 250 | `30` | `8` | `$2,379` | `$120` to `$1,619` |
| 500 | `60` | `15` | `$4,044` | `$952` to `$3,952` |
| 1000 | `120` | `30` | `$7,613` | `$2,380` to `$8,379` |

Read:

- Moderate usage is healthy above roughly `100` paying users.
- Heavy usage is fragile at small user counts and needs either higher pricing,
  video-minute caps, or excellent utilization.
- Once there are hundreds of subscribers, the model can work even with heavy
  live usage, but only if autoscaling and session admission are disciplined.

## Recommended Pricing Guardrails

The base `19.99/month` plan should not be unlimited live avatar time, but it can
be more generous than `300` minutes.

Suggested starter packaging:

| Plan | Price | Included live avatar minutes | Notes |
| --- | ---: | ---: | --- |
| Free | `$0` | `2/day`, no rollover | real trial, lower-priority capacity |
| Lite | `$9.99` | `180` | optional cheap paid tier |
| Core | `$19.99` | `600` | main plan |
| Unlimited | `$39.99` | `1500 priority minutes` | fair-use after allowance |
| Add-on | `$3-$5` | `100` extra minutes | keeps marginal cost covered |

At current GPU prices, `100` extra minutes costs about:

- `$0.61` at 50% GPU utilization
- `$1.22` at 25% GPU utilization

An add-on priced at `$3-$5` has room for video GPU cost, API cost, payment fees,
and margin.

## Operational Recommendations

1. Keep `4` smooth sessions/GPU as the current measured admission limit.
2. Use `3` sessions/GPU if you want SLA headroom.
3. Keep the optimized RTX 6000 Ada serving profile at `8,16`.
4. Treat C8/C12/C16 as degraded completion/stress modes, not premium smooth
   call capacity.
5. Keep the free tier at `2` live-avatar minutes/day with no rollover.
6. Put free sessions behind paid sessions in admission priority.
7. Track live-avatar minutes per user and per subscription tier.
8. Route users to text/audio-only mode when video capacity is full or allowance
   is exhausted.
9. Rebenchmark RTX A6000, L40S, and A40 before using them for production
   economics. Their Vast prices can be lower than RTX 6000 Ada, but throughput
   may also be lower.
10. Keep interruptible Vast instances out of live calls unless you can tolerate
   session drops. They are useful for batch jobs or warm spare experiments.

## Sources

- Local MuseTalk measurement:
  `load_test_webrtc_rtx6000ada_int8_trt_unet_split8_20fps_20260608.md`
- Lingua planning estimate:
  `/root/.codex/attachments/a3291375-8343-4aea-8fd0-ab49fc966844/pasted-text.txt`
- Vast pricing docs:
  `https://docs.vast.ai/guides/instances/pricing`
- Vast search offers API:
  `https://docs.vast.ai/api-reference/search/search-offers`
- OpenAI pricing:
  `https://platform.openai.com/docs/pricing/`
- GPT-4.1 mini model pricing:
  `https://platform.openai.com/docs/models/gpt-4.1-mini`
- GPT-4o transcribe model pricing:
  `https://platform.openai.com/docs/models/gpt-4o-transcribe`
- GPT-4o mini TTS model pricing:
  `https://developers.openai.com/api/docs/models/gpt-4o-mini-tts`
