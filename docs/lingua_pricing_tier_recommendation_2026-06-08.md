# Lingua Pricing Tier Recommendation - 2026-06-08

This document turns the MuseTalk profitability estimate into a concrete pricing
recommendation for a CallAnnie-like language companion app.

Source economics:

- MuseTalk profitability model:
  `docs/lingua_musetalk_profitability_estimate_2026-06-08.md`
- Measured MuseTalk live-avatar capacity:
  `4` smooth concurrent WebRTC sessions per RTX 6000 Ada at `20 fps`
- Current recommended MuseTalk server profile:
  `VAE INT8 + TRT UNet split8`, buckets `8,16`
- Current Vast planning price:
  RTX 6000 Ada at about `$0.731/hr`, or `$534/month` if run 24/7
- Current Vast RTX 3090 planning price:
  about `$0.205/hr`, or `$150/month` if run 24/7
- Conservative fixed stack:
  `1` RTX 6000 Ada MuseTalk worker + `1` RTX 3090 TTS worker + AWS =
  about `$714/month`

## Recommendation

Do not sell truly unlimited live-avatar video on the `$19.99` plan.

Use:

1. a real free trial allowance;
2. an optional cheap paid plan;
3. a generous `$19.99` core plan;
4. an unlimited/fair-use plan.

The best starting lineup:

| Tier | Price | Included live-avatar minutes | Best for |
| --- | ---: | ---: | --- |
| Free | `$0` | `2 min/day`, no rollover | real product trial |
| Lite | `$9.99/mo` | `180` | price-sensitive learners |
| Core | `$19.99/mo` | `600` | default paid plan |
| Unlimited | `$39.99/mo` | `1500 priority minutes`, then fair-use fallback | serious daily users |

If the product must launch with only two paid tiers:

| Tier | Price | Included live-avatar minutes | Notes |
| --- | ---: | ---: | --- |
| Core | `$19.99/mo` | `600` | default paid plan |
| Unlimited | `$39.99/mo` | `1500 priority minutes`, fair-use after | best starting unlimited price |

In that two-paid-tier version, the free plan replaces the cheap paid acquisition
tier. That is cleaner than asking users to pay `$9.99` for a small allowance
when free users already get `2` minutes per day.

## Why Unlimited Needs A Fair-Use Boundary

The current measured live avatar limit is:

```text
4 smooth concurrent 20 fps sessions per RTX 6000 Ada
```

That means one GPU can produce:

```text
4 sessions * 60 minutes * 730 hours = 175,200 smooth live-avatar minutes/month
```

That number assumes perfect 100% utilization, which real subscription products
do not get. More realistic usable capacity:

| GPU utilization | Smooth live-avatar minutes/month/GPU |
| ---: | ---: |
| 50% | `87,600` |
| 25% | `43,800` |
| 10% | `17,520` |

This is the core business risk: free daily usage and very heavy paid usage can
consume a lot of live video capacity if admission is not metered.

Free users at `2` minutes/day can use up to:

```text
2 min/day * 30 days = 60 live-avatar minutes/month
```

That is a good trial. It is also real GPU usage, so it needs no rollover,
per-account/device abuse controls, and lower-priority admission when paid demand
is high.

## Video Cost Per User

Using the current RTX 6000 Ada Vast estimate:

| Live-avatar minutes/user/month | Video GPU cost at 50% utilization | Video GPU cost at 25% utilization |
| ---: | ---: | ---: |
| 60 | `$0.37` | `$0.73` |
| 100 | `$0.61` | `$1.22` |
| 180 | `$1.10` | `$2.19` |
| 300 | `$1.83` | `$3.65` |
| 600 | `$3.65` | `$7.31` |
| 1000 | `$6.09` | `$12.18` |
| 1500 | `$9.14` | `$18.27` |

This table only covers live avatar GPU cost. It does not include app-store fees,
OpenAI/STT/LLM variable cost, TTS, storage, bandwidth, support, failed/idle
capacity, or profit.

## RTX 3090 Pool vs RTX 6000 Ada

The RTX 6000 Ada result is the newest measured result for the current optimized
path. The strongest RTX 3090 result in the local markdowns used five-stage VAE
INT8 plus TRT UNet split8 on a 300W RTX 3090 host and reached about `71-72`
aggregate FPS. That run completed C4/C6/C8, but C4 still missed the strict
`80` aggregate FPS target for four smooth `20 fps` sessions. For pricing and
capacity planning, treat one RTX 3090 as a `3` smooth-session unit until the
current code path is rebenchmarked on a current Vast RTX 3090 offer.

| Option | Planning price | Planning smooth 20 fps sessions | Aggregate FPS reference | Cost per smooth slot-hour |
| --- | ---: | ---: | ---: | ---: |
| `1 x RTX 6000 Ada` | `$0.731/hr` | `4` | `112-114` current measured | `$0.183` |
| `1 x RTX 3090` | `$0.205/hr` | `3` | `71-72` historical optimized | `$0.068` |
| `2 x RTX 3090` | `$0.410/hr` | `6` | `142-144` estimated pool total | `$0.068` |
| `3 x RTX 3090` | `$0.615/hr` | `9` | `213-216` estimated pool total | `$0.068` |
| `4 x RTX 3090` | `$0.820/hr` | `12` | `284-288` estimated pool total | `$0.068` |

Read: multiple RTX 3090s are much better on raw cost efficiency. Three RTX 3090s
cost slightly less than one RTX 6000 Ada and should provide roughly `9` planned
smooth sessions instead of `4`, assuming each 3090 is on a healthy, uncapped host
and the current optimized profile reproduces the historical result.

The tradeoff is operational complexity:

- RTX 3090 wins cost-per-slot, redundancy, and scale-out economics.
- RTX 6000 Ada wins simplicity, 48 GB VRAM headroom, fewer worker processes,
  easier warmup, and the cleanest current proof for the optimized WebRTC path.
- RTX 3090 hosts vary more. The repo already has a 150W-capped RTX 3090 example
  that degraded badly, so provider/host selection matters.
- A 3090 pool needs worker routing, session stickiness, per-worker warmup,
  duplicated model storage, WebRTC/TURN port handling, and capacity-aware
  admission control.

Recommendation:

- Use `1 x RTX 6000 Ada` as the first always-on production MuseTalk baseline
  because it is the exact current measured path.
- Use RTX 3090s for scale-out once the autoscaler and router can handle multiple
  workers and a same-code RTX 3090 rebenchmark confirms `3` smooth sessions/GPU.
- Do not let the advertised unlimited plan depend on manual 3090 provisioning.
  The unlimited tier needs fair-use minutes until autoscaling and admission
  control are proven.

## Tier Margin Check

Assumptions:

- App-store fee cases: `30%` and `15%`
- Free full-use other variable AI cost: `$0.25-$1.00/user/month`
- Lite other variable AI cost: `$0.75-$2.00/user/month`
- Core other variable AI cost: `$1.00-$4.00/user/month`
- Unlimited other variable AI cost: `$2.00-$6.00/user/month`
- Video GPU cost shown at both `50%` and `25%` GPU utilization

### Free

```text
Price: $0
Included live-avatar minutes: 2/day, no rollover
Max monthly live-avatar minutes: about 60
```

| Case | Estimated monthly cost per fully active free user |
| --- | ---: |
| 50% video utilization + `$0.25` other AI | `$0.62` |
| 25% video utilization + `$1.00` other AI | `$1.73` |

Read: this is a real trial, not a fake preview. It is worth doing, but it must
be no-rollover and lower-priority than paid traffic. At `1000` fully active free
users, the live-video GPU cost alone is roughly `$365-$731/month` depending on
utilization.

### Lite

```text
Price: $9.99/mo
Included live-avatar minutes: 180
```

| Store fee | Net revenue | Best-case contribution | Worse-case contribution |
| ---: | ---: | ---: | ---: |
| 30% | `$6.99` | `$5.14` | `$2.80` |
| 15% | `$8.49` | `$6.64` | `$4.30` |

Read: optional. If the free tier converts well, skip Lite at launch. If users
need a cheaper paid step, `180` minutes is enough to feel paid without
cannibalizing Core.

### Core

```text
Price: $19.99/mo
Included live-avatar minutes: 600
```

| Store fee | Net revenue | Best-case contribution | Worse-case contribution |
| ---: | ---: | ---: | ---: |
| 30% | `$13.99` | `$9.34` | `$2.68` |
| 15% | `$16.99` | `$12.34` | `$5.68` |

Read: this is the corrected default plan. `$19.99` should feel meaningfully more
generous than `300` minutes. `600` minutes is the aggressive but still defensible
point: about `20` live-avatar minutes/day on average.

### Unlimited

```text
Price: $39.99/mo
Included live-avatar minutes: 1500 priority minutes
After 1500 minutes: fair-use fallback
```

| Store fee | Net revenue | Best-case contribution | Worse-case contribution |
| ---: | ---: | ---: | ---: |
| 30% | `$27.99` | `$16.85` | `$3.72` |
| 15% | `$33.99` | `$22.85` | `$9.72` |

Read: `$39.99` works if "unlimited" means unlimited text/audio and fair-use
live-avatar video. After the priority allowance, fallback options should include
queueing, lower-priority video, lower fps, or audio-only mode.

## Best Bet

Best initial pricing:

```text
Free:      $0        - 2 live-avatar minutes/day, no rollover
Lite:      $9.99/mo  - 180 live-avatar minutes
Core:      $19.99/mo - 600 live-avatar minutes
Unlimited: $39.99/mo - unlimited text/audio + 1500 priority live-avatar minutes
```

Why this is the best fit:

- Free gives users enough live video to understand the product before paying.
- `$9.99` is optional and should be removed if it creates too much plan clutter.
- `$19.99` is the main plan and now gives more than `300` minutes without
  crossing into dangerous economics.
- `$39.99` gives serious users a meaningful upgrade while protecting the GPU
  economics through a priority-minute/fair-use boundary.

## If You Insist On Two Tiers

Use:

```text
Free: $0 with 2 live-avatar minutes/day
Core: $19.99/mo
Unlimited: $39.99/mo
```

Free:

- limited text/audio trial
- `2` live-avatar minutes/day
- no rollover
- lower-priority admission

Core:

- unlimited text chat
- unlimited or generous audio-only practice
- `600` live-avatar minutes/month
- paid top-ups for extra video

Unlimited:

- unlimited text chat
- unlimited or generous audio-only practice
- `1500` priority live-avatar minutes/month
- after that: lower-priority video queue, lower fps, or audio-only fallback

This is probably the cleanest launch lineup: Free, Core, Unlimited. Add Lite
only if conversion data shows too many users bouncing at `$19.99`.

## Add-On Pricing

Sell extra live-avatar time:

| Add-on | Price |
| --- | ---: |
| 100 extra live-avatar minutes | `$3-$5` |
| 300 extra live-avatar minutes | `$9-$12` |

At current GPU cost, `100` live-avatar minutes costs roughly:

- `$0.61` at 50% GPU utilization
- `$1.22` at 25% GPU utilization

So a `$3-$5` add-on leaves room for payment fees, API cost, support, and margin.

## Product Copy Guidance

Avoid saying:

```text
Unlimited live video calls
```

Safer wording:

```text
Unlimited AI conversation, with generous priority live-avatar minutes.
```

For Free:

```text
Try live avatar calls for 2 minutes each day.
```

For Core:

```text
Includes 600 live-avatar minutes per month.
```

For Unlimited:

```text
Includes 1500 priority live-avatar minutes per month. After that, continue in
audio mode or use live avatar subject to fair-use capacity.
```

This keeps the product honest and prevents the economics from being destroyed by
a small number of extreme users.

## Operational Rules

1. Enforce per-user live-avatar minute accounting from day one.
2. Keep free live-avatar minutes at `2/day`, no rollover.
3. Put free sessions behind paid sessions in admission priority.
4. Keep live-avatar admission at `4` sessions per RTX 6000 Ada, or `3` if you
   want stronger SLA headroom.
5. Do not route C8+ as premium smooth video; that is degraded/stress capacity.
6. Use audio-only fallback when GPU capacity is full.
7. Measure actual utilization before increasing included minutes.
8. Revisit prices after benchmarking cheaper 48 GB GPUs such as RTX A6000, A40,
   or L40/L40S.
