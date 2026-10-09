# A5 outcome: quality rejected; local model-startup improvement verified

No production change or approved400FPS release results from this experiment.

The native whole-FP16 UNet passes both original main/holdout gates and all16
frozen UNet metrics. Coupled output with the original portable TAESD fails103
of698 frozen checks:78 canonical-avatar and25 decoder checks. Bounds remain
unchanged. [Decision](quality/a5_whole_fp16_quality_decision_0804.json) records
the rejection; partial actual six-avatar middle-frame inspection cannot override it.

The separately tested default-off skip-eager startup flag saves5.924s on average:
original manager initialization9.123/9.095s versus3.189/3.181s, measured in
fresh local0/1/1/0 processes with warm filesystem cache. All tested UNet,
decoder and seeded diagnostic VAE-encoder outputs are bit-identical and finite.
[Four-run report](startup/a5_model_startup_pair_summary_0824.json) includes
exact receipts and ownership watches. This is NOT full avatar preparation,
WebRTC, fresh Docker boot or EC2 request-to-usable-call acceptance. Flag stays0.

Both new native UNet sets (26 payloads,2,554,853,412 archive bytes) and actual
six-avatar quality captures (30 payloads,274,645,430 archive bytes) passed
private conditional S3 PUT, exact-version GET, clean CPU restore and every
payload SHA check. Receipts are in `release/a5_*private_persistence*.json`.
No private media/plans are committed or made public.

Independent ownedA5 expiry completed08:35:03UTC on October9,2026: nonforced
destroy and fresh provider absence verified. No owned paid GPU remains.
Full quality-approved400+FPS, production48
pose/live acceptance, complete published image/template and approximately60s
EC2 usable-call startup remain unmet. Existing production is unchanged.
