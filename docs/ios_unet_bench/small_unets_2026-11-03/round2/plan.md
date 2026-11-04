# Round 2 — one controlled change

Round 1 failed: 0/12 student r5 passes versus 12/12 full fp16 control passes. Its rental is destroyed,
provider absence verified, expiry timer inactive, and estimated cumulative spend is $0.763 (<$5).

The **only changed training setting** is `DISTILL_STEPS=100000` (round 1: 30000). Keep bk_tiny, code
`d8826442c5d1bd8772185a7fc564eddd98faf19f`, the same teacher initialization/seed/data/split, batch 8,
gradient checkpointing, LR 1e-4, warmup 500, feature weight 1, eval every 250, and a 150-minute maximum.
Start a fresh experiment; do not resume round 1 with a different manifest. The larger step cap lets the
existing wall-time cosine schedule and 150-minute limit end the run, approximately 38k–40k actual steps.

Rationale: this uses the remaining authorized training time while retaining the only measured topology with
capacity above 25 fps on the A14. bk_small's measured 18.2–21.5 fps conflicts with that target. Lowering LR has
no stronger evidence than allowing the current stable optimization longer. Additional training is not a
promise that the quality gap will close. Acceptance is unchanged: all 12 clips must pass every r5 gate and
all six zero-margin visual comparisons against D.

Use exactly one second rental: `mtboot-dt-r2-20261104a`, stock launch, worker revision 4e2b02e, one RTX 3090,
price cap $0.30/h, lifetime ≤4 hours, expiry installed before create. Preflight quote: $0.2747743056/h.
At the hourly cap a four-hour lease adds at most $1.20 before transfer. Never recreate on an uncertain result;
reconcile with status. Never touch 51074906, production template hash 4dc0868e, or autoscaler settings.
Pull and verify all outputs, then destroy and verify provider absence and inactive expiry timer. No third
rental is authorized. After training/review, benchmark the selected trained student on the connected iPhone
against today's two-worker chain at matched 24 fps and an explicit 25-fps workload. Do not modify Lingua,
merge, or open a PR.
