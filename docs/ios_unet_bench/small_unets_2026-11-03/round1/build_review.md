# Distillation build review — 2026-11-03

Scope: `distill_small_unet.py`, `export_small_unets.py`, `review_distilled_unet.py`,
`distill_box/`, and the phone A/B runner. The unreviewed WIP has not been used for training.

Confirmed findings fixed before rental:

- BK-SDM up-block mapping: the student's second retained residual/attention block copies teacher block 2,
  with the tiny model's stage shift. All student tensors must copy successfully; random fallback is rejected.
- Training/selection reject all review and gate identities (including other poses) and any byte-identical gate
  frame. The box requires exactly 10 training and 2 validation characters, records their identities, and excludes
  both review audios. Gate results are reported at every evaluation and never select the checkpoint.
- Hashed data/cache manifests and optimizer/RNG/scheduler checkpoints keep resumed work consistent. Neither
  the box nor trainer can extend a saved training budget. Detached execution survives an SSH disconnect.
- Reviewer comparisons require all clips, all r5 gates, finite metrics, matching fps/frame counts/ROIs, and at
  least D's landmark coverage. The six existing visual comparisons against D allow zero deterioration, as
  requested. Latent error and the student fp16 arm are also reported. Positional encoding uses each arm's dtype.
- The rental uses an atomic local create-intent marker and the EC2 controller's pre-create expiry timer. Stock
  launch, pinned worker revision, one RTX 3090, $0.30/h and four-hour caps remain mandatory. Cleanup verifies
  provider absence and inactive timers. FaceMesh is mandatory; overrides cannot weaken the data/review rules.
- Trained Core ML export loads only the student, avoiding an unnecessary full teacher allocation on the Mac.
- The phone runner stops after exhausted retries and validates the complete requested arm order and matched
  inputs. Unknown phone process state blocks a run; its watchdog terminates only the exact harness executable.
- The last-resort `uv pip` dependency installer uses the supported command/option order.

Validation: four-step tiny CPU training, interrupted resume matching uninterrupted weights exactly, strict
student reload, held-out guards, gate results at steps 0/2/4, and student-only export checks pass. The topology
test passes with installed diffusers 0.42.1 and an isolated exact 0.30.2 wheel. Four acceptance regressions,
Python compilation/import/help checks, shell parsing, nested SSH/stdin round-trip mocks, complete box dry-run,
10/2 data manifest checks, and duplicate-create/price/lifetime/protected-instance refusal mocks pass. No full
model was allocated on the Mac for these tests. CUDA validation follows on the rental.

Read-only preflight for `mtboot-dt-r1-20261103a` found a stock RTX 3090 offer at $0.2747743056/h with
`template_hash_id: null`; no instance was created by that preflight. The rate is rechecked at create. A four-hour
lease at that quoted rate is $1.0991 before transfer. The user authorizes at most two sequential rentals and
$10 total, with round 2 conditional on round 1 failing and cumulative spend below $5.

Review clips use the existing 250-frame/10-second, 25-fps protocol matching `review_mixed_units` arm D.
Phone testing follows training/review: mirror today's two-worker model against the trained student at the
original 24-fps target, then test a 25-fps target. A 24-fps playback result alone cannot establish 25 fps.
The user explicitly authorized closing SoulX Studio for these tests. Lingua stays untouched.
