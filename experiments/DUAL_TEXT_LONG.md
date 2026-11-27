# MT / N0 / N7 paired 200-epoch training

The three fresh seed-42 runs compare single-slice Mean Teacher (MT), adaptive
three-slice context without text or auxiliary tissue supervision (N0), and the
unchanged full E2-token / E3-CLS aligned difference-fusion model (N7).

One epoch is one pass over the 1496 labeled slices, at 4 labeled examples per
batch: 374 updates. Each arm runs 200 epochs = 74800 updates; total = 224400.
The unlabeled pool contains 2236 slices and cycles independently. All arms
share center-slice ordering and augmentation, base U-Net initialization,
224x224 input size, batch 8 = 4 labeled + 4 unlabeled, and optimizer settings.
Context inputs increase computation, so equal updates are not equal compute.

Use the existing context reference as the base configuration, not the older
long dual-frequency configuration, which enables additional objectives.
Keep AdamW initial LR 0.0003 and weight decay 0.0001, polynomial power 0.9,
original EMA update rule targeting 0.99, unsupervised warmup 600 / ramp 1200,
confidence thresholds, augmentation and gradient clipping unchanged. N7 uses
its fixed embedded text bank and auxiliary alignment weight 0.1. No old
short-run checkpoint initializes these models.

Evaluate the full 21-case validation cohort every 5 epochs (1870 updates),
40 times per arm. Select best EMA teacher1 by case-equal GT-present foreground
Mean Dice, retaining the earlier step on ties. Never read test images/labels.
At epochs 100 and 200, retain both the exact state and the best-so-far state;
these milestones belong to a single 200-epoch LR schedule. The second half
restores the optimizer, EMA and RNG without restarting learning rates.
The per-invocation runtime limit is 86400 seconds, replacing the short-run
4-hour limit. Record each invocation's time and peak memory so costs include
both halves and interrupted work.

Prepare with `experiments/run_dual_text_long.py --output <new-dir> --prepare-only`.
The default milestones are `--epochs 100 200`, with `--eval-every-epochs 5`.
Preparation verifies actual paired center inputs, strong augmentation and
CutMix at epochs 0, 99, 100 and 199, common initialization, and N7 zero-gate
identity with N0. It freezes code, configurations, bank and hashes in a fresh
directory. The frozen copy's `--execute-frozen` runs MT, N0 and N7 serially on
GPU 2. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before launch. Hash mismatch,
nonfinite loss or an exception stops execution and preserves failure records;
there is no automatic retry or device change. Explicit `--resume` continues
the same immutable protocol. An exclusive lock prevents concurrent runners.

Reports distinguish exact milestone performance from best-through-milestone
performance, with overall Dice/IoU/Union, 19 foreground classes and background,
valid-case counts, per-case values, merged-tissue Dice, validation curves and
paired differences for N7-N0, N0-MT and N7-MT. The primary engineering reference
is N7-N0 best Dice >= 0.3 percentage points with nondecreasing Union. Flag
best/exact rank reversals. These three arms do not isolate routing and
difference components and do not establish multi-seed robustness.

Diagnostics retain all-step scalar means separately from sampled gates,
alignment, routing, attention and residual measurements. Resource reports
include total/participating parameters, cumulative invocation time and peak
allocated/reserved GPU memory.

After N7 finishes, rerun its global best and exact final checkpoints in five
states each: full, zero E2, zero E3, zero both, and fixed class-content
permutation [3,0,1,2] with anchors unchanged. Full inference must first exactly
reproduce saved validation. These ten evaluations perform no training and
never reselect a checkpoint. Equal best/final weights are not independent
replications.

Run `experiments/check_dual_text_long.py` in a separate temporary output
directory for three real-size correctness updates per arm. This smoke is
outside the formal training budget. Launch only after paired-input, CPU/GPU
exact-resume, old-loader compatibility and fresh-frozen-process tests pass.
After confirming the process, physical GPU and first valid training record,
leave the background runner to finish without continuous monitoring.
