# E2 Token + E3 CLS: semantic routing and difference fusion

This screen uses eight fresh seed-42 runs of 6000 updates on GPU 2, in order
N0, N1, N2, N3, N4, N5, N6, N7. The 48000-update budget excludes a separate
three-update-per-arm correctness smoke and fixed-checkpoint inference.

| Arm | Active stages | Class routing | Residual |
|---|---|---|---|
| N0 | none, original visual MT | none | none |
| N1 | auxiliary supervision only | none | none |
| N2 | E2 tokens, E3 CLS | independent attention | plain |
| N3 | E2 tokens, E3 CLS | supervised tissue logits | plain |
| N4 | E2 tokens, E3 CLS | independent attention | difference |
| N5 | E2 tokens | supervised tissue logits | difference |
| N6 | E3 CLS | supervised tissue logits | difference |
| N7 | E2 tokens, E3 CLS | supervised tissue logits | difference |

E2 and E3 modify the shared single-slice encoder before skip storage and the
next downsampling. E3 consumes E2's modified feature. Three slices share all
weights; the original adaptive neighbor fusion and decoder follow encoding.
E4 retains its auxiliary head without a text injection.

The frozen bank contains layer-12 word tokens, per-class CLS, masks and four
512-dimensional tissue anchors. It is embedded in every text checkpoint.
No text encoder or external bank is needed to load a trained checkpoint.
All N1–N7 tensors are initialized in the same order. Unused branches are frozen;
the total and participating parameter counts are reported separately. N6's
aligned single-CLS branch does not use K, so its K map is frozen.

Aligned routing computes 512-dimensional normalized pixel projections against
unit anchors at temperature 0.07. The same logits feed class mixing and the
existing area-pooled, class-balanced tissue CE. E2 retains within-class token
attention in 64 dimensions; E3 has one CLS per class. K/V are shared by both
stages. For every nonbaseline arm, the three auxiliary stage losses on labeled
center slices are averaged, then multiplied by 0.1. All slices are processed
identically before selecting these supervised centers.

Difference fusion computes Phi(Q,Z) - Phi(Q,0), then multiplies by tanh(g).
Both Phi calls share weights and participate in backpropagation. Gates start
at zero. Phi uses the existing 3x3 convolution, GroupNorm, GELU and 1x1
convolution. There are no stochastic layers or second encoder pass.

The fixed protocol is inherited from `configs/v3/context_efficiency_reference.yaml`:
1496 labeled / 2236 unlabeled slices, 224x224 triplets, batch 8 = 4L+4U,
AdamW 0.0003 with the existing polynomial decay, EMA 0.99, unsupervised warmup
600 and ramp 1200 updates. Every 500 updates, all 21 validation cases are
evaluated. No test data, image distillation, centered alignment, new prompts,
boundary objectives or extra combinations are introduced.

Best EMA selection uses case-equal GT-present foreground Mean Dice, retaining
the earlier checkpoint on ties. Best and final checkpoints are both reported.
N7 must improve over N0 by at least 0.3 percentage points and over N1/N2/N5/N6
by at least 0.1, without decreasing best-checkpoint Union-present Dice against
any of these controls. Rank reversals between best and final are flagged.
Routing contrasts are N7-N4 and N3-N2; difference contrasts are N7-N3 and N4-N2;
auxiliary supervision is N1-N0. These component effects are reported with
their signs and Union changes, without an invented significance threshold.

After training, N7 best and final are each evaluated in five states: full,
zero E2 text, zero E3 text, zero both, and class-content permutation [3,0,1,2].
The permutation moves K/V content and masks together, preserving anchors.
Full inference must first exactly reproduce historical validation metrics and
case confusion matrices. Interventions do not update weights or reselect a
checkpoint, and are not substitutes for training ablations or proof of medical
semantic understanding.

The frozen runner writes overall, 19 foreground classes plus background,
per-case, merged-tissue, validation-curve and paired-difference CSVs.
`training_window_summary.csv` averages every training step. The separate
`training_diagnostics.csv` samples step 1 and each logging interval, including
gates, route distributions, route/auxiliary agreement, token entropy, actual
residual RMS and raw with-text / zero-text / difference RMS. These statistics
use different sample sets and must not be interchanged.

Preparation creates a new directory and hashes code, configurations and bank.
Execution checks those hashes, takes an exclusive lock, initializes CUDA before
resetting memory statistics, and stops with failure records on exceptions or
nonfinite values. No retry, device change or protocol change is automatic.
An explicit resume uses existing checkpoints and exact saved configuration.

Entrypoints: `experiments/check_dual_text.py` for the isolated real-size smoke;
`experiments/run_dual_text.py --prepare-only` for freezing; the frozen copy's
`--execute-frozen` for execution. Set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before
launch. After confirming the physical GPU, process and first valid training
record, leave the runner to finish without continuous monitoring.
