# How optiMOFA came to be

A dated record of why this fork exists, what was tried, what failed and how the result was
checked. Written to be usable as source material for a methods section or an appendix.
Upstream: [bioFAM/mofapy2](https://github.com/bioFAM/mofapy2) 0.7.5 (commit `90418e5`,
2026-08-26), LGPL-3.0.

## 1. The problem (2026-11-17 to 2026-11-19)

The fork was made for a BSc thesis that builds epigenetic and transcriptomic aging clocks
from blood DNA methylation and RNA expression (n = 769 to 1,202 samples). One arm of the
comparison integrates both layers with MOFA+ (spike-and-slab and ARD priors on the weights,
one group, two Gaussian views) and uses the factors as predictors, inside a 5 x 5 nested
cross-validation. The MOFA+ settings tuned in the inner loop are up to 200,000 CpGs + 9,512
genes and 40, 60 or 100 factors, and every outer fold refits the chosen setting. Stability
and bootstrap analyses add more fits.

Measured with stock mofapy2 on that data (float32, 12 CPU cores):

| setting | s / iteration | iterations to converge | peak memory |
|---|---|---|---|
| 10,000 + 5,000 features, K = 40 | 2.3 | 70-131 | 2.2 GB |
| 50,000 + 9,512, K = 60 | 16.1 | 70-131 | 3.2 GB |
| 200,000 + 9,512, K = 100 | 94 | 70-131 | 7.4 GB (n = 769), 9.8 GB (n = 1,202) |

That adds up to about 13 days of inner-loop tuning and about 5 more days of refits, or 2-3
weeks of compute for one arm. Stock `gpu_mode` gives about 3x.

## 2. Where the time goes

Profiling the updates showed that the cost does not come from the model but from how the
updates are written. mofapy2 handles the most general case uniformly: the noise precision
`tau` is expanded to a full samples x features matrix (it can differ between groups, and it
is zeroed for missing values). The updates then work with N x D matrices:

- **W update** (spike-and-slab): for every factor k, one N x (K-1) by (K-1) x D product, so
  O(N D K^2) per iteration. This is the dominant term.
- **Z update**: the same structure, O(N D K^2).
- **Tau update**: builds the full N x D matrix of expected squared residuals.
- **Variance explained**, computed every iteration while factors may be dropped: one N x D
  outer product per factor.
- **ELBO**: counts observed entries with an N x D scan of the mask.
- **`gpu_mode`**: uploads the data matrix again in every W, Z and Tau update.

## 3. The idea: one assumption, exact closed forms

With **one group and no missing values** in a view, `tau_nd = t_d` for every sample by
construction of the model: mofapy2 stores one precision per (group, feature). The precision
then factors out of every sum over samples. All of these updates can be written with the
K x K Gram matrix `C = E[Z]^T E[Z]` and the D x K product `P = Y^T E[Z]`. The O(N D K^2)
terms become O(N D K + D K^2), and no N x D temporary is needed. These are algebraic
identities, not approximations, so the results equal stock up to floating-point summation
order. The derivations are in [fast_updates.md](fast_updates.md).

**Why upstream does not do this (our assessment):** mofapy2 targets the general case
(several groups, missing values, stochastic inference), where `tau` really varies by sample.
A single code path for that case is simpler to maintain. The special case only matters at
large D x K; for large data, mofapy2 itself only warns that factor dropping is slow and
suggests switching it off.
Nothing in the special case is new mathematics; it is the usual Gram-matrix trick for
factor models.

## 4. First version, inside the project (before 2026-11-19)

The first implementation was a runtime patch module in the thesis project (`mofa_fast.py`)
that replaced the five updates and added a GPU path. A unit test against stock found a Tau
mismatch: the stock W update also writes the new E[W] in place, and the patch had to do the
same. With that fixed, a full fit of 15,000 features / K = 40 took 6 s instead of 409 s.

## 5. An out-of-memory kill that was not the patch's fault (2026-11-19)

The first full parallel run of all arms was OOM-killed in the MOFA+ chain: the process held
10 GB plus 7.4 GB of swap. The cause was reference cycles. mofapy2 nodes reference each other
(Markov blankets), so a fitted model is freed only by Python's cycle collector, which large
NumPy arrays hardly trigger. Models piled up across the inner-CV fits. An explicit
`gc.collect()` after each fit fixed it: five sequential fits stay at 1.26 GB, against
1.6 → 2.9 GB without it. This affects stock mofapy2 equally, and the fix belongs in the
calling code.

## 6. Back to stock, then rigorous testing (2026-11-20)

The patch was removed from the project, which went back to pristine mofapy2 (verified
against the wheel's RECORD hashes), and the earlier results were archived. Before trusting
any speedup, the questions were: is it needed, is it really better, and does it give the same
results? The patches were moved to a separate harness outside the project. There each change
(W, Z, Tau, R2, YELBO, and GPU residency) can be switched on alone or in any combination.

**A bug the tests found.** With the ELBO shortcut on, one data set needed 131 iterations
instead of stock's 91. Stock's "TauTrick" (likelihood term = `E[tau] (b - b0)`) is only valid
right after a Tau update. mofapy2 computes one ELBO *before* training with the trick off,
and its convergence criterion is relative to that value. The rewrite had applied the trick
there too. The first version had matched stock only by accident, because a flag was not yet
set at that point. Fix: fall back to stock whenever `TauTrick` is off.

**Accuracy, synthetic data.** Ten data sets with different statistical properties: feature
noise, K close to N, pure noise, heavy tails with outliers, correlated and unscaled
features, sample-dependent noise, two groups, missing values, N >> D. Each was fitted with
all 64 combinations of the changes plus stock `gpu_mode`. **640/640 took the same iterations
and dropped the same factors as stock**; the worst relative difference in W, Z or tau was
5.3e-13.

**Accuracy, real data** (10,000 CpGs + 5,000 genes, to convergence): every combination took
the same 81 iterations and dropped the same factors (40 → 36). Differences were about 1e-13
in float64. In float32 they were about 1e-5, the same as stock `gpu_mode` vs stock CPU.

**Performance**, for all 64 combinations at two sizes and fix-level combinations at the full
size. The changes only pay off together. Alone, Z, Tau, R2 or YELBO gave no gain and W gave
1.5x, because each removes one of several equally expensive passes over the data. W+Z gave
7x (at 50k features), adding Tau 14x, and all five about 30x. At 200,000 + 9,512 features
and K = 100: stock 76.8 s/iteration, stock `gpu_mode` 24.1 s, all five changes 2.45 s on the
CPU (31x) and 1.48 s with the GPU (52x). Peak memory went *down* on the CPU (7.8 vs
8.2 GB). A full fit at 10,000 + 5,000 features took 6 s instead of 311 s.

**Is the assumption realistic?** It is exact relative to stock MOFA+, which makes the same
assumption. Whether the *data* satisfy it is a separate question, so it was checked. After a
fit, the per-sample residual variance varies about 20x more than equal noise would allow. It
depends on the processing chip (batch; η² 0.58 for expression, 0.23 for methylation, null
0.14) and, for methylation, on age (r = +0.27). This misspecification affects stock MOFA+
in exactly the same way. Remedies are batch correction, per-sample scaling, or MOFA+ groups,
and the fast forms then apply per group.

## 7. Shipping it: why a fork (2026-11-20)

The options considered:

- **Git submodule:** clones need `--recursive`, and it still needs a path dependency.
- **Runtime patch inside the project:** fragile, monkeypatches private methods, and is
  harder for a reader to audit than a diff.
- **Fork, installed as a pinned git dependency:** the lock file records the exact commit, a
  reviewer sees one readable diff against upstream, and the LGPL-3.0 condition that modified
  versions are distributed with source is met by the public fork itself.

The fork was chosen: `tristanRW/optiMOFA`, branched from upstream `master` at 0.7.5.

## 8. What the fork changes

- `mofapy2/core/fast_updates.py` (new): the closed forms for the spike-and-slab W, Z, Tau,
  variance explained and the ELBO likelihood term, plus `applies()` (full batch, one group,
  no missing values).
- `W_nodes.SW_Node`, `Z_nodes.Z_Node`, `Tau_nodes.TauD_Node`, `Y_nodes.Y_Node`,
  `BayesNet.calculate_variance_explained`: one early branch each into the fast form. The
  stock code below it is unchanged and runs whenever the fast form does not apply.
- `gpu_utils`: each view's data matrix is registered when the model is built and uploaded to
  the GPU once per fit. This also helps stock code paths, which call `gpu_utils.array(Y)`
  in every update.
- `entry_point.set_train_options(..., fast_updates=True)` and `mofa(..., fast_updates=True)`:
  `False` restores stock behaviour exactly, for verification.
- `tests/test_fast_updates.py`: the equivalence tests (10 data sets on the CPU, float32, no
  factor dropping, 4 on the GPU, single upload).

Benchmark of the fork itself on the real data (float32): 2.45 s/iteration on the CPU and
1.69 s with the GPU at 200,000 + 9,512 features and K = 100, against 76.8 s for stock.

## Not covered

- W without spike-and-slab (`W_Node`), spike-and-slab on Z (`SZ_Node`), non-Gaussian likelihoods, GP / MEFISTO nodes and
  stochastic inference use stock code unchanged.
- Several groups and views with missing values also fall back to stock code. The algebra
  extends to them (one Gram matrix per group, masked sums), but that is not implemented.
