# optiMOFA fast updates: derivations, validity and tests

This file derives the closed forms implemented in `mofapy2/core/fast_updates.py`, states
when they are exact, and reports the tests and benchmarks. The equivalence tests that ship with
the fork are in `tests/test_fast_updates.py`; the larger runs reported below (every subset
of the changes, real data) were made with a runtime-patching harness before the changes were
moved into this fork, and the fork was then re-benchmarked (last section).

## Setting and notation

One view (the views only meet in the Z update), one group, no missing values. Data
`Y` (N samples x D features, centred). Model

    Y_nd = sum_k Z_nk W_dk + e_nd,   e_nd ~ N(0, 1/tau_d)

with mean-field posteriors q(Z) q(W) q(tau); W has a spike-and-slab prior (W = S * What),
Z a Gaussian prior, tau_d a Gamma prior/posterior with parameters (a_d, b_d).

Expectations under q (all elementwise unless noted):

| symbol | meaning | size |
|---|---|---|
| `Zb = E[Z]`, `<Z2> = E[Z^2]` | factor means / second moments | N x K |
| `Wb = E[W]`, `<W2> = E[W^2]` | for spike-and-slab `Wb = theta*mu`, `<W2> = theta*(mu^2 + s2)` | D x K |
| `t_d = E[tau_d]` | noise precision of feature d | D |
| `C = Zb^T Zb` | Gram matrix of the factor means | K x K |
| `P = Y^T Zb` | data-factor cross product | D x K |

**The one assumption the shortcuts need** is that the precision of an entry, `tau_nd`,
does not depend on the sample: `tau_nd = t_d`. In mofapy2 this holds by construction within
a group: `TauD_Node` stores one precision per (group, feature), shape `(G, D)`, and expands
it to samples by indexing with each sample's group (`QExp["E"][self.groups, :]`). The only
other place where `tau_nd` varies is missing data: stock code sets `tau_nd = 0` for masked
entries. So "one group and no mask" is exactly the condition, and both are checked per view
at run time (`fast_updates.applies`; otherwise the stock update runs).

## Fix 1a: W update (spike-and-slab)

Stock mofapy2 (`SW_Node._updateParameters`) updates the factors one at a time; for factor k
and feature d it needs

    foo_dk  = sum_n tau_nd <Z2>_nk
    t1_dk   = sum_n tau_nd Y_nd Zb_nk
    t2_dk   = sum_n tau_nd Zb_nk sum_{j != k} Zb_nj Wb_dj        (Wb with factors < k already updated)

and computes `t2` as an N x (K-1) times (K-1) x D product for every k: O(N D K^2) per
iteration. With `tau_nd = t_d` the precision leaves the sum over samples:

    foo_dk = t_d * sum_n <Z2>_nk
    t1_dk  = t_d * P_dk
    t2_dk  = t_d * sum_{j != k} Wb_dj (sum_n Zb_nk Zb_nj) = t_d * [ (Wb C)_dk - Wb_dk C_kk ]

`P` is computed once per iteration (O(N D K)); inside the loop over k only `Wb C[:, k]`
(O(D K)) remains, so the loop costs O(D K^2). The rest of the update (precision, posterior
mean, inclusion probability, and the in-place refresh of `Wb[:, k]` before factor k+1) is
unchanged, which keeps the sequential (coordinate-ascent) order of stock mofapy2.

## Fix 1b: Z update

Stock (`Z_Node._updateParameters`), for factor k, sample n, summing over views m with
view weights `v_m` (1 unless `weight_views`):

    foo_nk = sum_m v_m sum_d tau_nd <W2>_dk
    bar_nk = sum_m v_m sum_d tau_nd Wb_dk (Y_nd - sum_{j != k} Zb_nj Wb_dj)

With `tau_nd = t_d`, `foo_nk = sum_m v_m (t^T <W2>)_k` is the same for all samples, and with
`G = sum_m v_m Wb^T diag(t) Wb` (K x K):

    bar_nk = sum_m v_m (Y diag(t) Wb)_nk - [ (Zb G)_nk - Zb_nk G_kk ]

`Y diag(t) Wb` is computed once per iteration (O(N D K)); the loop over k is O(N K^2).

## Fix 2a: Tau update

Stock (`TauD_Node._updateParameters`): `a_d = a0 + N/2`, `b_d = b0 + 1/2 sum_n E[(Y_nd - sum_k
Z_nk W_dk)^2]`, computed as a full N x D matrix of expected squared residuals. Expanding the
square, and using that Z and W are independent under q:

    E[(sum_k Z_nk W_dk)^2] = (sum_k Zb_nk Wb_dk)^2 - sum_k Zb_nk^2 Wb_dk^2 + sum_k <Z2>_nk <W2>_dk

Summing over samples gives, per feature d,

    sum_n E[res_nd^2] = sum_n Y_nd^2                       (constant, computed once)
                      - 2 (P_d . Wb_d)                      (row d of P with row d of Wb)
                      + Wb_d C Wb_d^T
                      - sum_k Wb_dk^2 (sum_n Zb_nk^2)
                      + sum_k <W2>_dk (sum_n <Z2>_nk)

This needs no assumption on tau (it is a sum per feature), only that every sample of the
feature is observed (a mask would restrict the sum over n) and one group (otherwise one such
sum per group). Cost O(N D K) for `P` (shared with fix 1a) plus O(D K^2); no N x D temporary.

## Fix 2b: variance explained (factor dropping)

Stock (`BayesNet.calculate_variance_explained`, run every iteration when factors may be
dropped) uses point estimates: `R2_k = 1 - sum_nd (Y_nd - Zb_nk Wb_dk)^2 / SS` with
`SS = sum_nd Y_nd^2`, building one N x D outer product per factor (O(N D K) and N x D memory
per factor). Expanding the square:

    sum_nd (Y_nd - Zb_nk Wb_dk)^2 = SS - 2 sum_d Wb_dk P_dk + (sum_n Zb_nk^2)(sum_d Wb_dk^2)

and for the total `||Y - Zb Wb^T||^2 = SS - 2 tr(Wb^T P) + tr(Wb C Wb^T)`. No assumption on
tau; needs no mask and one group (as stock, which zeroes masked entries).

## Fix 2c: likelihood term of the ELBO ("TauTrick")

    E_q[log p(Y | Z, W, tau)] = -N D/2 log(2 pi) + 1/2 sum_d n_d E[log tau_d]
                                - sum_d E[tau_d] * 1/2 sum_n E[res_nd^2]

with `n_d` the number of observed samples of feature d. Right after the Tau update,
`1/2 sum_n E[res_nd^2] = b_d - b0_d`, so the last term is `E[tau_d] (b_d - b0_d)` (stock's
"TauTrick"). Stock still counts `n_d` with an N x D scan of the mask every iteration; without
a mask `n_d = N`. **Validity condition found by the tests:** the trick holds only right after
a Tau update. mofapy2 computes one ELBO *before* training with `TauTrick = False` (exact
formula), and its convergence criterion is relative to that value. The rewrite must
therefore fall back to stock code whenever `TauTrick` is off; an earlier version did not and
only matched stock by accident (its no-mask flag was not yet set before the first update).

## Fix 3: data resident on the GPU

No change in the mathematics: stock `gpu_mode` uploads the data matrix again
(`gpu_utils.array(Y)`) in every W, Z and Tau update, i.e. several times per iteration. The fork
registers each view's data matrix when the model is built (`gpu_utils.register_resident`) and
uploads it once per fit; stock code paths that still call `gpu_utils.array(Y)` get the
cached device copy, and the products of fixes 1-2 run there. Results differ from the
CPU only by floating-point summation order.

## Is the equal-noise assumption realistic? (real data: DNA methylation + RNA, n = 769)

The speedups are exact *relative to stock MOFA+*, which makes the same assumption. Whether
the data satisfy it is a separate question. After a fit (10,000 CpGs + 5,000 genes, 40
factors), per sample `v_n = mean_d t_d res_nd^2` should be about 1 for every sample, with a
spread of about `sqrt(2/D)` if the noise were equal across samples:

| view | v_n 5 / 50 / 95 % | SD across samples (expected if equal) | chip (batch) eta^2 (null) | other |
|---|---|---|---|---|
| methylation | 0.71 / 0.91 / 1.60 | 0.276 (0.014) | 0.23 (0.14), p < 0.005 | corr with age +0.27 |
| expression | 0.65 / 0.93 / 1.37 | 0.253 (0.020) | 0.58 (0.14), p < 0.005 | corr with neutrophil share +0.26 |

So the noise is **not** equal across samples: it varies about 20 times more than chance would
allow, strongly with the processing chip for expression and with age for methylation (the
older the sample, the noisier its residual methylation, in line with age-related epigenetic
drift). This misspecification affects stock MOFA+ with one group in exactly the same way. Ways to address it, if wanted: batch correction per chip before MOFA+, per-sample
scaling, or MOFA+ groups (one noise level per group and feature; with 108 chips too many
groups, but e.g. a few batches or age bins would be possible, and the shortcuts then apply
per group).

## Tests

Changes tested: `W` (1a), `Z` (1b), `Tau` (2a), `R2` (2b), `YELBO` (2c), each alone and in all
32 combinations, each on the CPU and with `GPU` (fix 3: mofapy2 `gpu_mode` plus resident data),
i.e. 64 configurations, plus stock `gpu_mode` as shipped. Reference: stock mofapy2 0.7.5, same
data, seed and options (spike-and-slab + ARD on W, `convergence_mode="fast"`, `dropR2=1e-4`).

### Accuracy, synthetic data (float64, full fits to convergence)

Ten data sets with deliberately different statistical properties; 64 configurations each
(640 fits). "Same trajectory" = identical number of iterations *and* identical number of active
factors at every iteration (so convergence and factor dropping took the same decisions).

| data set | property | stock iters, K end | same trajectory | worst rel. diff W/Z/tau | worst ELBO rel. diff |
|---|---|---|---|---|---|
| baseline | low rank, sparse W, Gaussian noise | 166, 15->5 | 64/64 | 2.9e-14 | 1e-16 |
| hetero_features | feature noise SD 0.2..5, unequal views | 126, 12->5 | 64/64 | 5.3e-13 | 1e-16 |
| wide_small_n | N = 40, K = 30 (K close to N) | 91, 30->6 | 64/64 | 1.3e-13 | 3e-17 |
| noise_only | no structure, factors dropped | 121, 10->4 | 64/64 | 5.6e-14 | 6e-17 |
| heavy_tails | t(2.5) noise + 1 % gross outliers | 31, 12->4 | 64/64 | 4.0e-15 | 9e-17 |
| correlated_unscaled | AR(1) features, scales 1e-2..1e2, dense W | 26, 12->11 | 64/64 | 2.0e-14 | 9e-19 |
| sample_hetero | two batches, noise SD 0.5 vs 2 | 66, 12->4 | 64/64 | 1.3e-14 | 1e-16 |
| two_groups | same, modelled as 2 groups (fallback path) | 131, 12->4 | 64/64 | 4.7e-15 | 5e-17 |
| missing | 5 % NaN in view 0 (view 0 fallback) | 106, 15->5 | 64/64 | 1.8e-14 | 6e-17 |
| tall | N = 1000 >> D | 61, 8->2 | 64/64 | 1.7e-14 | 2e-16 |

All 640 configurations reproduce stock to floating-point rounding (worst 5.3e-13), including
the misspecified data sets (heavy tails, sample-dependent noise): the rewrites are algebraic
identities of stock's computation, not approximations of the model, so they agree with stock
whether or not the model fits the data. Variance explained differs by at most 1.1e-15.

### Accuracy, real data (10,000 CpGs + 5,000 genes, n = 769, K = 40, to convergence)

Fix-level combinations (`fix1` = W+Z, `fix2` = Tau+R2+YELBO), float64 against stock float64
and float32 against stock float32:

| config | float64: same trajectory | W / Z / tau rel. diff | wall (stock 311 s) | float32: same trajectory | W / Z / tau rel. diff | wall (stock 127 s) |
|---|---|---|---|---|---|---|
| GPU | yes | 5e-14 / 1e-14 / 1e-14 | 175 s | yes | 3e-5 / 6e-6 / 3e-6 | 78 s |
| fix2 | yes | 2e-14 / 4e-15 / 6e-14 | 175 s | yes | 5e-6 / 1e-6 / 2e-5 | 59 s |
| fix2+GPU | yes | 7e-14 / 2e-14 / 1e-13 | 59 s | yes | 3e-5 / 6e-6 / 1e-5 | 25 s |
| fix1 | yes | 5e-14 / 1e-14 / 6e-15 | 141 s | yes | 2e-5 / 2e-6 / 2e-6 | 126 s |
| fix1+GPU | yes | 6e-14 / 2e-14 / 7e-15 | 140 s | yes | 2e-5 / 2e-6 / 2e-6 | 128 s |
| **fix1+fix2** | yes | 7e-14 / 1e-14 / 7e-14 | **6 s** | yes | 2e-5 / 2e-6 / 2e-5 | **4 s** |
| fix1+fix2+GPU | yes | 1e-13 / 2e-14 / 2e-13 | 7 s | yes | 2e-5 / 2e-6 / 1e-5 | 5 s |
| stock gpu_mode | yes | 5e-14 / 1e-14 / 1e-14 | 189 s | yes | 3e-5 / 6e-6 / 3e-6 | 71 s |

Every configuration takes the same 81 iterations and drops the same factors (40 -> 36). In
float32 the differences (~1e-5) are float32 rounding: stock's own `gpu_mode` differs from stock
CPU by the same amount (2.7e-5).

### Performance (real data, float32, seconds per iteration, peak RSS)

Seconds per iteration = (time for 6 iterations - time for 1) / 5, each configuration in a fresh
process, WSL2, 12 GB RAM, 12 GB GPU. Single measurements: at size S differences below ~0.3
s/iter are noise.

Size L, the real MOFA+ arm setting (200,000 CpGs + 9,512 genes, K = 100):

| config | s/iter | speedup | peak RSS |
|---|---|---|---|
| stock | 76.8 | 1.0x | 8.2 GB |
| stock `gpu_mode` | 24.1 | 3.2x | 10.6 GB |
| GPU (resident data) | 42.0 | 1.8x | 10.6 GB |
| fix2 | 62.1 | 1.2x | 8.3 GB |
| fix2+GPU | 24.8 | 3.1x | 10.2 GB |
| fix1 | 5.1 | 15x | 9.8 GB |
| fix1+GPU | 3.6 | 21x | 11.5 GB |
| **fix1+fix2** | **2.45** | **31x** | **7.8 GB** |
| fix1+fix2+GPU | 1.48 | 52x | 8.8 GB |

Size M (50,000 + 9,512, K = 60), stock 10.7 s/iter, every change alone and the main combinations
(65 configurations measured):

| config | CPU s/iter (speedup) | with GPU s/iter (speedup) |
|---|---|---|
| none (stock) | 10.73 (1.0x) | 2.87 (3.7x) |
| W | 7.19 (1.5x) | 2.11 (5.1x) |
| Z | 10.38 (1.0x) | 2.24 (4.8x) |
| Tau | 10.06 (1.1x) | 2.85 (3.8x) |
| R2 | 11.16 (1.0x) | 2.88 (3.7x) |
| YELBO | 11.31 (0.9x) | 3.04 (3.5x) |
| W+Z (fix 1) | 1.56 (6.9x) | 0.64 (17x) |
| Tau+R2+YELBO (fix 2) | 8.55 (1.3x) | 2.68 (4.0x) |
| W+Z+Tau | 0.77 (14x) | 0.31 (35x) |
| W+Z+Tau+R2 | 0.31 (34x) | 0.24 (44x) |
| all five | 0.37 (29x) | 0.26 (41x) |

Size S (10,000 + 5,000, K = 40), stock 1.05 s/iter: all five 0.05 s (22x), W+Z+Tau 0.04 s
(25x), W+Z 0.21 s (5x), stock `gpu_mode` 0.41 s (2.5x); here the GPU variants of the full
rewrite are *slower* than the CPU ones (0.11 s), since the matrices are too small to pay for
the transfers.

What the numbers say:

- **The changes only pay off together.** Each update is one of several O(N D K) or
  O(N D K^2) passes per iteration; removing one leaves the others dominant (Z alone: 1.0x,
  Tau alone: 1.1x, R2 or YELBO alone: none). W+Z removes the K^2 terms (7x at M, 15x at L);
  Tau and R2 then remove the remaining N x D passes (another 4-5x); YELBO adds little because
  stock's TauTrick is already cheap and the remaining N x D mask count is all it saves.
- **Memory falls slightly** with the full CPU rewrite (7.8 vs 8.2 GB at L), since no N x D
  temporaries are built; GPU variants raise host memory by 1-3 GB (CUDA context and pinned
  copies).
- **The GPU is worth 1.5-2x on top of the full rewrite at L, nothing at S.** Stock `gpu_mode`
  alone gives 3.2x at L; the CPU rewrite alone gives 31x.
- Full fit to convergence (real S): stock 311 s -> 6 s (float64), 127 s -> 4 s (float32).

### The fork itself (fast_updates=True, float32, same real data)

| size | CPU s/iter (stock) | GPU s/iter (stock `gpu_mode`) | peak RSS CPU / GPU |
|---|---|---|---|
| M: 50,000 + 9,512, K = 60 | 0.41 (10.7) | 0.32 (2.97) | 2.0 / 2.4 GB |
| L: 200,000 + 9,512, K = 100 | 2.45 (76.8) | 1.69 (24.1) | 7.7 / 8.6 GB |

`pytest tests/` in the fork: 19 passed (10 data sets on the CPU in float64 to 1e-9, float32,
no factor dropping, 4 data sets on the GPU, and a check that each view's data is uploaded to
the GPU exactly once per fit).
