"""Equivalence of optiMOFA's fast updates with the stock mofapy2 updates.

Every data set is fitted to convergence twice with the same seed: `fast_updates=False` (stock
mofapy2 code everywhere) and `fast_updates=True`. The fits must take the same number of
iterations, drop the same factors at the same iterations, and agree in E[W], E[Z], E[tau],
the ELBO trajectory and the variance explained up to floating-point rounding.

The data sets differ in the statistical properties that could break a derivation that only
holds approximately: feature-specific noise, K close to N, no structure at all, heavy tails
with outliers, correlated and unscaled features, sample-dependent noise, several groups,
missing values, N >> D. The last two kinds exercise the fallback to stock code.
"""

import contextlib
import io

import numpy as np
import pytest

from mofapy2.run.entry_point import entry_point


# ---- data sets (samples x features, per view) --------------------------------------------------
def _std(y):
    return (y - y.mean(0)) / y.std(0)


def _signal(rng, n, dims, rank, density=0.3):
    z = rng.normal(size=(n, rank))
    return [
        z @ (rng.normal(size=(d, rank)) * (rng.random((d, rank)) < density)).T
        for d in dims
    ]


def ds_baseline(rng):
    """low rank, sparse weights, Gaussian homoscedastic noise"""
    s = _signal(rng, 200, (800, 200), 5)
    return [_std(x + rng.normal(size=x.shape)) for x in s], None, 15


def ds_hetero_features(rng):
    """feature-specific noise SD 0.2..5, views of very different size"""
    s = _signal(rng, 180, (1500, 60), 5)
    return [_std(x + rng.normal(size=x.shape) * rng.uniform(0.2, 5, x.shape[1])) for x in s], None, 12


def ds_wide_small_n(rng):
    """N = 40 < D, K = 30 close to N"""
    s = _signal(rng, 40, (120, 30), 3)
    return [_std(x + rng.normal(size=x.shape)) for x in s], None, 30


def ds_noise_only(rng):
    """no structure: most factors are dropped"""
    return [_std(rng.normal(size=(120, d))) for d in (400, 100)], None, 10


def ds_heavy_tails(rng):
    """t(2.5) noise plus 1 % gross outliers (misspecified likelihood)"""
    out = []
    for x in _signal(rng, 150, (600, 150), 4):
        y = x + rng.standard_t(2.5, size=x.shape)
        y[rng.random(y.shape) < 0.01] += 25
        out.append(_std(y))
    return out, None, 12


def ds_correlated_unscaled(rng):
    """AR(1) rho = 0.9 noise across features, feature scales 1e-2..1e2, dense weights"""
    out = []
    for x in _signal(rng, 160, (700, 120), 4, density=1.0):
        e = np.empty_like(x)
        e[:, 0] = rng.normal(size=x.shape[0])
        for j in range(1, x.shape[1]):
            e[:, j] = 0.9 * e[:, j - 1] + np.sqrt(1 - 0.81) * rng.normal(size=x.shape[0])
        y = x + e
        out.append((y - y.mean(0)) * 10 ** rng.uniform(-2, 2, x.shape[1]))
    return out, None, 12


def ds_sample_hetero(rng):
    """two batches with noise SD 0.5 and 2, modelled as one group"""
    s = _signal(rng, 200, (600, 150), 4)
    sd = np.where(np.arange(200) < 100, 0.5, 2.0)[:, None]
    return [_std(x + rng.normal(size=x.shape) * sd) for x in s], None, 12


def ds_two_groups(rng):
    """same data modelled as two groups: the fast forms do not apply (stock code runs)"""
    views, _, k = ds_sample_hetero(rng)
    return views, [np.arange(100), np.arange(100, 200)], k


def ds_missing(rng):
    """5 % missing values in view 0: view 0 uses stock code, view 1 the fast forms"""
    views, _, k = ds_baseline(rng)
    views[0] = views[0].copy()
    views[0][rng.random(views[0].shape) < 0.05] = np.nan
    return views, None, k


def ds_tall(rng):
    """N = 1000 >> D"""
    s = _signal(rng, 1000, (300, 50), 3)
    return [_std(x + rng.normal(size=x.shape)) for x in s], None, 8


DATASETS = {
    f.__name__[3:]: f
    for f in (
        ds_baseline,
        ds_hetero_features,
        ds_wide_small_n,
        ds_noise_only,
        ds_heavy_tails,
        ds_correlated_unscaled,
        ds_sample_hetero,
        ds_two_groups,
        ds_missing,
        ds_tall,
    )
}


def make(name):
    return DATASETS[name](np.random.default_rng(sum(map(ord, name))))


# ---- fitting and comparison ---------------------------------------------------------------------
def fit(views, k, groups=None, *, fast, gpu=False, f32=False, drop=1e-4, max_iter=1000):
    dtype = np.float32 if f32 else np.float64
    if groups is None:
        data, gnames = [[np.ascontiguousarray(v, dtype=dtype)] for v in views], None
    else:
        data = [[np.ascontiguousarray(v[g], dtype=dtype) for g in groups] for v in views]
        gnames = [f"g{i}" for i in range(len(groups))]
    with contextlib.redirect_stdout(io.StringIO()):
        ent = entry_point()
        ent.set_data_options(
            scale_views=False, scale_groups=False, center_groups=True, use_float32=f32
        )
        ent.set_data_matrix(
            data,
            likelihoods=["gaussian"] * len(views),
            views_names=[f"v{i}" for i in range(len(views))],
            groups_names=gnames,
        )
        ent.set_model_options(
            factors=k, spikeslab_weights=True, ard_weights=True,
            spikeslab_factors=False, ard_factors=False,
        )
        ent.set_train_options(
            iter=max_iter, convergence_mode="fast", seed=1, dropR2=drop,
            verbose=False, quiet=True, gpu_mode=gpu, fast_updates=fast,
        )
        ent.build()
        ent.run()
    nodes = ent.model.getNodes()
    st = ent.model.getTrainingStats()
    z = nodes["Z"].getExpectation()
    return {
        "elbo": np.asarray(st["elbo"], float),
        "nf": np.asarray(st["number_factors"], float),
        "W": [np.asarray(x["E"], float) for x in nodes["W"].getExpectations()],
        "Z": np.vstack(z) if isinstance(z, list) else np.asarray(z, float),
        "tau": [np.asarray(t.Q.getExpectation(), float) for t in nodes["Tau"].getNodes()],
        "r2": [np.asarray(r, float) for r in ent.model.calculate_variance_explained()],
    }


def rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    assert a.shape == b.shape
    m = np.isfinite(b)
    assert np.array_equal(np.isfinite(a), m)
    if not m.any():
        return 0.0
    return float(np.max(np.abs(a[m] - b[m])) / max(np.max(np.abs(b[m])), 1e-300))


def assert_equivalent(got, ref, tol):
    # same convergence and factor-dropping decisions
    np.testing.assert_array_equal(got["nf"], ref["nf"])
    assert rel(got["elbo"], ref["elbo"]) < tol
    assert max(rel(a, b) for a, b in zip(got["W"], ref["W"])) < tol
    assert rel(got["Z"], ref["Z"]) < tol
    assert max(rel(a, b) for a, b in zip(got["tau"], ref["tau"])) < tol
    for a, b in zip(got["r2"], ref["r2"]):
        assert np.max(np.abs(a - b)) < tol


def _has_gpu():
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


# ---- tests --------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", list(DATASETS))
def test_fast_equals_stock_cpu(name):
    views, groups, k = make(name)
    ref = fit(views, k, groups, fast=False)
    got = fit(views, k, groups, fast=True)
    assert_equivalent(got, ref, tol=1e-9)


@pytest.mark.parametrize("name", ["baseline", "missing"])
def test_fast_equals_stock_float32(name):
    views, groups, k = make(name)
    ref = fit(views, k, groups, fast=False, f32=True)
    got = fit(views, k, groups, fast=True, f32=True)
    assert_equivalent(got, ref, tol=1e-3)


def test_fast_without_factor_dropping():
    views, groups, k = make("baseline")
    ref = fit(views, k, groups, fast=False, drop=None, max_iter=60)
    got = fit(views, k, groups, fast=True, drop=None, max_iter=60)
    assert_equivalent(got, ref, tol=1e-9)


@pytest.mark.skipif(not _has_gpu(), reason="needs cupy and a CUDA device")
@pytest.mark.parametrize("name", ["baseline", "hetero_features", "two_groups", "missing"])
def test_fast_gpu_equals_stock_cpu(name):
    from mofapy2.core import gpu_utils

    views, groups, k = make(name)
    ref = fit(views, k, groups, fast=False)
    got = fit(views, k, groups, fast=True, gpu=True)
    gpu_utils.gpu_mode = False
    assert_equivalent(got, ref, tol=1e-9)


@pytest.mark.skipif(not _has_gpu(), reason="needs cupy and a CUDA device")
def test_data_stays_resident_on_gpu():
    """Each view's data matrix is uploaded once per fit, also by the stock code paths
    (which ask for it in every W, Z and Tau update; stock mofapy2 re-uploads it each time)."""
    import cupy as cp

    from mofapy2.core import gpu_utils

    views, groups, k = make("two_groups")  # two groups: stock code paths throughout
    count = {"requests": 0, "uploads": 0}
    orig_array, orig_cp = gpu_utils.array, cp.array

    def requests(obj):
        if id(obj) in gpu_utils._resident:
            count["requests"] += 1
        return orig_array(obj)

    def uploads(obj, *a, **kw):
        if id(obj) in gpu_utils._resident:
            count["uploads"] += 1
        return orig_cp(obj, *a, **kw)

    gpu_utils.array, cp.array = requests, uploads
    try:
        got = fit(views, k, groups, fast=False, gpu=True, max_iter=10)
    finally:
        gpu_utils.array, cp.array = orig_array, orig_cp
        gpu_utils.gpu_mode = False
    # iter=10 runs 9 update rounds: mofapy2 loops over range(1, maxiter), and its stats have
    # maxiter + 1 slots (slot 0 = before training, the last one stays NaN)
    rounds = 10 - 1
    assert np.isfinite(got["nf"]).sum() == rounds + 1
    # W, Z and Tau each ask for every view's data once per round ...
    assert count["requests"] == 3 * rounds * len(views)
    # ... and it is uploaded once per view for the whole fit
    assert count["uploads"] == len(views)
