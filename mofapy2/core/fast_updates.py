"""Exact fast forms of the Gaussian-likelihood updates (optiMOFA).

When the noise precision of a feature is the same for all samples -- one group and no missing
values in the view -- the sums over samples in the W, Z and Tau updates, the variance
explained and the likelihood term of the ELBO can be written with the K x K Gram matrix
C = E[Z]^T E[Z] and the D x K cross product P = Y^T E[Z]. This removes the O(N D K^2) terms
and every N x D temporary; the results equal the stock updates up to floating-point
summation order (derivations: docs/fast_updates.md).

Every function here is called from the corresponding stock method, which keeps running
unchanged whenever `applies` is False (several groups, missing values, stochastic
inference, or `fast_updates=False` in the training options).

With gpu_mode, the products with the data run on the GPU and each view's data matrix stays
resident there for the whole fit (see gpu_utils.register_resident).
"""

from __future__ import annotations

import numpy as np

from mofapy2.core import gpu_utils

# switched by the nodes' precompute() from the training option "fast_updates" (like gpu_mode)
enabled = True

# rows of W updated per block (bounds the D x K temporaries)
W_BLOCK, W_BLOCK_GPU = 16_384, 65_536


# ---- applicability -------------------------------------------------------------------------
def no_mask(ynode) -> bool:
    nm = getattr(ynode, "_fast_no_mask", None)
    if nm is None:
        nm = not bool(np.asarray(ynode.getMask(full=True)).any())
        ynode._fast_no_mask = nm
    return nm


def applies(ynode, taunode, ix=None) -> bool:
    """True if the fast form is exact for this view: full batch, one group, no missing values."""
    if not enabled or ix is not None:
        return False
    if getattr(ynode, "mini_batch", None) is not None:
        return False
    if taunode.n_groups != 1 or ynode.n_groups != 1:
        return False
    return no_mask(ynode)


# ---- helpers ---------------------------------------------------------------------------------
def _xp():
    if gpu_utils.gpu_mode:
        import cupy as cp

        return cp
    return np


def _host(x):
    return x.get() if hasattr(x, "get") else x


def _f64(x):
    return np.asarray(x, dtype=np.float64)


def _cols(x, shape):
    return np.array(np.broadcast_to(_f64(x), shape), dtype=np.float64, order="F")


def _tau(taunode):
    return np.asarray(taunode.Q.getExpectation()[0], dtype=np.float64)


class _ViewData:
    """Per-view data products; Y stays in its own dtype (float32 if use_float32)."""

    def __init__(self, y, gpu):
        self.gpu = gpu
        self.dtype = y.dtype
        self.sum_y2 = np.einsum("nd,nd->d", y, y, dtype=np.float64)
        self.y = gpu_utils.array(y) if gpu else y  # resident copy if registered

    def ytz(self, z, cols=slice(None), on_device=False):
        """Y[:, cols]^T Z in float64 (D x K)."""
        if self.gpu:
            import cupy as cp

            out = (self.y[:, cols].T @ cp.asarray(z, dtype=self.dtype)).astype(cp.float64)
            return out if on_device else cp.asnumpy(out)
        return (self.y[:, cols].T @ np.asarray(z).astype(self.dtype)).astype(np.float64)

    def yw(self, w):
        """Y W in float64 (N x K)."""
        if self.gpu:
            import cupy as cp

            return cp.asnumpy(self.y @ cp.asarray(w, dtype=self.dtype)).astype(np.float64)
        return (self.y @ w.astype(self.dtype)).astype(np.float64)


def view_data(ynode) -> _ViewData:
    gpu = bool(gpu_utils.gpu_mode)
    v = getattr(ynode, "_fast_view", None)
    if v is None or v.gpu != gpu:
        v = _ViewData(ynode.getExpectation(), gpu)
        ynode._fast_view = v
    return v


def reset(ynode) -> None:
    """Forget cached per-view data (called from Y_Node.precompute)."""
    ynode._fast_view = None
    ynode._fast_no_mask = None


# ---- W (spike-and-slab) ----------------------------------------------------------------------
def sw_update(node, ro=1.0):
    """SW_Node update. Per factor k and feature d, with t_d = E[tau_d]:
    sum_n tau Z_k Z_j W_j = t_d [(W C)_dk - W_dk C_kk],  sum_n tau Y Z_k = t_d P_dk."""
    mb = node.markov_blanket
    view = view_data(mb["Y"])
    zexp = mb["Z"].getExpectations()
    z = _f64(zexp["E"])
    t = _tau(mb["Tau"])
    if "AlphaW" in mb:
        alpha = mb["AlphaW"].getExpectation(expand=True)
    else:
        alpha = 1.0 / node.P.params["var_B1"]
    theta = mb["ThetaW"].getExpectations(expand=True)
    q = node.Q.getParameters()
    shape = tuple(node.dim)
    alpha = np.broadcast_to(alpha, shape)
    sw_all = node.Q.getExpectations()["E"]
    qmean_all, qvar_all, qtheta_all = q["mean_B1"], q["var_B1"], q["theta"]
    xp = _xp()
    ztz = xp.asarray(z.T @ z)
    sum_zz = xp.asarray(_f64(zexp["E2"]).sum(axis=0))
    z_dev = xp.asarray(z)
    block = W_BLOCK_GPU if gpu_utils.gpu_mode else W_BLOCK
    for d0 in range(0, shape[0], block):
        rows = slice(d0, min(d0 + block, shape[0]))

        def get(a, rows=rows):
            return xp.asfortranarray(
                xp.asarray(np.ascontiguousarray(a[rows]), dtype=xp.float64)
            )

        tb = xp.asarray(t[rows])
        alpha_b = get(alpha)
        prec = alpha_b + tb[:, None] * sum_zz[None, :]
        logit0 = get(np.broadcast_to(theta["lnE"], shape)) - get(
            np.broadcast_to(theta["lnEInv"], shape)
        )
        logit0 += 0.5 * xp.log(alpha_b) - 0.5 * xp.log(prec)
        tyz = xp.asfortranarray(view.ytz(z_dev, rows, on_device=True) * tb[:, None])
        qmean, qvar, qtheta, sw = get(qmean_all), get(qvar_all), get(qtheta_all), get(sw_all)
        # coordinate ascent over factors, as stock (W[:, k] refreshed before factor k + 1)
        for k in range(shape[1]):
            resid = tyz[:, k] - tb * (sw @ ztz[:, k] - sw[:, k] * ztz[k, k])
            p = prec[:, k]
            term = logit0[:, k] + 0.5 * resid * resid / p
            qtheta[:, k] = (1 - ro) * qtheta[:, k] + ro / (1.0 + xp.exp(-term))
            qvar[:, k] = (1 - ro) * qvar[:, k] + ro / p
            qmean[:, k] = (1 - ro) * qmean[:, k] + ro * resid / p
            sw[:, k] = qtheta[:, k] * qmean[:, k]
        qmean_all[rows] = _host(qmean)
        qvar_all[rows] = _host(qvar)
        qtheta_all[rows] = _host(qtheta)
        sw_all[rows] = _host(sw)  # stock also writes the new E[W] in place
    qvar0 = q["var_B0"] * (1 - ro) + ro / alpha
    node.Q.setParameters(
        mean_B0=np.zeros(shape),
        var_B0=qvar0,
        mean_B1=qmean_all,
        var_B1=qvar_all,
        theta=qtheta_all,
    )


# ---- Z -----------------------------------------------------------------------------------------
def z_update(node, ro=1.0):
    """Z_Node update. With G = sum_m v_m W^T diag(t) W:
    bar_nk = sum_m v_m (Y diag(t) W)_nk - [(Z G)_nk - Z_nk G_kk]."""
    mb = node.markov_blanket
    ynodes, taunodes = mb["Y"].nodes, mb["Tau"].nodes
    w_all = mb["W"].getExpectations()
    mu = mb["MuZ"].get_mini_batch() if "MuZ" in mb else node.P.getParameters()["mean"]
    alpha = mb["AlphaZ"].get_mini_batch() if "AlphaZ" in mb else 1.0 / node.P.params["var"]
    q = node.Q.getParameters()
    shape = q["mean"].shape
    n_views = len(ynodes)
    weights = np.ones(n_views)
    if node.weight_views and n_views > 1:
        sizes = np.asarray([y.dim[1] for y in ynodes], dtype=float)
        weights = np.asarray([sizes.sum() / (n_views * s) for s in sizes])
        weights = weights / weights.sum() * n_views
    foo = np.zeros(shape[1])
    pre = np.zeros(shape)
    gram = np.zeros((shape[1], shape[1]))
    for m in range(n_views):
        t = _tau(taunodes[m])
        w, ww = _f64(w_all[m]["E"]), _f64(w_all[m]["E2"])
        tw = t[:, None] * w
        foo += weights[m] * (t @ ww)
        pre += weights[m] * view_data(ynodes[m]).yw(tw)
        gram += weights[m] * (w.T @ tw)
    qmean, qvar = _cols(q["mean"], shape), _cols(q["var"], shape)
    alpha_f, mu_f = _cols(alpha, shape), _cols(mu, shape)
    for k in range(shape[1]):
        bar = pre[:, k] - (qmean @ gram[:, k] - qmean[:, k] * gram[k, k])
        qvar[:, k] = 1.0 / (alpha_f[:, k] + foo[k])
        qmean[:, k] = qvar[:, k] * (bar + alpha_f[:, k] * mu_f[:, k])
    node.Q.setParameters(mean=np.ascontiguousarray(qmean), var=np.ascontiguousarray(qvar))


# ---- Tau ---------------------------------------------------------------------------------------
def tau_update(node, ro=1.0):
    """TauD_Node update: sum_n E[res_nd^2] = sum_n Y^2 - 2 P_d.W_d + W_d C W_d^T
    - sum_k W_dk^2 sum_n Z_nk^2 + sum_k <W2>_dk sum_n <Z2>_nk."""
    mb = node.markov_blanket
    view = view_data(mb["Y"])
    wexp, zexp = mb["W"].getExpectations(), mb["Z"].getExpectations()
    w, ww = _f64(wexp["E"]), _f64(wexp["E2"])
    z, zz = _f64(zexp["E"]), _f64(zexp["E2"])
    s = (
        view.sum_y2
        + ww @ zz.sum(axis=0)
        - (w * w) @ (z * z).sum(axis=0)
        + np.einsum("dk,dk->d", w @ (z.T @ z), w)
        - 2.0 * np.einsum("dk,dk->d", view.ytz(z), w)
    )
    p, q = node.P.getParameters(), node.Q.getParameters()
    qa, qb = q["a"] * (1 - ro), q["b"] * (1 - ro)
    qa[0, :] += ro * (p["a"][0, :] + 0.5 * float(node.n_per_group[0]))
    qb[0, :] += ro * (p["b"][0, :] + 0.5 * s)
    node.Q.setParameters(a=qa, b=qb)


# ---- variance explained ------------------------------------------------------------------------
def variance_explained(net, total=False):
    """Point-estimate R2: ||Y - Z_k W_k^T||^2 = SS - 2 W_k.P_k + C_kk ||W_k||^2,
    ||Y - Z W^T||^2 = SS - 2 tr(W^T P) + tr(W C W^T). Returns None if not applicable."""
    ynodes = net.nodes["Y"].getNodes()
    taunodes = net.nodes["Tau"].getNodes()
    if net.dim["G"] != 1 or not all(applies(y, t) for y, t in zip(ynodes, taunodes)):
        return None
    z = _f64(net.nodes["Z"].getExpectation())
    ztz = z.T @ z
    ws = net.nodes["W"].getExpectation()
    out = np.zeros(net.dim["M"]) if total else np.zeros((net.dim["M"], net.dim["K"]))
    for m, ynode in enumerate(ynodes):
        view = view_data(ynode)
        ss = float(view.sum_y2.sum())
        w = _f64(ws[m])
        cross = np.einsum("dk,dk->k", view.ytz(z), w)
        if total:
            res = ss - 2.0 * cross.sum() + float(np.einsum("dk,dk->", w @ ztz, w))
            out[m] = 1.0 - res / ss
        else:
            out[m] = 1.0 - (ss - 2.0 * cross + np.diag(ztz) * (w * w).sum(axis=0)) / ss
    return [out]


# ---- likelihood term of the ELBO --------------------------------------------------------------
def y_elbo(ynode):
    """TauTrick without the N x D mask scan (valid right after a Tau update only, i.e. when
    ynode.TauTrick is set; the caller checks that). Returns None if not applicable."""
    tau = ynode.markov_blanket["Tau"]
    if not enabled or ynode.n_groups != 1 or tau.n_groups != 1 or not no_mask(ynode):
        return None
    qexp = tau.Q.getExpectations()
    qb, pb = tau.getParameters("Q")["b"], tau.getParameters("P")["b"]
    n = float(ynode.dim[0])
    return (
        ynode.likconst
        + 0.5 * n * qexp["lnE"][0, :].sum()
        - np.dot(qexp["E"][0, :], qb[0, :] - pb[0, :])
    )
