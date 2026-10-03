"""E-value fusion with per-window adaptive scale weighting.

Each channel's conformal p-value p is converted to an e-value with the
Vovk-Wang calibrator e = kappa * p^(kappa-1) (valid under arbitrary dependence
between channels). The TC-REJECT score is the weighted arithmetic mean of the
channels' e-values; a window is rejected when E >= 1/alpha.

Adaptive scale weighting. A single global weight cannot serve both novelty
kinds for which the scale channel is uninformative and those for which it is
dominant. The scale channel therefore receives a per-window weight that ramps
from ``orth_wmin`` to ``orth_wmax`` with the window's residual distance from the
affine hull spanned by the named-class centroids in scale-feature space (the
line through the two centroids when there are two classes). The remaining
weight is shared by the other channels in proportion to their probe weights.
Because the weight depends on the window's own features, this step does not
inherit the classical fixed-weights validity guarantee.
"""
import numpy as np


def p_to_e(p, kappa=0.4):
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    return kappa * p ** (kappa - 1.0)


def combine_eavg(pvals, weights, kappa=0.4):
    """Weighted mean of e-values. pvals: list of (N,) arrays; weights: same length."""
    weights = np.asarray(weights, dtype=float)
    E = np.zeros_like(np.asarray(pvals[0]), dtype=float)
    for w, p in zip(weights, pvals):
        E = E + w * p_to_e(p, kappa=kappa)
    return E


def combine_bonferroni_min(pvals):
    k = len(pvals)
    return np.minimum(1.0, k * np.minimum.reduce(pvals))


def combine_cauchy(pvals, weights=None):
    k = len(pvals)
    if weights is None:
        weights = np.full(k, 1.0 / k)
    eps = 1e-10
    T = np.zeros_like(pvals[0], dtype=float)
    for w, p in zip(weights, pvals):
        T = T + w * np.tan(np.pi * (0.5 - np.clip(p, eps, 1.0 - eps)))
    return 0.5 - np.arctan(T) / np.pi


# ------------------------------------------------------------ adaptive scale
def class_means(Z, y):
    """Centroids of the labelled classes in feature space, ordered by class index."""
    ks = sorted(set(y[y >= 0].tolist()))
    return np.stack([Z[y == k].mean(axis=0) for k in ks], axis=0)


def orth_dist_from_hull(X, means):
    """Distance of each row of X from the affine hull of the class centroids.

    With two classes this is the perpendicular distance from the line through
    the two centroids. With K classes it is the residual after projecting onto
    the (K-1)-dimensional affine subspace through the K centroids.
    """
    X = np.asarray(X, dtype=float)
    means = np.asarray(means, dtype=float)
    v = X - means[0]
    if len(means) < 2:
        return np.linalg.norm(v, axis=1)
    B = (means[1:] - means[0]).T                       # (D, K-1)
    U, s, _ = np.linalg.svd(B, full_matrices=False)
    keep = s > 1e-12 * max(s.max(), 1e-300)
    U = U[:, keep]
    resid = v - (v @ U) @ U.T
    return np.linalg.norm(resid, axis=1)


def scale_weight_from_orth(orth, lo, hi, w_min=0.0, w_max=0.55):
    orth = np.asarray(orth, dtype=float)
    t = np.clip((orth - lo) / (hi - lo), 0.0, 1.0)
    t = t * t * (3 - 2 * t)
    return w_min + t * (w_max - w_min)


def scale_weight_auto(orth, lo_pct=25.0, hi_pct=75.0, w_min=0.0, w_max=0.55):
    """Ramp with knees at percentiles of this run's own residual distances.
    Returns (w_scale, lo, hi)."""
    orth = np.asarray(orth, dtype=float)
    lo, hi = np.percentile(orth, [lo_pct, hi_pct])
    if hi <= lo:
        hi = lo + 1e-6
    return scale_weight_from_orth(orth, lo, hi, w_min, w_max), lo, hi


def combine_eavg_adaptive_scale(p_scale, w_scale, other_pvals, other_weights, kappa=0.4):
    """One channel ('scale') gets a per-window weight; the remainder is split over
    the other channels in proportion to their fixed weights."""
    w_scale = np.clip(np.asarray(w_scale, dtype=float), 0.0, 1.0)
    remainder = 1.0 - w_scale
    E = w_scale * p_to_e(p_scale, kappa=kappa)
    total_other = sum(other_weights.get(k, 0.0) for k in other_pvals)
    if total_other > 0:
        for k, p_k in other_pvals.items():
            E = E + remainder * (other_weights.get(k, 0.0) / total_other) * p_to_e(p_k, kappa=kappa)
    else:
        m = len(other_pvals)
        for k, p_k in other_pvals.items():
            E = E + remainder * (1.0 / m) * p_to_e(p_k, kappa=kappa)
    return E


def tc_reject_score(p, weights, Zscale, y, cfg, have):
    """Full TC-REJECT score for the windows in ``have``.

    p: dict channel -> (N,) p-values (all five channels).
    weights: capped probe weights, dict channel -> float (sum 1).
    Zscale: (N, D) scale features; y: (N,) class index (-1 unlabelled).
    Returns (E over ``have`` windows, info dict).
    """
    means = class_means(Zscale, y)
    orth = orth_dist_from_hull(Zscale[have], means)
    w_scale, lo, hi = scale_weight_auto(orth, cfg.orth_lo_pct, cfg.orth_hi_pct,
                                        cfg.orth_wmin, cfg.orth_wmax)
    others = [k for k in p if k != "scale"]
    other_w = {k: weights[k] for k in others}
    E = combine_eavg_adaptive_scale(p["scale"][have], w_scale, {k: p[k][have] for k in others},
                                    other_w, kappa=cfg.kappa)
    info = dict(orth_lo=float(lo), orth_hi=float(hi), w_scale_mean=float(w_scale.mean()),
                w_scale_min=float(w_scale.min()), w_scale_max=float(w_scale.max()),
                n_classes_in_hull=int(len(means)))
    return E, info
