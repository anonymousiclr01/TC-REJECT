"""Class-conditional conformal p-values for one feature space.

A *fold* supplies the windows a model may be fitted on (labelled "named"
windows) and the windows to score (the target set). For each fold we fit an
LDA scaffold on the fit windows, compute a per-class nonconformity score, and
turn each target window's score into a conformal p-value against the fit
windows of its nearest class. p-values from several folds are averaged.

Two kinds of nonconformity score:

* ``density``        0.5 * Mahalanobis distance to a class mean (shared covariance).
* ``discriminative`` 1 - LDA posterior probability of the class (label-relative).

Calibration distances of the fit windows to their own class are, by default,
leave-one-group-out (each group's own contribution to the class mean is
removed), so calibration is not optimistically biased by self-inclusion.
"""
from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp
from scipy.stats import weibull_min
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA


@dataclass
class Fold:
    fit: np.ndarray        # indices of labelled windows the fold may fit on
    target: np.ndarray     # indices of windows the fold scores
    held_groups: tuple = ()  # groups excluded from any fitting/pre-training in this fold


# ---------------------------------------------------------------- primitives
def rank_ge(calib_sorted, s_te):
    return len(calib_sorted) - np.searchsorted(calib_sorted, s_te, side="left")


def per_class_pvalue(d_te_all, d_tr_all, y_tr, K, min_class_n=5):
    """Conformal p-value of each test window against its nearest class.

    d_te_all: (n_te, K) score of each test window to each class (small = typical).
    d_tr_all: (n_tr, K) same for the fit windows; column y_tr[i] is the own-class score.
    """
    n_te = d_te_all.shape[0]
    k_star = d_te_all.argmin(axis=1)
    s_te = d_te_all[np.arange(n_te), k_star]
    pval = np.empty(n_te)
    pooled_calib = None
    for k in range(K):
        mask = k_star == k
        if not mask.any():
            continue
        calib_k = d_tr_all[y_tr == k, k]
        if len(calib_k) < min_class_n:
            if pooled_calib is None:
                pooled_calib = np.sort(d_tr_all.min(axis=1))
            calib_k = pooled_calib
        else:
            calib_k = np.sort(calib_k)
        pval[mask] = (1.0 + rank_ge(calib_k, s_te[mask])) / (1.0 + len(calib_k))
    return pval


def own_class_loo_maha(Z_tr, y_tr, group_tr, mus, P):
    """Own-class Mahalanobis of each fit window with its group's contribution
    to the class mean removed (leave-one-group-out class mean)."""
    n_tr = Z_tr.shape[0]
    own = np.empty(n_tr)
    for k in range(mus.shape[0]):
        idx = np.where(y_tr == k)[0]
        n_k = len(idx)
        if n_k < 2:
            if n_k == 1:
                d = Z_tr[idx] - mus[k]
                own[idx] = 0.5 * np.einsum("id,de,ie->i", d, P, d)
            continue
        Xk, gk = Z_tr[idx], group_tr[idx]
        for g in np.unique(gk):
            gidx = gk == g
            n_g = int(gidx.sum())
            if n_g >= n_k:
                Xg = Xk[gidx]
                mu_loo = (n_k * mus[k] - Xg) / max(n_k - 1, 1)
                diff = Xg - mu_loo
            else:
                mu_g_mean = Xk[gidx].mean(0)
                mu_loo = (n_k * mus[k] - n_g * mu_g_mean) / (n_k - n_g)
                diff = Xk[gidx] - mu_loo
            own[idx[gidx]] = 0.5 * np.einsum("id,de,ie->i", diff, P, diff)
    return own


def precision(cov, shrink):
    D = cov.shape[0]
    return np.linalg.inv(cov + shrink * np.trace(cov) / D * np.eye(D))


def maha_per_class(A, mus, P):
    d = A[:, None, :] - mus[None]
    return 0.5 * np.einsum("nkd,de,nke->nk", d, P, d)


def pca_dim(n_samples, n_features, max_dim):
    """PCA target dimension: ``max_dim``, but never more than the fit set can support."""
    return min(max_dim, max(n_samples - 1, 1), n_features)


def maybe_pca(Zfit, Ztgt, max_dim):
    if Zfit.shape[1] > max_dim:
        pca = PCA(n_components=pca_dim(Zfit.shape[0], Zfit.shape[1], max_dim), random_state=0).fit(Zfit)
        return pca.transform(Zfit), pca.transform(Ztgt)
    return Zfit, Ztgt


# ---------------------------------------------------------------- one fold
def score_fold(Zfit, yfit, gfit, Ztgt, kind, cfg, baselines=False, weibull_tail=25):
    """Scores Ztgt against an LDA scaffold fitted on (Zfit, yfit).

    Returns a dict with 'p' (conformal p-value per target window) and, for
    density scoring with baselines=True, the standard open-set baselines
    ('msp', 'maha_raw', 'energy', 'openmax').
    """
    lda = LDA(solver="lsqr", shrinkage=cfg.shrink).fit(Zfit, yfit)
    K = len(lda.classes_)
    yl = np.searchsorted(lda.classes_, yfit)
    out = {}
    if kind == "discriminative":
        d_te = 1.0 - lda.predict_proba(Ztgt)
        d_tr = 1.0 - lda.predict_proba(Zfit)
        out["p"] = per_class_pvalue(d_te, d_tr, yl, K, cfg.min_class_n)
        return out
    mus, cov = lda.means_, lda.covariance_
    P = precision(cov, cfg.shrink)
    d_te = maha_per_class(Ztgt, mus, P)
    d_tr = maha_per_class(Zfit, mus, P)
    if cfg.calibration == "group_loo":
        d_tr[np.arange(len(Zfit)), yl] = own_class_loo_maha(Zfit, yl, gfit, mus, P)
    out["p"] = per_class_pvalue(d_te, d_tr, yl, K, cfg.min_class_n)
    if baselines:
        pp = lda.predict_proba(Ztgt)
        out["maha_raw"] = d_te.min(1)
        out["msp"] = 1.0 - pp.max(axis=1)
        logits = lda.decision_function(Ztgt)
        if logits.ndim == 1:
            logits = np.stack([-logits, logits], axis=1)
        out["energy"] = -logsumexp(logits, axis=1)
        wb = {}
        for k in range(K):
            dk = d_tr[yl == k, k]
            dk = dk[np.argsort(dk)[-min(weibull_tail, len(dk)):]]
            if len(dk) < 5 or dk.max() <= 0:
                wb[k] = None
                continue
            try:
                shape_k, _, scale_k = weibull_min.fit(dk, floc=0)
                wb[k] = (shape_k, scale_k)
            except Exception:
                wb[k] = None
        om = np.zeros(len(Ztgt))
        for k in range(K):
            if wb[k] is None:
                continue
            om += pp[:, k] * weibull_min.cdf(d_te[:, k], wb[k][0], loc=0, scale=wb[k][1])
        out["openmax"] = om
    return out


# ---------------------------------------------------------------- all folds
def channel_pvalues(Z, y, folds, groups, kind, cfg, baselines=False):
    """Out-of-fold conformal p-values (mean over the folds scoring each window).

    y: (N,) integer class index for labelled windows, -1 elsewhere.
    Returns dict of (N,) arrays; entries are NaN for windows no fold scored.
    """
    n = len(Z)
    acc, hits = {}, np.zeros(n)
    for f in folds:
        fit, tgt = f.fit, f.target
        if len(fit) < 20 or len(set(y[fit].tolist())) < 2:
            continue
        Zfit, Ztgt = maybe_pca(Z[fit], Z[tgt], cfg.max_dim)
        out = score_fold(Zfit, y[fit], groups[fit], Ztgt, kind, cfg, baselines)
        for name, v in out.items():
            acc.setdefault(name, np.zeros(n))[tgt] += v
        hits[tgt] += 1
    res = {}
    scored = hits > 0
    names = list(acc) or (["p"] + (["msp", "maha_raw", "energy", "openmax"] if baselines and kind == "density" else []))
    for name in names:
        arr = np.full(n, np.nan)
        if name in acc:
            arr[scored] = acc[name][scored] / hits[scored]
        res[name] = arr
    return res
