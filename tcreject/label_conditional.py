"""Label-conditional scores: does a window fit the class its label names?

TC-REJECT scores a window against its NEAREST labelled class, so a window that looks like another labelled class is not novel. When the question is
whether a window's OWN label is right (contamination inside a labelled class), the relevant quantity is the fit to the window's own label. This module
computes it with the same LDA / Mahalanobis scaffold as the compound-block purification of the AJILE adapter (``ajile.refine_contam_windows``:
scale features, LDA shrinkage 0.1, precision with the extra ridge, conformal p-value against the fit windows), but OUT OF FOLD: the fold's fit windows
are the labelled windows of the training groups, the windows scored are those of the held-out groups (the folds of the dataset).

Returns arrays over all windows (NaN / -1 where no fold scored the window):
    k_star     index of the nearest class (smallest Mahalanobis distance)
    p_nearest  conformal p-value against the nearest class (calibrated on the in-sample own-class distances of the fit windows, as in the purification)
    p_own      conformal p-value of the window's OWN labelled class
    post       (N, K) LDA posterior probabilities
    d          (N, K) Mahalanobis distance to each class
"""
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from .conformal import per_class_pvalue, rank_ge
from .features import feat_scale


def own_class_scores(W, y, folds, shrink=0.1, min_class_n=5):
    X = feat_scale(W)
    N, K = len(X), int(y.max()) + 1
    out = dict(k_star=np.full(N, -1), p_nearest=np.full(N, np.nan), p_own=np.full(N, np.nan),
               post=np.full((N, K), np.nan), d=np.full((N, K), np.nan))
    for f in folds:
        fit, tgt = f.fit, f.target
        yl = y[fit]
        if len(fit) < 20 or len(set(yl.tolist())) < K:
            print(f"[label-conditional] fold with held-out groups {f.held_groups} skipped (fit set lacks a class)")
            continue
        lda = LDA(solver="lsqr", shrinkage=shrink).fit(X[fit], yl)
        mus, cov = lda.means_, lda.covariance_
        D = cov.shape[0]
        P = np.linalg.inv(cov + shrink * np.trace(cov) / D * np.eye(D))

        def maha(A):
            d = A[:, None, :] - mus[None]
            return np.einsum("nkd,de,nke->nk", d, P, d)
        d_te, d_tr = maha(X[tgt]), maha(X[fit])
        out["d"][tgt] = d_te
        out["k_star"][tgt] = d_te.argmin(axis=1)
        out["p_nearest"][tgt] = per_class_pvalue(d_te, d_tr, yl, K, min_class_n)
        out["post"][tgt] = lda.predict_proba(X[tgt])
        y_t = y[tgt]
        for k in range(K):
            m = y_t == k
            if m.any():
                calib = np.sort(d_tr[yl == k, k])
                out["p_own"][tgt[m]] = (1.0 + rank_ge(calib, d_te[m, k])) / (1.0 + len(calib))
    return out
