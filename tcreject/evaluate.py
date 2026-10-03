"""Evaluation: per-target AUROC for every score, block-bootstrap CIs, supervised ceiling."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold


def block_bootstrap_ci(y, s, groups, n=300, seed=0):
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    index = {g: np.where(groups == g)[0] for g in uniq}
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, len(uniq), replace=True)
        idx = np.concatenate([index[g] for g in pick])
        if 0 < y[idx].sum() < len(idx):
            vals.append(roc_auc_score(y[idx], s[idx]))
    if not vals:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def oracle_auroc(Zcat, pos, groups, n_splits=5):
    """Supervised ceiling: logistic regression separating novel from named, group-CV."""
    yy = pos.astype(int)
    n_eff = min(n_splits, len(np.unique(groups)))
    if n_eff < 2:
        return float("nan")
    pred = np.full(len(yy), np.nan)
    for tr, te in GroupKFold(n_eff).split(Zcat, yy, groups=groups):
        if len(set(yy[tr].tolist())) < 2:
            continue
        Ztr, Zte = Zcat[tr], Zcat[te]
        if Ztr.shape[1] > 200:
            pca = PCA(n_components=min(200, max(Ztr.shape[0] - 1, 1)), random_state=0).fit(Ztr)
            Ztr, Zte = pca.transform(Ztr), pca.transform(Zte)
        clf = LogisticRegression(max_iter=2000, C=0.5).fit(Ztr, yy[tr])
        pred[te] = clf.predict_proba(Zte)[:, 1]
    ok = ~np.isnan(pred)
    if ok.sum() < 30 or len(set(yy[ok].tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(yy[ok], pred[ok]))


def auroc_table(targets, truth_named, scores, groups, Zcat, n_boot, headline, min_n, oracle=True):
    """targets: list of (name, positive mask) over the scored windows.
    truth_named: mask of genuinely named scored windows (the negatives).
    scores: dict name -> array over the scored windows (larger = more novel).
    Returns a list of rows (one per target).
    """
    rows = []
    for name, pos in targets:
        if pos.sum() < min_n:
            continue
        m = pos | truth_named
        yy = pos[m].astype(int)
        if len(set(yy.tolist())) < 2:
            continue
        row = dict(target=name, n_pos=int(pos.sum()), n_neg=int(m.sum() - pos.sum()))
        row["auroc"] = {k: float(roc_auc_score(yy, s[m])) for k, s in scores.items()}
        if headline in scores and n_boot > 0:
            lo, hi = block_bootstrap_ci(yy, scores[headline][m], groups[m], n=n_boot)
            row["headline_ci"] = [lo, hi]
        if oracle and Zcat is not None:
            row["oracle"] = oracle_auroc(Zcat[m], yy.astype(bool), groups[m])
        rows.append(row)
    return rows
