"""Channel weights from label-only probes (never touches novel data).

Two probes score how well each channel would separate genuinely novel data,
using only labelled ("named") windows:

* leave-one-class-out probe: one class is treated as unknown and removed from
  the fit; the channel is scored on how well it flags that class in a held-out
  validation group.
* blend probe: pseudo-novel windows are synthesised as convex blends of two
  named classes in a validation pool; the channel is scored on flagging blends
  against pure windows.

The two AUC estimates are averaged, converted to weights proportional to the
excess over chance, and capped so that no single channel exceeds ``max_weight_share``.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import roc_auc_score

from .conformal import per_class_pvalue, precision, maha_per_class, pca_dim


def auc_to_weights_excess(mean_aucs, floor=0.5):
    excess = {k: max(v - floor, 0.0) for k, v in mean_aucs.items()}
    total = sum(excess.values())
    if total <= 0:
        return {k: 1.0 / len(mean_aucs) for k in mean_aucs}, excess
    return {k: excess[k] / total for k in excess}, excess


def cap_weight_share(weights, max_share=0.65):
    """Cap the largest weight at max_share; redistribute the remainder in
    proportion to the other weights (equally if they are all zero)."""
    if not weights:
        return weights
    top = max(weights, key=weights.get)
    if weights[top] <= max_share:
        return dict(weights)
    others = {k: v for k, v in weights.items() if k != top}
    remainder = 1.0 - max_share
    tot = sum(others.values())
    if tot > 0:
        red = {k: remainder * (v / tot) for k, v in others.items()}
    else:
        red = {k: remainder / len(others) for k in others} if others else {}
    out = {top: max_share}
    out.update(red)
    return out


def select_blend_val_units(class_ids, y_named, group_named, min_per_class=3, min_neg=5):
    """Pick the validation pool: the largest groups, added until at least two
    classes have >= min_per_class windows in the pool (always leaving one group out)."""
    uniq, counts = np.unique(group_named, return_counts=True)
    order = uniq[np.argsort(-counts)]
    chosen = []
    for u in order:
        if len(chosen) >= len(order) - 1:
            break
        chosen.append(u)
        val = np.isin(group_named, chosen)
        ok = [c for c in class_ids if int((y_named[val] == c).sum()) >= min_per_class]
        if len(ok) >= 2 and val.sum() >= min_neg:
            return chosen
    return chosen


def _scaffold(Ztr, ytr, kind, cfg, max_dim):
    """Returns (score_fn, d_tr_all, K, pca_transform)."""
    pca = None
    if Ztr.shape[1] > max_dim:
        pca = PCA(n_components=pca_dim(Ztr.shape[0], Ztr.shape[1], max_dim), random_state=0).fit(Ztr)
        Ztr = pca.transform(Ztr)
    lda = LDA(solver="lsqr", shrinkage=cfg.shrink).fit(Ztr, ytr)
    if kind == "discriminative":
        def score(A):
            return 1.0 - lda.predict_proba(A)
    else:
        mus, cov = lda.means_, lda.covariance_
        P = precision(cov, cfg.shrink)

        def score(A):
            return maha_per_class(A, mus, P)
    tf = (lambda A: pca.transform(A)) if pca is not None else (lambda A: A)
    return score, score(Ztr), len(lda.classes_), tf


def loco_probe(channels, yn, gn, is_named_idx, cfg):
    """Leave-one-class-out probe on the largest group. Returns (aucs, n_classes_used)."""
    uniq, counts = np.unique(gn, return_counts=True)
    if len(uniq) == 0:
        return {k: 0.5 for k in channels}, 0
    val_group = uniq[np.argmax(counts)]
    val_mask = gn == val_group
    aucs = {k: [] for k in channels}
    n_used = 0
    for c in sorted(set(yn.tolist())):
        pos = val_mask & (yn == c)
        neg = val_mask & (yn != c)
        train = ~val_mask & (yn != c)
        if pos.sum() < 5 or neg.sum() < 5 or train.sum() < 20:
            continue
        classes_present = sorted(set(yn[train].tolist()))
        if len(classes_present) < 2:
            continue
        relabel = {a: i for i, a in enumerate(classes_present)}
        ytr = np.array([relabel[a] for a in yn[train]])
        K = len(classes_present)
        yv = (yn[val_mask] == c).astype(int)
        if yv.sum() < 2 or len(yv) - yv.sum() < 2:
            continue
        used = False
        for name, (Zfull, kind) in channels.items():
            Zn = Zfull[is_named_idx]
            try:
                score, d_tr, _, tf = _scaffold(Zn[train], ytr, kind, cfg, cfg.max_dim)
            except Exception:
                continue
            pv = per_class_pvalue(score(tf(Zn[val_mask])), d_tr, ytr, K, cfg.min_class_n)
            aucs[name].append(roc_auc_score(yv, 1 - pv))
            used = True
        if used:
            n_used += 1
    return {k: float(np.mean(v)) if v else 0.5 for k, v in aucs.items()}, n_used


def blend_probe(channels, yn, gn, is_named_idx, cfg, val_units):
    """Blend probe. Returns (aucs, n_pairs_used)."""
    if len(np.unique(gn)) == 0:
        return {k: 0.5 for k in channels}, 0
    val_mask = np.isin(gn, val_units)
    train = ~val_mask
    classes_present = sorted(set(yn[train].tolist()))
    if len(classes_present) < 2:
        return {k: 0.5 for k in channels}, 0
    relabel = {a: i for i, a in enumerate(classes_present)}
    ytr = np.array([relabel[a] for a in yn[train]])
    K = len(classes_present)
    val_classes = sorted(set(yn[val_mask].tolist()) & set(classes_present))
    pairs = [(a, b) for i, a in enumerate(val_classes) for b in val_classes[i + 1:]]
    if not pairs:
        return {k: 0.5 for k in channels}, 0
    rng = np.random.RandomState(cfg.probe_seed)
    aucs = {k: [] for k in channels}
    used_pairs = set()
    yv_all = yn[val_mask]
    for name, (Zfull, kind) in channels.items():
        Zn = Zfull[is_named_idx]
        try:
            score, d_tr, _, tf = _scaffold(Zn[train], ytr, kind, cfg, cfg.max_dim)
        except Exception:
            continue
        Zval = tf(Zn[val_mask])
        for a, b in pairs:
            x1, x2 = Zval[yv_all == a], Zval[yv_all == b]
            neg = Zval[np.isin(yv_all, [a, b])]
            if len(x1) < 3 or len(x2) < 3 or len(neg) < 5:
                continue
            nb = min(cfg.n_blend_per_pair, len(x1), len(x2))
            i1 = rng.choice(len(x1), nb, replace=len(x1) < nb)
            i2 = rng.choice(len(x2), nb, replace=len(x2) < nb)
            al = rng.uniform(cfg.blend_alpha_lo, cfg.blend_alpha_hi, size=nb)
            blend = al[:, None] * x1[i1] + (1 - al[:, None]) * x2[i2]
            pv_b = per_class_pvalue(score(blend), d_tr, ytr, K, cfg.min_class_n)
            pv_n = per_class_pvalue(score(neg), d_tr, ytr, K, cfg.min_class_n)
            y = np.concatenate([np.ones(len(pv_b)), np.zeros(len(pv_n))])
            if y.sum() < 2 or len(y) - y.sum() < 2:
                continue
            aucs[name].append(roc_auc_score(y, 1 - np.concatenate([pv_b, pv_n])))
            used_pairs.add((a, b))
    return {k: float(np.mean(v)) if v else 0.5 for k, v in aucs.items()}, len(used_pairs)


def estimate_weights(channels, y, groups, cfg, val_units=None):
    """channels: name -> (feature matrix over all N windows, 'density'|'discriminative').
    y: (N,) class index for labelled windows, -1 otherwise. groups: (N,) group ids.
    """
    is_named_idx = np.where(y >= 0)[0]
    yn, gn = y[is_named_idx], groups[is_named_idx]
    if val_units is None:
        val_units = select_blend_val_units(sorted(set(yn.tolist())), yn, gn)
    aucs_loco, n_loco = loco_probe(channels, yn, gn, is_named_idx, cfg)
    aucs_blend, n_pairs = blend_probe(channels, yn, gn, is_named_idx, cfg, val_units)
    if n_pairs > 0:
        combined = {k: 0.5 * (aucs_loco[k] + aucs_blend[k]) for k in aucs_loco}
        raw, excess = auc_to_weights_excess(combined, cfg.weight_floor)
    else:
        combined = dict(aucs_loco)
        raw, excess = {k: 1.0 / len(channels) for k in channels}, {k: 0.0 for k in channels}
    capped = cap_weight_share(raw, cfg.max_weight_share)
    return dict(raw=raw, capped=capped, aucs_loco=aucs_loco, aucs_blend=aucs_blend,
                combined=combined, excess=excess, n_loco_classes=n_loco, n_pairs=n_pairs,
                val_units=[int(u) if float(u).is_integer() else u for u in val_units])
