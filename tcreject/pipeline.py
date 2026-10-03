"""The TC-REJECT pipeline: one function, every dataset.

Steps
 1. z-scored scale / shape / spectral features from the raw windows
 2. out-of-fold conformal p-values for five channels:
    scale, shape, spectral (density), disc (discriminative, on spectral), deep (learned)
 3. channel weights from the leave-one-class-out and blend probes, capped
 4. e-value and rank-average fusion -> the two TC-REJECT scores
 5. baselines, labelled ablations, calibration diagnostics, per-target AUROC
"""
from dataclasses import replace

import numpy as np
from sklearn.metrics import roc_auc_score

from .config import Config, DEFAULT, CHANNELS
from .conformal import channel_pvalues
from .features import compute_feature_spaces
from .weights import estimate_weights, select_blend_val_units, auc_to_weights_excess, cap_weight_share
from .fusion import (tc_reject_score, combine_eavg, combine_bonferroni_min, combine_cauchy)
from .fusion_rank_avg import rank_avg_score
from .deep import deep_pvalues, auxiliary_embedding, torch_embedder
from .diagnostics import null_uniformity, coverage_at_alpha, format_calibration
from .evaluate import auroc_table

HEADLINE = "TC-REJECT"
RANK_HEADLINE = "TC-REJECT(rank-avg)"


def _subset_weights(w, keep, cfg):
    """Probe weights restricted to the channels in ``keep`` (ablations only; no re-training)."""
    if w["n_pairs"] > 0:
        raw, _ = auc_to_weights_excess({k: w["combined"][k] for k in keep}, cfg.weight_floor)
    else:
        raw = {k: 1.0 / len(keep) for k in keep}
    return cap_weight_share(raw, cfg.max_weight_share)


def run(ds, cfg: Config = DEFAULT, embedder=None, device="auto", verbose=True, ablations=True):
    log = print if verbose else (lambda *a, **k: None)
    if embedder is None:                       # fail fast (before any heavy work) if PyTorch is missing
        embedder = torch_embedder(cfg, device=device, verbose=False)
    y = ds.y()
    named = ds.kind == "named"
    K = len(ds.named_vocab)
    log(f"[{ds.name}] {len(ds.W)} windows, {int(named.sum())} labelled, {K} classes {ds.named_vocab}, "
        f"{len(ds.folds)} folds")
    if K < 2:
        raise ValueError("TC-REJECT needs at least two labelled classes")

    # 1. features
    spaces = compute_feature_spaces(ds.W, ds.fs, cfg, named)
    Zs, Zh, Zp = spaces["scale"], spaces["shape"], spaces["spectral"]
    log(f"  feature dims: scale={Zs.shape[1]} shape={Zh.shape[1]} spectral={Zp.shape[1]}")

    # 2. channel p-values
    P = {}
    r_scale = channel_pvalues(Zs, y, ds.folds, ds.group, "density", cfg, baselines=True)
    P["scale"] = r_scale["p"]
    P["shape"] = channel_pvalues(Zh, y, ds.folds, ds.group, "density", cfg)["p"]
    r_spec = channel_pvalues(Zp, y, ds.folds, ds.group, "density", cfg, baselines=True)
    P["spectral"] = r_spec["p"]
    P["disc"] = channel_pvalues(Zp, y, ds.folds, ds.group, "discriminative", cfg)["p"]
    log("  deep channel: per-fold encoder pre-training ...")
    P["deep"] = deep_pvalues(ds.W, y, ds.folds, ds.group, cfg, embedder, seed=cfg.seed, verbose=verbose)

    # 3. weights
    named_idx = np.where(named)[0]
    val_units = select_blend_val_units(list(range(K)), y[named_idx], ds.group[named_idx])
    Zdeep = auxiliary_embedding(ds.W, ds.group, val_units, cfg, embedder, seed=cfg.seed)
    channels = {"scale": (Zs, "density"), "shape": (Zh, "density"), "spectral": (Zp, "density"),
                "disc": (Zp, "discriminative"), "deep": (Zdeep, "density")}
    wres = estimate_weights(channels, y, ds.group, cfg, val_units=val_units)
    log(f"  probe AUCs: loco={ {k: round(v, 3) for k, v in wres['aucs_loco'].items()} } "
        f"blend={ {k: round(v, 3) for k, v in wres['aucs_blend'].items()} } (pairs={wres['n_pairs']})")
    log(f"  weights raw   : { {k: round(v, 3) for k, v in wres['raw'].items()} }")
    log(f"  weights capped: { {k: round(v, 3) for k, v in wres['capped'].items()} }")

    # 4. fusion
    base_names = ("msp", "maha_raw", "energy", "openmax")
    have = np.ones(len(ds.W), dtype=bool)
    for k in CHANNELS:
        have &= ~np.isnan(P[k])
    for r in (r_scale, r_spec):
        for b in base_names:
            have &= ~np.isnan(r[b])
    E, finfo = tc_reject_score(P, wres["capped"], Zs, y, cfg, have)
    log(f"  {int(have.sum())}/{len(have)} windows scored by all channels; scale-weight ramp "
        f"knees lo={finfo['orth_lo']:.4f} hi={finfo['orth_hi']:.4f}")

    # 5. scores
    Ph = {k: P[k][have] for k in CHANNELS}
    scores = {HEADLINE: E}
    for k in CHANNELS:
        scores[f"p_{k}"] = 1 - Ph[k]
    for tag, r in (("", r_scale), ("(spec)", r_spec)):
        scores["MSP" + tag] = r["msp"][have]
        scores["Energy" + tag] = r["energy"][have]
        scores["Mahalanobis" + tag] = r["maha_raw"][have]
        scores["OpenMax" + tag] = r["openmax"][have]
    scores[RANK_HEADLINE] = rank_avg_score(
        Ph, scores, K, named[have], ds.group[have]
    )
    if ablations:
        w = wres["capped"]
        scores["abl:fixed-weights"] = combine_eavg([Ph[k] for k in CHANNELS], [w[k] for k in CHANNELS], cfg.kappa)
        scores["abl:uncapped-fixed"] = combine_eavg([Ph[k] for k in CHANNELS],
                                                    [wres["raw"][k] for k in CHANNELS], cfg.kappa)
        for drop in ("deep", "disc"):
            keep = [k for k in CHANNELS if k != drop]
            wk = _subset_weights(wres, keep, cfg)
            sub = {k: P[k] for k in keep}
            Ek, _ = tc_reject_score(sub, wk, Zs, y, cfg, have)
            scores[f"abl:no-{drop}"] = Ek
        c3 = [Ph["scale"], Ph["shape"], Ph["spectral"]]
        scores["abl:Bonferroni(scale,shape,spectral)"] = 1 - combine_bonferroni_min(c3)
        scores["abl:Cauchy(scale,shape,spectral)"] = 1 - combine_cauchy(c3)
        scores["abl:Bonferroni(all5)"] = 1 - combine_bonferroni_min([Ph[k] for k in CHANNELS])
        scores["abl:Cauchy(all5)"] = 1 - combine_cauchy([Ph[k] for k in CHANNELS])

    # calibration diagnostics on genuinely named windows
    named_h = (ds.truth == "named")[have]
    calib = {}
    log("  calibration on named windows (p-values should be uniform):")
    for k in CHANNELS:
        nu, cv = null_uniformity(Ph[k], named_h), coverage_at_alpha(Ph[k], named_h)
        calib[k] = dict(uniformity=nu, coverage=cv)
        log(format_calibration(k, nu, cv))

    # kappa sensitivity of the headline score
    kappa_auc = {}
    pos_all = ~named_h
    if pos_all.any() and named_h.any():
        for kp in (0.1, 0.2, 0.4, 0.6, 0.8):
            Ek, _ = tc_reject_score(P, wres["capped"], Zs, y, replace(cfg, kappa=kp), have)
            kappa_auc[str(kp)] = float(roc_auc_score(pos_all.astype(int), Ek))
        log(f"  kappa sensitivity (pooled novel vs named AUROC): {kappa_auc}")

    # per-target AUROC
    tg = [(n, m[have]) for n, m in ds.targets]
    Zcat_h = np.concatenate([Zs[have], Zh[have], Zp[have]], axis=1)
    table = auroc_table(tg, named_h, scores, ds.group[have], Zcat_h, cfg.n_boot, HEADLINE, cfg.min_n)
    for row in table:
        a = row["auroc"]
        ci = row.get("headline_ci")
        log(f"  {row['target']:34s} n={row['n_pos']:6d}  {HEADLINE}={a[HEADLINE]:.3f}"
            + (f" [{ci[0]:.2f},{ci[1]:.2f}]" if ci else "")
            + f"  best-single={max(a[f'p_{k}'] for k in CHANNELS):.3f}"
            + (f"  oracle={row['oracle']:.3f}" if row.get("oracle") == row.get("oracle") and "oracle" in row else ""))

    return dict(scores=scores, have=have, p=P, weights=wres, fusion=finfo, calibration=calib,
                kappa_auroc=kappa_auc, table=table, features=dict(Zs=Zs, Zh=Zh, Zp=Zp),
                config=cfg.to_dict())
