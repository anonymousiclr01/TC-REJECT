"""AJILE downstream utility: does removing TC-REJECT-flagged / purified windows from the
training pool help a published intracranial decoder, and how does that change with the
amount of clean training data?

Decoder. The pipeline of "A Combined Channel Approach for Decoding Intracranial EEG Signals"
(arXiv:2412.06336), the method previously run on AJILE12:
    * per electrode: gamma band 65-120 Hz -> Hilbert -> 18 features on 2 s windows
      (``paper_features.py``);
    * one weak classifier per electrode (XGBoost, the paper's best), SMOTE on its training part;
    * electrodes combined by majority vote, chosen by greedy forward selection on a
      validation part (start from the best electrode, add the one that helps most, stop when
      nothing helps);
    * 5-fold evaluation with a 64 / 16 / 20 train / validation / test ratio (here: 5 outer
      block-disjoint folds, then the training part is split 80 / 20 into train / validation).

Task. The paper decodes Move vs Rest, for which this dataset carries no labels. Here the same
decoder is applied to the named-activity task the rest of this study uses: Talk vs
Computer/phone, with the naive labelling of compound blocks ("Talk, TV" -> Talk) in the
training pool. This is the paper's METHOD on a DIFFERENT TASK; absolute accuracies are not
comparable to the paper's.

Windows. Each 10 s TC-REJECT window is cut into five non-overlapping 2 s windows. A 2 s
window inherits its parent's naive label, its purification status and its TC-REJECT flag, so
every exclusion mask applies at the 10 s level and the 2 s windows follow. (Adjacent 10 s
parents overlap by 5 s, so some raw seconds appear in two parents' children; all evaluation is
block-disjoint, so this cannot leak between train and test.)

Conditions (evaluated on true-named windows of held-out blocks):
    all_windows              naive pool, nothing removed                          (original labels)
    tcr_filtered             naive pool minus windows the score flags as reject (see threshold_mode below)
    random_filtered          naive pool minus a random draw count-matched to TC-REJECT's flags
    contam_purified          naive pool minus the compound windows purification rejected
    random_matched_compound  remove as many RANDOM compound windows as purification removes
    drop_all_compound        remove the whole compound pool (optional; not run by default)
contam_purified is the compound-block cleaning of the AJILE adapter (scale-channel conformal
test against the named class), not the fused TC-REJECT score; it is label-aware and its
circularity caveat is documented in ``datasets/ajile.py``.

Threshold mode (``--score_name`` picks WHICH saved score is read; ``--threshold_mode`` picks HOW
that score becomes a reject/keep decision for ``tcr_filtered`` -- the two are independent):
    native (default)   score >= 1/alpha for the e-value score.
    calibrated          a leave-one-group-out split-conformal threshold, computed per window from
                        the other recording blocks' true-named scores using ``loo_thresholds``
                        below; reject when the score is above that threshold.
Use ``calibrated`` for rank-average scores because they are not on the e-value scale.

Input. A finished ``run_tcreject.py ajile`` run directory (scores) plus the extracted
recordings (raw sampling rate, needed for the gamma band).

Cohort. ``--cohort dev`` targets ``tcreject.datasets.ajile.SUBJECT_SESSIONS``;
``--cohort fresh`` targets the prespecified sessions in
``tcreject.datasets.ajile.FRESH_SUBJECT_SESSIONS``. The flag selects the
subject/session list but does not change ``--results_root`` or ``--out_dir``.
"""
import csv
import json
import time
import traceback
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors

from ..datasets.ajile import SUBJECT_SESSIONS as SUBJECTS, FRESH_SUBJECT_SESSIONS, build_contam_map
from . import paper_features as PF

COHORTS = ("dev", "fresh")
DEFAULT_COHORT = "dev"
SUBJECTS_BY_COHORT = {"dev": SUBJECTS, "fresh": FRESH_SUBJECT_SESSIONS}

BASE_CONDS = ["all_windows", "tcr_filtered", "random_filtered", "contam_purified"]
ALL_CONDS = BASE_CONDS + ["random_matched_compound", "drop_all_compound"]
DEFAULT_CONDS = BASE_CONDS + ["random_matched_compound"]
SCORE_NAME = "TC-REJECT"
THRESHOLD_MODES = ("native", "calibrated")
DEFAULT_THRESHOLD_MODE = "native"
VAL_FRAC = 0.2          # of the training part, by block (paper: 64/16/20 overall)
SMOTE_K = 5


def loo_thresholds(s, named, groups, alphas):
    """Leave-one-group-out split-conformal threshold, per window: the k-th smallest score among
    the genuinely NAMED windows of every OTHER group (k = ceil((1 - alpha) * (n + 1))); +inf
    (never reject) if k > n. Returns an array shaped (len(alphas), len(s)).

    This implementation is local so the downstream module has no dependency on analysis scripts
    outside the package."""
    ug, inv = np.unique(groups, return_inverse=True)
    s_named, g_named = s[named], inv[named]
    thr = np.full((len(alphas), len(ug)), np.inf)
    for gi in range(len(ug)):
        others = np.sort(s_named[g_named != gi])
        n = len(others)
        for ai, a in enumerate(alphas):
            k = int(np.ceil((1.0 - a) * (n + 1)))
            if 1 <= k <= n:
                thr[ai, gi] = others[k - 1]
    return thr[:, inv]


# ------------------------------------------------------------------ run discovery / pool
def find_run(out_root, subject, session):
    """Newest run directory of this subject/session holding both scores and features."""
    tag = f"ajile_{subject.replace('-', '')}_{session.replace('-', '')}"
    base = Path(out_root) / tag
    if not base.is_dir():
        return None
    for d in sorted((p for p in base.iterdir() if p.is_dir()), reverse=True):
        s, f = d / f"scores_{tag}.npz", d / f"features_{tag}.npz"
        if s.exists() and f.exists():
            try:
                if "raw_kind" in np.load(s, allow_pickle=True).files:
                    return d
            except Exception:
                pass
    return None


def precheck_support(scores_npz, n_splits):
    d = np.load(scores_npz, allow_pickle=True)
    rk, rl = np.asarray(d["raw_kind"]), np.asarray(d["raw_label"])
    bid = np.asarray(d["block_id"]).astype(np.int64)
    m = rk == "named"
    classes = sorted(set(rl[m].tolist()))
    per_class = {c: dict(windows=int((m & (rl == c)).sum()),
                         blocks=int(len(np.unique(bid[m & (rl == c)])))) for c in classes}
    min_blocks = min([v["blocks"] for v in per_class.values()], default=0)
    return dict(classes=classes, per_class=per_class, n_named=int(m.sum()),
                skip=bool(len(classes) < 2 or min_blocks < n_splits))


def build_training_pool(run_dir, subject, session, score_name=SCORE_NAME, alpha=0.1, seed=0,
                        threshold_mode=DEFAULT_THRESHOLD_MODE):
    """Naive-contaminated training pool and the exclusion mask of every condition."""
    if threshold_mode not in THRESHOLD_MODES:
        raise SystemExit(f"threshold_mode={threshold_mode!r}; choose from {THRESHOLD_MODES}")
    rng = np.random.default_rng(seed)
    tag = f"ajile_{subject.replace('-', '')}_{session.replace('-', '')}"
    sc = np.load(Path(run_dir) / f"scores_{tag}.npz", allow_pickle=True)
    key = f"score__{score_name}"
    if key not in sc.files:
        raise SystemExit(f"{key} not in run; available: {[k for k in sc.files if k.startswith('score__')]}")
    score = np.asarray(sc[key], dtype=float)
    block_id, kind_h, label_h = sc["block_id"], sc["kind"], sc["label"]
    raw_kind, raw_label = sc["raw_kind"], sc["raw_label"]
    named_vocab = sorted(set(raw_label[raw_kind == "named"].tolist()))
    binary = len(named_vocab) == 2
    contam_map = build_contam_map(raw_label, raw_kind, named_vocab)
    is_orig_named = raw_kind == "named"
    is_elig = (raw_kind == "novel_semi") & np.array([l in contam_map for l in raw_label])
    naive_label = raw_label.copy()
    ei = np.where(is_elig)[0]
    naive_label[ei] = [contam_map[raw_label[i]] for i in ei]
    universe = is_orig_named | is_elig
    if threshold_mode == "native":
        thr_repr = "E >= 1/alpha"
        tcr_mask = (score >= 1.0 / alpha) & universe
    else:  # "calibrated"
        thr_repr = "LOO-conformal (calibrated on held-out true-named windows)"
        thr_per_window = loo_thresholds(score, is_orig_named, block_id, [alpha])[0]
        tcr_mask = (score > thr_per_window) & universe
    n_tcr = int(tcr_mask.sum())
    contam_mask = is_elig & (kind_h != "named")
    n_contam = int(contam_mask.sum())
    rand_mask = np.zeros(len(universe), dtype=bool)
    if n_tcr > 0:
        pool_idx = np.where(universe)[0]
        rand_mask[rng.choice(pool_idx, size=min(n_tcr, len(pool_idx)), replace=False)] = True
    print(f"  pool: {int(is_orig_named.sum())} true-named + {int(is_elig.sum())} compound-eligible windows; "
          f"{score_name!r} [{threshold_mode}: {thr_repr}] flags {n_tcr}; purification removes {n_contam}")
    conditions = {"all_windows": np.zeros(len(universe), dtype=bool), "tcr_filtered": tcr_mask,
                  "random_filtered": rand_mask, "contam_purified": contam_mask}
    return dict(window_index=np.asarray(sc["window_index"]), block_id=block_id, kind_h=kind_h, label_h=label_h,
                raw_label=raw_label, raw_kind=raw_kind, naive_label=naive_label, is_orig_named=is_orig_named,
                is_compound_eligible=is_elig, named_vocab=named_vocab, binary=binary, conditions=conditions,
                n_tcr=n_tcr, n_contam=n_contam, training_universe=universe, contam_map=contam_map)


def build_cv_folds(block_id, is_orig_named, n_splits):
    """Block-disjoint GroupKFold over TRUE-named blocks only."""
    block_id_i64 = block_id.astype(np.int64)
    named_block_ids = np.array(sorted(set(block_id_i64[is_orig_named].tolist())))
    n_eff = min(n_splits, len(named_block_ids))
    if n_eff < 2:
        raise SystemExit(f"only {len(named_block_ids)} named block(s); need >= 2")
    folds = list(GroupKFold(n_splits=n_eff).split(np.zeros((len(named_block_ids), 1)), groups=named_block_ids))
    print(f"  {len(named_block_ids)} named blocks -> {n_eff}-fold block-disjoint CV")
    return block_id_i64, named_block_ids, folds


# ------------------------------------------------------------------ the decoder
def make_clf(kind, n_estimators, seed):
    if kind == "xgboost":
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise SystemExit("xgboost is not installed (pip install xgboost). It is the paper's best classifier and "
                             "the default; --clf rf is the paper's second-best and only meant as a fallback.")
        return XGBClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=1, tree_method="hist",
                             eval_metric="logloss", verbosity=0)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=n_estimators, random_state=seed, n_jobs=1)
    raise SystemExit(f"unknown --clf {kind!r}")


def smote(X, y, seed, k=SMOTE_K):
    """Minimal SMOTE (Chawla et al. 2002): oversample the minority class to the majority count by
    interpolating between a minority sample and one of its k nearest minority neighbours."""
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n0 == n1 or min(n0, n1) < 2:
        return X, y
    minority = 1 if n1 < n0 else 0
    Xm = X[y == minority]
    need = abs(n0 - n1)
    kk = min(k, len(Xm) - 1)
    nbr = NearestNeighbors(n_neighbors=kk + 1).fit(Xm).kneighbors(Xm, return_distance=False)[:, 1:]
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(Xm), need)
    j = nbr[i, rng.integers(0, kk, need)]
    Xn = Xm[i] + rng.random((need, 1)) * (Xm[j] - Xm[i])
    return np.vstack([X, Xn]), np.concatenate([y, np.full(need, minority, dtype=y.dtype)])


def _channel_probs(Xtr, ytr, Xva, Xte, clf, n_estimators, seed):
    mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-8
    Xtr, Xva, Xte = (Xtr - mu) / sd, (Xva - mu) / sd, (Xte - mu) / sd
    Xs, ys = smote(Xtr, ytr, seed)
    m = make_clf(clf, n_estimators, seed).fit(Xs, ys)
    return m.predict_proba(Xva)[:, 1].astype(np.float32), m.predict_proba(Xte)[:, 1].astype(np.float32)


def _bal_acc_rows(pred, y):
    pos = y == 1
    return 0.5 * ((pred[:, pos] == 1).mean(axis=1) + (pred[:, ~pos] == 0).mean(axis=1))


def _vote(votes, psum, m):
    """Majority vote of m members; ties broken by the members' mean probability."""
    return np.where(2 * votes > m, 1, np.where(2 * votes < m, 0, (psum / m >= 0.5).astype(int)))


def greedy_select(Pva, yva):
    """Greedy forward channel selection for the majority vote, on validation balanced accuracy."""
    n_ch = len(Pva)
    pred = (Pva >= 0.5).astype(np.int32)
    single = _bal_acc_rows(pred, yva)
    first = int(np.argmax(single))
    sel, cur = [first], float(single[first])
    votes, psum = pred[first].copy(), Pva[first].astype(np.float64)
    used = np.zeros(n_ch, dtype=bool)
    used[first] = True
    while used.sum() < n_ch:
        cand = np.where(~used)[0]
        m = len(sel) + 1
        cp = _vote(votes[None, :] + pred[cand], psum[None, :] + Pva[cand], m)
        acc = _bal_acc_rows(cp, yva)
        b = int(np.argmax(acc))
        if acc[b] <= cur + 1e-12:
            break
        c = int(cand[b])
        sel.append(c); used[c] = True; cur = float(acc[b])
        votes += pred[c]; psum += Pva[c]
    return sel, cur, first


def split_train_val(idx, y, blk, parent, prio_block, prio_parent):
    """Split the training children ``idx`` into train / validation parts by block (validation
    = the VAL_FRAC of each class's blocks with the lowest fold-level priority). If some class
    has fewer than two blocks in the pool, split by parent window instead (recorded)."""
    yi, bi, pi = y[idx], blk[idx], parent[idx]
    for units, prio, mode in ((bi, prio_block, "block"), (pi, prio_parent, "parent_window")):
        val_units, ok = [], True
        for cls in (0, 1):
            u = np.unique(units[yi == cls])
            if len(u) < 2:
                ok = False
                break
            n_val = min(max(1, int(round(VAL_FRAC * len(u)))), len(u) - 1)
            val_units.append(u[np.argsort(prio[u])[:n_val]])
        if ok:
            is_val = np.isin(units, np.concatenate(val_units))
            return idx[~is_val], idx[is_val], mode
    return None, None, "unsplittable"


def fit_eval(D, tr_all, te_child, args, fold_seed, prio_block, prio_parent):
    """One decoder fit on the children in ``tr_all``; evaluate on the children ``te_child``."""
    y = D["y_naive"]
    if len(set(y[tr_all].tolist())) < 2 or len(te_child) == 0:
        return None
    tr, va, mode = split_train_val(tr_all, y, D["blk"], D["parent"], prio_block, prio_parent)
    if tr is None:
        return None
    F = D["F"]
    Ftr, Fva, Fte = F[tr], F[va], F[te_child]
    ytr, yva, yte = y[tr], y[va], D["y_true"][te_child]
    out = Parallel(n_jobs=args.n_jobs)(
        delayed(_channel_probs)(Ftr[:, c, :], ytr, Fva[:, c, :], Fte[:, c, :], args.clf, args.n_estimators,
                                fold_seed * 1009 + c) for c in range(F.shape[1]))
    Pva = np.stack([o[0] for o in out])
    Pte = np.stack([o[1] for o in out])
    sel, val_acc, first = greedy_select(Pva, yva)
    votes = (Pte[sel] >= 0.5).sum(axis=0)
    mean_p = Pte[sel].mean(axis=0)
    pred = _vote(votes, Pte[sel].sum(axis=0), len(sel))
    res = dict(bal_acc=float(balanced_accuracy_score(yte, pred)),
               macro_f1=float(f1_score(yte, pred, average="macro")), auroc=None,
               n_train=int(len(tr_all)), n_sel=int(len(sel)), val_bal_acc=float(val_acc), val_mode=mode,
               single_bal_acc=float(balanced_accuracy_score(yte, (Pte[first] >= 0.5).astype(int))))
    if len(set(yte.tolist())) == 2:
        res["auroc"] = float(roc_auc_score(yte, mean_p))
    return res


# ------------------------------------------------------------------ result bookkeeping
def summarize(r):
    if not r["fold_bal_acc"]:
        return None
    bal = np.array(r["fold_bal_acc"]); f1s = np.array(r["fold_macro_f1"])
    au = [a for a in r.get("fold_auroc", []) if a is not None]
    s = dict(n_folds_ran=int(len(bal)), bal_acc_mean=float(bal.mean()), bal_acc_std=float(bal.std()),
             macro_f1_mean=float(f1s.mean()), macro_f1_std=float(f1s.std()),
             mean_n_train=float(np.mean(r["fold_n_train"])),
             mean_n_channels_selected=float(np.mean(r["fold_n_sel"])),
             single_best_channel_bal_acc_mean=float(np.mean(r["fold_single_bal_acc"])))
    if au:
        s["auroc_mean"] = float(np.mean(au)); s["auroc_std"] = float(np.std(au))
    if r.get("fold_n_train_named"):
        s["mean_n_train_named"] = float(np.mean(r["fold_n_train_named"]))
    return s


def empty_res():
    return dict(fold_ids=[], fold_bal_acc=[], fold_macro_f1=[], fold_auroc=[], fold_n_test=[], fold_n_train=[],
                fold_n_train_named=[], fold_n_sel=[], fold_val_bal_acc=[], fold_single_bal_acc=[], fold_val_mode=[])


def add_res(r, fold_i, ev, n_test, n_train_named=None):
    r["fold_ids"].append(int(fold_i)); r["fold_bal_acc"].append(ev["bal_acc"])
    r["fold_macro_f1"].append(ev["macro_f1"]); r["fold_auroc"].append(ev["auroc"])
    r["fold_n_test"].append(int(n_test)); r["fold_n_train"].append(ev["n_train"])
    r["fold_n_sel"].append(ev["n_sel"]); r["fold_val_bal_acc"].append(ev["val_bal_acc"])
    r["fold_single_bal_acc"].append(ev["single_bal_acc"]); r["fold_val_mode"].append(ev["val_mode"])
    if n_train_named is not None:
        r["fold_n_train_named"].append(int(n_train_named))


def _mean_ev(evs):
    au = [e["auroc"] for e in evs if e["auroc"] is not None]
    return dict(bal_acc=float(np.mean([e["bal_acc"] for e in evs])), macro_f1=float(np.mean([e["macro_f1"] for e in evs])),
                auroc=float(np.mean(au)) if au else None, n_train=int(np.mean([e["n_train"] for e in evs])),
                n_sel=float(np.mean([e["n_sel"] for e in evs])), val_bal_acc=float(np.mean([e["val_bal_acc"] for e in evs])),
                single_bal_acc=float(np.mean([e["single_bal_acc"] for e in evs])), val_mode=evs[0]["val_mode"])


def random_matched_masks(pool, args, fold_i, tag):
    """Masks removing as many RANDOM compound windows as purification removes; seeded per (tag, fold, draw)."""
    elig = pool["is_compound_eligible"]
    idx = np.where(elig)[0]
    k = min(int(pool["n_contam"]), len(idx))
    draws = []
    for d in range(args.n_random_matched):
        rng = np.random.default_rng(args.seed * 7919 + fold_i * 131 + tag * 17 + d)
        m = np.zeros(len(elig), dtype=bool)
        if k > 0:
            m[rng.choice(idx, size=k, replace=False)] = True
        draws.append(m)
    return draws


# ------------------------------------------------------------------ experiments
def _units_for_fold(pool, args, tn, fold_i, tag):
    """Parent-level training masks: {unit name: mask}. Conditions map onto one or more units."""
    elig = pool["is_compound_eligible"]
    base = tn | elig
    units = {}
    for cn in args.conditions:
        if cn in pool["conditions"]:
            units[cn] = base & ~pool["conditions"][cn]
        elif cn == "random_matched_compound":
            for d, m in enumerate(random_matched_masks(pool, args, fold_i, tag)):
                units[f"random_matched_compound#{d}"] = base & ~m
        elif cn == "drop_all_compound":
            units[cn] = base & ~elig
    return units


def _run_units(units, D, te_child, args, fold_i, prio_block, prio_parent, cache):
    """Evaluate each distinct training mask once (identical masks give identical results)."""
    out = {}
    for name, mask in units.items():
        key = mask.tobytes()
        if key not in cache:
            child_idx = np.where(mask[D["parent"]])[0]
            cache[key] = fit_eval(D, child_idx, te_child, args, args.seed * 100003 + fold_i, prio_block, prio_parent)
        out[name] = cache[key]
    return out


def _assemble(res, unit_out, n_test, tn_named_count, args):
    for cn in args.conditions:
        if cn == "random_matched_compound":
            evs = [unit_out[k] for k in unit_out if k.startswith("random_matched_compound#") and unit_out[k] is not None]
            if evs:
                add_res(res[cn], res["_fold"], _mean_ev(evs), n_test, tn_named_count)
                res[cn].setdefault("draws_bal_acc", []).append([e["bal_acc"] for e in evs])
        elif unit_out.get(cn) is not None:
            add_res(res[cn], res["_fold"], unit_out[cn], n_test, tn_named_count)


def run_experiments(pool, D, folds, named_block_ids, block_id_i64, args, do_richness):
    is_orig_named = pool["is_orig_named"]
    conds = list(args.conditions)
    n_blocks = int(block_id_i64.max()) + 1

    def fold_setup(fold_i, tg, sg):
        tr_blocks = set(named_block_ids[tg].tolist()); te_blocks = set(named_block_ids[sg].tolist())
        test_parent = is_orig_named & np.isin(block_id_i64, list(te_blocks))
        te_child = np.where(test_parent[D["parent"]])[0]
        tn_full = is_orig_named & np.isin(block_id_i64, list(tr_blocks))
        rng = np.random.default_rng(args.seed * 65537 + fold_i)
        return te_child, tn_full, rng.random(n_blocks), rng.random(len(is_orig_named)), int(test_parent.sum())

    def run_budget(budget, tag, label):
        res = {c: empty_res() for c in conds}
        for fold_i, (tg, sg) in enumerate(folds):
            t0 = time.time()
            te_child, tn_full, pb, pp, n_te_parent = fold_setup(fold_i, tg, sg)
            if budget is not None and int(tn_full.sum()) > budget:
                sub = np.random.default_rng(args.seed * 1_000_003 + fold_i * 97 + budget)
                tn = np.zeros_like(tn_full)
                tn[sub.choice(np.where(tn_full)[0], size=budget, replace=False)] = True
            else:
                tn = tn_full
            units = _units_for_fold(pool, args, tn, fold_i, tag)
            uo = _run_units(units, D, te_child, args, fold_i, pb, pp, cache={})
            res["_fold"] = fold_i
            _assemble(res, uo, len(te_child), None, args)
            for cn in conds:      # named windows actually used, at the 10 s level
                if res[cn]["fold_ids"] and res[cn]["fold_ids"][-1] == fold_i:
                    key = cn if cn != "random_matched_compound" else "random_matched_compound#0"
                    if key in units:
                        res[cn]["fold_n_train_named"].append(int((units[key] & is_orig_named).sum()))
            print(f"    [{label}] fold {fold_i} ({time.time() - t0:.0f} s): " + "  ".join(
                f"{c[:12]}={res[c]['fold_bal_acc'][-1]:.3f}" for c in conds
                if res[c]["fold_ids"] and res[c]["fold_ids"][-1] == fold_i), flush=True)
        res.pop("_fold", None)
        return res

    res_full = run_budget(None, 0, "full")
    cls_sum = {c: summarize(res_full[c]) for c in conds}
    rich = {}
    if do_richness:
        budgets = [None if t.strip().lower() == "full" else int(t) for t in args.budgets.split(",")]
        for budget in budgets:
            bk = "full" if budget is None else str(budget)
            if budget is None:
                rich[bk] = dict(summary=cls_sum, results_raw=res_full, reused_full=True)
                continue
            avail = [int((is_orig_named & np.isin(block_id_i64, list(set(named_block_ids[tg].tolist())))).sum())
                     for tg, _ in folds]
            if max(avail) <= budget:      # not a subsample for any fold: identical to 'full'
                rich[bk] = dict(summary=cls_sum, results_raw=res_full, reused_full=True)
                print(f"    budget {bk:<5s} >= every fold's available true-named windows (max {max(avail)}): "
                      f"identical to 'full', reused", flush=True)
                continue
            res = run_budget(budget, 1 + budget, f"budget {bk}")
            rich[bk] = dict(summary={c: summarize(res[c]) for c in conds}, results_raw=res)
            s = rich[bk]["summary"]
            if s.get("all_windows") and s.get("contam_purified"):
                print("    budget %-5s named_used=%5.0f  purified-all=%+.4f" % (
                    bk, s["all_windows"].get("mean_n_train_named", float("nan")),
                    s["contam_purified"]["bal_acc_mean"] - s["all_windows"]["bal_acc_mean"]), flush=True)
    return dict(classifier=dict(summary=cls_sum, results_raw=res_full), richness=rich)


# ------------------------------------------------------------------ per subject
def _pipeline_meta(args, n_channels):
    return dict(method="arXiv:2412.06336 combined-channel decoder (per-channel classifiers, majority vote, greedy "
                        "forward channel selection, SMOTE)", task="named-activity (Talk vs Computer/phone), naive "
                        "compound labelling in the training pool -- NOT the paper's Move vs Rest",
                gamma_band=list(PF.GAMMA_BAND), window_sec=PF.CHILD_SEC, n_child_per_parent=PF.N_CHILD,
                n_features=PF.N_FEAT, feature_names=PF.FEATURE_NAMES, feature_version=PF.FEATURE_VERSION,
                classifier=args.clf, n_estimators=args.n_estimators, val_frac=VAL_FRAC, smote_k=SMOTE_K,
                n_channels=int(n_channels), score_name=args.score_name,
                threshold_mode=getattr(args, "threshold_mode", DEFAULT_THRESHOLD_MODE))


def run_subject(subj, ses, args, out_dir: Path):
    out_json = out_dir / f"downstream_{subj}_{ses}.json"
    if out_json.exists() and not args.force:
        prev = json.loads(out_json.read_text(encoding="utf-8"))
        st = prev.get("status")
        final = st in ("SKIPPED_named_support", "NOT_APPLICABLE_no_compound_pool", "NOT_BINARY") or \
            (st == "OK" and (args.skip_richness or prev.get("richness")))
        if final:
            print(f"[{subj}/{ses}] already done (status={st}); use --force to redo", flush=True)
            return prev
        print(f"[{subj}/{ses}] previous status={st} is not final for this request; redoing", flush=True)
    rec = dict(subject=subj, session=ses, status=None, args=vars(args))
    t0 = time.time()
    run_dir = find_run(args.results_root, subj, ses)
    if run_dir is None:
        rec["status"] = "NO_QUALIFYING_RUN"
        out_json.write_text(json.dumps(rec, indent=1), encoding="utf-8")   # not final: re-checked next time
        print(f"[{subj}/{ses}] no finished run with scores + features under {args.results_root}", flush=True)
        return rec
    tag = f"ajile_{subj.replace('-', '')}_{ses.replace('-', '')}"
    rec["run_dir"] = str(run_dir)
    print("=" * 78 + f"\n[{subj}/{ses}] using {run_dir}", flush=True)
    pre = precheck_support(run_dir / f"scores_{tag}.npz", args.n_splits)
    rec.update(precheck_per_class_scored=pre["per_class"], n_true_named=pre["n_named"])
    if pre["skip"]:
        rec.update(status="SKIPPED_named_support", per_class_scored=pre["per_class"],
                   reason=(f"scored true-named classes={pre['classes']}, blocks per class="
                           f"{ {c: v['blocks'] for c, v in pre['per_class'].items()} } "
                           f"(< n_splits={args.n_splits} for at least one, or < 2 classes)"),
                   seconds=round(time.time() - t0, 1))
        out_json.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
        print(f"[{subj}/{ses}] status={rec['status']}: {rec['reason']}", flush=True)
        return rec
    try:
        pool = build_training_pool(run_dir, subj, ses, args.score_name, args.alpha, seed=args.seed,
                                   threshold_mode=getattr(args, "threshold_mode", DEFAULT_THRESHOLD_MODE))
    except SystemExit as e:
        rec.update(status="POOL_BUILD_FAILED", error=str(e))
        out_json.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(f"[{subj}/{ses}] pool build stopped: {e}", flush=True)
        return rec
    except Exception:
        rec.update(status="POOL_BUILD_ERROR", error=traceback.format_exc())
        out_json.write_text(json.dumps(rec, indent=1), encoding="utf-8")
        print(rec["error"], flush=True)
        return rec
    is_named, elig = pool["is_orig_named"], pool["is_compound_eligible"]
    lab, blk = pool["label_h"], pool["block_id"].astype(np.int64)
    classes = sorted(set(lab[is_named].tolist()))
    per_class = {c: dict(windows=int((is_named & (lab == c)).sum()),
                         blocks=int(len(np.unique(blk[is_named & (lab == c)])))) for c in classes}
    n_named, n_elig, n_contam = int(is_named.sum()), int(elig.sum()), int(pool["n_contam"])
    rec.update(named_vocab_rebuilt=pool["named_vocab"], scored_named_classes=classes, per_class_scored=per_class,
               n_true_named=n_named, n_compound_eligible=n_elig, n_contam_purified=n_contam,
               n_tcr_excluded=int(pool["n_tcr"]), contam_map=pool["contam_map"],
               touched_of_compound=(n_contam / n_elig) if n_elig else None,
               touched_of_pool=(n_contam / (n_named + n_elig)) if (n_named + n_elig) else None)
    min_blocks = min([v["blocks"] for v in per_class.values()], default=0)
    if len(classes) < 2 or min_blocks < args.n_splits:
        rec["status"] = "SKIPPED_named_support"
        rec["reason"] = f"scored true-named classes={classes}, blocks per class={ {c: v['blocks'] for c, v in per_class.items()} }"
    elif not pool["binary"]:
        rec["status"] = "NOT_BINARY"
        rec["reason"] = f"named vocabulary {pool['named_vocab']} is not two classes"
    elif n_elig == 0:
        rec["status"] = "NOT_APPLICABLE_no_compound_pool"
        rec["reason"] = "no compound-eligible windows: contam_purified is identical to all_windows by construction"
    else:
        try:
            rows = np.where(pool["training_universe"])[0]
            feats = PF.get_or_compute(Path(args.out_dir) / "_feature_cache", subj, ses, args.data_root,
                                      pool["window_index"], pool["block_id"], pool["raw_label"], pool["raw_kind"],
                                      rows, recompute=args.recompute_features, n_jobs=args.n_jobs)
        except SystemExit as e:
            rec.update(status="FEATURES_FAILED", error=str(e))
            out_json.write_text(json.dumps(rec, indent=1), encoding="utf-8")
            print(f"[{subj}/{ses}] {e}", flush=True)
            return rec
        if feats is None:
            rec.update(status="NO_RAW_DATA", error=f"recording not found under {args.data_root}")
            out_json.write_text(json.dumps(rec, indent=1), encoding="utf-8")
            print(f"[{subj}/{ses}] recording not found under {args.data_root}", flush=True)
            return rec
        F, parent, meta = feats
        pos = pool["named_vocab"][1]
        D = dict(F=F, parent=parent, blk=blk[parent], y_naive=(pool["naive_label"][parent] == pos).astype(np.int64),
                 y_true=(lab[parent] == pos).astype(np.int64))
        rec["pipeline"] = _pipeline_meta(args, F.shape[1])
        rec["positive_class"] = pos
        print(f"  {F.shape[0]} nested 2 s windows x {F.shape[1]} channels x {F.shape[2]} features; positive class = {pos}",
              flush=True)
        block_id_i64, named_block_ids, folds = build_cv_folds(pool["block_id"], is_named, args.n_splits)
        rec.update(run_experiments(pool, D, folds, named_block_ids, block_id_i64, args, not args.skip_richness))
        rec["status"] = "OK"
    rec["seconds"] = round(time.time() - t0, 1)
    out_json.write_text(json.dumps(rec, indent=1, default=str), encoding="utf-8")
    print(f"[{subj}/{ses}] status={rec['status']}  ({rec['seconds']} s)  wrote {out_json.name}", flush=True)
    return rec


# ------------------------------------------------------------------ aggregate
def fold_wins(r_a, r_b):
    """folds where a > b / == / <, aligned by fold id."""
    da = dict(zip(r_a["fold_ids"], r_a["fold_bal_acc"])); db = dict(zip(r_b["fold_ids"], r_b["fold_bal_acc"]))
    ids = sorted(set(da) & set(db))
    w = sum(da[i] > db[i] + 1e-12 for i in ids); l = sum(da[i] < db[i] - 1e-12 for i in ids)
    return w, len(ids) - w - l, l, len(ids)


def _bal(S, c):
    return S[c]["bal_acc_mean"] if S.get(c) else float("nan")


CONTRASTS = [("tcr_filtered", "all_windows"), ("contam_purified", "all_windows"),
             ("contam_purified", "random_matched_compound")]


def aggregate(out_dir: Path, cohort_subjects=SUBJECTS):
    """Aggregate per-session results for the selected cohort."""
    recs = []
    for subj, ses in cohort_subjects:
        p = out_dir / f"downstream_{subj}_{ses}.json"
        if p.exists():
            recs.append(json.loads(p.read_text(encoding="utf-8")))
    if not recs:
        print("no per-subject JSONs found in", out_dir); return
    print("\n" + "=" * 100 + "\nSTATUS AND SUPPORT (touched fractions: of compound pool | of whole training pool)")
    print(f"{'subj':7}{'status':34}{'n_named':>8}{'n_comp':>7}{'n_purif':>8}{'n_tcr':>7}{'tch/comp':>9}{'tch/pool':>9}  named classes: blocks")
    for r in recs:
        pc = r.get("per_class_scored", {})
        tc, tp = r.get("touched_of_compound"), r.get("touched_of_pool")
        dash = lambda v, w: f"{'-':>{w}}" if v is None else f"{v:{w}d}"
        print(f"{r['subject']:7}{str(r['status']):34}{r.get('n_true_named', 0):8d}"
              f"{dash(r.get('n_compound_eligible'), 7)}{dash(r.get('n_contam_purified'), 8)}{dash(r.get('n_tcr_excluded'), 7)}"
              f"{f'{chr(45):>9}' if tc is None else format(tc, '9.3f')}{f'{chr(45):>9}' if tp is None else format(tp, '9.3f')}"
              f"  {pc and {c: v['blocks'] for c, v in pc.items()}}")
    ok = [r for r in recs if r["status"] == "OK"]
    if not ok:
        return
    conds = [c for c in ALL_CONDS if all(c in r["classifier"]["summary"] for r in ok)]
    contrasts = [(a, b) for a, b in CONTRASTS if a in conds and b in conds]
    print("\n" + "=" * 100 + "\nCLASSIFIER: paper decoder, balanced accuracy on 2 s windows of held-out true-named blocks (mean over folds)")
    print(f"{'subj':7}" + "".join(f"{c[:13]:>15}" for c in conds) + "".join(f"{(a[:4] + '-' + b[:4]):>11}" for a, b in contrasts)
          + "   folds a>b (of n) per contrast;  mean #channels voting | single-best-channel bal acc (all_windows)")
    rows = []
    for r in ok:
        S = r["classifier"]["summary"]; R = r["classifier"]["results_raw"]
        b = {c: _bal(S, c) for c in conds}
        d = {(a, bb): b[a] - b[bb] for a, bb in contrasts}
        w = {(a, bb): fold_wins(R[a], R[bb]) for a, bb in contrasts}
        extra = S["all_windows"]
        print(f"{r['subject']:7}" + "".join(f"{b[c]:15.4f}" for c in conds) + "".join(f"{d[k]:+11.4f}" for k in d) + "   "
              + " | ".join(f"{w[k][0]}/{w[k][3]}" for k in w)
              + f";  {extra['mean_n_channels_selected']:.1f} | {extra['single_best_channel_bal_acc_mean']:.4f}")
        row = dict(subject=r["subject"], session=r["session"], n_true_named=r["n_true_named"],
                   n_compound_eligible=r["n_compound_eligible"], n_contam_purified=r["n_contam_purified"],
                   n_tcr_excluded=r["n_tcr_excluded"], touched_of_compound=r["touched_of_compound"],
                   touched_of_pool=r["touched_of_pool"], **{f"bal_{c}": b[c] for c in conds})
        for (a, bb) in contrasts:
            row[f"d_{a}_minus_{bb}"] = d[(a, bb)]; row[f"folds_{a}_gt_{bb}"] = w[(a, bb)][0]; row["n_folds"] = w[(a, bb)][3]
        rows.append(row)
    for a, bb in contrasts:
        d = np.array([x[f"d_{a}_minus_{bb}"] for x in rows])
        print(f"\nAcross {len(d)} informative subjects: {a} - {bb}: mean {d.mean():+.4f} "
              f"(SD {d.std(ddof=1) if len(d) > 1 else float('nan'):.4f}), positive in {(d > 0).sum()}/{len(d)}, "
              f"negative in {(d < 0).sum()}/{len(d)}, zero in {(d == 0).sum()}/{len(d)}")
    print("(sub-01 is the subject the earlier confirmatory result came from -- not independent evidence.)")
    with open(out_dir / "downstream_all_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    print("\n" + "=" * 100 + "\nRICHNESS: margin in balanced accuracy by true-named training budget (10 s windows; each yields 5 nested 2 s windows)")
    budgets = []
    for r in ok:
        for bk in r.get("richness", {}):
            if bk not in budgets:
                budgets.append(bk)
    rrows = []
    for a, bb in contrasts:
        print(f"\n  margin = bal_acc({a}) - bal_acc({bb})")
        print(f"{'subj':7}" + "".join(f"{('b=' + b):>14}" for b in budgets) + "     (named windows actually used, at each budget)")
        per_budget = {b: [] for b in budgets}
        for r in ok:
            cells, used = [], []
            for bk in budgets:
                s = r.get("richness", {}).get(bk, {}).get("summary") if r.get("richness") else None
                if s and s.get(a) and s.get(bb):
                    m = s[a]["bal_acc_mean"] - s[bb]["bal_acc_mean"]
                    nu = s["all_windows"].get("mean_n_train_named", float("nan")) if s.get("all_windows") else float("nan")
                    cells.append(f"{m:+14.4f}"); used.append(f"{nu:.0f}"); per_budget[bk].append(m)
                    rrows.append(dict(subject=r["subject"], contrast=f"{a}-{bb}", budget=bk, named_used=nu, margin=m))
                else:
                    cells.append(f"{'n/a':>14}"); used.append("-")
            print(f"{r['subject']:7}" + "".join(cells) + "     " + "/".join(used))
        print(f"{'mean':7}" + "".join(f"{np.mean(per_budget[b]):+14.4f}" if per_budget[b] else f"{'n/a':>14}" for b in budgets))
        print(f"{'#pos':7}" + "".join(f"{sum(x > 0 for x in per_budget[b]):>14d}" for b in budgets))
    if rrows:
        with open(out_dir / "downstream_all_richness.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rrows[0].keys())); w.writeheader(); w.writerows(rrows)
    print("\nNOTE: budgets larger than a subject's available true-named windows are capped to 'all of them' "
          "(see the last column) -- compare margins only where 'named used' actually differs.")
    print("wrote downstream_all_summary.csv and downstream_all_richness.csv in", out_dir)


def run_all(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.conditions = [c.strip() for c in args.conditions.split(",")] if isinstance(args.conditions, str) else args.conditions
    bad = [c for c in args.conditions if c not in ALL_CONDS]
    if bad:
        raise SystemExit(f"unknown condition(s) {bad}; choose from {ALL_CONDS}")
    if "all_windows" not in args.conditions:
        raise SystemExit("all_windows (the original-label baseline) must be among the conditions")
    cohort = getattr(args, "cohort", DEFAULT_COHORT) or DEFAULT_COHORT
    if cohort not in COHORTS:
        raise SystemExit(f"--cohort {cohort!r}; choose from {COHORTS}")
    cohort_subjects = SUBJECTS_BY_COHORT[cohort]
    only = args.only_subjects.split(",") if args.only_subjects else None
    targets = [s for s in cohort_subjects if not only or s[0] in only]
    if args.list:
        for s, e in targets:
            print(s, e, "->", find_run(args.results_root, s, e))
        return
    if not args.aggregate_only:
        make_clf(args.clf, 2, 0)          # fail fast if the classifier library is missing
        for s, e in targets:
            run_subject(s, e, args, out_dir)
    aggregate(out_dir, cohort_subjects)

