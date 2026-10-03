"""PAMAP2 adapter (wearable accelerometry).

Nine subjects, three IMUs (9 accelerometer axes at 100 Hz), 10 s windows with 50 %
overlap. In each evaluation split a few activities are held out: their windows
are unlabelled for the method (kind 'heldout') and serve as the novelty targets;
all other activities are the labelled classes. Folds are leave-one-subject-out.
"""
import glob
import os

import numpy as np

from ..folds import leave_one_group_out
from .base import Dataset

ACTIVITY_NAMES = {
    1: "lying", 2: "sitting", 3: "standing", 4: "walking", 5: "running",
    6: "cycling", 7: "nordic_walking", 12: "ascending_stairs",
    13: "descending_stairs", 16: "vacuum_cleaning", 17: "ironing",
    24: "rope_jumping",
}
ACCEL_COLS = [4, 5, 6, 21, 22, 23, 38, 39, 40]
HOLDOUT_SPLITS = {
    "A": {24, 16},   # rope_jumping, vacuum_cleaning
    "B": {7},        # nordic_walking (walking stays labelled)
    "C": {12, 13},   # ascending_stairs, descending_stairs
}
FS = 100.0
WIN = 1000
HOP = 500


def _fill_nan_nearest(x):
    x = x.copy()
    for c in range(x.shape[1]):
        col = x[:, c]
        nans = np.isnan(col)
        if not nans.any():
            continue
        good = np.flatnonzero(~nans)
        if len(good) == 0:
            col[:] = 0.0
            continue
        fill = np.clip(np.searchsorted(good, np.flatnonzero(nans)), 0, len(good) - 1)
        col[nans] = col[good[fill]]
    return x


def _load_subject(path):
    arr = np.loadtxt(path)
    return arr[:, 1].astype(int), _fill_nan_nearest(arr[:, ACCEL_COLS])


def _segment_runs(act, accel, min_len):
    change = np.flatnonzero(np.diff(act) != 0) + 1
    bounds = np.concatenate(([0], change, [len(act)]))
    runs = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        a = int(act[s])
        if a in ACTIVITY_NAMES and (e - s) >= min_len:
            runs.append((a, accel[s:e]))
    return runs


def _sliding_windows(x, win, hop):
    T, d = x.shape
    N = max(0, (T - win) // hop + 1)
    if N == 0:
        return np.zeros((0, win, d))
    idx = np.arange(win)[None, :] + np.arange(N)[:, None] * hop
    return x[idx]


def load_windows(data_root, win=WIN, hop=HOP, cache_dir=None):
    """Raw windows W (M, win, 9), activity id (M,), subject id (M,)."""
    cache = None
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        cache = os.path.join(cache_dir, f"pamap2_windows_w{win}_h{hop}.npz")
        if os.path.exists(cache):
            z = np.load(cache)
            return z["W"], z["act"], z["subj"]
    files = sorted(glob.glob(os.path.join(data_root, "Protocol", "subject*.dat")))
    if not files:
        raise FileNotFoundError(f"no subject*.dat under {os.path.join(data_root, 'Protocol')}")
    W, A, S = [], [], []
    for path in files:
        sid = int(os.path.basename(path).replace("subject", "").replace(".dat", ""))
        act, accel = _load_subject(path)
        for a, seg in _segment_runs(act, accel, min_len=win):
            w = _sliding_windows(seg, win, hop)
            if len(w) == 0:
                continue
            W.append(w)
            A.append(np.full(len(w), a, dtype=int))
            S.append(np.full(len(w), sid, dtype=int))
    W, A, S = np.concatenate(W), np.concatenate(A), np.concatenate(S)
    if cache:
        np.savez(cache, W=W, act=A, subj=S)
    return W, A, S


def make_dataset(split, data_root, win=WIN, hop=HOP, cache_dir=None, windows=None):
    """One evaluation split ('A', 'B' or 'C'). ``windows`` may pass pre-loaded (W, act, subj)."""
    W, act, subj = windows if windows is not None else load_windows(data_root, win, hop, cache_dir)
    held = HOLDOUT_SPLITS[split]
    is_named = ~np.isin(act, list(held))
    names = np.array([ACTIVITY_NAMES[a] for a in act], dtype=object)
    kind = np.where(is_named, "named", "heldout").astype(object)
    truth = kind.copy()
    truth[~is_named] = names[~is_named]
    vocab = sorted({ACTIVITY_NAMES[a] for a in set(act.tolist()) - held})
    targets = [(ACTIVITY_NAMES[a], act == a) for a in sorted(held)]
    return Dataset(name=f"pamap2_split{split}", W=W, fs=FS, label=names, kind=kind, group=subj,
                   named_vocab=vocab, folds=leave_one_group_out(subj, is_named), truth=truth,
                   truth_label=names, targets=targets,
                   meta=dict(split=split, heldout=sorted(ACTIVITY_NAMES[a] for a in held), win=win, hop=hop),
                   extras=dict(raw_kind=kind.copy(), raw_label=names.copy()))
