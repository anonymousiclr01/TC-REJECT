"""TC-REJECT(rank-avg): the paper's second fusion route.

This module implements the single frozen configuration reported in the paper. Its parameters
were selected on development data and fixed before evaluation on 12 previously unused AJILE12
sessions:

  * each of the four density channels (scale, shape, spectral, deep/"learned") and the
    discriminative channel is first *oriented* so larger = more novel, then mapped to a
    leave-one-group-out reference rank (``ref_cdf`` below): the fraction of labelled ("named")
    windows from *other* recording groups whose oriented score is <= this window's score;
  * with exactly two named classes, the scale, spectral and discriminative channels use their
    raw statistic (Mahalanobis, spectral Mahalanobis, max-softmax rejection) instead of the
    conformal p-value; with three or more named classes they use ``1 - p``. Shape and deep
    (the "learned" channel) always use ``1 - p``. This is the ``ksw`` switch below.
  * ``D(x)`` is the mean of the four density-channel reference ranks;
  * the final score is ``S_rank(x) = 0.70 * D(x) + 0.30 * R_disc(x)`` — a fixed 0.30 weight on
    the discriminative channel, prespecified from the development-data search and never refit
    per dataset or per run.

Operationally for the rank-average route, a window is flagged when its score exceeds the
``(1 - alpha)`` leave-one-group-out quantile of the labelled reference scores of every *other*
group (``loo_threshold`` below); this is not an e-value, so it does not use the ``E >= 1/alpha``
rule of ``tcreject/fusion.py``.

The implementation is intentionally restricted to that reported design; it does not expose a
search over alternative weights or transforms.
"""
from typing import Dict

import numpy as np

DENSITY_CHANNELS = ("scale", "shape", "spectral", "deep")
RAW_OF = {"scale": "Mahalanobis", "spectral": "Mahalanobis(spec)", "disc": "MSP(spec)"}
W_DISC = 0.30


def orientation_mode(K: int) -> str:
    """'raw' (use the channel's raw statistic) when there are exactly two named classes,
    else 'p' (use 1 - conformal p-value). See module docstring."""
    return "raw" if K == 2 else "p"


def channel_score(P: Dict[str, np.ndarray], scores: Dict[str, np.ndarray], channel: str, mode: str) -> np.ndarray:
    """Oriented per-window score for one channel (larger = more novel), before rank transform.

    P: channel -> (N,) conformal p-value, already restricted to the scored windows
      (``res["p"][channel][res["have"]]`` from ``tcreject.pipeline.run`` — the same subsetting
      ``tcreject.io.save_run`` applies before writing ``p__<channel>`` to ``scores_*.npz``).
    scores: the 'scores' dict of the same pipeline run (already ``have``-length), which carries
      the raw baseline statistics ("Mahalanobis", "Mahalanobis(spec)", "MSP(spec)") alongside the
      headline score.
    """
    if mode == "raw" and channel in RAW_OF:
        return np.asarray(scores[RAW_OF[channel]], dtype=float)
    return 1.0 - np.asarray(P[channel], dtype=float)


def ref_cdf(s: np.ndarray, is_ref: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Leave-one-group-out reference rank of every window in ``s``: the fraction of labelled
    ("named") windows from *other* groups whose score is <= this window's score. Depends only on
    the window's own score and the labelled reference, not on the composition of the scored pool
    (so it is deployable, unlike a rank computed over the whole scored pool). Values in [0, 1);
    a window far beyond the whole reference saturates near 1.
    """
    s = np.asarray(s, dtype=float)
    out = np.empty(len(s))
    ug, inv = np.unique(groups, return_inverse=True)
    order = np.argsort(s[is_ref])
    ss, gg = s[is_ref][order], inv[is_ref][order]
    for gi in range(len(ug)):
        m = inv == gi
        ref_other = ss[gg != gi]
        out[m] = np.searchsorted(ref_other, s[m], side="right") / (len(ref_other) + 1.0)
    return out


def channel_rank(P, scores, channel: str, mode: str, is_ref: np.ndarray, groups: np.ndarray) -> np.ndarray:
    s = channel_score(P, scores, channel, mode)
    return ref_cdf(s, is_ref, groups)


def rank_avg_score(P: Dict[str, np.ndarray], scores: Dict[str, np.ndarray], K: int,
                    is_ref: np.ndarray, groups: np.ndarray, w_disc: float = W_DISC) -> np.ndarray:
    """TC-REJECT(rank-avg): S_rank(x) = (1 - w_disc) * D(x) + w_disc * R_disc(x).

    P: ``res["p"]`` from ``tcreject.pipeline.run``, each channel restricted to ``res["have"]``
      (see ``channel_score`` above) — NOT the raw, full-length ``res["p"]``.
    scores: ``res["scores"]`` (already ``have``-length, as ``pipeline.run`` returns it).
    K: number of named classes (``len(ds.named_vocab)``); selects the orientation mode.
    is_ref: (N,) boolean, True for labelled ("named" kind) windows eligible as the reference set.
    groups: (N,) group id per window (the same grouping used for the folds).
    w_disc: discriminative-channel weight; 0.30 is the paper's prespecified value and should not
      be changed per dataset or run (see module docstring).
    """
    mode = orientation_mode(K)
    D = np.mean([channel_rank(P, scores, c, mode, is_ref, groups) for c in DENSITY_CHANNELS], axis=0)
    R_disc = channel_rank(P, scores, "disc", mode, is_ref, groups)
    return (1.0 - w_disc) * D + w_disc * R_disc


def loo_threshold(s: np.ndarray, named: np.ndarray, groups: np.ndarray, alpha: float) -> np.ndarray:
    """Per-window operational threshold at level ``alpha``: the k-th smallest score among the
    labelled ("named") windows of every *other* group, k = ceil((1 - alpha) * (n + 1));
    +inf (never reject) if k > n. Returns a (N,) array, one threshold per window (shared by all
    windows of the same group). Used for TC-REJECT(rank-avg) since, unlike the e-value route,
    it has no fixed universal threshold.
    """
    s = np.asarray(s, dtype=float)
    ug, inv = np.unique(groups, return_inverse=True)
    s_named, g_named = s[named], inv[named]
    thr = np.full(len(ug), np.inf)
    for gi in range(len(ug)):
        others = np.sort(s_named[g_named != gi])
        n = len(others)
        k = int(np.ceil((1.0 - alpha) * (n + 1)))
        if 1 <= k <= n:
            thr[gi] = others[k - 1]
    return thr[inv]
