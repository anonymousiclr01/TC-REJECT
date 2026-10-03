"""Per-channel feature extraction of the published AJILE decoder
"A Combined Channel Approach for Decoding Intracranial EEG Signals" (arXiv:2412.06336).

What the paper states (and what is used here as stated)
    * gamma band 65-120 Hz extracted first, then its Hilbert transform;
    * 2 s windows;
    * 18 features per channel: 12 "time-domain" and 6 "frequency-domain", named in its
      Tables I and II.

What the paper does NOT state, and therefore is a choice made in this file
    * the paper gives names and one-line glosses only -- no formulas, no lags, no
      entropy order, no bin counts, no band edges, no spike threshold, no filter type
      and no statement whether features are taken from the Hilbert envelope or from the
      analytic signal. Every such choice is listed in the table below and is a
      standard textbook definition, not a copy of a reference implementation (the
      referenced publication did not provide enough detail to recover those choices).
    * the choices are confined to this one file so that they can be replaced by the
      an alternative reference implementation without touching anything else: keep the signature
      ``window_features(X, fs) -> (n, 18)`` and bump ``FEATURE_VERSION``.

Definitions used (X = amplitude envelope |analytic signal| of the 65-120 Hz band-passed
signal, one 2 s window per row)
    time-domain
      mean          mean of X after a 20 ms moving average
      rms           sqrt(mean(X^2))
      max_peak      max(X)
      variance      var(X)
      skewness      Fisher-Pearson skewness
      kurtosis      excess kurtosis
      autocorr      normalised autocorrelation at a lag of 50 ms
      nonlinear_energy   mean Teager-Kaiser energy  x[n]^2 - x[n-1] x[n+1]
      spikes        number of local maxima above mean + 2 SD of the window
      higuchi_fd    Higuchi fractal dimension, kmax = 10
      shannon_entropy    Shannon entropy (bits) of a 32-bin amplitude histogram
      renyi_entropy      Renyi entropy of order 2 (bits) of the same histogram
    frequency-domain (the paper's own grouping, which also lists coastline and Hjorth here)
      coastline     sum |x[n+1] - x[n]|
      band_power    mean of the Welch PSD over all frequencies (0.5 s segments, 50 % overlap,
                    mean-detrended); the paper's gloss is "mean of power spectral density"
      spectral_edge_90   frequency below which 90 % of the Welch power lies
      hjorth_mobility    sqrt(var(dx)/var(x))
      hjorth_complexity  mobility(dx)/mobility(x)
      spectral_entropy   Shannon entropy (bits) of the normalised Welch PSD
"""
import csv
import time
from math import gcd
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from scipy.fft import next_fast_len
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, hilbert, sosfiltfilt, welch
from scipy.stats import kurtosis, skew

from ..datasets.ajile import HOP, KEEP_KINDS, TARGET_FS, WIN, _paths

FEATURE_VERSION = 1
GAMMA_BAND = (65.0, 120.0)
CHILD_SEC = 2.0
PARENT_SEC = WIN / TARGET_FS                         # 10 s TC-REJECT windows
N_CHILD = int(round(PARENT_SEC / CHILD_SEC))         # 5 non-overlapping 2 s windows per parent
FEATURE_NAMES = ["mean", "rms", "max_peak", "variance", "skewness", "kurtosis", "autocorr",
                 "nonlinear_energy", "spikes", "higuchi_fd", "shannon_entropy", "renyi_entropy",
                 "coastline", "band_power", "spectral_edge_90", "hjorth_mobility",
                 "hjorth_complexity", "spectral_entropy"]
N_FEAT = len(FEATURE_NAMES)                          # 12 time-domain + 6 frequency-domain
_EPS = 1e-12


# ------------------------------------------------------------------ the 18 features
def _higuchi(X, kmax=10):
    n, N = X.shape
    Lk = np.zeros((n, kmax))
    for k in range(1, kmax + 1):
        Lm = np.zeros((n, k))
        for m in range(k):
            xm = X[:, m::k]
            n_inc = xm.shape[1] - 1                          # floor((N - 1 - m) / k)
            if n_inc < 1:
                continue
            Lm[:, m] = np.abs(np.diff(xm, axis=1)).sum(axis=1) * (N - 1) / (n_inc * k) / k
        Lk[:, k - 1] = Lm.mean(axis=1)
    a = np.log(1.0 / np.arange(1, kmax + 1))
    b = np.log(Lk + _EPS)
    a0 = a - a.mean()
    return (b - b.mean(axis=1, keepdims=True)) @ a0 / (a0 @ a0)


def _hist_entropies(X, bins=32):
    n, T = X.shape
    lo, hi = X.min(axis=1, keepdims=True), X.max(axis=1, keepdims=True)
    idx = np.clip(((X - lo) / (hi - lo + _EPS) * bins).astype(np.int64), 0, bins - 1)
    counts = np.bincount((idx + np.arange(n)[:, None] * bins).ravel(), minlength=n * bins).reshape(n, bins)
    p = counts / T
    shannon = -(p * np.log2(np.where(p > 0, p, 1.0))).sum(axis=1)
    renyi2 = -np.log2((p ** 2).sum(axis=1) + _EPS)
    return shannon, renyi2


def window_features(X, fs):
    """(n, T) envelope windows -> (n, 18) features in ``FEATURE_NAMES`` order."""
    X = np.asarray(X, dtype=np.float64)
    n, T = X.shape
    sm = max(3, int(round(0.02 * fs)))
    lag = max(1, int(round(0.05 * fs)))
    Xc = X - X.mean(axis=1, keepdims=True)
    var = X.var(axis=1)
    dx = np.diff(X, axis=1)
    ddx = np.diff(dx, axis=1)

    f = np.zeros((n, N_FEAT))
    f[:, 0] = uniform_filter1d(X, size=sm, axis=1, mode="nearest").mean(axis=1)
    f[:, 1] = np.sqrt((X ** 2).mean(axis=1))
    f[:, 2] = X.max(axis=1)
    f[:, 3] = var
    with np.errstate(all="ignore"):
        f[:, 4] = skew(X, axis=1)
        f[:, 5] = kurtosis(X, axis=1)
    f[:, 6] = (Xc[:, :-lag] * Xc[:, lag:]).sum(axis=1) / ((Xc ** 2).sum(axis=1) + _EPS)
    f[:, 7] = (X[:, 1:-1] ** 2 - X[:, :-2] * X[:, 2:]).mean(axis=1)
    thr = (X.mean(axis=1) + 2.0 * X.std(axis=1))[:, None]
    f[:, 8] = ((X[:, 1:-1] > X[:, :-2]) & (X[:, 1:-1] > X[:, 2:]) & (X[:, 1:-1] > thr)).sum(axis=1)
    f[:, 9] = _higuchi(X)
    f[:, 10], f[:, 11] = _hist_entropies(X)
    f[:, 12] = np.abs(dx).sum(axis=1)
    freqs, P = welch(X, fs=fs, nperseg=min(T, int(round(0.5 * fs))), noverlap=None, axis=1)
    f[:, 13] = P.mean(axis=1)
    cum = np.cumsum(P, axis=1)
    f[:, 14] = freqs[np.minimum((cum < 0.9 * cum[:, -1:]).sum(axis=1), len(freqs) - 1)]
    var1, var2 = dx.var(axis=1), ddx.var(axis=1)
    mob = np.sqrt(var1 / (var + _EPS))
    f[:, 15] = mob
    f[:, 16] = np.sqrt(var2 / (var1 + _EPS)) / (mob + _EPS)
    pn = P / (P.sum(axis=1, keepdims=True) + _EPS)
    f[:, 17] = -(pn * np.log2(np.where(pn > 0, pn, 1.0))).sum(axis=1)
    return f


# ------------------------------------------------------------------ window bookkeeping
def enumerate_windows(blocks, n_raw, fs_raw):
    """Reproduce, without the signal, the window list ``datasets.ajile.build_windows`` makes.

    Returns arrays (block_index, k_within_block, start_sample_at_100Hz), one row per 10 s
    window in the same order, plus the block label/kind of each window."""
    up, down = int(TARGET_FS), int(fs_raw)
    g = gcd(up, down)
    T = -(-n_raw * (up // g) // (down // g))                 # length resample_poly gives
    bi_l, k_l, st_l, lab_l, kind_l = [], [], [], [], []
    for bi, (lab, s, e, kind) in enumerate(blocks):
        if kind not in KEEP_KINDS:
            continue
        i0, i1 = int(s * TARGET_FS), min(T, int(e * TARGET_FS))
        if i1 - i0 < WIN:
            continue
        n = (i1 - i0 - WIN) // HOP + 1
        if n <= 0:
            continue
        bi_l.append(np.full(n, bi)); k_l.append(np.arange(n)); st_l.append(i0 + np.arange(n) * HOP)
        lab_l.append(np.full(n, lab, dtype=object)); kind_l.append(np.full(n, kind, dtype=object))
    return (np.concatenate(bi_l), np.concatenate(k_l), np.concatenate(st_l),
            np.concatenate(lab_l), np.concatenate(kind_l))


def child_starts(parent_start_100hz, fs_raw):
    """Start samples (raw rate) of the N_CHILD non-overlapping 2 s windows nested in each
    10 s parent. Returns (n_parent, N_CHILD) int64."""
    t0 = np.asarray(parent_start_100hz, dtype=np.float64) / TARGET_FS
    t = t0[:, None] + CHILD_SEC * np.arange(N_CHILD)[None, :]
    return np.round(t * fs_raw).astype(np.int64)


def gamma_envelope(x, fs):
    if fs / 2.0 <= GAMMA_BAND[1] + 10:
        raise SystemExit(f"raw sampling rate {fs} Hz is too low for the 65-120 Hz gamma band "
                         f"(need > {2 * (GAMMA_BAND[1] + 10):.0f} Hz)")
    sos = butter(4, [b / (fs / 2.0) for b in GAMMA_BAND], btype="bandpass", output="sos")
    xf = sosfiltfilt(sos, x)
    return np.abs(hilbert(xf, N=next_fast_len(len(xf))))[:len(xf)]


def gather(env, starts, length):
    return env[starts[:, None] + np.arange(length)[None, :]]


# ------------------------------------------------------------------ per subject
def _channel_features(yc, fs, starts_flat, length, chunk):
    env = gamma_envelope(np.asarray(yc, dtype=np.float64), fs)
    out = np.zeros((len(starts_flat), N_FEAT), dtype=np.float32)
    for a in range(0, len(starts_flat), chunk):
        out[a:a + chunk] = window_features(gather(env, starts_flat[a:a + chunk], length), fs)
    return out


def load_raw(subject, session, data_root):
    npz_path, labels_path, _ = _paths(data_root, subject, session)
    if not npz_path.exists():
        return None
    z = np.load(npz_path, allow_pickle=False)
    with open(labels_path, newline="") as f:
        blocks = [(r["label"], float(r["start_sec"]), float(r["end_sec"]), r["kind"]) for r in csv.DictReader(f)]
    return z["y"], float(z["fs"]), z["channel_ids"], blocks


def extract_child_features(subject, session, data_root, window_index, block_id, raw_label, raw_kind,
                           rows, chunk=2000, verbose=True, n_jobs=1):
    """Features of the nested 2 s windows of the parent rows ``rows``.

    ``window_index`` .. ``raw_kind`` are the scored-window arrays of a TC-REJECT run; they are
    checked against the re-enumerated window list, and the function stops if any row disagrees.
    Returns F (n_rows * N_CHILD, n_channels, 18) float32 and the row index of every child."""
    raw = load_raw(subject, session, data_root)
    if raw is None:
        return None
    y, fs, channel_ids, blocks = raw
    if np.isnan(y).any():
        raise SystemExit("NaN in the extracted recording; re-run the extraction step")
    bi, kk, st, lab, kind = enumerate_windows(blocks, y.shape[1], fs)
    wi = np.asarray(window_index, dtype=np.int64)
    if wi.max() >= len(bi):
        raise SystemExit("run has more windows than the re-enumerated recording: the recording or labels changed "
                         "since the TC-REJECT run; refusing to continue")
    bad = (bi[wi] != np.asarray(block_id)) | (lab[wi] != np.asarray(raw_label)) | (kind[wi] != np.asarray(raw_kind))
    if bad.any():
        raise SystemExit(f"{int(bad.sum())} of {len(wi)} scored windows do not match the re-enumerated window list "
                         "(block / label / kind); the recording or labels changed since the TC-REJECT run")
    rows = np.asarray(rows, dtype=np.int64)
    starts = child_starts(st[wi[rows]], fs)                       # (n_rows, N_CHILD)
    length = int(round(CHILD_SEC * fs))
    if starts.max() + length > y.shape[1]:
        raise SystemExit("a nested window runs past the end of the recording")
    starts_flat = starts.ravel()
    parent_of_child = np.repeat(rows, N_CHILD)
    F = np.zeros((len(starts_flat), y.shape[0], N_FEAT), dtype=np.float32)
    t0 = time.time()
    gen = Parallel(n_jobs=n_jobs, return_as="generator")(
        delayed(_channel_features)(y[c], fs, starts_flat, length, chunk) for c in range(y.shape[0]))
    for c, fc in enumerate(gen):
        F[:, c, :] = fc
        if verbose and (c % 10 == 0 or c == y.shape[0] - 1):
            print(f"    channel {c + 1}/{y.shape[0]}  ({time.time() - t0:.0f} s)", flush=True)
    bad = ~np.isfinite(F)
    n_bad = int(bad.sum())
    if n_bad:
        print(f"    replaced {n_bad} non-finite feature values ({n_bad / F.size:.2e} of all) with 0", flush=True)
        # where they are: per feature and per channel (fraction of that feature's / channel's values)
        by_feat = bad.sum(axis=(0, 1)) / (F.shape[0] * F.shape[1])
        by_chan = bad.sum(axis=(0, 2)) / (F.shape[0] * F.shape[2])
        top_f = [f"{FEATURE_NAMES[i]}={by_feat[i]:.3f}" for i in np.argsort(-by_feat)[:6] if by_feat[i] > 0]
        top_c = [f"ch{channel_ids[c]}={by_chan[c]:.3f}" for c in np.argsort(-by_chan)[:8] if by_chan[c] > 0]
        print(f"    non-finite by feature (top): {', '.join(top_f)}", flush=True)
        print(f"    non-finite by channel (top): {', '.join(top_c)}", flush=True)
        print(f"    channels with any non-finite: {int((by_chan > 0).sum())} of {F.shape[1]}; "
              f"channels with >50% non-finite: {int((by_chan > 0.5).sum())}", flush=True)
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F, parent_of_child, dict(fs_raw=fs, n_channels=int(y.shape[0]), channel_ids=np.asarray(channel_ids),
                                    band=GAMMA_BAND, child_sec=CHILD_SEC, n_nonfinite=n_bad)


def get_or_compute(cache_dir, subject, session, data_root, window_index, block_id, raw_label, raw_kind,
                   rows, recompute=False, verbose=True, n_jobs=1):
    """Cached wrapper. The cache is valid only for the same parent rows and feature version."""
    tag = f"ajile_{subject.replace('-', '')}_{session.replace('-', '')}"
    path = Path(cache_dir) / f"paper_features_{tag}.npz"
    rows = np.asarray(rows, dtype=np.int64)
    if path.exists() and not recompute:
        z = np.load(path, allow_pickle=False)
        if (int(z["version"]) == FEATURE_VERSION and np.array_equal(z["window_index"], np.asarray(window_index)[rows])
                and np.array_equal(z["rows"], rows)):
            if verbose:
                print(f"  paper features: loaded cache {path.name}", flush=True)
            return z["F"], z["parent"], dict(n_channels=int(z["F"].shape[1]), cached=True)
        print("  paper features: cache does not match this run / version; recomputing", flush=True)
    out = extract_child_features(subject, session, data_root, window_index, block_id, raw_label, raw_kind, rows,
                                 verbose=verbose, n_jobs=n_jobs)
    if out is None:
        return None
    F, parent, meta = out
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    np.savez(path, F=F, parent=parent, rows=rows, window_index=np.asarray(window_index)[rows],
             version=FEATURE_VERSION, feature_names=np.array(FEATURE_NAMES), fs_raw=meta["fs_raw"],
             band=np.array(GAMMA_BAND), channel_ids=meta["channel_ids"])
    return F, parent, meta
