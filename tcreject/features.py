"""Window features shared by every dataset.

Input windows are arrays of shape (N, T, C): N windows, T samples, C channels.

* scale:    log-compressed covariance and lagged autocovariance (magnitude of activity)
* shape:    the same statistics after variance normalisation (correlation structure)
* spectral: Welch power binned into frequency bands, log-compressed

All functions are chunked over windows so memory stays bounded; results do not
depend on the chunk size.
"""
import numpy as np

_LAGS = (2, 5, 10)


def _chunked(fn, w, chunk):
    if len(w) <= chunk:
        return fn(w)
    return np.concatenate([fn(w[i:i + chunk]) for i in range(0, len(w), chunk)], axis=0)


def _feat_scale(w):
    N, win, d = w.shape
    wc = w - w.mean(axis=1, keepdims=True)
    cov = np.einsum("nwd,nwe->nde", wc, wc) / win
    iu = np.triu_indices(d)
    feats = [cov[:, iu[0], iu[1]]]
    for lag in _LAGS:
        if win > lag + 2:
            a, b = wc[:, :-lag, :], wc[:, lag:, :]
            feats.append((a * b).mean(axis=1))
    f = np.concatenate(feats, axis=1)
    return np.sign(f) * np.log1p(np.abs(f))


def _feat_shape(w):
    N, win, d = w.shape
    wc = w - w.mean(axis=1, keepdims=True)
    cov = np.einsum("nwd,nwe->nde", wc, wc) / win
    var = np.diagonal(cov, axis1=1, axis2=2).clip(min=1e-8)
    denom = np.sqrt(var[:, :, None] * var[:, None, :])
    corr = cov / denom
    iu = np.triu_indices(d)
    feats = [corr[:, iu[0], iu[1]]]
    for lag in _LAGS:
        if win > lag + 2:
            a, b = wc[:, :-lag, :], wc[:, lag:, :]
            feats.append((a * b).mean(axis=1) / var)
    f = np.concatenate(feats, axis=1)
    return np.sign(f) * np.log1p(np.abs(f))


def feat_scale(w, chunk=2048):
    return _chunked(_feat_scale, np.asarray(w), chunk)


def feat_shape(w, chunk=2048):
    return _chunked(_feat_shape, np.asarray(w), chunk)


def default_bands(fs, seg):
    """Band edges in Hz: 0-3, 3-6, 6-10, 10-15, 15-Nyquist (clipped to Nyquist)."""
    freqs = np.fft.rfftfreq(seg, d=1.0 / fs)
    nyq = freqs[-1]
    edges = [e for e in (0.0, 3.0, 6.0, 10.0, 15.0) if e < nyq] + [nyq]
    return list(zip(edges[:-1], edges[1:]))


def _feat_spectral(w, welch_seg, welch_hop, fs, bands):
    N, win, d = w.shape
    seg = min(welch_seg, win)
    hop = max(1, min(welch_hop, seg))
    nseg = max(1, (win - seg) // hop + 1)
    idx2 = np.arange(seg)[None, :] + np.arange(nseg)[:, None] * hop
    sub = w[:, idx2, :]
    sub = sub - sub.mean(axis=2, keepdims=True)
    hann = np.hanning(seg)
    spec = np.fft.rfft(sub * hann[None, None, :, None], axis=2)
    power = (spec.real ** 2 + spec.imag ** 2).mean(axis=1)
    freqs = np.fft.rfftfreq(seg, d=1.0 / fs)
    if bands is None:
        bands = default_bands(fs, seg)
    band_feats = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs <= hi)
        if not mask.any():
            j = int(np.argmin(np.abs(freqs - 0.5 * (lo + hi))))
            bp = power[:, j, :]
        else:
            bp = power[:, mask, :].mean(axis=1)
        band_feats.append(np.log1p(bp))
    f = np.stack(band_feats, axis=1)
    return f.transpose(0, 2, 1).reshape(N, -1)


def feat_spectral(w, welch_seg=200, welch_hop=100, fs=100.0, bands=None, chunk=256):
    w = np.asarray(w)
    return _chunked(lambda x: _feat_spectral(x, welch_seg, welch_hop, fs, bands), w, chunk)


def zscore(X, ref_mask=None):
    """Column-wise z-score using the rows in ref_mask (all rows if None)."""
    ref = X if ref_mask is None else X[ref_mask]
    return (X - ref.mean(0)) / (ref.std(0) + 1e-6)


def compute_feature_spaces(W, fs, cfg, named_mask):
    """Returns dict name -> z-scored feature matrix for scale, shape, spectral."""
    ref = named_mask if cfg.standardize_on == "named" else None
    Xs, Xh = feat_scale(W), feat_shape(W)
    Xp = feat_spectral(W, welch_seg=cfg.welch_seg, welch_hop=cfg.welch_hop, fs=fs)
    return {"scale": zscore(Xs, ref), "shape": zscore(Xh, ref), "spectral": zscore(Xp, ref)}
