"""Synthetic benchmark: a latent linear-dynamical system observed through mixing
matrices, with two labelled ("named") regimes and three kinds of novelty.

Novelty kinds (ground truth, used only for evaluation):
  STRUCTURED    replacement by a clean, distinct regime
  UNSTRUCTURED  dynamics resampled every 0.5 s (no stable template)
  SEMI          the enclosing named dynamics with extra unstructured drive superposed

Every recording is split into blocks carrying one instructed label. Novel
segments are inserted *inside* labelled blocks (open-set label noise), and some
segments of the other named regime are inserted too (closed-set label noise), so
the method sees only noisy block labels and never the true state.
"""
import numpy as np
from scipy.linalg import expm

from ..conformal import Fold
from ..folds import group_kfold_over_labelled
from .base import Dataset

GATE_SEEDS = [11, 23, 37, 41, 59, 67, 73, 89, 97, 101]
DEV_SEEDS = [1001, 1002, 1003]
DEFAULT_OPEN = 0.25
DEFAULT_CLOSED = 0.10
WIN = 50      # 0.5 s at 100 Hz
HOP = 20      # 0.2 s


class SynthConfig:
    fs = 100.0
    n_regions = 4
    n_channels_per_region = 3
    latent_dim = 4
    n_named = 2
    struct_idx = 2
    unstruct_idx = 3
    unstruct_radius = 0.3
    unstruct_drive = 0.25
    n_novel_true = 3
    n_recordings = 8
    n_blocks = 8
    block_sec = 30.0
    semi_idx = 4
    semi_perturb = 0.2
    kappa = [1.0, 0.6, 0.25, 0.0]
    snr_db = 10.0
    drive = 0.05
    closed_contam = 0.1
    open_contam = 0.3
    contam_spread = 0.08
    n_segments = (1, 3)
    min_seg_sec = 1.0


def _stable_rotation(rng, d, radius, ang=0.35):
    A = rng.normal(size=(d, d))
    A = A - A.T
    return expm(A * ang) * radius


def _state_dynamics(rng, cfg, kappa_r, n_total):
    d = cfg.latent_dim
    shared = _stable_rotation(rng, d, 0.92)
    mats = []
    n_spread = max(2, cfg.n_named + 1)
    for k in range(n_total):
        if k == cfg.unstruct_idx:
            specific = _stable_rotation(rng, d, cfg.unstruct_radius, ang=0.05)
        elif k == cfg.semi_idx:
            specific = shared
        else:
            ang = 0.1 + 0.35 * k / max(1, n_spread - 1)
            rad = 0.99 - 0.1 * k / max(1, n_spread - 1)
            specific = _stable_rotation(rng, d, rad, ang=ang)
        mats.append((1.0 - kappa_r) * shared + kappa_r * specific)
    return np.stack(mats)


def _make_blocks(cfg, rng):
    lens, labels = ([], [])
    for b in range(cfg.n_blocks):
        lens.append(int(round(cfg.block_sec * rng.uniform(0.85, 1.15) * cfg.fs)))
        labels.append(b % cfg.n_named)
    return (np.array(lens), np.array(labels))


def _insert_segments(state, target_states, frac, cfg, rng):
    L = len(state)
    n_dev = int(round(frac * L))
    min_seg = int(cfg.min_seg_sec * cfg.fs)
    if n_dev < min_seg or L <= 2 * min_seg:
        import warnings
        warnings.warn(f'_insert_segments: frac={frac:.3f} of L={L} gives n_dev={n_dev} < min_seg={min_seg}; inserting nothing')
        return 0.0
    n_seg = int(rng.integers(cfg.n_segments[0], cfg.n_segments[1] + 1))
    n_seg = max(1, min(n_seg, n_dev // min_seg))
    seg_len = max(min_seg, n_dev // n_seg)
    used = np.zeros(L, dtype=bool)
    placed = 0
    for _ in range(n_seg):
        for _try in range(20):
            start = int(rng.integers(0, max(1, L - seg_len)))
            end = start + seg_len
            if not used[start:end].any():
                state[start:end] = int(rng.choice(target_states))
                used[start:end] = True
                placed += end - start
                break
    return placed / L


def generate_recording(rng, cfg, closed_frac, open_frac, snr_db, block_offset, region_mats, region_W):
    lens, instructed = _make_blocks(cfg, rng)
    T = int(lens.sum())
    n_total = cfg.n_named + cfg.n_novel_true
    state_true = np.zeros(T, dtype=np.int64)
    label_block = np.zeros(T, dtype=np.int64)
    block_id = np.zeros(T, dtype=np.int64)
    open_true = np.zeros(cfg.n_blocks)
    closed_true = np.zeros(cfg.n_blocks)
    novels = list(range(cfg.n_named, n_total))
    pos = 0
    for b, (Lb, lab) in enumerate(zip(lens, instructed)):
        st = np.full(Lb, lab, dtype=np.int64)
        others = [k for k in range(cfg.n_named) if k != lab]
        of = max(0.0, rng.uniform(open_frac - cfg.contam_spread, open_frac + cfg.contam_spread))
        cf = max(0.0, rng.uniform(closed_frac - cfg.contam_spread, closed_frac + cfg.contam_spread))
        placed = 0.0
        for nk in novels:
            placed += _insert_segments(st, [nk], of / len(novels), cfg, rng)
        open_true[b] = placed
        closed_true[b] = _insert_segments(st, others, cf, cfg, rng)
        state_true[pos:pos + Lb] = st
        label_block[pos:pos + Lb] = lab
        block_id[pos:pos + Lb] = block_offset + b
        pos += Lb
    enclosing = label_block.copy()
    kappa = np.asarray(cfg.kappa, dtype=float)
    y = np.zeros((cfg.n_regions, cfg.n_channels_per_region, T))
    drive = np.full(n_total, cfg.drive)
    drive[cfg.unstruct_idx] = cfg.unstruct_drive
    seg = int(0.5 * cfg.fs)
    for r in range(cfg.n_regions):
        mats = region_mats[r]
        unstruct_mats = [_stable_rotation(rng, cfg.latent_dim, rng.uniform(0.7, 0.98), ang=rng.uniform(0.05, 1.2)) for _ in range(max(1, T // seg + 1))]
        x = np.zeros((cfg.latent_dim, T))
        x[:, 0] = rng.normal(size=cfg.latent_dim)
        for t in range(1, T):
            k = state_true[t - 1]
            if k == cfg.unstruct_idx:
                A, dr, extra = (unstruct_mats[t // seg], drive[k], 0.0)
            elif k == cfg.semi_idx:
                A, dr, extra = (mats[enclosing[t - 1]], cfg.drive, cfg.semi_perturb)
            else:
                A, dr, extra = (mats[k], drive[k], 0.0)
            x[:, t] = A @ x[:, t - 1] + dr * rng.normal(size=cfg.latent_dim)
            if extra > 0:
                x[:, t] += extra * rng.normal(size=cfg.latent_dim)
        W = region_W[r]
        clean = W @ x
        sig_p = clean.var(axis=1, keepdims=True) + 1e-09
        noise_p = sig_p / 10.0 ** (snr_db / 10.0)
        y[r] = clean + np.sqrt(noise_p) * rng.normal(size=clean.shape)
    y = (y - y.mean(axis=2, keepdims=True)) / (y.std(axis=2, keepdims=True) + 1e-08)
    return dict(y=y.astype(np.float32), state_true=state_true, is_novel=(state_true >= cfg.n_named).astype(np.int64), is_struct=(state_true == cfg.struct_idx).astype(np.int64), is_unstruct=(state_true == cfg.unstruct_idx).astype(np.int64), is_semi=(state_true == cfg.semi_idx).astype(np.int64), label_block=label_block, block_id=block_id, closed_true=closed_true, open_true=open_true, instructed=instructed, block_lens=lens, kappa=kappa)


def generate_dataset(seed, closed_frac=None, open_frac=None, snr_db=None, cfg=None):
    cfg = cfg or SynthConfig()
    closed_frac = SynthConfig.closed_contam if closed_frac is None else closed_frac
    open_frac = SynthConfig.open_contam if open_frac is None else open_frac
    snr_db = SynthConfig.snr_db if snr_db is None else snr_db
    rng = np.random.default_rng(seed)
    n_total = cfg.n_named + cfg.n_novel_true
    kappa = np.asarray(cfg.kappa, dtype=float)
    region_mats = [_state_dynamics(rng, cfg, kappa[r], n_total) for r in range(cfg.n_regions)]
    region_W = [rng.normal(size=(cfg.n_channels_per_region, cfg.latent_dim)) for _ in range(cfg.n_regions)]
    recs, offset = ([], 0)
    for _ in range(cfg.n_recordings):
        recs.append(generate_recording(rng, cfg, closed_frac, open_frac, snr_db, offset, region_mats, region_W))
        offset += cfg.n_blocks
    return dict(recordings=recs, n_blocks_total=offset, kappa=kappa, closed_true=np.concatenate([r['closed_true'] for r in recs]), open_true=np.concatenate([r['open_true'] for r in recs]), meta=dict(seed=seed, closed_frac=closed_frac, open_frac=open_frac, snr_db=snr_db, fs=cfg.fs, n_regions=cfg.n_regions, n_channels=cfg.n_channels_per_region, n_named=cfg.n_named, n_novel_true=cfg.n_novel_true, struct_idx=cfg.struct_idx, unstruct_idx=cfg.unstruct_idx, semi_idx=cfg.semi_idx))


KINDS = {"struct_idx": "STRUCTURED", "semi_idx": "SEMI", "unstruct_idx": "UNSTRUCTURED"}


def _windows(ds, win, hop):
    W, ST, LB, GRP = [], [], [], []
    for ri, rec in enumerate(ds["recordings"]):
        y = rec["y"]
        R, d, T = y.shape
        y2 = y.reshape(R * d, T)
        N = max(1, (T - win) // hop + 1)
        idx = np.clip(np.arange(win)[None, :] + np.arange(N)[:, None] * hop, 0, T - 1)
        W.append(np.transpose(y2[:, idx], (1, 2, 0)).astype(np.float32))
        c = (np.arange(N) * hop + win // 2).clip(max=T - 1)
        ST.append(rec["state_true"][c])
        LB.append(rec["label_block"][c])
        GRP.append(ri * 10_000 + rec["block_id"][c])
    return np.concatenate(W), np.concatenate(ST), np.concatenate(LB), np.concatenate(GRP)


def make_dataset(seed, open_frac=DEFAULT_OPEN, closed_frac=DEFAULT_CLOSED, snr_db=None,
                 n_splits=5, win=WIN, hop=HOP):
    """One synthetic dataset. The method sees block labels only; the true state
    is used for the evaluation targets."""
    cfg = SynthConfig()
    gen = generate_dataset(seed, closed_frac, open_frac, snr_db, cfg=cfg)
    meta = gen["meta"]
    W, ST, LB, GRP = _windows(gen, win, hop)
    n_named = meta["n_named"]
    vocab = [f"class{i}" for i in range(n_named)]
    label = np.array([vocab[i] for i in LB], dtype=object)
    kind = np.full(len(W), "named", dtype=object)          # every window carries a block label
    truth = np.full(len(W), "named", dtype=object)
    for key, nm in KINDS.items():
        truth[ST == meta[key]] = nm
    targets = [(nm, truth == nm) for nm in KINDS.values() if int((truth == nm).sum()) > 0]
    folds = group_kfold_over_labelled(GRP, np.ones(len(W), dtype=bool), n_splits)
    return Dataset(name=f"synthetic_seed{seed}", W=W, fs=float(meta["fs"]), label=label, kind=kind,
                   group=GRP, named_vocab=vocab, folds=folds, truth=truth, truth_label=truth.copy(),
                   targets=targets, meta=dict(meta, win=win, hop=hop),
                   extras=dict(raw_kind=kind.copy(), raw_label=label.copy()))
