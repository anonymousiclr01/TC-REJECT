"""AJILE12 adapter (human intracranial ECoG, DANDI 000055).

Expects the extracted recording of one subject/session under
``<data_root>/AJILE12_<subject>/`` as two files:

    ajile_<subject>_<session>_full.npz         y (channels, samples), fs, channel_ids
    ajile_<subject>_<session>_full_labels.csv  label, start_sec, end_sec, kind

Signals are band-passed (1-40 Hz), resampled to 100 Hz, z-scored per channel and
grouped by electrode region. Windows are 10 s with 50 % overlap; windows never
cross block boundaries.

Block kinds:  named (the two labelled activities), novel_structured (single
unlabelled activity), novel_semi (compound label such as "Talk, TV"). Blocks of
other kinds (blocklist, other) are not windowed.

Compound-block purification (``refine_contam=True``): every window of a compound
block that pairs exactly one named class with other content is scored against
its own named class and reclassified as named if it fits, marked ambiguous if it
fits a different named class, and left as novel_semi otherwise.
"""
import csv
import json
from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt, resample_poly
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from ..conformal import per_class_pvalue
from ..features import feat_scale
from ..folds import group_kfold_over_labelled
from .base import Dataset

DANDISET_ID = "000055"
TARGET_FS = 100.0
BAND = (1.0, 40.0)
WIN = 1000        # 10 s at 100 Hz
HOP = 500         # 50 % overlap
MIN_REGION_CHANNELS = 6
KEEP_KINDS = {"named", "novel_structured", "novel_semi"}

SUBJECT_SESSIONS = [("sub-01", "ses-4"), ("sub-03", "ses-3"), ("sub-05", "ses-7"), ("sub-10", "ses-5"),
                    ("sub-02", "ses-6"), ("sub-04", "ses-4"), ("sub-06", "ses-3"),
                    ("sub-07", "ses-5"), ("sub-08", "ses-7"), ("sub-11", "ses-3")]

# Prespecified 12-session evaluation cohort, disjoint from SUBJECT_SESSIONS on
# the (subject, session) pair.
FRESH_SUBJECT_SESSIONS = [("sub-01", "ses-3"), ("sub-03", "ses-4"), ("sub-04", "ses-3"), ("sub-04", "ses-6"),
                          ("sub-05", "ses-4"), ("sub-07", "ses-6"), ("sub-08", "ses-6"), ("sub-10", "ses-3"),
                          ("sub-10", "ses-4"), ("sub-10", "ses-6"), ("sub-12", "ses-4"), ("sub-12", "ses-5")]


def _paths(data_root, subject, session):
    st, se = subject.replace("-", ""), session.replace("-", "")
    d = Path(data_root) / f"AJILE12_{st}"
    return (d / f"ajile_{st}_{se}_full.npz", d / f"ajile_{st}_{se}_full_labels.csv",
            d / f"ajile_{st}_{se}_group_names.json")


def _electrode_group_names(subject, session, sidecar):
    """Electrode region of every electrode in the NWB table (cached next to the data)."""
    if sidecar.exists():
        return json.loads(sidecar.read_text(encoding="utf-8"))
    import remfile, h5py, pynwb                      # noqa: E401  (only needed on first use)
    from dandi.dandiapi import DandiAPIClient
    asset_path = f"{subject}/{subject}_{session}_behavior+ecephys.nwb"
    with DandiAPIClient() as client:
        asset = client.get_dandiset(DANDISET_ID, "draft").get_asset_by_path(asset_path)
        url = asset.get_content_url(follow_redirects=1, strip_query=True)
    io = pynwb.NWBHDF5IO(file=h5py.File(remfile.File(url), "r"), load_namespaces=True)
    try:
        names = list(io.read().electrodes.to_dataframe()["group_name"])
    finally:
        io.close()
    sidecar.write_text(json.dumps(names), encoding="utf-8")
    return names


def load_and_preprocess(subject, session, data_root):
    npz_path, labels_path, sidecar = _paths(data_root, subject, session)
    z = np.load(npz_path, allow_pickle=False)
    y_raw, fs_raw = z["y"], float(z["fs"])
    channel_ids = z["channel_ids"]
    assert not np.isnan(y_raw).any(), "NaN in the extracted recording; re-run the extraction step"
    with open(labels_path, newline="") as f:
        blocks = [(r["label"], float(r["start_sec"]), float(r["end_sec"]), r["kind"])
                  for r in csv.DictReader(f)]
    sos = butter(4, [b / (fs_raw / 2) for b in BAND], btype="bandpass", output="sos")
    y_filt = sosfiltfilt(sos, y_raw, axis=1)
    up, down = int(TARGET_FS), int(fs_raw)
    g = gcd(up, down)
    y = resample_poly(y_filt, up // g, down // g, axis=1).astype(np.float32)
    y = (y - y.mean(1, keepdims=True)) / (y.std(1, keepdims=True) + 1e-08)
    all_names = _electrode_group_names(subject, session, sidecar)
    group_names = [all_names[i] for i in channel_ids]
    print(f"[ajile] {subject}/{session}: {y.shape[0]} channels, {y.shape[1] / TARGET_FS / 3600:.2f} h at {TARGET_FS} Hz")
    return y, blocks, group_names


def group_regions(y, group_names):
    names = sorted(set(group_names))
    groups, counts = [], []
    for name in names:
        idx = [i for i, gn in enumerate(group_names) if gn == name]
        if idx:
            groups.append(y[idx])
            counts.append((name, len(idx)))
    kept = [(g, c) for g, c in zip(groups, counts) if c[1] >= MIN_REGION_CHANNELS]
    dropped = [c for c in counts if c[1] < MIN_REGION_CHANNELS]
    if dropped:
        print(f"[ajile] dropping regions below {MIN_REGION_CHANNELS} channels: {dropped}")
    if not kept:
        raise SystemExit(f"all regions below {MIN_REGION_CHANNELS} channels")
    n_ch = min(g.shape[0] for g, _ in kept)
    print(f"[ajile] regions: {[c for _, c in kept]} -> trimmed to {n_ch} channels each")
    return np.stack([g[:n_ch] for g, _ in kept], axis=0), [c[0] for _, c in kept], n_ch


def _sliding_windows(x, win, hop):
    T, d = x.shape
    N = max(0, (T - win) // hop + 1)
    if N == 0:
        return np.zeros((0, win, d), dtype=x.dtype)
    idx = np.arange(win)[None, :] + np.arange(N)[:, None] * hop
    return x[idx]


def build_windows(subject, session, data_root):
    y, blocks, group_names = load_and_preprocess(subject, session, data_root)
    y_regions, region_names, n_ch = group_regions(y, group_names)
    R, C, T = y_regions.shape
    x = y_regions.reshape(R * C, T).T
    Ws, blks, labs, kinds = [], [], [], []
    n_skipped_kind = n_short = 0
    for bi, (lab, s, e, k) in enumerate(blocks):
        if k not in KEEP_KINDS:
            n_skipped_kind += 1
            continue
        i0, i1 = int(s * TARGET_FS), min(T, int(e * TARGET_FS))
        if i1 - i0 < WIN:
            n_short += 1
            continue
        w = _sliding_windows(x[i0:i1], WIN, HOP)
        if len(w) == 0:
            continue
        Ws.append(w)
        blks.append(np.full(len(w), bi, dtype=np.int64))
        labs.append(np.full(len(w), lab, dtype=object))
        kinds.append(np.full(len(w), k, dtype=object))
    if not Ws:
        raise RuntimeError("no windows built")
    W = np.concatenate(Ws)
    print(f"[ajile] {len(blocks)} blocks, {n_skipped_kind} skipped (blocklist/other), {n_short} shorter than "
          f"{WIN / TARGET_FS:.0f} s -> {len(W)} windows, {W.shape[2]} channels ({region_names})")
    return W, np.concatenate(blks), np.concatenate(labs), np.concatenate(kinds)


# ------------------------------------------------------ compound-block purification
def build_contam_map(lab, kind, named_vocab):
    """{compound label: its single named class} for compound labels pairing exactly
    one named class with other content (e.g. 'Talk, TV' -> 'Talk')."""
    named = set(named_vocab)
    cmap, excluded = {}, []
    for raw in sorted(set(lab[kind == "novel_semi"].tolist())):
        parts = [p.strip() for p in raw.split(",")]
        named_parts = [p for p in parts if p in named]
        other_parts = [p for p in parts if p not in named]
        if len(named_parts) == 1 and len(other_parts) >= 1:
            cmap[raw] = named_parts[0]
        else:
            excluded.append(raw)
    print(f"[contam] {len(cmap)} usable compound label(s): {cmap}; {len(excluded)} excluded (0 or 2 named parts)")
    return cmap


def refine_contam_windows(Xref, lab, kind, blk, named_vocab, contam_map, shrink=0.1, alpha=0.1):
    """Purify compound blocks window by window. Returns (lab', kind', report)."""
    is_named = kind == "named"
    is_elig = np.array([l in contam_map for l in lab]) & (kind == "novel_semi")
    report = {"n_contam_eligible": int(is_elig.sum())}
    if not is_elig.any():
        print("[contam] no eligible compound-block windows; nothing to refine")
        return lab.copy(), kind.copy(), report
    named_sorted = sorted(named_vocab)
    label_map = {c: i for i, c in enumerate(named_sorted)}
    K = len(named_sorted)
    y_named = np.array([label_map[l] for l in lab[is_named]])
    X_named = Xref[is_named]
    lda = LDA(solver="lsqr", shrinkage=shrink).fit(X_named, y_named)
    mus, cov = lda.means_, lda.covariance_
    D = cov.shape[0]
    P = np.linalg.inv(cov + shrink * np.trace(cov) / D * np.eye(D))

    def maha_all(X):
        d = X[:, None, :] - mus[None]
        return np.einsum("nkd,de,nke->nk", d, P, d)

    idxs = np.where(is_elig)[0]
    d_te = maha_all(Xref[idxs])
    p_nearest = per_class_pvalue(d_te, maha_all(X_named), y_named, K)
    k_star = d_te.argmin(axis=1)
    own_str = np.array([contam_map[l] for l in lab[idxs]])
    own_idx = np.array([label_map[s] for s in own_str])
    fits_own = (k_star == own_idx) & (p_nearest >= alpha)
    fits_other = (k_star != own_idx) & (p_nearest >= alpha)
    lab2, kind2 = lab.copy(), kind.copy()
    lab2[idxs[fits_own]] = own_str[fits_own]
    kind2[idxs[fits_own]] = "named"
    kind2[idxs[fits_other]] = "other"
    stays = ~fits_own & ~fits_other
    report.update(n_blocks_eligible=len(set(blk[idxs].tolist())), frac_reclassified_named=float(fits_own.mean()),
                  frac_ambiguous=float(fits_other.mean()), frac_remaining_novel_semi=float(stays.mean()), alpha=alpha)
    print(f"[contam] {len(idxs)} windows in {report['n_blocks_eligible']} compound blocks: "
          f"{fits_own.mean():.1%} reclassified named, {fits_other.mean():.1%} ambiguous (excluded), "
          f"{stays.mean():.1%} remain novel_semi")
    print("[contam] NOTE: purification uses the same conformal scoring as the scale channel, so novelty AUROC "
          "measured against the purified novel_semi set is circular for that channel; use it as a cleaning step.")
    return lab2, kind2, report


def make_dataset(subject, session, data_root, refine_contam=True, shrink=0.1, contam_alpha=0.1,
                 n_splits=5, min_n=10, windows=None):
    W, blk, lab, kind = windows if windows is not None else build_windows(subject, session, data_root)
    named_vocab = sorted(set(lab[kind == "named"].tolist()))
    if len(named_vocab) < 2:
        raise SystemExit("fewer than 2 named classes; check the labels")
    raw_lab, raw_kind = lab, kind
    report = None
    if refine_contam:
        cmap = build_contam_map(lab, kind, named_vocab)
        lab, kind, report = refine_contam_windows(feat_scale(W), lab, kind, blk, named_vocab, cmap,
                                                  shrink=shrink, alpha=contam_alpha)
    targets = [("novel_structured (all)", kind == "novel_structured"),
               ("novel_semi (all)", kind == "novel_semi")]
    for k in ("novel_structured", "novel_semi"):
        for lb in sorted(set(lab[kind == k].tolist())):
            m = (kind == k) & (lab == lb)
            if m.sum() >= min_n:
                targets.append((f"  -> {lb}", m))
    return Dataset(name=f"ajile_{subject.replace('-', '')}_{session.replace('-', '')}", W=W, fs=TARGET_FS,
                   label=lab, kind=kind, group=blk, named_vocab=named_vocab,
                   folds=group_kfold_over_labelled(blk, kind == "named", n_splits),
                   truth=kind.copy(), truth_label=lab.copy(), targets=targets,
                   meta=dict(subject=subject, session=session, refine_contam=refine_contam,
                             contam_report=report, win=WIN, hop=HOP),
                   extras=dict(raw_kind=raw_kind, raw_label=raw_lab))
