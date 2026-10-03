"""DBS adapter (two patients, one microelectrode recording each; clinical labels vs ground-truth labels).

This adapter is released without any clinical data. The layout below is the schema expected for
an authorized local copy of the protected recordings.

Layout under ``<root>``:

    S1/<recording_id>/{Data,Labels}                   S2/<recording_id>/{Data,Labels}
    Data/micro<STRUCT>_<L|R>_<k>_CommonFiltered.mat   MATLAB v7.3, variables ``data`` (N, 1) float64 and ``fs`` (24414.0625 Hz)
    Labels/Clinical Labels.txt, Labels/Ground Truth.txt    lines ``Note-n: [start end] [label] ""`` (seconds from sample 0 of the recording)

``<recording_id>`` is whatever the data-access process names it; ``find_recording`` below only requires
that exactly one such subfolder (containing ``Data`` and ``Labels``) exists under each patient directory,
and ``patient_tag`` derives "s1"/"s2" directly from the ``S1``/``S2`` folder names, so no internal
case or session identifier needs to appear anywhere in this codebase.

What the method sees: the CLINICAL labels (Rest, Movement) as given. What is used only for evaluation: the GROUND-TRUTH labels, which differ from
the clinical ones only by an extra label "Extra Movement" carved out of clinical Rest (the patient moved while labelled Rest).

Signal preparation (fixed, identical in spirit to the AJILE adapter): per channel two zero-phase decimations by 10 (24414.0625 -> 244.140625 Hz),
zero-phase band-pass 1-40 Hz, rational resampling 256/625 to exactly 100 Hz, z-score per channel over the recording. Channels are grouped by
structure and side (GPi1, GPi2, VIM, VOSTN x L, R); a group needs at least ``min_group_channels`` (5) channels; every kept group is trimmed to its
first 5 channels by file index, and the groups are concatenated in alphabetical order (8 groups x 5 = 40 channels).

Windows: 2 s (200 samples), hop 1 s (50 % overlap), built inside every ground-truth segment separately (window start = segment start + 0, 1, 2, ...
s while the window fits), so no window crosses a label boundary of either label file. Not Visible segments (S2, 0-23 s) are not windowed.

Per window: label = clinical label (kind 'named' for all of them); truth = 'named' when the ground-truth label equals the clinical one, 'extra_movement'
when the ground truth says Extra Movement; group = the clinical segment (Note index) the window lies in, so a whole clinical segment is held out
together; folds = GroupKFold over the clinical segments (5 folds).
"""
import re
from pathlib import Path

import numpy as np
from scipy.signal import butter, decimate, resample_poly, sosfiltfilt

from ..folds import group_kfold_over_labelled
from .base import Dataset

FS_STORED = 24414.0625
TARGET_FS = 100.0
BAND = (1.0, 40.0)
WIN = 200            # 2 s at 100 Hz
HOP = 100            # 1 s, 50 % overlap
MIN_GROUP_CHANNELS = 5
FILE_RE = re.compile(r"^micro(?P<struct>[A-Za-z0-9]+?)_(?P<side>[LR])_(?P<idx>\d+)_CommonFiltered\.mat$")
NOTE_RE = re.compile(r"Note-\d+\s*:?\s*\[\s*([\d.]+)\s+([\d.]+)\s*\]\s*\[(.+?)\]")   # the colon after Note-N is optional (one label file omits it)
NOT_VISIBLE = "Not Visible"
EXTRA = "Extra Movement"


# ------------------------------------------------------------------ labels
def parse_labels(path):
    """[(start, end, label)] in seconds. A line that starts with 'Note' but cannot be parsed raises: it is never silently dropped."""
    segs = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        m = NOTE_RE.search(line)
        if m:
            segs.append((float(m.group(1)), float(m.group(2)), m.group(3).strip()))
        elif line.strip().lower().startswith("note"):
            raise ValueError(f"{Path(path).name}: cannot parse label line {line.strip()!r}")
    if not segs:
        raise ValueError(f"{Path(path).name}: no label segments found")
    return sorted(segs)


# ------------------------------------------------------------------ signal
def read_channel(path):
    """(signal float64, fs) from a MATLAB v7.3 (HDF5) or older file with variables ``data`` and ``fs``"""
    try:
        import h5py
        if h5py.is_hdf5(str(path)):
            with h5py.File(str(path), "r") as f:
                return np.asarray(f["data"], dtype=float).ravel(), float(np.ravel(np.array(f["fs"]))[0])
    except ImportError:
        pass
    from scipy.io import loadmat
    d = loadmat(str(path), variable_names=["data", "fs"])
    return np.asarray(d["data"], float).ravel(), float(np.ravel(d["fs"])[0])


def to_100hz(x, fs):
    if abs(fs - FS_STORED) > 1e-3:
        raise ValueError(f"fs = {fs} is not {FS_STORED} Hz; the 100 Hz route of this adapter assumes the stored rate")
    y = decimate(decimate(x, 10, ftype="iir", zero_phase=True), 10, ftype="iir", zero_phase=True)     # 244.140625 Hz
    fs2 = fs / 100.0
    sos = butter(4, [BAND[0] / (fs2 / 2), BAND[1] / (fs2 / 2)], btype="bandpass", output="sos")
    return resample_poly(sosfiltfilt(sos, y), 256, 625)          # 244.140625 * 256 / 625 = 100.0 Hz exactly


def load_signals(rec_dir, cache_file=None, min_group_channels=MIN_GROUP_CHANNELS):
    """(Y (channels, samples) float32 at 100 Hz z-scored, channel tags, group names). Cached in cache_file (npz) when given."""
    if cache_file is not None and Path(cache_file).exists():
        z = np.load(cache_file, allow_pickle=False)
        print(f"[dbs] loaded cached 100 Hz signals {cache_file}")
        return z["Y"], [str(t) for t in z["tags"]], [str(g) for g in z["groups"]]
    groups = {}
    for f in sorted((Path(rec_dir) / "Data").glob("*.mat")):
        m = FILE_RE.match(f.name)
        if m:
            groups.setdefault(f"{m['struct']}_{m['side']}", []).append((int(m["idx"]), f))
    kept = {g: sorted(v)[:min_group_channels] for g, v in groups.items() if len(v) >= min_group_channels}
    dropped = {g: len(v) for g, v in groups.items() if len(v) < min_group_channels}
    if dropped:
        print(f"[dbs] dropping groups with fewer than {min_group_channels} channels: {dropped}")
    if not kept:
        raise SystemExit(f"no structure/side group with {min_group_channels} channels under {rec_dir}")
    Y, tags, gnames = [], [], []
    for g in sorted(kept):
        for idx, f in kept[g]:
            x, fs = read_channel(f)
            if not np.isfinite(x).all():
                raise ValueError(f"{f.name}: non-finite samples; the adapter does not repair them")
            y = to_100hz(x, fs).astype(np.float32)
            Y.append((y - y.mean()) / (y.std() + 1e-8))
            tags.append(f"{g}_{idx}")
            gnames.append(g)
        print(f"[dbs] {Path(rec_dir).parent.name}: group {g}: channels {[i for i, _ in kept[g]]}", flush=True)
    L = min(len(y) for y in Y)
    Y = np.stack([y[:L] for y in Y]).astype(np.float32)
    if cache_file is not None:
        Path(cache_file).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_file, Y=Y, tags=np.array(tags), groups=np.array(gnames))
    return Y, tags, gnames


# ------------------------------------------------------------------ windows
def build_windows(rec_dir, cache_file=None):
    lab = Path(rec_dir) / "Labels"
    clin = parse_labels(lab / "Clinical Labels.txt")
    truth = parse_labels(lab / "Ground Truth.txt")
    Y, tags, gnames = load_signals(rec_dir, cache_file)
    C, T = Y.shape
    span = max(e for _, e, _ in clin + truth)
    if span * TARGET_FS > T + 1e-6:
        raise ValueError(f"labels reach {span} s but the signal has only {T / TARGET_FS:.2f} s")
    # clinical segment index for grouping (Not Visible segments get none)
    clin_ids = {}
    for i, (a, b, l) in enumerate(clin):
        if l != NOT_VISIBLE:
            clin_ids[i] = len(clin_ids)
    Ws, lab_c, lab_t, kind_t, grp, tseg, tstart = [], [], [], [], [], [], []
    n_skipped_nv = 0
    for ti, (a, b, tl) in enumerate(truth):
        if tl == NOT_VISIBLE:
            n_skipped_nv += 1
            continue
        owners = [i for i, (ca, cb, _) in enumerate(clin) if ca - 1e-9 <= a and b <= cb + 1e-9]
        if len(owners) != 1:
            raise ValueError(f"ground-truth segment {a}-{b} [{tl}] is not inside exactly one clinical segment ({owners})")
        ci = owners[0]
        cl = clin[ci][2]
        if cl == NOT_VISIBLE:
            n_skipped_nv += 1
            continue
        if tl == cl:
            tk = "named"
        elif tl == EXTRA:
            tk = "extra_movement"
        else:
            raise ValueError(f"ground-truth label {tl!r} inside clinical {cl!r} at {a}-{b}: no rule for this pair")
        t = a
        while t + WIN / TARGET_FS <= b + 1e-9:
            i0 = int(round(t * TARGET_FS))
            Ws.append(Y[:, i0:i0 + WIN].T)
            lab_c.append(cl)
            lab_t.append(tl)
            kind_t.append(tk)
            grp.append(clin_ids[ci])
            tseg.append(ti)
            tstart.append(t)
            t += HOP / TARGET_FS
    W = np.stack(Ws).astype(np.float32)
    print(f"[dbs] {len(W)} windows of {WIN / TARGET_FS:.0f} s, {C} channels; {n_skipped_nv} Not Visible segment(s) not windowed; "
          f"clinical segments {len(clin_ids)}")
    return dict(W=W, label=np.array(lab_c, dtype=object), truth_label=np.array(lab_t, dtype=object), truth=np.array(kind_t, dtype=object),
                group=np.array(grp, dtype=np.int64), truth_seg=np.array(tseg, dtype=np.int64), t_start=np.array(tstart, dtype=float),
                channel_tags=tags, group_names=sorted(set(gnames)))


def patient_tag(patient_dir):
    m = re.search(r"\((\d+)\)", Path(patient_dir).name)
    return f"s{m.group(1)}" if m else re.sub(r"\W+", "", Path(patient_dir).name).lower()


def find_recording(patient_dir):
    recs = [r for r in Path(patient_dir).iterdir() if r.is_dir() and (r / "Data").exists() and (r / "Labels").exists()]
    if len(recs) != 1:
        raise SystemExit(f"expected exactly one recording folder under {patient_dir}, found {[r.name for r in recs]}")
    return recs[0]


def make_dataset(patient_dir, cache_dir=None, n_splits=5, windows=None):
    patient_dir = Path(patient_dir)
    rec = find_recording(patient_dir)
    tag = patient_tag(patient_dir)
    cache = None if cache_dir is None else Path(cache_dir) / f"dbs_{tag}_100hz.npz"
    d = windows if windows is not None else build_windows(rec, cache)
    named_vocab = sorted(set(d["label"].tolist()))
    if len(named_vocab) < 2:
        raise SystemExit("fewer than 2 clinical classes; check the labels")
    kind = np.full(len(d["W"]), "named", dtype=object)
    targets = [("extra_movement (clinical Rest, ground-truth Extra Movement)", d["truth"] == "extra_movement")]
    return Dataset(name=f"dbs_{tag}", W=d["W"], fs=TARGET_FS, label=d["label"], kind=kind, group=d["group"], named_vocab=named_vocab,
                   folds=group_kfold_over_labelled(d["group"], np.ones(len(kind), bool), n_splits),
                   truth=d["truth"], truth_label=d["truth_label"], targets=targets,
                   meta=dict(patient=patient_dir.name, recording=rec.name, win=WIN, hop=HOP, n_channels=int(d["W"].shape[2]),
                             channel_tags=d["channel_tags"], group_names=d["group_names"]),
                   extras=dict(truth_seg=d["truth_seg"], t_start=d["t_start"]))
