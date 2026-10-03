"""Extract one AJILE12 session from DANDI into the format used by TC-REJECT.

The raw download is cached before preprocessing. Missing spans of at most
``MAX_INTERP_GAP_SEC`` are interpolated per channel; longer spans are removed
across all channels, and overlapping annotation blocks are split and remapped
to the compacted timeline.

Optional ``--pose`` extraction places keypoint position and velocity on the
same excision-aware timeline. Use ``--inspect --pose`` to confirm that a
session's NWB pose interface is supported before extracting pose data.

Install:
    pip install dandi pynwb h5py remfile
"""
from pathlib import Path
import argparse
import numpy as np
import csv
import time

DANDISET_ID = "000055"
# Defaults preserve the original sub-01/ses-4 behavior exactly; --subject/
# --session (added for the multi-subject extraction, decision #4) override
# these at the top of main() via `global`, computing subject/session-keyed
# paths so different subjects never collide on disk.
ASSET_PATH = "sub-01/sub-01_ses-4_behavior+ecephys.nwb"
OUT_DIR = Path(".")

NAMED_LABELS = {"Talk", "Computer/phone"}
STRUCTURED_NOVEL_LABELS = {"Sleep/rest", "TV", "Inactive", "Other activity"}

MAX_INTERP_GAP_SEC = 2.0
CHUNK_SAMPLES = 2_000_000

RAWCACHE_PATH = OUT_DIR / "ajile_sub01_ses4_full_RAWCACHE.npy"
RAWCACHE_META_PATH = OUT_DIR / "ajile_sub01_ses4_full_RAWCACHE_meta.npz"
POSE_RAWCACHE_PATH = OUT_DIR / "ajile_sub01_ses4_full_POSE_RAWCACHE.npz"


def classify(label):
    if label in NAMED_LABELS:
        return "named"
    if label in STRUCTURED_NOVEL_LABELS:
        return "novel_structured"
    if "Blocklist" in label:
        return "blocklist"
    if "," in label:
        return "novel_semi"
    return "other"


def _open_remote(asset_path):
    import remfile, h5py, pynwb
    from dandi.dandiapi import DandiAPIClient
    with DandiAPIClient() as client:
        asset = client.get_dandiset(DANDISET_ID, "draft").get_asset_by_path(asset_path)
        url = asset.get_content_url(follow_redirects=1, strip_query=True)
    f = remfile.File(url)
    h5 = h5py.File(f, "r")
    io = pynwb.NWBHDF5IO(file=h5, load_namespaces=True)
    return io, io.read()


def find_runs(mask_1d):
    """Contiguous True runs in a 1D boolean array -> list of (start, end)."""
    diffs = np.diff(np.concatenate([[0], mask_1d.astype(np.int8), [0]]))
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]
    return list(zip(starts.tolist(), ends.tolist()))


def merge_spans(spans):
    if not spans:
        return []
    spans = sorted(spans)
    merged = [list(spans[0])]
    for s, e in spans[1:]:
        if s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return [tuple(x) for x in merged]


def complement(spans, T):
    """Spans NOT covered by `spans`, i.e. the parts of [0, T) to KEEP."""
    keep, prev = [], 0
    for s, e in spans:
        if s > prev:
            keep.append((prev, s))
        prev = max(prev, e)
    if prev < T:
        keep.append((prev, T))
    return keep


def split_and_remap_block(bs, be, keep_intervals):
    """Intersect [bs, be) with keep_intervals, return list of (new_s, new_e)
    in the COMPACTED (post-excision) timeline. A block overlapping an
    excised span comes back as >1 piece; a block fully inside an excised
    span comes back empty (dropped)."""
    out, cum = [], 0
    for s, e in keep_intervals:
        os_, oe_ = max(bs, s), min(be, e)
        if os_ < oe_:
            out.append((cum + (os_ - s), cum + (oe_ - s)))
        cum += (e - s)
    return out


def remap_points_through_keep_intervals(t_abs, keep_intervals_sec):
    """
    Point-time analogue of `split_and_remap_block`: given an array of
    absolute times `t_abs` (seconds, need not be sorted) and a list of
    non-overlapping, ascending (start_sec, end_sec) intervals to KEEP,
    return (t_new, keep_mask) where t_new[keep_mask] gives each surviving
    point's time on the compacted (post-excision) clock -- the same clock
    `y_clean`'s sample index / fs is on. Points that fall inside an excised
    gap (i.e. in none of keep_intervals_sec) are dropped via keep_mask.
    """
    t_abs = np.asarray(t_abs, dtype=np.float64)
    t_new = np.full_like(t_abs, np.nan)
    keep_mask = np.zeros_like(t_abs, dtype=bool)
    cum = 0.0
    for s, e in keep_intervals_sec:
        m = (t_abs >= s) & (t_abs < e) & ~keep_mask
        t_new[m] = cum + (t_abs[m] - s)
        keep_mask |= m
        cum += (e - s)
    return t_new, keep_mask


def handle_nans(y, fs, max_gap_sec):
    """Per-channel: interpolate short NaN runs in place. Across channels:
    any run LONGER than max_gap_sec anywhere becomes a global excision span
    (cut from every channel, not just the one that was NaN there) so the
    array stays rectangular and one label timeline applies to all channels.
    Returns (y_clean, keep_intervals, n_interpolated, n_excised_samples,
    n_long_spans)."""
    max_gap_samples = int(max_gap_sec * fs)
    n_channels, T = y.shape

    long_spans = []
    n_short_runs = 0
    for c in range(n_channels):
        for s, e in find_runs(np.isnan(y[c])):
            if (e - s) > max_gap_samples:
                long_spans.append((s, e))
            else:
                n_short_runs += 1
    long_spans = merge_spans(long_spans)

    if long_spans:
        total_excised = sum(e - s for s, e in long_spans)
        print(f"  {len(long_spans)} long gap(s) found, {total_excised} samples "
              f"({total_excised/fs:.1f}s) total -- excising across all channels:")
        for s, e in long_spans[:20]:
            print(f"    [{s/fs:.1f}s -> {e/fs:.1f}s]  length {(e-s)/fs:.1f}s")
        if len(long_spans) > 20:
            print(f"    ... and {len(long_spans) - 20} more")
    else:
        print("  no gaps exceed the long-gap threshold -- nothing to excise")

    keep_intervals = complement(long_spans, T)
    y_clean = np.concatenate([y[:, s:e] for s, e in keep_intervals], axis=1)
    del y

    # remaining NaNs (all now guaranteed <= max_gap_samples per channel,
    # since anything longer was already cut above) -- interpolate per channel
    total_interpolated = 0
    for c in range(n_channels):
        row = y_clean[c]
        nan_mask = np.isnan(row)
        if not nan_mask.any():
            continue
        for s, e in find_runs(nan_mask):
            gap_len = e - s
            assert gap_len <= max_gap_samples, (
                f"channel {c}: leftover gap {gap_len/fs:.1f}s after excision "
                f"-- bug in the excision logic, do not proceed silently.")
            left = row[s - 1] if s > 0 else row[e]
            right = row[e] if e < row.shape[0] else row[s - 1]
            row[s:e] = np.linspace(left, right, gap_len, endpoint=False)
            total_interpolated += gap_len

    print(f"  interpolated {total_interpolated} short-gap NaN samples "
          f"({n_short_runs} runs) across {n_channels} channels")
    return y_clean, keep_intervals, total_interpolated, sum(e - s for s, e in long_spans), len(long_spans)


# ---------------------------------------------------------------------------
# Optional pose extraction and alignment to the compacted signal timeline.
# ---------------------------------------------------------------------------
def _describe_pose_interface(name, v):
    kind = type(v).__name__
    print(f"    {name}: {kind}")
    series_dict = getattr(v, "pose_estimation_series", None)
    if series_dict:
        print(f"      (looks like ndx-pose PoseEstimation) nodes={list(series_dict.keys())}")
        for node, s in series_dict.items():
            d = getattr(s, "data", None)
            conf = getattr(s, "confidence", None)
            print(f"        {node}: data.shape={getattr(d, 'shape', '?')} "
                  f"rate={getattr(s, 'rate', None)} "
                  f"has_timestamps={getattr(s, 'timestamps', None) is not None} "
                  f"has_confidence={conf is not None}")
        return
    spatial_dict = getattr(v, "spatial_series", None)
    if spatial_dict:
        print(f"      (looks like pynwb Position) series={list(spatial_dict.keys())}")
        for node, s in spatial_dict.items():
            d = getattr(s, "data", None)
            print(f"        {node}: data.shape={getattr(d, 'shape', '?')} "
                  f"rate={getattr(s, 'rate', None)} "
                  f"has_timestamps={getattr(s, 'timestamps', None) is not None}")
        return
    print(f"      (unrecognized pose-like container; attrs: "
          f"{[a for a in dir(v) if not a.startswith('_')][:25]})")


def inspect_pose(nwb):
    """Called from --inspect when --pose is also passed. Prints candidate
    pose interfaces so the _find_pose accessor guesses can be checked/fixed
    BEFORE running a real --pose extraction."""
    print("\n=== pose interfaces (best-effort; fix _find_pose if this "
          "doesn't match what's actually there) ===")
    found = False
    for mod_name, mod in nwb.processing.items():
        for k, v in mod.data_interfaces.items():
            looks_pose = (
                hasattr(v, "pose_estimation_series")
                or hasattr(v, "spatial_series")
                or "pose" in k.lower()
                or "position" in k.lower()
            )
            if looks_pose:
                found = True
                print(f"  [{mod_name}] ->", end=" ")
                _describe_pose_interface(k, v)
    if not found:
        print("  (nothing matched by name/shape heuristics -- pose may live "
              "under a different processing module name, or in a SEPARATE "
              "NWB file for this subject/session, e.g. a "
              "'*_behavior+image.nwb' asset -- check --list)")


def _find_pose(nwb):
    """
    Return {node_name: (timestamps_sec[np.ndarray], xy[np.ndarray T x 2],
    confidence[np.ndarray T] or None)} for every pose keypoint found.
    UNVERIFIED against a real file -- see module docstring.
    """
    def _series_timestamps(s, n):
        ts = getattr(s, "timestamps", None)
        if ts is not None:
            return np.asarray(ts[:], dtype=np.float64)
        rate = getattr(s, "rate", None)
        starting_time = getattr(s, "starting_time", 0.0) or 0.0
        if rate:
            return starting_time + np.arange(n, dtype=np.float64) / float(rate)
        raise ValueError("pose series has neither timestamps nor rate")

    for mod in nwb.processing.values():
        for k, v in mod.data_interfaces.items():
            series_dict = getattr(v, "pose_estimation_series", None)
            if series_dict:
                out = {}
                for node, s in series_dict.items():
                    data = np.asarray(s.data[:], dtype=np.float64)
                    xy = data[:, :2]
                    ts = _series_timestamps(s, xy.shape[0])
                    conf = getattr(s, "confidence", None)
                    conf = np.asarray(conf[:], dtype=np.float64) if conf is not None else None
                    out[node] = (ts, xy, conf)
                print(f"[pose] using ndx-pose interface '{k}' in module "
                      f"'{mod.name}': nodes={list(out.keys())}")
                return out

    for mod in nwb.processing.values():
        for k, v in mod.data_interfaces.items():
            spatial_dict = getattr(v, "spatial_series", None)
            if spatial_dict:
                out = {}
                for node, s in spatial_dict.items():
                    data = np.asarray(s.data[:], dtype=np.float64)
                    xy = data[:, :2]
                    ts = _series_timestamps(s, xy.shape[0])
                    out[node] = (ts, xy, None)
                print(f"[pose] using pynwb Position interface '{k}' in module "
                      f"'{mod.name}': nodes={list(out.keys())} (no confidence field)")
                return out

    raise KeyError(
        "no pose interface found -- run `--inspect ... --pose` and look at "
        "the 'pose interfaces' section. AJILE12 may store pose in a "
        "SEPARATE NWB file per session (check --list for a second asset "
        "alongside the behavior+ecephys one) rather than in this file.")


def _resample_and_diff(ts, xy, conf, t0, t1, out_rate, conf_thresh=0.3, max_gap_s=0.5):
    """
    Interpolate one keypoint's (possibly irregular, possibly low-confidence)
    position onto a uniform grid over [t0, t1) at out_rate Hz, then finite-
    difference to velocity. Low-confidence samples (if confidence is
    available) are treated as missing and interpolated through; gaps longer
    than max_gap_s are left as NaN rather than papered over with a straight
    line across, e.g., an occlusion.

    IMPORTANT for the real AJILE12 layout (confirmed via --inspect --pose):
    the plain pynwb Position/SpatialSeries container has NO confidence field
    at all -- so if a keypoint is occluded, the ONLY signal is a literal NaN
    in the (T,2) position array itself, not a confidence dip or a timestamp
    gap (this container has a fixed rate, so `ts` is a synthetic, gapless
    grid regardless of tracking quality). A NaN-valued sample is therefore
    explicitly dropped from the mask below, same as a low-confidence sample
    would be -- this routes it through the SAME gap-length bridge/excise
    logic just below (short holes interpolated across, long ones left NaN)
    instead of letting a raw NaN leak into np.interp's arithmetic uncontrolled
    (NaN propagates through +/-/* to more output samples than the actual gap).
    """
    mask = (ts >= t0) & (ts < t1)
    if conf is not None:
        mask &= conf >= conf_thresh
    mask &= ~np.isnan(xy).any(axis=1)
    ts_v, xy_v = ts[mask], xy[mask]
    if ts_v.shape[0] < 2:
        raise ValueError(f"fewer than 2 valid pose samples in [{t0}, {t1}) "
                          f"-- confidence threshold {conf_thresh} may be too strict, "
                          f"or this window has no pose coverage")

    grid = np.arange(t0, t1, 1.0 / out_rate)
    pos = np.full((grid.shape[0], 2), np.nan, dtype=np.float64)
    for dim in range(2):
        pos[:, dim] = np.interp(grid, ts_v, xy_v[:, dim], left=np.nan, right=np.nan)

    gap_before = np.searchsorted(ts_v, grid)
    gap_before = np.clip(gap_before, 1, ts_v.shape[0] - 1)
    local_gap = ts_v[gap_before] - ts_v[gap_before - 1]
    pos[local_gap > max_gap_s, :] = np.nan

    vel = np.full_like(pos, np.nan)
    vel[1:-1] = (pos[2:] - pos[:-2]) * (out_rate / 2.0)   # central difference
    # plain np.sum (NOT nansum): a NaN in either component must make the
    # sample's speed NaN too, rather than silently reading as zero velocity.
    speed = np.sqrt(np.sum(vel ** 2, axis=1))
    return grid - t0, pos, vel, speed


def _load_or_fetch_pose(open_nwb_fn, cache_path):
    """
    Load raw (per-keypoint timestamps/xy/confidence) pose from cache if
    present; otherwise call open_nwb_fn() -> (io, nwb), run _find_pose,
    cache the result, and close io. Kept separate from the ECoG RAWCACHE so
    a pose-only re-run (e.g. after fixing a confidence threshold) never
    re-touches the (much larger) ECoG cache, and vice versa.
    """
    if cache_path.exists():
        print(f"[pose cache] loading {cache_path}", flush=True)
        d = np.load(cache_path, allow_pickle=True)
        nodes = list(d["nodes"])
        pose_raw = {}
        for node in nodes:
            safe = str(node)
            conf_key = f"{safe}__conf"
            pose_raw[node] = (
                d[f"{safe}__ts"], d[f"{safe}__xy"],
                d[conf_key] if conf_key in d.files else None,
            )
        return pose_raw

    io, nwb = open_nwb_fn()
    try:
        pose_raw = _find_pose(nwb)
    finally:
        io.close()

    save_kwargs = {"nodes": np.array(list(pose_raw.keys()))}
    for node, (ts, xy, conf) in pose_raw.items():
        safe = str(node)
        save_kwargs[f"{safe}__ts"] = ts
        save_kwargs[f"{safe}__xy"] = xy
        if conf is not None:
            save_kwargs[f"{safe}__conf"] = conf
    np.savez_compressed(cache_path, **save_kwargs)
    print(f"[pose cache] saved {cache_path}", flush=True)
    return pose_raw


def extract_and_remap_pose(pose_raw, orig_duration_sec, keep_intervals_sec,
                            out_rate=30.0, conf_thresh=0.3):
    """
    For every keypoint: resample to a uniform grid over the FULL original
    (pre-excision) session, compute velocity, then drop samples that fall
    inside an excised ECoG gap and shift the survivors onto the SAME
    compacted clock y_clean's samples are on (see module docstring).
    Returns {node: {t, pos, vel, speed}} on that compacted clock.
    """
    out = {}
    for node, (ts, xy, conf) in pose_raw.items():
        try:
            t_grid, pos, vel, speed = _resample_and_diff(
                ts, xy, conf, 0.0, orig_duration_sec, out_rate, conf_thresh)
        except ValueError as e:
            print(f"[pose] skipping keypoint '{node}': {e}")
            continue
        t_new, keep_mask = remap_points_through_keep_intervals(t_grid, keep_intervals_sec)
        n_dropped_by_excision = int((~keep_mask).sum())
        t_new, pos, vel, speed = (t_new[keep_mask], pos[keep_mask],
                                   vel[keep_mask], speed[keep_mask])
        coverage = 1.0 - np.mean(np.isnan(speed)) if speed.size else 0.0
        print(f"[pose] '{node}': {speed.shape[0]} samples survive excision "
              f"({n_dropped_by_excision} dropped by ECoG-gap excision), "
              f"{100*coverage:.1f}% coverage after confidence/gap filtering")
        out[node] = dict(t=t_new, pos=pos, vel=vel, speed=speed)
    if not out:
        raise ValueError("every keypoint failed to extract -- see per-keypoint "
                          "messages above")
    return out


def main():
    global OUT_DIR, RAWCACHE_PATH, RAWCACHE_META_PATH, POSE_RAWCACHE_PATH, ASSET_PATH

    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=str, default="sub-01",
                    help="AJILE12 subject id, e.g. sub-01, sub-05 (decision #4, "
                         "cross-subject extraction). Default sub-01 reproduces "
                         "the original single-subject behavior exactly.")
    ap.add_argument("--session", type=str, default="ses-4",
                    help="session id, e.g. ses-4, ses-7. Default ses-4.")
    ap.add_argument(
        "--data_root",
        type=Path,
        required=True,
        help="Directory under which AJILE12_<subject> extraction folders are written.",
    )
    ap.add_argument("--inspect", action="store_true",
                    help="print NWB structure (incl. pose interfaces if --pose "
                         "is also given) and exit -- run this before a real "
                         "--pose extraction.")
    ap.add_argument("--pose", action="store_true",
                    help="also extract continuous keypoint position/velocity, "
                         "remapped onto the same excision-compacted clock as "
                         "the ECoG array (UNVERIFIED accessor -- see module "
                         "docstring; run --inspect --pose first)")
    ap.add_argument("--pose_rate", type=float, default=30.0,
                    help="output sample rate (Hz) for the interpolated pose grid")
    ap.add_argument("--pose_conf_thresh", type=float, default=0.3,
                    help="drop pose samples below this confidence before "
                         "interpolating (ignored if the pose interface has no "
                         "confidence field)")
    args = ap.parse_args()

    subject_tag = args.subject.replace("-", "")  # sub-05 -> sub05, for filenames
    session_tag = args.session.replace("-", "")  # ses-7 -> ses7

    ASSET_PATH = f"{args.subject}/{args.subject}_{args.session}_behavior+ecephys.nwb"
    OUT_DIR = args.data_root.resolve() / f"AJILE12_{subject_tag}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RAWCACHE_PATH = OUT_DIR / f"ajile_{subject_tag}_{session_tag}_full_RAWCACHE.npy"
    RAWCACHE_META_PATH = OUT_DIR / f"ajile_{subject_tag}_{session_tag}_full_RAWCACHE_meta.npz"
    POSE_RAWCACHE_PATH = OUT_DIR / f"ajile_{subject_tag}_{session_tag}_full_POSE_RAWCACHE.npz"

    print(f"[extract] subject={args.subject} session={args.session} -> "
          f"asset={ASSET_PATH}  out_dir={OUT_DIR}", flush=True)

    if args.inspect:
        io, nwb = _open_remote(ASSET_PATH)
        try:
            print("=== session ===")
            print(" id:", nwb.session_id, " start:", nwb.session_start_time)
            print(" subject:", nwb.subject)
            print("\n=== acquisition ===")
            for k, v in nwb.acquisition.items():
                d = getattr(v, "data", None)
                print(f"  {k}: {type(v).__name__} "
                      f"shape={d.shape if d is not None else '?'} "
                      f"rate={getattr(v, 'rate', None)}")
            print("\n=== processing modules ===")
            for mod_name, mod in nwb.processing.items():
                print(f"  [{mod_name}]")
                for k, v in mod.data_interfaces.items():
                    print(f"    {k}: {type(v).__name__}")
            if args.pose:
                inspect_pose(nwb)
            print("\n=== intervals ===")
            if nwb.intervals:
                for k, tbl in nwb.intervals.items():
                    print(f"  {k}: n={len(tbl)} cols={list(tbl.colnames)}")
        finally:
            io.close()
        return

    t0 = time.time()
    pose_raw = None

    if RAWCACHE_PATH.exists() and RAWCACHE_META_PATH.exists():
        print(f"[cache] loading raw download from {RAWCACHE_PATH} "
              f"(skipping re-download)", flush=True)
        y = np.load(RAWCACHE_PATH)
        meta = np.load(RAWCACHE_META_PATH, allow_pickle=True)
        fs = float(meta["fs"])
        channel_ids = meta["channel_ids"]
        rows = [tuple(r) for r in meta["rows"]]
        rows = [(str(lab), float(s), float(e)) for lab, s, e in rows]
        print(f"[cache] loaded array {y.shape}  [{time.time()-t0:.0f}s]", flush=True)
        if args.pose:
            pose_raw = _load_or_fetch_pose(lambda: _open_remote(ASSET_PATH), POSE_RAWCACHE_PATH)
    else:
        io, nwb = _open_remote(ASSET_PATH)
        try:
            series = None
            for key in ("ElectricalSeries", "ECoG", "ecog", "raw"):
                if key in nwb.acquisition:
                    series = nwb.acquisition[key]
                    break
            if series is None:
                for k, v in nwb.acquisition.items():
                    if hasattr(v, "electrodes"):
                        print(f"[loader] using acquisition['{k}'] as ECoG")
                        series = v
                        break
            if series is None:
                raise KeyError("no ECoG series found")

            fs = float(series.rate)
            n_total, n_ch = series.data.shape
            print(f"fs={fs}  total_samples={n_total}  channels={n_ch}  "
                  f"({n_total/fs/3600:.2f}h)", flush=True)

            tbl = nwb.intervals["epochs"] if (nwb.intervals and "epochs" in nwb.intervals) else \
                  next(iter(nwb.intervals.values()))
            df = tbl.to_dataframe()
            label_col = "label" if "label" in df.columns else "labels"
            start_col = "start_time" if "start_time" in df.columns else "start_sec"
            stop_col = "stop_time" if "stop_time" in df.columns else "end_sec"
            rows = [(str(r[label_col]), float(r[start_col]), float(r[stop_col]))
                    for _, r in df.iterrows()]
            kinds = {}
            for lab, s, e in rows:
                k = classify(lab)
                kinds[k] = kinds.get(k, 0) + (e - s)
            print("time budget by kind (s):", {k: round(v) for k, v in kinds.items()},
                  flush=True)

            chunks = []
            for i0 in range(0, n_total, CHUNK_SAMPLES):
                i1 = min(n_total, i0 + CHUNK_SAMPLES)
                chunk = series.data[i0:i1, :].astype(np.float32).T
                chunks.append(chunk)
                print(f"  downloaded {i1}/{n_total} samples "
                      f"[{time.time()-t0:.0f}s elapsed]", flush=True)
            y = np.concatenate(chunks, axis=1)
            del chunks
            print(f"full array: {y.shape}  [{time.time()-t0:.0f}s]", flush=True)

            channel_ids = np.array(nwb.electrodes["id"][:]) if nwb.electrodes is not None else None

            print(f"[cache] saving raw download to {RAWCACHE_PATH} so a NaN-handling "
                  f"bug doesn't cost another download...", flush=True)
            np.save(RAWCACHE_PATH, y)
            np.savez(RAWCACHE_META_PATH, fs=fs, channel_ids=channel_ids,
                     rows=np.array(rows, dtype=object))
            print(f"[cache] saved  [{time.time()-t0:.0f}s]", flush=True)

            if args.pose:
                # pose is fetched from the SAME already-open nwb handle here,
                # so a fresh full download never pays for a second remote
                # connection just to also get pose.
                if POSE_RAWCACHE_PATH.exists():
                    pose_raw = _load_or_fetch_pose(lambda: (None, None), POSE_RAWCACHE_PATH)
                else:
                    pose_raw = _find_pose(nwb)
                    save_kwargs = {"nodes": np.array(list(pose_raw.keys()))}
                    for node, (ts, xy, conf) in pose_raw.items():
                        safe = str(node)
                        save_kwargs[f"{safe}__ts"] = ts
                        save_kwargs[f"{safe}__xy"] = xy
                        if conf is not None:
                            save_kwargs[f"{safe}__conf"] = conf
                    np.savez_compressed(POSE_RAWCACHE_PATH, **save_kwargs)
                    print(f"[pose cache] saved {POSE_RAWCACHE_PATH}", flush=True)

        finally:
            io.close()

    # orig_duration_sec must be measured on the PRE-excision array (same
    # timeline the label rows and pose timestamps are both expressed in).
    orig_duration_sec = y.shape[1] / fs

    # --- NaN handling: interpolate short gaps, EXCISE long ones ---
    y_clean, keep_intervals, n_interp, n_excised, n_long = handle_nans(
        y, fs, MAX_INTERP_GAP_SEC)
    del y
    assert not np.isnan(y_clean).any(), "NaN remains after handling -- bug"

    # --- remap every label row into the compacted timeline, splitting any
    #     block that straddled an excised span ---
    labels_path = OUT_DIR / f"ajile_{subject_tag}_{session_tag}_full_labels.csv"
    n_split = 0
    with open(labels_path, "w", newline="") as lf:
        lw = csv.writer(lf)
        lw.writerow(["label", "start_sec", "end_sec", "kind"])
        for lab, s, e in rows:
            bs, be = int(round(s * fs)), int(round(e * fs))
            pieces = split_and_remap_block(bs, be, keep_intervals)
            if len(pieces) > 1:
                n_split += 1
            for ns, ne in pieces:
                lw.writerow([lab, ns / fs, ne / fs, classify(lab)])
    print(f"wrote {labels_path}  ({n_split} block(s) split by excision)", flush=True)

    pose_out = None
    if args.pose:
        keep_intervals_sec = [(s / fs, e / fs) for s, e in keep_intervals]
        pose_out = extract_and_remap_pose(
            pose_raw, orig_duration_sec, keep_intervals_sec,
            out_rate=args.pose_rate, conf_thresh=args.pose_conf_thresh)

    npz_path = OUT_DIR / f"ajile_{subject_tag}_{session_tag}_full.npz"
    save_kwargs = dict(y=y_clean, fs=fs, channel_ids=channel_ids)
    if pose_out:
        save_kwargs["pose_nodes"] = np.array(list(pose_out.keys()))
        save_kwargs["pose_rate"] = args.pose_rate
        for node, d in pose_out.items():
            safe = str(node).replace(" ", "_")
            save_kwargs[f"pose_{safe}_t"] = d["t"]
            save_kwargs[f"pose_{safe}_pos"] = d["pos"]
            save_kwargs[f"pose_{safe}_vel"] = d["vel"]
            save_kwargs[f"pose_{safe}_speed"] = d["speed"]
    np.savez_compressed(npz_path, **save_kwargs)
    print(f"wrote {npz_path} ({npz_path.stat().st_size/1e9:.2f} GB)  "
          f"shape={y_clean.shape}  [{time.time()-t0:.0f}s total]", flush=True)
    print(f"\nSUMMARY: {n_long} long gap(s) excised ({n_excised/fs:.1f}s total), "
          f"{n_interp} short-gap samples interpolated, {n_split} label block(s) "
          f"split around excisions.")
    print("If n_long/n_excised looks unexpectedly large relative to the session "
          "length, stop and look at the printed gap list above before trusting "
          "this output -- it should be a small fraction of 24h.")
    if pose_out:
        print("\npose coverage per keypoint, on the compacted clock (fraction "
              "of surviving samples not NaN after confidence/gap filtering):")
        for node, d in pose_out.items():
            cov = 1.0 - np.mean(np.isnan(d["speed"])) if d["speed"].size else 0.0
            print(f"  {node:20s} {100*cov:.1f}%  ({d['speed'].shape[0]} samples "
                  f"@ {args.pose_rate} Hz)")


if __name__ == "__main__":
    main()
