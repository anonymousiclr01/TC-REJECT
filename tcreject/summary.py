"""Cohort tables from finished runs (reads results_<name>.json; recomputes nothing)."""
import json
from pathlib import Path

import numpy as np

from .config import CHANNELS
from .datasets.ajile import SUBJECT_SESSIONS

TARGETS = ("novel_structured (all)", "novel_semi (all)")


def _latest_results(out_root, subject, session):
    tag = f"ajile_{subject.replace('-', '')}_{session.replace('-', '')}"
    base = Path(out_root) / tag
    if not base.is_dir():
        return None, None
    for d in sorted((p for p in base.iterdir() if p.is_dir()), reverse=True):
        f = d / f"results_{tag}.json"
        if f.exists():
            return d, json.loads(f.read_text(encoding="utf-8"))
    return None, None


def summarize_ajile(out_root):
    rows = {t: [] for t in TARGETS}
    print("AJILE12: TC-REJECT (all five channels, fixed configuration) per subject/session\n"
          "AUROC of novel vs genuinely named windows; the latest finished run of each subject is used.\n")
    hdr = (f"{'subject':9}{'target':26}{'n_pos':>7}{'TC-REJECT':>10}{'CI 95%':>14}{'best chan':>10}"
           f"{'oracle':>8}  " + " ".join(f"{k[:6]:>7}" for k in CHANNELS) + "   run")
    print(hdr)
    for sub, ses in SUBJECT_SESSIONS:
        d, r = _latest_results(out_root, sub, ses)
        if r is None:
            print(f"{sub:9}(no finished run)")
            continue
        for t in TARGETS:
            row = next((x for x in r["table"] if x["target"] == t), None)
            if row is None:
                print(f"{sub:9}{t:26}  not reported (fewer than min_n positives)")
                continue
            a = row["auroc"]
            ci = row.get("headline_ci") or [float("nan")] * 2
            best = max(a[f"p_{k}"] for k in CHANNELS)
            orc = row.get("oracle", float("nan"))
            print(f"{sub:9}{t:26}{row['n_pos']:7d}{a['TC-REJECT']:10.3f}{f'[{ci[0]:.2f},{ci[1]:.2f}]':>14}{best:10.3f}"
                  f"{orc:8.3f}  " + " ".join(f"{a[f'p_{k}']:7.3f}" for k in CHANNELS) + f"   {d.name}")
            rows[t].append((sub, a["TC-REJECT"]))
    print()
    for t in TARGETS:
        v = np.array([x[1] for x in rows[t]])
        if len(v):
            print(f"{t:26} subjects with a value: {len(v)}   mean {v.mean():.3f}   min {v.min():.3f}   max {v.max():.3f}")
    print("\nNo cohort headline is selected here: report per-subject values, and state how subjects were "
          "included, alongside any mean.")
