"""Persistence: everything needed to recompute any number without re-running."""
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o


def make_run_dir(root, dataset_name):
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    d = Path(root) / dataset_name / stamp
    d.mkdir(parents=True, exist_ok=True)
    return d


def _versions():
    v = {"python": platform.python_version(), "numpy": np.__version__}
    for mod in ("scipy", "sklearn", "torch"):
        try:
            m = __import__(mod)
            v[mod] = m.__version__
        except Exception:
            pass
    return v


def save_run(out_dir, ds, res, cli_args=None, save_features=True):
    out_dir = Path(out_dir)
    have = res["have"]
    widx = np.where(have)[0]
    tag = ds.name.replace("/", "_")
    arrays = {f"score__{k}": v for k, v in res["scores"].items()}
    arrays.update(window_index=widx, block_id=ds.group[have], kind=ds.kind[have],
                  label=ds.label[have], truth=ds.truth[have], truth_label=ds.truth_label[have])
    for k, v in ds.extras.items():
        arrays[k] = v[have]
    for k, v in res["p"].items():
        arrays[f"p__{k}"] = v[have]
    np.savez(out_dir / f"scores_{tag}.npz", **arrays)
    if save_features:
        Zcat = np.concatenate([res["features"][k][have] for k in ("Zs", "Zh", "Zp")], axis=1).astype(np.float32)
        np.savez(out_dir / f"features_{tag}.npz", Zcat=Zcat, window_index=widx)
    summary = dict(dataset=ds.name, meta=ds.meta, n_windows=int(len(have)), n_scored=int(have.sum()),
                   named_vocab=ds.named_vocab, weights=res["weights"], fusion=res["fusion"],
                   calibration=res["calibration"], kappa_auroc=res["kappa_auroc"], table=res["table"])
    (out_dir / f"results_{tag}.json").write_text(json.dumps(_jsonable(summary), indent=1), encoding="utf-8")
    (out_dir / "config.json").write_text(json.dumps(_jsonable(dict(
        config=res["config"], cli_args=cli_args or {}, versions=_versions(), dataset=ds.name)), indent=1),
        encoding="utf-8")
    cols = list(res["scores"].keys())
    with open(out_dir / f"results_table_{tag}.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["target", "n_pos", "n_neg", "oracle", "headline_ci_lo", "headline_ci_hi"] + cols)
        for r in res["table"]:
            ci = r.get("headline_ci", [None, None])
            w.writerow([r["target"], r["n_pos"], r["n_neg"], r.get("oracle"), ci[0], ci[1]]
                       + [r["auroc"].get(c) for c in cols])
    return out_dir
