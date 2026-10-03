"""Run TC-REJECT with the shared default configuration on authorized DBS recordings.

For each patient folder under tcreject\\datasets\\DBS:
  1. tcreject.datasets.dbs_adapter.make_dataset builds the dataset using the fixed adapter settings
  2. tcreject.run runs the shared five-channel pipeline and both reported fusion routes
  3. tcreject.io.save_run                 scores_<name>.npz etc. into  Results\\dbs\\<name>\\<UTC stamp>\\
  4. label-conditional scores             tcreject.label_conditional.own_class_scores (out of fold), saved as own_scores_<name>.npz beside the run
Nothing is tuned; nothing reads the ground-truth labels except the adapter, which stores them as `truth` for evaluation only.
The 100 Hz signals are cached in Results\\dbs\\_cache (a few hundred MB) so a re-run does not re-read the 6 GB of .mat files.

Usage (from the repository root, in an environment that has torch):
    python run_dbs.py                    # both patients
    python run_dbs.py --patient S1       # one patient (folder name starts with S1)
"""
import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(HERE / "tcreject" / "datasets" / "DBS"))
    ap.add_argument("--out_root", default=str(HERE / "Results" / "dbs"))
    ap.add_argument("--patient", default="all", help="all, or a prefix of the patient folder name such as S1")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--n_boot", type=int, default=300)
    a = ap.parse_args()

    from dataclasses import replace
    from tcreject import run, DEFAULT
    from tcreject.datasets import dbs_adapter as dbs
    from tcreject.deep import resolve_device
    from tcreject.io import make_run_dir, save_run
    from tcreject.label_conditional import own_class_scores

    resolve_device(a.device)                     # fail before any long loading if the deep channel cannot run
    cfg = replace(DEFAULT, n_boot=a.n_boot) if a.n_boot != DEFAULT.n_boot else DEFAULT
    pats = sorted(p for p in Path(a.root).iterdir() if p.is_dir() and (a.patient == "all" or p.name.startswith(a.patient)))
    if not pats:
        sys.exit(f"no patient folder matching {a.patient!r} under {a.root}")
    failed = []
    for p in pats:
        t0 = time.time()
        try:
            ds = dbs.make_dataset(p, cache_dir=Path(a.out_root) / "_cache")
            res = run(ds, cfg, device=a.device, verbose=True)
            out = make_run_dir(a.out_root, ds.name)
            save_run(out, ds, res, cli_args={k: str(v) for k, v in vars(a).items()})
            y = ds.y()
            own = own_class_scores(ds.W, y, ds.folds)
            np.savez(Path(out) / f"own_scores_{ds.name}.npz", named_vocab=np.array(ds.named_vocab), y=y, group=ds.group, **own)
            print(f"[{ds.name}] done in {time.time() - t0:.0f} s -> {out}\n", flush=True)
        except BaseException as e:   # noqa: BLE001  report, never hide; a failed patient is listed, not replaced
            if isinstance(e, KeyboardInterrupt):
                raise
            print(f"[{p.name}] FAILED: {e!r}\n{traceback.format_exc()}", flush=True)
            failed.append((p.name, repr(e)))
    if failed:
        print("FAILED patients:", failed)
        sys.exit(1)
    print("all requested DBS runs finished")


if __name__ == "__main__":
    main()
