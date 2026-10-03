#!/usr/bin/env python
"""Single entry point for TC-REJECT. The same method, with the same constants,
runs on every dataset; only the data adapter differs.

    python run_tcreject.py synthetic                       # the 10 gate seeds
    python run_tcreject.py synthetic --seeds 11,23
    python run_tcreject.py pamap2 --split all
    python run_tcreject.py ajile --subject sub-01 --session ses-4
    python run_tcreject.py ajile --subjects all            # 10 subject/sessions, resumable
    python run_tcreject.py summarize ajile                 # cohort table from finished runs
    python run_tcreject.py downstream ajile                # original vs purified labels

Outputs go to  <out_root>/<dataset>/<UTC timestamp>/  (default out_root: ./Results).
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEFAULT_DATA = HERE.parent / "Data"
DEFAULT_OUT = HERE / "Results"


def _preflight(args):
    """Fail before any long data loading if the deep channel cannot run (all channels always run)."""
    from tcreject.deep import resolve_device
    resolve_device(args.device)


def _run_and_save(ds, args, cfg):
    from tcreject import run
    from tcreject.io import make_run_dir, save_run
    t0 = time.time()
    res = run(ds, cfg, device=args.device, verbose=True)
    out = make_run_dir(args.out_root, ds.name)
    save_run(out, ds, res, cli_args={k: str(v) for k, v in vars(args).items()})
    print(f"[{ds.name}] done in {time.time() - t0:.0f} s -> {out}\n", flush=True)
    return out


def _cfg(args):
    from dataclasses import replace
    from tcreject import DEFAULT
    return replace(DEFAULT, n_boot=args.n_boot) if args.n_boot != DEFAULT.n_boot else DEFAULT


def cmd_synthetic(args):
    from tcreject.datasets import synthetic
    seeds = ([int(s) for s in args.seeds.split(",")] if args.seeds
             else (synthetic.DEV_SEEDS if args.dev else synthetic.GATE_SEEDS))
    cfg = _cfg(args)
    _preflight(args)
    for s in seeds:
        ds = synthetic.make_dataset(s, open_frac=args.open_frac, closed_frac=args.closed_frac, snr_db=args.snr_db)
        _run_and_save(ds, args, cfg)


def cmd_pamap2(args):
    from tcreject.datasets import pamap2
    root = args.data_root or str(DEFAULT_DATA / "PAMAP2" / "PAMAP2_Dataset")
    cache = args.cache_dir or str(Path(args.out_root) / "_cache")
    _preflight(args)
    windows = pamap2.load_windows(root, cache_dir=cache)
    splits = ["A", "B", "C"] if args.split == "all" else [args.split]
    cfg = _cfg(args)
    for sp in splits:
        _run_and_save(pamap2.make_dataset(sp, root, windows=windows), args, cfg)


def _ajile_done(out_root, subject, session):
    from tcreject.downstream.ajile_classifier import find_run
    return find_run(out_root, subject, session) is not None


def cmd_ajile(args):
    from tcreject.datasets import ajile
    root = args.data_root or str(DEFAULT_DATA)
    if args.subjects:
        pairs = ajile.SUBJECT_SESSIONS if args.subjects == "all" else \
            [p for p in ajile.SUBJECT_SESSIONS if p[0] in args.subjects.split(",")]
    else:
        pairs = [(args.subject, args.session)]
    cfg = _cfg(args)
    _preflight(args)
    failed = []
    for sub, ses in pairs:
        if args.subjects and not args.force and _ajile_done(args.out_root, sub, ses):
            print(f"[{sub}/{ses}] a finished run already exists; skipping (use --force to redo)", flush=True)
            continue
        try:
            ds = ajile.make_dataset(sub, ses, root, refine_contam=not args.no_refine_contam)
            _run_and_save(ds, args, cfg)
        except BaseException as e:   # noqa: BLE001  keep the batch going; report at the end
            if isinstance(e, KeyboardInterrupt):
                raise
            print(f"[{sub}/{ses}] FAILED: {e!r}\n{traceback.format_exc()}", flush=True)
            failed.append((sub, ses, repr(e)))
    if failed:
        print("FAILED subjects:", failed)
        sys.exit(1)


def cmd_summarize(args):
    from tcreject.summary import summarize_ajile
    summarize_ajile(args.out_root)


def cmd_downstream(args):
    from tcreject.downstream.ajile_classifier import run_all
    args.results_root = args.results_root or str(Path(args.out_root))
    args.data_root = args.data_root or str(DEFAULT_DATA)
    args.out_dir = args.out_dir or str(Path(args.out_root) / "downstream_ajile")
    run_all(args)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_root", default=str(DEFAULT_OUT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    root = argparse.ArgumentParser(add_help=False)          # lets --out_root follow the subcommand too
    root.add_argument("--out_root", default=argparse.SUPPRESS)

    def common(p):
        p.add_argument("--device", default="auto", help="auto | cpu | cuda")
        p.add_argument("--n_boot", type=int, default=300)

    p = sub.add_parser("synthetic", parents=[root]); common(p)
    p.add_argument("--seeds", default=None, help="comma list; default = the 10 gate seeds")
    p.add_argument("--dev", action="store_true", help="use the development seeds instead")
    p.add_argument("--open_frac", type=float, default=0.25)
    p.add_argument("--closed_frac", type=float, default=0.10)
    p.add_argument("--snr_db", type=float, default=None)
    p.set_defaults(fn=cmd_synthetic)

    p = sub.add_parser("pamap2", parents=[root]); common(p)
    p.add_argument("--split", default="all", choices=["A", "B", "C", "all"])
    p.add_argument("--data_root", default=None, help="PAMAP2_Dataset folder (contains Protocol/)")
    p.add_argument("--cache_dir", default=None)
    p.set_defaults(fn=cmd_pamap2)

    p = sub.add_parser("ajile", parents=[root]); common(p)
    p.add_argument("--subject", default="sub-01")
    p.add_argument("--session", default="ses-4")
    p.add_argument("--subjects", default=None, help="'all' or a comma list such as sub-01,sub-06 (resumable)")
    p.add_argument("--data_root", default=None, help="folder holding AJILE12_subXX/ (default ../Data)")
    p.add_argument("--no_refine_contam", action="store_true")
    p.add_argument("--force", action="store_true")
    p.set_defaults(fn=cmd_ajile)

    p = sub.add_parser("summarize", parents=[root]); p.add_argument("what", choices=["ajile"]); p.set_defaults(fn=cmd_summarize)

    p = sub.add_parser("downstream", parents=[root])
    p.add_argument("what", choices=["ajile"])
    p.add_argument("--cohort", default="dev", choices=["dev", "fresh"],
                   help="AJILE12 cohort to analyze. Supply --results_root and --out_dir for the selected cohort.")
    p.add_argument("--results_root", default=None, help="where the TC-REJECT runs are (default: --out_root)")
    p.add_argument("--data_root", default=None, help="folder holding AJILE12_subXX/ (raw recordings; default ../Data)")
    p.add_argument("--out_dir", default=None)
    p.add_argument("--only_subjects", default=None)
    p.add_argument("--score_name", default="TC-REJECT")
    p.add_argument("--threshold_mode", default="native", choices=["native", "calibrated"],
                   help="how the saved --score_name becomes a reject/keep decision for the tcr_filtered "
                        "condition. 'native' (default) uses score >= 1/alpha for the e-value score. "
                        "'calibrated' uses a leave-one-group-out conformal threshold from held-out "
                        "true-named windows and is appropriate for rank-average scores.")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--clf", default="xgboost", choices=["xgboost", "rf"],
                   help="per-electrode classifier; xgboost = the paper's best (rf only as a fallback if it cannot be installed)")
    p.add_argument("--n_estimators", type=int, default=100)
    p.add_argument("--conditions", default="all_windows,tcr_filtered,random_filtered,contam_purified,random_matched_compound",
                   help="comma list from: all_windows,tcr_filtered,random_filtered,contam_purified,random_matched_compound,drop_all_compound")
    p.add_argument("--recompute_features", action="store_true", help="ignore the gamma-feature cache")
    p.add_argument("--n_jobs", type=int, default=-1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--budgets", default="full,2500,1000,500,250,100")
    p.add_argument("--n_random_matched", type=int, default=3)
    p.add_argument("--skip_richness", action="store_true")
    p.add_argument("--aggregate_only", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--list", action="store_true")
    p.set_defaults(fn=cmd_downstream)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
