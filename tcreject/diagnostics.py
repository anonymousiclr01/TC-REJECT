"""Calibration diagnostics: conformal p-values of genuinely named windows should be
uniform on [0, 1], so the empirical rejection rate at level alpha should be ~alpha."""
import numpy as np
from scipy.stats import kstest


def null_uniformity(p, is_named, n_bins=10):
    p = np.asarray(p)[np.asarray(is_named)]
    p = p[~np.isnan(p)]
    if len(p) < 20:
        return dict(n=int(len(p)), ks_D=None, ks_p=None, decile_counts=None)
    stat, pval = kstest(p, "uniform")
    counts, _ = np.histogram(p, bins=n_bins, range=(0.0, 1.0))
    return dict(n=int(len(p)), ks_D=float(stat), ks_p=float(pval), decile_counts=counts.tolist())


def coverage_at_alpha(p, is_named, alphas=(0.01, 0.05, 0.10, 0.20)):
    p = np.asarray(p)[np.asarray(is_named)]
    p = p[~np.isnan(p)]
    if len(p) < 20:
        return dict(n=int(len(p)), rows=None)
    rows = []
    for a in alphas:
        emp = float(np.mean(p <= a))
        rows.append(dict(alpha=a, empirical=emp, ratio=emp / a))
    return dict(n=int(len(p)), rows=rows)


def format_calibration(name, nu, cov):
    s = f"    {name:10s} "
    if nu["ks_p"] is None:
        return s + f"n={nu['n']} too few named windows"
    flag = "  <-- NOT UNIFORM (KS p<0.01)" if nu["ks_p"] < 0.01 else ""
    cov_s = "  ".join(f"a={r['alpha']:.2f}:{r['ratio']:.2f}x" for r in (cov["rows"] or []))
    return s + f"n={nu['n']:5d} KS D={nu['ks_D']:.4f} p={nu['ks_p']:.3g}  coverage ratio {cov_s}{flag}"
