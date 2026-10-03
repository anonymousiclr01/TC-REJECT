"""Single configuration for TC-REJECT.

Every hyperparameter of the method lives here, with one value that is used on
every dataset. Dataset adapters may supply data-specific *inputs* (window
length, sampling rate, fold scheme) but never override these method constants.
"""
from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Config:
    # --- per-channel conformal calibration -------------------------------
    shrink: float = 0.1          # LDA shrinkage; also the extra ridge on the precision matrix
    max_dim: int = 200           # per-fold PCA cap for high-dimensional feature spaces
    min_class_n: int = 5         # smaller classes fall back to the pooled calibration set
    calibration: str = "group_loo"   # own-class calibration distances: "group_loo" | "insample"
    standardize_on: str = "named"    # features z-scored on labelled ("named") windows | "all"
    n_splits: int = 5            # GroupKFold splits (datasets using block folds)

    # --- e-value fusion ---------------------------------------------------
    kappa: float = 0.4           # Vovk-Wang calibrator e = kappa * p^(kappa-1)
    alpha: float = 0.10          # reject when E >= 1/alpha

    # --- channel weights (probes) ----------------------------------------
    weight_floor: float = 0.5    # AUC floor for "excess over chance" weighting
    max_weight_share: float = 0.65   # weight-collapse safeguard
    n_blend_per_pair: int = 30
    blend_alpha_lo: float = 0.3
    blend_alpha_hi: float = 0.7
    probe_seed: int = 0

    # --- adaptive per-window scale weighting ------------------------------
    orth_lo_pct: float = 25.0    # ramp knees are percentiles of the run's own
    orth_hi_pct: float = 75.0    # residual-distance distribution
    orth_wmin: float = 0.0
    orth_wmax: float = 0.55

    # --- features ---------------------------------------------------------
    welch_seg: int = 200
    welch_hop: int = 100

    # --- deep channel -----------------------------------------------------
    deep_embed_dim: int = 32
    deep_epochs: int = 20
    deep_batch_size: int = 128
    deep_lr: float = 1e-3
    deep_temperature: float = 0.2

    # --- evaluation -------------------------------------------------------
    n_boot: int = 300
    min_n: int = 10              # minimum positives to report a target (n is always printed)

    seed: int = 0

    def to_dict(self):
        return asdict(self)


DEFAULT = Config()

# The channels of the method. Every run scores all of them.
CHANNELS = ("scale", "shape", "spectral", "disc", "deep")
