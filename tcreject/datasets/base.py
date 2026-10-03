"""Dataset container shared by every adapter.

An adapter turns a raw recording into windows plus the information the method is
allowed to see (which windows are labelled, their class, their group and how
to fold them) and, separately, the ground truth used only for evaluation.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import numpy as np

from ..conformal import Fold


@dataclass
class Dataset:
    name: str
    W: np.ndarray                 # (N, T, C) raw windows
    fs: float                     # sampling rate in Hz
    label: np.ndarray             # (N,) label seen by the method (object array)
    kind: np.ndarray              # (N,) 'named' for labelled windows the method may fit on
    group: np.ndarray             # (N,) integer group id (subject, recording block, ...)
    named_vocab: List[str]        # sorted class names of the labelled windows
    folds: List[Fold]
    truth: np.ndarray             # (N,) ground truth for evaluation: 'named' or a novelty kind
    truth_label: np.ndarray       # (N,) finer ground-truth label for per-label targets
    targets: List[Tuple[str, np.ndarray]]  # evaluation targets: (name, positive mask over all windows)
    meta: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, np.ndarray] = field(default_factory=dict)  # e.g. raw_kind / raw_label

    def y(self):
        """Integer class index for labelled windows, -1 elsewhere."""
        idx = {c: i for i, c in enumerate(self.named_vocab)}
        out = np.full(len(self.label), -1, dtype=int)
        named = self.kind == "named"
        out[named] = [idx[l] for l in self.label[named]]
        return out
