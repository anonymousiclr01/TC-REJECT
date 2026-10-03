"""Fold constructors. A dataset picks one; the method itself is fold-agnostic.

* ``leave_one_group_out``      each group (e.g. subject) is held out in turn; the
                               fold scores every window of the held-out group and
                               fits on labelled windows of all other groups.
* ``group_kfold_over_labelled`` GroupKFold over the groups (e.g. recording blocks)
                               that contain labelled windows; a fold scores its
                               held-out groups plus every unlabelled window.
"""
import numpy as np
from sklearn.model_selection import GroupKFold

from .conformal import Fold


def leave_one_group_out(group, is_labelled):
    folds = []
    for g in np.unique(group):
        te = np.where(group == g)[0]
        tr = np.where(group != g)[0]
        folds.append(Fold(fit=tr[is_labelled[tr]], target=te, held_groups=(g,)))
    return folds


def group_kfold_over_labelled(group, is_labelled, n_splits=5):
    ni = np.where(is_labelled)[0]
    ng = np.unique(group[ni])
    n_eff = min(n_splits, len(ng))
    if n_eff < 2:
        return []
    folds = []
    gkf = GroupKFold(n_splits=n_eff)
    for tr, _ in gkf.split(ni, groups=group[ni]):
        fit = ni[tr]
        held = tuple(sorted(set(ng.tolist()) - set(np.unique(group[fit]).tolist())))
        target = np.where(~is_labelled | np.isin(group, list(held)))[0]
        folds.append(Fold(fit=fit, target=target, held_groups=held))
    return folds
