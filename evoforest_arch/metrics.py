from __future__ import annotations

import numpy as np


def roc_auc_score(y_true: np.ndarray, score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    s = np.asarray(score, dtype=np.float64)
    pos = y == 1
    neg = y == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = s[order]
    i = 0
    while i < len(s):
        j = i + 1
        while j < len(s) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    rank_sum_pos = float(np.sum(ranks[pos]))
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg)


def stratified_folds(y: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    folds: list[list[int]] = [[] for _ in range(n_splits)]
    for label in sorted(np.unique(y)):
        idx = np.flatnonzero(y == label)
        rng.shuffle(idx)
        for fold_id, part in enumerate(np.array_split(idx, n_splits)):
            folds[fold_id].extend(part.tolist())
    all_idx = np.arange(y.shape[0])
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for fold in folds:
        val = np.asarray(sorted(fold), dtype=np.int64)
        mask = np.ones(y.shape[0], dtype=bool)
        mask[val] = False
        out.append((all_idx[mask], val))
    return out


def stratified_group_folds(y: np.ndarray, groups: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    group_array = np.asarray(groups)
    if y.ndim != 1:
        raise ValueError("y must be a 1-D array.")
    if group_array.ndim != 1 or group_array.shape[0] != y.shape[0]:
        raise ValueError("groups must be a 1-D array with the same length as y.")
    unique_groups, inverse = np.unique(group_array, return_inverse=True)
    effective_splits = max(2, min(int(n_splits), int(unique_groups.shape[0])))
    group_labels = np.zeros(unique_groups.shape[0], dtype=np.float64)
    for group_index in range(unique_groups.shape[0]):
        group_labels[group_index] = float(np.max(y[inverse == group_index]))

    fold_group_indices: list[list[int]] = [[] for _ in range(effective_splits)]
    labels, counts = np.unique(group_labels, return_counts=True)
    if labels.shape[0] > 1 and np.all(counts >= effective_splits):
        for label in sorted(labels):
            group_idx = np.flatnonzero(group_labels == label)
            rng.shuffle(group_idx)
            for fold_id, part in enumerate(np.array_split(group_idx, effective_splits)):
                fold_group_indices[fold_id].extend(part.tolist())
    else:
        shuffled = np.arange(unique_groups.shape[0])
        rng.shuffle(shuffled)
        for fold_id, part in enumerate(np.array_split(shuffled, effective_splits)):
            fold_group_indices[fold_id].extend(part.tolist())

    all_idx = np.arange(y.shape[0])
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for group_idx in fold_group_indices:
        if not group_idx:
            continue
        val_groups = unique_groups[np.asarray(group_idx, dtype=np.int64)]
        val_mask = np.isin(group_array, val_groups)
        val = np.flatnonzero(val_mask)
        train = all_idx[~val_mask]
        if train.size and val.size:
            folds.append((train, val))
    if len(folds) < 2:
        return stratified_folds(y, min(max(2, int(n_splits)), y.shape[0]), seed)
    return folds
