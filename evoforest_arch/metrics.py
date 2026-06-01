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
