from __future__ import annotations

import ast
import numpy as np

from submissions.structural_break_real_time import submission


def test_realtime_submission_trains_and_streams_scores(tmp_path) -> None:
    train_data = make_realtime_series(n_series=24, seed=3)

    submission.train(train_data, str(tmp_path))

    test_items = [(historical, iter(online)) for _id, historical, online, _tau in train_data[:6]]
    generator = submission.infer(test_items, str(tmp_path))
    assert next(generator) is None
    scores = list(generator)
    expected_count = sum(len(row[2]) for row in train_data[:6])

    assert len(scores) == expected_count
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert np.std(scores) > 0.0


def test_realtime_submission_gets_signal_on_simple_mean_shift(tmp_path) -> None:
    train_data = make_realtime_series(n_series=40, seed=9)
    submission.train(train_data, str(tmp_path))

    test_items = [(historical, online) for _id, historical, online, _tau in train_data]
    generator = submission.infer(test_items, str(tmp_path))
    next(generator)
    predictions = list(generator)
    labels = []
    for _id, _historical, online, tau in train_data:
        labels.extend(float(tau is not None and idx >= tau) for idx in range(len(online)))

    assert auc(np.asarray(labels), np.asarray(predictions)) > 0.65


def test_realtime_submission_survives_crunch_export_without_top_level_constants(tmp_path) -> None:
    """Crunch's notebook converter preserves functions but can drop assignments."""

    source_path = submission.__file__
    assert source_path is not None
    source = open(source_path, encoding="utf-8").read()
    tree = ast.parse(source)
    dropped_names = {
        "MODEL_FILE",
        "DEFAULT_MAX_ROWS_PER_SERIES",
        "DEFAULT_SERIES_LENGTH",
        "DEFAULT_MAX_ONLINE_LENGTH",
        "ALPHAS",
        "TAIL_WINDOWS",
        "EPS",
    }
    tree.body = [
        node
        for node in tree.body
        if not (
            isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id in dropped_names for target in node.targets)
        )
    ]
    ast.fix_missing_locations(tree)
    namespace: dict[str, object] = {}
    exec(compile(tree, str(source_path), "exec"), namespace)

    train_data = make_realtime_series(n_series=12, seed=11)
    namespace["train"](train_data, str(tmp_path))  # type: ignore[index,operator]
    generator = namespace["infer"]([(row[1], row[2]) for row in train_data[:3]], str(tmp_path))  # type: ignore[index,operator]
    assert next(generator) is None
    scores = list(generator)

    assert len(scores) == sum(len(row[2]) for row in train_data[:3])
    assert all(0.0 <= score <= 1.0 for score in scores)


def make_realtime_series(*, n_series: int, seed: int) -> list[tuple[int, np.ndarray, np.ndarray, int | None]]:
    rng = np.random.default_rng(seed)
    rows = []
    for idx in range(n_series):
        historical = rng.normal(0.0, 1.0, size=160)
        online = rng.normal(0.0, 1.0, size=80)
        tau = None
        if idx % 2 == 0:
            tau = 28 + (idx % 9)
            online[tau:] += 2.0 + 0.1 * (idx % 5)
        rows.append((idx, historical, online, tau))
    return rows


def auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    positives = scores[y_true >= 0.5]
    negatives = scores[y_true < 0.5]
    if positives.size == 0 or negatives.size == 0:
        return 0.5
    wins = np.sum(positives.reshape(-1, 1) > negatives.reshape(1, -1))
    ties = np.sum(positives.reshape(-1, 1) == negatives.reshape(1, -1))
    return float((wins + 0.5 * ties) / float(positives.size * negatives.size))
