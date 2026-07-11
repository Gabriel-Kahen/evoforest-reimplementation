"""Isolated FEAT command boundary used by the research baseline adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from feat import FeatRegressor


def _fit_predict(
    train_path: Path,
    test_path: Path,
    output_path: Path,
    *,
    seed: int,
    generations: int,
    population: int,
) -> dict[str, object]:
    train = np.loadtxt(train_path, delimiter=",", ndmin=2)
    test = np.loadtxt(test_path, delimiter=",", ndmin=2)
    if train.shape[1] != test.shape[1] + 1:
        raise ValueError("Training CSV must contain input columns followed by one target column.")
    model = FeatRegressor(
        pop_size=population,
        gens=generations,
        n_jobs=1,
        random_state=seed,
        verbosity=0,
    )
    model.fit(train[:, :-1], train[:, -1])
    predictions = np.asarray(model.predict(test), dtype=float).reshape(-1)
    if predictions.shape != (test.shape[0],) or not np.all(np.isfinite(predictions)):
        raise RuntimeError("FEAT returned invalid predictions.")
    np.savetxt(output_path, predictions, delimiter=",")
    return {"rows": len(predictions), "representation": model.get_representation()}


def _self_test() -> dict[str, object]:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(96, 4))
    y = np.sin(x[:, 0] * x[:, 1]) + 0.25 * x[:, 2]
    model = FeatRegressor(pop_size=12, gens=1, n_jobs=1, random_state=7, verbosity=0)
    model.fit(x[:72], y[:72])
    prediction = np.asarray(model.predict(x[72:]), dtype=float)
    return {
        "rows": len(prediction),
        "finite": bool(np.isfinite(prediction).all()),
        "rmse": float(np.sqrt(np.mean((prediction - y[72:]) ** 2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path)
    parser.add_argument("--test-csv", type=Path)
    parser.add_argument("--predictions-csv", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--generations", type=int, default=100)
    parser.add_argument("--population", type=int, default=100)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        result = _self_test()
    else:
        if args.train_csv is None or args.test_csv is None or args.predictions_csv is None:
            parser.error("train, test, and predictions paths are required")
        result = _fit_predict(
            args.train_csv,
            args.test_csv,
            args.predictions_csv,
            seed=args.seed,
            generations=args.generations,
            population=args.population,
        )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
