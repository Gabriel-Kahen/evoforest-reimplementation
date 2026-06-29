from __future__ import annotations

from pathlib import Path

from evoforest_arch.evaluator import RidgeEvaluator
from evoforest_arch.evolution import EvolutionLoop
from evoforest_arch.seed import build_structural_break_seed_graph
from evoforest_arch.synthetic import make_structural_break_data


def main() -> None:
    dataset = make_structural_break_data(seed=7)
    graph = build_structural_break_seed_graph()
    loop = EvolutionLoop(graph, evaluator=RidgeEvaluator(n_splits=3, seed=7), seed=7)
    result = loop.run(dataset.inputs(), dataset.y, steps=8, output_dir=Path("runs/example"))
    print(f"best_score={result.score:.6f}")


if __name__ == "__main__":
    main()
