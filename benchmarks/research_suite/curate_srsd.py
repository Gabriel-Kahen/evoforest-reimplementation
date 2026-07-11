from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SELECTED_EASY_TASKS = (
    "feynman-ii.8.31",
    "feynman-i.12.1",
    "feynman-i.18.12",
    "feynman-i.27.6",
    "feynman-i.18.16",
    "feynman-ii.2.42",
)


def curate(source: Path, output: Path) -> list[Path]:
    supplement = json.loads((source / "supp_info.json").read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=True)
    manifests: list[Path] = []
    index: list[dict[str, object]] = []
    for task in SELECTED_EASY_TASKS:
        arrays = [np.loadtxt(source / split / f"{task}.txt") for split in ("train", "val", "test")]
        widths = {array.shape[1] for array in arrays}
        if len(widths) != 1:
            raise ValueError(f"SRSD split width mismatch for {task}.")
        X = np.vstack([array[:, :-1] for array in arrays])
        y = np.concatenate([array[:, -1] for array in arrays])
        counts = [array.shape[0] for array in arrays]
        starts = np.cumsum([0, *counts])
        splits = {
            name: list(range(int(starts[index]), int(starts[index + 1])))
            for index, name in enumerate(("train", "validation", "test"))
        }
        stem = task.replace(".", "_")
        data_path = output / f"{stem}.npz"
        split_path = output / f"{stem}.splits.json"
        manifest_path = output / f"{stem}.manifest.json"
        np.savez_compressed(data_path, X=X, y=y)
        split_path.write_text(json.dumps({"splits": splits}, indent=2) + "\n", encoding="utf-8")
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
        info = supplement[task]
        manifest = {
            "version": 1,
            "dataset_id": f"srsd-easy/{task}",
            "data": data_path.name,
            "feature_key": "X",
            "target_key": "y",
            "feature_names": [f"x{index}" for index in range(X.shape[1])],
            "split_file": split_path.name,
            "sha256": digest,
            "metadata": {
                "family": "srsd_feynman_easy",
                "difficulty": "easy",
                "true_equation": info["sympy_eq_str"],
                "source": "https://huggingface.co/datasets/yoshitomo-matsubara/srsd-feynman_easy",
                "paper": "https://arxiv.org/abs/2206.10540",
                "license": "CC BY 4.0",
                "selection_rule": "predeclared coverage of one through five variables plus a rational three-variable task",
                "original_split_preserved": True,
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifests.append(manifest_path)
        index.append({"dataset_id": manifest["dataset_id"], "manifest": manifest_path.name, "rows": len(y), "features": X.shape[1]})
    (output / "index.json").write_text(json.dumps({"version": 1, "datasets": index}, indent=2) + "\n", encoding="utf-8")
    return manifests


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the preregistered six-task SRSD easy pilot subset.")
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    curate(args.source, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
