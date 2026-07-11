# External Regression Manifests

Place SRBench, SRSD, PMLB, or other numeric regression arrays in local `.npz`
files and commit only their versioned manifests and frozen split files when data
licensing permits. The research runner never downloads data or invents a split.

Example manifest:

```json
{
  "version": 1,
  "dataset_id": "suite/task-name",
  "data": "task-name.npz",
  "feature_key": "X",
  "target_key": "y",
  "split_file": "task-name.splits.json",
  "sha256": "<64-character dataset hash>",
  "metadata": {"family": "symbolic_regression"}
}
```

The split file contains non-overlapping `train`, `validation`, and `test` index
lists. Test labels remain behind the sealed evaluation protocol.
