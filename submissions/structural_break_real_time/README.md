# Structural Break Real-Time Submission

This folder adapts the best current EvoForest row graph idea for the CrunchDAO
ADIA Lab Structural Break Challenge: Real-Time Edition.

The upload artifact is:

```text
submissions/structural_break_real_time/evoforest_realtime_submission.ipynb
```

The notebook is self-contained. It defines the two functions expected by the
Crunch runner:

- `train(datasets, model_directory_path)`
- `infer(datasets, model_directory_path)`

The implementation trains a ridge readout over streaming row features inspired
by the committed `row_baseline_time_tail_graph`: historical-reference features,
expanded time/observed-position features, and multiscale recent-tail features.
At inference time it emits exactly one score per online observation.

## Local Test

After running your authenticated Crunch setup command:

```bash
crunch setup-notebook structural-break-real-time <your-token>
```

open the notebook and run:

```python
import crunch
crunch_tools = crunch.load_notebook()
crunch_tools.test(no_determinism_check=True)
```

The setup token is user-specific, so it is intentionally not committed here.

## Caveat

This is not the same benchmark as the earlier row-parquet audit. The real-time
competition scores per-online-step predictions with time-stratified AUC, so this
submission retrains the readout from the competition's streaming tuples rather
than reusing the offline parquet model coefficients.
