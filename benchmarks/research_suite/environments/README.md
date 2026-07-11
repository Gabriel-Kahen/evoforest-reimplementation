# Isolated symbolic-regression baselines

## FEAT

FEAT is not the package named `feat` on PyPI; that package is an unrelated
Python 2 agent toolkit. The research baseline is built from the Cava Lab
repository at commit `2967e6e5f7eee75ecf34062708e7b0b87c0b9145`.

Create the isolated environment without changing the project virtualenv:

```bash
MAMBA_EXE=/absolute/path/to/micromamba \
FEAT_ENV_PREFIX=/absolute/path/to/feat-env \
bash benchmarks/research_suite/environments/setup_feat.sh
```

The setup finishes with a real one-generation fit/predict smoke test. Use it
through `FEATCommandAdapter` with the environment Python as the executable and
this runner as the first argument:

```python
FEATCommandAdapter(
    executable="/absolute/path/to/feat-env/bin/python",
    arguments=[
        "/absolute/path/to/feat_runner.py",
        "--train-csv", "{train_csv}",
        "--test-csv", "{test_csv}",
        "--predictions-csv", "{predictions_csv}",
        "--seed", "7",
        "--generations", "100",
        "--population", "100",
    ],
)
```

The `nlohmann_json=3.11.2` pin is intentional. A build using 3.12.0 succeeded
but failed at import with an undefined FEAT serialization symbol.

## Operon decision

`pyoperon` 0.5.0 through 0.6.1 require NumPy 2 or newer. The main research
environment currently uses NumPy 1.26.4 with AutoFeat, so Operon must not be
installed into that environment. This is an environment-isolation issue, not a
scientific reason to omit Operon. Keep it out of the core confirmatory matrix
until it has its own pinned command environment, then include it in the
symbolic-regression lane if compute permits. Never downgrade to `pyoperon`
0.4.0 merely to evade declared dependencies without separately validating its
older API and numerical behavior.
