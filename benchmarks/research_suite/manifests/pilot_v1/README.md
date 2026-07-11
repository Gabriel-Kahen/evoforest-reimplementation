# Frozen External Regression Pilot v1

This directory contains three small real-world regression datasets for the
engineering pilot. It is frozen as of 2026-07-10. These tasks are for debugging
runtime, failure handling, metric collection, and budget choices; they must not
be reported as untouched confirmatory evidence after pilot results have been
inspected.

## Included tasks

| Dataset | Rows | Features | Target | Train / validation / test |
| --- | ---: | ---: | --- | --- |
| UCI Energy Efficiency | 768 | 8 | Heating load (`Y1`) | 461 / 153 / 154 |
| UCI Airfoil Self-Noise | 1,503 | 5 | Scaled sound pressure | 902 / 300 / 301 |
| UCI Concrete Compressive Strength | 1,030 | 8 | Compressive strength | 618 / 206 / 206 |

All feature names exposed to agents are blinded (`x0`, `x1`, ...). Original
column names are retained only as provenance metadata. For Energy Efficiency,
the cooling-load response is excluded rather than used as a predictor.

## Frozen split policy

- A `numpy.random.Generator(PCG64)` seeded with `20260710` shuffled groups.
- Rows with exactly identical complete feature vectors were kept in the same
  partition to prevent duplicate leakage.
- Groups were assigned toward 60% training, 20% validation, and 20% sealed test
  partitions. The concrete grouping makes its final counts differ slightly
  from exact percentages.
- No scaling, imputation, filtering, or target-aware preprocessing was applied.
- The test partitions are for one final evaluation after graph and baseline
  choices are frozen.

## Provenance and license

The numeric arrays were derived from the UCI Machine Learning Repository's
official `data.csv` endpoints. Each manifest records the dataset DOI, source
URL, retrieval date, source CSV SHA-256, derived NPZ SHA-256, split-file
SHA-256, attribution, preprocessing, and original schema.

At retrieval, each UCI dataset page identified its dataset as licensed under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Redistribution of
the derived numeric arrays therefore requires preserving attribution. Cite the
individual DOI recorded in each manifest and UCI in any published artifact.

## Integrity check

`index.json` is the collection-level inventory. The existing
`load_regression_dataset` loader checks each NPZ hash and validates that the
three frozen partitions are nonempty, disjoint, and in range.
