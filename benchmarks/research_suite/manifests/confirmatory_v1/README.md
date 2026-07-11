# Frozen Confirmatory Real-Regression Suite v1

This directory freezes the twelve real-world tasks selected by
`../confirmatory_registry_v1.json`. Selection used the registry's first four
eligible candidates in each size stratum and did not inspect the performance of
EvoForest or any baseline. `selection_audit.json` records the audit result and
source-specific decisions. No candidate was replaced.

## Tasks

| Priority | Task | Rows | Numeric features after encoding | Split |
| ---: | --- | ---: | ---: | --- |
| 1 | Yacht Hydrodynamics | 308 | 6 | duplicate-group random |
| 2 | Real Estate Valuation | 414 | 6 | duplicate-group random |
| 3 | Forest Fires | 517 | 30 | duplicate-group random |
| 4 | QSAR Fish Toxicity | 908 | 6 | duplicate-group random |
| 5 | Wine Quality — White | 4,898 | 11 | duplicate-group random |
| 6 | Parkinsons Total UPDRS | 5,875 | 19 | subject-group random |
| 7 | Abalone | 4,177 | 11 | duplicate-group random |
| 8 | Combined Cycle Power Plant | 9,568 | 4 | duplicate-group random |
| 9 | Bike Sharing — Hourly | 17,379 | 64 | chronological |
| 10 | Appliances Energy | 19,735 | 25 | chronological |
| 11 | Superconductivity | 21,263 | 81 | duplicate-group random |
| 12 | Protein Tertiary Structure | 45,730 | 9 | duplicate-group random |

## Frozen protocol

- IID tasks use a 60/20/20 target allocation with exact duplicate feature
  vectors kept together and `numpy.random.Generator(PCG64)` seed `20260710`.
- Parkinsons observations are grouped by subject before the seeded split.
- Bike Sharing and Appliances Energy are sorted by their source timestamps and
  split chronologically 60/20/20.
- Categorical one-hot vocabularies are learned from the frozen training
  partition only and include an explicit unknown-category column.
- Agent-facing names are blinded to `x0`, `x1`, and so on. Semantic and encoded
  names remain in provenance metadata but must not be supplied to feature-
  generating agents in the blinded confirmatory condition.
- No scaling, target transformation, target-aware filtering, or outlier removal
  is baked into the arrays. Any model-specific normalization or imputation must
  be fitted using its training boundary only.
- Validation guides method and graph selection. Test labels remain sealed until
  all choices are complete.

## Leakage-related exclusions

- Real Estate: transaction row identifier `No`.
- Energy Efficiency is not part of this suite because it was used in the pilot.
- Parkinsons: subject identifier and alternate `motor_UPDRS` response.
- Bike Sharing: row/date identifiers and `casual`/`registered`, whose sum is the
  target. A numeric day-of-month is derived before removing the date; together
  with the source year, month, and hour fields, this prevents identical
  predictor vectors from crossing chronological partitions.
- Appliances Energy: timestamp plus synthetic duplicate random variables `rv1`
  and `rv2`.
- Wine Quality: the current UCI CSV combines red and white records; the registry
  preregistered white wine, so only rows with `color == "white"` are retained
  and the now-constant color field is removed.

## Provenance, license, and integrity

All data came from official UCI Machine Learning Repository endpoints. The
legacy Yacht, QSAR, and Protein tasks came from the official static ZIP archive;
the other tasks came from official `data.csv` endpoints. Every manifest records
the UCI DOI, repository and download URLs, source update and retrieval dates,
download and archive-member SHA-256 hashes, transformation details, split hash,
and derived NPZ hash.

At retrieval, the UCI pages displayed [CC BY
4.0](https://creativecommons.org/licenses/by/4.0/) for these datasets. Preserve
the per-dataset attribution and DOI in downstream publications and releases.

`index.json` is the collection inventory. The standard external-dataset loader
verifies each NPZ checksum and all frozen partition constraints.
