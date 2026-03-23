# Data Statement

## Dataset

- Name: DuEE-Fin (financial event extraction)
- Location in this repository: `data/raw/DuEE-Fin/`
- Splits used by this project: `train`, `dev`, `test`

## Provenance

The dataset is used for research benchmarking of structured event extraction in Chinese financial text.
The repository stores data files under `data/raw/DuEE-Fin/` for local experimentation.

## Label Availability Caveat

In this repository snapshot, `duee_fin_test.json` typically does not include gold `event_list`.
Therefore, `evaluate_api.py --split test` is prediction-only and does not report F1.

## Licensing and Access Conditions

This repository does not redefine or override upstream DuEE-Fin licensing terms.
Users are responsible for complying with the original dataset provider's terms of use and redistribution constraints.

## Data Risks

- Financial text may contain stale or entity-sensitive information.
- LLM-based extraction can hallucinate unsupported entities/arguments.
- Evaluation includes hallucination-rate diagnostics, but manual review is still required for high-stakes usage.
