# Ethics and Limitations

## Intended Use

This project is intended for academic research on event extraction and preference optimization.
It is not intended as a sole decision system for compliance, legal, or investment decisions.

## Known Risks

- Hallucinated arguments: LLM outputs may include entities not supported by source text.
- Extraction bias: prompt and schema design may favor frequent event types.
- Domain drift: model behavior can degrade on out-of-domain or temporally shifted news.

## Mitigations Implemented

- Hallucination diagnostics in evaluation (`evaluate.py`, `evaluate_api.py`)
- Schema compliance checks for event types and argument structures
- Parse diagnostics and explicit parse failure accounting
- Multi-seed reproducibility suite with confidence intervals and significance tests

## Residual Limitations

- API model aliases can evolve over time; the project records returned `response_model`, but upstream behavior may still change.
- Automatic metrics do not replace manual error analysis for deployment scenarios.
- Dataset licensing and governance are inherited from upstream DuEE-Fin release policies.

## Resource and Environmental Reporting

Evaluation summaries include runtime wall-clock, token usage, and environment manifest.
These fields should be reported in papers for transparency.
