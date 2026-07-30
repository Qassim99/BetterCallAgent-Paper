# Experiments

This folder keeps provenance adapters and exploratory work outside the default online
and offline pipelines.

- `historical_validation/` reconstructs the checksum-bound July 2026 validation
  snapshot, including its incomplete verifier batch.
- `model_comparison/` runs fresh, fingerprinted verifier models on the exact
  100-candidate validation input and evaluates complete score sets through the
  existing citation gate.
- `reranker_finetuning/` documents why audited training prototypes are not shipped as
  part of the reported pipeline.

Every new experiment should have:

- a dedicated README and an explicit research question;
- immutable data, model, prompt, and source identities;
- fixed seeds where applicable;
- a train/validation/test protocol defined before threshold selection;
- a unique output directory;
- complete failure and missing-data reporting; and
- an explicit statement of whether it changes a reported result.

Generated datasets, checkpoints, logs, predictions, and provider responses are
external artifacts. Their checksums and redistribution status must be recorded before
making a metric claim.
