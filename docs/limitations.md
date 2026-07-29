# Limitations

## Experimental evidence

- The fixed-vote result (`0.4709962618` Macro-F1) and saved sparse-support result
  (`0.4806246255438345`) use only ten validation queries.
- The verifier artifact is incomplete: 19 of 20 batches completed, 95 of 100
  candidates have scores, and `val_002` ranks 6–10 are missing.
- Gate and support thresholds were selected on that same small validation split.
  Reported differences are exploratory and may not generalize.
- No held-out result is claimed for this variant.
- The support improvement depends on saved balanced sparse traces. A later,
  simplified candidate-keyword regeneration did not recover the same gain.
- A fresh strict 100-score run is a new experiment, not the historical result.

## Reproduction boundary

- The external artifact archive currently has no DOI or stable URL. Until the authors
  publish an authorized archive, this is a source-plus-manifest release.
- No dense-index builder is included. The exact index must be obtained and verified
  by tree hash.
- The predecessor index manifest omitted embedding model and revision metadata. The
  release compensates by binding its tree hash to a hashed configuration containing
  the exact model revision.
- Cached verifier scores are valid only for the exact candidate identities, order,
  prompt, batch composition, model, and input hashes.
- Hosted endpoints can change behavior without changing a model label.

## Interpretation and use

- The closed citation vocabulary and evaluation labels inherit the competition
  dataset's coverage and annotation limitations.
- Retrieval support does not establish that a citation is legally correct.
- Project-specific RAGAS-style metrics, linear surrogates, and perturbation proxies
  must not be presented as official-library metrics, model-internal explanations, or
  causal attributions.
- Case-law and query text may contain personal or confidential information. Provider
  use requires a separate privacy and legal assessment.
- BetterCallAgent is research software and does not provide legal advice.
