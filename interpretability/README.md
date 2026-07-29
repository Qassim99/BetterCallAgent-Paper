# BetterCallAgent interpretability

This folder contains small, deterministic analyses for saved citation-pipeline
artifacts. It is intentionally offline: the code makes no network calls, loads
no model, and never runs retrieval. Gold citations are accepted only by these
research-time analyses and must not be imported into the online pipeline.

The implementation uses only the Python standard library. Python 3.10 or newer
is sufficient.

## What is included

| Command | Question answered | Output |
|---|---|---|
| `attribute-errors` | At which observable stage did citation errors appear? | Retrieval misses, combined selector/gate misses, and supported or unsupported false positives |
| `ragas-style` | How precise, complete, supported, and correct are citation sets? | Transparent citation-set metrics named `ragas_style` |
| `gate-surrogate` | Which recorded gate features drive a simple fitted approximation? | Logistic/additive log-odds contributions and surrogate fidelity |
| `faithfulness-proxy` | Do evidence removals reduce a supplied black-box score more than controls? | Targeted AOPC, seeded random baseline, and their gap |
| `render-report` | How can the compact results be reviewed together? | One JSON file and one Markdown file |

`ragas_style` is a local metric family defined below. It is not the official
RAGAS library and its values should not be compared to official-library results
without first matching the formulas.

## Run the synthetic example

Run these commands from the repository root. Outputs go to a new temporary
directory rather than into the source tree.

```bash
BCA_INTERP_OUT="$(mktemp -d)"

python -m interpretability attribute-errors \
  --queries interpretability/examples/queries.jsonl \
  --retrieval interpretability/examples/retrieval.jsonl \
  --predictions interpretability/examples/predictions.jsonl \
  --output "$BCA_INTERP_OUT/error_attribution.json"

python -m interpretability ragas-style \
  --queries interpretability/examples/queries.jsonl \
  --retrieval interpretability/examples/retrieval.jsonl \
  --predictions interpretability/examples/predictions.jsonl \
  --context-k 10 \
  --output "$BCA_INTERP_OUT/ragas_style.json"

python -m interpretability gate-surrogate \
  --input interpretability/examples/gate_candidates.jsonl \
  --output "$BCA_INTERP_OUT/gate_surrogate.json"

python -m interpretability faithfulness-proxy \
  --input interpretability/examples/perturbations.jsonl \
  --seed 17 \
  --random-trials 1000 \
  --output "$BCA_INTERP_OUT/faithfulness_proxy.json"

python -m interpretability render-report \
  --input "$BCA_INTERP_OUT/error_attribution.json" \
  --input "$BCA_INTERP_OUT/ragas_style.json" \
  --input "$BCA_INTERP_OUT/gate_surrogate.json" \
  --input "$BCA_INTERP_OUT/faithfulness_proxy.json" \
  --json-output "$BCA_INTERP_OUT/report.json" \
  --markdown-output "$BCA_INTERP_OUT/report.md"

cat "$BCA_INTERP_OUT/report.md"
```

The example citation strings and scores are synthetic. They test the schemas
and formulas; they are not Swiss-law findings or paper results.

## Input schemas

All inputs are UTF-8 JSONL: one JSON object per line. Query identifiers must be
unique. For citation analyses, the three files must contain exactly the same
query identifiers. Citation lists must contain unique, canonical strings.
Invalid or inconsistent artifacts cause a clear error.

### Query and gold artifact

```json
{"query_id":"val_001","query":"Question text","gold_citations":["Art. 1 ZGB"]}
```

### Retrieval artifact

The list order is the retrieval rank.

```json
{"query_id":"val_001","retrieved_citations":["Art. 1 ZGB","Art. 2 ZGB"]}
```

### Prediction artifact

```json
{"query_id":"val_001","predicted_citations":["Art. 1 ZGB"]}
```

### Gate-candidate artifact

Every row must use the same observable numeric feature names. `accepted` is the
recorded gate decision, not a gold relevance label.

```json
{"query_id":"val_001","citation":"Art. 1 ZGB","accepted":true,"features":{"best_score":9.1,"bm25_support":2,"top3_anchor":1,"vote_count":5}}
```

### Perturbation artifact

`full_score` and `score_without_feature` are caller-supplied black-box scores.
The analysis does not invoke a verifier. Each query needs at least one evidence
ablation and at least as many non-evidence controls as evidence features.

```json
{"query_id":"val_001","full_score":0.9,"ablations":[{"feature":"evidence_doc_1","score_without_feature":0.5,"is_evidence":true},{"feature":"unrelated_doc_1","score_without_feature":0.89,"is_evidence":false}]}
```

## Metric definitions

For a gold citation set \(G\), the first \(k\) retrieved citations \(C_k\), and
predictions \(P\):

- `context_precision_at_k` is \(|G ∩ C_k| / |C_k|\).
- `gold_context_recall_at_k` is \(|G ∩ C_k| / |G|\).
- `citation_faithfulness` is \(|P ∩ C_k| / |P|\). It measures retrieval
  support, not whether a citation is legally correct.
- `answer_f1` is citation-set F1 between \(P\) and \(G\).

The error attribution is deliberately conservative. A gold citation absent
from retrieval is a retrieval miss. A retrieved gold citation absent from the
prediction is a combined `selection_or_gate_miss`; without a saved finer trace,
the code does not guess which of those two stages removed it.

The gate explanation fits a deterministic standardized logistic regression.
For each row it reports coefficient-times-feature contributions whose sum plus
the intercept equals the surrogate log-odds. Fidelity measures agreement with
recorded gate decisions. These terms describe the fitted linear surrogate, not
model internals or causal effects.

The perturbation proxy computes the mean score decrease after evidence-feature
removal and compares it with equally sized samples of non-evidence removals.
The random baseline is reproducible from `--seed`. This proxy uses supplied
black-box scores; it does not access verifier logits or gradients and does not
establish causal faithfulness.

## Exploratory validation results

The incomplete historical artifact set from the ten-query validation split reported:

| Configuration | Macro citation F1 |
|---|---:|
| Dense Qwen candidates with the fixed-vote gate | 0.470996 |
| Same pipeline plus saved SIRA/BM25 citation-support traces | 0.480625 |

The associated custom RAGAS-style analysis reported context precision
**0.7273**, gold context recall **0.4996**, and citation faithfulness **0.8486**.

These are exploratory results on **n=10**, not held-out test estimates. Only 19 of
20 verifier batches completed: 95 of 100 candidates have scores, with `val_002`
ranks 6–10 missing. Both thresholds were selected on the same split. The 0.480625
result depends on previously saved balanced sparse traces; a later
candidate-keyword regeneration did not reproduce the same improvement.
Consequently, the values must not be presented as generalization evidence or as
outputs of the synthetic example. A paper should publish the exact input artifact
hashes and configuration beside any regenerated report.

## Testing

```bash
python -m unittest discover -s tests/interpretability -v
```

Tests cover citation formulas, strict stage attribution, deterministic
surrogate contributions, seeded perturbation baselines, and the complete CLI
workflow.

## Scope and limitations

- Analyses depend on the completeness and alignment of the saved artifacts.
- Gold-based diagnostics are suitable for offline validation only.
- Retrieval support is not equivalent to substantive legal correctness.
- A linear surrogate can faithfully approximate a simple gate while still
  omitting interactions.
- Perturbation scores depend on the caller's scoring function and ablation
  design.
- No confidence interval from ten queries should be interpreted as evidence of
  broad reliability.
