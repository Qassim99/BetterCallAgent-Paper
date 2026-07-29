# Interpretability and diagnostics

BetterCallAgent exposes operational traces and deterministic offline diagnostics. It
does not request, store, or claim hidden model chain-of-thought.

| Analysis | What it reports | What it does not claim |
|---|---|---|
| Citation error attribution | Retrieval misses, combined selection-or-gate misses, and false positives | It cannot separate selection from gating without a finer saved trace |
| RAGAS-style citation metrics | Explicit set formulas for context precision, gold recall, support, and answer F1 | It is not the official RAGAS package |
| Gate surrogate | Standardized logistic/additive contributions and fidelity to recorded gate decisions | Contributions are not SHAP values or verifier internals |
| Perturbation proxy | Seeded evidence-removal AOPC relative to random controls | It is not Integrated Gradients or causal attribution |

The command output records the quantities required by each analysis. For example, the
perturbation command records its seed and trial count, while the surrogate records
features, coefficients, intercept, contributions, and fidelity. Input artifacts and
command lines should be archived alongside any paper report; the tools do not embed
every source file into every output.

All gold-based diagnostics are offline-only. The online package does not import them.
See [the interpretability guide](../interpretability/README.md) for schemas, formulas,
copy-paste commands, and limitations.
