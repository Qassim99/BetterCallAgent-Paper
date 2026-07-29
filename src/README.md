# Python package map

Installable shared code lives under `src/bettercallagent`:

```text
bettercallagent/
├── citations/   Exact extraction, vocabulary checks, fixed policy, answer validation
├── evaluation/  Citation-set metrics used by offline evaluation
├── providers/   Explicit model-provider boundary
├── retrieval/   Five query views and weighted reciprocal-rank fusion
├── schemas.py   Shared immutable records
└── settings.py  Validated online configuration
```

The package is framework-independent. Importing it does not read files, inspect
environment variables, or contact a service. Component-specific orchestration remains
in `offline`, `online`, and `interpretability`.

See [bettercallagent/README.md](bettercallagent/README.md) for stable imports and
behavioral guarantees.
