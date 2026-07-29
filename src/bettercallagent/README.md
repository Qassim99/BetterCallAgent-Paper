# Shared Python core

`bettercallagent` contains the framework-independent code reused by online
demonstrations and offline evaluation. Importing it never reads environment
variables, opens files, or contacts a model endpoint.

## Stable APIs

```python
from bettercallagent.citations import (
    CitationExtractor,
    CsvCitationVocabulary,
    FixedVotePolicy,
    select_citations,
    validate_answer_citations,
)
from bettercallagent.evaluation.metrics import macro_f1
from bettercallagent.retrieval import build_query_views, weighted_rrf
```

- `CitationExtractor` reproduces the audited law aliases, bare-article lists,
  paragraph ranges, BGE/ATF references, and docket considerations.
- `CsvCitationVocabulary` loads an exact closed vocabulary from explicit law
  and court CSV paths.
- `FixedVotePolicy` is the single configured citation gate. It has no
  keep-one fallback.
- `validate_answer_citations` fails if generated text cites an identifier that
  is outside the vocabulary or was not accepted by the gate.
- `weighted_rrf` rejects missing weights and uses stable lexical tie-breaking.
- `macro_f1` computes query-level Macro-F1 over the gold query set.

## Configuration boundary

`OnlineSettings.from_environment(mapping)` is the only environment loader. It
requires the caller to supply a mapping, validates CORS and model allowlists,
and requires an HTTPS model endpoint in live mode. Secrets are represented only
as runtime values and are excluded from dataclass representations.
