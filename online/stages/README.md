# Pipeline stages

Each numbered module implements one stage and exposes the same small coroutine:

```python
async def run(
    context: RunContext,
    dependencies: OnlineDependencies,
) -> dict[str, object]: ...
```

`RunContext` is request-local. `OnlineDependencies` contains only validated,
application-lifetime services. No stage reads environment variables or creates
its own provider.

| Module | Responsibility |
|---|---|
| `stage_01_understand.py` | strict legal-issue analysis |
| `stage_02_generate_queries.py` | five views with byte-exact artifact validation |
| `stage_03_retrieve.py` | versioned dense/BM25 ranking replay and weighted RRF |
| `stage_04_rerank.py` | bounded independent relevance scoring |
| `stage_05_control_citations.py` | configured fixed-vote citation gate |
| `stage_06_answer.py` | grounded synthesis and exact citation validation |

The orchestrator in `online/pipeline.py` is deliberately thin: it orders these
functions and emits the stable event contract used by the frontend.

Stage 3 does not contact a dynamic search index. The configured version-2 asset
contains the rankings and the exact five views used to create them. Stage 2
rejects any generated view drift before ranking replay begins.
