# Models

## Historical validation snapshot

| Role | Exact identifier | Evidence scope |
|---|---|---|
| Embedding | `Qwen/Qwen3-Embedding-8B` at revision `1d8ad4ca9b3dd8059ad90a75d4983776a23d44af` | Five-view dense index and query encoding |
| Verifier reranker | `qwen3-30b-a3b-instruct-2507` | Provider-specific identifier in the incomplete ten-query validation run |

The verifier identifier is not a portable model revision. Reproducing it requires the
original provider behavior and the exact prompt/input fingerprints. A label match
alone is insufficient.

## Requested recommendations

These are candidates for new experiments, not models validated by the historical
snapshot:

| Model | Recommended role | Exact reference identifier | Availability note |
|---|---|---|---|
| DeepSeek V4 Flash | Latency-conscious local or hosted reranking | official local ID `deepseek-ai/DeepSeek-V4-Flash`; provider example `deepseek/deepseek-v4-flash` | Confirm the exact endpoint label and terms |
| Qwen3.6-27B | Open-weight local reranking | `Qwen/Qwen3.6-27B` | Pin a concrete model revision before running |
| Soofi S | German/English hosted deployment evaluation | OpenAI-compatible base URL `https://soofi-owu.l3s.de/api`; verified runtime model `Soofi-S-RLVR-Isar` | Authorized API access is required; the deployment model ID may change |

References:

- [DeepSeek V4 Flash model card](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash)
- [Qwen3.6-27B model card](https://huggingface.co/Qwen/Qwen3.6-27B)
- [Soofi](https://www.soofi.info/)

Provider identifiers, local checkpoint identifiers, and immutable revisions are
different fields. Record the one actually used. Never replay cached scores under a
new label. Compare alternatives only after rerunning the complete validation
protocol.

## SOOFI hosted experiment contract

SOOFI is configured as a new hosted experiment, not as evidence for the
historical validation snapshot. The selected service exposes the
OpenAI-compatible route
`https://soofi-owu.l3s.de/api/chat/completions`. Configure clients with the base
URL `https://soofi-owu.l3s.de/api` and let them append `/chat/completions`.

The verified runtime model is
`Soofi-S-RLVR-Isar`.
Treat this value as configurable rather than permanent. For every experiment,
store both the requested model ID and the `model` identifier returned in the
actual completion response. The returned identifier is the authoritative record
of the served deployment.

SOOFI requests for this configuration must include:

```json
{
  "chat_template_kwargs": {
    "enable_thinking": false
  }
}
```

Supply the API key only at runtime through the experiment runner's secret
environment variable. Never place it in this file, a command-line argument, a
committed `.env` file, or an experiment artifact.
