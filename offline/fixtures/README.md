# Synthetic offline fixture

These tiny records are invented for software testing. They are not legal advice,
benchmark examples, or evidence for any paper result.

The fixture deliberately uses saved per-view rankings, pre-materialized documents,
saved sparse traces, and fingerprint-bound verifier scores. It therefore tests
fusion, joins, citation extraction, cache alignment, evidence gating, evaluation,
and run provenance without downloading a model or contacting a provider.

Run it from the repository root:

```bash
BCA_FIXTURE_OUT="$(mktemp -d)/bettercallagent-fixture"
uv run --locked python -m offline.run \
  --config offline/fixtures/config.toml \
  --output "$BCA_FIXTURE_OUT"
```

Full GPU retrieval and live reranking use the same stage implementations but must
be configured explicitly; see [`../README.md`](../README.md).
