# Data

No competition input, case-law text, model output, or gold answer is committed to
this repository. Tiny fixtures are invented and are located in their component
folders.

## Authorized acquisition

The evaluation CSV files and closed citation vocabularies are distributed through
the Kaggle competition
[LLM Agentic Legal Information Retrieval](https://www.kaggle.com/competitions/llm-agentic-legal-information-retrieval).
Reviewers must accept the applicable terms before downloading them:

```bash
mkdir -p artifacts/downloads/kaggle
kaggle competitions download \
  -c llm-agentic-legal-information-retrieval \
  -p artifacts/downloads/kaggle
```

Do not assume the archive's internal layout matches the release contract. Place each
authorized file at its path under `artifacts/downloads` from
`artifacts/manifest.toml`, then verify:

```bash
uv run --locked python scripts/verify_artifacts.py \
  --manifest artifacts/manifest.toml \
  --root artifacts/downloads
```

The 111-file case-law snapshot is derived from `voilaj/swiss-caselaw`. Its upstream
dataset can change, so reproduction requires the committed tree checksum, not merely
the dataset name.

## Derived artifacts

The dense index, dense candidates, complete reranker input, incomplete historical
scores, balanced sparse traces, reference submissions, and audits are separate
hash-bound artifacts. No dense-index builder or sparse-evidence generator is included
in this release.

The authors' external artifact set currently has no DOI or stable download URL. This
release provides source and a checksum manifest; full reproduction is blocked until
an authorized archive is published. Do not redistribute competition or case-law
files without confirming their terms.

## Fixtures

Synthetic fixtures contain no competition answers. Fixture mode is always explicit;
the software never silently substitutes a fixture for a missing real artifact.
