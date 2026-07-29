# Third-party notices and redistribution boundaries

BetterCallAgent integrates with third-party data, models, APIs, and libraries. Naming
them does not imply endorsement.

## Data and derived artifacts

- The Swiss legal evaluation files and closed citation vocabularies are distributed
  through the Kaggle competition **LLM Agentic Legal Information Retrieval**. They are
  not committed. Users must accept and follow the competition terms.
- The case-law corpus snapshot is derived from `voilaj/swiss-caselaw`. Its verified
  snapshot is identified by a tree checksum because the upstream dataset may change.
- The dense index, saved verifier inputs and outputs, sparse evidence traces, and
  reference submissions are derived artifacts with separate size and redistribution
  constraints.
- `artifacts/manifest.toml` records checksums without redistributing these files.
  The external artifact set currently has no DOI or stable URL.

The authors must confirm the redistribution rights for every external and derived
artifact before publishing an archive.

## Models and services

- Qwen, DeepSeek, and Soofi models remain subject to their model cards, licenses,
  acceptable-use policies, and provider terms.
- An OpenAI-compatible protocol does not imply that a service is operated by OpenAI.
- Queries and selected document excerpts sent to a remote endpoint leave the local
  machine. Review the provider's privacy, retention, location, and processing terms.

## Dependencies

Python and JavaScript dependencies retain their own copyrights and licenses. Exact
resolved versions are recorded in `uv.lock` and `frontend/package-lock.json`.

No project-wide open-source license has been approved. See `LICENSE-PENDING.md`.
