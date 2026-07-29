# Refactoring provenance

The paper release was reconstructed from two audited source lineages:

1. predecessor repository commit
   `9ae83a3ffc8d9ddf6247bf889d741dded0edcc6f`; and
2. the July 2026 research artifact tree containing the online interface, dense
   retrieval experiment, reranker outputs, sparse evidence traces, and diagnostic
   results.

Behavior was retained only after comparing source, saved outputs, and documentation.
Direct differential checks found:

- zero differences across all ten saved five-view query transformations; and
- byte-identical ordering, scores, and metadata across all 10,000 dense candidates
  when retaining the historical fresh-model-per-view lifecycle (artifact SHA-256
  `8c4d5c617fd9c1f04981db5a4765d254a0b0055b759fd432d705e4f1cd6be5b8`); and
- zero citation-extraction differences across 3,577 occurrences in the first 100
  audited reranker documents.

The release refactor:

- separates the online six-stage trace from the offline seven-stage experiment;
- shares one exact citation extractor, fusion implementation, and evidence gate;
- removes hard-coded citations and undocumented keep-one fallbacks;
- confines gold labels to offline evaluation and diagnostics;
- binds replay scores to immutable inputs rather than rank alone;
- exposes the historical missing verifier batch instead of concealing it;
- removes duplicate interfaces, generated environments, logs, caches, operational
  tooling, and machine-specific paths; and
- names custom metrics and surrogate diagnostics according to their actual claims.

The predecessor source tree is deliberately not copied into the reviewer-facing
release. The clean publication history starts from this audited artifact. Potentially
live credentials from predecessor development history must be rotated before
publication; no predecessor Git objects belong in the release repository.

The historical results remain incomplete and exploratory. Refactoring does not turn
them into held-out or fully reproduced evidence.
