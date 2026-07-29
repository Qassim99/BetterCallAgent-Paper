# Online fixtures

`demo.json` is a synthetic, version-2 integration fixture. It contains no
benchmark query, gold label, private document, or credential. Its scripted
model outputs are consumed only when `BCA_ONLINE_MODE=fixture`; live mode never
uses them.

The loader validates:

- asset version `2`;
- unique query and document identifiers;
- exact query-to-retrieval-view coverage and all five required view fields;
- exact query-to-ranking coverage;
- ranking references against the document table;
- declared citations, when present, against deterministic extraction from the
  document text;
- one reranker response for every fixture document.

Citation support counts are not stored as answer labels. Stage 5 derives them
from citations found in the configured BM25 top-k documents.

The scripted query-plan output and `retrieval_views` are intentionally stored
separately. Stage 2 reconstructs the five views from the scripted output and
requires a byte-exact match before stage 3 replays the saved rankings.
