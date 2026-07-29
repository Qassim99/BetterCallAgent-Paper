# Contributing

BetterCallAgent is a paper artifact. Changes must preserve traceability and must never
use validation answers as online hints.

1. Create a focused branch.
2. Add or update tests before changing a pipeline rule.
3. Install locked dependencies with
   `uv sync --locked --extra dev --extra online` and
   `npm --prefix frontend ci`.
4. Run `make check`.
5. Record every metric with its exact split, query count, configuration, model and
   prompt identity, input checksums, missing-data status, and output path.
6. Keep gold labels in offline evaluation and diagnostic code only.
7. Do not commit generated data, model weights, indexes, logs, provider responses, or
   credentials.
8. Treat a complete rerun as a new result; never overwrite historical provenance.

Code, comments, commit messages, and documentation are written in English. Prefer
small explicit functions, immutable records, strict joins, and visible failures over
implicit behavior.

No open-source license or contributor agreement has been approved. The owners must
establish contribution terms before accepting external contributions for public
redistribution.
