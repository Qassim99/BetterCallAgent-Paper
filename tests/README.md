# Tests

Tests are organized by public component:

- `online/` covers configuration, asset validation, six-stage behavior, security
  boundaries, streamed event order, and fixture execution;
- `offline/` covers document-view parsing, deterministic/resumable index construction,
  builder-to-Stage-1 retrieval, replay fingerprints, strict joins, sparse balancing,
  output isolation, and a complete synthetic run;
- `interpretability/` covers formulas, conservative error attribution, deterministic
  surrogate contributions, seeded perturbations, and CLI composition; and
- `repository/` covers the artifact manifest and release hygiene.

Run the locked Python suite:

```bash
uv run --locked pytest
```

Run all Python, frontend, release, fixture, and distribution checks:

```bash
make check
```

Fixtures are synthetic. Passing tests validates software behavior and invariants; it
does not reproduce `0.4709962618` or `0.4806246255438345`, establish legal
correctness, or provide a held-out estimate.
