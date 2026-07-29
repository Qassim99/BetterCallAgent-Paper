# Repository utilities

| Script | Purpose |
|---|---|
| `verify_artifacts.py` | Validate external files and directory trees by size, count, and SHA-256 |
| `check_release.py` | Reject credentials, machine-specific paths, generated content, and oversized release files in the worktree and reachable history |
| `check_distribution.py` | Confirm that the built wheel contains intended packages and no research/runtime data |
| `clean_generated.py` | Remove known generated caches, builds, and package metadata |

Run the release and distribution checks from the repository root:

```bash
uv run --locked python scripts/check_release.py
uv run --locked python -m build
uv run --locked python scripts/check_distribution.py dist
```

Verify external artifacts only after placing the authorized files:

```bash
uv run --locked python scripts/verify_artifacts.py \
  --manifest artifacts/manifest.toml \
  --root artifacts/downloads
```

A source-only clone is expected to fail artifact verification with a clear missing
file report. It must not download or fabricate replacements.

`make check` runs Python checks, the synthetic offline pipeline, frontend checks,
release checks, and the wheel validation in one command.
