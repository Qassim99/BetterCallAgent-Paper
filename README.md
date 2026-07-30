# BetterCallAgent

BetterCallAgent is a reproducible research artifact for Swiss legal citation
retrieval. It contains:

- a seven-stage offline experiment for citation prediction;
- a separate six-stage online API for interactive demonstrations;
- one React interface for fixture and live modes;
- framework-independent citation, retrieval, provider, and evaluation code; and
- deterministic interpretability utilities for saved artifacts.

The online demonstration and offline benchmark deliberately share small audited
primitives without pretending to be the same experiment. Gold labels are available
only to offline evaluation and diagnostic code.

## Demo video

[Watch or download the BetterCallAgent demo video](https://github.com/Qassim99/BetterCallAgent-Paper/releases/download/demo-video-v1/Videoprojekt.mp4)
(MP4, 675 MiB).

> **Artifact status:** this repository is source-only. The external artifact set has
> checksums but currently has no DOI or stable download URL. This release therefore
> provides source and a checksum manifest; full reproduction is blocked until the
> authors publish an authorized archive. See
> [External artifacts](artifacts/README.md) and [License status](LICENSE-PENDING.md).

## Results and claims

The historical July 2026 snapshot contains ten validation queries. Its verifier run
completed only 19 of 20 batches: 95 of 100 candidates have scores, and `val_002`
ranks 6–10 are missing. The gate and support thresholds were selected on this same
split.

| Historical configuration | Macro citation F1 | Scope |
|---|---:|---|
| Five-view dense retrieval plus fixed-vote gate | 0.4709962618 | 10 validation queries; same incomplete 95-score verifier artifact |
| Same candidates plus saved balanced sparse support | 0.4806246255438345 | 10 validation queries; same incomplete verifier artifact and tuned thresholds |

The [historical adapter](experiments/historical_validation/README.md) checksum-binds
the exact inputs, exposes all five missing score identities, and reconstructs the
saved result. These values are exploratory validation measurements, not held-out
estimates.

Three activities must not be conflated:

1. The bundled **synthetic fixture** validates schemas, joins, policies, metrics, and
   software execution only. It does not reproduce a paper metric.
2. The **historical adapter** reconstructs the two reported values from the exact
   incomplete July 2026 artifact set.
3. A **fresh strict run** requires all 100 verifier scores and produces a new result.
   It is methodologically cleaner, but it cannot be presented as the historical run.

## Architecture

```text
offline/          Seven explicit benchmark stages and a strict runner
online/           Six explicit FastAPI stages with fixture and live modes
frontend/         React/Vite reviewer interface
interpretability/ Deterministic analyses for saved citation artifacts
src/              Shared framework-independent Python core
experiments/      Historical replay and clearly separated exploratory work
configs/          Non-secret model and full-run configuration
artifacts/        Checksummed external-artifact contract
data/             Authorized data-acquisition guidance
docs/             Architecture, provenance, limitations, and security
scripts/          Artifact, release, and distribution checks
tests/            Component and repository tests
```

The offline path is:

```text
five query views → dense retrieval → sparse-support loading → document materialization
→ verifier reranking → fixed citation gate + support union → validation
```

See [Architecture](docs/architecture.md) for the exact data flow and
[Offline pipeline](offline/README.md) for the file-per-stage map.

## Reviewer quick start

Requirements:

- Python 3.11 or 3.12;
- [uv](https://docs.astral.sh/uv/);
- Node.js 20.19+, 22.13+, or 24+ with npm 11+ for the frontend; and
- Git.

From the repository root:

```bash
python3 -m pip install --user uv
uv sync --locked --extra dev --extra online --extra offline-index
npm --prefix frontend ci
```

`uv.lock` and `frontend/package-lock.json` pin the resolved Python and JavaScript
dependency graphs. Run every source, test, release, frontend, and wheel check with:

```bash
make check
```

### Offline synthetic fixture

This run is deterministic, network-free, and does not require legal data, a model, or
a credential:

```bash
BCA_FIXTURE_OUT="$(mktemp -d)/bettercallagent-fixture"
uv run --locked python -m offline.run \
  --config offline/fixtures/config.toml \
  --output "$BCA_FIXTURE_OUT"
cat "$BCA_FIXTURE_OUT/07_metrics.json"
```

The fixture's score is a software assertion over invented records, not a research
result.

### Online backend in fixture mode

```bash
cp online/.env.example online/.env
set -a
. online/.env
set +a

uv run --locked python -m uvicorn online.app:create_app \
  --factory --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
curl --fail http://127.0.0.1:8000/api/health
curl --fail http://127.0.0.1:8000/api/queries
```

The complete fixture request is in [online/README.md](online/README.md).

### Frontend with no backend

```bash
VITE_DATA_SOURCE=fixture npm --prefix frontend run dev
```

Open the URL printed by Vite, normally <http://localhost:5173>. This explicit fixture
mode never falls back to live data or contains validation answers.

### Frontend with the running backend

```bash
cp frontend/.env.example frontend/.env
npm --prefix frontend run dev
```

The example configuration uses `VITE_DATA_SOURCE=live` and
`VITE_API_BASE_URL=http://localhost:8000`.

## Live online configuration

Use a versioned online asset and a model endpoint that implements the expected
OpenAI-compatible chat contract:

```bash
export BCA_ONLINE_MODE=live
export BCA_ASSET_PATH="$PWD/artifacts/downloads/online/versioned_asset.json"
export BCA_ALLOWED_MODELS="your-provider-model-id"
export BCA_DEFAULT_MODEL="your-provider-model-id"
export BCA_CORS_ORIGINS="http://localhost:5173"
export BCA_LLM_BASE_URL="https://your-provider.example/v1"
export BCA_LLM_API_KEY="read-at-runtime-from-your-secret-manager"

uv run --locked python -m uvicorn online.app:create_app \
  --factory --host 127.0.0.1 --port 8000
```

| Variable | Required | Meaning |
|---|---:|---|
| `BCA_ONLINE_MODE` | yes | Exactly `fixture` or `live` |
| `BCA_ASSET_PATH` | yes | Validated version-2 online retrieval asset |
| `BCA_ALLOWED_MODELS` | yes | Comma-separated provider model allowlist |
| `BCA_DEFAULT_MODEL` | yes | One identifier from the allowlist |
| `BCA_CORS_ORIGINS` | yes | Comma-separated browser origins |
| `BCA_LLM_BASE_URL` | live | HTTPS model API base URL |
| `BCA_LLM_API_KEY` | live | Server-side model credential |
| `BCA_AUTH_TOKEN` | no | Optional API bearer token of at least 16 characters |
| `BCA_RERANK_BASE_URL` | full offline live reranking | HTTPS verifier API base URL |
| `BCA_RERANK_API_KEY` | full offline live reranking | Verifier credential |
| `VITE_DATA_SOURCE` | frontend | Exactly `fixture` or `live` |
| `VITE_API_BASE_URL` | frontend live mode | Online API base URL |
| `VITE_ENABLE_DEBUG` | no | Show normalized observable events when `true` |

Every `VITE_*` value is public in the browser bundle. Provider secrets belong only in
the backend environment and must never be committed.

## Full offline reproduction

The current repository cannot download the complete paper artifact automatically.
The authors must first publish the authorized archive named in
`artifacts/manifest.toml`. Once those exact files are available:

```bash
uv sync --locked --extra dev --extra online --extra offline-gpu

uv run --locked python scripts/verify_artifacts.py \
  --manifest artifacts/manifest.toml \
  --root artifacts/downloads

mkdir -p runs
cp configs/offline.example.toml runs/paper_validation.toml
# Review paths, immutable model IDs, device, and provider environment names.

export BCA_RERANK_BASE_URL="https://your-provider.example/v1"
export BCA_RERANK_API_KEY="read-at-runtime-from-your-secret-manager"

uv run --locked python -m offline.run \
  --config runs/paper_validation.toml
```

The strict runner checks every input checksum, requires complete score coverage,
records resolved configuration and provenance, and rejects a non-empty output
directory. Stage 1 preserves every non-blank query-view string exactly before
tokenization and maps only blank values to one space. It uses one pinned Qwen encoder
for all five views and records the text-preparation, lifecycle, and placement
settings. Its output is a fresh result.

The exact historical dense index is approximately 40.5 GB. A clean, resumable
builder is now available as `bca-build-index`; see
[offline/indexing/README.md](offline/indexing/README.md). The audited predecessor
index manifest did not record the embedding
model or revision, so the release binds the index with both its complete tree hash
and the hashed full-run configuration, which pins
`Qwen/Qwen3-Embedding-8B@1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`.

For the incomplete historical reconstruction, follow the copy-paste command in
[experiments/historical_validation/README.md](experiments/historical_validation/README.md).
The complete contract and caveats are in
[Reproducibility](docs/reproducibility.md) and [Limitations](docs/limitations.md).

## Interpretability

The interpretability package provides citation error attribution, transparent
project-specific RAGAS-style metrics, a deterministic gate surrogate, and a seeded
perturbation proxy. It does not expose chain-of-thought, claim official RAGAS
equivalence, call surrogate contributions SHAP, or describe perturbations as
Integrated Gradients.

```bash
BCA_INTERP_OUT="$(mktemp -d)"
uv run --locked python -m interpretability ragas-style \
  --queries interpretability/examples/queries.jsonl \
  --retrieval interpretability/examples/retrieval.jsonl \
  --predictions interpretability/examples/predictions.jsonl \
  --output "$BCA_INTERP_OUT/ragas_style.json"
cat "$BCA_INTERP_OUT/ragas_style.json"
```

See [interpretability/README.md](interpretability/README.md) for every command,
formula, and limitation.

## Models

The historical snapshot used:

- embedding model
  `Qwen/Qwen3-Embedding-8B@1d8ad4ca9b3dd8059ad90a75d4983776a23d44af`; and
- provider-specific reranker identifier `qwen3-30b-a3b-instruct-2507`.

Requested alternatives are recommendations only; none produced the reported values:

| Recommendation | Exact reference identifier | Status |
|---|---|---|
| DeepSeek V4 Flash | local: `deepseek-ai/DeepSeek-V4-Flash`; provider example: `deepseek/deepseek-v4-flash` | Hosted or local reranking candidate |
| Qwen3.6-27B | `Qwen/Qwen3.6-27B` | Open-weight local reranking candidate |
| Soofi S | provider example `Soofi-S-Instruct` | Early access and deployment-specific; no generally downloadable checkpoint recorded as of 2026-07-29 |

Model labels are not interchangeable. A provider change or model change requires a
new complete run. See [Models](docs/models.md).

## Data, security, and release status

- Competition inputs, the case-law snapshot, the dense index, saved evidence traces,
  and model outputs are not committed. Their expected sizes and SHA-256 hashes are in
  `artifacts/manifest.toml`.
- The case-law corpus snapshot is derived from `voilaj/swiss-caselaw`; use the
  committed tree hash because the upstream dataset can change.
- Real online queries can send questions and selected document excerpts to the
  configured provider. Review its retention, location, and data-processing terms.
- Historical credentials potentially present in predecessor development history
  must be rotated immediately. That history and its Git objects are not part of this
  clean repository.
- This is a public **source snapshot**, not an open-source or complete artifact
  release. No open-source license has been selected, so all rights remain reserved.
- Public visibility does not imply that external data, indexes, model outputs, or
  derived artifacts may be redistributed. Their availability and permissions remain
  separate release decisions.

Before publishing an open-source or complete artifact release, the owners must
approve a license, confirm contributor and redistribution rights, complete credential
rotation, and publish an authorized artifact archive. See [Security](SECURITY.md),
[Third-party notices](NOTICE.md), and [License status](LICENSE-PENDING.md).

BetterCallAgent is research software, not legal advice.
