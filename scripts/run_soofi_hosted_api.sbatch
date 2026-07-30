#!/usr/bin/env bash
#SBATCH --job-name=bca-soofi-rerank
#SBATCH --partition=ampere
#SBATCH --account=ampere
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --output=soofi-reranker-%j.out

set -euo pipefail

REPOSITORY_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
cd "$REPOSITORY_ROOT"

ENV_FILE="${SOOFI_ENV_FILE:-$REPOSITORY_ROOT/.env}"
if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing environment file: $ENV_FILE" >&2
  exit 2
fi
set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

if [[ -z "${SOOFI_API_KEY:-}" ]]; then
  echo "SOOFI_API_KEY is not set in the environment file." >&2
  exit 2
fi

export SOOFI_BASE_URL="${SOOFI_BASE_URL:-https://soofi-owu.l3s.de/api}"
if [[ -z "${SOOFI_MODEL:-}" ]]; then
  echo "SOOFI_MODEL is not set in the environment file." >&2
  exit 2
fi
export SOOFI_MODEL

PYTHON_EXECUTABLE="${BCA_PYTHON:-$REPOSITORY_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_EXECUTABLE" ]]; then
  echo "Missing Python environment: $PYTHON_EXECUTABLE (run 'uv sync --locked --extra dev')." >&2
  exit 2
fi

OUTPUT_DIR="${SOOFI_OUTPUT_DIR:-$REPOSITORY_ROOT/runs/model_comparison/soofi}"
arguments=(
  -m experiments.model_comparison.run_soofi
  --output-dir "$OUTPUT_DIR"
  --max-tokens 16384
  --no-enable-thinking
  --no-json-response
)
if [[ -n "${SOOFI_LIMIT_BATCHES:-}" ]]; then
  arguments+=(--limit-batches "$SOOFI_LIMIT_BATCHES")
fi

exec "$PYTHON_EXECUTABLE" "${arguments[@]}"
