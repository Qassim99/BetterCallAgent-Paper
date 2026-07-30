"""Stage 5: score each retrieved document with an explicit verifier model.

Live mode uses one OpenAI-compatible HTTPS provider.  Replay mode is an explicit,
network-free adapter whose records are bound to the exact query, document, prompt
version, and model by SHA-256 fingerprints.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from bettercallagent.providers.openai_compatible import (
    ChatProvider,
    OpenAICompatibleProvider,
    ProviderError,
    parse_json_object,
)
from bettercallagent.schemas import ChatMessage
from offline.identity import candidate_id
from offline.io import JsonObject, atomic_write_jsonl, normalize_text, read_jsonl

PROMPT_VERSION = "swiss-origin-verifier-de-v2"
SYSTEM_PROMPT = """Du bist ein juristischer Retrieval-Reranker fuer Schweizer Gerichtsentscheide.

Ziel: Finde zu einer Kaggle-Query das urspruengliche Gerichtsdokument. Die Query wurde oft aus dem Originalfall erzeugt, aber stark anonymisiert/paraphrasiert:
- Namen, Firmen, Orte, Laender, Daten, Betraege und Produktnamen koennen ersetzt sein.
- Die Branche kann leicht verschoben sein.
- Der rechtliche Kern, Verfahrenskontext, Anspruchstyp, Normenfamilie und die wichtigsten Sachverhaltsanker bleiben aber meist erhalten.

Bewerte jeden Kandidaten unabhaengig mit Score 0 bis 10:
10 = praktisch sicher das Ursprungdokument.
8-9 = sehr wahrscheinlich derselbe Fall, trotz anonymisierter Details.
6-7 = enge Schwester / gleicher Rechts- und Sachverhaltskern, aber wichtige Details fehlen.
4-5 = nur gleiche Rechtsfamilie oder allgemeine Dogmatik.
0-3 = wahrscheinlich nicht der Ursprung.

Wichtig:
- Nicht an synthetischen Namen/Orten/Jahren kleben.
- Hoeher gewichten: seltene Faktenkombination, Prozessart, Normen, Rechtsfrage, konkrete Rechtsbegehren, Unfall-/Vertrags-/Familienrechtsstruktur.
- Niedriger gewichten: nur gleiche Norm ohne passenden Sachverhalt.
- Wenn ein Dokument nur ein Leitentscheid zur Dogmatik ist, aber nicht die konkrete Story traegt, score eher 4-6.

Antworte NUR als valides JSON, kein Markdown:
{"scores":[{"candidate_id":"...","score":0.0,"confidence":0.0,"rationale_de":"kurze Begruendung","matching_anchors":["..."],"missing_or_wrong_anchors":["..."]}]}
"""


def _effective_document(row: Mapping[str, Any], document_char_limit: int) -> str:
    if document_char_limit < 0:
        raise ValueError("document_char_limit cannot be negative.")
    text = str(row.get("document_text") or "")
    if document_char_limit and len(text) > document_char_limit:
        return text[:document_char_limit] + " …[truncated]"
    return text


def input_fingerprint(
    row: Mapping[str, Any],
    *,
    model: str,
    document_char_limit: int = 0,
) -> str:
    """Bind a cached score to its exact semantic input and verifier model."""

    payload = {
        "prompt_version": PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(SYSTEM_PROMPT.encode("utf-8")).hexdigest(),
        "model": model,
        "query_id": str(row.get("query_id") or ""),
        "query": normalize_text(row.get("query")),
        "meta_query": normalize_text(row.get("meta_query")),
        "keywords_query": normalize_text(row.get("keywords_query")),
        "doc_id": str(row.get("doc_id") or ""),
        "rank": int(row.get("rank") or 0),
        "global_idx": row.get("global_idx"),
        "fusion_score": row.get("fusion_score"),
        "hits": row.get("hits") or [],
        "document_char_limit": document_char_limit,
        "effective_document": _effective_document(row, document_char_limit),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _validate_inputs(rows: Sequence[JsonObject]) -> list[JsonObject]:
    if not rows:
        raise ValueError("Reranker input is empty.")
    seen: set[str] = set()
    output: list[JsonObject] = []
    for row in rows:
        identifier = candidate_id(row)
        if identifier in seen:
            raise ValueError(f"Duplicate reranker candidate identity: {identifier}")
        seen.add(identifier)
        query = normalize_text(row.get("query"))
        document = str(row.get("document_text") or "").strip()
        rank = int(row.get("rank") or 0)
        if not query or not document or rank <= 0:
            raise ValueError(f"{identifier}: query, document_text, and positive rank are required")
        output.append(row)
    return output


def _prompt_for_batch(rows: Sequence[JsonObject], *, document_char_limit: int) -> str:
    if document_char_limit < 0:
        raise ValueError("document_char_limit cannot be negative.")
    query_values = {normalize_text(row.get("query")) for row in rows}
    if len(query_values) != 1:
        raise ValueError("A verifier batch may contain only one query.")
    query = next(iter(query_values))
    meta_values = {normalize_text(row.get("meta_query")) for row in rows}
    keyword_values = {normalize_text(row.get("keywords_query")) for row in rows}
    if len(meta_values) != 1 or len(keyword_values) != 1:
        raise ValueError("A verifier batch must share its generated query views.")
    parts = [
        "ORIGINAL-QUERY / KAGGLE-QUERY:",
        query,
        "",
        "HEURISTISCHER META-SEARCHTERM:",
        next(iter(meta_values)),
        "",
        "HEURISTISCHE KEYWORDS:",
        next(iter(keyword_values)),
        "",
        "KANDIDATEN. Bewerte jeden candidate_id:",
    ]
    for row in rows:
        text = _effective_document(row, document_char_limit)
        parts.append(
            "\n---\n"
            f"candidate_id: {candidate_id(row)}\n"
            f"retrieval_rank: {int(row['rank'])}\n"
            f"fusion_score: {row.get('fusion_score')}\n"
            f"doc_id: {row.get('doc_id')}\n"
            f"embedding_hits: {json.dumps(row.get('hits') or [], ensure_ascii=False)}\n"
            f"document:\n{text}"
        )
    parts.append("\nGib exakt JSON mit einem Score fuer jeden candidate_id zurueck.")
    return "\n".join(parts)


def batch_fingerprint(
    rows: Sequence[JsonObject],
    *,
    model: str,
    document_char_limit: int,
    max_tokens: int = 4_096,
    json_response: bool = True,
    temperature: float = 0.0,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> str:
    """Bind a replay to the exact prompt and ordered batch composition."""

    generation: JsonObject = {
        "json_response": json_response,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if chat_template_kwargs is not None:
        generation["chat_template_kwargs"] = dict(chat_template_kwargs)
    payload = {
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "generation": generation,
        "system_prompt": SYSTEM_PROMPT,
        "user_prompt": _prompt_for_batch(
            rows,
            document_char_limit=document_char_limit,
        ),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _make_batches(
    rows: Sequence[JsonObject],
    *,
    batch_size: int,
) -> list[list[JsonObject]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    grouped: dict[str, list[JsonObject]] = defaultdict(list)
    for row in rows:
        grouped[str(row["query_id"])].append(row)
    batches: list[list[JsonObject]] = []
    for query_id in sorted(grouped):
        ordered = sorted(grouped[query_id], key=lambda row: int(row["rank"]))
        batches.extend(
            ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)
        )
    return batches


def prepare_batches(
    rows: Sequence[JsonObject],
    *,
    batch_size: int = 5,
) -> list[list[JsonObject]]:
    """Validate inputs and return the exact batches consumed by Stage 5."""

    return _make_batches(_validate_inputs(rows), batch_size=batch_size)


def _parse_scores(
    content: str,
    *,
    expected_ids: Sequence[str],
) -> dict[str, tuple[float, float, str]]:
    payload = parse_json_object(content)
    raw_scores = payload.get("scores")
    if not isinstance(raw_scores, list):
        raise ValueError("Verifier JSON must contain a 'scores' array.")
    parsed: dict[str, tuple[float, float, str]] = {}
    for item in raw_scores:
        if not isinstance(item, dict):
            raise ValueError("Every verifier score must be an object.")
        identifier = str(item.get("candidate_id") or "")
        if not identifier:
            raise ValueError("Verifier returned an empty candidate_id.")
        if identifier in parsed:
            raise ValueError(f"Verifier returned duplicate candidate_id {identifier!r}.")
        raw_score = item.get("score")
        raw_confidence = item.get("confidence")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise ValueError(f"{identifier}: score must be a JSON number.")
        if isinstance(raw_confidence, bool) or not isinstance(raw_confidence, (int, float)):
            raise ValueError(f"{identifier}: confidence must be a JSON number.")
        score = float(raw_score)
        confidence = float(raw_confidence)
        if not math.isfinite(score) or not math.isfinite(confidence):
            raise ValueError(f"{identifier}: score and confidence must be finite.")
        rationale = normalize_text(item.get("rationale") or item.get("rationale_de"))
        if not 0.0 <= score <= 10.0:
            raise ValueError(f"{identifier}: score must be in [0, 10].")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"{identifier}: confidence must be in [0, 1].")
        if not rationale:
            raise ValueError(f"{identifier}: rationale must not be empty.")
        parsed[identifier] = (score, confidence, rationale)
    if set(parsed) != set(expected_ids):
        missing = sorted(set(expected_ids) - set(parsed))
        extra = sorted(set(parsed) - set(expected_ids))
        raise ValueError(f"Verifier IDs differ: missing={missing}, extra={extra}")
    return parsed


def rerank_from_replay(
    rows: Sequence[JsonObject],
    *,
    replay_path: Path,
    model: str,
    batch_size: int = 5,
    document_char_limit: int = 0,
    max_tokens: int = 4_096,
    json_response: bool = True,
    temperature: float = 0.0,
) -> list[JsonObject]:
    """Load fingerprint-bound scores and reject stale or misaligned caches."""

    inputs = _validate_inputs(rows)
    replay: dict[str, JsonObject] = {}
    for record in read_jsonl(replay_path):
        identifier = str(record.get("candidate_id") or "")
        if not identifier or identifier in replay:
            raise ValueError(f"Duplicate or empty replay candidate_id {identifier!r}.")
        replay[identifier] = record
    expected_ids = {candidate_id(row) for row in inputs}
    if set(replay) != expected_ids:
        missing = sorted(expected_ids - set(replay))
        extra = sorted(set(replay) - expected_ids)
        raise ValueError(f"Replay IDs differ: missing={missing}, extra={extra}")

    batch_hashes = {
        candidate_id(row): batch_fingerprint(
            batch,
            model=model,
            document_char_limit=document_char_limit,
            max_tokens=max_tokens,
            json_response=json_response,
            temperature=temperature,
        )
        for batch in _make_batches(inputs, batch_size=batch_size)
        for row in batch
    }

    output: list[JsonObject] = []
    for row in inputs:
        identifier = candidate_id(row)
        record = replay[identifier]
        if record.get("model") != model:
            raise ValueError(f"{identifier}: replay model does not match {model!r}.")
        fingerprint = input_fingerprint(
            row,
            model=model,
            document_char_limit=document_char_limit,
        )
        if record.get("input_sha256") != fingerprint:
            raise ValueError(f"{identifier}: stale replay fingerprint.")
        batch_hash = batch_hashes[identifier]
        if record.get("batch_sha256") != batch_hash:
            raise ValueError(f"{identifier}: stale replay batch fingerprint.")
        if int(record.get("rank") or 0) != int(row["rank"]):
            raise ValueError(f"{identifier}: replay rank does not match.")
        score = float(record["score"])
        confidence = float(record["confidence"])
        rationale = normalize_text(record.get("rationale"))
        _parse_scores(
            json.dumps(
                {
                    "scores": [
                        {
                            "candidate_id": identifier,
                            "score": score,
                            "confidence": confidence,
                            "rationale": rationale,
                        }
                    ]
                }
            ),
            expected_ids=[identifier],
        )
        output.append(
            {
                "query_id": row["query_id"],
                "doc_id": row["doc_id"],
                "global_idx": row.get("global_idx"),
                "candidate_id": identifier,
                "rank": int(row["rank"]),
                "score": score,
                "confidence": confidence,
                "rationale": rationale,
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "input_sha256": fingerprint,
                "batch_sha256": batch_hash,
                "source": "verified_replay",
            }
        )
    return output


async def rerank_live(
    rows: Sequence[JsonObject],
    *,
    provider: ChatProvider,
    model: str,
    batch_size: int,
    concurrency: int,
    document_char_limit: int,
    checkpoint_path: Path | None = None,
    max_attempts: int = 3,
    retry_delay_seconds: float = 1.0,
    max_tokens: int = 4_096,
    json_response: bool = True,
    temperature: float = 0.0,
    chat_template_kwargs: Mapping[str, Any] | None = None,
) -> list[JsonObject]:
    """Score batches with bounded retries and an atomically durable checkpoint."""

    inputs = _validate_inputs(rows)
    if min(batch_size, concurrency, max_attempts, max_tokens) <= 0:
        raise ValueError("batch_size, concurrency, max_attempts, and max_tokens must be positive.")
    if retry_delay_seconds < 0:
        raise ValueError("retry_delay_seconds cannot be negative.")
    batches = _make_batches(inputs, batch_size=batch_size)
    semaphore = asyncio.Semaphore(concurrency)
    checkpoint_lock = asyncio.Lock()
    expected_batches = {
        batch_fingerprint(
            batch,
            model=model,
            document_char_limit=document_char_limit,
            max_tokens=max_tokens,
            json_response=json_response,
            temperature=temperature,
            chat_template_kwargs=chat_template_kwargs,
        ): batch
        for batch in batches
    }
    checkpoint_records: dict[str, JsonObject] = {}
    batch_scores: dict[str, dict[str, tuple[float, float, str]]] = {}

    if checkpoint_path is not None and checkpoint_path.exists():
        for record in read_jsonl(checkpoint_path):
            fingerprint = str(record.get("batch_sha256") or "")
            if not fingerprint or fingerprint in checkpoint_records:
                raise ValueError("Checkpoint contains a duplicate or empty batch hash.")
            batch = expected_batches.get(fingerprint)
            if batch is None:
                raise ValueError("Checkpoint contains a stale or unknown batch.")
            identifiers = [candidate_id(row) for row in batch]
            if record.get("requested_model") != model:
                raise ValueError("Checkpoint requested_model does not match.")
            if record.get("prompt_version") != PROMPT_VERSION:
                raise ValueError("Checkpoint prompt_version does not match.")
            if record.get("candidate_ids") != identifiers:
                raise ValueError("Checkpoint candidate order does not match.")
            normalized_content = record.get("normalized_content")
            if not isinstance(normalized_content, str):
                raise ValueError("Checkpoint normalized_content must be a string.")
            batch_scores[fingerprint] = _parse_scores(
                normalized_content,
                expected_ids=identifiers,
            )
            record.pop("raw_response", None)
            checkpoint_records[fingerprint] = record

    batch_order = list(expected_batches)

    def write_checkpoint() -> None:
        if checkpoint_path is None:
            return
        atomic_write_jsonl(
            checkpoint_path,
            (
                checkpoint_records[fingerprint]
                for fingerprint in batch_order
                if fingerprint in checkpoint_records
            ),
        )

    if checkpoint_records:
        write_checkpoint()

    async def score_batch(fingerprint: str, batch: list[JsonObject]) -> None:
        if fingerprint in batch_scores:
            return
        identifiers = [candidate_id(row) for row in batch]
        last_error: Exception | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                async with semaphore:
                    messages = (
                        ChatMessage(role="system", content=SYSTEM_PROMPT),
                        ChatMessage(
                            role="user",
                            content=_prompt_for_batch(
                                batch,
                                document_char_limit=document_char_limit,
                            ),
                        ),
                    )
                    response = await provider.complete(
                        messages,
                        model=model,
                        purpose="offline_reranking",
                        metadata={"query_id": str(batch[0]["query_id"])},
                        json_response=json_response,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )
                parsed = _parse_scores(
                    response.content,
                    expected_ids=identifiers,
                )
                record: JsonObject = {
                    "schema_version": 1,
                    "batch_sha256": fingerprint,
                    "candidate_ids": identifiers,
                    "input_sha256": {
                        candidate_id(row): input_fingerprint(
                            row,
                            model=model,
                            document_char_limit=document_char_limit,
                        )
                        for row in batch
                    },
                    "prompt_version": PROMPT_VERSION,
                    "requested_model": model,
                    "provider_model": response.model,
                    "generation": {
                        "json_response": json_response,
                        "max_tokens": max_tokens,
                        "temperature": temperature,
                        "chat_template_kwargs": dict(chat_template_kwargs or {}),
                    },
                    "usage": {
                        "prompt_tokens": response.usage_prompt_tokens,
                        "completion_tokens": response.usage_completion_tokens,
                        "reasoning_tokens": response.usage_reasoning_tokens,
                        "total_tokens": response.usage_total_tokens,
                    },
                    "latency_seconds": response.latency_seconds,
                    "attempts": attempt,
                    "final_content": response.content,
                    "reasoning_content": response.reasoning_content,
                    "normalized_content": response.content,
                }
                async with checkpoint_lock:
                    batch_scores[fingerprint] = parsed
                    checkpoint_records[fingerprint] = record
                    write_checkpoint()
                return
            except (ProviderError, ValueError, KeyError, TypeError) as exc:
                last_error = exc
                if attempt < max_attempts and retry_delay_seconds:
                    await asyncio.sleep(retry_delay_seconds * attempt)
        assert last_error is not None
        raise RuntimeError(
            f"Verifier batch failed after {max_attempts} attempts: {fingerprint}"
        ) from last_error

    async with asyncio.TaskGroup() as group:
        for fingerprint, batch in expected_batches.items():
            group.create_task(score_batch(fingerprint, batch))

    scores = {
        identifier: value
        for result in batch_scores.values()
        for identifier, value in result.items()
    }
    batch_hash_by_candidate = {
        candidate_id(row): fingerprint
        for fingerprint, batch in expected_batches.items()
        for row in batch
    }
    output: list[JsonObject] = []
    for row in inputs:
        identifier = candidate_id(row)
        score, confidence, rationale = scores[identifier]
        output.append(
            {
                "query_id": row["query_id"],
                "doc_id": row["doc_id"],
                "global_idx": row.get("global_idx"),
                "candidate_id": identifier,
                "rank": int(row["rank"]),
                "score": score,
                "confidence": confidence,
                "rationale": rationale,
                "model": model,
                "prompt_version": PROMPT_VERSION,
                "input_sha256": input_fingerprint(
                    row,
                    model=model,
                    document_char_limit=document_char_limit,
                ),
                "batch_sha256": batch_hash_by_candidate[identifier],
                "provider_model": checkpoint_records[batch_hash_by_candidate[identifier]][
                    "provider_model"
                ],
                "source": "live_provider",
            }
        )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--replay", type=Path, help="Explicit fingerprint-bound replay.")
    parser.add_argument("--base-url", help="Required in live mode; must use HTTPS.")
    parser.add_argument("--api-key-env", default="BCA_RERANK_API_KEY")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=4_096)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Durable batch envelope; defaults beside --output in live mode.",
    )
    parser.add_argument(
        "--document-char-limit",
        type=int,
        default=0,
        help="Optional verifier-only truncation; 0 preserves full text.",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Forward chat_template_kwargs.enable_thinking when the provider supports it.",
    )
    return parser


async def _run(args: argparse.Namespace) -> list[JsonObject]:
    rows = list(read_jsonl(args.input))
    if args.replay is not None:
        return rerank_from_replay(
            rows,
            replay_path=args.replay,
            model=args.model,
            batch_size=args.batch_size,
            document_char_limit=args.document_char_limit,
            max_tokens=args.max_tokens,
        )
    if not args.base_url:
        raise ValueError("--base-url is required unless --replay is supplied.")
    api_key = os.environ.get(args.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing required environment variable: {args.api_key_env}")
    provider = OpenAICompatibleProvider(
        base_url=args.base_url,
        api_key=api_key,
        timeout_seconds=args.timeout,
        chat_template_kwargs=(
            {"enable_thinking": args.enable_thinking} if args.enable_thinking is not None else None
        ),
    )
    try:
        checkpoint = args.checkpoint or args.output.with_name(f"{args.output.stem}.batches.jsonl")
        return await rerank_live(
            rows,
            provider=provider,
            model=args.model,
            batch_size=args.batch_size,
            concurrency=args.concurrency,
            document_char_limit=args.document_char_limit,
            checkpoint_path=checkpoint,
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
            max_tokens=args.max_tokens,
            chat_template_kwargs=(
                {"enable_thinking": args.enable_thinking}
                if args.enable_thinking is not None
                else None
            ),
        )
    finally:
        await provider.close()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    rows = asyncio.run(_run(args))
    atomic_write_jsonl(args.output, rows)
    print(f"Reranked {len(rows)} candidates -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
