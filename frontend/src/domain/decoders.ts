import type {
  CitationDecision,
  CitationValidationData,
  CuratedQuery,
  DataMode,
  EvaluationMetrics,
  EvidenceDocument,
  FinalAnswerStepData,
  HealthResponse,
  ModelsResponse,
  PipelineStepNumber,
  QueriesResponse,
  QueryGenerationData,
  RerankingData,
  RetrievalData,
  RunEvent,
  StepCompleteEvent,
  UnderstandingData,
} from "./models";

type JsonRecord = Record<string, unknown>;

export class ProtocolError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ProtocolError";
  }
}

function record(value: unknown, path: string): JsonRecord {
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new ProtocolError(`${path} must be an object.`);
  }
  return value as JsonRecord;
}

function stringValue(object: JsonRecord, key: string, path: string): string {
  const value = object[key];
  if (typeof value !== "string") {
    throw new ProtocolError(`${path}.${key} must be a string.`);
  }
  return value;
}

function optionalString(object: JsonRecord, key: string, path: string): string | undefined {
  const value = object[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "string") {
    throw new ProtocolError(`${path}.${key} must be a string when present.`);
  }
  return value;
}

function numberValue(object: JsonRecord, key: string, path: string): number {
  const value = object[key];
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new ProtocolError(`${path}.${key} must be a finite number.`);
  }
  return value;
}

function optionalNumber(object: JsonRecord, key: string, path: string): number | undefined {
  const value = object[key];
  if (value === undefined || value === null) {
    return undefined;
  }
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new ProtocolError(`${path}.${key} must be a finite number when present.`);
  }
  return value;
}

function booleanValue(object: JsonRecord, key: string, path: string): boolean {
  const value = object[key];
  if (typeof value !== "boolean") {
    throw new ProtocolError(`${path}.${key} must be a boolean.`);
  }
  return value;
}

function arrayValue<T>(
  object: JsonRecord,
  key: string,
  path: string,
  decode: (value: unknown, itemPath: string) => T,
): T[] {
  const value = object[key];
  if (!Array.isArray(value)) {
    throw new ProtocolError(`${path}.${key} must be an array.`);
  }
  return value.map((item, index) => decode(item, `${path}.${key}[${String(index)}]`));
}

function stringArray(object: JsonRecord, key: string, path: string): string[] {
  return arrayValue(object, key, path, (value, itemPath) => {
    if (typeof value !== "string") {
      throw new ProtocolError(`${itemPath} must be a string.`);
    }
    return value;
  });
}

function dataMode(value: unknown, path: string): DataMode {
  if (value !== "live" && value !== "fixture") {
    throw new ProtocolError(`${path} must be "live" or "fixture".`);
  }
  return value;
}

function stepNumber(value: unknown, path: string): PipelineStepNumber {
  if (value === 1 || value === 2 || value === 3 || value === 4 || value === 5 || value === 6) {
    return value;
  }
  throw new ProtocolError(`${path} must be an integer from 1 to 6.`);
}

function evidenceDocument(value: unknown, path: string): EvidenceDocument {
  const object = record(value, path);
  return {
    doc_ref: stringValue(object, "doc_ref", path),
    court: optionalString(object, "court", path),
    decision_id: optionalString(object, "decision_id", path),
    docket_number: optionalString(object, "docket_number", path),
    decision_date: optionalString(object, "decision_date", path),
    snippet: stringValue(object, "snippet", path),
    score: numberValue(object, "score", path),
    score_kind: stringValue(object, "score_kind", path),
    rank: optionalNumber(object, "rank", path),
    confidence: optionalNumber(object, "confidence", path),
    rationale_de: optionalString(object, "rationale_de", path),
  };
}

function citationDecision(value: unknown, path: string): CitationDecision {
  const object = record(value, path);
  const type = stringValue(object, "type", path);
  if (type !== "law" && type !== "court") {
    throw new ProtocolError(`${path}.type must be "law" or "court".`);
  }
  return {
    citation: stringValue(object, "citation", path),
    type,
    reason: stringValue(object, "reason", path),
    votes: optionalString(object, "votes", path),
  };
}

function understandingData(value: unknown, path: string): UnderstandingData {
  const object = record(value, path);
  const route = stringValue(object, "route", path);
  if (route !== "legal") {
    throw new ProtocolError(`${path}.route must be "legal".`);
  }
  return {
    kind: "understanding",
    route,
    restated_question: stringValue(object, "restated_question", path),
    legal_topic: stringValue(object, "legal_topic", path),
    languages_considered: stringArray(object, "languages_considered", path),
    key_legal_concepts: stringArray(object, "key_legal_concepts", path),
  };
}

function queryGenerationData(value: unknown, path: string): QueryGenerationData {
  const object = record(value, path);
  return {
    kind: "query_generation",
    search_queries: stringArray(object, "search_queries", path),
    meta_searchterm_de: stringValue(object, "meta_searchterm_de", path),
    keywords: stringArray(object, "keywords", path),
  };
}

function retrievalData(value: unknown, path: string): RetrievalData {
  const object = record(value, path);
  const counts = record(object.counts, `${path}.counts`);
  return {
    kind: "retrieval",
    counts: {
      dense: numberValue(counts, "dense", `${path}.counts`),
      bm25: numberValue(counts, "bm25", `${path}.counts`),
      hybrid_unique: numberValue(counts, "hybrid_unique", `${path}.counts`),
    },
    dense_available: booleanValue(object, "dense_available", path),
    dense: arrayValue(object, "dense", path, evidenceDocument),
    bm25: arrayValue(object, "bm25", path, evidenceDocument),
    hybrid: arrayValue(object, "hybrid", path, evidenceDocument),
  };
}

function rerankingData(value: unknown, path: string): RerankingData {
  const object = record(value, path);
  return {
    kind: "reranking",
    model: stringValue(object, "model", path),
    before: arrayValue(object, "before", path, (item, itemPath) => {
      const before = record(item, itemPath);
      return {
        rank: numberValue(before, "rank", itemPath),
        doc_ref: stringValue(before, "doc_ref", itemPath),
      };
    }),
    after: arrayValue(object, "after", path, evidenceDocument),
    top_select: numberValue(object, "top_select", path),
  };
}

function citationValidationData(value: unknown, path: string): CitationValidationData {
  const object = record(value, path);
  const support = record(object.bm25_support, `${path}.bm25_support`);
  const rawCounts = record(support.counts, `${path}.bm25_support.counts`);
  const counts: Record<string, number> = {};
  for (const [citation, count] of Object.entries(rawCounts)) {
    if (typeof count !== "number" || !Number.isFinite(count)) {
      throw new ProtocolError(`${path}.bm25_support.counts.${citation} must be a number.`);
    }
    counts[citation] = count;
  }
  return {
    kind: "citation_validation",
    rule: stringValue(object, "rule", path),
    qwen_rule: stringValue(object, "qwen_rule", path),
    bm25_rule: stringValue(object, "bm25_rule", path),
    accepted: arrayValue(object, "accepted", path, citationDecision),
    rejected: arrayValue(object, "rejected", path, citationDecision),
    predicted_citations: stringArray(object, "predicted_citations", path),
    bm25_support: {
      top_k: numberValue(support, "top_k", `${path}.bm25_support`),
      min_votes: numberValue(support, "min_votes", `${path}.bm25_support`),
      counts,
    },
  };
}

function finalAnswerStepData(value: unknown, path: string): FinalAnswerStepData {
  const object = record(value, path);
  return {
    kind: "final_answer_step",
    grounded_on: stringArray(object, "grounded_on", path),
    document_count: numberValue(object, "document_count", path),
  };
}

function evaluationMetrics(value: unknown, path: string): EvaluationMetrics {
  const object = record(value, path);
  return {
    precision: numberValue(object, "precision", path),
    recall: numberValue(object, "recall", path),
    f1: numberValue(object, "f1", path),
    true_positives: optionalNumber(object, "true_positives", path),
    false_positives: optionalNumber(object, "false_positives", path),
    false_negatives: optionalNumber(object, "false_negatives", path),
  };
}

function baseEvent(object: JsonRecord, path: string): { ts: number; run_id?: string } {
  return {
    ts: numberValue(object, "ts", path),
    run_id: optionalString(object, "run_id", path),
  };
}

function stepCompleteEvent(
  object: JsonRecord,
  step: PipelineStepNumber,
  path: string,
): StepCompleteEvent {
  const common = {
    ...baseEvent(object, path),
    type: "step_complete" as const,
    name: stringValue(object, "name", path),
    summary: stringValue(object, "summary", path),
  };
  switch (step) {
    case 1:
      return { ...common, step, data: understandingData(object.data, `${path}.data`) };
    case 2:
      return { ...common, step, data: queryGenerationData(object.data, `${path}.data`) };
    case 3:
      return { ...common, step, data: retrievalData(object.data, `${path}.data`) };
    case 4:
      return { ...common, step, data: rerankingData(object.data, `${path}.data`) };
    case 5:
      return { ...common, step, data: citationValidationData(object.data, `${path}.data`) };
    case 6:
      return { ...common, step, data: finalAnswerStepData(object.data, `${path}.data`) };
  }
}

export function decodeRunEvent(value: unknown): RunEvent {
  const path = "event";
  const object = record(value, path);
  const type = stringValue(object, "type", path);
  const common = baseEvent(object, path);

  switch (type) {
    case "run_start": {
      const queryId = object.query_id;
      if (queryId !== null && typeof queryId !== "string") {
        throw new ProtocolError("event.query_id must be a string or null.");
      }
      return {
        ...common,
        type,
        query: stringValue(object, "query", path),
        query_id: queryId,
        model: stringValue(object, "model", path),
        mode: dataMode(object.mode, "event.mode"),
      };
    }
    case "step_start":
      return {
        ...common,
        type,
        step: stepNumber(object.step, "event.step"),
        name: stringValue(object, "name", path),
      };
    case "step_complete": {
      const step = stepNumber(object.step, "event.step");
      return stepCompleteEvent(object, step, path);
    }
    case "final_answer":
      return {
        ...common,
        type,
        markdown: stringValue(object, "markdown", path),
        grounded_on: stringArray(object, "grounded_on", path),
        metrics:
          object.metrics === undefined || object.metrics === null
            ? undefined
            : evaluationMetrics(object.metrics, "event.metrics"),
      };
    case "run_complete":
      return {
        ...common,
        type,
        elapsed_s: numberValue(object, "elapsed_s", path),
        usage_total_tokens: numberValue(object, "usage_total_tokens", path),
      };
    case "stream_end":
      return { ...common, type };
    case "error": {
      const code = stringValue(object, "code", path);
      if (code !== "invalid_request" && code !== "unauthorized" && code !== "pipeline_error") {
        throw new ProtocolError("event.code is not a supported error code.");
      }
      const rawStep = object.step;
      return {
        ...common,
        type,
        code,
        message: stringValue(object, "message", path),
        step: rawStep === undefined || rawStep === null ? undefined : stepNumber(rawStep, "event.step"),
      };
    }
    default:
      throw new ProtocolError(`Unsupported event type: ${type}.`);
  }
}

export function decodeRunEventJson(data: string): RunEvent {
  let value: unknown;
  try {
    value = JSON.parse(data);
  } catch {
    throw new ProtocolError("SSE data is not valid JSON.");
  }
  return decodeRunEvent(value);
}

export function decodeHealthResponse(value: unknown): HealthResponse {
  const object = record(value, "health");
  const status = stringValue(object, "status", "health");
  if (status !== "ok") {
    throw new ProtocolError('health.status must be "ok".');
  }
  return {
    status,
    ready: booleanValue(object, "ready", "health"),
    mode: dataMode(object.mode, "health.mode"),
    default_model: stringValue(object, "default_model", "health"),
    artifact_documents: numberValue(object, "artifact_documents", "health"),
  };
}

export function decodeModelsResponse(value: unknown): ModelsResponse {
  const object = record(value, "models");
  return {
    default: stringValue(object, "default", "models"),
    models: stringArray(object, "models", "models"),
  };
}

function curatedQuery(value: unknown, path: string): CuratedQuery {
  const object = record(value, path);
  return {
    query_id: stringValue(object, "query_id", path),
    query: stringValue(object, "query", path),
    has_dense: booleanValue(object, "has_dense", path),
    split: dataMode(object.split, `${path}.split`),
  };
}

export function decodeQueriesResponse(value: unknown): QueriesResponse {
  const object = record(value, "queries");
  return {
    queries: arrayValue(object, "queries", "queries", curatedQuery),
  };
}
