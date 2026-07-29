import type {
  CitationValidationData,
  CuratedQuery,
  FinalAnswerStepData,
  HealthResponse,
  ModelsResponse,
  QueriesResponse,
  QueryGenerationData,
  RerankingData,
  RetrievalData,
  RunEvent,
  RunRequest,
  UnderstandingData,
} from "../domain/models";
import type { AgentDataSource } from "./data-source";
import { DataSourceError } from "./data-source";

const FIXTURE_MODEL = "fixture-deterministic";

const FIXTURE_QUERY = {
  query_id: "fixture_detention",
  query:
    "When may Swiss criminal authorities extend pre-trial detention because of a concrete risk of collusion?",
  has_dense: true,
  split: "fixture",
} satisfies CuratedQuery;

const FIXTURE_QUERIES: QueriesResponse = {
  queries: [FIXTURE_QUERY],
};

const understanding: UnderstandingData = {
  kind: "understanding",
  route: "legal",
  restated_question:
    "Under which conditions is an extension of pre-trial detention for risk of collusion lawful?",
  legal_topic: "Swiss criminal procedure — pre-trial detention",
  languages_considered: ["English", "German", "French", "Italian"],
  key_legal_concepts: ["risk of collusion", "proportionality", "detention review"],
};

const queryGeneration: QueryGenerationData = {
  kind: "query_generation",
  search_queries: [
    "Swiss pre-trial detention concrete risk of collusion proportionality",
    "Untersuchungshaft Kollusionsgefahr Verhältnismässigkeit Verlängerung",
  ],
  meta_searchterm_de: "Untersuchungshaft Kollusionsgefahr konkrete Anhaltspunkte",
  keywords: ["Untersuchungshaft", "Kollusionsgefahr", "Verhältnismässigkeit"],
};

const denseDocument = {
  doc_ref: "fixture-court-001",
  court: "Swiss Federal Supreme Court",
  docket_number: "1B_000/2025",
  decision_date: "2025-01-15",
  snippet:
    "The risk of collusion must rest on concrete circumstances and diminish as the investigation progresses.",
  score: 0.91,
  score_kind: "dense",
  rank: 1,
};

const bm25Document = {
  doc_ref: "fixture-law-001",
  snippet:
    "Pre-trial detention may be ordered where there is a serious risk that the accused will influence persons or interfere with evidence.",
  score: 18.42,
  score_kind: "bm25",
  rank: 1,
};

const retrieval: RetrievalData = {
  kind: "retrieval",
  counts: { dense: 10, bm25: 10, hybrid_unique: 17 },
  dense_available: true,
  dense: [denseDocument],
  bm25: [bm25Document],
  hybrid: [denseDocument, bm25Document],
};

const reranking: RerankingData = {
  kind: "reranking",
  model: FIXTURE_MODEL,
  before: [
    { rank: 1, doc_ref: denseDocument.doc_ref },
    { rank: 2, doc_ref: bm25Document.doc_ref },
  ],
  after: [
    { ...denseDocument, score: 9.4, score_kind: "reranker", rank: 1 },
    { ...bm25Document, score: 8.8, score_kind: "reranker", rank: 2 },
  ],
  top_select: 2,
};

const validation: CitationValidationData = {
  kind: "citation_validation",
  rule: "Retain citations supported by the selected evidence documents.",
  qwen_rule: "Evidence vote or high-confidence top-ranked support.",
  bm25_rule: "Require repeated sparse-retrieval support.",
  accepted: [
    {
      citation: "Art. 221 Abs. 1 lit. b StPO",
      type: "law",
      reason: "Directly governs detention based on risk of collusion.",
      votes: "2 supporting documents",
    },
  ],
  rejected: [
    {
      citation: "Art. 221 Abs. 1 lit. a StPO",
      type: "law",
      reason: "Concerns a different detention ground.",
    },
  ],
  predicted_citations: ["Art. 221 Abs. 1 lit. b StPO"],
  bm25_support: {
    top_k: 10,
    min_votes: 2,
    counts: { "Art. 221 Abs. 1 lit. b StPO": 3 },
  },
};

const finalStep: FinalAnswerStepData = {
  kind: "final_answer_step",
  grounded_on: ["Art. 221 Abs. 1 lit. b StPO"],
  document_count: 2,
};

function timestamp(): number {
  return Date.now() / 1000;
}

function abortError(): DOMException {
  return new DOMException("The operation was aborted.", "AbortError");
}

async function delay(milliseconds: number, signal: AbortSignal): Promise<void> {
  if (signal.aborted) {
    throw abortError();
  }
  await new Promise<void>((resolve, reject) => {
    const timer = globalThis.setTimeout(() => {
      signal.removeEventListener("abort", onAbort);
      resolve();
    }, milliseconds);
    const onAbort = (): void => {
      globalThis.clearTimeout(timer);
      reject(abortError());
    };
    signal.addEventListener("abort", onAbort, { once: true });
  });
}

export class FixtureDataSource implements AgentDataSource {
  async getHealth(): Promise<HealthResponse> {
    return {
      status: "ok",
      ready: true,
      mode: "fixture",
      default_model: FIXTURE_MODEL,
      artifact_documents: 2,
    };
  }

  async getModels(): Promise<ModelsResponse> {
    return { default: FIXTURE_MODEL, models: [FIXTURE_MODEL] };
  }

  async getQueries(): Promise<QueriesResponse> {
    return FIXTURE_QUERIES;
  }

  async streamRun(
    request: RunRequest,
    onEvent: (event: RunEvent) => void,
    signal: AbortSignal,
  ): Promise<void> {
    if (request.model !== FIXTURE_MODEL) {
      throw new DataSourceError(`Fixture mode only supports ${FIXTURE_MODEL}.`);
    }
    if (request.query_id !== FIXTURE_QUERY.query_id) {
      throw new DataSourceError(
        `Fixture mode only supports query_id ${FIXTURE_QUERY.query_id}.`,
      );
    }
    if (request.query !== FIXTURE_QUERY.query) {
      throw new DataSourceError(
        "The fixture question must exactly match its curated query_id.",
      );
    }

    const runId = `fixture-${crypto.randomUUID()}`;
    const events: RunEvent[] = [
      {
        type: "run_start",
        ts: timestamp(),
        run_id: runId,
        query: request.query,
        query_id: request.query_id,
        model: request.model,
        mode: "fixture",
      },
      { type: "step_start", ts: timestamp(), run_id: runId, step: 1, name: "Question understanding" },
      {
        type: "step_complete",
        ts: timestamp(),
        run_id: runId,
        step: 1,
        name: "Question understanding",
        summary: "Legal topic and multilingual concepts identified.",
        data: understanding,
      },
      { type: "step_start", ts: timestamp(), run_id: runId, step: 2, name: "Query generation" },
      {
        type: "step_complete",
        ts: timestamp(),
        run_id: runId,
        step: 2,
        name: "Query generation",
        summary: "English and German retrieval queries prepared.",
        data: queryGeneration,
      },
      { type: "step_start", ts: timestamp(), run_id: runId, step: 3, name: "Evidence retrieval" },
      {
        type: "step_complete",
        ts: timestamp(),
        run_id: runId,
        step: 3,
        name: "Evidence retrieval",
        summary: "Dense and sparse evidence merged.",
        data: retrieval,
      },
      { type: "step_start", ts: timestamp(), run_id: runId, step: 4, name: "Candidate reranking" },
      {
        type: "step_complete",
        ts: timestamp(),
        run_id: runId,
        step: 4,
        name: "Candidate reranking",
        summary: "Two evidence documents selected.",
        data: reranking,
      },
      { type: "step_start", ts: timestamp(), run_id: runId, step: 5, name: "Citation validation" },
      {
        type: "step_complete",
        ts: timestamp(),
        run_id: runId,
        step: 5,
        name: "Citation validation",
        summary: "One citation accepted and one rejected.",
        data: validation,
      },
      { type: "step_start", ts: timestamp(), run_id: runId, step: 6, name: "Grounded answer" },
      {
        type: "step_complete",
        ts: timestamp(),
        run_id: runId,
        step: 6,
        name: "Grounded answer",
        summary: "Answer grounded in two evidence documents.",
        data: finalStep,
      },
      {
        type: "final_answer",
        ts: timestamp(),
        run_id: runId,
        markdown:
          "An extension requires a **specific and current risk of collusion**, not an abstract possibility. The authority must identify concrete investigative acts that remain vulnerable to interference and reassess proportionality as the proceedings advance.",
        grounded_on: ["Art. 221 Abs. 1 lit. b StPO"],
      },
      {
        type: "run_complete",
        ts: timestamp(),
        run_id: runId,
        elapsed_s: 1.8,
        usage_total_tokens: 0,
      },
      { type: "stream_end", ts: timestamp(), run_id: runId },
    ];

    for (const event of events) {
      await delay(90, signal);
      onEvent(event);
    }
  }
}
