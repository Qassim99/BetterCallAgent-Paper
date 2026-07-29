export type DataMode = "live" | "fixture";
export type PipelineStepNumber = 1 | 2 | 3 | 4 | 5 | 6;
export type RunStatus = "idle" | "running" | "completed" | "error" | "cancelled";
export type StepStatus = "running" | "completed" | "error";

export const PIPELINE_STEPS: ReadonlyArray<{
  number: PipelineStepNumber;
  title: string;
}> = [
  { number: 1, title: "Question understanding" },
  { number: 2, title: "Query generation" },
  { number: 3, title: "Evidence retrieval" },
  { number: 4, title: "Candidate reranking" },
  { number: 5, title: "Citation validation" },
  { number: 6, title: "Grounded answer" },
];

export interface HealthResponse {
  status: "ok";
  ready: boolean;
  mode: DataMode;
  default_model: string;
  artifact_documents: number;
}

export interface ModelsResponse {
  default: string;
  models: string[];
}

export interface CuratedQuery {
  query_id: string;
  query: string;
  has_dense: boolean;
  split: DataMode;
}

export interface QueriesResponse {
  queries: CuratedQuery[];
}

export interface ConversationTurn {
  role: "user" | "assistant";
  content: string;
}

export interface RunRequest {
  query: string;
  query_id: string;
  model: string;
  history?: ConversationTurn[];
}

export interface EvidenceDocument {
  doc_ref: string;
  court?: string;
  decision_id?: string;
  docket_number?: string;
  decision_date?: string;
  snippet: string;
  score: number;
  score_kind: string;
  rank?: number;
  confidence?: number;
  rationale_de?: string;
}

export interface CitationDecision {
  citation: string;
  type: "law" | "court";
  reason: string;
  votes?: string;
}

export interface UnderstandingData {
  kind: "understanding";
  route: "legal";
  restated_question: string;
  legal_topic: string;
  languages_considered: string[];
  key_legal_concepts: string[];
}

export interface QueryGenerationData {
  kind: "query_generation";
  search_queries: string[];
  meta_searchterm_de: string;
  keywords: string[];
}

export interface RetrievalData {
  kind: "retrieval";
  counts: {
    dense: number;
    bm25: number;
    hybrid_unique: number;
  };
  dense_available: boolean;
  dense: EvidenceDocument[];
  bm25: EvidenceDocument[];
  hybrid: EvidenceDocument[];
}

export interface RerankingData {
  kind: "reranking";
  model: string;
  before: Array<{
    rank: number;
    doc_ref: string;
  }>;
  after: EvidenceDocument[];
  top_select: number;
}

export interface CitationValidationData {
  kind: "citation_validation";
  rule: string;
  qwen_rule: string;
  bm25_rule: string;
  accepted: CitationDecision[];
  rejected: CitationDecision[];
  predicted_citations: string[];
  bm25_support: {
    top_k: number;
    min_votes: number;
    counts: Record<string, number>;
  };
}

export interface FinalAnswerStepData {
  kind: "final_answer_step";
  grounded_on: string[];
  document_count: number;
}

export interface StepDataByNumber {
  1: UnderstandingData;
  2: QueryGenerationData;
  3: RetrievalData;
  4: RerankingData;
  5: CitationValidationData;
  6: FinalAnswerStepData;
}

export type PipelineStepData = StepDataByNumber[PipelineStepNumber];

interface BaseRunEvent {
  ts: number;
  run_id?: string;
}

export interface RunStartEvent extends BaseRunEvent {
  type: "run_start";
  query: string;
  query_id: string | null;
  model: string;
  mode: DataMode;
}

export interface StepStartEvent extends BaseRunEvent {
  type: "step_start";
  step: PipelineStepNumber;
  name: string;
}

export type StepCompleteEvent = {
  [Step in PipelineStepNumber]: BaseRunEvent & {
    type: "step_complete";
    step: Step;
    name: string;
    summary: string;
    data: StepDataByNumber[Step];
  };
}[PipelineStepNumber];

export interface EvaluationMetrics {
  precision: number;
  recall: number;
  f1: number;
  true_positives?: number;
  false_positives?: number;
  false_negatives?: number;
}

export interface FinalAnswerEvent extends BaseRunEvent {
  type: "final_answer";
  markdown: string;
  grounded_on: string[];
  metrics?: EvaluationMetrics;
}

export interface RunCompleteEvent extends BaseRunEvent {
  type: "run_complete";
  elapsed_s: number;
  usage_total_tokens: number;
}

export interface StreamEndEvent extends BaseRunEvent {
  type: "stream_end";
}

export interface RunErrorEvent extends BaseRunEvent {
  type: "error";
  code: "invalid_request" | "unauthorized" | "pipeline_error";
  message: string;
  step?: PipelineStepNumber;
}

export type RunEvent =
  | RunStartEvent
  | StepStartEvent
  | StepCompleteEvent
  | FinalAnswerEvent
  | RunCompleteEvent
  | StreamEndEvent
  | RunErrorEvent;

export interface PipelineStepState {
  number: PipelineStepNumber;
  name: string;
  status: StepStatus;
  summary?: string;
  data?: PipelineStepData;
  error?: string;
}

export interface RunState {
  status: RunStatus;
  clientRunId?: string;
  serverRunId?: string;
  request?: RunRequest;
  mode?: DataMode;
  steps: Partial<Record<PipelineStepNumber, PipelineStepState>>;
  answer?: FinalAnswerEvent;
  elapsedSeconds?: number;
  usageTotalTokens?: number;
  errorMessage?: string;
  rawEvents: RunEvent[];
}
