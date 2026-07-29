import type {
  CitationDecision,
  EvidenceDocument,
  PipelineStepData,
} from "../domain/models";
import { EvidenceDocumentCard } from "./EvidenceDocumentCard";

function ChipList({ values }: { values: string[] }) {
  if (values.length === 0) {
    return <span className="muted">None reported</span>;
  }
  return (
    <ul className="chip-list" aria-label="Reported values">
      {values.map((value) => (
        <li key={value}>{value}</li>
      ))}
    </ul>
  );
}

function DocumentColumn({
  title,
  documents,
  emptyMessage,
}: {
  title: string;
  documents: EvidenceDocument[];
  emptyMessage: string;
}) {
  return (
    <section className="document-column" aria-label={title}>
      <h4>{title}</h4>
      {documents.length === 0 ? (
        <p className="empty-message">{emptyMessage}</p>
      ) : (
        <div className="document-list">
          {documents.map((document, index) => (
            <EvidenceDocumentCard
              key={`${document.doc_ref}-${String(document.rank ?? index)}`}
              document={document}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function CitationList({
  title,
  decisions,
  accepted,
}: {
  title: string;
  decisions: CitationDecision[];
  accepted: boolean;
}) {
  return (
    <section className="citation-column">
      <h4>
        {title} <span className="count-badge">{decisions.length}</span>
      </h4>
      {decisions.length === 0 ? (
        <p className="empty-message">None</p>
      ) : (
        <ul className="citation-list">
          {decisions.map((decision) => (
            <li key={`${accepted ? "accepted" : "rejected"}-${decision.citation}`}>
              <span
                className={`decision-icon decision-icon--${accepted ? "accepted" : "rejected"}`}
                aria-hidden="true"
              >
                {accepted ? "✓" : "×"}
              </span>
              <div>
                <div className="citation-list__title">
                  <code>{decision.citation}</code>
                  <span className="type-badge">{decision.type}</span>
                </div>
                <p>{decision.reason}</p>
                {decision.votes !== undefined && <small>{decision.votes}</small>}
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

export function StepDetails({ data }: { data: PipelineStepData }) {
  switch (data.kind) {
    case "understanding":
      return (
        <div className="step-details">
          <dl className="fact-list">
            <div>
              <dt>Restated question</dt>
              <dd>{data.restated_question}</dd>
            </div>
            <div>
              <dt>Legal topic</dt>
              <dd>{data.legal_topic}</dd>
            </div>
            <div>
              <dt>Route</dt>
              <dd>{data.route}</dd>
            </div>
          </dl>
          <div className="detail-group">
            <h4>Languages considered</h4>
            <ChipList values={data.languages_considered} />
          </div>
          <div className="detail-group">
            <h4>Key legal concepts</h4>
            <ChipList values={data.key_legal_concepts} />
          </div>
        </div>
      );
    case "query_generation":
      return (
        <div className="step-details">
          <div className="detail-group">
            <h4>Search queries</h4>
            <ol className="query-list">
              {data.search_queries.map((query) => (
                <li key={query}>
                  <code>{query}</code>
                </li>
              ))}
            </ol>
          </div>
          <dl className="fact-list">
            <div>
              <dt>German meta-search term</dt>
              <dd>
                <code lang="de">{data.meta_searchterm_de}</code>
              </dd>
            </div>
          </dl>
          <div className="detail-group">
            <h4>Keywords</h4>
            <ChipList values={data.keywords} />
          </div>
        </div>
      );
    case "retrieval":
      return (
        <div className="step-details">
          <dl className="metric-strip">
            <div>
              <dt>Dense</dt>
              <dd>{data.counts.dense}</dd>
            </div>
            <div>
              <dt>BM25</dt>
              <dd>{data.counts.bm25}</dd>
            </div>
            <div>
              <dt>Unique candidates</dt>
              <dd>{data.counts.hybrid_unique}</dd>
            </div>
            <div>
              <dt>Dense available</dt>
              <dd>{data.dense_available ? "Yes" : "No"}</dd>
            </div>
          </dl>
          <div className="retrieval-grid">
            <DocumentColumn
              title="Dense retrieval"
              documents={data.dense}
              emptyMessage="No dense results were returned."
            />
            <DocumentColumn
              title="Sparse retrieval"
              documents={data.bm25}
              emptyMessage="No BM25 results were returned."
            />
            <DocumentColumn
              title="Merged candidates"
              documents={data.hybrid}
              emptyMessage="No merged candidates were returned."
            />
          </div>
        </div>
      );
    case "reranking":
      return (
        <div className="step-details reranking-grid">
          <section>
            <h4>Candidate order before reranking</h4>
            <ol className="before-list">
              {data.before.map((candidate) => (
                <li key={candidate.doc_ref}>
                  <span>#{candidate.rank}</span>
                  <code>{candidate.doc_ref}</code>
                </li>
              ))}
            </ol>
          </section>
          <DocumentColumn
            title={`Reranked by ${data.model} — top ${String(data.top_select)} selected`}
            documents={data.after}
            emptyMessage="The reranker returned no candidates."
          />
        </div>
      );
    case "citation_validation": {
      const support = Object.entries(data.bm25_support.counts).sort(
        ([, left], [, right]) => right - left,
      );
      return (
        <div className="step-details">
          <dl className="fact-list">
            <div>
              <dt>Selection rule</dt>
              <dd>{data.rule}</dd>
            </div>
            <div>
              <dt>LLM evidence rule</dt>
              <dd>{data.qwen_rule}</dd>
            </div>
            <div>
              <dt>BM25 support rule</dt>
              <dd>{data.bm25_rule}</dd>
            </div>
          </dl>
          <div className="support-panel">
            <div>
              <h4>BM25 citation support</h4>
              <p>
                Top {data.bm25_support.top_k}; minimum votes{" "}
                {data.bm25_support.min_votes}
              </p>
            </div>
            {support.length === 0 ? (
              <p className="empty-message">No supported citations reported.</p>
            ) : (
              <ul className="support-list">
                {support.map(([citation, votes]) => (
                  <li key={citation}>
                    <code>{citation}</code>
                    <strong>{votes} votes</strong>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div className="citation-grid">
            <CitationList title="Accepted" decisions={data.accepted} accepted />
            <CitationList title="Rejected" decisions={data.rejected} accepted={false} />
          </div>
        </div>
      );
    }
    case "final_answer_step":
      return (
        <div className="step-details">
          <dl className="metric-strip">
            <div>
              <dt>Evidence documents</dt>
              <dd>{data.document_count}</dd>
            </div>
            <div>
              <dt>Grounded citations</dt>
              <dd>{data.grounded_on.length}</dd>
            </div>
          </dl>
          <div className="detail-group">
            <h4>Citations supplied to answer generation</h4>
            <ChipList values={data.grounded_on} />
          </div>
        </div>
      );
  }
}
