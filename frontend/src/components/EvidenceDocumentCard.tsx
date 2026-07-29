import type { EvidenceDocument } from "../domain/models";

interface EvidenceDocumentCardProps {
  document: EvidenceDocument;
}

function formatScore(score: number): string {
  return Number.isInteger(score) ? score.toString() : score.toFixed(3);
}

export function EvidenceDocumentCard({ document }: EvidenceDocumentCardProps) {
  return (
    <article className="document-card">
      <div className="document-card__header">
        <div className="document-card__identity">
          {document.rank !== undefined && <span className="rank-badge">#{document.rank}</span>}
          <code title={document.doc_ref}>{document.doc_ref}</code>
        </div>
        <span className="score-badge">
          {document.score_kind}: {formatScore(document.score)}
        </span>
      </div>

      <dl className="document-card__metadata">
        {document.court !== undefined && (
          <>
            <dt>Court</dt>
            <dd>{document.court}</dd>
          </>
        )}
        {document.docket_number !== undefined && (
          <>
            <dt>Docket</dt>
            <dd>{document.docket_number}</dd>
          </>
        )}
        {document.decision_date !== undefined && (
          <>
            <dt>Date</dt>
            <dd>{document.decision_date}</dd>
          </>
        )}
        {document.confidence !== undefined && (
          <>
            <dt>Confidence</dt>
            <dd>{document.confidence.toFixed(3)}</dd>
          </>
        )}
      </dl>

      <p className="document-card__snippet">{document.snippet}</p>
      {document.rationale_de !== undefined && (
        <blockquote className="document-card__rationale" lang="de">
          {document.rationale_de}
        </blockquote>
      )}
    </article>
  );
}
