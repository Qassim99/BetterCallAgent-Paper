import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { FinalAnswerEvent } from "../domain/models";

export function FinalAnswerCard({ answer }: { answer: FinalAnswerEvent }) {
  return (
    <article className="final-answer" aria-labelledby="final-answer-heading">
      <div className="final-answer__header">
        <div>
          <p className="eyebrow">Grounded output</p>
          <h3 id="final-answer-heading">Final answer</h3>
        </div>
        <span className="citation-count">
          {answer.grounded_on.length} grounded{" "}
          {answer.grounded_on.length === 1 ? "citation" : "citations"}
        </span>
      </div>

      {answer.metrics !== undefined && (
        <aside className="evaluation-metrics" aria-label="Optional evaluation metrics">
          <div>
            <strong>Evaluation-only metrics</strong>
            <p>Displayed because the backend explicitly returned them for this run.</p>
          </div>
          <dl>
            <div>
              <dt>F1</dt>
              <dd>{answer.metrics.f1.toFixed(3)}</dd>
            </div>
            <div>
              <dt>Precision</dt>
              <dd>{answer.metrics.precision.toFixed(3)}</dd>
            </div>
            <div>
              <dt>Recall</dt>
              <dd>{answer.metrics.recall.toFixed(3)}</dd>
            </div>
          </dl>
        </aside>
      )}

      <div className="markdown">
        <ReactMarkdown remarkPlugins={[remarkGfm]}>{answer.markdown}</ReactMarkdown>
      </div>

      {answer.grounded_on.length > 0 && (
        <div className="grounded-citations">
          <h4>Grounded on</h4>
          <ul>
            {answer.grounded_on.map((citation) => (
              <li key={citation}>
                <code>{citation}</code>
              </li>
            ))}
          </ul>
        </div>
      )}
    </article>
  );
}
