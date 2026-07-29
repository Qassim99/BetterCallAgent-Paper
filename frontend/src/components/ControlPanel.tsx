import type { FormEvent } from "react";
import type {
  CuratedQuery,
  DataMode,
  HealthResponse,
  RunStatus,
} from "../domain/models";

interface ControlPanelProps {
  sourceMode: DataMode;
  health?: HealthResponse;
  connectionState: "loading" | "ready" | "error";
  connectionMessage?: string;
  queries: CuratedQuery[];
  models: string[];
  selectedQueryId: string;
  query: string;
  model: string;
  runStatus: RunStatus;
  onSelectedQueryChange: (queryId: string) => void;
  onModelChange: (model: string) => void;
  onSubmit: (event: FormEvent<HTMLFormElement>) => void;
  onCancel: () => void;
  onRetryConnection: () => void;
}

function connectionLabel(
  state: ControlPanelProps["connectionState"],
  health: HealthResponse | undefined,
): string {
  if (state === "loading") {
    return "Checking API";
  }
  if (state === "error") {
    return "API unavailable";
  }
  return health?.ready === true ? "Ready" : "Not ready";
}

export function ControlPanel({
  sourceMode,
  health,
  connectionState,
  connectionMessage,
  queries,
  models,
  selectedQueryId,
  query,
  model,
  runStatus,
  onSelectedQueryChange,
  onModelChange,
  onSubmit,
  onCancel,
  onRetryConnection,
}: ControlPanelProps) {
  const running = runStatus === "running";
  const canRun =
    connectionState === "ready" &&
    health?.ready === true &&
    selectedQueryId.length > 0 &&
    query.trim().length > 0 &&
    model.length > 0 &&
    !running;

  return (
    <section className="control-panel" aria-labelledby="run-heading">
      <div className="control-panel__heading">
        <div>
          <p className="eyebrow">Run configuration</p>
          <h2 id="run-heading">Inspect one retrieval run</h2>
        </div>
        <div
          className={`connection-badge connection-badge--${connectionState}`}
          role="status"
          aria-live="polite"
        >
          <span className="connection-badge__dot" aria-hidden="true" />
          <span>{connectionLabel(connectionState, health)}</span>
          <span className="connection-badge__mode">{sourceMode} source</span>
        </div>
      </div>

      {connectionState === "error" && (
        <div className="notice notice--error" role="alert">
          <div>
            <strong>Could not initialize the data source.</strong>
            <p>{connectionMessage}</p>
          </div>
          <button type="button" className="button button--secondary" onClick={onRetryConnection}>
            Retry
          </button>
        </div>
      )}

      {connectionState === "ready" && health?.ready === false && (
        <div className="notice notice--warning" role="alert">
          The API responded but is not ready to run the pipeline.
        </div>
      )}

      <form onSubmit={onSubmit}>
        <div className="control-grid">
          <div className="field field--wide">
            <label htmlFor="curated-query">Curated reproduction query</label>
            <select
              id="curated-query"
              value={selectedQueryId}
              disabled={connectionState !== "ready" || running}
              onChange={(event) => onSelectedQueryChange(event.target.value)}
            >
              {queries.length === 0 && <option value="">No curated queries available</option>}
              {queries.map((option) => (
                <option key={option.query_id} value={option.query_id}>
                  {option.query_id} — {option.query}
                </option>
              ))}
            </select>
            <p className="field__help">
              Every run sends the selected question together with its explicit{" "}
              <code>query_id</code>.
            </p>
          </div>

          <div className="field">
            <label htmlFor="model">Model</label>
            <select
              id="model"
              value={model}
              disabled={models.length === 0 || running}
              onChange={(event) => onModelChange(event.target.value)}
            >
              {models.length === 0 && <option value="">No models available</option>}
              {models.map((modelId) => (
                <option key={modelId} value={modelId}>
                  {modelId}
                </option>
              ))}
            </select>
            <p className="field__help">Available models are reported by the selected data source.</p>
          </div>

          <div className="field field--full">
            <label htmlFor="legal-question">Curated legal question (read-only)</label>
            <textarea id="legal-question" rows={4} value={query} readOnly />
            <div className="field__meta">
              <span>
                Select a curated query above; retrieval may expand it into German, French,
                and Italian.
              </span>
              <span>{query.length.toLocaleString()} characters</span>
            </div>
          </div>
        </div>

        <div className="control-panel__actions">
          <button type="submit" className="button button--primary" disabled={!canRun}>
            Run pipeline
          </button>
          {running && (
            <button type="button" className="button button--danger" onClick={onCancel}>
              Stop run
            </button>
          )}
          {health !== undefined && (
            <p className="control-panel__corpus">
              {health.artifact_documents.toLocaleString()} artifact documents
            </p>
          )}
        </div>
      </form>
    </section>
  );
}
