import { useState } from "react";
import { PIPELINE_STEPS } from "../domain/models";
import type { PipelineStepState, RunState } from "../domain/models";
import { FinalAnswerCard } from "./FinalAnswerCard";
import { StepDetails } from "./StepDetails";

function statusLabel(status: PipelineStepState["status"]): string {
  switch (status) {
    case "running":
      return "Running";
    case "completed":
      return "Completed";
    case "error":
      return "Failed";
  }
}

function StepCard({ step, runId }: { step: PipelineStepState; runId: string }) {
  const [expanded, setExpanded] = useState(true);
  const panelId = `run-${runId}-step-${String(step.number)}-details`;

  return (
    <li>
      <article className={`step-card step-card--${step.status}`}>
        <button
          type="button"
          className="step-card__toggle"
          aria-expanded={expanded}
          aria-controls={panelId}
          onClick={() => setExpanded((current) => !current)}
        >
          <span className="step-number" aria-hidden="true">
            {step.number}
          </span>
          <span className="step-card__title">
            <strong>{step.name}</strong>
            {step.summary !== undefined && <small>{step.summary}</small>}
          </span>
          <span className={`status-label status-label--${step.status}`}>
            <span aria-hidden="true" />
            {statusLabel(step.status)}
          </span>
          <span className="chevron" aria-hidden="true">
            {expanded ? "−" : "+"}
          </span>
        </button>

        {expanded && (
          <div id={panelId} className="step-card__body">
            {step.status === "running" && (
              <p className="working-message" role="status">
                Processing this stage…
              </p>
            )}
            {step.status === "error" && (
              <p className="notice notice--error" role="alert">
                {step.error ?? "This pipeline stage failed."}
              </p>
            )}
            {step.data !== undefined && <StepDetails data={step.data} />}
          </div>
        )}
      </article>
    </li>
  );
}

function runStatusMessage(state: RunState): string {
  switch (state.status) {
    case "idle":
      return "Select a curated query and run the pipeline to inspect its operational trace.";
    case "running":
      return "The pipeline is running. Completed stages remain available for inspection.";
    case "completed":
      return "The run completed successfully.";
    case "cancelled":
      return "The run was cancelled.";
    case "error":
      return state.errorMessage ?? "The run failed.";
  }
}

export function PipelineTrace({ state }: { state: RunState }) {
  const steps = PIPELINE_STEPS.flatMap((definition) => {
    const step = state.steps[definition.number];
    return step === undefined ? [] : [step];
  });
  const runId = state.clientRunId ?? "idle";

  return (
    <section className="trace-section" aria-labelledby="trace-heading">
      <div className="section-heading">
        <div>
          <p className="eyebrow">Operational transparency</p>
          <h2 id="trace-heading">Pipeline trace</h2>
        </div>
        {state.status !== "idle" && (
          <span className={`run-status run-status--${state.status}`}>{state.status}</span>
        )}
      </div>

      <p className="trace-intro" role="status" aria-live="polite">
        {runStatusMessage(state)}
      </p>

      {state.status === "error" && (
        <div className="notice notice--error" role="alert">
          {state.errorMessage}
        </div>
      )}

      {steps.length === 0 ? (
        <div className="trace-empty">
          <div className="trace-empty__diagram" aria-hidden="true">
            <span>Question</span>
            <span>→</span>
            <span>Retrieve</span>
            <span>→</span>
            <span>Validate</span>
            <span>→</span>
            <span>Answer</span>
          </div>
          <p>
            The interface shows concrete queries, retrieved evidence, scores, and citation
            decisions. It does not expose hidden chain-of-thought.
          </p>
        </div>
      ) : (
        <ol className="step-list">
          {steps.map((step) => (
            <StepCard key={`${runId}-${String(step.number)}`} step={step} runId={runId} />
          ))}
        </ol>
      )}

      {state.answer !== undefined && <FinalAnswerCard answer={state.answer} />}

      {(state.elapsedSeconds !== undefined || state.usageTotalTokens !== undefined) && (
        <dl className="run-metadata">
          {state.elapsedSeconds !== undefined && (
            <div>
              <dt>Elapsed time</dt>
              <dd>{state.elapsedSeconds.toFixed(2)} seconds</dd>
            </div>
          )}
          {state.usageTotalTokens !== undefined && (
            <div>
              <dt>Reported token usage</dt>
              <dd>{state.usageTotalTokens.toLocaleString()}</dd>
            </div>
          )}
          {state.serverRunId !== undefined && (
            <div>
              <dt>Server run ID</dt>
              <dd>
                <code>{state.serverRunId}</code>
              </dd>
            </div>
          )}
        </dl>
      )}
    </section>
  );
}
