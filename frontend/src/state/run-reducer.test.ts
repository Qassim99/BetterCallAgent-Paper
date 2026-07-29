import { describe, expect, it } from "vitest";
import type { RunEvent, RunRequest } from "../domain/models";
import { createClientRunId, INITIAL_RUN_STATE, runReducer } from "./run-reducer";

const request: RunRequest = {
  query: "What is the applicable law?",
  query_id: "fixture_001",
  model: "fixture-deterministic",
};

function eventSequence(): RunEvent[] {
  const common = { ts: 1, run_id: "server-run-1" };
  return [
    {
      ...common,
      type: "run_start",
      query: request.query,
      query_id: request.query_id ?? null,
      model: request.model,
      mode: "fixture",
    },
    { ...common, type: "step_start", step: 1, name: "Question understanding" },
    {
      ...common,
      type: "step_complete",
      step: 1,
      name: "Question understanding",
      summary: "Understood.",
      data: {
        kind: "understanding",
        route: "legal",
        restated_question: request.query,
        legal_topic: "Test",
        languages_considered: ["English", "German"],
        key_legal_concepts: ["test"],
      },
    },
    {
      ...common,
      type: "final_answer",
      markdown: "Answer",
      grounded_on: ["Art. 1 ZGB"],
    },
    {
      ...common,
      type: "run_complete",
      elapsed_s: 2.5,
      usage_total_tokens: 42,
    },
    { ...common, type: "stream_end" },
  ];
}

describe("runReducer", () => {
  it("reduces a complete run without losing metadata", () => {
    let state = runReducer(INITIAL_RUN_STATE, {
      type: "begin",
      clientRunId: "client-run-1",
      request,
    });
    for (const event of eventSequence()) {
      state = runReducer(state, { type: "event", event });
    }

    expect(state.status).toBe("completed");
    expect(state.serverRunId).toBe("server-run-1");
    expect(state.steps[1]?.status).toBe("completed");
    expect(state.answer?.markdown).toBe("Answer");
    expect(state.elapsedSeconds).toBe(2.5);
    expect(state.usageTotalTokens).toBe(42);
    expect(state.rawEvents).toHaveLength(6);
  });

  it("marks an incomplete stream as an error", () => {
    let state = runReducer(INITIAL_RUN_STATE, {
      type: "begin",
      clientRunId: "client-run-1",
      request,
    });
    state = runReducer(state, {
      type: "event",
      event: { type: "stream_end", ts: 1, run_id: "server-run-1" },
    });

    expect(state.status).toBe("error");
    expect(state.errorMessage).toContain("before the run completed");
  });

  it("rejects events from a different server run", () => {
    let state = runReducer(INITIAL_RUN_STATE, {
      type: "begin",
      clientRunId: "client-run-1",
      request,
    });
    state = runReducer(state, {
      type: "event",
      event: eventSequence()[0]!,
    });
    state = runReducer(state, {
      type: "event",
      event: {
        type: "step_start",
        ts: 2,
        run_id: "different-run",
        step: 1,
        name: "Question understanding",
      },
    });

    expect(state.status).toBe("error");
    expect(state.errorMessage).toContain("different run identifiers");
  });

  it("creates a unique client identifier for every run", () => {
    expect(createClientRunId()).not.toBe(createClientRunId());
  });
});
