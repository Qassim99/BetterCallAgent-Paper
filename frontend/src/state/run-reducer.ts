import { PIPELINE_STEPS } from "../domain/models";
import type {
  PipelineStepNumber,
  PipelineStepState,
  RunEvent,
  RunRequest,
  RunState,
} from "../domain/models";

export const INITIAL_RUN_STATE: RunState = {
  status: "idle",
  steps: {},
  rawEvents: [],
};

export type RunAction =
  | {
      type: "begin";
      clientRunId: string;
      request: RunRequest;
    }
  | {
      type: "event";
      event: RunEvent;
    }
  | {
      type: "cancel";
    }
  | {
      type: "fail";
      message: string;
    }
  | {
      type: "reset";
    };

function defaultStepName(number: PipelineStepNumber): string {
  return PIPELINE_STEPS.find((step) => step.number === number)?.title ?? `Step ${String(number)}`;
}

function eventBelongsToRun(state: RunState, event: RunEvent): boolean {
  return (
    state.serverRunId === undefined ||
    event.run_id === undefined ||
    state.serverRunId === event.run_id
  );
}

function protocolFailure(state: RunState, message: string, event: RunEvent): RunState {
  return {
    ...state,
    status: "error",
    errorMessage: message,
    rawEvents: [...state.rawEvents, event],
  };
}

export function runReducer(state: RunState, action: RunAction): RunState {
  switch (action.type) {
    case "begin":
      return {
        status: "running",
        clientRunId: action.clientRunId,
        request: action.request,
        steps: {},
        rawEvents: [],
      };
    case "reset":
      return INITIAL_RUN_STATE;
    case "cancel":
      if (state.status !== "running") {
        return state;
      }
      return {
        ...state,
        status: "cancelled",
        errorMessage: undefined,
      };
    case "fail":
      return {
        ...state,
        status: "error",
        errorMessage: action.message,
      };
    case "event":
      break;
  }

  const event = action.event;
  if (!eventBelongsToRun(state, event)) {
    return protocolFailure(
      state,
      "The server mixed events from different run identifiers.",
      event,
    );
  }

  const rawEvents = [...state.rawEvents, event];
  switch (event.type) {
    case "run_start":
      return {
        ...state,
        serverRunId: event.run_id,
        mode: event.mode,
        rawEvents,
      };
    case "step_start": {
      const step: PipelineStepState = {
        number: event.step,
        name: event.name,
        status: "running",
      };
      return {
        ...state,
        steps: { ...state.steps, [event.step]: step },
        rawEvents,
      };
    }
    case "step_complete": {
      const step: PipelineStepState = {
        number: event.step,
        name: event.name,
        status: "completed",
        summary: event.summary,
        data: event.data,
      };
      return {
        ...state,
        steps: { ...state.steps, [event.step]: step },
        rawEvents,
      };
    }
    case "final_answer":
      return {
        ...state,
        answer: event,
        rawEvents,
      };
    case "run_complete":
      return {
        ...state,
        status: "completed",
        elapsedSeconds: event.elapsed_s,
        usageTotalTokens: event.usage_total_tokens,
        rawEvents,
      };
    case "stream_end":
      if (state.status === "running") {
        return {
          ...state,
          status: "error",
          errorMessage: "The stream ended before the run completed.",
          rawEvents,
        };
      }
      return { ...state, rawEvents };
    case "error": {
      const currentStep = event.step === undefined ? undefined : state.steps[event.step];
      const failedStep: PipelineStepState | undefined =
        event.step === undefined
          ? undefined
          : {
              number: event.step,
              name: currentStep?.name ?? defaultStepName(event.step),
              status: "error",
              error: event.message,
            };
      return {
        ...state,
        status: "error",
        errorMessage: event.message,
        steps:
          event.step === undefined || failedStep === undefined
            ? state.steps
            : { ...state.steps, [event.step]: failedStep },
        rawEvents,
      };
    }
  }
}

export function createClientRunId(): string {
  return crypto.randomUUID();
}
