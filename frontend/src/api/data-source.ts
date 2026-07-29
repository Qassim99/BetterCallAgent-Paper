import type {
  HealthResponse,
  ModelsResponse,
  QueriesResponse,
  RunEvent,
  RunRequest,
} from "../domain/models";

export interface AgentDataSource {
  getHealth(signal?: AbortSignal): Promise<HealthResponse>;
  getModels(signal?: AbortSignal): Promise<ModelsResponse>;
  getQueries(signal?: AbortSignal): Promise<QueriesResponse>;
  streamRun(
    request: RunRequest,
    onEvent: (event: RunEvent) => void,
    signal: AbortSignal,
  ): Promise<void>;
}

export class DataSourceError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DataSourceError";
  }
}

export function isAbortError(error: unknown): boolean {
  return error instanceof DOMException && error.name === "AbortError";
}

export function safeErrorMessage(error: unknown): string {
  if (error instanceof DataSourceError) {
    return error.message;
  }
  if (error instanceof Error && error.name === "ProtocolError") {
    return "The API returned data that does not match the documented frontend contract.";
  }
  return "The request failed unexpectedly. Verify the API configuration and try again.";
}
