import {
  decodeHealthResponse,
  decodeModelsResponse,
  decodeQueriesResponse,
  decodeRunEventJson,
} from "../domain/decoders";
import type {
  HealthResponse,
  ModelsResponse,
  QueriesResponse,
  RunEvent,
  RunRequest,
} from "../domain/models";
import type { AgentDataSource } from "./data-source";
import { DataSourceError } from "./data-source";
import { consumeSseStream } from "./sse";

interface LiveDataSourceOptions {
  apiBaseUrl: string;
  bearerToken?: string;
}

export class LiveDataSource implements AgentDataSource {
  private readonly apiBaseUrl: string;
  private readonly bearerToken: string | undefined;

  constructor(options: LiveDataSourceOptions) {
    this.apiBaseUrl = options.apiBaseUrl;
    this.bearerToken = options.bearerToken;
  }

  getHealth(signal?: AbortSignal): Promise<HealthResponse> {
    return this.getJson("/api/health", decodeHealthResponse, signal);
  }

  getModels(signal?: AbortSignal): Promise<ModelsResponse> {
    return this.getJson("/api/models", decodeModelsResponse, signal);
  }

  getQueries(signal?: AbortSignal): Promise<QueriesResponse> {
    return this.getJson("/api/queries", decodeQueriesResponse, signal);
  }

  async streamRun(
    request: RunRequest,
    onEvent: (event: RunEvent) => void,
    signal: AbortSignal,
  ): Promise<void> {
    const headers = this.headers();
    headers.set("Accept", "text/event-stream");
    headers.set("Content-Type", "application/json");

    const response = await fetch(`${this.apiBaseUrl}/api/runs/stream`, {
      method: "POST",
      headers,
      signal,
      body: JSON.stringify(request),
    });
    if (!response.ok) {
      throw new DataSourceError(`The API rejected the run request (HTTP ${String(response.status)}).`);
    }
    const contentType = response.headers.get("content-type") ?? "";
    if (!contentType.toLowerCase().includes("text/event-stream")) {
      throw new DataSourceError("The run endpoint did not return an SSE response.");
    }
    if (response.body === null) {
      throw new DataSourceError("The run endpoint returned an empty response body.");
    }

    await consumeSseStream(
      response.body,
      (message) => {
        onEvent(decodeRunEventJson(message.data));
      },
      signal,
    );
  }

  private async getJson<T>(
    path: string,
    decode: (value: unknown) => T,
    signal?: AbortSignal,
  ): Promise<T> {
    const response = await fetch(`${this.apiBaseUrl}${path}`, {
      headers: this.headers(),
      signal,
    });
    if (!response.ok) {
      throw new DataSourceError(`${path} failed (HTTP ${String(response.status)}).`);
    }

    let value: unknown;
    try {
      value = await response.json();
    } catch {
      throw new DataSourceError(`${path} did not return valid JSON.`);
    }
    return decode(value);
  }

  private headers(): Headers {
    const headers = new Headers();
    if (this.bearerToken !== undefined) {
      headers.set("Authorization", `Bearer ${this.bearerToken}`);
    }
    return headers;
  }
}
