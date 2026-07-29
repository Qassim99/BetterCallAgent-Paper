import type { DataMode } from "../domain/models";

interface EnvironmentInput {
  VITE_DATA_SOURCE?: string;
  VITE_API_BASE_URL?: string;
  VITE_ENABLE_DEBUG?: string;
  VITE_API_BEARER_TOKEN?: string;
}

export interface AppConfig {
  dataSource: DataMode;
  apiBaseUrl?: string;
  enableDebug: boolean;
  bearerToken?: string;
}

export class ConfigurationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "ConfigurationError";
  }
}

function optionalTrimmed(value: string | undefined): string | undefined {
  const trimmed = value?.trim();
  return trimmed === "" ? undefined : trimmed;
}

function parseDebugFlag(value: string | undefined): boolean {
  const normalized = optionalTrimmed(value)?.toLowerCase();
  if (normalized === undefined || normalized === "false") {
    return false;
  }
  if (normalized === "true") {
    return true;
  }
  throw new ConfigurationError('VITE_ENABLE_DEBUG must be either "true" or "false".');
}

function normalizeApiBaseUrl(value: string | undefined): string {
  const candidate = optionalTrimmed(value);
  if (candidate === undefined) {
    throw new ConfigurationError(
      "VITE_API_BASE_URL is required when VITE_DATA_SOURCE=live.",
    );
  }

  let parsed: URL;
  try {
    parsed = new URL(candidate);
  } catch {
    throw new ConfigurationError("VITE_API_BASE_URL must be an absolute HTTP(S) URL.");
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new ConfigurationError("VITE_API_BASE_URL must use HTTP or HTTPS.");
  }
  return candidate.replace(/\/+$/, "");
}

export function readAppConfig(environment: EnvironmentInput): AppConfig {
  const source = optionalTrimmed(environment.VITE_DATA_SOURCE);
  if (source !== "live" && source !== "fixture") {
    throw new ConfigurationError('VITE_DATA_SOURCE must be either "live" or "fixture".');
  }

  return {
    dataSource: source,
    apiBaseUrl:
      source === "live" ? normalizeApiBaseUrl(environment.VITE_API_BASE_URL) : undefined,
    enableDebug: parseDebugFlag(environment.VITE_ENABLE_DEBUG),
    bearerToken: optionalTrimmed(environment.VITE_API_BEARER_TOKEN),
  };
}
