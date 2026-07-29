import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { readAppConfig } from "./api/config";
import type { AppConfig } from "./api/config";
import type { AgentDataSource } from "./api/data-source";
import { FixtureDataSource } from "./api/fixture-data-source";
import { LiveDataSource } from "./api/live-data-source";
import { ConfigurationFailure } from "./components/ConfigurationFailure";
import { ErrorBoundary } from "./components/ErrorBoundary";
import "./styles.css";

function createDataSource(config: AppConfig): AgentDataSource {
  if (config.dataSource === "fixture") {
    return new FixtureDataSource();
  }
  if (config.apiBaseUrl === undefined) {
    throw new Error("Live data source is missing its validated API base URL.");
  }
  return new LiveDataSource({
    apiBaseUrl: config.apiBaseUrl,
    bearerToken: config.bearerToken,
  });
}

const rootElement = document.getElementById("root");
if (rootElement === null) {
  throw new Error("The root application element is missing.");
}

const root = createRoot(rootElement);
try {
  const config = readAppConfig(import.meta.env);
  const dataSource = createDataSource(config);
  root.render(
    <StrictMode>
      <ErrorBoundary>
        <App config={config} dataSource={dataSource} />
      </ErrorBoundary>
    </StrictMode>,
  );
} catch (error: unknown) {
  const message =
    error instanceof Error ? error.message : "The frontend configuration is invalid.";
  root.render(<ConfigurationFailure message={message} />);
}
