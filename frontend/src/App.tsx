import { useEffect, useReducer, useRef, useState } from "react";
import type { FormEvent } from "react";
import type { AppConfig } from "./api/config";
import type { AgentDataSource } from "./api/data-source";
import { isAbortError, safeErrorMessage } from "./api/data-source";
import { ControlPanel } from "./components/ControlPanel";
import { DebugPanel } from "./components/DebugPanel";
import { PipelineTrace } from "./components/PipelineTrace";
import type {
  HealthResponse,
  ModelsResponse,
  QueriesResponse,
  RunRequest,
} from "./domain/models";
import {
  createClientRunId,
  INITIAL_RUN_STATE,
  runReducer,
} from "./state/run-reducer";

type BootstrapState =
  | { status: "loading" }
  | {
      status: "ready";
      health: HealthResponse;
      models: ModelsResponse;
      queries: QueriesResponse;
    }
  | {
      status: "error";
      message: string;
    };

interface AppProps {
  config: AppConfig;
  dataSource: AgentDataSource;
}

export function App({ config, dataSource }: AppProps) {
  const [bootstrap, setBootstrap] = useState<BootstrapState>({ status: "loading" });
  const [connectionAttempt, setConnectionAttempt] = useState(0);
  const [selectedQueryId, setSelectedQueryId] = useState("");
  const [model, setModel] = useState("");
  const [runState, dispatch] = useReducer(runReducer, INITIAL_RUN_STATE);
  const activeRun = useRef<AbortController | null>(null);

  useEffect(() => {
    const controller = new AbortController();

    Promise.all([
      dataSource.getHealth(controller.signal),
      dataSource.getModels(controller.signal),
      dataSource.getQueries(controller.signal),
    ])
      .then(([health, models, queries]) => {
        if (controller.signal.aborted) {
          return;
        }
        if (!models.models.includes(models.default)) {
          setBootstrap({
            status: "error",
            message: "The API default model is missing from its model list.",
          });
          return;
        }
        const firstQuery = queries.queries[0];
        if (firstQuery === undefined) {
          setBootstrap({
            status: "error",
            message: "The API did not return any curated queries.",
          });
          return;
        }
        setModel(models.default);
        setSelectedQueryId(firstQuery.query_id);
        setBootstrap({ status: "ready", health, models, queries });
      })
      .catch((error: unknown) => {
        if (!isAbortError(error)) {
          setBootstrap({ status: "error", message: safeErrorMessage(error) });
        }
      });

    return () => controller.abort();
  }, [connectionAttempt, dataSource]);

  useEffect(() => {
    return () => activeRun.current?.abort();
  }, []);

  const queries = bootstrap.status === "ready" ? bootstrap.queries.queries : [];
  const models = bootstrap.status === "ready" ? bootstrap.models.models : [];
  const health = bootstrap.status === "ready" ? bootstrap.health : undefined;
  const selectedQuery = queries.find(
    (candidate) => candidate.query_id === selectedQueryId,
  );
  const query = selectedQuery?.query ?? "";

  function handleSelectedQueryChange(queryId: string): void {
    if (queries.some((candidate) => candidate.query_id === queryId)) {
      setSelectedQueryId(queryId);
    }
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (
      bootstrap.status !== "ready" ||
      !bootstrap.health.ready ||
      selectedQuery === undefined ||
      runState.status === "running"
    ) {
      return;
    }

    if (selectedQuery.query.trim() === "" || model === "") {
      return;
    }

    const request: RunRequest = {
      query: selectedQuery.query,
      query_id: selectedQuery.query_id,
      model,
    };
    const clientRunId = createClientRunId();
    const controller = new AbortController();
    activeRun.current = controller;
    dispatch({ type: "begin", clientRunId, request });

    try {
      await dataSource.streamRun(
        request,
        (runEvent) => dispatch({ type: "event", event: runEvent }),
        controller.signal,
      );
    } catch (error: unknown) {
      if (isAbortError(error)) {
        dispatch({ type: "cancel" });
      } else {
        dispatch({ type: "fail", message: safeErrorMessage(error) });
      }
    } finally {
      if (activeRun.current === controller) {
        activeRun.current = null;
      }
    }
  }

  function cancelRun(): void {
    activeRun.current?.abort();
    dispatch({ type: "cancel" });
  }

  function retryConnection(): void {
    setSelectedQueryId("");
    setModel("");
    setBootstrap({ status: "loading" });
    setConnectionAttempt((attempt) => attempt + 1);
  }

  return (
    <div className="app-shell">
      <header className="site-header">
        <div className="site-header__brand">
          <span className="brand-mark" aria-hidden="true">
            B
          </span>
          <div>
            <strong>BetterCallAgent</strong>
            <span>Swiss legal citation retrieval</span>
          </div>
        </div>
        <div className="site-header__context">
          <span>Paper artifact</span>
          <span aria-hidden="true">·</span>
          <span>Operational trace</span>
        </div>
      </header>

      <main className="page">
        <section className="hero" aria-labelledby="page-title">
          <p className="eyebrow">Reproducible legal information retrieval</p>
          <h1 id="page-title">See how evidence becomes a citation-grounded answer.</h1>
          <p>
            BetterCallAgent exposes the pipeline operations reviewers can verify: multilingual
            query construction, retrieval scores, reranking, and deterministic citation
            decisions. Hidden model reasoning is neither requested nor displayed.
          </p>
        </section>

        <ControlPanel
          sourceMode={config.dataSource}
          health={health}
          connectionState={bootstrap.status}
          connectionMessage={bootstrap.status === "error" ? bootstrap.message : undefined}
          queries={queries}
          models={models}
          selectedQueryId={selectedQueryId}
          query={query}
          model={model}
          runStatus={runState.status}
          onSelectedQueryChange={handleSelectedQueryChange}
          onModelChange={setModel}
          onSubmit={(formEvent) => {
            void handleSubmit(formEvent);
          }}
          onCancel={cancelRun}
          onRetryConnection={retryConnection}
        />

        <PipelineTrace state={runState} />
        {config.enableDebug && <DebugPanel events={runState.rawEvents} />}
      </main>

      <footer className="site-footer">
        <p>Research software — outputs are not legal advice.</p>
        <p>Swiss legal sources remain authoritative.</p>
      </footer>
    </div>
  );
}
