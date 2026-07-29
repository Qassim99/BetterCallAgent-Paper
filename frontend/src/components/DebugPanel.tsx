import type { RunEvent } from "../domain/models";

export function DebugPanel({ events }: { events: RunEvent[] }) {
  if (events.length === 0) {
    return null;
  }
  return (
    <details className="debug-panel">
      <summary>Raw normalized events ({events.length})</summary>
      <p>
        Debug output is enabled explicitly through <code>VITE_ENABLE_DEBUG=true</code>.
      </p>
      <pre>{events.map((event) => JSON.stringify(event)).join("\n")}</pre>
    </details>
  );
}
