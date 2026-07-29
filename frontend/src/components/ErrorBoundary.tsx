import { Component } from "react";
import type { ReactNode } from "react";

interface ErrorBoundaryProps {
  children: ReactNode;
}

interface ErrorBoundaryState {
  failed: boolean;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  state: ErrorBoundaryState = { failed: false };

  static getDerivedStateFromError(): ErrorBoundaryState {
    return { failed: true };
  }

  componentDidCatch(): void {
    // Error details are intentionally not rendered or persisted by the paper UI.
  }

  render(): ReactNode {
    if (this.state.failed) {
      return (
        <main className="fatal-error" role="alert">
          <div className="fatal-error__card">
            <p className="eyebrow">BetterCallAgent</p>
            <h1>The interface could not be rendered.</h1>
            <p>
              Reload the page. If the problem continues, run <code>npm run check</code> and
              inspect the browser developer console locally.
            </p>
            <button type="button" onClick={() => window.location.reload()}>
              Reload interface
            </button>
          </div>
        </main>
      );
    }
    return this.props.children;
  }
}
