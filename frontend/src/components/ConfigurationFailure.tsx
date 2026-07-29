export function ConfigurationFailure({ message }: { message: string }) {
  return (
    <main className="fatal-error" role="alert">
      <div className="fatal-error__card">
        <p className="eyebrow">Configuration required</p>
        <h1>BetterCallAgent cannot start.</h1>
        <p>{message}</p>
        <p>
          Copy <code>.env.example</code> to <code>.env</code>, select a data source, and restart
          the development server.
        </p>
      </div>
    </main>
  );
}
