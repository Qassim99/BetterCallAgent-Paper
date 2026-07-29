import { describe, expect, it, vi } from "vitest";
import type { RunRequest } from "../domain/models";
import { FixtureDataSource } from "./fixture-data-source";

async function fixtureRequest(source: FixtureDataSource): Promise<RunRequest> {
  const models = await source.getModels();
  const queries = await source.getQueries();
  const query = queries.queries[0];
  if (query === undefined) {
    throw new Error("Fixture query is missing.");
  }
  return {
    query: query.query,
    query_id: query.query_id,
    model: models.default,
  };
}

describe("FixtureDataSource", () => {
  it("exposes one query matching its detention trace", async () => {
    const source = new FixtureDataSource();

    await expect(source.getHealth()).resolves.toEqual({
      status: "ok",
      ready: true,
      mode: "fixture",
      default_model: "fixture-deterministic",
      artifact_documents: 2,
    });
    await expect(source.getQueries()).resolves.toEqual({
      queries: [
        {
          query_id: "fixture_detention",
          query:
            "When may Swiss criminal authorities extend pre-trial detention because of a concrete risk of collusion?",
          has_dense: true,
          split: "fixture",
        },
      ],
    });
  });

  it("rejects a query_id that does not match the fixture trace", async () => {
    const source = new FixtureDataSource();
    const request = await fixtureRequest(source);

    await expect(
      source.streamRun(
        { ...request, query_id: "fixture_contract" },
        vi.fn(),
        new AbortController().signal,
      ),
    ).rejects.toThrow("only supports query_id fixture_detention");
  });

  it("rejects any query text that is not the exact curated question", async () => {
    const source = new FixtureDataSource();
    const request = await fixtureRequest(source);

    await expect(
      source.streamRun(
        { ...request, query: `${request.query} ` },
        vi.fn(),
        new AbortController().signal,
      ),
    ).rejects.toThrow("must exactly match its curated query_id");
  });
});
