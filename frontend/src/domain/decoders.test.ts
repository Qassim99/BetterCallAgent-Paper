import { describe, expect, it } from "vitest";
import { decodeHealthResponse } from "./decoders";

describe("decodeHealthResponse", () => {
  it("decodes the artifact document count", () => {
    expect(
      decodeHealthResponse({
        status: "ok",
        ready: true,
        mode: "live",
        default_model: "model-1",
        artifact_documents: 42,
      }),
    ).toEqual({
      status: "ok",
      ready: true,
      mode: "live",
      default_model: "model-1",
      artifact_documents: 42,
    });
  });

  it("rejects the retired corpus_documents field", () => {
    expect(() =>
      decodeHealthResponse({
        status: "ok",
        ready: true,
        mode: "live",
        default_model: "model-1",
        corpus_documents: 42,
      }),
    ).toThrow("health.artifact_documents must be a finite number");
  });
});
