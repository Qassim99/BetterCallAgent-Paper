import { describe, expect, it } from "vitest";
import { ConfigurationError, readAppConfig } from "./config";

describe("readAppConfig", () => {
  it("requires an explicit data source", () => {
    expect(() => readAppConfig({})).toThrow(ConfigurationError);
  });

  it("requires an API URL in live mode", () => {
    expect(() => readAppConfig({ VITE_DATA_SOURCE: "live" })).toThrow(
      "VITE_API_BASE_URL is required",
    );
  });

  it("does not require an API URL in fixture mode", () => {
    expect(readAppConfig({ VITE_DATA_SOURCE: "fixture" })).toEqual({
      dataSource: "fixture",
      apiBaseUrl: undefined,
      enableDebug: false,
      bearerToken: undefined,
    });
  });

  it("normalizes a live URL and validates the debug flag", () => {
    expect(
      readAppConfig({
        VITE_DATA_SOURCE: "live",
        VITE_API_BASE_URL: "https://example.test/",
        VITE_ENABLE_DEBUG: "true",
      }),
    ).toEqual({
      dataSource: "live",
      apiBaseUrl: "https://example.test",
      enableDebug: true,
      bearerToken: undefined,
    });
  });
});
