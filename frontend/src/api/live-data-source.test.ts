import { afterEach, describe, expect, it, vi } from "vitest";
import type { RunEvent, RunRequest } from "../domain/models";
import { LiveDataSource } from "./live-data-source";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("LiveDataSource", () => {
  it("posts the documented request and decodes a fragmented SSE response", async () => {
    const encoded = new TextEncoder().encode(
      'data: {"type":"stream_end","ts":1,"run_id":"server-1"}\r\n\r\n',
    );
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(encoded.slice(0, 7));
        controller.enqueue(encoded.slice(7, 29));
        controller.enqueue(encoded.slice(29));
        controller.close();
      },
    });
    const fetchMock = vi.fn(
      async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
        void input;
        void init;
        return new Response(stream, {
          status: 200,
          headers: { "Content-Type": "text/event-stream; charset=utf-8" },
        });
      },
    );
    vi.stubGlobal("fetch", fetchMock);

    const source = new LiveDataSource({
      apiBaseUrl: "https://api.example.test",
      bearerToken: "development-token",
    });
    const request: RunRequest = {
      query: "Which Swiss rule applies?",
      query_id: "val_001",
      model: "model-1",
    };
    const events: RunEvent[] = [];

    await source.streamRun(
      request,
      (event) => events.push(event),
      new AbortController().signal,
    );

    expect(events).toEqual([
      { type: "stream_end", ts: 1, run_id: "server-1" },
    ]);
    expect(fetchMock).toHaveBeenCalledOnce();
    const call = fetchMock.mock.calls[0];
    expect(call?.[0]).toBe("https://api.example.test/api/runs/stream");
    expect(call?.[1]?.method).toBe("POST");
    expect(call?.[1]?.body).toBe(JSON.stringify(request));
    expect(new Headers(call?.[1]?.headers).get("Authorization")).toBe(
      "Bearer development-token",
    );
  });
});
