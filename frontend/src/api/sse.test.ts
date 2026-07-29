import { describe, expect, it } from "vitest";
import { consumeSseStream, SseParser } from "./sse";

describe("SseParser", () => {
  it("parses LF-delimited multiline data", () => {
    const parser = new SseParser();
    const messages = parser.push(
      "event: update\ndata: first line\ndata: second line\nid: run-1\n\n",
    );

    expect(messages).toEqual([
      {
        data: "first line\nsecond line",
        event: "update",
        id: "run-1",
        retry: undefined,
      },
    ]);
  });

  it("parses CRLF when every character arrives separately", () => {
    const parser = new SseParser();
    const messages = Array.from('data: {"type":"stream_end"}\r\n\r\n').flatMap((character) =>
      parser.push(character),
    );

    expect(messages).toHaveLength(1);
    expect(messages[0]?.data).toBe('{"type":"stream_end"}');
  });

  it("flushes a final event without a trailing blank line", () => {
    const parser = new SseParser();
    expect(parser.push("data: final payload")).toEqual([]);
    expect(parser.finish()).toEqual([
      {
        data: "final payload",
        event: undefined,
        id: undefined,
        retry: undefined,
      },
    ]);
  });

  it("ignores comments and unknown fields", () => {
    const parser = new SseParser();
    const messages = parser.push(": keepalive\nunknown: value\ndata: ok\n\n");
    expect(messages[0]?.data).toBe("ok");
  });
});

describe("consumeSseStream", () => {
  it("preserves UTF-8 characters split across byte chunks and performs final flush", async () => {
    const bytes = new TextEncoder().encode("data: Grüezi 🇨🇭");
    const chunks = Array.from(bytes, (byte) => new Uint8Array([byte]));
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) {
          controller.enqueue(chunk);
        }
        controller.close();
      },
    });
    const data: string[] = [];

    await consumeSseStream(stream, (message) => data.push(message.data));

    expect(data).toEqual(["Grüezi 🇨🇭"]);
  });
});
