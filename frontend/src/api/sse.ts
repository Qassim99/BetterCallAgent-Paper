export interface SseMessage {
  data: string;
  event?: string;
  id?: string;
  retry?: number;
}

/**
 * Incremental parser for the event-stream wire format.
 *
 * It accepts arbitrary text fragments, including fragments that split CRLF
 * delimiters or field names. A final event is dispatched by finish() even when
 * the server closes without a trailing blank line.
 */
export class SseParser {
  private buffer = "";
  private dataLines: string[] = [];
  private eventType: string | undefined;
  private lastEventId: string | undefined;
  private retry: number | undefined;

  push(chunk: string): SseMessage[] {
    this.buffer += chunk;
    const messages: SseMessage[] = [];

    while (true) {
      const lineBreak = this.findLineBreak();
      if (lineBreak === undefined) {
        break;
      }
      const line = this.buffer.slice(0, lineBreak.index);
      this.buffer = this.buffer.slice(lineBreak.index + lineBreak.width);
      this.processLine(line, messages);
    }

    return messages;
  }

  finish(): SseMessage[] {
    const messages: SseMessage[] = [];

    if (this.buffer.endsWith("\r")) {
      this.processLine(this.buffer.slice(0, -1), messages);
    } else if (this.buffer.length > 0) {
      this.processLine(this.buffer, messages);
    }
    this.buffer = "";

    const finalMessage = this.dispatch();
    if (finalMessage !== undefined) {
      messages.push(finalMessage);
    }
    return messages;
  }

  private findLineBreak(): { index: number; width: number } | undefined {
    for (let index = 0; index < this.buffer.length; index += 1) {
      const character = this.buffer[index];
      if (character === "\n") {
        return { index, width: 1 };
      }
      if (character === "\r") {
        if (index === this.buffer.length - 1) {
          return undefined;
        }
        return {
          index,
          width: this.buffer[index + 1] === "\n" ? 2 : 1,
        };
      }
    }
    return undefined;
  }

  private processLine(line: string, messages: SseMessage[]): void {
    if (line === "") {
      const message = this.dispatch();
      if (message !== undefined) {
        messages.push(message);
      }
      return;
    }

    if (line.startsWith(":")) {
      return;
    }

    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) {
      value = value.slice(1);
    }

    switch (field) {
      case "data":
        this.dataLines.push(value);
        break;
      case "event":
        this.eventType = value;
        break;
      case "id":
        if (!value.includes("\0")) {
          this.lastEventId = value;
        }
        break;
      case "retry":
        if (/^\d+$/.test(value)) {
          this.retry = Number(value);
        }
        break;
      default:
        break;
    }
  }

  private dispatch(): SseMessage | undefined {
    if (this.dataLines.length === 0) {
      this.eventType = undefined;
      this.retry = undefined;
      return undefined;
    }

    const message: SseMessage = {
      data: this.dataLines.join("\n"),
      event: this.eventType,
      id: this.lastEventId,
      retry: this.retry,
    };
    this.dataLines = [];
    this.eventType = undefined;
    this.retry = undefined;
    return message;
  }
}

function abortError(): DOMException {
  return new DOMException("The operation was aborted.", "AbortError");
}

export async function consumeSseStream(
  stream: ReadableStream<Uint8Array>,
  onMessage: (message: SseMessage) => void,
  signal?: AbortSignal,
): Promise<void> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  const parser = new SseParser();

  try {
    while (true) {
      if (signal?.aborted === true) {
        throw abortError();
      }
      const result = await reader.read();
      if (result.done) {
        break;
      }
      const text = decoder.decode(result.value, { stream: true });
      for (const message of parser.push(text)) {
        onMessage(message);
      }
    }

    const finalText = decoder.decode();
    for (const message of parser.push(finalText)) {
      onMessage(message);
    }
    for (const message of parser.finish()) {
      onMessage(message);
    }
  } finally {
    reader.releaseLock();
  }
}
