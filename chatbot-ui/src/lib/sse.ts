/**
 * Typed SSE client for POST /api/chat.
 *
 * The browser's EventSource only does GET, so the stream is read off a
 * fetch() body and parsed by hand. Events are separated by a blank line
 * and may arrive split across network chunks, so a partial trailing block
 * is buffered until the next read.
 */

export type SourceChunk = {
  chunk_id: number;
  doc_id: number;
  score: number;
  section_title: string | null;
};

export type UsagePayload = {
  model: string;
  input_tokens: number;
  output_tokens: number;
  cache_read_input_tokens: number;
  cache_creation_input_tokens: number;
  total_tokens: number;
  cost_usd: number | null;
  stop_reason: string | null;
};

export type ChatEvent =
  | { type: "sources"; chunks: SourceChunk[] }
  | { type: "token"; text: string }
  | { type: "usage"; usage: UsagePayload }
  | { type: "done" }
  | { type: "error"; message: string };

export type HistoryMessage = { role: "user" | "assistant"; content: string };

export type ChatRequest = {
  question: string;
  history?: HistoryMessage[];
  session_id?: string;
};

/** Parse one "event: …\ndata: …" block into a typed event, or null. */
function parseBlock(block: string): ChatEvent | null {
  let event: string | null = null;
  let data: string | null = null;
  for (const line of block.split("\n")) {
    if (line.startsWith("event:")) event = line.slice(6).trim();
    else if (line.startsWith("data:")) data = line.slice(5).trim();
  }
  if (!event || data === null) return null;

  let payload: unknown;
  try {
    payload = JSON.parse(data);
  } catch {
    return null;
  }
  const p = payload as Record<string, unknown>;

  switch (event) {
    case "sources":
      return { type: "sources", chunks: p.chunks as SourceChunk[] };
    case "token":
      return { type: "token", text: String(p.text ?? "") };
    case "usage":
      return { type: "usage", usage: payload as UsagePayload };
    case "done":
      return { type: "done" };
    case "error":
      return { type: "error", message: String(p.message ?? "unknown") };
    default:
      return null;
  }
}

/**
 * POST the request and yield typed events as they arrive.
 * Abort via the signal to stop generation server-side too.
 */
export async function* streamChat(
  baseUrl: string,
  body: ChatRequest,
  signal?: AbortSignal,
): AsyncGenerator<ChatEvent> {
  const res = await fetch(`${baseUrl}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });

  if (!res.ok || !res.body) {
    throw new Error(`chat request failed: ${res.status} ${res.statusText}`);
  }

  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    // SSE allows CRLF line endings and sse-starlette emits them, so
    // normalise first — otherwise "\n\n" never matches and the whole
    // stream collapses into a single block.
    buffer += value.replace(/\r\n/g, "\n");

    const blocks = buffer.split("\n\n");
    // The last piece may be an incomplete event — keep it for the next read.
    buffer = blocks.pop() ?? "";

    for (const block of blocks) {
      const ev = parseBlock(block);
      if (ev) yield ev;
    }
  }

  // Flush anything left once the stream closes cleanly.
  const tail = parseBlock(buffer);
  if (tail) yield tail;
}
