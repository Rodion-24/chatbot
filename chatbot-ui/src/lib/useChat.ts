import { useCallback, useRef, useState } from "react";
import {
  streamChat,
  type HistoryMessage,
  type SourceChunk,
  type UsagePayload,
} from "./sse";

export type ChatMessage = HistoryMessage & {
  id: string;
  sources?: SourceChunk[];
  usage?: UsagePayload;
  error?: string;
  streaming?: boolean;
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://localhost:8000";

/** Holds the conversation and drives one streaming request at a time. */
export function useChat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const abortRef = useRef<AbortController | null>(null);
  // Stable per-tab id so Langfuse can group the whole conversation.
  const sessionId = useRef(crypto.randomUUID()).current;

  const patchLast = useCallback(
    (patch: Partial<ChatMessage> | ((m: ChatMessage) => ChatMessage)) => {
      setMessages((prev) => {
        if (prev.length === 0) return prev;
        const last = prev[prev.length - 1];
        const next = typeof patch === "function" ? patch(last) : { ...last, ...patch };
        return [...prev.slice(0, -1), next];
      });
    },
    [],
  );

  const send = useCallback(
    async (question: string) => {
      const q = question.trim();
      if (!q || loading) return;

      // History is everything before this turn; the API appends the question.
      const history: HistoryMessage[] = messages
        .filter((m) => !m.error)
        .map(({ role, content }) => ({ role, content }));

      setMessages((prev) => [
        ...prev,
        { id: crypto.randomUUID(), role: "user", content: q },
        { id: crypto.randomUUID(), role: "assistant", content: "", streaming: true },
      ]);
      setLoading(true);

      const controller = new AbortController();
      abortRef.current = controller;

      try {
        for await (const ev of streamChat(
          API_BASE,
          { question: q, history, session_id: sessionId },
          controller.signal,
        )) {
          switch (ev.type) {
            case "sources":
              patchLast({ sources: ev.chunks });
              break;
            case "token":
              patchLast((m) => ({ ...m, content: m.content + ev.text }));
              break;
            case "usage":
              patchLast({ usage: ev.usage });
              break;
            case "error":
              patchLast({ error: ev.message, streaming: false });
              break;
            case "done":
              patchLast({ streaming: false });
              break;
          }
        }
      } catch (err) {
        if ((err as Error).name !== "AbortError") {
          patchLast({ error: (err as Error).message, streaming: false });
        } else {
          patchLast({ streaming: false });
        }
      } finally {
        setLoading(false);
        abortRef.current = null;
      }
    },
    [messages, loading, sessionId, patchLast],
  );

  const stop = useCallback(() => abortRef.current?.abort(), []);
  const clear = useCallback(() => setMessages([]), []);

  return { messages, loading, send, stop, clear };
}
