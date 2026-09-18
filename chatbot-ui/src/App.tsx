import { useEffect, useRef } from "react";
import AiBackdrop from "./components/AiBackdrop";
import ChatInput from "./components/ChatInput";
import ChatMessage from "./components/ChatMessage";
import Robot3D from "./components/Robot3D";
import { useChat } from "./lib/useChat";

export default function App() {
  const { messages, loading, send, stop, clear } = useChat();
  const bottomRef = useRef<HTMLDivElement>(null);
  const chatZoneRef = useRef<HTMLDivElement>(null);

  // Keep the newest token in view while an answer streams in.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  return (
    <div className="flex h-dvh flex-col bg-slate-50 text-slate-900 dark:bg-slate-900 dark:text-slate-100">
      {/* Three columns: a robot in each corner, chat info dead centre.
          The side cells are equal width so the title is truly centred. */}
      <header className="grid grid-cols-[112px_1fr_112px] items-center border-b border-slate-200 px-4 py-2 dark:border-slate-800">
        <Robot3D className="h-24 w-28" />

        <div className="text-center">
          <h1 className="text-base font-semibold">pgvector docs assistant</h1>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Answers come only from the indexed documentation.
          </p>
          {messages.length > 0 && (
            <button
              type="button"
              onClick={clear}
              disabled={loading}
              className="mt-1 text-xs text-slate-500 hover:text-slate-800 disabled:opacity-40 dark:text-slate-400 dark:hover:text-slate-100"
            >
              New chat
            </button>
          )}
        </div>

        <Robot3D className="h-24 w-28 justify-self-end" />
      </header>

      {/* `relative` anchors the absolutely-positioned canvas to <main>. */}
      <main className="relative flex-1 overflow-y-auto px-4 py-6">
        {/* Drawn first so it sits behind. The chat zone below is measured
            live, so particles never enter it — even before any message. */}
        <AiBackdrop keepOutRef={chatZoneRef} />
        <div
          ref={chatZoneRef}
          className="relative mx-auto flex min-h-full max-w-3xl flex-col gap-4 rounded-2xl bg-slate-50/70 px-4 py-2 ring-1 ring-slate-200/60 backdrop-blur-[1px] dark:bg-slate-900/70 dark:ring-slate-800/60"
        >
          {messages.length === 0 && (
            <p className="mt-16 text-center text-sm text-slate-400">
              Try: “How do I create an HNSW index?”
            </p>
          )}
          {messages.map((m) => (
            <ChatMessage key={m.id} message={m} />
          ))}
          <div ref={bottomRef} />
        </div>
      </main>

      <footer className="border-t border-slate-200 px-4 py-3 dark:border-slate-800">
        <div className="mx-auto max-w-3xl">
          <ChatInput disabled={loading} onSend={send} onStop={stop} />
        </div>
      </footer>
    </div>
  );
}
