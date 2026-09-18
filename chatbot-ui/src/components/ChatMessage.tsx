import type { ChatMessage as Msg } from "../lib/useChat";
import SourcesPanel from "./SourcesPanel";
import UsageBadge from "./UsageBadge";

type Props = { message: Msg };

export default function ChatMessage({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={[
          "max-w-[80%] rounded-2xl px-4 py-3 text-sm leading-relaxed",
          isUser
            ? "bg-sky-600 text-white"
            : "bg-white text-slate-800 shadow-sm ring-1 ring-slate-200 dark:bg-slate-800 dark:text-slate-100 dark:ring-slate-700",
        ].join(" ")}
      >
        {/* whitespace-pre-wrap keeps the model's newlines and code fences. */}
        <div className="whitespace-pre-wrap break-words">
          {message.content}
          {message.streaming && (
            <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-current align-text-bottom" />
          )}
        </div>

        {message.error && (
          <div className="mt-2 text-xs text-red-600 dark:text-red-400">
            error: {message.error}
          </div>
        )}
        {!isUser && message.sources && <SourcesPanel chunks={message.sources} />}
        {!isUser && message.usage && <UsageBadge usage={message.usage} />}
      </div>
    </div>
  );
}
