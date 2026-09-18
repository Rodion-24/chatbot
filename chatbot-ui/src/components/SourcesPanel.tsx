import type { SourceChunk } from "../lib/sse";

type Props = { chunks: SourceChunk[] };

/** Chunks the retriever handed to the model, ranked by cosine distance. */
export default function SourcesPanel({ chunks }: Props) {
  if (chunks.length === 0) return null;

  return (
    <details className="mt-2 text-xs text-slate-500 dark:text-slate-400">
      <summary className="cursor-pointer select-none hover:text-slate-700 dark:hover:text-slate-200">
        {chunks.length} source{chunks.length === 1 ? "" : "s"}
      </summary>
      <ul className="mt-1 space-y-0.5 pl-1">
        {chunks.map((c) => (
          <li key={c.chunk_id} className="flex items-baseline gap-2 font-mono">
            <span className="text-slate-700 dark:text-slate-300">
              chunk {c.chunk_id}
            </span>
            {c.section_title && (
              <span className="truncate font-sans">{c.section_title}</span>
            )}
            {/* Lower is closer — cosine distance, not similarity. */}
            <span className="ml-auto tabular-nums" title="cosine distance">
              {c.score.toFixed(3)}
            </span>
          </li>
        ))}
      </ul>
    </details>
  );
}
