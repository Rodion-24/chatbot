import type { UsagePayload } from "../lib/sse";

type Props = { usage: UsagePayload };

/** Token counts and cost for one answer, straight from the `usage` event. */
export default function UsageBadge({ usage }: Props) {
  const cost =
    usage.cost_usd === null ? "n/a" : `$${usage.cost_usd.toFixed(4)}`;
  return (
    <div className="mt-2 flex flex-wrap gap-x-3 font-mono text-[11px] text-slate-400 dark:text-slate-500">
      <span>{usage.model}</span>
      <span title="input tokens">in {usage.input_tokens}</span>
      <span title="output tokens">out {usage.output_tokens}</span>
      <span title="estimated cost">{cost}</span>
    </div>
  );
}
