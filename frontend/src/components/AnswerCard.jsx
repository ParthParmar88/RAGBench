// Reusable card that renders one answer: the text, its source citations, and a
// collapsible usage footer.
import { formatCost, formatTokens } from "../utils/format.js";

function ChevronIcon() {
  return (
    <svg
      width="14"
      height="14"
      viewBox="0 0 24 24"
      fill="none"
      className="ml-auto text-muted transition-transform group-open:rotate-180"
      aria-hidden="true"
    >
      <path
        d="M6 9l6 6 6-6"
        stroke="currentColor"
        strokeWidth="1.8"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function UsageRow({ label, value }) {
  return (
    <>
      <dt className="text-muted">{label}</dt>
      <dd className="truncate text-right font-medium text-ink">{value}</dd>
    </>
  );
}

function UsageFooter({ usage }) {
  // No usage recorded (e.g. the provider didn't report token counts).
  if (!usage) return null;
  return (
    <details className="group [&_summary]:list-none">
      <summary className="flex cursor-pointer items-center gap-2 text-xs text-muted">
        <span className="font-medium text-secondary">
          {formatTokens(usage.total_tokens)} tokens
        </span>
        <span aria-hidden="true">·</span>
        <span>{formatCost(usage.cost, usage.currency)}</span>
        <ChevronIcon />
      </summary>
      <dl className="mt-2 grid grid-cols-2 gap-x-6 gap-y-1.5 rounded-lg border border-line bg-surface-2 px-3 py-2.5 text-xs">
        <UsageRow label="Input tokens" value={formatTokens(usage.input_tokens)} />
        <UsageRow label="Output tokens" value={formatTokens(usage.output_tokens)} />
        <UsageRow label="Total tokens" value={formatTokens(usage.total_tokens)} />
        <UsageRow label="Model cost" value={formatCost(usage.cost, usage.currency)} />
        <UsageRow label="Model" value={usage.model} />
      </dl>
    </details>
  );
}

function SourceList({ sources }) {
  if (!sources?.length) return null;
  return (
    <div className="flex flex-wrap items-center gap-1.5 border-t border-line px-4 py-2">
      <span className="text-[10px] font-semibold uppercase tracking-wider text-muted">
        Sources
      </span>
      {sources.map((s) => (
        <details key={s.number} className="group [&_summary]:list-none">
          <summary
            title={s.title}
            className="flex max-w-[160px] cursor-pointer items-center gap-1 rounded-md border border-line bg-surface-2 px-1.5 py-0.5 text-[11px] text-secondary hover:bg-hover"
          >
            <span className="font-semibold text-muted">{s.number}</span>
            <span className="min-w-0 truncate text-ink">{s.title}</span>
            {s.score != null && (
              <span className="text-muted">{(s.score * 100).toFixed(0)}%</span>
            )}
          </summary>
          <p className="mt-1 max-w-xs rounded-md border border-line bg-surface-2 p-2 text-[11px] leading-relaxed text-muted">
            {s.snippet}…
          </p>
        </details>
      ))}
    </div>
  );
}

/**
 * @param {object} props
 * @param {object} props.data        - { direct_answer, sources, usage }
 * @param {string} [props.className]
 */
export default function AnswerCard({ data, className = "" }) {
  return (
    <div className={`card overflow-hidden rounded-2xl ${className}`}>
      <div className="px-4 py-3.5">
        <p className="whitespace-pre-wrap text-[15px] leading-relaxed text-ink">
          {data.direct_answer}
        </p>
      </div>

      <SourceList sources={data.sources} />

      {data.usage && (
        <div className="border-t border-line px-4 py-3">
          <UsageFooter usage={data.usage} />
        </div>
      )}
    </div>
  );
}
