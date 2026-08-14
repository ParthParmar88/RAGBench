import { useMemo, useRef, useState } from "react";
import { askQuestion } from "../api.js";
import { formatCost, formatTokens } from "../utils/format.js";
import AnswerCard from "./AnswerCard.jsx";

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <path
        d="M12 19V5m0 0l-6 6m6-6l6 6"
        stroke="currentColor"
        strokeWidth="1.9"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function EmptyState() {
  return (
    <div className="flex h-full flex-col items-center justify-center px-6 text-center">
      <div className="grid h-12 w-12 place-items-center rounded-xl border border-line bg-surface-2 shadow-soft">
        <svg width="22" height="22" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <path
            d="M21 12a8 8 0 0 1-11.6 7.1L4 20l1-5.3A8 8 0 1 1 21 12Z"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinejoin="round"
            className="text-muted"
          />
        </svg>
      </div>
      <h2 className="mt-4 text-base font-semibold text-ink">Ask your documents</h2>
      <p className="mt-1.5 max-w-sm text-sm text-muted">
        Upload a PDF in the sidebar, then ask a question. Answers are grounded in your
        sources and cited.
      </p>
    </div>
  );
}

function ThinkingDots() {
  return (
    <div className="flex w-fit items-center gap-1.5 rounded-2xl border border-line bg-surface px-4 py-3 shadow-soft">
      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted [animation-delay:-0.3s]" />
      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted [animation-delay:-0.15s]" />
      <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-muted" />
    </div>
  );
}

function UserMessage({ text }) {
  return (
    <div className="flex justify-end">
      <div className="max-w-[80%] rounded-2xl rounded-br-md bg-hover px-4 py-2.5 text-[15px] leading-relaxed text-ink">
        {text}
      </div>
    </div>
  );
}

function ErrorMessage({ text }) {
  return (
    <div className="flex justify-start">
      <div className="max-w-[80%] rounded-2xl rounded-bl-md border border-line bg-surface px-4 py-3 text-sm text-danger">
        {text}
      </div>
    </div>
  );
}

function AssistantMessage({ data }) {
  return (
    <div className="flex justify-start">
      <AnswerCard data={data} className="max-w-[85%] rounded-bl-md" />
    </div>
  );
}

function Message({ msg }) {
  if (msg.role === "user") return <UserMessage text={msg.text} />;
  if (msg.role === "error") return <ErrorMessage text={msg.text} />;
  return <AssistantMessage data={msg.data} />;
}

export default function ChatWindow() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const endRef = useRef(null);

  const sessionTotals = useMemo(() => {
    let tokens = 0;
    let cost = 0;
    let hasCost = false;
    let any = false;
    for (const m of messages) {
      const u = m.role === "assistant" ? m.data?.usage : null;
      if (!u) continue;
      any = true;
      tokens += u.total_tokens || 0;
      if (u.cost != null) {
        cost += u.cost;
        hasCost = true;
      }
    }
    return any ? { tokens, cost: hasCost ? cost : null } : null;
  }, [messages]);

  function scroll() {
    setTimeout(() => endRef.current?.scrollIntoView({ behavior: "smooth" }), 50);
  }

  async function send(e) {
    e.preventDefault();
    const question = input.trim();
    if (!question || loading) return;

    setMessages((m) => [...m, { role: "user", text: question }]);
    setInput("");
    setLoading(true);
    scroll();

    try {
      const data = await askQuestion(question);
      setMessages((m) => [...m, { role: "assistant", data }]);
    } catch (err) {
      setMessages((m) => [...m, { role: "error", text: err.message }]);
    } finally {
      setLoading(false);
      scroll();
    }
  }

  return (
    <main className="flex min-h-0 flex-col bg-surface">
      {/* Session usage total */}
      {sessionTotals && (
        <div className="flex justify-end border-b border-line px-6 py-2">
          <span className="badge badge-neutral">
            Session · {formatTokens(sessionTotals.tokens)} tokens ·{" "}
            {formatCost(sessionTotals.cost)}
          </span>
        </div>
      )}

      {/* Messages */}
      <div className="min-h-0 flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          <div className="mx-auto flex max-w-3xl flex-col gap-6 px-6 py-8">
            {messages.map((m, i) => (
              <Message key={i} msg={m} />
            ))}
            {loading && <ThinkingDots />}
            <div ref={endRef} />
          </div>
        )}
      </div>

      {/* Composer */}
      <div className="border-t border-line bg-surface px-6 py-4">
        <form
          onSubmit={send}
          className="mx-auto flex max-w-3xl items-center gap-2"
        >
          <input
            type="text"
            className="input"
            placeholder="Ask a question about your documents…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={loading}
          />
          <button
            type="submit"
            className="btn btn-primary btn-icon"
            disabled={loading || !input.trim()}
            aria-label="Send"
          >
            <SendIcon />
          </button>
        </form>
        <p className="mx-auto mt-2 max-w-3xl text-center text-xs text-muted">
          Answers are generated from your uploaded documents.
        </p>
      </div>
    </main>
  );
}
