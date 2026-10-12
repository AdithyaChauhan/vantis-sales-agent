import { ChevronDown } from "lucide-react";

/**
 * "How I answered · 4 steps · 6.4 s": the agent's steps for one reply (from the API's `steps`),
 * collapsed by default. The total is the time the page waited for the reply; every time is in seconds.
 */

const NODE_LABELS = {
  understand: "Understood the message",
  retrieve: "Searched the knowledge base",
  agent: "Model call",
  sync: "Saved new details",
  guard: "Safety check on the reply",
  reply: "Fixed reply, no model call",
};
const MS_IN_DETAIL = /,?\s*(\d+) ms\b/;

/** A plain label for one step. */
export function stepLabel(step) {
  if (step.node === "tool") return `Tool: ${step.name ?? "unknown"}`;
  return NODE_LABELS[step.node] ?? step.node;
}

/** "6.4 s" for 6400 ms. */
export function seconds(ms) {
  return `${(ms / 1000).toFixed(1)} s`;
}

/** The step's time in seconds: from its data, or from an "N ms" in its detail. */
function stepTime(step) {
  const ms = step.data?.latency_ms;
  if (Number.isFinite(ms)) return seconds(ms);
  const match = step.detail?.match(MS_IN_DETAIL);
  return match ? seconds(Number(match[1])) : null;
}

/** The detail without its "N ms" (the time is shown separately, in seconds). */
function stepDetail(step) {
  return (step.detail ?? "").replace(MS_IN_DETAIL, "").trim();
}

export default function StepsPanel({ steps, elapsedMs, degraded }) {
  if (!steps?.length && !degraded) return null;
  const parts = ["How I answered", steps.length === 1 ? "1 step" : `${steps.length} steps`];
  if (Number.isFinite(elapsedMs)) parts.push(seconds(elapsedMs));
  return (
    <details className="steps">
      <summary>
        <span className="steps-summary">{parts.join(" · ")}</span>
        <ChevronDown className="steps-chevron" aria-hidden="true" />
      </summary>
      <ol className="steps-list">
        {steps.map((step, i) => (
          <li key={i}>
            <span className="step-label">{stepLabel(step)}</span>
            <span className="step-detail">{stepDetail(step)}</span>
            <span className="step-time">{stepTime(step) ?? ""}</span>
          </li>
        ))}
        {degraded ? (
          <li>
            <span className="step-label">Fallback used</span>
            <span className="step-detail">A backup path answered (keyword search or a backup model)</span>
            <span className="step-time" />
          </li>
        ) : null}
      </ol>
    </details>
  );
}
