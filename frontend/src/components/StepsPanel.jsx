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
  save: "Saved the lead",
  guard: "Safety check on the reply",
  reply: "Fixed reply, no model call",
  fallback: "Backup reply, no model",
  area_check: "Service area check",
};
const MS_IN_DETAIL = /,?\s*(\d+) ms\b/;

/**
 * T13d: the last line says who wrote the reply (the API's `answered_by`). Only a model's reply on a degraded turn
 * "used a backup path"; when no model answered, the line says so (the Vercel page once claimed "A backup path
 * answered" on a turn no model answered).
 */
const NO_MODEL = {
  knowledge_base: "The reply quotes the knowledge base directly",
  tools: "The reply says what this turn saved",
  fixed: "A fixed reply was sent",
};

/** The last line of the panel: [label, detail], or null when nothing needs saying. */
export function answerLine(answeredBy, degraded) {
  if (NO_MODEL[answeredBy] && (degraded || answeredBy !== "fixed")) return ["No model answered", NO_MODEL[answeredBy]];
  if (degraded) return ["Fallback used", "Part of this turn used a backup path (see the steps above)"];
  return null;
}

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

export default function StepsPanel({ steps, elapsedMs, degraded, answeredBy }) {
  const last = answerLine(answeredBy, degraded);
  if (!steps?.length && !last) return null;
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
        {last ? (
          <li>
            <span className="step-label">{last[0]}</span>
            <span className="step-detail">{last[1]}</span>
            <span className="step-time" />
          </li>
        ) : null}
      </ol>
    </details>
  );
}
