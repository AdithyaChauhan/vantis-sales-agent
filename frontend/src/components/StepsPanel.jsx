/**
 * "How I answered": the agent's steps for one reply (from the API's `steps`), collapsed by default.
 * The total is the time the page waited for the reply.
 */

const NODE_LABELS = {
  understand: "Understood the message",
  retrieve: "Searched the knowledge base",
  agent: "Model call",
  sync: "Saved new details",
  reply: "Fixed reply, no model call",
};

/** A plain label for one step. */
export function stepLabel(step) {
  if (step.node === "tool") return `Tool: ${step.name ?? "unknown"}`;
  return NODE_LABELS[step.node] ?? step.node;
}

function seconds(ms) {
  return `${(ms / 1000).toFixed(1)} s`;
}

/** The step's time, unless its detail already says it (a model call's detail is "gemini, 984 ms"). */
function stepTime(step) {
  const ms = step.data?.latency_ms;
  if (!Number.isFinite(ms) || step.detail?.includes(`${ms} ms`)) return null;
  return `${ms} ms`;
}

export default function StepsPanel({ steps, elapsedMs, degraded }) {
  if (!steps?.length && !degraded) return null;
  const count = steps.length === 1 ? "1 step" : `${steps.length} steps`;
  const total = Number.isFinite(elapsedMs) ? `, ${seconds(elapsedMs)}` : "";
  return (
    <details className="steps">
      <summary>{`How I answered (${count}${total})`}</summary>
      <ol>
        {steps.map((step, i) => (
          <li key={i}>
            <span className="step-label">{stepLabel(step)}</span>
            <span className="step-detail">{step.detail}</span>
            {stepTime(step) ? <span className="step-time">{stepTime(step)}</span> : null}
          </li>
        ))}
        {degraded ? (
          <li>
            <span className="step-label">Fallback used</span>
            <span className="step-detail">A backup path answered (keyword search or a backup model)</span>
          </li>
        ) : null}
      </ol>
    </details>
  );
}
