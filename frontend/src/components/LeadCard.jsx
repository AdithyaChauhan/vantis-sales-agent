/**
 * Demo view of what the sales team sees, from /api/chat's `lead` summary. Status and score are
 * shown only when the API sends them (CHAT_SHOW_LEAD_SCORE is on for the demo, off in production).
 */

export const DEMO_NOTE = "What the sales team sees. Customers don't see this in production.";

/** A few words for the phone strip: "no lead yet", "not saved yet", "lead saved, callback requested". */
export function leadHeadline(lead) {
  if (!lead) return "no lead yet";
  if (lead.id == null) return "not saved yet";
  return lead.callback_status ? `lead saved, callback ${lead.callback_status}` : "lead saved";
}

function callbackText(status) {
  if (!status) return "None yet";
  if (status === "requested") return "Requested, the team will confirm";
  return status;
}

export default function LeadCard({ lead, showTitle = true }) {
  const saved = lead?.id != null;
  const missing = lead?.missing_fields ?? [];
  return (
    <section className="lead-card" aria-labelledby={showTitle ? "lead-card-title" : undefined}>
      {showTitle ? (
        <h2 id="lead-card-title" className="lead-title">
          Sales view (demo)
        </h2>
      ) : null}
      <p className="lead-note">{DEMO_NOTE}</p>
      {!lead ? (
        <p className="lead-empty">No lead yet. Details appear here as the customer shares them.</p>
      ) : (
        // A new key when anything changes replays the highlight, so a change is easy to spot.
        <div className="lead-body" key={JSON.stringify(lead)}>
          <dl>
            <div className="lead-row">
              <dt>Lead</dt>
              <dd>{saved ? `Saved as lead #${lead.id}` : "Not saved yet"}</dd>
            </div>
            {lead.status != null ? (
              <div className="lead-row">
                <dt>Status</dt>
                <dd>
                  <span className={`badge badge-${String(lead.status).toLowerCase()}`}>{lead.status}</span>
                </dd>
              </div>
            ) : null}
            {lead.score != null ? (
              <div className="lead-row">
                <dt>Score</dt>
                <dd className="lead-score">{lead.score} / 100</dd>
              </div>
            ) : null}
            <div className="lead-row">
              <dt>Callback</dt>
              <dd>{callbackText(lead.callback_status)}</dd>
            </div>
          </dl>
          {missing.length ? (
            <div className="lead-missing">
              <h3>{saved ? "Would add points" : "Still needed to save"}</h3>
              <ul>
                {missing.map((item) => (
                  <li key={item}>{item}</li>
                ))}
              </ul>
            </div>
          ) : null}
        </div>
      )}
    </section>
  );
}
