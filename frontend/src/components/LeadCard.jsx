import { useId } from "react";
import {
  CalendarDays,
  ChartColumn,
  Circle,
  CircleCheck,
  Flame,
  House,
  MapPin,
  Phone,
  ReceiptText,
  Snowflake,
  SolarPanel,
  Sun,
  UserRound,
} from "lucide-react";

/**
 * Demo view of what the sales team sees, from /api/chat's `lead` summary: the name, city and type,
 * the status and score (only when the API sends them: CHAT_SHOW_LEAD_SCORE is on for the demo, off
 * in production), which details are collected (from the known field names), what is still missing,
 * and the callback.
 */

export const DEMO_NOTE = "What the sales team sees; hidden from customers in production.";

/** The "Details collected" rows and the field names that tick each one. */
const DETAILS = [
  { key: "city", label: "City", Icon: MapPin, fields: ["city"] },
  { key: "need", label: "Need", Icon: House, fields: ["system_size_kw", "requirement"] },
  { key: "bill", label: "Bill", Icon: ReceiptText, fields: ["monthly_bill_inr", "monthly_units_kwh"] },
  { key: "timeline", label: "Timeline", Icon: CalendarDays, fields: ["timeline", "timeline_bucket"] },
  { key: "contact", label: "Contact", Icon: UserRound, fields: ["phone", "email"] },
  { key: "rooftop", label: "Rooftop", Icon: SolarPanel, fields: ["rooftop_available"] },
];

const STATUS = {
  HOT: { Icon: Flame, className: "badge-hot" },
  WARM: { Icon: Sun, className: "badge-warm" },
  COLD: { Icon: Snowflake, className: "badge-cold" },
};

/** [{key, label, done}] for the checklist: done when any of the row's fields is known. */
export function checklist(lead) {
  const known = new Set(lead?.known_fields ?? []);
  return DETAILS.map(({ key, label, fields }) => ({ key, label, done: fields.some((field) => known.has(field)) }));
}

/** A few words for the phone strip: "no lead yet", "not saved yet", "lead saved, callback requested". */
export function leadHeadline(lead) {
  if (!lead) return "no lead yet";
  if (lead.id == null) return "not saved yet";
  return lead.callback_status ? `lead saved, callback ${lead.callback_status}` : "lead saved";
}

function callbackText(status) {
  if (!status) return "not requested yet";
  if (status === "requested") return "requested, the team will confirm";
  return status;
}

const capitalize = (text) => (text ? text[0].toUpperCase() + text.slice(1) : text);

/** "Lucknow · Residential": the city and the customer type (or the property type). */
function subtitle(lead) {
  return [lead.city, capitalize(lead.customer_type ?? lead.property_type)].filter(Boolean).join(" · ");
}

function StatusBadge({ status }) {
  const { Icon, className } = STATUS[status] ?? { Icon: null, className: "badge-other" };
  return (
    <span className={`badge ${className}`}>
      {Icon ? <Icon className="badge-icon" aria-hidden="true" /> : null}
      {status}
    </span>
  );
}

function Score({ score, children }) {
  const width = `${Math.max(0, Math.min(100, score))}%`;
  return (
    <div className="score">
      <div className="score-row">
        <p className="score-value">
          <span className="score-number">{score}</span>
          <span className="score-max"> / 100</span>
        </p>
        {children}
      </div>
      <div className="score-bar" aria-hidden="true">
        <span style={{ width }} />
      </div>
    </div>
  );
}

function Identity({ lead }) {
  if (!lead) {
    return (
      <div className="lead-panel lead-identity lead-identity-empty">
        <p className="lead-name lead-name-empty">No lead yet</p>
        <p className="lead-sub">Details appear here as the customer shares them.</p>
      </div>
    );
  }
  const saved = lead.id != null;
  const line = subtitle(lead);
  const savedChip = (
    <p className={`lead-saved${saved ? " is-saved" : ""}`}>{saved ? `Saved as lead #${lead.id}` : "Not saved yet"}</p>
  );
  return (
    // A new key when anything changes replays the highlight, so a change is easy to spot.
    <div className="lead-panel lead-identity lead-body" key={JSON.stringify(lead)}>
      <div className="lead-head">
        <div className="lead-who">
          <p className="lead-name">{lead.name ?? "New lead"}</p>
          {line ? <p className="lead-sub">{line}</p> : null}
        </div>
        {lead.status != null ? <StatusBadge status={lead.status} /> : null}
      </div>
      {/* With a score, the saved chip sits in the score's row; without one, under the city line */}
      {lead.score != null ? <Score score={lead.score}>{savedChip}</Score> : savedChip}
    </div>
  );
}

function Details({ lead }) {
  const headingId = useId();
  const missing = lead?.missing_fields ?? [];
  const saved = lead?.id != null;
  return (
    <section className="lead-panel lead-details" aria-labelledby={headingId}>
      <h3 id={headingId} className="lead-details-title">
        Details collected
      </h3>
      <ul className="checklist">
        {checklist(lead).map(({ key, label, done }) => {
          const { Icon } = DETAILS.find((row) => row.key === key);
          return (
            <li key={key} className={done ? "is-done" : undefined}>
              <Icon className="row-icon" aria-hidden="true" />
              <span className="row-label">{label}</span>
              {done ? (
                <CircleCheck className="row-state row-state-done" aria-hidden="true" />
              ) : (
                <Circle className="row-state row-state-empty" aria-hidden="true" />
              )}
              <span className="visually-hidden">{done ? "collected" : "not yet"}</span>
            </li>
          );
        })}
      </ul>
      {missing.length ? (
        <div className="lead-missing">
          <p className="lead-missing-label">{saved ? "Would add points" : "Still needed to save"}</p>
          <ul>
            {missing.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </div>
      ) : null}
    </section>
  );
}

export default function LeadCard({ lead, showTitle = true }) {
  const titleId = useId();
  return (
    <section className="lead-card" aria-labelledby={showTitle ? titleId : undefined} aria-label={showTitle ? undefined : "Sales view (demo)"}>
      {showTitle ? (
        <div className="lead-card-head">
          <h2 id={titleId} className="lead-title">
            <ChartColumn className="lead-title-icon" aria-hidden="true" />
            Sales view (demo)
          </h2>
          <p className="lead-note">{DEMO_NOTE}</p>
        </div>
      ) : (
        <p className="lead-note">{DEMO_NOTE}</p>
      )}
      <Identity lead={lead} />
      <Details lead={lead} />
      <div className="lead-panel lead-callback">
        <Phone className="row-icon" aria-hidden="true" />
        <p>
          <strong>Callback:</strong> {callbackText(lead?.callback_status)}
        </p>
      </div>
    </section>
  );
}
