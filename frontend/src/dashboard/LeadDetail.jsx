import { ArrowLeft, Circle, CircleCheck, ListTodo, Mail, Phone, TriangleAlert } from "lucide-react";

import { checklist, DETAILS, StatusBadge } from "../components/LeadCard.jsx";
import { WAKE_NOTICE } from "../useWelcome.js";
import { detailValues, formatPhone, knownFields, otherDetails, placeAndType, timeAgo } from "./format.js";
import { DemoTag } from "./LeadTable.jsx";
import { useSlow } from "./useSalesData.js";

/**
 * The selected lead (GET /api/leads/{id}): who it is, the API's score and status, why it scored that
 * way (score_breakdown: each rule's points and maximum, and any cap), the reason, the full next action,
 * contact details, callbacks, and everything else the customer told us. On phones it covers the screen
 * and has a Back button. Customer-typed text is always rendered as plain text.
 */

function WhyThisScore({ breakdown, score }) {
  const rules = breakdown?.rules ?? [];
  const caps = breakdown?.caps ?? [];
  const raw = breakdown?.raw_score;
  return (
    <section className="detail-section">
      <h3 className="detail-heading">
        Why this score
      </h3>
      <ul className="why">
        {rules.map((rule) => (
          <li key={rule.rule}>
            <span>{rule.label}</span>
            <span className="why-points">
              {rule.points}/{rule.max}
            </span>
          </li>
        ))}
        {caps.length ? (
          <li className="why-cap">
            <TriangleAlert className="why-cap-icon" aria-hidden="true" />
            <span className="why-cap-lines">
              {caps.map((cap) => (
                <span key={cap.rule}>{cap.label}</span>
              ))}
              <span className="why-cap-before">
                {raw > score ? `Score before the cap: ${raw} → ${score}` : "The cap didn't change this score."}
              </span>
            </span>
          </li>
        ) : null}
      </ul>
    </section>
  );
}

function Contact({ lead }) {
  return (
    <section className="detail-section">
      <h3 className="detail-heading">
        Contact
      </h3>
      {lead.phone || lead.email ? (
        <ul className="contact">
          {lead.phone ? (
            <li>
              <Phone className="contact-icon" aria-hidden="true" />
              <a href={`tel:${lead.phone}`}>{formatPhone(lead.phone)}</a>
            </li>
          ) : null}
          {lead.email ? (
            <li>
              <Mail className="contact-icon" aria-hidden="true" />
              <a href={`mailto:${lead.email}`}>{lead.email}</a>
            </li>
          ) : null}
        </ul>
      ) : (
        <p className="detail-muted">No contact details.</p>
      )}
    </section>
  );
}

function Callbacks({ callbacks }) {
  const newestFirst = [...(callbacks ?? [])].reverse();
  if (!newestFirst.length) {
    return (
      <div className="detail-callback">
        <Phone className="row-icon" aria-hidden="true" />
        <p>
          <strong>Callback:</strong> not requested
        </p>
      </div>
    );
  }
  return newestFirst.map((callback) => (
    <div className="detail-callback" key={callback.id}>
      <Phone className="row-icon" aria-hidden="true" />
      <p>
        <strong>Callback:</strong> {callback.status}, {callback.preferred_time}
        <span className="detail-callback-when"> · asked {timeAgo(callback.created_at)}</span>
      </p>
    </div>
  ));
}

function DetailsCollected({ lead }) {
  const values = detailValues(lead);
  const rows = checklist({ known_fields: knownFields(lead) });
  return (
    <section className="detail-section">
      <h3 className="detail-heading">
        Details collected
      </h3>
      <ul className="checklist detail-checklist">
        {rows.map(({ key, label, done }) => {
          const { Icon } = DETAILS.find((row) => row.key === key);
          const value = values[key];
          return (
            <li key={key} className={done ? "is-done" : undefined}>
              <Icon className="row-icon" aria-hidden="true" />
              <span className="row-text">
                <span className="row-label">{label}</span>
                {value ? <span className="row-value">{value}</span> : null}
                {key === "need" && lead.requirement ? <span className="row-value row-words">{lead.requirement}</span> : null}
                {!done ? <span className="row-value">not yet</span> : null}
              </span>
              {done ? (
                <CircleCheck className="row-state row-state-done" aria-hidden="true" />
              ) : (
                <Circle className="row-state row-state-empty" aria-hidden="true" />
              )}
            </li>
          );
        })}
      </ul>
      <dl className="other-details">
        {otherDetails(lead).map(([label, value]) => (
          <div key={label}>
            <dt>{label}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function Body({ lead }) {
  const breakdown = lead.score_breakdown;
  const where = placeAndType(lead);
  return (
    <>
      <div className="detail-head">
        <div className="detail-who">
          <h2 className="detail-name">{lead.name ?? "No name"}</h2>
          <p className="detail-sub">
            {where ? <span>{where}</span> : null}
            <DemoTag lead={lead} />
          </p>
        </div>
        <StatusBadge status={lead.lead_status} />
      </div>
      <div className="detail-score">
        <p className="score-value">
          <span className="score-number">{lead.lead_score}</span>
          <span className="score-max"> / 100</span>
        </p>
        <div className="score-bar" aria-hidden="true">
          <span style={{ width: `${Math.max(0, Math.min(100, lead.lead_score))}%` }} />
        </div>
      </div>
      <WhyThisScore breakdown={breakdown} score={lead.lead_score} />
      <section className="detail-section">
        <h3 className="detail-heading">
          Reason
        </h3>
        <p className="detail-text">{lead.reason}</p>
      </section>
      <div className="next-action">
        <ListTodo className="next-action-icon" aria-hidden="true" />
        <div>
          <h3 className="detail-heading">Next action</h3>
          <p className="detail-text">{lead.recommended_action}</p>
        </div>
      </div>
      <Contact lead={lead} />
      <Callbacks callbacks={lead.callbacks} />
      <DetailsCollected lead={lead} />
    </>
  );
}

/**
 * detail: {status: "idle" | "loading" | "ready" | "error", lead, error, retry} from useLeadDetail.
 * overlay: the phone version (full screen, with Back).
 */
export default function LeadDetail({ detail, overlay = false, onBack }) {
  const slow = useSlow(detail.status === "loading" && !detail.lead);
  let content;
  if (detail.lead) {
    content = <Body lead={detail.lead} />;
  } else if (detail.status === "loading") {
    content = (
      <div className="detail-loading" aria-busy="true">
        <div className="skeleton" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        {slow ? (
          <p className="wake" role="status">
            {WAKE_NOTICE}
          </p>
        ) : (
          <p className="visually-hidden">Loading the lead</p>
        )}
      </div>
    );
  } else if (detail.status === "error") {
    content = (
      <div className="dash-state">
        <p role="alert">
          {detail.error?.kind === "not_found"
            ? "This lead wasn't found. It may have been removed."
            : "Couldn't load this lead. Check your connection and try again."}
        </p>
        {detail.error?.kind === "not_found" ? null : (
          <button type="button" className="btn btn-secondary btn-retry" onClick={detail.retry}>
            Try again
          </button>
        )}
      </div>
    );
  } else {
    content = <p className="detail-muted dash-state">Select a lead to see its details.</p>;
  }

  return (
    <aside className={`dash-card detail${overlay ? " detail-overlay" : ""}`} aria-label="Lead details">
      {overlay ? (
        <button type="button" className="btn btn-secondary btn-back" onClick={onBack}>
          <ArrowLeft className="btn-icon" aria-hidden="true" />
          Back to leads
        </button>
      ) : null}
      {content}
    </aside>
  );
}
