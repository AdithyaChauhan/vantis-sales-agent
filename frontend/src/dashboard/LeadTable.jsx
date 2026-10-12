import { ArrowDown } from "lucide-react";

import { StatusBadge } from "../components/LeadCard.jsx";
import { Link } from "../router.jsx";
import { firstSentence, fullDate, placeAndType, requirementSummary, timeAgo, typeLabel } from "./format.js";

/**
 * The leads list: a table on wide screens (LeadTable) and cards on phones (LeadList). Rows show the
 * API's own score and status, short versions of the free text (the full text is in the detail panel)
 * and every customer-typed value as plain text.
 */

/** A small "Demo" tag for rows from scripts/seed_demo_leads.py (source_channel "demo"). */
export function DemoTag({ lead }) {
  return lead.source_channel === "demo" ? <span className="demo-tag">Demo</span> : null;
}

/** The score number over a thin bar (the bar is decoration; the number is the value). */
export function ScoreCell({ score }) {
  return (
    <span className="score-cell">
      <span className="score-cell-number">{score}</span>
      <span className="mini-bar" aria-hidden="true">
        <span style={{ width: `${Math.max(0, Math.min(100, score))}%` }} />
      </span>
    </span>
  );
}

function Created({ at }) {
  return (
    <time dateTime={at} title={fullDate(at)}>
      {timeAgo(at)}
    </time>
  );
}

/** A column header that sorts the list on the server (highest score first, or newest first). */
function SortHeader({ label, value, sort, onSort }) {
  const active = sort === value;
  return (
    <th scope="col" aria-sort={active ? "descending" : undefined}>
      <button type="button" className={`th-sort${active ? " is-active" : ""}`} onClick={() => onSort(value)}>
        {label}
        <span className="visually-hidden">{value === "score" ? ", sort highest first" : ", sort newest first"}</span>
        {active ? <ArrowDown className="th-sort-icon" aria-hidden="true" /> : null}
      </button>
    </th>
  );
}

/**
 * Desktop table. The name is a real link to /leads/<id> (it can open in a new tab); a click anywhere
 * else on the row selects the lead too.
 */
export function LeadTable({ items, selectedId, sort, onSort, onSelect }) {
  function rowClick(event, id) {
    if (event.target.closest("a, button")) return; // the link handles its own clicks (incl. new tabs)
    onSelect(id);
  }
  return (
    <table className="lead-table">
      <caption className="visually-hidden">Leads</caption>
      <colgroup>
        <col className="col-name" />
        <col className="col-city" />
        <col className="col-need" />
        <col className="col-score" />
        <col className="col-status" />
        <col className="col-action" />
        <col className="col-created" />
      </colgroup>
      <thead>
        <tr>
          <th scope="col">Name</th>
          <th scope="col">City</th>
          <th scope="col">Requirement</th>
          <SortHeader label="Score" value="score" sort={sort} onSort={onSort} />
          <th scope="col">Status</th>
          <th scope="col">Next action</th>
          <SortHeader label="Created" value="newest" sort={sort} onSort={onSort} />
        </tr>
      </thead>
      <tbody>
        {items.map((lead) => {
          const selected = lead.id === selectedId;
          const need = requirementSummary(lead);
          const action = firstSentence(lead.recommended_action);
          return (
            <tr key={lead.id} className={selected ? "is-selected" : undefined} onClick={(event) => rowClick(event, lead.id)}>
              <td>
                <Link to={`/leads/${lead.id}`} replace className="lead-link" aria-current={selected ? "true" : undefined}>
                  {lead.name ?? "No name"}
                </Link>
                <span className="cell-sub cell-sub-row">
                  {typeLabel(lead) ?? ""}
                  <DemoTag lead={lead} />
                </span>
              </td>
              <td>
                <span className="cell-line">{lead.city ?? "Not given"}</span>
              </td>
              <td>
                <span className="cell-line" title={lead.requirement ?? need ?? undefined}>
                  {need ?? "Not given"}
                </span>
              </td>
              <td>
                <ScoreCell score={lead.lead_score} />
              </td>
              <td>
                <StatusBadge status={lead.lead_status} compact />
              </td>
              <td>
                <span className="cell-two-lines" title={lead.recommended_action}>
                  {action}
                </span>
              </td>
              <td className="cell-created">
                <Created at={lead.created_at} />
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

/** Phone: one card per lead; a tap opens its details full screen (the list stays underneath). */
export function LeadList({ items }) {
  return (
    <ul className="lead-tiles">
      {items.map((lead) => (
        <li key={lead.id}>
          <Link to={`/leads/${lead.id}`} state={{ fromList: true }} className="lead-tile">
            <span className="tile-top">
              <span className="tile-name">{lead.name ?? "No name"}</span>
              <DemoTag lead={lead} />
              <StatusBadge status={lead.lead_status} compact />
            </span>
            <span className="tile-sub">{placeAndType(lead) || "City not given"}</span>
            <ScoreCell score={lead.lead_score} />
            <span className="tile-line">{requirementSummary(lead) ?? "Requirement not given"}</span>
            <span className="tile-line tile-action">{firstSentence(lead.recommended_action)}</span>
            <span className="tile-created">
              <Created at={lead.created_at} />
            </span>
          </Link>
        </li>
      ))}
    </ul>
  );
}
