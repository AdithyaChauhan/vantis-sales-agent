import { useState } from "react";

import { StatusBadge } from "../components/LeadCard.jsx";
import { navigate } from "../router.jsx";
import { WAKE_NOTICE } from "../useWelcome.js";
import { useWideScreen } from "../useWideScreen.js";
import LeadDetail from "./LeadDetail.jsx";
import { LeadList, LeadTable } from "./LeadTable.jsx";
import { useDebounced, useLeadDetail, useLeadList, useSlow } from "./useSalesData.js";

const STATUSES = ["HOT", "WARM", "COLD"];
export const NO_LEADS = "No leads yet. They appear here when customers share their details in the chat.";
export const NO_MATCH = "No leads match your search or filter.";

/** All, HOT, WARM and COLD: the API's counts for the whole table, whatever the filter or page. */
function CountCards({ counts }) {
  const all = counts ? STATUSES.reduce((sum, status) => sum + (counts[status] ?? 0), 0) : null;
  return (
    <ul className="counts" aria-label="Lead counts">
      <li className="count-card">
        <span className="count-label">All leads</span>
        <span className="count-number">{all ?? "–"}</span>
      </li>
      {STATUSES.map((status) => (
        <li key={status} className="count-card">
          <StatusBadge status={status} compact />
          <span className="count-number">{counts?.[status] ?? "–"}</span>
        </li>
      ))}
    </ul>
  );
}

/** The status filter: All, HOT, WARM, COLD (one at a time). */
function StatusChips({ value, onChange }) {
  return (
    <div className="filter-chips" role="group" aria-label="Show leads">
      {[null, ...STATUSES].map((status) => (
        <button
          key={status ?? "all"}
          type="button"
          className="filter-chip"
          aria-pressed={value === status}
          onClick={() => onChange(status)}
        >
          {status ?? "All"}
        </button>
      ))}
    </div>
  );
}

/** Skeleton rows while the first page loads, with the waking-up notice after 3 s. */
function Loading({ slow }) {
  return (
    <div className="dash-state" aria-busy="true">
      <div className="skeleton skeleton-rows" aria-hidden="true">
        <span />
        <span />
        <span />
        <span />
      </div>
      {slow ? (
        <p className="wake" role="status">
          {WAKE_NOTICE}
        </p>
      ) : (
        <p className="visually-hidden">Loading leads</p>
      )}
    </div>
  );
}

/** The API's message for a 503 or 429 (plain words from our own server), else a short one. */
export function loadError(error, what) {
  if (error?.status === 503 || error?.status === 429) return error.detail || `Couldn't load the ${what}.`;
  return `Couldn't load the ${what}. Check your connection and try again.`;
}

/**
 * The leads page: count cards, status chips, the list (search, filter, sort and paging run on the
 * server) and the selected lead. On desktop the first lead is selected until another one is clicked;
 * the URL (/leads/<id>) holds the choice, so a reload or a shared link opens the same lead.
 */
export default function LeadsView({ apiKey, leadId, search, refreshTick, onUnauthorized }) {
  const wide = useWideScreen();
  const [status, setStatus] = useState(null);
  const [sort, setSort] = useState("newest");
  const q = useDebounced(search.trim(), 300);
  const list = useLeadList(apiKey, { status, q, sort }, refreshTick, onUnauthorized);
  const firstLoad = list.status === "loading" && !list.body;
  const slow = useSlow(firstLoad);
  const selectedId = leadId ?? (wide ? (list.items[0]?.id ?? null) : null);
  const detail = useLeadDetail(apiKey, selectedId, refreshTick, onUnauthorized);

  function select(id) {
    if (wide) navigate(`/leads/${id}`, { replace: true });
    else navigate(`/leads/${id}`, { state: { fromList: true } });
  }

  function back() {
    if (window.history.state?.fromList) window.history.back(); // keeps the list where it was
    else navigate("/leads");
  }

  let body;
  if (firstLoad) {
    body = <Loading slow={slow} />;
  } else if (list.status === "error") {
    body = (
      <div className="dash-state">
        <p role="alert">{loadError(list.error, "leads")}</p>
        <button type="button" className="btn btn-secondary btn-retry" onClick={list.reload}>
          Try again
        </button>
      </div>
    );
  } else if (list.items.length === 0) {
    const none = list.counts && STATUSES.every((s) => !list.counts[s]);
    body = <p className="dash-state dash-empty">{none ? NO_LEADS : NO_MATCH}</p>;
  } else {
    body = wide ? (
      <LeadTable items={list.items} selectedId={selectedId} sort={sort} onSort={setSort} onSelect={select} />
    ) : (
      <LeadList items={list.items} />
    );
  }

  return (
    <>
      <CountCards counts={list.counts} />
      <StatusChips value={status} onChange={setStatus} />
      <section className="dash-card leads-card" aria-label="Leads" aria-busy={list.status === "loading"}>
        {body}
        {list.status === "ready" && list.hasMore ? (
          <div className="load-more">
            {list.moreError ? <p role="alert">{loadError(list.moreError, "next leads")}</p> : null}
            <button type="button" className="btn btn-secondary" onClick={list.more} disabled={list.loadingMore}>
              {list.loadingMore ? "Loading…" : `Load more (${list.items.length} of ${list.total})`}
            </button>
          </div>
        ) : null}
      </section>
      {wide ? <LeadDetail detail={detail} /> : null}
      {!wide && leadId != null ? <LeadDetail detail={detail} overlay onBack={back} /> : null}
    </>
  );
}
