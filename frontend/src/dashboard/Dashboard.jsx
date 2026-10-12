import { useState } from "react";
import { Lock, RotateCw, Search } from "lucide-react";

import Logo from "../components/Logo.jsx";
import Sidebar, { SectionNav } from "../components/Sidebar.jsx";
import { clearSalesKey, loadSalesKey, saveSalesKey } from "../storage.js";
import LeadsView from "./LeadsView.jsx";
import SupportView from "./SupportView.jsx";
import UnlockCard from "./UnlockCard.jsx";
import "./dashboard.css";

const HEADINGS = {
  leads: { title: "Sales dashboard", sub: "Leads from the website chat" },
  support: { title: "Support requests", sub: "Issues from existing customers" },
};
export const KEY_STOPPED = "That key no longer works. Enter the current one.";

/**
 * The sales dashboard: "/leads" (and "/leads/<id>") and "/support". Without a sales key it shows the
 * unlock card. The key lives in sessionStorage only (src/storage.js); Lock forgets it and so does any
 * 401 from the API (for example after the key was changed on the server).
 */
export default function Dashboard({ view, leadId }) {
  const [key, setKey] = useState(loadSalesKey);
  const [lockedNote, setLockedNote] = useState(null);
  const [search, setSearch] = useState("");
  const [refreshTick, setRefreshTick] = useState(0);

  function unlock(newKey) {
    saveSalesKey(newKey);
    setLockedNote(null);
    setKey(newKey);
  }

  function lock(note = null) {
    clearSalesKey();
    setKey(null);
    setSearch("");
    setLockedNote(note);
  }

  const heading = HEADINGS[view];
  return (
    <div className="dash">
      <Sidebar current={view} staff brand />
      <main className={`dash-main${key ? ` dash-${view}` : " dash-locked"}`}>
        {key ? (
          <>
            <header className="dash-head">
              <div className="dash-titles">
                <Logo className="dash-logo" />
                <div>
                  <h1 className="dash-title">{heading.title}</h1>
                  <p className="dash-sub">{heading.sub}</p>
                </div>
              </div>
              {view === "leads" ? (
                <label className="dash-search">
                  <Search className="dash-search-icon" aria-hidden="true" />
                  <span className="visually-hidden">Search leads by name or city</span>
                  <input
                    type="search"
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                    placeholder="Search name or city"
                    maxLength={100}
                    autoComplete="off"
                  />
                </label>
              ) : null}
              <div className="dash-buttons">
                <button type="button" className="btn btn-primary btn-refresh" onClick={() => setRefreshTick((n) => n + 1)}>
                  <RotateCw className="btn-icon" aria-hidden="true" />
                  <span className="dash-button-label">Refresh</span>
                </button>
                <button type="button" className="btn btn-secondary btn-lock" onClick={() => lock()}>
                  <Lock className="btn-icon" aria-hidden="true" />
                  <span className="dash-button-label">Lock</span>
                </button>
              </div>
            </header>
            <SectionNav current={view} staff className="nav tabs" />
            {view === "leads" ? (
              <LeadsView
                apiKey={key}
                leadId={leadId}
                search={search}
                refreshTick={refreshTick}
                onUnauthorized={() => lock(KEY_STOPPED)}
              />
            ) : (
              <SupportView apiKey={key} refreshTick={refreshTick} onUnauthorized={() => lock(KEY_STOPPED)} />
            )}
          </>
        ) : (
          <UnlockCard note={lockedNote} onUnlock={unlock} />
        )}
      </main>
    </div>
  );
}
