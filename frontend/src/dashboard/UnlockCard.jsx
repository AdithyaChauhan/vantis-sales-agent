import { useId, useState } from "react";
import { KeyRound, LockOpen } from "lucide-react";

import { getLeads } from "../api.js";
import { formatWait } from "../components/Composer.jsx";
import { Link } from "../router.jsx";
import { WAKE_NOTICE } from "../useWelcome.js";
import { useSlow } from "./useSalesData.js";

/** What to say when the key check fails, from the API's answer. */
export function unlockMessage(error) {
  if (error?.status === 401) return "That key didn't work.";
  if (error?.status === 429) {
    return `Too many wrong keys from this network. Try again in ${formatWait(error.retryAfter ?? 60)}.`;
  }
  if (error?.status === 503) return error.detail || "The server isn't set up for the dashboard yet.";
  if (error?.kind === "network") return "Couldn't reach the server. Check your connection and try again.";
  return "Something went wrong. Please try again.";
}

/**
 * Asks for the sales API key and checks it with one small request (GET /api/leads?limit=1). The input
 * has no name and the form never submits to a URL, so the key can't end up in an address or a log;
 * the dashboard keeps it in sessionStorage only. `note` says why the card is back (e.g. a key the
 * server no longer accepts).
 */
export default function UnlockCard({ note, onUnlock }) {
  const [value, setValue] = useState("");
  const [checking, setChecking] = useState(false);
  const [error, setError] = useState(null);
  const slow = useSlow(checking);
  const ids = { title: useId(), input: useId(), error: useId() };

  async function submit(event) {
    event.preventDefault();
    const key = value.trim();
    if (!key || checking) return;
    setChecking(true);
    setError(null);
    try {
      await getLeads({ limit: 1 }, key);
      onUnlock(key); // the dashboard replaces this card
    } catch (failure) {
      setError(unlockMessage(failure));
      setChecking(false);
    }
  }

  return (
    <section className="dash-card unlock-card" aria-labelledby={ids.title}>
      <span className="unlock-icon" aria-hidden="true">
        <KeyRound />
      </span>
      <h1 id={ids.title} className="unlock-title">
        Sales dashboard
      </h1>
      <p className="unlock-text">
        Enter the sales API key to see leads and support requests. It stays in this browser tab only and is
        forgotten when you close the tab or press Lock.
      </p>
      {note ? (
        <p className="unlock-note" role="status">
          {note}
        </p>
      ) : null}
      <form className="unlock-form" onSubmit={submit} autoComplete="off">
        <label htmlFor={ids.input} className="unlock-label">
          Sales API key
        </label>
        <div className="unlock-row">
          <input
            id={ids.input}
            className="unlock-input"
            type="password"
            value={value}
            onChange={(event) => setValue(event.target.value)}
            autoComplete="off"
            autoCapitalize="off"
            spellCheck={false}
            aria-invalid={error ? true : undefined}
            aria-describedby={error ? ids.error : undefined}
          />
          <button type="submit" className="btn btn-primary btn-unlock" disabled={!value.trim() || checking}>
            <LockOpen className="btn-icon" aria-hidden="true" />
            {checking ? "Checking…" : "Unlock"}
          </button>
        </div>
      </form>
      {error ? (
        <p id={ids.error} className="unlock-error" role="alert">
          {error}
        </p>
      ) : null}
      {slow ? (
        <p className="wake" role="status">
          {WAKE_NOTICE}
        </p>
      ) : null}
      <Link to="/" className="unlock-back">
        Back to the chat
      </Link>
    </section>
  );
}
