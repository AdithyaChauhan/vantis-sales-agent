import { useEffect, useState } from "react";

import { getWelcome } from "./api.js";

// No welcome after this long: the server is probably asleep (Render's free plan), so say so.
export const WAKE_NOTICE_AFTER_MS = 3000;

const LOADING = { status: "loading", slow: false, welcome: null };

/**
 * GET /api/welcome once on load. status: "loading" | "ready" | "error"; slow turns true after
 * WAKE_NOTICE_AFTER_MS while still loading. retry() tries again after an error.
 */
export function useWelcome() {
  const [state, setState] = useState(LOADING);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    const timer = setTimeout(() => active && setState((s) => ({ ...s, slow: true })), WAKE_NOTICE_AFTER_MS);
    getWelcome()
      .then((welcome) => active && setState({ status: "ready", slow: false, welcome }))
      .catch(() => active && setState({ status: "error", slow: false, welcome: null }))
      .finally(() => clearTimeout(timer));
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, [attempt]);

  function retry() {
    setState(LOADING);
    setAttempt((n) => n + 1);
  }

  return { ...state, retry };
}
