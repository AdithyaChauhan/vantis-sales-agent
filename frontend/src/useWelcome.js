import { useEffect, useState } from "react";

import { getWelcome } from "./api.js";

const LOADING = { status: "loading", welcome: null };

/**
 * GET /api/welcome once on load. status: "loading" | "ready" | "error". retry() tries again after an error.
 * T13a: no waking note of its own; the page-load ping (useServerWake) says when the server is waking up.
 */
export function useWelcome() {
  const [state, setState] = useState(LOADING);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    let active = true;
    getWelcome()
      .then((welcome) => active && setState({ status: "ready", welcome }))
      .catch(() => active && setState({ status: "error", welcome: null }));
    return () => {
      active = false;
    };
  }, [attempt]);

  function retry() {
    setState(LOADING);
    setAttempt((n) => n + 1);
  }

  return { ...state, retry };
}
