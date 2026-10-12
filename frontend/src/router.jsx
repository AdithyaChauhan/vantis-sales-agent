import { useSyncExternalStore } from "react";

/**
 * A tiny router on the browser's History API (no dependency): usePath() for the current path,
 * navigate() to change it, and <Link>, a real link whose plain left click stays inside the app.
 * The server must answer every path with index.html so that a reload or a deep link works (Vite does
 * this locally; on Vercel it needs the rewrite noted for T13 in docs/PLAN.md).
 */

const NAVIGATED = "vantis:navigated";

function subscribe(onChange) {
  window.addEventListener("popstate", onChange); // the browser's Back and Forward
  window.addEventListener(NAVIGATED, onChange); // navigate() below
  return () => {
    window.removeEventListener("popstate", onChange);
    window.removeEventListener(NAVIGATED, onChange);
  };
}

/** The current path, e.g. "/leads/12"; the component re-renders when it changes. */
export function usePath() {
  return useSyncExternalStore(subscribe, () => window.location.pathname);
}

/**
 * Go to a path inside the app. replace: swap the current history entry instead of adding one.
 * state: saved with the entry (window.history.state), e.g. where the visitor came from.
 */
export function navigate(to, { replace = false, state = null } = {}) {
  if (to === window.location.pathname) return;
  window.history[replace ? "replaceState" : "pushState"](state, "", to);
  window.dispatchEvent(new Event(NAVIGATED));
}

/**
 * A real <a href>, so it can be opened in a new tab or copied. A plain left click navigates inside
 * the app; a click with Ctrl, Cmd, Shift or Alt, or with another button, keeps the browser's behaviour.
 */
export function Link({ to, replace = false, state = null, onClick, ...props }) {
  function handleClick(event) {
    onClick?.(event);
    const modified = event.metaKey || event.ctrlKey || event.shiftKey || event.altKey;
    if (event.defaultPrevented || event.button !== 0 || modified) return;
    event.preventDefault();
    navigate(to, { replace, state });
  }
  return <a href={to} onClick={handleClick} {...props} />;
}
