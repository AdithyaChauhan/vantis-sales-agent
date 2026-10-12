import { useServerWake, WAKE_NOTE } from "../useServerWake.js";

/**
 * The "waking up" note (T13a), on every page while the page-load health ping is still waiting. The polite live
 * region is always there, so screen readers hear the note when it appears; it is fixed, so it never moves the page.
 */
export default function WakeNote() {
  const waking = useServerWake();
  return (
    <div className="wake-region" role="status" aria-live="polite">
      {waking ? <p className="wake-note">{WAKE_NOTE}</p> : null}
    </div>
  );
}
