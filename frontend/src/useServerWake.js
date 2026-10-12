import { useEffect, useState } from "react";

import { pingServer } from "./api.js";

// A warm server answers the ping in about 0.2 s, so a normal visit never shows the note; a sleeping one (Render's
// free plan sleeps after 15 idle minutes and takes 30-40 s to wake) shows it after this long.
export const WAKE_NOTE_AFTER_MS = 1500;
export const WAKE_NOTE = "Waking up the server. The first reply can take up to a minute.";

/**
 * True while the page-load health ping (pingServer) is still waiting after WAKE_NOTE_AFTER_MS (T13a); false
 * again once it answers, and quietly false when it fails. Slow replies later on never turn it on.
 */
export function useServerWake() {
  const [waking, setWaking] = useState(false);
  useEffect(() => {
    let active = true;
    let answered = false;
    const timer = setTimeout(() => active && !answered && setWaking(true), WAKE_NOTE_AFTER_MS);
    pingServer().then(() => {
      answered = true;
      clearTimeout(timer);
      if (active) setWaking(false);
    });
    return () => {
      active = false;
      clearTimeout(timer);
    };
  }, []);
  return waking;
}
