import { useEffect, useState } from "react";

// Same breakpoint as styles.css and dashboard.css: below it, the phone layout.
export const WIDE_SCREEN = "(min-width: 900px)";

/** True on screens wide enough for side panels (true when matchMedia is missing, e.g. in tests). */
export function useWideScreen() {
  const [wide, setWide] = useState(() => window.matchMedia?.(WIDE_SCREEN).matches ?? true);
  useEffect(() => {
    const query = window.matchMedia?.(WIDE_SCREEN);
    if (!query) return undefined;
    const update = () => setWide(query.matches);
    query.addEventListener("change", update);
    return () => query.removeEventListener("change", update);
  }, []);
  return wide;
}
