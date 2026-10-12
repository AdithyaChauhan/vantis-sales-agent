/**
 * The chat (session id, transcript, latest lead) in sessionStorage: it survives a reload and is gone
 * when the tab closes, so a name and phone number don't stay on a shared computer. Storage can be
 * blocked (private mode, site settings), so nothing here ever throws.
 */

const KEY = "vantis-chat";

/** The saved chat, or null when there is none (or it can't be read). */
export function loadChat() {
  try {
    const data = JSON.parse(sessionStorage.getItem(KEY));
    if (!data || !Array.isArray(data.messages)) return null;
    return {
      sessionId: typeof data.sessionId === "string" ? data.sessionId : null,
      messages: data.messages,
      lead: data.lead ?? null,
    };
  } catch {
    return null;
  }
}

/** Save the chat ({sessionId, messages, lead}). */
export function saveChat(chat) {
  try {
    sessionStorage.setItem(KEY, JSON.stringify(chat));
  } catch {
    // Storage full or blocked: the chat still works, it just won't survive a reload.
  }
}

/** Forget the saved chat. */
export function clearChat() {
  try {
    sessionStorage.removeItem(KEY);
  } catch {
    // Nothing to do.
  }
}

// The "Sales view" switch is a viewer preference, not chat data, so it lives in localStorage
// (remembered across visits on this browser). On for the demo when nothing is saved.
const SALES_VIEW_KEY = "vantis-sales-view";

/** Whether the sales card is shown (true unless the viewer turned it off). */
export function loadSalesView() {
  try {
    return localStorage.getItem(SALES_VIEW_KEY) !== "off";
  } catch {
    return true;
  }
}

/** Remember the switch position. */
export function saveSalesView(on) {
  try {
    localStorage.setItem(SALES_VIEW_KEY, on ? "on" : "off");
  } catch {
    // Blocked storage: the switch still works for this visit.
  }
}

// The sales API key for the dashboard: sessionStorage only, so it is gone when the tab closes and a
// shared computer doesn't keep it. Never localStorage, the URL, the logs or the build. Lock removes it.
const SALES_KEY = "vantis-sales-key";

/** The saved sales key, or null. */
export function loadSalesKey() {
  try {
    return sessionStorage.getItem(SALES_KEY) || null;
  } catch {
    return null;
  }
}

/** Keep the sales key for this tab. */
export function saveSalesKey(key) {
  try {
    sessionStorage.setItem(SALES_KEY, key);
  } catch {
    // Blocked storage: the key lasts until the page is reloaded.
  }
}

/** Forget the sales key (Lock, or a key the server no longer accepts). */
export function clearSalesKey() {
  try {
    sessionStorage.removeItem(SALES_KEY);
  } catch {
    // Nothing to do.
  }
}
