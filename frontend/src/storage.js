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
