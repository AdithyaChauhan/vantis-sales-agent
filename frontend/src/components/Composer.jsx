import { Send } from "lucide-react";

// The API accepts 1-1,000 characters (backend/app/api/chat.py).
export const MAX_MESSAGE_CHARS = 1000;

/**
 * The message box. Enter sends, Shift+Enter adds a line. While a reply is on its way the box is
 * read-only (it keeps focus) and Send is disabled; during a 429 wait Send shows the countdown.
 * The box starts as one line and grows with the text.
 */
export default function Composer({ draft, setDraft, onSend, pending, waitLeft }) {
  const blocked = pending || waitLeft > 0;
  const canSend = !blocked && draft.trim().length > 0;

  function submit(event) {
    event.preventDefault();
    if (canSend) onSend(draft);
  }

  function onKeyDown(event) {
    // isComposing: Enter that confirms an input-method word (e.g. Hindi typing) must not send
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      if (canSend) onSend(draft);
    }
  }

  return (
    <form className="composer" onSubmit={submit}>
      <label htmlFor="message" className="visually-hidden">
        Your message
      </label>
      <div className="composer-field">
        <textarea
          id="message"
          rows={1}
          maxLength={MAX_MESSAGE_CHARS}
          value={draft}
          readOnly={pending}
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Type a message…"
          aria-describedby="composer-hint"
        />
      </div>
      <p id="composer-hint" className="visually-hidden">
        Enter to send, Shift+Enter for a new line
      </p>
      <button type="submit" className="btn btn-primary btn-send" disabled={!canSend}>
        <Send className="btn-icon" aria-hidden="true" />
        <span>{waitLeft > 0 ? `Wait ${formatWait(waitLeft)}` : "Send"}</span>
      </button>
    </form>
  );
}

/** "12 s", "3 min" or "about 5 h": how long a 429 wait still lasts. */
export function formatWait(seconds) {
  if (seconds <= 90) return `${seconds} s`;
  if (seconds < 3600) return `${Math.ceil(seconds / 60)} min`;
  return `about ${Math.round(seconds / 3600)} h`;
}
