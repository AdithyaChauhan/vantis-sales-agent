import { useEffect, useRef, useState } from "react";

import Composer, { formatWait } from "./components/Composer.jsx";
import LeadCard, { leadHeadline } from "./components/LeadCard.jsx";
import Message from "./components/Message.jsx";
import ReplyText from "./components/ReplyText.jsx";
import { useChat } from "./useChat.js";
import { useWelcome } from "./useWelcome.js";

const WIDE_SCREEN = "(min-width: 900px)"; // same breakpoint as styles.css
const TYPING = "Assistant is typing";

/** True on screens wide enough for the side panel (true when matchMedia is missing, e.g. in tests). */
function useWideScreen() {
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

function prefersReducedMotion() {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}

/** The welcome (F0) as the first assistant message, with its starter questions. */
function Welcome({ welcome, showStarters, startersDisabled, onStarter }) {
  if (welcome.status === "loading") {
    return (
      <div className="message message-assistant" aria-busy="true">
        <div className="skeleton" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        {welcome.slow ? (
          <p className="wake" role="status">
            Waking up the server. This can take up to a minute.
          </p>
        ) : (
          <p className="visually-hidden">Loading</p>
        )}
      </div>
    );
  }
  if (welcome.status === "error") {
    return (
      <div className="message message-assistant">
        <p role="alert">Couldn't reach the server. Check your connection and try again.</p>
        <button type="button" className="btn btn-secondary" onClick={welcome.retry}>
          Try again
        </button>
      </div>
    );
  }
  return (
    <div className="message message-assistant">
      <ReplyText text={welcome.welcome.text} />
      {showStarters ? (
        <div className="starters" role="group" aria-label="Suggested questions">
          {welcome.welcome.starter_questions.map((question) => (
            <button
              key={question}
              type="button"
              className="btn btn-secondary starter"
              disabled={startersDisabled}
              onClick={() => onStarter(question)}
            >
              {question}
            </button>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function Typing({ slow }) {
  return (
    <div className="message message-assistant typing">
      <span className="dots" aria-hidden="true">
        <span />
        <span />
        <span />
      </span>
      <span className="visually-hidden">{TYPING}</span>
      {slow ? (
        <p className="typing-slow" role="status">
          Still working. The first answer after the server wakes up can take about 20 seconds.
        </p>
      ) : null}
    </div>
  );
}

function Notice({ notice, waitLeft }) {
  if (notice?.kind === "error") {
    return (
      <p className="notice notice-error" role="alert">
        {notice.text}
      </p>
    );
  }
  if (notice?.kind === "rate_limit" && waitLeft > 0) {
    return (
      <div className="notice">
        <p role="alert">{notice.text}</p>
        <p className="notice-countdown">You can send again in {formatWait(waitLeft)}.</p>
      </div>
    );
  }
  return null;
}

export default function App() {
  const welcome = useWelcome();
  const chat = useChat();
  const wide = useWideScreen();
  const endRef = useRef(null);

  // Screen readers hear each new reply once; replies restored after a reload are not read again.
  const lastReply = chat.messages.findLast((m) => m.role === "assistant");
  const announcedId = useRef(lastReply?.id);
  const [announcement, setAnnouncement] = useState("");
  useEffect(() => {
    if (chat.pending) {
      setAnnouncement(TYPING);
    } else if (lastReply && lastReply.id !== announcedId.current) {
      announcedId.current = lastReply.id;
      setAnnouncement(`Vantis assistant: ${lastReply.text}`);
    } else {
      setAnnouncement((current) => (current === TYPING ? "" : current));
    }
  }, [chat.pending, lastReply]);

  useEffect(() => {
    endRef.current?.scrollIntoView?.({ block: "end", behavior: prefersReducedMotion() ? "auto" : "smooth" });
  }, [chat.messages.length, chat.pending, welcome.status]);

  const busy = chat.pending || chat.waitLeft > 0;
  const hasChat = chat.messages.length > 0 || chat.sessionId !== null;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="mark" aria-hidden="true" />
          <div>
            <h1 className="brand-name">Vantis Energy Solutions</h1>
            <p className="brand-sub">Rooftop solar assistant</p>
          </div>
        </div>
        <button type="button" className="btn btn-secondary" onClick={chat.newChat} disabled={chat.pending || !hasChat}>
          New chat
        </button>
      </header>

      {!wide ? (
        <details className="sales-strip">
          <summary>Sales view (demo): {leadHeadline(chat.lead)}</summary>
          <LeadCard lead={chat.lead} showTitle={false} />
        </details>
      ) : null}

      <div className="layout">
        <main className="chat">
          <section className="transcript" role="log" aria-live="off" aria-label="Conversation" aria-busy={chat.pending}>
            <Welcome
              welcome={welcome}
              showStarters={chat.messages.length === 0}
              startersDisabled={busy}
              onStarter={chat.send}
            />
            {chat.messages.map((message) => (
              <Message key={message.id} message={message} />
            ))}
            {chat.pending ? <Typing slow={chat.slow} /> : null}
            <div ref={endRef} />
          </section>
          <Notice notice={chat.notice} waitLeft={chat.waitLeft} />
          <Composer
            draft={chat.draft}
            setDraft={chat.setDraft}
            onSend={chat.send}
            pending={chat.pending}
            waitLeft={chat.waitLeft}
          />
        </main>
        {wide ? (
          <aside className="sales-panel">
            <LeadCard lead={chat.lead} />
          </aside>
        ) : null}
      </div>

      <div className="visually-hidden" aria-live="polite">
        {announcement}
      </div>
    </div>
  );
}
