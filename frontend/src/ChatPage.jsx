import { useEffect, useRef, useState } from "react";
import { ChartColumn, ChevronDown } from "lucide-react";

import Composer, { formatWait } from "./components/Composer.jsx";
import Header from "./components/Header.jsx";
import LeadCard, { leadHeadline } from "./components/LeadCard.jsx";
import Message, { AssistantAvatar } from "./components/Message.jsx";
import ReplyText from "./components/ReplyText.jsx";
import Sidebar from "./components/Sidebar.jsx";
import { loadSalesView, saveSalesView } from "./storage.js";
import { STILL_WORKING, useChat } from "./useChat.js";
import { useWelcome } from "./useWelcome.js";
import { useWideScreen } from "./useWideScreen.js";

const TYPING = "Assistant is typing";

/** The "Sales view" switch: on by default, remembered in this browser. */
function useSalesView() {
  const [on, setOn] = useState(loadSalesView);
  useEffect(() => saveSalesView(on), [on]);
  return [on, () => setOn((value) => !value)];
}

function prefersReducedMotion() {
  return window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false;
}

/** An assistant row without a saved message: the welcome, the typing dots. */
function AssistantRow({ children, busy = false }) {
  return (
    <div className="msg msg-assistant" aria-busy={busy || undefined}>
      <AssistantAvatar />
      <div className="msg-body">{children}</div>
    </div>
  );
}

/** The welcome (F0) as the first assistant message, with its starter questions. */
function Welcome({ welcome, showStarters, startersDisabled, onStarter }) {
  if (welcome.status === "loading") {
    return (
      <AssistantRow busy>
        <div className="bubble bubble-assistant bubble-text">
          <div className="skeleton" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>
          <p className="visually-hidden">Loading</p>
        </div>
      </AssistantRow>
    );
  }
  if (welcome.status === "error") {
    return (
      <AssistantRow>
        <div className="bubble bubble-assistant bubble-text">
          <p role="alert">Couldn't reach the server. Check your connection and try again.</p>
          <button type="button" className="btn btn-secondary btn-retry" onClick={welcome.retry}>
            Try again
          </button>
        </div>
      </AssistantRow>
    );
  }
  return (
    <AssistantRow>
      <div className="bubble bubble-assistant bubble-text">
        <ReplyText text={welcome.welcome.text} />
      </div>
      {showStarters ? (
        <div className="starters" role="group" aria-label="Suggested questions">
          {welcome.welcome.starter_questions.map((question) => (
            <button
              key={question}
              type="button"
              className="starter"
              disabled={startersDisabled}
              onClick={() => onStarter(question)}
            >
              {question}
            </button>
          ))}
        </div>
      ) : null}
    </AssistantRow>
  );
}

function Typing({ slow }) {
  return (
    <AssistantRow>
      <div className="bubble bubble-assistant bubble-text typing">
        <span className="dots" aria-hidden="true">
          <span />
          <span />
          <span />
        </span>
        <span className="visually-hidden">{TYPING}</span>
        {slow ? (
          <p className="typing-slow" role="status">
            {STILL_WORKING}
          </p>
        ) : null}
      </div>
    </AssistantRow>
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

/** Phone: the sales card folds into a strip under the header. */
function SalesStrip({ lead }) {
  return (
    <details className="sales-strip">
      <summary>
        <ChartColumn className="strip-icon" aria-hidden="true" />
        <span className="strip-title">Sales view (demo):</span>
        <span className="strip-headline">{leadHeadline(lead)}</span>
        {lead?.status ? <span className={`strip-badge badge-${String(lead.status).toLowerCase()}`}>{lead.status}</span> : null}
        <ChevronDown className="strip-chevron" aria-hidden="true" />
      </summary>
      <div className="strip-body">
        <LeadCard lead={lead} showTitle={false} />
      </div>
    </details>
  );
}

/** The customer chat page ("/"): header, sections, the conversation and the demo sales card. */
export default function ChatPage() {
  const welcome = useWelcome();
  const chat = useChat();
  const wide = useWideScreen();
  const [salesView, toggleSalesView] = useSalesView();
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
    const toEnd = () => endRef.current?.scrollIntoView?.({ block: "end", behavior: prefersReducedMotion() ? "auto" : "smooth" });
    toEnd();
    // T14c: a chat restored on page load is first laid out in a fallback font; when the web font arrives the
    // messages wrap differently and grow, which left the newest one cut off at the bottom. Scroll again then.
    document.fonts?.ready.then(toEnd);
  }, [chat.messages.length, chat.pending, welcome.status]);

  const busy = chat.pending || chat.waitLeft > 0;
  const hasChat = chat.messages.length > 0 || chat.sessionId !== null;

  return (
    <div className="app">
      <div className={`frame${salesView ? "" : " frame-no-sales"}`}>
        <Header
          salesView={salesView}
          onToggleSalesView={toggleSalesView}
          onNewChat={chat.newChat}
          newChatDisabled={chat.pending || !hasChat}
        />
        {!wide && salesView ? <SalesStrip lead={chat.lead} /> : null}
        {/* The sales sections show only with the Sales view on, so customers never see staff links */}
        <Sidebar current="chat" staff={salesView} />
        <main className="chat-card">
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
        {wide && salesView ? (
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
