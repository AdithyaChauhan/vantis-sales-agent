import { UserRound } from "lucide-react";

import { SunMark } from "./Logo.jsx";
import ReplyText from "./ReplyText.jsx";
import SourceChips from "./SourceChips.jsx";
import StepsPanel from "./StepsPanel.jsx";

const TIME_FORMAT = new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" });

/** "11:02 AM" (in the browser's language) under a message; nothing for messages saved without a time. */
export function MessageTime({ at }) {
  const date = at ? new Date(at) : null;
  if (!date || Number.isNaN(date.getTime())) return null;
  return (
    <time className="msg-time" dateTime={date.toISOString()}>
      {TIME_FORMAT.format(date)}
    </time>
  );
}

/** The assistant's avatar: the Vantis sun on a peach circle. */
export function AssistantAvatar() {
  return (
    <span className="avatar avatar-assistant" aria-hidden="true">
      <SunMark className="avatar-sun" />
    </span>
  );
}

/** One chat message: the customer's text, or a reply with its steps, sources and time. */
export default function Message({ message }) {
  if (message.role === "user") {
    return (
      <div className="msg msg-user">
        <div className="msg-body">
          <div className="bubble bubble-user">
            <p className="visually-hidden">You said:</p>
            {message.text
              .split("\n")
              .filter((line) => line.trim())
              .map((line, i) => (
                <p key={i}>{line}</p>
              ))}
          </div>
          <MessageTime at={message.at} />
        </div>
        <span className="avatar avatar-user" aria-hidden="true">
          <UserRound className="avatar-person" />
        </span>
      </div>
    );
  }
  return (
    <div className="msg msg-assistant">
      <AssistantAvatar />
      <div className="msg-body">
        <div className="bubble bubble-assistant">
          <div className="bubble-text">
            <p className="visually-hidden">Vantis assistant:</p>
            <ReplyText text={message.text} />
          </div>
          <StepsPanel
            steps={message.steps ?? []}
            elapsedMs={message.elapsedMs}
            degraded={message.degraded}
            answeredBy={message.answeredBy}
          />
        </div>
        <SourceChips sources={message.sources} />
        <MessageTime at={message.at} />
      </div>
    </div>
  );
}
