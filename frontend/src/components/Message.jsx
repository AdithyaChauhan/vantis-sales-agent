import ReplyText from "./ReplyText.jsx";
import SourceChips from "./SourceChips.jsx";
import StepsPanel from "./StepsPanel.jsx";

/** One chat message: the customer's text, or a reply with its sources and steps. */
export default function Message({ message }) {
  if (message.role === "user") {
    return (
      <div className="message message-user">
        <p className="visually-hidden">You said:</p>
        {message.text
          .split("\n")
          .filter((line) => line.trim())
          .map((line, i) => (
            <p key={i}>{line}</p>
          ))}
      </div>
    );
  }
  return (
    <div className="message message-assistant">
      <p className="visually-hidden">Vantis assistant:</p>
      <ReplyText text={message.text} />
      <SourceChips sources={message.sources} />
      <StepsPanel steps={message.steps ?? []} elapsedMs={message.elapsedMs} degraded={message.degraded} />
    </div>
  );
}
