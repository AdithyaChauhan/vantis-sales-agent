/**
 * A reply as paragraphs and lists, built only from React text nodes: nothing in a reply is ever
 * treated as HTML, so "<script>" shows as text. Small extras the agent's replies use:
 * "- " or "* " lines become a list, "1. " lines a numbered list, **x** is bold, and
 * "[KB Section 8: Product & Service Catalog]" becomes a small "KB Section 8" tag.
 */

const CITATION = /\[KB Section (\d+): ([^\]\n]+)\]/g;
const BOLD = /\*\*([^*\n]+)\*\*/g;
const BULLET = /^[-*•]\s+/;
const NUMBERED = /^\d+[.)]\s+/;

/** Split text on a pattern: plain strings stay strings, each match becomes render(match, key). */
function splitOn(text, pattern, render, keyPrefix) {
  const parts = [];
  let last = 0;
  for (const match of text.matchAll(pattern)) {
    if (match.index > last) parts.push(text.slice(last, match.index));
    parts.push(render(match, `${keyPrefix}-${match.index}`));
    last = match.index + match[0].length;
  }
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

function Citation({ section, title }) {
  return (
    <span className="cite" title={`KB Section ${section}: ${title}`}>
      {`KB Section ${section}`}
      <span className="visually-hidden">{`: ${title}`}</span>
    </span>
  );
}

/** One line of text with citations and bold turned into elements. */
export function inline(text, key = "i") {
  return splitOn(
    text,
    CITATION,
    (m, k) => <Citation key={k} section={m[1]} title={m[2].trim()} />,
    key,
  ).flatMap((part, i) =>
    typeof part === "string"
      ? splitOn(part, BOLD, (m, k) => <strong key={k}>{m[1]}</strong>, `${key}-${i}`)
      : [part],
  );
}

/** Group the reply's non-empty lines into paragraphs and lists. */
export function toBlocks(text) {
  const blocks = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    const kind = BULLET.test(line) ? "ul" : NUMBERED.test(line) ? "ol" : "p";
    const content = kind === "p" ? line : line.replace(kind === "ul" ? BULLET : NUMBERED, "");
    const previous = blocks[blocks.length - 1];
    if (kind !== "p" && previous?.kind === kind) previous.items.push(content);
    else blocks.push({ kind, items: [content] });
  }
  return blocks;
}

export default function ReplyText({ text }) {
  return (
    <div className="reply-text">
      {toBlocks(text).map((block, b) => {
        if (block.kind === "p") return <p key={b}>{inline(block.items[0], `p${b}`)}</p>;
        const List = block.kind;
        return (
          <List key={b}>
            {block.items.map((item, i) => (
              <li key={i}>{inline(item, `l${b}-${i}`)}</li>
            ))}
          </List>
        );
      })}
    </div>
  );
}
