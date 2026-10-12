/**
 * A reply as paragraphs and lists, built only from React text nodes: nothing in a reply is ever
 * treated as HTML, so "<script>" shows as text. The small bit of Markdown the agent's replies use:
 * "- " or "* " lines become a list, "1. " lines a numbered list (starting at the first number),
 * "# " at the start of a line a heading, **x** is bold and *x* italic; any other asterisk is dropped,
 * so none ever shows. Underscores are left alone (they appear in email addresses). And
 * "[KB Section 8: Product & Service Catalog]" becomes a small "KB 8" tag.
 */

const CITATION = /\[KB Section (\d+): ([^\]\n]+)\]/g;
const BOLD = /\*\*([^*\n]+)\*\*/g;
// *x* not touching a letter, digit or another asterisk, so "5*3" and "**" leftovers are not italics
const ITALIC = /(?<![\w*])\*(?=\S)([^*\n]+?)(?<=\S)\*(?![\w*])/g;
const STRAY_ASTERISKS = /\*+/g;
const HEADING = /^#{1,6}\s+/; // "# Steps" is a heading, "#1" is not
const BULLET = /^[-*•]\s+/;
const NUMBERED = /^(\d+)[.)]\s+/;

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

/** Text with any asterisk left over from Markdown removed. */
const plain = (text) => text.replace(STRAY_ASTERISKS, "");

function Citation({ section, title }) {
  const label = `KB Section ${section}: ${title}`;
  return (
    <span className="cite" title={label}>
      <span aria-hidden="true">{`KB ${section}`}</span>
      <span className="visually-hidden">{label}</span>
    </span>
  );
}

/** Apply `pattern` to the string parts of `parts` (elements pass through unchanged). */
function splitStrings(parts, pattern, render, key) {
  return parts.flatMap((part, i) => (typeof part === "string" ? splitOn(part, pattern, render, `${key}-${i}`) : [part]));
}

/** One line of text with citations, bold and italics turned into elements. */
export function inline(text, key = "i") {
  let parts = splitOn(text, CITATION, (m, k) => <Citation key={k} section={m[1]} title={m[2].trim()} />, key);
  parts = splitStrings(parts, BOLD, (m, k) => <strong key={k}>{plain(m[1])}</strong>, `${key}b`);
  parts = splitStrings(parts, ITALIC, (m, k) => <em key={k}>{plain(m[1])}</em>, `${key}e`);
  return parts.map((part) => (typeof part === "string" ? plain(part) : part));
}

/** Group the reply's non-empty lines into paragraphs, headings and lists. */
export function toBlocks(text) {
  const blocks = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (!line) continue;
    if (HEADING.test(line)) {
      blocks.push({ kind: "h", items: [line.replace(HEADING, "")] });
      continue;
    }
    const number = line.match(NUMBERED);
    const kind = BULLET.test(line) ? "ul" : number ? "ol" : "p";
    const content = kind === "p" ? line : line.replace(kind === "ul" ? BULLET : NUMBERED, "");
    const previous = blocks[blocks.length - 1];
    if (kind !== "p" && previous?.kind === kind) previous.items.push(content);
    else blocks.push({ kind, items: [content], start: kind === "ol" ? Number(number[1]) : undefined });
  }
  return blocks;
}

export default function ReplyText({ text }) {
  return (
    <div className="reply-text">
      {toBlocks(text).map((block, b) => {
        if (block.kind === "p") return <p key={b}>{inline(block.items[0], `p${b}`)}</p>;
        if (block.kind === "h") {
          return (
            <p key={b} className="reply-heading">
              <strong>{inline(block.items[0], `h${b}`)}</strong>
            </p>
          );
        }
        const List = block.kind;
        const start = block.kind === "ol" && block.start !== 1 ? block.start : undefined;
        return (
          <List key={b} start={start}>
            {block.items.map((item, i) => (
              <li key={i}>{inline(item, `l${b}-${i}`)}</li>
            ))}
          </List>
        );
      })}
    </div>
  );
}
