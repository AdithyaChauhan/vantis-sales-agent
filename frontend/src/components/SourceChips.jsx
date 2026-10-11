/**
 * The KB chunks a reply is based on, one chip each. A chip shows the KB section and title; opening it
 * (click, Enter or Space) shows the snippet and the page.
 */
export default function SourceChips({ sources }) {
  if (!sources?.length) return null;
  return (
    <ul className="sources" aria-label="Sources">
      {sources.map((source, i) => (
        <li key={`${source.section}-${i}`}>
          <details className="chip">
            <summary>
              <span className="chip-section">KB Section {source.section}</span>
              <span className="chip-title">{source.title}</span>
            </summary>
            <p className="chip-snippet">{source.snippet}</p>
            {source.page ? <p className="chip-page">Page {source.page} of the knowledge base</p> : null}
          </details>
        </li>
      ))}
    </ul>
  );
}
