import { BookOpen } from "lucide-react";

/**
 * The KB chunks a reply is based on, one pill each: "KB 8 · Product & Service Catalog". Opening a pill
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
              <BookOpen className="chip-icon" aria-hidden="true" />
              <span className="chip-section">
                KB <span className="visually-hidden">Section </span>
                {source.section}
              </span>
              <span className="chip-dot" aria-hidden="true">
                ·
              </span>
              <span className="chip-title">{source.title}</span>
            </summary>
            <div className="chip-panel">
              <p className="chip-snippet">{source.snippet}</p>
              {source.page ? <p className="chip-page">Page {source.page} of the knowledge base</p> : null}
            </div>
          </details>
        </li>
      ))}
    </ul>
  );
}
