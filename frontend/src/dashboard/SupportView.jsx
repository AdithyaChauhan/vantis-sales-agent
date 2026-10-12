import { WAKE_NOTICE } from "../useWelcome.js";
import { useWideScreen } from "../useWideScreen.js";
import { formatPhone, fullDate, timeAgo } from "./format.js";
import { loadError } from "./LeadsView.jsx";
import { useSlow, useSupportList } from "./useSalesData.js";

export const NO_SUPPORT = "No support requests yet. They appear here when existing customers report an issue in the chat.";

function Who({ request }) {
  const contact = [formatPhone(request.phone), request.email].filter(Boolean).join(" · ");
  return (
    <>
      <span className="cell-strong">{request.name}</span>
      {contact ? <span className="cell-sub">{contact}</span> : null}
    </>
  );
}

function Issue({ request }) {
  return (
    <>
      <span className="support-issue">{request.issue}</span>
      {request.project_details ? <span className="cell-sub">{request.project_details}</span> : null}
    </>
  );
}

function Created({ at }) {
  return (
    <time dateTime={at} title={fullDate(at)}>
      {timeAgo(at)}
    </time>
  );
}

/**
 * Support requests from existing customers (GET /api/support-requests), newest first: who, the issue
 * and when. They are never sales leads. Every value is shown as plain text.
 */
export default function SupportView({ apiKey, refreshTick, onUnauthorized }) {
  const wide = useWideScreen();
  const list = useSupportList(apiKey, refreshTick, onUnauthorized);
  const firstLoad = list.status === "loading" && !list.body;
  const slow = useSlow(firstLoad);

  let body;
  if (firstLoad) {
    body = (
      <div className="dash-state" aria-busy="true">
        <div className="skeleton skeleton-rows" aria-hidden="true">
          <span />
          <span />
          <span />
        </div>
        {slow ? (
          <p className="wake" role="status">
            {WAKE_NOTICE}
          </p>
        ) : (
          <p className="visually-hidden">Loading support requests</p>
        )}
      </div>
    );
  } else if (list.status === "error") {
    body = (
      <div className="dash-state">
        <p role="alert">{loadError(list.error, "support requests")}</p>
        <button type="button" className="btn btn-secondary btn-retry" onClick={list.reload}>
          Try again
        </button>
      </div>
    );
  } else if (list.items.length === 0) {
    body = <p className="dash-state dash-empty">{NO_SUPPORT}</p>;
  } else if (wide) {
    body = (
      <table className="support-table">
        <caption className="visually-hidden">Support requests</caption>
        <thead>
          <tr>
            <th scope="col">Who</th>
            <th scope="col">Issue</th>
            <th scope="col">Created</th>
          </tr>
        </thead>
        <tbody>
          {list.items.map((request) => (
            <tr key={request.id}>
              <td>
                <Who request={request} />
              </td>
              <td>
                <Issue request={request} />
              </td>
              <td className="cell-created">
                <Created at={request.created_at} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  } else {
    body = (
      <ul className="support-tiles">
        {list.items.map((request) => (
          <li key={request.id} className="support-tile">
            <Who request={request} />
            <Issue request={request} />
            <span className="tile-created">
              <Created at={request.created_at} />
            </span>
          </li>
        ))}
      </ul>
    );
  }

  return (
    <section className="dash-card support-card" aria-label="Support requests" aria-busy={list.status === "loading"}>
      {body}
      {list.status === "ready" && list.hasMore ? (
        <div className="load-more">
          {list.moreError ? <p role="alert">{loadError(list.moreError, "next requests")}</p> : null}
          <button type="button" className="btn btn-secondary" onClick={list.more} disabled={list.loadingMore}>
            {list.loadingMore ? "Loading…" : `Load more (${list.items.length} of ${list.total})`}
          </button>
        </div>
      ) : null}
    </section>
  );
}
