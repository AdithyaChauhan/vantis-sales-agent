import { useEffect } from "react";

import ChatPage from "./ChatPage.jsx";
import Dashboard from "./dashboard/Dashboard.jsx";
import { Link, usePath } from "./router.jsx";

/**
 * Which page a path shows: "/" the customer chat; "/leads" (or "/leads/<id>" with that lead open) and
 * "/support" the sales dashboard; anything else a short "not found" page. A trailing slash is ignored.
 */
export function route(path) {
  const clean = path.length > 1 ? path.replace(/\/+$/, "") : path;
  if (clean === "/") return { page: "chat" };
  if (clean === "/leads") return { page: "leads", leadId: null };
  const lead = /^\/leads\/([1-9]\d{0,8})$/.exec(clean);
  if (lead) return { page: "leads", leadId: Number(lead[1]) };
  if (clean === "/support") return { page: "support" };
  return { page: "missing" };
}

const TITLES = {
  chat: "Vantis Solar Assistant",
  leads: "Leads · Vantis sales dashboard",
  support: "Support requests · Vantis sales dashboard",
  missing: "Page not found · Vantis",
};

function NotFound() {
  return (
    <main className="not-found">
      <div className="not-found-card">
        <h1>Page not found</h1>
        <p>There is nothing at this address.</p>
        <Link to="/" className="btn btn-primary btn-not-found">
          Go to the chat
        </Link>
      </div>
    </main>
  );
}

/** The app: one page per path (src/router.jsx), and the browser tab's title for it. */
export default function App() {
  const where = route(usePath());
  useEffect(() => {
    document.title = TITLES[where.page];
  }, [where.page]);

  if (where.page === "chat") return <ChatPage />;
  if (where.page === "missing") return <NotFound />;
  return <Dashboard view={where.page} leadId={where.leadId} />;
}
