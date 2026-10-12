import { MessageSquare } from "lucide-react";

/**
 * The left column on wide screens: the sections of the app (only Chat for now; T11 adds Leads as a
 * real link) and a rooftop solar photo at the bottom. The photo is a CSS background, so phones, where
 * this column is hidden, never download it.
 */
export default function Sidebar() {
  return (
    <div className="sidebar">
      <nav aria-label="Sections">
        <ul className="nav">
          <li>
            <span className="nav-item" aria-current="page">
              <MessageSquare className="nav-icon" aria-hidden="true" />
              Chat
            </span>
          </li>
        </ul>
      </nav>
      <div className="sidebar-photo" aria-hidden="true" />
    </div>
  );
}
