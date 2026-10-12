import { ChartColumn, Plus } from "lucide-react";

import Logo from "./Logo.jsx";

/**
 * The top bar: the brand, the "Sales view" switch (shows or hides the demo sales card) and New chat.
 * On phones the switch and button keep their names for screen readers but show only their icons.
 */
export default function Header({ salesView, onToggleSalesView, onNewChat, newChatDisabled }) {
  return (
    <header className="topbar">
      <div className="brand">
        <Logo className="brand-logo" />
        <div className="brand-text">
          <h1 className="brand-name">Vantis Energy Solutions</h1>
          <p className="brand-sub">Rooftop solar assistant</p>
        </div>
      </div>
      <div className="topbar-actions">
        <button
          type="button"
          role="switch"
          aria-checked={salesView}
          className="switch"
          onClick={onToggleSalesView}
        >
          <ChartColumn className="switch-icon" aria-hidden="true" />
          <span className="switch-label">Sales view</span>
          <span className="switch-track" aria-hidden="true">
            <span className="switch-thumb" />
          </span>
        </button>
        <button type="button" className="btn btn-primary btn-new" onClick={onNewChat} disabled={newChatDisabled}>
          <Plus className="btn-icon" aria-hidden="true" />
          <span className="btn-new-label">New chat</span>
        </button>
      </div>
    </header>
  );
}
