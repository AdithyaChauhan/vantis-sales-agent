import { Headset, MessageSquare, UsersRound } from "lucide-react";

import { Link } from "../router.jsx";
import Logo from "./Logo.jsx";

/** The app's sections. Leads and Support requests are for the sales team (`staff`). */
const SECTIONS = [
  { key: "chat", label: "Chat", to: "/", Icon: MessageSquare },
  { key: "leads", label: "Leads", to: "/leads", Icon: UsersRound, staff: true },
  { key: "support", label: "Support requests", to: "/support", Icon: Headset, staff: true },
];

/**
 * The sections as a list: the current one is a marker (aria-current, not a link to itself), the
 * others are links. The sales sections are listed only when `staff` is true, so customers never see
 * them. `className` styles the list (the sidebar's column or the dashboard's row of tabs).
 */
export function SectionNav({ current, staff = false, className = "nav" }) {
  return (
    <nav aria-label="Sections">
      <ul className={className}>
        {SECTIONS.filter((section) => staff || !section.staff).map(({ key, label, to, Icon }) => (
          <li key={key}>
            {key === current ? (
              <span className="nav-item" aria-current="page">
                <Icon className="nav-icon" aria-hidden="true" />
                {label}
              </span>
            ) : (
              <Link to={to} className="nav-item nav-link">
                <Icon className="nav-icon" aria-hidden="true" />
                {label}
              </Link>
            )}
          </li>
        ))}
      </ul>
    </nav>
  );
}

/**
 * The left column on wide screens: the sections and a rooftop solar photo at the bottom. The photo is
 * a CSS background, so phones, where this column is hidden, never download it. The dashboard's
 * version (`brand`) is wider and starts with the company name, as in its reference image.
 */
export default function Sidebar({ current, staff = false, brand = false }) {
  return (
    <div className={`sidebar${brand ? " sidebar-wide" : ""}`}>
      {brand ? (
        <div className="sidebar-brand">
          <Logo className="sidebar-logo" />
          <div className="brand-text">
            <p className="sidebar-brand-name">Vantis Energy Solutions</p>
            <p className="brand-sub">Rooftop solar assistant</p>
          </div>
        </div>
      ) : null}
      <SectionNav current={current} staff={staff} />
      <div className="sidebar-photo" aria-hidden="true" />
    </div>
  );
}
