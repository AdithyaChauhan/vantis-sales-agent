import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import LeadCard, { DEMO_NOTE, leadHeadline } from "./LeadCard.jsx";

const DRAFT = { id: null, status: null, score: null, missing_fields: ["name", "phone or email"], callback_status: null };
const SAVED_HIDDEN = { id: 7, status: null, score: null, missing_fields: ["ownership"], callback_status: "requested" };
const SAVED_SHOWN = { ...SAVED_HIDDEN, status: "HOT", score: 97 };

describe("LeadCard", () => {
  it("is labelled as a demo view of what the sales team sees", () => {
    render(<LeadCard lead={null} />);
    expect(screen.getByRole("heading", { name: "Sales view (demo)" })).toBeTruthy();
    expect(screen.getByText(DEMO_NOTE)).toBeTruthy();
    expect(screen.getByText(/No lead yet/)).toBeTruthy();
  });

  it("shows what is still needed before the lead is saved", () => {
    render(<LeadCard lead={DRAFT} />);
    expect(screen.getByText("Not saved yet")).toBeTruthy();
    expect(screen.getByText("Still needed to save")).toBeTruthy();
    expect(screen.getByText("phone or email")).toBeTruthy();
    expect(screen.getByText("None yet")).toBeTruthy();
  });

  it("hides score and status when the API doesn't send them", () => {
    const { container } = render(<LeadCard lead={SAVED_HIDDEN} />);
    expect(screen.getByText("Saved as lead #7")).toBeTruthy();
    expect(screen.getByText("Requested, the team will confirm")).toBeTruthy();
    expect(screen.getByText("Would add points")).toBeTruthy();
    expect(screen.queryByText("Status")).toBeNull();
    expect(screen.queryByText("Score")).toBeNull();
    expect(container.textContent).not.toMatch(/HOT|WARM|COLD|\/ 100/);
  });

  it("shows score and status when the API sends them (CHAT_SHOW_LEAD_SCORE)", () => {
    render(<LeadCard lead={SAVED_SHOWN} />);
    expect(screen.getByText("HOT")).toBeTruthy();
    expect(screen.getByText("97 / 100")).toBeTruthy();
  });

  it("gives the phone strip a short headline", () => {
    expect(leadHeadline(null)).toBe("no lead yet");
    expect(leadHeadline(DRAFT)).toBe("not saved yet");
    expect(leadHeadline(SAVED_HIDDEN)).toBe("lead saved, callback requested");
  });
});
