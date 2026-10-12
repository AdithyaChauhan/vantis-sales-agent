import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import LeadCard, { checklist, DEMO_NOTE, leadHeadline } from "./LeadCard.jsx";

const DRAFT = {
  id: null, status: null, score: null, missing_fields: ["name", "phone or email"], callback_status: null,
  name: null, city: "Lucknow", property_type: "house", customer_type: "residential",
  known_fields: ["city", "customer_type", "monthly_bill_inr", "property_type", "system_size_kw", "timeline"],
};
const SAVED_HIDDEN = {
  ...DRAFT, id: 7, name: "Rahul Sharma", missing_fields: ["ownership"], callback_status: "requested",
  known_fields: [...DRAFT.known_fields, "email", "name", "phone"],
};
const SAVED_SHOWN = { ...SAVED_HIDDEN, status: "HOT", score: 97 };

/** {label: "collected" | "not yet"} for the "Details collected" rows on screen. */
function rows() {
  const list = screen.getByRole("heading", { name: "Details collected" }).nextElementSibling;
  return Object.fromEntries(
    within(list).getAllByRole("listitem").map((li) => [li.querySelector(".row-label").textContent, li.textContent.endsWith("collected") ? "collected" : "not yet"]),
  );
}

describe("LeadCard", () => {
  it("is labelled as a demo view of what the sales team sees, with a designed empty state", () => {
    render(<LeadCard lead={null} />);
    expect(screen.getByRole("heading", { name: "Sales view (demo)" })).toBeTruthy();
    expect(screen.getByText(DEMO_NOTE)).toBeTruthy();
    expect(screen.getByText(/No lead yet/)).toBeTruthy();
    expect(Object.values(rows())).toEqual(Array(6).fill("not yet"));
    expect(screen.getByText("not requested yet")).toBeTruthy();
  });

  it("shows what is still needed before the lead is saved", () => {
    render(<LeadCard lead={DRAFT} />);
    expect(screen.getByText("Not saved yet")).toBeTruthy();
    expect(screen.getByText("Still needed to save")).toBeTruthy();
    expect(screen.getByText("phone or email")).toBeTruthy();
    expect(screen.getByText("not requested yet")).toBeTruthy();
    expect(screen.getByText("New lead")).toBeTruthy(); // no name yet
    expect(screen.getByText("Lucknow · Residential")).toBeTruthy();
  });

  it("ticks the checklist from the known field names", () => {
    render(<LeadCard lead={DRAFT} />);
    expect(rows()).toEqual({
      City: "collected", Need: "collected", Bill: "collected", Timeline: "collected", Contact: "not yet", Rooftop: "not yet",
    });
    expect(checklist({ known_fields: ["email", "rooftop_available"] }).filter((r) => r.done).map((r) => r.key)).toEqual([
      "contact", "rooftop",
    ]);
    expect(checklist(null).every((r) => !r.done)).toBe(true);
  });

  it("hides score and status when the API doesn't send them", () => {
    const { container } = render(<LeadCard lead={SAVED_HIDDEN} />);
    expect(screen.getByText("Rahul Sharma")).toBeTruthy();
    expect(screen.getByText("Saved as lead #7")).toBeTruthy();
    expect(screen.getByText("requested, the team will confirm")).toBeTruthy();
    expect(screen.getByText("Would add points")).toBeTruthy();
    expect(rows().Contact).toBe("collected");
    expect(container.querySelector(".score")).toBeNull();
    expect(container.textContent).not.toMatch(/HOT|WARM|COLD|\/ 100/);
  });

  it("shows score and status when the API sends them (CHAT_SHOW_LEAD_SCORE)", () => {
    const { container } = render(<LeadCard lead={SAVED_SHOWN} />);
    expect(screen.getByText("HOT")).toBeTruthy();
    expect(container.querySelector(".score-value").textContent).toBe("97 / 100");
    expect(container.querySelector(".score-bar span").style.width).toBe("97%");
  });

  it("gives WARM and COLD their own colours, always with the word", () => {
    const { rerender } = render(<LeadCard lead={{ ...SAVED_SHOWN, status: "WARM", score: 60 }} />);
    expect(screen.getByText("WARM").className).toContain("badge-warm");
    rerender(<LeadCard lead={{ ...SAVED_SHOWN, status: "COLD", score: 30 }} />);
    expect(screen.getByText("COLD").className).toContain("badge-cold");
  });

  it("gives the phone strip a short headline", () => {
    expect(leadHeadline(null)).toBe("no lead yet");
    expect(leadHeadline(DRAFT)).toBe("not saved yet");
    expect(leadHeadline(SAVED_HIDDEN)).toBe("lead saved, callback requested");
  });
});
