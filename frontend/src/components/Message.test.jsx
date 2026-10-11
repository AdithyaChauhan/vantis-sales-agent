import { render, screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import Message from "./Message.jsx";

const SOURCES = [
  { section: "8", title: "Product & Service Catalog", page: 3, snippet: "Vantis HomePro 5: 5 kW rooftop system" },
  { section: "9", title: "Package Guidance", page: 4, snippet: "Packages are reference options" },
];

const STEPS = [
  { node: "understand", detail: "intents: question", data: { latency_ms: 1200 } },
  { node: "retrieve", detail: "4 chunks (hybrid)", data: { latency_ms: 900 } },
  { node: "agent", detail: "gemini, 984 ms, calls check_service_area", data: { latency_ms: 984 } },
  { node: "tool", name: "check_service_area", detail: "Lucknow: listed", data: {} },
];

function assistant(extra = {}) {
  return { id: "1", role: "assistant", text: "Answer.", sources: [], steps: [], elapsedMs: 6400, ...extra };
}

describe("source chips", () => {
  it("shows one chip per source with its KB section, title and snippet", () => {
    render(<Message message={assistant({ sources: SOURCES })} />);
    const chips = within(screen.getByRole("list", { name: "Sources" })).getAllByRole("listitem");
    expect(chips).toHaveLength(2);
    expect(within(chips[0]).getByText("KB Section 8")).toBeTruthy();
    expect(within(chips[0]).getByText("Product & Service Catalog")).toBeTruthy();
    expect(within(chips[0]).getByText("Vantis HomePro 5: 5 kW rooftop system")).toBeTruthy();
    expect(within(chips[0]).getByText("Page 3 of the knowledge base")).toBeTruthy();
  });

  it("are collapsed until opened", () => {
    const { container } = render(<Message message={assistant({ sources: SOURCES })} />);
    expect(container.querySelector("details.chip").open).toBe(false);
  });

  it("are absent when a reply has no sources", () => {
    render(<Message message={assistant()} />);
    expect(screen.queryByRole("list", { name: "Sources" })).toBeNull();
  });
});

describe("How I answered", () => {
  it("is collapsed and lists each step with its timing", () => {
    const { container } = render(<Message message={assistant({ steps: STEPS })} />);
    const panel = container.querySelector("details.steps");
    expect(panel.open).toBe(false);
    expect(within(panel).getByText("How I answered (4 steps, 6.4 s)")).toBeTruthy();
    expect(within(panel).getByText("Understood the message")).toBeTruthy();
    expect(within(panel).getByText("1200 ms")).toBeTruthy();
    expect(within(panel).queryByText("984 ms")).toBeNull(); // already in the model call's detail
    expect(within(panel).getByText("Tool: check_service_area")).toBeTruthy();
    expect(within(panel).getByText("Lucknow: listed")).toBeTruthy();
  });

  it("says when a fallback was used", () => {
    render(<Message message={assistant({ steps: STEPS, degraded: true })} />);
    expect(screen.getByText("Fallback used")).toBeTruthy();
  });

  it("is not shown on the customer's own messages", () => {
    const { container } = render(<Message message={{ id: "2", role: "user", text: "<b>hi</b>" }} />);
    expect(container.querySelector("details")).toBeNull();
    expect(container.querySelector("b")).toBeNull();
    expect(screen.getByText("<b>hi</b>")).toBeTruthy();
  });
});
