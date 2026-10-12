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
    expect(chips[0].querySelector(".chip-section").textContent).toBe("KB Section 8"); // shown as "KB 8"
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
    expect(within(panel).getByText("How I answered · 4 steps · 6.4 s")).toBeTruthy();
    expect(within(panel).getByText("Understood the message")).toBeTruthy();
    expect(within(panel).getByText("1.2 s")).toBeTruthy(); // every time in seconds
    expect(within(panel).getByText("1.0 s")).toBeTruthy(); // the model call's 984 ms
    expect(within(panel).queryByText(/\d+ ms/)).toBeNull();
    expect(within(panel).getByText("gemini, calls check_service_area")).toBeTruthy();
    expect(within(panel).getByText("Tool: check_service_area")).toBeTruthy();
    expect(within(panel).getByText("Lucknow: listed")).toBeTruthy();
  });

  it("names the city check the code made before the model (T12b)", () => {
    const steps = [{ node: "area_check", detail: "Kanpur: listed", data: { by: "code" } }];
    const { container } = render(<Message message={assistant({ steps })} />);
    const panel = container.querySelector("details.steps");
    expect(within(panel).getByText("Service area check")).toBeTruthy();
    expect(within(panel).getByText("Kanpur: listed")).toBeTruthy();
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

describe("time", () => {
  it("shows when a message was sent, as a <time> with the exact moment", () => {
    const at = "2026-10-12T05:32:00.000Z";
    const { container } = render(<Message message={assistant({ at })} />);
    expect(container.querySelector("time").getAttribute("datetime")).toBe(at);
    expect(container.querySelector("time").textContent).toMatch(/\d{1,2}[:.]\d{2}/);
  });

  it("shows nothing for a message saved without a time", () => {
    const { container } = render(<Message message={{ id: "3", role: "user", text: "Old message" }} />);
    expect(container.querySelector("time")).toBeNull();
  });
});
