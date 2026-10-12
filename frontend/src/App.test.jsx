import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import App, { route } from "./App.jsx";
import { Link } from "./router.jsx";
import { chatReply, fakeApi, jsonResponse, WELCOME } from "./test-utils.js";

describe("route", () => {
  it.each([
    ["/", { page: "chat" }],
    ["/leads", { page: "leads", leadId: null }],
    ["/leads/", { page: "leads", leadId: null }],
    ["/leads/12", { page: "leads", leadId: 12 }],
    ["/support", { page: "support" }],
    ["/support/", { page: "support" }],
    ["/leads/0", { page: "missing" }],
    ["/leads/abc", { page: "missing" }],
    ["/leads/12/edit", { page: "missing" }],
    ["/admin", { page: "missing" }],
  ])("%s", (path, expected) => {
    expect(route(path)).toEqual(expected);
  });
});

describe("pages", () => {
  it("moves between the chat, the dashboard and a not-found page with links and the browser's Back", async () => {
    fakeApi({ chat: () => jsonResponse(200, chatReply("Hello!")) });
    window.history.pushState(null, "", "/nowhere");
    render(<App />);
    expect(screen.getByRole("heading", { name: "Page not found" })).toBeTruthy();
    expect(document.title).toBe("Page not found · Vantis");

    fireEvent.click(screen.getByRole("link", { name: "Go to the chat" }));
    expect(window.location.pathname).toBe("/");
    expect(await screen.findByText(WELCOME.text)).toBeTruthy();
    expect(document.title).toBe("Vantis Solar Assistant");

    // The chat's sidebar links to the dashboard while the Sales view is on (the default)
    fireEvent.click(screen.getByRole("link", { name: "Leads" }));
    expect(window.location.pathname).toBe("/leads");
    expect(screen.getByLabelText("Sales API key")).toBeTruthy(); // no key yet in this tab
    expect(document.title).toBe("Leads · Vantis sales dashboard");

    act(() => window.history.back());
    await waitFor(() => expect(window.location.pathname).toBe("/"));
    expect(await screen.findByText(WELCOME.text)).toBeTruthy();
  });
});

describe("Link", () => {
  it("navigates inside the app on a plain click and leaves other clicks (new tab) to the browser", () => {
    render(<Link to="/support">Support</Link>);
    const link = screen.getByRole("link", { name: "Support" });
    expect(link.getAttribute("href")).toBe("/support"); // a real link: it can be copied or opened in a tab

    // Record whether the app took each click, then stop jsdom from trying to follow the link
    const taken = [];
    const record = (event) => {
      taken.push(event.defaultPrevented);
      event.preventDefault();
    };
    document.addEventListener("click", record);
    try {
      for (const init of [{ ctrlKey: true }, { metaKey: true }, { shiftKey: true }, { altKey: true }, { button: 1 }]) {
        fireEvent.click(link, init);
        expect(window.location.pathname).toBe("/");
      }
      fireEvent.click(link);
    } finally {
      document.removeEventListener("click", record);
    }
    expect(taken).toEqual([false, false, false, false, false, true]);
    expect(window.location.pathname).toBe("/support");
  });
});
