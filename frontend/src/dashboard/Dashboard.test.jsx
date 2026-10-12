import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "../App.jsx";
import {
  fakeApi,
  jsonResponse,
  leadDetail,
  leadItem,
  leadList,
  phoneScreen,
  SALES_KEY,
  salesCalls,
  supportItem,
} from "../test-utils.js";
import { WAKE_NOTE } from "../useServerWake.js";
import { KEY_STOPPED } from "./Dashboard.jsx";
import { NO_LEADS, NO_MATCH } from "./LeadsView.jsx";
import { NO_SUPPORT } from "./SupportView.jsx";

afterEach(() => {
  vi.useRealTimers();
});

/** Render the app at a path, with the sales key already in this tab (or not). */
function openAt(path, { unlocked = true } = {}) {
  window.history.pushState(null, "", path);
  if (unlocked) sessionStorage.setItem("vantis-sales-key", SALES_KEY);
  return render(<App />);
}

function unlock(key) {
  fireEvent.change(screen.getByLabelText("Sales API key"), { target: { value: key } });
  fireEvent.click(screen.getByRole("button", { name: "Unlock" }));
}

const leadsTable = () => screen.findByRole("table", { name: "Leads" });
const bodyRows = (table) => within(table).getAllByRole("row").slice(1);
const searchBox = () => screen.getByRole("searchbox", { name: "Search leads by name or city" });
const lastLeadsCall = (fetchMock) => salesCalls(fetchMock).filter((call) => call.path === "/api/leads").at(-1);

async function detailPanel() {
  const panel = await screen.findByRole("complementary", { name: "Lead details" });
  await within(panel).findByText("Why this score");
  return panel;
}

describe("unlock", () => {
  it("asks for the key first; a wrong key (401) says so and nothing is kept", async () => {
    const fetchMock = fakeApi();
    openAt("/leads", { unlocked: false });
    expect(screen.getByRole("heading", { name: "Sales dashboard" })).toBeTruthy();
    expect(screen.getByRole("button", { name: "Unlock" }).disabled).toBe(true); // nothing typed yet
    expect(salesCalls(fetchMock)).toEqual([]); // no sales call without a key

    unlock("wrong-key");
    expect((await screen.findByRole("alert")).textContent).toBe("That key didn't work.");
    expect(sessionStorage.getItem("vantis-sales-key")).toBeNull();
    expect(screen.queryByRole("table")).toBeNull();
  });

  it("429: says how long to wait, from retry_after or the Retry-After header", async () => {
    let body = { detail: "Too many wrong API keys from your network.", retry_after: 840 };
    fakeApi({ leads: () => jsonResponse(429, body, { "Retry-After": String(body.retry_after ?? 90) }) });
    openAt("/leads", { unlocked: false });
    unlock(SALES_KEY);
    expect((await screen.findByRole("alert")).textContent).toBe(
      "Too many wrong keys from this network. Try again in 14 min.",
    );

    body = { detail: "Too many wrong API keys from your network." }; // header only
    fireEvent.click(screen.getByRole("button", { name: "Unlock" }));
    await waitFor(() =>
      expect(screen.getByRole("alert").textContent).toBe("Too many wrong keys from this network. Try again in 90 s."),
    );
  });

  it("503: says plainly that the server isn't set up, in the API's words", async () => {
    fakeApi({ leads: () => jsonResponse(503, { detail: "The sales API is not configured on this server." }) });
    openAt("/leads", { unlocked: false });
    unlock(SALES_KEY);
    expect((await screen.findByRole("alert")).textContent).toBe("The sales API is not configured on this server.");
    expect(sessionStorage.getItem("vantis-sales-key")).toBeNull();
  });

  it("keeps the key in sessionStorage only, sends it as X-API-Key (never in a URL), and Lock forgets it", async () => {
    const fetchMock = fakeApi();
    openAt("/leads", { unlocked: false });
    unlock(`  ${SALES_KEY}  `); // pasted with spaces
    await leadsTable();
    await detailPanel();

    expect(sessionStorage.getItem("vantis-sales-key")).toBe(SALES_KEY);
    expect(Object.keys(localStorage).filter((name) => localStorage.getItem(name).includes(SALES_KEY))).toEqual([]);
    expect(window.location.href).not.toContain(SALES_KEY);
    const calls = salesCalls(fetchMock);
    expect(calls.length).toBeGreaterThan(1);
    for (const call of calls) {
      expect(call.key).toBe(SALES_KEY);
      expect(call.url).not.toContain(SALES_KEY);
    }

    fireEvent.click(screen.getByRole("button", { name: "Lock" }));
    expect(sessionStorage.getItem("vantis-sales-key")).toBeNull();
    expect(screen.getByLabelText("Sales API key").value).toBe("");
    expect(screen.queryByText("Rahul Sharma")).toBeNull(); // the leads are gone from the page too
  });

  it("goes back to the unlock card when a later call gets a 401 (the key was changed on the server)", async () => {
    let accepted = true;
    fakeApi({
      leads: () =>
        accepted ? jsonResponse(200, leadList([leadItem()])) : jsonResponse(401, { detail: "Missing or invalid API key." }),
    });
    openAt("/leads");
    await leadsTable();
    accepted = false;
    fireEvent.click(screen.getByRole("button", { name: "Refresh" }));
    expect(await screen.findByText(KEY_STOPPED)).toBeTruthy();
    expect(screen.getByLabelText("Sales API key")).toBeTruthy();
    expect(sessionStorage.getItem("vantis-sales-key")).toBeNull();
  });
});

describe("leads", () => {
  it("count cards show the API's counts for the whole table, not the rows on this page", async () => {
    const page = [leadItem(), leadItem({ id: 2, name: "Priya Verma" })];
    fakeApi({ leads: () => jsonResponse(200, leadList(page, { total: 230, counts: { HOT: 120, WARM: 80, COLD: 30 } })) });
    openAt("/leads");
    const counts = await screen.findByRole("list", { name: "Lead counts" });
    expect(within(counts).getAllByRole("listitem").map((card) => card.textContent)).toEqual([
      "All leads230",
      "HOT120",
      "WARM80",
      "COLD30",
    ]);
    expect(bodyRows(await leadsTable())).toHaveLength(2);
    expect(screen.getByRole("button", { name: "Load more (2 of 230)" })).toBeTruthy();
  });

  it("the chips, search and sort headers ask the server, from the first page; Load more asks for the next", async () => {
    const fetchMock = fakeApi({
      leads: (params) => jsonResponse(200, leadList([leadItem({ id: Number(params.offset) + 1 })], { total: 3 })),
    });
    openAt("/leads");
    await leadsTable();
    expect(lastLeadsCall(fetchMock).params).toEqual({ sort: "newest", limit: "50", offset: "0" });

    fireEvent.click(screen.getByRole("button", { name: "HOT" }));
    await waitFor(() => expect(lastLeadsCall(fetchMock).params).toMatchObject({ status: "HOT", offset: "0" }));
    expect(screen.getByRole("button", { name: "HOT" }).getAttribute("aria-pressed")).toBe("true");

    fireEvent.change(searchBox(), { target: { value: "  pune " } });
    await waitFor(() => expect(lastLeadsCall(fetchMock).params).toMatchObject({ status: "HOT", q: "pune", offset: "0" }));

    fireEvent.click(screen.getByRole("button", { name: /^Score/ }));
    await waitFor(() =>
      expect(lastLeadsCall(fetchMock).params).toEqual({ status: "HOT", q: "pune", sort: "score", limit: "50", offset: "0" }),
    );
    expect(screen.getByRole("columnheader", { name: /^Score/ }).getAttribute("aria-sort")).toBe("descending");

    fireEvent.click(await screen.findByRole("button", { name: "Load more (1 of 3)" }));
    await waitFor(() => expect(lastLeadsCall(fetchMock).params).toMatchObject({ sort: "score", q: "pune", offset: "1" }));
    expect(await screen.findByRole("button", { name: "Load more (2 of 3)" })).toBeTruthy();
    expect(bodyRows(await leadsTable())).toHaveLength(2);
  });

  it("shows short versions in the table and the full text in the detail panel", async () => {
    const messy = leadItem({
      id: 2,
      name: "Rohit Patel",
      lead_score: 31,
      lead_status: "COLD",
      system_size_kw: null,
      monthly_bill_inr: null,
      requirement: "i want to install one at my organisation\nand maybe the warehouse too",
      recommended_action:
        "Add to the nurture list, share product information and follow up in 30 days. Customer asked for a callback: next year.",
    });
    fakeApi({
      leads: () => jsonResponse(200, leadList([leadItem(), messy])),
      lead: (id) => jsonResponse(200, id === 2 ? leadDetail(messy) : leadDetail()),
    });
    openAt("/leads");
    const [rahul, rohit] = bodyRows(await leadsTable());
    expect(within(rahul).getByText("5 kW house")).toBeTruthy(); // the size when it is known
    expect(within(rahul).getByText("Contact the lead within 1 hour and offer a site assessment.")).toBeTruthy();
    expect(within(rohit).getByText("i want to install one at my organisation")).toBeTruthy(); // the first line
    expect(within(rohit).getByText("Add to the nurture list, share product information and follow up in 30 days.")).toBeTruthy();

    // On a wide screen the first lead is open; its panel has the whole action
    let panel = await detailPanel();
    expect(within(panel).getByText(leadItem().recommended_action)).toBeTruthy();

    fireEvent.click(within(rohit).getByRole("link", { name: "Rohit Patel" }));
    expect(window.location.pathname).toBe("/leads/2");
    panel = await detailPanel();
    expect(await within(panel).findByText(messy.recommended_action)).toBeTruthy();
    expect(within(panel).getByText("i want to install one at my organisation and maybe the warehouse too")).toBeTruthy();
    expect(within(await leadsTable()).getByRole("link", { name: "Rohit Patel" }).getAttribute("aria-current")).toBe("true");
  });

  it("shows what to do first: next action, callback and contact, then why this score, reason and details", async () => {
    fakeApi();
    openAt("/leads/1");
    const panel = await detailPanel();
    const headings = within(panel).getAllByRole("heading", { level: 3 }).map((heading) => heading.textContent);
    expect(headings).toEqual(["Next action", "Contact", "Why this score", "Reason", "Details collected"]);
    // The callback row sits between the next action and the contact details
    const nextAction = within(panel).getByRole("heading", { name: "Next action" });
    const callback = within(panel).getByText("Callback:").closest(".detail-callback");
    const contact = within(panel).getByRole("heading", { name: "Contact" });
    const FOLLOWS = Node.DOCUMENT_POSITION_FOLLOWING;
    expect(nextAction.compareDocumentPosition(callback) & FOLLOWS).toBeTruthy();
    expect(callback.compareDocumentPosition(contact) & FOLLOWS).toBeTruthy();
  });

  it("explains the score: every rule's points out of its maximum, and a cap with the score before it", async () => {
    const breakdown = {
      rules: [
        { rule: "intent", value: "ready", points: 25, max: 25, label: "Ready to buy" },
        { rule: "location", value: "unverified", points: 5, max: 20, label: "City is not in the listed service areas" },
      ],
      raw_score: 85,
      caps: [{ rule: "unverified_location", label: "City not in the listed service areas: capped at WARM" }],
      lead_score: 74,
      lead_status: "WARM",
    };
    const sana = { id: 3, name: "Sana Khan", city: "Pune", lead_score: 74, lead_status: "WARM", location_status: "unverified" };
    fakeApi({
      leads: () => jsonResponse(200, leadList([leadItem(sana)])),
      lead: () => jsonResponse(200, leadDetail({ ...sana, score_breakdown: breakdown })),
    });
    openAt("/leads/3");
    const panel = await detailPanel();
    const why = within(panel).getByRole("heading", { name: "Why this score" }).nextElementSibling;
    expect(within(why).getAllByRole("listitem").map((row) => row.textContent)).toEqual([
      "Ready to buy25/25",
      "City is not in the listed service areas5/20",
      "City not in the listed service areas: capped at WARMScore before the cap: 85 → 74",
    ]);
    expect(within(panel).getByText("Pune, not in the listed service areas")).toBeTruthy();
  });

  it("says when a cap didn't change the score (a COLD lead in an unlisted city)", async () => {
    const breakdown = {
      rules: [{ rule: "intent", value: "exploring", points: 5, max: 25, label: "Only exploring" }],
      raw_score: 31,
      caps: [{ rule: "unverified_location", label: "City not in the listed service areas: capped at WARM" }],
      lead_score: 31,
      lead_status: "COLD",
    };
    fakeApi({ lead: () => jsonResponse(200, leadDetail({ lead_score: 31, lead_status: "COLD", score_breakdown: breakdown })) });
    openAt("/leads");
    const panel = await detailPanel();
    expect(within(panel).getByText("The cap didn't change this score.")).toBeTruthy();
    expect(within(panel).queryByText(/Score before the cap/)).toBeNull();
  });

  it("shows the API's score and status as they are; the page never works out a status", async () => {
    fakeApi({
      leads: () => jsonResponse(200, leadList([leadItem({ lead_score: 76, lead_status: "WARM" })])),
      lead: () => jsonResponse(200, leadDetail({ lead_score: 76, lead_status: "WARM" })),
    });
    openAt("/leads");
    const [row] = bodyRows(await leadsTable());
    expect(within(row).getByText("WARM")).toBeTruthy();
    expect(within(row).queryByText("HOT")).toBeNull(); // 76 would be HOT by the rules; the API said WARM
    const panel = await detailPanel();
    expect(within(panel).getByText("WARM")).toBeTruthy();
    expect(within(panel).getByText("76")).toBeTruthy();
  });

  it("tags demo leads, and only those", async () => {
    fakeApi({ leads: () => jsonResponse(200, leadList([leadItem(), leadItem({ id: 2, name: "Priya Verma", source_channel: "demo" })])) });
    openAt("/leads");
    const [web, demo] = bodyRows(await leadsTable());
    expect(within(web).queryByText("Demo")).toBeNull();
    expect(within(demo).getByText("Demo")).toBeTruthy();
  });

  it("shows contact details, the callback and every other known detail", async () => {
    fakeApi({ lead: () => jsonResponse(200, leadDetail({ ownership: "own", other_info: "Prefers WhatsApp" })) });
    openAt("/leads");
    const panel = await detailPanel();
    expect(within(panel).getByRole("link", { name: "+91 98765 43210" }).getAttribute("href")).toBe("tel:+919876543210");
    expect(within(panel).getByRole("link", { name: "rahul.sharma@example.com" }).getAttribute("href")).toBe(
      "mailto:rahul.sharma@example.com",
    );
    expect(within(panel).getByText(/requested, tomorrow evening/)).toBeTruthy();
    expect(within(panel).getByText("Owns the property")).toBeTruthy();
    expect(within(panel).getByText("Prefers WhatsApp")).toBeTruthy();
    expect(within(panel).getByText("Website chat")).toBeTruthy();
  });
});

describe("customer-typed text", () => {
  it("is always plain text, never HTML", async () => {
    const evil = "<img src=x onerror=alert(1)>";
    fakeApi({
      leads: () => jsonResponse(200, leadList([leadItem({ name: evil, system_size_kw: null, requirement: "<b>bold</b>" })])),
      lead: () =>
        jsonResponse(
          200,
          leadDetail({ name: evil, system_size_kw: null, requirement: "<b>bold</b>", other_info: "<script>alert(2)</script>" }),
        ),
      support: () =>
        jsonResponse(200, { items: [supportItem({ name: evil, issue: "<img src=y onerror=alert(3)>" })], total: 1, limit: 50, offset: 0 }),
    });
    const { container } = openAt("/leads");
    const panel = await detailPanel();
    expect(within(panel).getByText("<script>alert(2)</script>")).toBeTruthy();
    expect(screen.getAllByText(evil).length).toBeGreaterThanOrEqual(2); // the table and the panel
    expect(screen.getAllByText("<b>bold</b>").length).toBeGreaterThanOrEqual(2);
    expect(container.querySelectorAll("img, b, script")).toHaveLength(0);

    fireEvent.click(screen.getAllByRole("link", { name: "Support requests" })[0]);
    expect(await screen.findByText("<img src=y onerror=alert(3)>")).toBeTruthy();
    expect(screen.getByText(evil)).toBeTruthy();
    expect(container.querySelectorAll("img, b, script")).toHaveLength(0);
  });
});

describe("support requests", () => {
  it("lists who, the issue and when, newest first", async () => {
    const anil = supportItem({ id: 2, name: "Anil Kumar", phone: null, email: "anil@example.com", issue: "No output since the storm", project_details: null });
    fakeApi({ support: () => jsonResponse(200, { items: [supportItem(), anil], total: 2, limit: 50, offset: 0 }) });
    openAt("/support");
    const table = await screen.findByRole("table", { name: "Support requests" });
    expect(screen.getByRole("heading", { name: "Support requests" })).toBeTruthy();
    expect(within(table).getAllByRole("columnheader").map((th) => th.textContent)).toEqual(["Who", "Issue", "Created"]);
    const [meera, second] = bodyRows(table);
    expect(meera.textContent).toContain("Meera Iyer");
    expect(meera.textContent).toContain("+91 90000 00009 · meera.iyer@example.com");
    expect(meera.textContent).toContain("3 h ago");
    expect(second.textContent).toContain("No output since the storm");
    expect(screen.queryByRole("searchbox")).toBeNull(); // no search here
  });

  it("says when there are none, and offers Try again after an error", async () => {
    let failing = true;
    fakeApi({
      support: () =>
        failing
          ? jsonResponse(503, { detail: "The database is not reachable right now. Please try again in a minute." })
          : jsonResponse(200, { items: [], total: 0, limit: 50, offset: 0 }),
    });
    openAt("/support");
    expect((await screen.findByRole("alert")).textContent).toBe(
      "The database is not reachable right now. Please try again in a minute.",
    );
    failing = false;
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(await screen.findByText(NO_SUPPORT)).toBeTruthy();
  });
});

describe("states", () => {
  it("says the server is waking up while the page-load ping waits, then shows the leads (T13a)", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    let wake;
    let answer;
    fakeApi({
      health: () => new Promise((resolve) => (wake = resolve)),
      leads: () => new Promise((resolve) => (answer = resolve)),
    });
    openAt("/leads");
    await act(() => vi.advanceTimersByTimeAsync(1400));
    expect(screen.queryByText(WAKE_NOTE)).toBeNull();
    await act(() => vi.advanceTimersByTimeAsync(100));
    expect(screen.getByText(WAKE_NOTE)).toBeTruthy();
    await act(async () => wake(jsonResponse(200, { status: "ok" })));
    expect(screen.queryByText(WAKE_NOTE)).toBeNull();
    await act(async () => answer(jsonResponse(200, leadList([leadItem()]))));
    await act(() => vi.advanceTimersByTimeAsync(0));
    expect(screen.getByRole("table", { name: "Leads" })).toBeTruthy();
  });

  it("a slow leads request after the server answered shows no waking note (T13a)", async () => {
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    fakeApi({ leads: () => new Promise(() => {}) });
    openAt("/leads");
    await act(() => vi.advanceTimersByTimeAsync(20_000));
    expect(screen.queryByText(WAKE_NOTE)).toBeNull();
    expect(screen.getByText("Loading leads")).toBeTruthy();
  });

  it("empty: says how leads get here; a search with no match says so instead", async () => {
    fakeApi({
      leads: (params) =>
        jsonResponse(200, params.q ? leadList([], { counts: { HOT: 1, WARM: 0, COLD: 0 } }) : leadList([])),
    });
    openAt("/leads");
    expect(await screen.findByText(NO_LEADS)).toBeTruthy();
    fireEvent.change(searchBox(), { target: { value: "zz" } });
    expect(await screen.findByText(NO_MATCH)).toBeTruthy();
  });

  it("error: says so, and Try again loads the leads", async () => {
    let failing = true;
    fakeApi({
      leads: () => {
        if (failing) throw new TypeError("Failed to fetch");
        return jsonResponse(200, leadList([leadItem()]));
      },
    });
    openAt("/leads");
    expect((await screen.findByRole("alert")).textContent).toBe("Couldn't load the leads. Check your connection and try again.");
    failing = false;
    fireEvent.click(screen.getByRole("button", { name: "Try again" }));
    expect(bodyRows(await leadsTable())).toHaveLength(1);
  });
});

describe("phone", () => {
  it("shows cards, opens a lead full screen with Back, and keeps the list underneath", async () => {
    phoneScreen();
    fakeApi();
    openAt("/leads");
    const card = await screen.findByRole("link", { name: /Rahul Sharma/ });
    expect(screen.queryByRole("table")).toBeNull(); // cards, not a table
    expect(screen.queryByRole("complementary", { name: "Lead details" })).toBeNull(); // nothing open yet
    expect(within(screen.getByRole("list", { name: "Lead counts" })).getAllByRole("listitem")).toHaveLength(4);

    fireEvent.click(card);
    expect(window.location.pathname).toBe("/leads/1");
    const panel = await detailPanel();
    fireEvent.click(within(panel).getByRole("button", { name: "Back to leads" }));
    await waitFor(() => expect(window.location.pathname).toBe("/leads"));
    expect(screen.queryByRole("complementary", { name: "Lead details" })).toBeNull();
    expect(screen.getByRole("link", { name: /Rahul Sharma/ })).toBeTruthy();
  });
});

describe("controls", () => {
  it("has no button or link without an action", async () => {
    fakeApi({ leads: () => jsonResponse(200, leadList([leadItem()], { total: 2 })) });
    openAt("/leads");
    await leadsTable();
    await detailPanel();
    const buttons = screen.getAllByRole("button").map((button) => button.textContent.trim());
    expect(buttons.sort()).toEqual(
      [
        "All", "HOT", "WARM", "COLD", // the status filter
        "Score, sort highest first", "Created, sort newest first", // sorting on the server
        "Load more (1 of 2)", "Refresh", "Lock",
      ].sort(),
    );
    // Every link goes to a page that exists, the lead, or the customer's phone or email
    const links = screen.getAllByRole("link").map((link) => link.getAttribute("href"));
    for (const href of links) expect(href).toMatch(/^(\/|\/support|\/leads\/\d+|tel:\+91\d{10}|mailto:\S+@\S+)$/);
    expect(links).toEqual(
      expect.arrayContaining(["/", "/support", "/leads/1", "tel:+919876543210", "mailto:rahul.sharma@example.com"]),
    );
  });
});
