import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App.jsx";
import { chatPayloads, chatReply, fakeApi, jsonResponse, SESSION, WELCOME } from "./test-utils.js";

const FAKE_TIMERS = { toFake: ["setTimeout", "clearTimeout", "setInterval", "clearInterval"] };
const RATE_DETAIL = "You're sending messages a little fast. Please wait 3 seconds and try again.";

afterEach(() => {
  vi.useRealTimers();
});

const box = () => screen.getByLabelText("Your message");
const sendButton = () => screen.getByRole("button", { name: /^(Send|Wait .*)$/ });

function type(text) {
  fireEvent.change(box(), { target: { value: text } });
}

function send(text) {
  type(text);
  fireEvent.click(screen.getByRole("button", { name: "Send" }));
}

/** Let pending promises (fake fetch, state updates) settle while timers are faked. */
const settle = (ms = 0) => act(() => vi.advanceTimersByTimeAsync(ms));

describe("welcome", () => {
  it("shows the welcome text and starter questions, and a starter sends its question", async () => {
    const fetchMock = fakeApi({ chat: () => jsonResponse(200, chatReply("We offer three packages.")) });
    render(<App />);
    expect(await screen.findByText(WELCOME.text)).toBeTruthy();
    fireEvent.click(screen.getByRole("button", { name: WELCOME.starter_questions[0] }));
    expect(await screen.findByText("We offer three packages.")).toBeTruthy();
    expect(chatPayloads(fetchMock)[0].message).toBe(WELCOME.starter_questions[0]);
    // Starters are only offered before the conversation starts
    expect(screen.queryByRole("button", { name: WELCOME.starter_questions[1] })).toBeNull();
  });

  it("says the server is waking up when the welcome takes more than 3 seconds", async () => {
    vi.useFakeTimers(FAKE_TIMERS);
    let answer;
    fakeApi({ welcome: () => new Promise((resolve) => (answer = resolve)) });
    render(<App />);
    await settle(2900);
    expect(screen.queryByText(/Waking up the server/)).toBeNull();
    await settle(100);
    expect(screen.getByText("Waking up the server. This can take up to a minute.")).toBeTruthy();
    await act(async () => answer(jsonResponse(200, WELCOME)));
    expect(screen.queryByText(/Waking up the server/)).toBeNull();
    expect(screen.getByText(WELCOME.text)).toBeTruthy();
    expect(screen.getByRole("button", { name: WELCOME.starter_questions[0] })).toBeTruthy();
  });

  it("offers to try again when the server can't be reached", async () => {
    let calls = 0;
    fakeApi({
      welcome: () => {
        calls += 1;
        if (calls === 1) throw new TypeError("Failed to fetch");
        return jsonResponse(200, WELCOME);
      },
    });
    render(<App />);
    fireEvent.click(await screen.findByRole("button", { name: "Try again" }));
    expect(await screen.findByText(WELCOME.text)).toBeTruthy();
  });
});

describe("session", () => {
  it("sends no session_id first, then the server's id, also after a reload; New chat starts fresh", async () => {
    const fetchMock = fakeApi({ chat: (_payload, n) => jsonResponse(200, chatReply(`Reply ${n}`)) });
    render(<App />);
    await screen.findByText(WELCOME.text);

    send("First question");
    await screen.findByText("Reply 1");
    send("Second question");
    await screen.findByText("Reply 2");

    // Reload: the transcript and the session come back from sessionStorage
    cleanup();
    render(<App />);
    expect(await screen.findByText("Reply 2")).toBeTruthy();
    expect(screen.getByText("First question")).toBeTruthy();
    send("Third question");
    await screen.findByText("Reply 3");

    fireEvent.click(screen.getByRole("button", { name: "New chat" }));
    expect(screen.queryByText("Reply 3")).toBeNull();
    send("Fresh start");
    await screen.findByText("Reply 4");

    expect(chatPayloads(fetchMock)).toEqual([
      { message: "First question", channel: "web" },
      { message: "Second question", channel: "web", session_id: SESSION },
      { message: "Third question", channel: "web", session_id: SESSION },
      { message: "Fresh start", channel: "web" },
    ]);
  });
});

describe("sending", () => {
  it("Enter sends, Shift+Enter does not", async () => {
    const fetchMock = fakeApi({ chat: () => jsonResponse(200, chatReply("Hello!")) });
    render(<App />);
    await screen.findByText(WELCOME.text);
    type("Hi");
    fireEvent.keyDown(box(), { key: "Enter", shiftKey: true });
    expect(chatPayloads(fetchMock)).toHaveLength(0);
    fireEvent.keyDown(box(), { key: "Enter" });
    expect(await screen.findByText("Hello!")).toBeTruthy();
    expect(chatPayloads(fetchMock)).toHaveLength(1);
  });

  it("allows one message at a time and explains a slow first answer", async () => {
    vi.useFakeTimers(FAKE_TIMERS);
    let answer;
    const fetchMock = fakeApi({ chat: () => new Promise((resolve) => (answer = resolve)) });
    render(<App />);
    await settle();
    send("Hello");
    await settle();
    expect(sendButton().disabled).toBe(true);
    expect(box().readOnly).toBe(true);
    fireEvent.keyDown(box(), { key: "Enter" });
    expect(chatPayloads(fetchMock)).toHaveLength(1);
    expect(screen.queryByText(/about 20 seconds/)).toBeNull();
    await settle(8000);
    expect(screen.getByText(/The first answer after the server wakes up can take about 20 seconds/)).toBeTruthy();
    await act(async () => answer(jsonResponse(200, chatReply("Here you go."))));
    expect(screen.getByText("Here you go.")).toBeTruthy();
    expect(box().readOnly).toBe(false);
  });
});

describe("errors", () => {
  it("429: shows the API's message, counts down, then lets the customer send again", async () => {
    vi.useFakeTimers(FAKE_TIMERS);
    fakeApi({
      chat: () => jsonResponse(429, { detail: RATE_DETAIL, retry_after: 3 }, { "Retry-After": "3" }),
    });
    render(<App />);
    await settle();
    send("Do you install in Pune?");
    await settle();

    expect(screen.getByRole("alert").textContent).toBe(RATE_DETAIL);
    expect(screen.getByText("You can send again in 3 s.")).toBeTruthy();
    expect(sendButton().textContent).toBe("Wait 3 s");
    expect(sendButton().disabled).toBe(true);
    expect(box().value).toBe("Do you install in Pune?"); // the typed message is not lost
    expect(screen.queryByText("Do you install in Pune?", { selector: "p" })).toBeNull(); // no sent bubble left

    await settle(1000);
    expect(screen.getByText("You can send again in 2 s.")).toBeTruthy();
    await settle(1000);
    expect(screen.getByText("You can send again in 1 s.")).toBeTruthy();
    await settle(1000);
    expect(screen.queryByText(/You can send again/)).toBeNull();
    expect(sendButton().textContent).toBe("Send");
    expect(sendButton().disabled).toBe(false);
  });

  it("422: shows the field's message and keeps the text", async () => {
    const message = "message must be 1-1000 characters after trimming";
    fakeApi({ chat: () => jsonResponse(422, { detail: "Some values are not valid.", errors: { message } }) });
    render(<App />);
    await screen.findByText(WELCOME.text);
    send("Hello");
    expect((await screen.findByRole("alert")).textContent).toBe(message);
    expect(box().value).toBe("Hello");
  });

  it("422 on session_id: the next message starts a new chat", async () => {
    sessionStorage.setItem("vantis-chat", JSON.stringify({ sessionId: "not-a-uuid", messages: [], lead: null }));
    const fetchMock = fakeApi({
      chat: (_payload, n) =>
        n === 1
          ? jsonResponse(422, { detail: "Some values are not valid.", errors: { session_id: "session_id must be ..." } })
          : jsonResponse(200, chatReply("Welcome back.")),
    });
    render(<App />);
    await screen.findByText(WELCOME.text);
    send("Hello");
    await screen.findByRole("alert");
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await screen.findByText("Welcome back.");
    expect(chatPayloads(fetchMock).map((p) => p.session_id)).toEqual(["not-a-uuid", undefined]);
  });

  it("network failure: a short 'please try again', the text is kept, nothing crashes", async () => {
    fakeApi({
      chat: () => {
        throw new TypeError("Failed to fetch");
      },
    });
    render(<App />);
    await screen.findByText(WELCOME.text);
    send("Is there a warranty?");
    expect((await screen.findByRole("alert")).textContent).toBe("Something went wrong. Please try again.");
    expect(box().value).toBe("Is there a warranty?");
    expect(sendButton().disabled).toBe(false);
  });

  it("a server error shows the same short message", async () => {
    fakeApi({ chat: () => jsonResponse(503, { detail: "Our knowledge base is not reachable right now." }) });
    render(<App />);
    await screen.findByText(WELCOME.text);
    send("Hello");
    expect((await screen.findByRole("alert")).textContent).toBe("Something went wrong. Please try again.");
  });
});

describe("reply and lead", () => {
  it("shows sources, steps and the lead card from the reply", async () => {
    const lead = { id: 12, status: null, score: null, missing_fields: [], callback_status: "requested" };
    fakeApi({
      chat: () =>
        jsonResponse(
          200,
          chatReply("I've logged a callback request for tomorrow evening; our team will confirm.", {
            sources: [{ section: "10", title: "Reference Service Locations", page: 3, snippet: "Lucknow" }],
            steps: [{ node: "tool", name: "schedule_callback", detail: "requested", data: {} }],
            lead,
          }),
        ),
    });
    render(<App />);
    await screen.findByText(WELCOME.text);
    send("Yes, call me tomorrow evening.");
    await screen.findByText(/logged a callback request/);
    expect(screen.getByText("Reference Service Locations")).toBeTruthy();
    expect(screen.getByText("Tool: schedule_callback")).toBeTruthy();
    expect(screen.getByText("Saved as lead #12")).toBeTruthy();
    expect(screen.getByText("Requested, the team will confirm")).toBeTruthy();
  });
});
