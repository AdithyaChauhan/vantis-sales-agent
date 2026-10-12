import { StrictMode } from "react";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";

import App from "./App.jsx";
import { navigate } from "./router.jsx";
import { chatReply, fakeApi, jsonResponse } from "./test-utils.js";
import { STILL_WORKING } from "./useChat.js";
import { WAKE_NOTE } from "./useServerWake.js";

// T13a: one health ping when the page loads; the note shows only while that ping is still waiting.

const FAKE_TIMERS = { toFake: ["setTimeout", "clearTimeout"] };
const advance = (ms) => act(() => vi.advanceTimersByTimeAsync(ms));
const healthCalls = (fetchMock) => fetchMock.mock.calls.filter(([url]) => url.endsWith("/api/health/live"));

afterEach(() => {
  vi.useRealTimers();
});

/** A fake server whose health ping answers only when `wake()` is called (a sleeping Render service). */
function sleepingServer(extra = {}) {
  let wake;
  const fetchMock = fakeApi({ health: () => new Promise((resolve) => (wake = resolve)), ...extra });
  return { fetchMock, wake: () => wake(jsonResponse(200, { status: "ok" })) };
}

describe("waking note", () => {
  it("shows after 1.5 s while the ping waits, and goes when the server answers", async () => {
    vi.useFakeTimers(FAKE_TIMERS);
    const server = sleepingServer();
    render(<App />);
    await advance(1400);
    expect(screen.queryByText(WAKE_NOTE)).toBeNull();
    await advance(100);
    expect(screen.getByText(WAKE_NOTE)).toBeTruthy();
    expect(screen.getByText(WAKE_NOTE).closest("[role=status]")).toBeTruthy();
    await advance(30_000); // a sleeping Render service takes 30-40 s
    expect(screen.getByText(WAKE_NOTE)).toBeTruthy();
    await act(async () => server.wake());
    expect(screen.queryByText(WAKE_NOTE)).toBeNull();
  });

  it("goes quietly when the ping fails", async () => {
    vi.useFakeTimers(FAKE_TIMERS);
    let fail;
    fakeApi({ health: () => new Promise((_, reject) => (fail = reject)) });
    render(<App />);
    await advance(2000);
    expect(screen.getByText(WAKE_NOTE)).toBeTruthy();
    await act(async () => fail(new TypeError("Failed to fetch")));
    expect(screen.queryByText(WAKE_NOTE)).toBeNull();
    expect(screen.queryByRole("alert")).toBeNull(); // no error shown for the ping
  });

  it("never shows when the server answers at once", async () => {
    vi.useFakeTimers(FAKE_TIMERS);
    fakeApi();
    render(<App />);
    await advance(5000);
    expect(screen.queryByText(WAKE_NOTE)).toBeNull();
  });

  it("sends one ping per page load, also in StrictMode and after moving between pages", async () => {
    const fetchMock = fakeApi();
    render(
      <StrictMode>
        <App />
      </StrictMode>,
    );
    await act(async () => navigate("/leads"));
    await act(async () => navigate("/support"));
    await act(async () => navigate("/"));
    expect(healthCalls(fetchMock)).toHaveLength(1);
    expect(healthCalls(fetchMock)[0][0]).toBe("http://127.0.0.1:8000/api/health/live");
  });

  it("a slow chat reply after the server answered shows no waking note, only the neutral line", async () => {
    vi.useFakeTimers(FAKE_TIMERS);
    let answer;
    fakeApi({ chat: () => new Promise((resolve) => (answer = resolve)) });
    render(<App />);
    await advance(0);
    fireEvent.change(screen.getByLabelText("Your message"), { target: { value: "Hello" } });
    fireEvent.click(screen.getByRole("button", { name: "Send" }));
    await advance(20_000);
    expect(screen.queryByText(WAKE_NOTE)).toBeNull();
    expect(screen.getByText(STILL_WORKING)).toBeTruthy();
    await act(async () => answer(jsonResponse(200, chatReply("Here you go."))));
    expect(screen.getByText("Here you go.")).toBeTruthy();
  });
});
