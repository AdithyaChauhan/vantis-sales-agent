import { describe, expect, it, vi } from "vitest";

import {
  apiUrl,
  errorFrom,
  GENERIC_ERROR,
  getLead,
  getLeads,
  getSupportRequests,
  pingServer,
  postChat,
  REQUEST_TIMEOUT_MS,
  RETRY_DELAY_MS,
} from "./api.js";
import { chatReply, fakeApi, jsonResponse, SALES_KEY } from "./test-utils.js";

describe("apiUrl", () => {
  it.each([
    ["http://localhost:8000", "http://localhost:8000/api/chat"],
    ["http://localhost:8000/", "http://localhost:8000/api/chat"],
    ["https://x.onrender.com//", "https://x.onrender.com/api/chat"],
    [" https://x.onrender.com/ ", "https://x.onrender.com/api/chat"],
  ])("VITE_API_URL %j gives one slash before /api", (base, expected) => {
    vi.stubEnv("VITE_API_URL", base);
    expect(apiUrl("/api/chat")).toBe(expected);
  });

  it("defaults to the local backend", () => {
    vi.stubEnv("VITE_API_URL", "");
    expect(apiUrl("/api/welcome")).toBe("http://127.0.0.1:8000/api/welcome");
  });
});

describe("errorFrom", () => {
  it("429 takes the detail and retry_after from the body", () => {
    const error = errorFrom(new Response(null, { status: 429 }), { detail: "Please wait 12 seconds.", retry_after: 12 });
    expect(error).toMatchObject({ kind: "rate_limit", message: "Please wait 12 seconds.", retryAfter: 12 });
  });

  it("429 falls back to the Retry-After header", () => {
    const response = new Response(null, { status: 429, headers: { "Retry-After": "7" } });
    expect(errorFrom(response, { detail: "Slow down." }).retryAfter).toBe(7);
  });

  it("422 shows the first field message", () => {
    const body = { detail: "Some values are not valid.", errors: { message: "message must be 1-1000 characters after trimming" } };
    const error = errorFrom(new Response(null, { status: 422 }), body);
    expect(error).toMatchObject({ kind: "invalid", field: "message", message: "message must be 1-1000 characters after trimming" });
  });

  it("other statuses get the short generic message, not the server's text", () => {
    const error = errorFrom(new Response(null, { status: 503 }), { detail: "database down" });
    expect(error).toMatchObject({ kind: "server", status: 503, message: GENERIC_ERROR });
  });

  it("keeps the API's detail for the dashboard (the chat still shows its short message)", () => {
    const error = errorFrom(new Response(null, { status: 503 }), { detail: "The sales API is not configured on this server." });
    expect(error).toMatchObject({ message: GENERIC_ERROR, detail: "The sales API is not configured on this server." });
  });

  it("401 and 404 get their own kinds", () => {
    expect(errorFrom(new Response(null, { status: 401 }), { detail: "Missing or invalid API key." })).toMatchObject({
      kind: "unauthorized",
      status: 401,
      message: GENERIC_ERROR,
    });
    expect(errorFrom(new Response(null, { status: 404 }), { detail: "No lead with id 9." })).toMatchObject({
      kind: "not_found",
      status: 404,
    });
  });
});

describe("sales calls", () => {
  it("send the key in the X-API-Key header, never in the URL, with only the parameters that are set", async () => {
    vi.stubEnv("VITE_API_URL", "");
    const fetchMock = fakeApi();
    await getLeads({ status: "HOT", q: "", sort: "score", limit: 50, offset: 0 }, SALES_KEY);
    await getLead(7, SALES_KEY);
    await getSupportRequests({ limit: 50, offset: 50 }, SALES_KEY);
    expect(fetchMock.mock.calls.map(([url]) => url)).toEqual([
      "http://127.0.0.1:8000/api/leads?status=HOT&sort=score&limit=50&offset=0",
      "http://127.0.0.1:8000/api/leads/7",
      "http://127.0.0.1:8000/api/support-requests?limit=50&offset=50",
    ]);
    for (const [url, init] of fetchMock.mock.calls) {
      expect(init.headers).toEqual({ "X-API-Key": SALES_KEY });
      expect(url).not.toContain(SALES_KEY);
    }
  });

  it("treat an answer without the expected shape as a server error", async () => {
    fakeApi({
      leads: () => jsonResponse(200, { items: [], total: 0 }), // no counts
      lead: () => jsonResponse(200, { id: 7 }), // no score breakdown
      support: () => jsonResponse(200, { total: 0 }), // no items
    });
    await expect(getLeads({}, SALES_KEY)).rejects.toMatchObject({ kind: "server" });
    await expect(getLead(7, SALES_KEY)).rejects.toMatchObject({ kind: "server" });
    await expect(getSupportRequests({}, SALES_KEY)).rejects.toMatchObject({ kind: "server" });
  });

  it("a wrong key is an unauthorized error", async () => {
    fakeApi();
    await expect(getLeads({}, "not-the-key")).rejects.toMatchObject({ kind: "unauthorized", status: 401 });
  });
});

const UUID_V4 = /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/;

/** Run a postChat call with fake timers, so the retry's pause passes at once. */
async function withFakeTimers(call) {
  vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
  try {
    const result = call().then(
      (value) => ({ value }),
      (error) => ({ error }),
    );
    await vi.advanceTimersByTimeAsync(RETRY_DELAY_MS + 10);
    return await result;
  } finally {
    vi.useRealTimers();
  }
}

describe("postChat", () => {
  it("leaves session_id out of the first message and gives each message a random id", async () => {
    const fetchMock = fakeApi({ chat: () => jsonResponse(200, { session_id: "s", reply: "ok" }) });
    await postChat({ message: "hi", sessionId: null });
    await postChat({ message: "again", sessionId: "s" });
    const [first, second] = fetchMock.mock.calls.map(([, init]) => JSON.parse(init.body));
    expect(first).toEqual({ message: "hi", channel: "web", client_message_id: expect.stringMatching(UUID_V4) });
    expect(second.client_message_id).not.toBe(first.client_message_id);
  });

  it("turns a network failure into a friendly error after one retry with the same id (T13d)", async () => {
    const fetchMock = vi.fn(() => Promise.reject(new TypeError("Failed to fetch")));
    vi.stubGlobal("fetch", fetchMock);
    const { error } = await withFakeTimers(() => postChat({ message: "hi", sessionId: null }));
    expect(error).toMatchObject({ kind: "network", message: GENERIC_ERROR });
    expect(fetchMock).toHaveBeenCalledTimes(2);
    const ids = fetchMock.mock.calls.map(([, init]) => JSON.parse(init.body).client_message_id);
    expect(ids[0]).toMatch(UUID_V4);
    expect(ids[1]).toBe(ids[0]);
  });

  it.each([
    ["a network error", () => Promise.reject(new TypeError("Failed to fetch"))],
    ["a proxy's 502 page", async () => new Response("<html>Bad gateway</html>", { status: 502 })],
    ["a proxy's 503 page", async () => new Response("upstream unavailable", { status: 503 })],
  ])("sends a message again once after %s, and the reply that comes then is shown", async (_, firstAnswer) => {
    let calls = 0;
    const fetchMock = vi.fn(() => (++calls === 1 ? firstAnswer() : Promise.resolve(jsonResponse(200, chatReply("ok")))));
    vi.stubGlobal("fetch", fetchMock);
    const { value } = await withFakeTimers(() => postChat({ message: "hi", sessionId: null }));
    expect(value).toMatchObject({ reply: "ok" });
    expect(fetchMock).toHaveBeenCalledTimes(2);
  });

  it.each([
    ["our own 503 (the database is down)", () => jsonResponse(503, { detail: "We can't reach our database right now." })],
    ["a 422", () => jsonResponse(422, { detail: "x", errors: { message: "too long" } })],
    ["a 429", () => jsonResponse(429, { detail: "Slow down.", retry_after: 5 })],
  ])("never sends again after %s: the server answered", async (_, answer) => {
    const fetchMock = vi.fn(async () => answer());
    vi.stubGlobal("fetch", fetchMock);
    const { error } = await withFakeTimers(() => postChat({ message: "hi", sessionId: null }));
    expect(error).toBeTruthy();
    expect(fetchMock).toHaveBeenCalledTimes(1);
  });

  it("treats a 200 that isn't JSON as a server error", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("<html>proxy page</html>", { status: 200 })));
    await expect(postChat({ message: "hi", sessionId: null })).rejects.toMatchObject({ kind: "server" });
  });
});

describe("timeouts and the health ping (T13a)", () => {
  it("no request gives up before 60 s: a chat reply at 61 s still arrives", async () => {
    expect(REQUEST_TIMEOUT_MS).toBeGreaterThanOrEqual(60_000);
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    try {
      vi.stubGlobal(
        "fetch",
        vi.fn(
          (url, init) =>
            new Promise((resolve, reject) => {
              const late = setTimeout(() => resolve(jsonResponse(200, chatReply("late but fine"))), 61_000);
              init.signal.addEventListener("abort", () => {
                clearTimeout(late);
                reject(new DOMException("aborted", "AbortError"));
              });
            }),
        ),
      );
      const reply = postChat({ message: "hi", sessionId: null });
      await vi.advanceTimersByTimeAsync(61_000);
      await expect(reply).resolves.toMatchObject({ reply: "late but fine" });
    } finally {
      vi.useRealTimers();
    }
  });

  it("pingServer asks /api/health/live once per page load and never throws", async () => {
    vi.stubEnv("VITE_API_URL", "https://x.onrender.com/");
    const fetchMock = vi.fn(() => Promise.reject(new TypeError("Failed to fetch")));
    vi.stubGlobal("fetch", fetchMock);
    await expect(pingServer()).resolves.toBe(false);
    await expect(pingServer()).resolves.toBe(false);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe("https://x.onrender.com/api/health/live");
  });
});
