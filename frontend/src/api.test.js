import { describe, expect, it, vi } from "vitest";

import { apiUrl, errorFrom, GENERIC_ERROR, postChat } from "./api.js";
import { fakeApi, jsonResponse } from "./test-utils.js";

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
});

describe("postChat", () => {
  it("leaves session_id out of the first message", async () => {
    const fetchMock = fakeApi({ chat: () => jsonResponse(200, { session_id: "s", reply: "ok" }) });
    await postChat({ message: "hi", sessionId: null });
    expect(JSON.parse(fetchMock.mock.calls[0][1].body)).toEqual({ message: "hi", channel: "web" });
  });

  it("turns a network failure into a friendly error", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("Failed to fetch"))));
    await expect(postChat({ message: "hi", sessionId: null })).rejects.toMatchObject({ kind: "network", message: GENERIC_ERROR });
  });

  it("treats a 200 that isn't JSON as a server error", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("<html>proxy page</html>", { status: 200 })));
    await expect(postChat({ message: "hi", sessionId: null })).rejects.toMatchObject({ kind: "server" });
  });
});
