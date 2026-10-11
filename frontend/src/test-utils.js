import { vi } from "vitest";

export const SESSION = "6f1c2a90-1b2c-4d3e-8f40-123456789abc";
export const WELCOME = {
  text: "Hi! I'm the Vantis Energy Solutions assistant. What would you like to know?",
  starter_questions: ["What solar systems do you offer for homes?", "Do you install in my city?"],
};

/**
 * What fetch resolves to, with a JSON body. A plain object rather than a real Response: its json()
 * resolves through promises only, so tests with fake timers never wait on Node's stream internals.
 */
export function jsonResponse(status, body, headers = {}) {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ "Content-Type": "application/json", ...headers }),
    json: async () => body,
  };
}

/** A /api/chat 200 body. */
export function chatReply(reply, extra = {}) {
  return { session_id: SESSION, reply, sources: [], steps: [], lead: null, degraded: false, ...extra };
}

/**
 * Replace fetch with a fake API. welcome() and chat(payload, callNumber) return a Response (or a
 * promise of one, or throw). Returns the mock, so tests can read the chat payloads that were sent.
 */
export function fakeApi({ welcome = () => jsonResponse(200, WELCOME), chat } = {}) {
  let chatCalls = 0;
  const fetchMock = vi.fn(async (url, init = {}) => {
    if (url.endsWith("/api/welcome")) return welcome();
    if (url.endsWith("/api/chat")) return chat(JSON.parse(init.body), ++chatCalls);
    throw new Error(`unexpected URL ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

/** The JSON bodies sent to /api/chat so far. */
export function chatPayloads(fetchMock) {
  return fetchMock.mock.calls.filter(([url]) => url.endsWith("/api/chat")).map(([, init]) => JSON.parse(init.body));
}
