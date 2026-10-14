/**
 * Calls to the Vantis API (backend/app/api). The base URL comes from VITE_API_URL; a trailing slash
 * is fine. Every failure becomes an ApiError whose message can be shown to the customer as it is.
 * The sales calls (dashboard) send the sales key in the X-API-Key header, never in the URL.
 */

// 127.0.0.1, not localhost: uvicorn listens on IPv4 only, and "localhost" can resolve to ::1 first
// (on this machine a Docker/WSL relay answers there with 404).
const DEFAULT_API_URL = "http://127.0.0.1:8000";
// Render's free plan sleeps; waking it took about 33 s, and a chat turn can wait on LLM retries.
export const REQUEST_TIMEOUT_MS = 90_000;
export const GENERIC_ERROR = "Something went wrong. Please try again.";
// T13d: a chat message that got no reply at all (a network error, our timeout, or a proxy's 502/503/504 without our
// JSON) is sent once more after this pause, with the same client_message_id, so the server can answer it with the reply
// it already made instead of running the turn twice. On 13 Oct a message at 12:01:08 UTC never reached the server.
export const RETRY_DELAY_MS = 1500;
const PROXY_STATUSES = [502, 503, 504];
const RATE_LIMIT_FALLBACK = "You're sending messages a little fast. Please wait a moment and try again.";

/** The full URL for an API path such as "/api/chat", without a double slash. */
export function apiUrl(path) {
  const base = (import.meta.env.VITE_API_URL || DEFAULT_API_URL).trim().replace(/\/+$/, "");
  return base + path;
}

/**
 * A failed call. kind: "rate_limit" (429), "invalid" (422), "unauthorized" (401), "not_found" (404),
 * "server" or "network". `detail` is the API's own message when it sent one (the dashboard shows a
 * 503's detail; the chat keeps its short message).
 */
export class ApiError extends Error {
  constructor(kind, message, { status = null, retryAfter = null, field = null, detail = null, noReply = false } = {}) {
    super(message);
    this.kind = kind;
    this.status = status;
    this.retryAfter = retryAfter; // seconds, for 429
    this.field = field; // the first invalid field, for 422
    this.detail = detail;
    this.noReply = noReply; // T13d: no answer from our API at all (network, timeout, a proxy's error page)
  }
}

function positiveSeconds(value) {
  const seconds = Math.ceil(Number(value));
  return Number.isFinite(seconds) && seconds > 0 ? seconds : null;
}

/** Turn an error response (and its JSON body, or null) into an ApiError. */
export function errorFrom(response, body) {
  const detail = typeof body?.detail === "string" ? body.detail : null;
  if (response.status === 429) {
    const retryAfter =
      positiveSeconds(body?.retry_after) ?? positiveSeconds(response.headers.get("Retry-After")) ?? 60;
    return new ApiError("rate_limit", detail || RATE_LIMIT_FALLBACK, { status: 429, retryAfter, detail });
  }
  if (response.status === 422) {
    const errors = body?.errors && typeof body.errors === "object" ? body.errors : {};
    const [field, message] = Object.entries(errors)[0] ?? [null, null];
    const text = typeof message === "string" ? message : GENERIC_ERROR;
    return new ApiError("invalid", text, { status: 422, field, detail });
  }
  if (response.status === 401) return new ApiError("unauthorized", GENERIC_ERROR, { status: 401, detail });
  if (response.status === 404) return new ApiError("not_found", GENERIC_ERROR, { status: 404, detail });
  const noReply = PROXY_STATUSES.includes(response.status) && body === null; // not our JSON: a proxy answered
  return new ApiError("server", GENERIC_ERROR, { status: response.status, detail, noReply });
}

async function request(path, options = {}) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
  try {
    let response;
    try {
      response = await fetch(apiUrl(path), { ...options, signal: controller.signal });
    } catch {
      throw new ApiError("network", GENERIC_ERROR, { noReply: true }); // offline, CORS refused, DNS, or our timeout
    }
    const body = await response.json().catch(() => null);
    if (!response.ok || body === null) throw errorFrom(response, body);
    return body;
  } finally {
    clearTimeout(timer);
  }
}

let ping = null; // the one health ping of this page load (pingServer)

/**
 * GET /api/health/live once per page load (T13a), so a sleeping Render server starts waking while the visitor
 * reads. Resolves to true when the server answered and false when it didn't; it never throws. Later calls get the
 * same promise, so React's double effects in development and moving between pages send no second ping.
 */
export function pingServer() {
  if (!ping) {
    ping = request("/api/health/live").then(
      () => true,
      () => false,
    );
  }
  return ping;
}

/** Tests only: the next pingServer() sends a new ping. */
export function forgetPing() {
  ping = null;
}

/** GET /api/welcome: {text, starter_questions}. */
export async function getWelcome() {
  const body = await request("/api/welcome");
  if (typeof body.text !== "string" || !Array.isArray(body.starter_questions)) {
    throw new ApiError("server", GENERIC_ERROR);
  }
  return body;
}

/** A random id for one chat message (UUID v4), or null where the browser can't make one (then nothing is retried). */
function newMessageId() {
  return globalThis.crypto?.randomUUID?.() ?? null;
}

const pause = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * POST /api/chat. Leave sessionId null on the first message; the reply carries the id to send next.
 * Returns {session_id, reply, sources, steps, lead, degraded, answered_by}. T13d: each message has its own
 * client_message_id; when no reply arrived at all it is sent once more with the same id (RETRY_DELAY_MS).
 */
export async function postChat({ message, sessionId, clientMessageId = newMessageId() }) {
  const payload = { message, channel: "web" };
  if (sessionId) payload.session_id = sessionId;
  if (clientMessageId) payload.client_message_id = clientMessageId;
  const send = () =>
    request("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  let body;
  try {
    body = await send();
  } catch (error) {
    if (!(error instanceof ApiError) || !error.noReply || !clientMessageId) throw error;
    await pause(RETRY_DELAY_MS);
    body = await send(); // once, with the same id: the server sends the reply it made, if it made one
  }
  if (typeof body.reply !== "string" || typeof body.session_id !== "string") {
    throw new ApiError("server", GENERIC_ERROR);
  }
  return body;
}

/** "?status=HOT&q=pune" from the values that are set (null, undefined and "" are left out). */
function queryString(params = {}) {
  const query = new URLSearchParams();
  for (const [name, value] of Object.entries(params)) {
    if (value !== null && value !== undefined && value !== "") query.set(name, String(value));
  }
  const text = query.toString();
  return text ? `?${text}` : "";
}

function salesGet(path, key) {
  return request(path, { headers: { "X-API-Key": key } });
}

/**
 * GET /api/leads with the sales key. params: status (HOT | WARM | COLD), q (name or city), sort
 * (newest | score), limit, offset. Returns {items, total, limit, offset, counts}; counts has the leads
 * per status in the whole table, whatever the filter or page.
 */
export async function getLeads(params, key) {
  const body = await salesGet(`/api/leads${queryString(params)}`, key);
  if (!Array.isArray(body.items) || typeof body.total !== "number" || typeof body.counts !== "object" || !body.counts) {
    throw new ApiError("server", GENERIC_ERROR);
  }
  return body;
}

/** GET /api/leads/{id} with the sales key: every field, the score breakdown, reason, action and callbacks. */
export async function getLead(id, key) {
  const body = await salesGet(`/api/leads/${encodeURIComponent(id)}`, key);
  if (typeof body.id !== "number" || typeof body.score_breakdown !== "object" || !body.score_breakdown) {
    throw new ApiError("server", GENERIC_ERROR);
  }
  return body;
}

/** GET /api/support-requests with the sales key (params: limit, offset): {items, total, limit, offset}. */
export async function getSupportRequests(params, key) {
  const body = await salesGet(`/api/support-requests${queryString(params)}`, key);
  if (!Array.isArray(body.items) || typeof body.total !== "number") throw new ApiError("server", GENERIC_ERROR);
  return body;
}
