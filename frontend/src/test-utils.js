import { vi } from "vitest";

export const SESSION = "6f1c2a90-1b2c-4d3e-8f40-123456789abc";
export const WELCOME = {
  text: "Hi! I'm the Vantis Energy Solutions assistant. What would you like to know?",
  starter_questions: ["What solar systems do you offer for homes?", "Do you install in my city?"],
};
export const SALES_KEY = "test-sales-key-0123456789";

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

const hoursAgo = (hours) => new Date(Date.now() - hours * 3_600_000).toISOString();

/** One row of GET /api/leads (the demo lead, HOT 97, two hours old). */
export function leadItem(overrides = {}) {
  return {
    id: 1,
    name: "Rahul Sharma",
    customer_type: "residential",
    property_type: "house",
    lead_score: 97,
    lead_status: "HOT",
    city: "Lucknow",
    location_status: "listed",
    requirement: "5 kW system for my house",
    system_size_kw: 5,
    monthly_bill_inr: 4000,
    recommended_action:
      "Contact the lead within 1 hour and offer a site assessment. Customer asked for a callback: tomorrow evening.",
    source_channel: "web",
    created_at: hoursAgo(2),
    ...overrides,
  };
}

/** A GET /api/leads body. counts defaults to the statuses of `items`. */
export function leadList(items, { total = items.length, counts = null, limit = 50, offset = 0 } = {}) {
  const byStatus = { HOT: 0, WARM: 0, COLD: 0 };
  for (const item of items) byStatus[item.lead_status] += 1;
  return { items, total, limit, offset, counts: counts ?? byStatus };
}

const RAHUL_RULES = [
  { rule: "intent", value: "ready", points: 25, max: 25, label: "Ready to buy" },
  { rule: "requirement", value: "clear", points: 20, max: 20, label: "Clear requirement: 5 kW system, monthly bill given" },
  { rule: "location", value: "listed", points: 20, max: 20, label: "City is in the listed service areas" },
  { rule: "timeline", value: "lt_1m", points: 20, max: 20, label: "Wants it within about a month" },
  { rule: "contact", value: "both", points: 10, max: 10, label: "Phone and email given" },
  { rule: "property", value: "partly_known", points: 2, max: 5, label: "Property partly known" },
];

/** A GET /api/leads/{id} body: every field, the breakdown, reason, action and callbacks. */
export function leadDetail(overrides = {}) {
  const item = leadItem();
  return {
    ...item,
    phone: "+919876543210",
    email: "rahul.sharma@example.com",
    city_input: "Lucknow",
    state: null,
    ownership: null,
    rooftop_available: null,
    monthly_units_kwh: null,
    existing_solar: null,
    timeline: "next month",
    timeline_bucket: "lt_1m",
    intent_level: "ready",
    requirement_outside_catalog: false,
    motivation: null,
    budget_inr: null,
    business_type: null,
    other_info: null,
    score_breakdown: { rules: RAHUL_RULES, raw_score: 97, caps: [], lead_score: 97, lead_status: "HOT" },
    reason: "Ready buyer with a clear 5 kW requirement in a listed city.",
    reason_source: "llm",
    updated_at: item.created_at,
    callbacks: [
      { id: 4, lead_id: 1, preferred_time: "tomorrow evening", notes: null, status: "requested", created_at: hoursAgo(1) },
    ],
    ...overrides,
  };
}

/** A support request row. */
export function supportItem(overrides = {}) {
  return {
    id: 1,
    name: "Meera Iyer",
    phone: "+919000000009",
    email: "meera.iyer@example.com",
    project_details: "3 kW rooftop system installed in 2025",
    issue: "The inverter shows a red error light since Monday",
    status: "open",
    created_at: hoursAgo(3),
    ...overrides,
  };
}

/**
 * Replace fetch with a fake API. Each handler returns a Response (or a promise of one, or throws):
 * welcome(), chat(payload, callNumber), and for the sales API (which checks the X-API-Key header
 * first: anything but SALES_KEY gets a 401) leads(params), lead(id) and support(params).
 * Returns the mock, so tests can read what was sent.
 */
export function fakeApi({
  welcome = () => jsonResponse(200, WELCOME),
  chat,
  leads = () => jsonResponse(200, leadList([leadItem()])),
  lead = (id) => jsonResponse(200, leadDetail({ id })),
  support = () => jsonResponse(200, { items: [supportItem()], total: 1, limit: 50, offset: 0 }),
  key = SALES_KEY,
} = {}) {
  let chatCalls = 0;
  const fetchMock = vi.fn(async (url, init = {}) => {
    const { pathname, searchParams } = new URL(url);
    if (pathname === "/api/welcome") return welcome();
    if (pathname === "/api/chat") return chat(JSON.parse(init.body), ++chatCalls);
    if (init.headers?.["X-API-Key"] !== key) {
      return jsonResponse(401, { detail: "Missing or invalid API key. Send it in the X-API-Key header." });
    }
    const params = Object.fromEntries(searchParams);
    if (pathname === "/api/leads") return leads(params);
    const one = /^\/api\/leads\/(\d+)$/.exec(pathname);
    if (one) return lead(Number(one[1]));
    if (pathname === "/api/support-requests") return support(params);
    throw new Error(`unexpected URL ${url}`);
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

/** The JSON bodies sent to /api/chat so far. */
export function chatPayloads(fetchMock) {
  return fetchMock.mock.calls.filter(([url]) => url.endsWith("/api/chat")).map(([, init]) => JSON.parse(init.body));
}

/** The sales API calls so far: [{path, params, key}]. */
export function salesCalls(fetchMock) {
  return fetchMock.mock.calls
    .filter(([url]) => /\/api\/(leads|support-requests)/.test(url))
    .map(([url, init]) => {
      const { pathname, searchParams } = new URL(url);
      return { path: pathname, params: Object.fromEntries(searchParams), key: init?.headers?.["X-API-Key"] ?? null, url };
    });
}

/** Pretend the screen is a phone (useWideScreen reads this media query). */
export function phoneScreen() {
  vi.stubGlobal("matchMedia", (query) => ({
    matches: false,
    media: query,
    addEventListener() {},
    removeEventListener() {},
  }));
}
