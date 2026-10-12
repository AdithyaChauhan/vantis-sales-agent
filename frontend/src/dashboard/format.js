/**
 * Text helpers for the dashboard (pure functions, tested in format.test.js): the table's short
 * versions of free text, and the labels of the detail panel. Every value comes from the API as it is;
 * nothing here works out a score or a status.
 */

const INR = new Intl.NumberFormat("en-IN");
const DATE = new Intl.DateTimeFormat("en-IN", { day: "numeric", month: "short" });
const DATE_TIME = new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" });
// Property types that read well after a size ("20 kW warehouse"); "other" becomes "system"
const PROPERTY_WORDS = new Set(["house", "villa", "apartment", "shop", "office", "clinic", "warehouse"]);

const capitalize = (text) => (text ? text[0].toUpperCase() + text.slice(1) : text);

/** "Residential": the customer type, else the property type; null when neither is known. */
export function typeLabel(lead) {
  return capitalize(lead?.customer_type ?? lead?.property_type) ?? null;
}

/** "Lucknow · Residential" (the parts that are known). */
export function placeAndType(lead) {
  return [lead?.city, typeLabel(lead)].filter(Boolean).join(" · ");
}

/** "₹4,500" in Indian digit grouping. */
export function rupees(amount) {
  return `₹${INR.format(amount)}`;
}

/** "5 kW" or "5.5 kW". */
export function kilowatts(size) {
  return `${Number(Number(size).toFixed(2))} kW`;
}

/**
 * The table's one-line requirement: the size and property when the size is known ("20 kW
 * warehouse"), else the first line of the customer's own words, else the bill; null if none is known.
 */
export function requirementSummary(lead) {
  if (lead.system_size_kw) {
    return `${kilowatts(lead.system_size_kw)} ${PROPERTY_WORDS.has(lead.property_type) ? lead.property_type : "system"}`;
  }
  const words = lead.requirement?.split("\n").find((line) => line.trim());
  if (words) return words.trim();
  if (lead.monthly_bill_inr) return `${rupees(lead.monthly_bill_inr)} a month bill`;
  return null;
}

/** The first sentence of a text (the first step of a recommended action), or all of it. */
export function firstSentence(text) {
  const clean = (text ?? "").trim();
  return /^[\s\S]*?[.!?](?=\s|$)/.exec(clean)?.[0] ?? clean;
}

/** "just now", "5 min ago", "2 h ago", "3 d ago", then the date ("12 Oct"). `now` is for tests. */
export function timeAgo(iso, now = Date.now()) {
  const then = new Date(iso).getTime();
  if (Number.isNaN(then)) return "";
  const minutes = Math.floor((now - then) / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes} min ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours} h ago`;
  const days = Math.floor(hours / 24);
  if (days < 7) return `${days} d ago`;
  return DATE.format(new Date(then));
}

/** The full date and time in the browser's language (for a tooltip). */
export function fullDate(iso) {
  const date = new Date(iso);
  return Number.isNaN(date.getTime()) ? "" : DATE_TIME.format(date);
}

/** "+91 98765 43210" for a stored "+919876543210"; anything else as it is. */
export function formatPhone(phone) {
  const parts = /^\+91(\d{5})(\d{5})$/.exec(phone ?? "");
  return parts ? `+91 ${parts[1]} ${parts[2]}` : phone;
}

/** The names of the lead's known fields (not null), like the chat's lead summary sends them. */
export function knownFields(lead) {
  return Object.keys(lead ?? {}).filter((name) => lead[name] !== null && lead[name] !== undefined);
}

const LOCATION_WORDS = {
  listed: "in the listed service areas",
  unverified: "not in the listed service areas",
  unknown: "not checked",
};

/** The value for each "Details collected" row of the detail panel ({city, need, ...}), or null. */
export function detailValues(lead) {
  const usage = [
    lead.monthly_bill_inr ? `${rupees(lead.monthly_bill_inr)} a month` : null,
    lead.monthly_units_kwh ? `${INR.format(lead.monthly_units_kwh)} kWh a month` : null,
  ].filter(Boolean);
  const contact = [lead.phone ? "Phone" : null, lead.email ? "email" : null].filter(Boolean).join(" and ");
  const rooftop = { true: "Usable rooftop", false: "No usable rooftop" }[lead.rooftop_available] ?? null;
  return {
    city: lead.city ? `${lead.city}, ${LOCATION_WORDS[lead.location_status] ?? LOCATION_WORDS.unknown}` : null,
    need: lead.system_size_kw ? kilowatts(lead.system_size_kw) : null, // the words are shown under it
    bill: usage.join(", ") || null,
    timeline: lead.timeline ?? null,
    contact: capitalize(contact) || null,
    rooftop,
  };
}

const OWNERSHIP = { own: "Owns the property", rent: "Rents the property" };
const CHANNELS = { web: "Website chat", whatsapp: "WhatsApp", demo: "Demo data" };

/**
 * The detail panel's other known fields as [label, value] pairs, e.g. ["Ownership", "Owns the
 * property"]. Free text stays as the customer wrote it (the page shows it as plain text).
 */
export function otherDetails(lead) {
  const rows = [
    ["Property", capitalize(lead.property_type)],
    ["Ownership", OWNERSHIP[lead.ownership]],
    ["Business", lead.business_type],
    ["State", lead.state],
    ["Existing solar", { true: "Yes", false: "No" }[lead.existing_solar]],
    ["Budget", lead.budget_inr ? rupees(lead.budget_inr) : null],
    ["Why solar", lead.motivation],
    ["Other info", lead.other_info],
    ["City as typed", lead.city_input && lead.city_input !== lead.city ? lead.city_input : null],
    ["Source", CHANNELS[lead.source_channel] ?? lead.source_channel],
    ["Created", fullDate(lead.created_at)],
    ["Updated", lead.updated_at && lead.updated_at !== lead.created_at ? fullDate(lead.updated_at) : null],
  ];
  return rows.filter(([, value]) => value);
}
