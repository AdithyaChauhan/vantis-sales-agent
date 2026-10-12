import { describe, expect, it } from "vitest";

import {
  detailValues,
  firstSentence,
  formatPhone,
  knownFields,
  otherDetails,
  placeAndType,
  requirementSummary,
  timeAgo,
  typeLabel,
} from "./format.js";

describe("requirementSummary (the table's one line)", () => {
  it.each([
    [{ system_size_kw: 20, property_type: "warehouse", requirement: "20 kW on the roof" }, "20 kW warehouse"],
    [{ system_size_kw: 5.5, property_type: "other" }, "5.5 kW system"],
    [{ system_size_kw: 4, property_type: null }, "4 kW system"],
    [{ requirement: "  i want to install one at my organisation\nand the warehouse" }, "i want to install one at my organisation"],
    [{ requirement: "\n\nThe first line with words" }, "The first line with words"],
    [{ monthly_bill_inr: 125000 }, "₹1,25,000 a month bill"],
    [{}, null],
  ])("%j → %j", (lead, expected) => {
    expect(requirementSummary(lead)).toBe(expected);
  });
});

describe("firstSentence (the table's next action)", () => {
  it.each([
    [
      "Contact the lead within 1 hour and offer a site assessment. Customer asked for a callback: tomorrow evening.",
      "Contact the lead within 1 hour and offer a site assessment.",
    ],
    ["Follow up within 24 hours. Verify service availability for Pune.", "Follow up within 24 hours."],
    ["No full stop at all", "No full stop at all"],
    ["  ", ""],
    [null, ""],
  ])("%j → %j", (text, expected) => {
    expect(firstSentence(text)).toBe(expected);
  });
});

describe("timeAgo", () => {
  const now = Date.parse("2026-10-12T12:00:00Z");
  it.each([
    ["2026-10-12T11:59:40Z", "just now"],
    ["2026-10-12T12:05:00Z", "just now"], // a clock slightly ahead
    ["2026-10-12T11:55:00Z", "5 min ago"],
    ["2026-10-12T10:00:00Z", "2 h ago"],
    ["2026-10-10T12:00:00Z", "2 d ago"],
    ["2026-10-01T12:00:00Z", "1 Oct"],
    ["not a date", ""],
  ])("%s → %j", (iso, expected) => {
    expect(timeAgo(iso, now)).toBe(expected);
  });
});

describe("labels", () => {
  it("formats an Indian mobile number and leaves anything else alone", () => {
    expect(formatPhone("+919876543210")).toBe("+91 98765 43210");
    expect(formatPhone("12345")).toBe("12345");
    expect(formatPhone(null)).toBeNull();
  });

  it("names the customer type, else the property type", () => {
    expect(typeLabel({ customer_type: "commercial", property_type: "shop" })).toBe("Commercial");
    expect(typeLabel({ customer_type: null, property_type: "villa" })).toBe("Villa");
    expect(typeLabel({})).toBeNull();
    expect(placeAndType({ city: "Pune", customer_type: "residential" })).toBe("Pune · Residential");
    expect(placeAndType({ city: null, customer_type: null })).toBe("");
  });

  it("gives each 'Details collected' row its value", () => {
    const values = detailValues({
      city: "Pune",
      location_status: "unverified",
      system_size_kw: 4,
      monthly_bill_inr: 3500,
      monthly_units_kwh: 1200,
      timeline: "next month",
      phone: "+919000000006",
      email: null,
      rooftop_available: false,
    });
    expect(values).toEqual({
      city: "Pune, not in the listed service areas",
      need: "4 kW",
      bill: "₹3,500 a month, 1,200 kWh a month",
      timeline: "next month",
      contact: "Phone",
      rooftop: "No usable rooftop",
    });
    expect(detailValues({ rooftop_available: null, phone: "+91", email: "a@example.com" })).toMatchObject({
      rooftop: null,
      contact: "Phone and email",
      city: null,
    });
  });

  it("lists the other known details only, and the source", () => {
    const rows = otherDetails({
      property_type: "house",
      ownership: "rent",
      business_type: null,
      existing_solar: false,
      budget_inr: 250000,
      city: "Gurugram",
      city_input: "Gurgaon",
      source_channel: "demo",
      created_at: "2026-10-12T10:00:00Z",
      updated_at: "2026-10-12T10:00:00Z",
    });
    const labels = rows.map(([label]) => label);
    expect(labels).toEqual(["Property", "Ownership", "Existing solar", "Budget", "City as typed", "Source", "Created"]);
    expect(Object.fromEntries(rows)).toMatchObject({
      Ownership: "Rents the property",
      "Existing solar": "No",
      Budget: "₹2,50,000",
      "City as typed": "Gurgaon",
      Source: "Demo data",
    });
  });

  it("knows which fields are known (false counts, null does not)", () => {
    expect(knownFields({ city: "Pune", rooftop_available: false, email: null, phone: undefined })).toEqual([
      "city",
      "rooftop_available",
    ]);
  });
});
