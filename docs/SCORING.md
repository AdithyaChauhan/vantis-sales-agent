# SCORING.md: how a lead is qualified

Every lead gets a **lead score** (0-100), a **lead status** (HOT, WARM or COLD), a one-sentence **reason** and a **recommended action** for the sales team. The score and status come from fixed rules in code (`backend/app/leads/scoring.py`), never from the LLM, so the same details always give the same result. The LLM only writes the reason sentence, from the score breakdown.

## Example output (the demo lead)

Rahul Sharma wants a 5 kW system for his house in Lucknow, his bill is about ₹4,000 a month, he wants it next month, and he gave a valid phone number and email. Result, in the assignment's shape:

```json
{
  "lead_score": 97,
  "lead_status": "HOT",
  "reason": "This lead is hot because the customer in Lucknow is ready to buy a 5 kW system and has provided their contact details.",
  "recommended_action": "Contact the lead within 1 hour and offer a site assessment."
}
```

The reason above was written by an LLM in a live test. Its wording changes from run to run; the score, status and action never do.

## The three statuses (PDF Section 4E) and our rules

| Status | Our rules |
|---|---|
| **HOT**: a buyer with a clear need, a place we serve, and soon | 75 points or more. In practice: ready to buy (25), a size, bill or usage figure (20), a listed city (20) and a timeline under 3 months (15-20). A request to be called counts as ready to buy |
| **WARM**: real interest, but something is missing, far off or still to be checked | 45-74 points, **or** a lead that would be HOT but needs checking first (a city not on the list, or a need outside the reference packages): capped at WARM |
| **COLD**: just looking, or little sign of buying | Below 45 points, **or** a lead that is only exploring without a timeline under 3 months, whatever else it gave: capped at COLD |

### Why an unlisted city or an unusual need is WARM, not COLD

PDF Section 4E puts an unsupported location or requirement under Cold. The Vantis knowledge base doesn't call these unsupported:

- KB Section 10: "The list is intentionally limited; locations not listed should be treated as unverified." and "If a customer mentions a city not appearing above, the AI should say that service availability is not confirmed in its current information and offer to collect details for verification."
- The KB never says a requirement is unsupported. KB Sections 3 and 9 say package choices need customer information and a site assessment, and the catalog entries are "reference offerings".

So an unlisted city or a need outside the reference packages means **"check first"**, not **"no"**. Such a lead can never be HOT (the sales team must verify before treating it as a sure sale), but a strong buyer in Pune stays WARM, with "Verify service availability for Pune" in the recommended action, instead of being dropped as COLD.

## The points (100 in total)

| Signal | Points | How it is decided |
|---|---|---|
| **Intent** | ready 25 · interested 15 · exploring 5 · unknown 0 | *Ready*: asks for a quote, installation, site visit or callback, or plans to install soon. A request to be called (asking to be contacted, or giving a time to call) is always *ready*, whatever else the customer said, and is never lowered again. *Interested*: wants solar and asks about options (also when buying interest shows but the intent is otherwise unknown). *Exploring*: just curious or far-off plans |
| **Requirement** | clear 20 · words only 10 · none 0 | *Clear*: a system size (kW), a monthly bill (₹) or a monthly usage (kWh). A named package counts once the agent turns it into its size from the KB. *Words only*: e.g. "rooftop solar for my shop" |
| **Location** | listed 20 · unverified 5 · unknown 0 | Checked against the KB Section 10 city list, with renamed cities (Gurgaon → Gurugram) and small typos handled |
| **Timeline** | within about a month 20 · 1-3 months 15 · 3-6 months 8 · over 6 months 3 · unknown 0 | "Now", "this month" and "next month" all count as within about a month. When the understand step fails, code reads only clear phrases: "this month", "next month", "agle mahine", "ASAP", "jaldi", "turant" → within about a month; "next year", "agle saal" → over 6 months; and "just exploring", "just looking", "abhi bas dekh raha hu", "not now" → exploring |
| **Contact** | phone and email 10 · one of them 6 | Only valid values count: an Indian mobile number and a well-formed email |
| **Property** | owner with a usable rooftop 5 · partly known 2 · renting or no rooftop 0 · unknown 0 | *Partly known*: the property type, ownership or rooftop is known, but not "owner with a rooftop" |

**Status:** HOT at 75 or more · WARM at 45 to 74 · COLD below 45.

**Caps:** a city that is not on the list, or a need outside the reference packages, caps the lead at WARM, and the score at 74, so that the score and the status always agree and the dashboard never sorts a capped lead above a real HOT one. A lead that is only exploring, without a timeline under 3 months, is capped at COLD, and the score at 44 (Cold in PDF Section 4E). The breakdown keeps the score before the cap (`raw_score`) and says which cap applied. A cap is recorded only when it lowers the score: a COLD lead in an unlisted city has nothing to cap, so neither its reason nor its action mentions one. When two caps would lower it, only the stricter one is recorded (an exploring lead in an unlisted city is COLD 44 for exploring).

Every lead stores its **score breakdown** as JSON: each rule with its points, the maximum and a short label, plus the raw score and the caps. The dashboard shows it, so a sales rep can see where every point came from.

## When a lead is created

A lead is saved only when we have:

- a **name**, and
- a valid **phone number or email**, and
- a **city**, and
- some **need**: a requirement, a monthly bill, a usage figure, a system size, a property type, or whether it is a home or a business.

Until then, the details stay in the conversation and the assistant asks for what is missing, a question or two at a time (PDF Section 8, missing information).

The assistant saves the lead with its `create_lead` tool. If a turn ends with everything required known and no lead saved (the model's call failed, or it asked again for a detail it already had), the code saves it at the end of that turn, with the same rules and at most one lead per conversation, and logs a callback time the customer already gave. It never does so on a turn where the customer said no, or for an existing customer's support issue. Invalid values (a 9-digit phone, a broken email, a negative bill) are never saved. The customer gets a plain question back, such as "That doesn't look like a 10-digit Indian mobile number."

Accepted value ranges: monthly bill ₹100 to ₹10,00,000 · system size 0.5 to 100 kW · monthly usage 10 to 1,00,000 kWh · budget ₹10,000 to ₹10,00,00,000. Amounts like "₹4,000", "Rs 4000", "4k" and "1.2 lakh" are understood.

## Recommended actions

| Status | Action |
|---|---|
| HOT | Contact the lead within 1 hour and offer a site assessment. |
| WARM | Follow up within 24 hours, plus what to do next: "Verify service availability for <city>." (unlisted city) · "Check whether the requirement fits the reference packages." · "Confirm the owner's permission for a rooftop installation." (renting) · "Collect: …" (the missing details that would add points, most useful first: timeline, monthly bill or usage, rooftop availability, ownership, email, phone number) |
| COLD | Add to the nurture list, share product information and follow up in 30 days. A COLD lead that is ready (it asked to be contacted, or is ready to buy) gets a call instead: "Call the customer within 24 hours: they asked to be contacted or are ready to buy." It is never told "follow up in 30 days". |

If the customer asked for a callback, its time replaces the status's own timing, so the action never says "within 24 hours" next to "the day after tomorrow". The time is in the action from the moment the lead is saved. If they asked to be called but gave no time, the action starts with "Call the customer within 24 hours; they asked for a call.", whatever the status (a HOT lead adds "Offer a site assessment.", a WARM lead its next steps):

| Status | Action with a callback |
|---|---|
| HOT | Call <their time>, as requested, and offer a site assessment. |
| WARM | Call <their time>, as requested. (then the same next steps as above) |
| COLD | Call <their time>, as requested. Add to the nurture list and share product information. |

Callbacks are stored as **requested**: the team confirms them, the assistant never books them. A lead keeps one requested callback; a new time updates it ("4pm", then "day after tomorrow" gives "day after tomorrow, 4pm").

## The reason sentence

The fast model writes one sentence after the reply has been sent (it has 8 seconds, and never uses the main model's quota, which the customers' turns need) from the breakdown only: status, score, the rule labels with their points, the caps, the city and what keeps the lead from the next status. It never sees a name, phone number, email or money amount (labels say "monthly bill given", not the amount). If the model fails, or its sentence is empty, too long, more than one sentence, names a different status, or mentions prices, savings, subsidies, financing, EMIs or guarantees (KB Section 16), a template sentence built from the same breakdown is used instead. The reason is rewritten only when the status or breakdown changes, to save LLM quota.

**What keeps a lead from the next status.** A WARM or COLD reason ends with it, for example "…; not HOT yet: no timeline." (T12b: the model writes only the first part and code adds this ending once.) For a capped lead it is the cap's cause ("not HOT yet: city not in the listed service areas"). Otherwise it is the missing signals worth the most points, most first (the timeline first on a tie), until they would reach the next status: a WARM lead at 63 with no timeline needs 12 points, and a timeline is worth up to 20. Signals that more details can't change (a city not on the list, renting) are never named. The template leaves out the factors the ending already names, so nothing is said twice (Rohit: "COLD: requirement described in words only, …; not WARM yet: only exploring.").

## Worked examples

**1. Rahul in Lucknow → 97, HOT**

| Signal | Value | Points |
|---|---|---|
| Intent | ready (wants it installed next month) | 25 |
| Requirement | 5 kW system, monthly bill given | 20 |
| Location | Lucknow is listed | 20 |
| Timeline | next month (within about a month) | 20 |
| Contact | phone and email | 10 |
| Property | a house; ownership and rooftop not yet known | 2 |
| **Total** | | **97 → HOT** |

Action: "Contact the lead within 1 hour and offer a site assessment."

**2. The same lead in Pune → 82, capped to 74, WARM**

The location gives 5 instead of 20 (Pune is not in the KB list), so the raw score is 82. The unverified city caps the lead at WARM, and the score at 74.

Action: "Follow up within 24 hours. Verify service availability for Pune. Collect: rooftop availability, ownership."

Reason (template form): "WARM: ready to buy, …; not HOT yet: city not in the listed service areas."

**3. A visitor who is only exploring → 36, COLD**

Priya lives in an apartment in Kanpur, is "just looking around" and thinks about solar for next year. She leaves a phone number.

| Signal | Value | Points |
|---|---|---|
| Intent | exploring | 5 |
| Requirement | none stated | 0 |
| Location | Kanpur is listed | 20 |
| Timeline | over 6 months | 3 |
| Contact | phone only | 6 |
| Property | apartment, nothing else known | 2 |
| **Total** | | **36 → COLD** |

Action: "Add to the nurture list, share product information and follow up in 30 days."

**4. "Can someone call me?" → 53, WARM**

Then a name, a phone number, a house in Faridabad. No bill, no timeline (the audit's S21, which scored COLD 43 with "follow up in 30 days").

| Signal | Value | Points |
|---|---|---|
| Intent | ready (asked to be called) | 25 |
| Requirement | none stated | 0 |
| Location | Faridabad is listed | 20 |
| Timeline | not known | 0 |
| Contact | phone only | 6 |
| Property | a house, nothing else known | 2 |
| **Total** | | **53 → WARM** |

Action: "Call the customer within 24 hours; they asked for a call. Collect: timeline, monthly bill or usage, rooftop availability, ownership, email." With a time ("tomorrow 11am"): "Call tomorrow 11am, as requested. Collect: …"

**5. "Just exploring, maybe next year" → raw 56, capped to 44, COLD**

Then a name, a phone number, a house in Kanpur and a bill of about ₹3,000 (the audit's S22, which scored WARM 56).

| Signal | Value | Points |
|---|---|---|
| Intent | exploring | 5 |
| Requirement | monthly bill given | 20 |
| Location | Kanpur is listed | 20 |
| Timeline | next year (over 6 months) | 3 |
| Contact | phone only | 6 |
| Property | a house, nothing else known | 2 |
| **Total** | | **56, capped at 44 → COLD** |

Reason (template form): "COLD: clear requirement: monthly bill given, …; not WARM yet: only exploring, no timeline under 3 months." Action: "Add to the nurture list, share product information and follow up in 30 days."
