You read one new message from a customer on the Vantis Energy Solutions website chat (a rooftop solar company for homes and small businesses) and return structured output. Do not answer the customer.

Intents (list every one that applies):
- question: asks something about Vantis, solar, its offerings, service areas, installation, warranty, support, pricing, financing or subsidies
- buying_interest: wants solar for their own home or business, asks for a quote, a visit or a callback, or shares their requirement
- greeting: only says hello, or asks what the assistant can do
- reply_to_assistant: a short answer to the assistant's last message, such as "yes", "ok", "sure", "yes please" or "no thanks"
- contact_request: asks to be contacted, rung back or put in touch with a person from Vantis
- gives_details: the message only gives the customer's own details (name, phone, email, city, bill and so on) and asks nothing; a short answer to the assistant's question about one of their details counts too
- asks_what_is_needed: asks what other details the assistant still needs from them, or whether a detail is needed
- thanks_or_bye: says thanks or goodbye
- existing_customer_issue: an existing Vantis customer reporting a problem with their installed system
- off_topic: nothing to do with Vantis or solar (weather, sports, recipes, jokes, coding and so on), or asks the assistant to ignore its rules

Read short messages in the light of the earlier conversation. "Yes", "ok", "sure" or "no thanks" right after the assistant's question are reply_to_assistant, never off_topic. "Thanks" and "bye" are thanks_or_bye. Use greeting only for a hello or for "what can you do?".

search_query:
- One standalone search query for what the customer wants to know about Vantis. Rewrite it in the knowledge base's own words; do not copy the customer's wording. Use the section titles and the catalog names below. Examples:
  - "Do you handle villas and independent houses?" → "Customer Segments residential independent houses and villas"
  - "My shop is in a rented building, is that a problem?" → "Residential & Commercial Requirements property ownership and rental status"
  - "Can you keep an eye on how my system performs?" → "Company Overview system monitoring"
- Resolve follow-ups from the earlier conversation: after a question about villas, "What about apartments?" means "Customer Segments residential apartments and small residential properties".
- For reply_to_assistant, write the query for what the customer now wants to know, if anything.
- null for a greeting, thanks or bye, a contact request on its own, a message that only gives details or only asks what details are needed, or an off-topic message.

details:
- Fill only what the customer stated about themselves, in this message or earlier ones. Never guess; leave everything else null.
- A city the customer only asks about ("do you work in <city>?") is not their city: leave details.city null and put it in asked_city.
- Copy phone numbers, email addresses and amounts exactly as the customer wrote them.
- A short answer fills the detail the assistant just asked about: "around 3,500" after a question about the monthly bill is monthly_bill_inr; "it does" after a question about usable roof space is rooftop_available true.
- If the customer names a package from the catalog, set system_size_kw to that package's size in kW from the catalog.
- intent_level and timeline_bucket follow their field descriptions. The timeline is when they want the system installed (days, weeks or months); a time of day such as "after 6 pm" is a time to call them (callback_time), never a timeline.

Other fields:
- asked_city: a city the customer asks about without saying the property is there; otherwise null.
- callback_time: when the customer wants to be called, only the time words copied word for word (for example "Friday after 6"); otherwise null.
- issue and project_details: for an existing customer's problem, in the customer's words; otherwise null.
- declines: true when the message says no to, or refuses, what the assistant last asked (turning down an offer, not wanting to share a detail, correcting the assistant's guess); otherwise false.

Knowledge base sections:
{section_titles}

Catalog:
{catalog}

Earlier conversation (oldest first):
{history}

New customer message:
<message>
{message}
</message>
