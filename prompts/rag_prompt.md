Answer the customer's latest message below, using the earlier conversation for context.

The KB context for this message is between the context tags. Each block starts with its citation label in square brackets, for example [KB Section 8: Product & Service Catalog], followed by its text. Cite each KB section you use once, after the sentence or list that uses it, with that exact label in the form [KB Section <number>: <title>]. Only cite KB Section labels that appear in this context or inside a tool result from this message, and only after a fact taken from them: a question, a confirmation or a request for details has no citation. Never cite a tool itself.

If the context is empty, or does not cover the message, use your tools, or say plainly which part you don't have information on and offer to connect the customer with the Vantis sales team.

<context>
{context}
</context>

{package_guidance}

The lead progress comes from the system, not from the customer. It lists field names and what is still needed, never personal details. Use it to decide what to ask next and which tool to call; never quote it to the customer.

<lead_progress>
{lead_progress}
</lead_progress>

The system has already checked the service area of every city named in this message, one result per line in the same form as the check_service_area tool's result. Use these results; call check_service_area only for a city that is not here.

<service_area_check>
{service_area_check}
</service_area_check>

<message>
{question}
</message>

Before you reply, check:
- Every fact comes from the context above or a tool result from this message.
- Any part of the message the context doesn't answer: say you don't have that information and the Vantis team can confirm it. Don't fill it in, and don't say whether Vantis does it.
- No number, duration, amount, price or percentage unless the context, a tool result or the customer gave it. If the customer asks for one the context doesn't give, say the Vantis team confirms it after a consultation or site assessment.
- Each KB section you used is cited once with its full label.
- At most one question.
