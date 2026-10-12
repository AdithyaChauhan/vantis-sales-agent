"""Fixed texts the chat sends without calling an LLM. The chat page (T10) shows the same welcome.

The welcome lists what the assistant can help with and its main limits (KB Section 16).
"""

WELCOME_TEXT = (
    "Hi! I'm the Vantis Energy Solutions assistant. I can help with our rooftop solar packages, "
    "the cities we serve, the installation process, warranty and support, financing and subsidy "
    "questions, and getting a quote. I can't give exact prices, savings or subsidy amounts, or "
    "guarantee installation dates; the Vantis team confirms those after a consultation. "
    "What would you like to know?"
)

STARTER_QUESTIONS = [
    "What solar systems do you offer for homes?",
    "Do you install in my city?",
    "What is your installation process?",
    "What warranty do you provide?",
]

# Thanks or bye, with nothing else in the message
CLOSING_TEXT = "Thanks for chatting with Vantis! If you have more questions about rooftop solar, just ask."

# Off-topic only (flow F5): short and polite; the rules do not change
REDIRECT_TEXT = (
    "I can only help with questions about Vantis Energy Solutions and rooftop solar. "
    "Is there something about solar for your home or business I can help with?"
)

# T12a: what the reply says instead of a false "your details are saved" (app/agent/leads.py,
# remove_false_claims), when no lead exists after the turn and the reply has no question left.
# T12a: what a reply says about a callback, so it always matches what schedule_callback did
# (app/agent/leads.py, fix_callback_claims). Callbacks are requests; the team confirms them.
CALLBACK_LOGGED_TEXT = "I've logged a callback request for {time}; the team will confirm it."
CALLBACK_UPDATED_TEXT = "I've updated your callback request to {time}; the team will confirm it."
CALLBACK_EXISTS_TEXT = "Your callback request for {time} is logged; the team will confirm it."
CALLBACK_TIME_ASK = "What time would suit you for the call?"
STILL_NEEDED_ASK = "Could you share your {items}? Then I can pass your details to the sales team."
SUPPORT_STILL_NEEDED_ASK = "Could you share your {items}? Then I can pass your request to the Vantis team."
READY_TO_PASS_ASK = "Shall I pass your details to the sales team so they can follow up?"
# The missing items in everyday words (keys: NEED_ORDER and SUPPORT_NEEDS in app/agent/leads.py)
CUSTOMER_WORDS = {
    "city": "city",
    "requirement, monthly bill, usage, system size or property type": "monthly electricity bill",
    "name": "name",
    "phone or email": "phone number",
    "issue": "description of the problem",
    "project details": "system details (size and when it was installed)",
}
