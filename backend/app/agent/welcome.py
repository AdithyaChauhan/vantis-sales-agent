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
# T12b: what replaces a sentence that stated an invented duration, amount or percentage (app/rag/numbers.py),
# unless the reply already says the team confirms something (KB Section 16).
NUMBERS_CONFIRM_TEXT = (
    "The Vantis team can confirm exact numbers, such as costs, timings or amounts, after a consultation or "
    "site assessment."
)
# T12b: when the model fails after a tool already wrote something this turn, the reply says what was done
# (app/agent/graph.py, tool_outcome_reply) instead of "try again", which would be false
SAVED_FALLBACK_TEXT = (
    "Your details are saved, and the Vantis sales team can follow up with you. What time would suit you for a "
    "callback?"
)
SUPPORT_SAVED_FALLBACK_TEXT = "I've saved your support request (id {id}); the Vantis team can follow up on it."
CALLBACK_TIME_ASK = "What time would suit you for the call?"
# T13c: what replaces a sentence that stated a policy on something the KB doesn't cover, such as discounts, offers or
# brands (app/rag/answer.py, remove_uncovered_policies): never a policy either way
UNCOVERED_TEXT = "I can't confirm that from the information I have; the Vantis sales team can tell you."
# T13c: the reply when the all-or-none package guard (app/agent/packages.py) would leave nothing
PACKAGES_FIT_TEXT = (
    "Which reference package fits depends on your property and electricity use; a site assessment decides it."
)
STILL_NEEDED_ASK = "Could you share your {items}? Then I can pass your details to the sales team."
SUPPORT_STILL_NEEDED_ASK = "Could you share your {items}? Then I can pass your request to the Vantis team."
READY_TO_PASS_ASK = "Shall I pass your details to the sales team so they can follow up?"
# T13d: the safety net (app/agent/fallback.py), when no model answered. City lines from the code's own check, with the
# KB Section 10 tag after them; the lines for a message the KB sentences don't match; the one closing question
AREA_LISTED_TEXT = "Vantis currently covers {city}."
AREA_UNVERIFIED_TEXT = ("I can't confirm service in {city} yet, as it isn't among the service areas in my information; "
                        "the Vantis team can check it for you.")
NOT_COVERED_TEXT = "That isn't covered in the information I have; the Vantis team can confirm it for you."
DETAILS_THANKS_TEXT = "Thanks for sharing that."
CONTINUE_TEXT = "Happy to help with that."
FALLBACK_OFFER = "Would you like the Vantis sales team to help you with this?"
ANYTHING_ELSE_ASK = "Anything else you'd like to know?"
# The next detail as the safety net's one question (keys: the items of leads.ask_targets)
NEXT_QUESTION = {
    "city": "Which city is the property in?",
    "requirement, monthly bill, usage, system size or property type": (
        "What is your approximate monthly electricity bill?"),
    "name": "Could you share your name?",
    "phone or email": "Could you share your phone number or email?",
    "property type": "Is this for a home or a business?",
    "issue": "Could you describe the problem with your system?",
    "project details": "Could you share your system details (size and when it was installed)?",
    "timeline": "When would you like the system installed?",
    "monthly bill or usage": "What is your approximate monthly electricity bill?",
    "rooftop availability": "Does the property have usable roof space for the panels?",
    "ownership": "Do you own the property?",
    "email": "Could you share your email address?",
    "phone number": "Could you share your phone number?",
}
# The missing items in everyday words (keys: NEED_ORDER and SUPPORT_NEEDS in app/agent/leads.py)
CUSTOMER_WORDS = {
    "city": "city",
    "requirement, monthly bill, usage, system size or property type": "monthly electricity bill",
    "name": "name",
    "phone or email": "phone number",
    "issue": "description of the problem",
    "project details": "system details (size and when it was installed)",
}
