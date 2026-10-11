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
