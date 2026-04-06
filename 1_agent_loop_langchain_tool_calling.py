from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

MAX_ITERATIONS = 5

MODEL = "qwen3:1.7b"


@tool
def get_product_price(product_name: str) -> float:
    """Look up the price of a product in the catalog."""

    print(f"  >> Looking up price for: {product_name}")

    prices = {
        "laptop": 999.99,
        "smartphone": 499.85,
        "headphones": 199.50,
        "keyboard": 59.17,
    }
    return prices.get(product_name.lower(), 0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """
    Apply a discount to a price based on the discount tier.
    Available discount tiers:
    - "bronze": 5% discount
    - "silver": 10% discount
    - "gold": 15% discount
    """

    print(f"  >> Applying {discount_tier} discount to price: {price}")

    discounts = {
        "bronze": 0.05,
        "silver": 0.10,
        "gold": 0.15,
    }
    discount_rate = discounts.get(discount_tier.lower(), 0)
    discounted_price = price * (1 - discount_rate)
    return round(discounted_price, 2)

def run_agent(query: str) -> str:
    return "No tool calls were made."

if __name__ == "__main__":
    print("Starting agent loop...")
    print()

    user_query = "What is the price of a laptop with a gold discount?"
    result = run_agent(user_query)
    print(f"Final result: {result}")
