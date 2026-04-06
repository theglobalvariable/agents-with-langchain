from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

import ollama
from ollama import Message

MAX_ITERATIONS = 10

MODEL = "qwen3:1.7b"

SYSTEM_PROMPT = """
You are a helpful shopping assistant that can call tools to answer user queries.
You have access to a product catalog tool and a discount tool.
STRICT RULES - You must follow these rules:
1. NEVER guess or assume any product prices or discount rates. Always use the tools to get accurate information.
2. Only call apply_discount after you have obtained a product price from the get_product_price tool. Always use the price obtained from the get_product_price tool as the input for the apply_discount tool.
3. NEVER calculate discounts or final prices manually. Always use the apply_discount tool to get the correct discounted price.
4. If user is not satisfied with the answer, ask them for more details about the product or discount they are interested in, and then use the tools to get the information needed to provide a complete answer.

If you have the answer to the user's query and do not need to call any tools, respond with a JSON object in the following format:
```json
{
  "answer": "Your final answer to the user's query."
}
```
"""


@traceable(name="Get Product Price Tool", run_type="tool")
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


@traceable(name="Apply Discount Tool", run_type="tool")
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


# define tools JSON for LLM
tools_for_llm = [
    {
        "type": "function",
        "function": {
            "name": "get_product_price",
            "description": "Look up the price of a product in the catalog.",
            "parameters": {
                "type": "object",
                "properties": {
                    "product_name": {
                        "type": "string",
                        "description": "The name of the product to look up.",
                    }
                },
                "required": ["product_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "apply_discount",
            "description": "Apply a discount to a price based on the discount tier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "price": {
                        "type": "number",
                        "description": "The original price of the product.",
                    },
                    "discount_tier": {
                        "type": "string",
                        "description": 'The discount tier to apply. Available tiers: "bronze", "silver", "gold".',
                    },
                },
                "required": ["price", "discount_tier"],
            },
        },
    },
]


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat(messages):
    return ollama.chat(model=MODEL, tools=tools_for_llm, messages=messages)


@traceable(name="Agent Loop with Raw Function Calling")
def run_agent(question: str):
    tool_dict = {
        "get_product_price": get_product_price,
        "apply_discount": apply_discount,
    }

    messages: list[Message] = [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=question),
    ]

    print(f"Question: {question}")
    print("=" * 50)

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration} ---")

        response = ollama_chat(messages)
        ai_message = response.message
        messages.append(ai_message)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final AI response (no tool calls): {ai_message.content}")
            return ai_message.content

        print(f"AI called {len(tool_calls)} tool(s):")

        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments

            print(f"  - [Tool] {tool_name} with args: {tool_args}")

            tool_to_call = tool_dict.get(tool_name)
            if tool_to_call is None:
                print(f"    >> Error: Tool '{tool_name}' not found.")
            else:
                tool_result = tool_to_call(**tool_args)
                print(f"    >> Tool result: {tool_result}")

                messages.append(Message(role="tool", content=str(tool_result)))

    print("ERROR: Reached maximum iterations without a final answer.")
    return None


if __name__ == "__main__":
    print("Starting agent loop...")
    print()

    question = "What is the price of a laptop with a gold discount?"
    result = run_agent(question)
    print(f"Final result: {result}")
