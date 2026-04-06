from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

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

@traceable(name="LangChain Agent Loop with Tool Calling")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tool_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("="   *50)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print (f"--- Iteration {iteration} ---")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        if not tool_calls:
            print(f"Final AI response (no tool calls): {ai_message.content}")
            return ai_message.content

        print(f"AI called {len(tool_calls)} tool(s):")

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})
            tool_call_id = tool_call.get("id")

            print(f"  - [Tool] {tool_name} with args: {tool_args}")

            tool_to_call = tool_dict.get(tool_name)
            if tool_to_call is None:
                print(f"    >> Error: Tool '{tool_name}' not found.")
            else:
                tool_result = tool_to_call.invoke(tool_args)
                print(f"    >> Tool result: {tool_result}")

                messages.append(ai_message)
                messages.append(ToolMessage(content=str(tool_result), tool_call_id=tool_call_id))

    print("ERROR: Reached maximum iterations without a final answer.")
    return None

if __name__ == "__main__":
    print("Starting agent loop...")
    print()

    question = "What is the price of a laptop with a gold discount?"
    result = run_agent(question)
    print(f"Final result: {result}")
