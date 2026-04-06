import inspect
import re

from dotenv import load_dotenv
from langsmith import traceable

load_dotenv()

import ollama
from ollama import Message

MAX_ITERATIONS = 10

MODEL = "qwen3:1.7b"


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
def apply_discount(price: str, discount_tier: str) -> float:
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
    price_float = float(price)
    discount_rate = discounts.get(discount_tier.lower(), 0)
    discounted_price = price_float * (1 - discount_rate)
    return round(discounted_price, 2)


tools = {
    "get_product_price": get_product_price,
    "apply_discount": apply_discount,
}


def get_tool_descriptions(tool_dict) -> str:
    descriptions = []

    for tool_name, tool_func in tool_dict.items():
        original_function = getattr(tool_func, "__wrapped__", tool_func)
        signature = inspect.signature(original_function)
        docstring = inspect.getdoc(original_function) or ""
        descriptions.append(f"{tool_name}{signature} - {docstring}")

    return "\n".join(descriptions)


tools_descriptions = get_tool_descriptions(tools)
tool_names = ", ".join(tools.keys())

react_prompt = f"""
STRICT RULES - You must follow these rules:
1. NEVER guess or assume any product prices or discount rates. Always use the tools to get accurate information.
2. Only call apply_discount after you have obtained a product price from the get_product_price tool. Always use the price obtained from the get_product_price tool as the input for the apply_discount tool.
3. NEVER calculate discounts or final prices manually. Always use the apply_discount tool to get the correct discounted price.
4. If user is not satisfied with the answer, ask them for more details about the product or discount they are interested in, and then use the tools to get the information needed to provide a complete answer.

Answer the following questions as best you can. You have access to the following tools:

{tools_descriptions}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action, as comma separated values
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {{question}}
Thought:"""


@traceable(name="Ollama Chat", run_type="llm")
def ollama_chat(model, messages, options):
    return ollama.chat(model=model, messages=messages, options=options)


@traceable(name="Agent Loop with Raw Function Calling")
def run_agent(question: str):
    print(f"Question: {question}")
    print("=" * 50)

    prompt = react_prompt.format(question=question)
    scratchpad = ""

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"--- Iteration {iteration} ---")

        full_prompt = prompt + scratchpad

        # print(f"  [LLM] Sending prompt to LLM:\n{full_prompt}\n")
        print("  [Thinking]")

        response = ollama_chat(
            model=MODEL,
            messages=[Message(role="user", content=full_prompt)],
            options={"stop": ["\nObservation"], "temperature": 0},
        )
        output = response.message.content
        # print(f"LLM Output:\n{output}\n")
        print("  LLM Output received.")

        print("  [Parsing] Looking for final answer in LLM output...")

        final_answer_match = re.search(r"Final Answer:\s*(.+)", str(output))
        if final_answer_match:
            final_answer = final_answer_match.group(1).strip()
            print(f"  [Parsed] Final Answer: {final_answer}")
            print("=" * 50)
            return final_answer

        print("  [Parsing] No final answer found. Looking for tool calls...")

        action_match = re.search(r"Action:\s*(\w+)", str(output))
        action_input_match = re.search(r"Action Input:\s*(.+)", str(output))
        if not action_match or not action_input_match:
            print("    >> No valid tool call found in LLM output. Ending loop.")
            break

        tool_name = action_match.group(1).strip()
        tool_input_raw = action_input_match.group(1).strip()

        print(
            f"  [Parsed] Detected tool call: {tool_name} with input: {tool_input_raw}"
        )

        raw_args = [x.strip() for x in tool_input_raw.split(",")]
        args = [x.split("=", 1)[-1].strip().strip("'\"") for x in raw_args]

        print(f"  [Tool Executing] {tool_name}({args})...")

        tool_to_call = tools.get(tool_name)
        if tool_to_call is None:
            observation = (
                f"Error: Tool '{tool_name}' not found. Available tools: {tool_names}"
            )
        else:
            observation = str(tools[tool_name](*args))

        print(f"  [Tool Result] Observation: {observation}")

        scratchpad += f"{output}\nObservation: {observation}\n"

    print("ERROR: Reached maximum iterations without a final answer.")
    return None


if __name__ == "__main__":
    print("Starting agent loop...")
    print()

    question = "What is the price of a laptop with a gold discount?"
    result = run_agent(question)
    print(f"Final result: {result}")
