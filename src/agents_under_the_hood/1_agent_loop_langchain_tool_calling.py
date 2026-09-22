from dotenv import load_dotenv
import os

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage, SystemMessage
from langsmith import traceable
from langchain_ollama import ChatOllama

MAX_ITERATIONS = 10
MODEL = "qwen3:1.7b"

# -- Tools (Langchani @tool decorator) ---
@tool
def get_product_price(product: str) -> float:
    """lookup the price of a product in the catalog"""
    print(f" >>Executing get_product_price(product='{product}')")
    prices = {"laptop": 1299.99, "headphones": 149.95, "keyboard": 89.50}
    return prices.get(product, 0)

@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """apply a discount tier to a price and return the final price
    available tiers : Bronze, Silver, Gold"""
    print(f" >>Executing apply_discount(price='{price}, discount_tier='{discount_tier}')")
    discounts_percentages = {"Bronze": 5, "Silver": 12, "Gold": 23}
    discount = discounts_percentages.get(discount_tier, 0)
    return round(price * (1 - discount / 100), 2)

# -- Agent Loop  ---
@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {t.name: t for t in tools}
    llm = init_chat_model(f"ollama: {MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    print(f"Question: {question}")
    print("=" * 60)





    messages = [
        SystemMessage(
            content=("You are a helpful shopping assistant. "
                        "You have access to product catalog tool "
                        "and discount tier tool. \n\n"
                        "STRICT RULES - you must follow these exactly:\n"
                        "1. NEVER guess or assume any product price. "
                        "you must call get_product_price tool to get the price of a product."
                        "2. Only call apply_discount tool after getting the price of a product from "
                        "get_product_price tool. Pass the exact price"
                        "returned by get_product_price tool - do not pass a made-up number."
                        "3. Never calculate discounts yourself using math. "
                        "Always use the apply_discount tool to calculate the discount."
                        " If the user does not specify discount tier, "
                        "ask them which tier to use - do not assume one."
                        
            )
        ),

        HumanMessage(content=question)
    ]
   
    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"Iteration {iteration}")

        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls
        
        # If there are tool calls, this is the final answer

        if not tool_calls:
            print(f"Final answer: {ai_message.content}")
            return ai_message.content


        # Process only the FIRST tool call — force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {})
        tool_call_id = tool_call.get("id")

        print(f"  [Tool Selected] {tool_name} with args: {tool_args}")


        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool '{tool_name}' not found")

        observation = tool_to_use.invoke(tool_args)

        print(f"  [Tool Result] {observation}")

        messages.append(ai_message)
        messages.append(
            ToolMessage(content=str(observation),
            tool_call_id=tool_call_id)
            
        )


    print("ERROR: Max iterations reached without a final answer")
    return None

if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop after applying a gold discount?")







