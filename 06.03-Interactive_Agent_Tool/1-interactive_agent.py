from dotenv import dotenv_values
import numpy as np
import pandas as pd
from pprint import pprint
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
import glob
import os
from typing import List, Optional, Dict, Any
import json


# MiniMax-compatible OpenAI API settings
LLM_KEY_LOCATION = r"C:\Users\user\Desktop\Coding\minimax_Key.txt"
LLM_MODEL_NAME = "MiniMax-M3"
#LLM_MODEL_BASE_URL = "https://api.minimaxi.chat/v1/text/chatcompletion_v2"
LLM_MODEL_BASE_URL = "https://api.minimaxi.chat/v1"


def load_api_key(key_file: str) -> str:
    values = dotenv_values(key_file)
    api_key = values.get("MINIMAX_API_KEY")
    group_id = values.get("MINIMAX_GROUP_ID")

    api_key = api_key.strip() if isinstance(api_key, str) else ""
    group_id = group_id.strip() if isinstance(group_id, str) else ""

    if not api_key:
        raise ValueError("MINIMAX_API_KEY is missing or empty.")
    if not group_id:
        raise ValueError("MINIMAX_GROUP_ID is missing or empty.")

    #return api_key, group_id
    return api_key


def call_llm() -> ChatOpenAI:
    """Create and return a configured ChatOpenAI model instance."""
    api_key = load_api_key(LLM_KEY_LOCATION)

    llm = ChatOpenAI(
        model=LLM_MODEL_NAME,
        openai_api_key=api_key,
        openai_api_base=LLM_MODEL_BASE_URL,
        temperature=0.5,
    )
    return llm


def build_prompt() -> str:
    """Build a prompt for the LLM."""
    response = call_llm().invoke("What is tool calling in langchain?")
    print("\nResponse Content: ", response.content)
    return response.content if isinstance(response.content, str) else str(response.content)


@tool
def add(a: int, b: int) -> int:
    """
    Add a and b.
    
    Args:
        a (int): first integer to be added
        b (int): second integer to be added

    Return:
        int: sum of a and b
    """
    return a + b

@tool
def subtract(a: int, b:int) -> int:
    """Subtract b from a."""
    return a - b


@tool
def multiply(a: int, b:int) -> int:
    """Multiply a and b."""
    return a * b


class ToolCallingAgent:
    """Wrap an agent to perform explicit tool-calling flow:

    1. Send user query to the agent (LLM decides whether to call a tool).
    2. If the LLM requests a tool, call the corresponding local tool.
    3. Send the tool result back to the agent and get the final response.
    """

    def __init__(self, agent, tool_map: Dict[str, Any]):
        self.agent = agent
        self.tool_map = tool_map

    def run(self, query: str) -> Any:
        chat_history = {"messages": [{"role": "user", "content": query}]}
        first_response = self.agent.invoke(chat_history)

        # Find first AI message that requested a tool call
        messages = first_response.get("messages", [])
        print()
        print("First response: ", messages)
        print()
        tool_call = None
        for message in messages:
            # message objects may have attributes like `type` and `tool_calls`
            if getattr(message, "type", None) == "ai" and getattr(message, "tool_calls", None):
                tool_call = message.tool_calls[0]
                break

        if not tool_call:
            # No tool requested: return the LLM's direct reply
            if messages:
                last = messages[-1]
                return getattr(last, "text", str(last))
            return str(first_response)

        tool_id = tool_call["id"]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Call the actual tool
        # Please note that in a real-world scenario, you would want to validate the tool name and arguments before invoking the tool. For validation, you can check if the tool name exists in the tool_map and if the arguments match the expected types for that tool. This is important to prevent errors and ensure that the tool is called correctly.
        if tool_name not in self.tool_map:
            raise ValueError(f"Unknown tool requested: {tool_name}")
        tool_result = self.tool_map[tool_name].invoke(tool_args)
        print()
        print(f"Tool '{tool_name}' (id: {tool_id}) called with args {tool_args}, result: {tool_result}")
        print()
        # Append tool result into chat history as assistant content and get final response
        #chat_history["messages"].append({"role": "assistant", "content": f"Result from tool {tool_name}: {tool_result}"})
        chat_history["messages"].append({'role': 'assistant', 'content': None, 'tool_calls': [{'id': tool_id, 'type': 'function', 'function': {'name': tool_name, 'arguments': tool_args}}]})
        chat_history["messages"].append({"role": "tool", "tool_call_id": tool_id, "content": str(tool_result)})
        print()
        print("Chat history after tool call: ", chat_history)
        print()
        final_response = self.agent.invoke(chat_history)
        print()
        print("Full response: ", final_response)
        print()
        final_response = final_response["messages"][-1].text
        print("Full response (extracted): ", final_response)
        print()
        return final_response


def _tool_name(t: Any) -> str:
    """Safely determine a tool's external name without triggering attribute access errors."""
    try:
        name = getattr(t, "name")
        if name:
            return name
    except Exception:
        pass
    try:
        return t.__name__
    except Exception:
        return t.__class__.__name__


if __name__ == "__main__":
    print()
    inputs = {
        "a": 1,
        "b": 2
    }
    add_result = add.invoke(inputs)
    print("Add Result:", add_result)
    print()
    print("Agent with explicit tool-calling flow demonstration: ")
    agent = create_agent(
        model=call_llm(),
        tools=[add, subtract, multiply],
        system_prompt="You are a helpful assistant that can perform basic arithmetic operations using the provided tools. Use the tools when necessary to provide accurate answers.",
    )
    # Build a mapping from tool name to tool object (some tool decorators expose `name`)
    tool_list = [add, subtract, multiply]
    tool_map = {_tool_name(t): t for t in tool_list}
    # Demonstrate the explicit tool-calling flow
    wrapper = ToolCallingAgent(agent=agent, tool_map=tool_map)
    print('\n--- Explicit tool-calling flow ---')
    query = "What is 3 + 2?"
    final_response = wrapper.run(query)
    print()

