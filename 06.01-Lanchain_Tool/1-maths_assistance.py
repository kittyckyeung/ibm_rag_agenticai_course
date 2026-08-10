from dotenv import dotenv_values
import re
from typing import List, Dict, Union
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool, Tool
from langchain.agents import create_agent


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


def add__numbers(inputs:str) -> dict:
    """
    Adds a list of numbers provided in the input dictionary or extracts numbers from a string.

    Parameters:
    - inputs (str): 
    string, it should contain numbers that can be extracted and summed.

    Returns:
    - dict: A dictionary with a single key "result" containing the sum of the numbers.

    Example Input (Dictionary):
    {"numbers": [10, 20, 30]}

    Example Input (String):
    "Add the numbers 10, 20, and 30."

    Example Output:
    {"result": 60}
    """
    #numbers = [int(num) for num in re.findall(r'\d+', inputs)]
    numbers = [int(x) for x in inputs.replace(",", "").split() if x.isdigit()]

    result = sum(numbers)
    return {"result": result}


@tool
def add_numbers(inputs:str) -> dict:
    """
    Adds a list of numbers provided in the input dictionary or extracts numbers from a string.

    Parameters:
    - inputs (str): 
    string, it should contain numbers that can be extracted and summed.

    Returns:
    - dict: A dictionary with a single key "result" containing the sum of the numbers.

    Example Input (Dictionary):
    {"numbers": [10, 20, 30]}

    Example Input (String):
    "Add the numbers 10, 20, and 30."

    Example Output:
    {"result": 60}
    """
    numbers = [int(num) for num in re.findall(r'\d+', inputs)]
    #numbers = [int(x) for x in inputs.replace(",", "").split() if x.isdigit()]

    result = sum(numbers)
    return {"result": result}


@tool
def add_numbers_with_options(numbers: List[float], absolute: bool = False) -> float:
    """
    Adds a list of numbers provided as input.

    Parameters:
    - numbers (List[float]): A list of numbers to be summed.
    - absolute (bool): If True, use the absolute values of the numbers before summing.

    Returns:
    - float: The total sum of the numbers.
    """
    if absolute:
        numbers = [abs(n) for n in numbers]
    return sum(numbers)


@tool
def sum_numbers_with_complex_output(inputs: str) -> Dict[str, Union[float, str]]:
    """
    Extracts and sums all integers and decimal numbers from the input string.

    Parameters:
    - inputs (str): A string that may contain numeric values.

    Returns:
    - dict: A dictionary with the key "result". If numbers are found, the value is their sum (float). 
            If no numbers are found or an error occurs, the value is a corresponding message (str).

    Example Input:
    "Add 10, 20.5, and -3."

    Example Output:
    {"result": 27.5}
    """
    matches = re.findall(r'-?\d+(?:\.\d+)?', inputs)
    if not matches:
        return {"result": "No numbers found in input."}
    try:
        numbers = [float(num) for num in matches]
        total = sum(numbers)
        return {"result": total}
    except Exception as e:
        return {"result": f"Error during summation: {str(e)}"}


@tool
def sum_numbers_from_text(inputs: str) -> float:
    """
    Adds a list of numbers provided in the input string.
    
    Args:
        text: A string containing numbers that should be extracted and summed.
        
    Returns:
        The sum of all numbers found in the input.
    """
    # Use regular expressions to extract all numbers from the input
    numbers = [int(num) for num in re.findall(r'\d+', inputs)]
    result = sum(numbers)
    return result


if __name__ == "__main__":
    answer = build_prompt()
    print()
    # Using the non-@tool
    print("Using the non-@tool decorated function:")
    print(add__numbers("1 2"))
    print()
    add_tool=Tool(
        name="AddTool",
        func=add__numbers,
        description="Adds a list of numbers and returns the result.")
    print("Tool Object: ",add_tool)
    print()
    # Tool name
    print("Tool Name: ", add_tool.name)
    print()
    # Tool description
    print("Tool Description: ", add_tool.description)
    print()
    # Tool function
    print("Tool Function: ", add_tool.invoke)
    print()
    test_input = "10 20 30 a b" 
    print("Calling Tool Function: ", add_tool.invoke(test_input))
    print()
    # Using the @tool
    print("Using the @tool decorated function:")
    print("Name: ", add_numbers.name)
    print()
    print("Description: ", add_numbers.description) 
    print()
    print("Args: ", add_numbers.args) 
    print()
    test_input = "what is the sum between 10, 20 and 30 " 
    print(add_numbers.invoke(test_input))
    print()
    # Comparing the two approaches
    print("Tool Constructor Approach:")
    print(f"Has Schema: {hasattr(add_tool, 'args_schema')}")
    print()
    print("@tool Decorator Approach:")
    print(f"Has Schema: {hasattr(add_numbers, 'args_schema')}")
    print(f"Args Schema Info: {add_numbers.args}")
    print()
    print(f"Args Schema Info: {add_numbers_with_options.args}")
    print(f"Args Schema Info: {add_numbers.args}")
    print()
    print(add_numbers_with_options.invoke({"numbers":[-1.1,-2.1,-3.0],"absolute":False}))
    print(add_numbers_with_options.invoke({"numbers":[-1.1,-2.1,-3.0],"absolute":True}))
    print()
    agent = create_agent(
        model=call_llm(),
        tools=[add_numbers, add_numbers_with_options, sum_numbers_with_complex_output, sum_numbers_from_text],
        system_prompt="You are a helpful math assistant. Use tools when arithmetic is requested.",
    )
    agent_result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the sum of 10, 20.5, and -3?"}]}
    )
    print("Agent Result:", agent_result)
    print()
    agent_result = agent.invoke(
        {"messages": [{"role": "user", "content": "In 2023, the US GDP was approximately $27.72 trillion, while Canada's was around $2.14 trillion and Mexico's was about $1.79 trillion what is the total."}]}
    )
    print("Agent Result:", agent_result)
    print()
    agent_result = agent.invoke(
        {"messages": [{"role": "user", "content": "Add 10, 20, two and 30"}]}
    )
    print("Agent Result:", agent_result)


