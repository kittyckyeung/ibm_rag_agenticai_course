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


@tool
def subtract_numbers(inputs: str) -> dict:
    """
    Extracts numbers from a string, negates the first number, and successively subtracts 
    the remaining numbers in the list.

    This function is designed to handle input in string format, where numbers are separated 
    by spaces, commas, or other delimiters. It parses the string, extracts valid numeric values, 
    and performs a step-by-step subtraction operation starting with the first number negated.

    Parameters:
    - inputs (str): 
      A string containing numbers to subtract. The string may include spaces, commas, or 
      other delimiters between the numbers.

    Returns:
    - dict: 
      A dictionary containing the key "result" with the calculated difference as its value. 
      If no valid numbers are found in the input string, the result defaults to 0.

    Example Input:
    "100, 20, 10"

    Example Output:
    {"result": -130}

    Notes:
    - Non-numeric characters in the input are ignored.
    - If the input string contains only one valid number, the result will be that number negated.
    - Handles a variety of delimiters (e.g., spaces, commas) but does not validate input formats 
      beyond extracting numeric values.
    """
    # Extract numbers from the string
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]

    # If no numbers are found, return 0
    if not numbers:
        return {"result": 0}

    # Start with the first number negated
    result = -1 * numbers[0]

    # Subtract all subsequent numbers
    for num in numbers[1:]:
        result -= num

    return {"result": result}


@tool
def multiply_numbers(inputs: str) -> dict:
    """
    Extracts numbers from a string and calculates their product.

    Parameters:
    - inputs (str): A string containing numbers separated by spaces, commas, or other delimiters.

    Returns:
    - dict: A dictionary with the key "result" containing the product of the numbers.

    Example Input:
    "2, 3, 4"

    Example Output:
    {"result": 24}

    Notes:
    - If no numbers are found, the result defaults to 1 (neutral element for multiplication).
    """
    # Extract numbers from the string
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]
    print(numbers)

    # If no numbers are found, return 1
    if not numbers:
        return {"result": 1}

    # Calculate the product of the numbers
    result = 1
    for num in numbers:
        result *= num
        print(num)

    return {"result": result}


@tool
def divide_numbers(inputs: str) -> dict:
    """
    Extracts numbers from a string and calculates the result of dividing the first number 
    by the subsequent numbers in sequence.

    Parameters:
    - inputs (str): A string containing numbers separated by spaces, commas, or other delimiters.

    Returns:
    - dict: A dictionary with the key "result" containing the quotient.

    Example Input:
    "100, 5, 2"

    Example Output:
    {"result": 10.0}

    Notes:
    - If no numbers are found, the result defaults to 0.
    - Division by zero will raise an error.
    """
    # Extract numbers from the string
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]


    # If no numbers are found, return 0
    if not numbers:
        return {"result": 0}

    # Calculate the result of dividing the first number by subsequent numbers
    result = numbers[0]
    for num in numbers[1:]:
        result /= num

    return {"result": result}


@tool
def new_subtract_numbers(inputs: str) -> dict:
    """
    Extracts numbers from a string and performs subtraction sequentially, starting with the first number.

    This function is designed to handle input in string format, where numbers may be separated by spaces, 
    commas, or other delimiters. It parses the input string, extracts numeric values, and calculates 
    the result by subtracting each subsequent number from the first. inputs[0]-inputs[1]-inputs[2]

    Parameters:
    - inputs (str): 
      A string containing numbers to subtract. The string can include spaces, commas, or other 
      delimiters between the numbers.

    Returns:
    - dict: 
      A dictionary containing the key "result" with the calculated difference as its value. 
      If no valid numbers are found in the input string, the result defaults to 0.

    Example Usage:
    - Input: "100, 20, 10"
    - Output: {"result": 70}

    Limitations:
    - The function does not handle cases where numbers are formatted with decimals or other non-integer representations.
    """
    # Extract numbers from the string
    numbers = [int(num) for num in inputs.replace(",", "").split() if num.isdigit()]

    # If no numbers are found, return 0
    if not numbers:
        return {"result": 0}

    # Start with the first number
    result = numbers[0]

    # Subtract all subsequent numbers
    for num in numbers[1:]:
        result -= num

    return {"result": result}


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
    print()
    print("Name: ", subtract_numbers.name)
    print("Description: ", subtract_numbers.description)
    print("Args: ", subtract_numbers.args)
    print()
    test_input = "10 20 30 and four a b" 
    print(f"Calling Tool Function: {subtract_numbers.invoke(test_input)}")
    print()
    multiply_test_input = "2, 3, and four "
    multiply_result = multiply_numbers.invoke(multiply_test_input)
    print("--- Testing MultiplyTool ---")
    print(f"Input: {multiply_test_input}")
    print(f"Output: {multiply_result}")
    print()
    divide_test_input = "100, 5, two"
    divide_result = divide_numbers.invoke(divide_test_input)
    print("--- Testing DivideTool ---")
    print(f"Input: {divide_test_input}")
    print(f"Output: {divide_result}")
    print()

