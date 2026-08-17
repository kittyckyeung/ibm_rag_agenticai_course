## Attempt to create a data visualization agent using langchain and MiniMax API. There is no quick and easy way to do this, but we can use the langchain tools and agent framework to create a simple agent that can visualize data using pandas and matplotlib. The agent will be able to take a pandas DataFrame and generate visualizations based on user input. The agent will also be able to answer questions about the data using the MiniMax API.
## Half done
from dotenv import dotenv_values
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_agent
import io
import sys
from typing import Dict, Any
import os


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


# Wrap the dataframe in a tool context (simple pattern)
@tool
def run_pandas_code(code: str) -> str:
    """
    The run_pandas_code tool allows the agent to execute pandas code on a DataFrame. The agent can use this tool to perform data analysis and manipulation tasks.
    
    Args:
        code (str): The pandas code to execute.

    Return:
        str: The output of the executed code or an error message.
    """

    globals_dict: Dict[str, Any] = {"df": df, "pd": pd}
    locals_dict: Dict[str, Any] = {}

    old_stdout = sys.stdout
    captured = io.StringIO()
    sys.stdout = captured

    try:
        exec(code, globals_dict, locals_dict)
    except Exception as e:
        return f"Error executing code: {e}"
    finally:
        sys.stdout = old_stdout

    return captured.getvalue()


if __name__ == "__main__":
    print()
    # Get the data (download locally if needed) and run row-count checks
    #DATA_URL = (
    #    "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZNoKMJ9rssJn-QbJ49kOzA/student-mat.csv"
    #)
    LOCAL_CSV = os.path.join(os.path.dirname(__file__), "student-mat.csv")
    df = pd.read_csv(LOCAL_CSV)

    print("DataFrame Head: ", df.head(5))
    print()
    print("DataFrame Info: ", df.info())
    print()
    agent = create_agent(
        model=call_llm(),
        tools=[run_pandas_code],
        system_prompt="You are a pandas agent. Always respond using Action/Action Input/Final Answer format. Never output raw data directly. The file is loaded into a pandas DataFrame called 'df'. Always provide a concise summary of your findings in the Final Answer.",
    )
    agent_result = agent.invoke(
        {"messages": [{"role": "user", "content": "how many rows of data are in this file?"}]}
    )
    print("Number of rows in DataFrame (Agent Result): ", agent_result)
    print()
    print("Number of rows in DataFrame: ", len(df))
    print()
    agent_result = agent.invoke(
        {"messages": [{"role": "user", "content": "Give me all the data where student's age is over 18 years old."}]}
    )
    print("Students over 18 (Agent Result): ", agent_result)
    print()
