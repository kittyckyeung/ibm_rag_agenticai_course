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


# MiniMax-compatible OpenAI API settings
LLM_KEY_LOCATION = r"C:\Users\user\Desktop\Coding\minimax_Key.txt"
LLM_MODEL_NAME = "MiniMax-M3"
#LLM_MODEL_BASE_URL = "https://api.minimaxi.chat/v1/text/chatcompletion_v2"
LLM_MODEL_BASE_URL = "https://api.minimaxi.chat/v1"


DATAFRAME_CACHE = {}


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
def list_csv_files() -> Optional[List[str]]:
    """List all CSV file names in the local directory.

    Returns:
        A list containing CSV file names.
        If no CSV files are found, returns None.
    """
    csv_files = glob.glob(os.path.join(os.getcwd(), "*.csv"))
    if not csv_files:
        return None
    return [os.path.basename(file) for file in csv_files]


@tool
def preload_datasets(paths: List[str]) -> str:
    """
    Loads CSV files into a global cache if not already loaded.
    
    This function helps to efficiently manage datasets by loading them once
    and storing them in memory for future use. Without caching, you would
    waste tokens describing dataset contents repeatedly in agent responses.
    
    Args:
        paths: A list of file paths to CSV files.

    Returns:
        A message summarizing which datasets were loaded or already cached.
    """
    loaded = []
    cached = []
    for path in paths:
        if path not in DATAFRAME_CACHE:
            DATAFRAME_CACHE[path] = pd.read_csv(path)
            loaded.append(path)
        else:
            cached.append(path)
    
    return (
        f"Loaded datasets: {loaded}\n"
        f"Already cached: {cached}"
    )


@tool
def get_dataset_summaries(dataset_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Analyze multiple CSV files and return metadata summaries for each.

    Args:
        dataset_paths (List[str]): 
            A list of file paths to CSV datasets.

    Returns:
        List[Dict[str, Any]]: 
            A list of summaries, one per dataset, each containing:
            - "file_name": The path of the dataset file.
            - "column_names": A list of column names in the dataset.
            - "data_types": A dictionary mapping column names to their data types (as strings).
    """
    summaries = []

    for path in dataset_paths:
        # Load and cache the dataset if not already cached
        if path not in DATAFRAME_CACHE:
            DATAFRAME_CACHE[path] = pd.read_csv(path)
        
        df = DATAFRAME_CACHE[path]

        # Build summary
        summary = {
            "file_name": path,
            "column_names": df.columns.tolist(),
            "data_types": df.dtypes.astype(str).to_dict()
        }

        summaries.append(summary)

    return summaries


@tool
def call_dataframe_method(file_name: str, method: str) -> str:
   """
   Execute a method on a DataFrame and return the result.
   This tool lets you run simple DataFrame methods like 'head', 'tail', or 'describe' 
   on a dataset that has already been loaded and cached using 'preload_datasets'.
   Args:
       file_name (str): The path or name of the dataset in the global cache.
       method (str): The name of the method to call on the DataFrame. Only no-argument 
                     methods are supported (e.g., 'head', 'describe', 'info').
   Returns:
       str: The output of the method as a formatted string, or an error message if 
            the dataset is not found or the method is invalid.
   Example:
       call_dataframe_method(file_name="data.csv", method="head")
   """
   # Try to get the DataFrame from cache, or load it if not already cached
   if file_name not in DATAFRAME_CACHE:
       try:
           DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)
       except FileNotFoundError:
           return f"DataFrame '{file_name}' not found in cache or on disk."
       except Exception as e:
           return f"Error loading '{file_name}': {str(e)}"
   
   df = DATAFRAME_CACHE[file_name]
   func = getattr(df, method, None)
   if not callable(func):
       return f"'{method}' is not a valid method of DataFrame."
   try:
       result = func()
       return str(result)
   except Exception as e:
       return f"Error calling '{method}' on '{file_name}': {str(e)}"


@tool
def evaluate_classification_dataset(file_name: str, target_column: str) -> Dict[str, float]:
    """
    Train and evaluate a classifier on a dataset using the specified target column.
    Args:
        file_name (str): The name or path of the dataset stored in DATAFRAME_CACHE.
        target_column (str): The name of the column to use as the classification target.
    Returns:
        Dict[str, float]: A dictionary with the model's accuracy score.
    """
    # Try to get the DataFrame from cache, or load it if not already cached
    if file_name not in DATAFRAME_CACHE:
        try:
            DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)
        except FileNotFoundError:
            return {"error": f"DataFrame '{file_name}' not found in cache or on disk."}
        except Exception as e:
            return {"error": f"Error loading '{file_name}': {str(e)}"}
    
    df = DATAFRAME_CACHE[file_name]
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in '{file_name}'."}
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return {"accuracy": acc}

@tool
def evaluate_regression_dataset(file_name: str, target_column: str) -> Dict[str, float]:
    """
    Train and evaluate a regression model on a dataset using the specified target column.
    Args:
        file_name (str): The name or path of the dataset stored in DATAFRAME_CACHE.
        target_column (str): The name of the column to use as the regression target.
    Returns:
        Dict[str, float]: A dictionary with R² score and Mean Squared Error.
    """
    # Try to get the DataFrame from cache, or load it if not already cached
    if file_name not in DATAFRAME_CACHE:
        try:
            DATAFRAME_CACHE[file_name] = pd.read_csv(file_name)
        except FileNotFoundError:
            return {"error": f"DataFrame '{file_name}' not found in cache or on disk."}
        except Exception as e:
            return {"error": f"Error loading '{file_name}': {str(e)}"}
    
    df = DATAFRAME_CACHE[file_name]
    if target_column not in df.columns:
        return {"error": f"Target column '{target_column}' not found in '{file_name}'."}
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    return {
        "r2_score": r2,
        "mean_squared_error": mse
    }


if __name__ == "__main__":
    print()
    print("Tool Name: ", list_csv_files.name)
    print("Tool Description: ", list_csv_files.description)
    print("Tool Arguments: ", list_csv_files.args)
    print()
    print("Calling Tool Function: ", list_csv_files.invoke({}))
    print()
    print("Tool Name: ", preload_datasets.name)
    print("Tool Description: ", preload_datasets.description)
    print("Tool Arguments: ", preload_datasets.args)
    print()
    print("Calling Tool Function: ", preload_datasets.invoke({"paths": list_csv_files.invoke({})}))
    print()
    print("Tool Name: ", get_dataset_summaries.name)
    print("Tool Description: ", get_dataset_summaries.description)
    print("Tool Arguments: ", get_dataset_summaries.args)
    print("Calling Tool Function: ", get_dataset_summaries.invoke({"dataset_paths": list_csv_files.invoke({})}))
    print()
    agent = create_agent(
        model=call_llm(),
        tools=[list_csv_files, preload_datasets, get_dataset_summaries, call_dataframe_method, evaluate_classification_dataset, evaluate_regression_dataset],
        system_prompt="You are a data science assistant. Use the available tools to analyze CSV files. Your job is to determine whether each dataset is for classification or regression, based on its structure. Do not answer from prior knowledge.",
    )
    inputs = {
        "messages": [{"role": "user", "content": "List classification-dataset.csv and regression-dataset.csv files, inspect their columns, and tell me whether each dataset is classification or regression."}]
    }
    print("=== Step-by-step agent execution ===")
    # Stream with safe finalization. Collect the last event and derive final messages from it.
    last_event = None
    try:
        for step, event in enumerate(agent.stream(inputs, stream_mode="updates"), start=1):
            last_event = event
            print(f"\nStep {step}")
            print("Updated nodes:", list(event.keys()))

            # Collect tools invoked in this step (from model tool_calls and tools node messages)
            step_tools: list[str] = []

            model_update = event.get("model")
            if isinstance(model_update, dict):
                model_messages = model_update.get("messages", [])
                if model_messages:
                    last_model_message = model_messages[-1]
                    tool_calls = getattr(last_model_message, "tool_calls", None)
                    if tool_calls:
                        # record tool names from model's tool_calls
                        for tc in tool_calls:
                            try:
                                tname = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                            except Exception:
                                tname = None
                            if tname and tname not in step_tools:
                                step_tools.append(tname)
                        print("Model requested tool call(s):")
                        pprint(tool_calls)

            tools_update = event.get("tools")
            if isinstance(tools_update, dict):
                tool_messages = tools_update.get("messages", [])
                for tool_msg in tool_messages:
                        tool_name = getattr(tool_msg, "name", None)
                        tool_content = getattr(tool_msg, "content", "")
                        if tool_name and tool_name not in step_tools:
                            step_tools.append(tool_name)
                        print(f"Tool result from {tool_name if tool_name else 'unknown_tool'}:")
                        print(tool_content)

            # Print tools used this step
            if step_tools:
                print("Tools this step:", step_tools)
            else:
                print("Tools this step: None")

    except Exception as e:
        # Streaming occasionally fails in langgraph; fall back to single invoke.
        print("Streaming error:", repr(e))
        print("Falling back to single agent.invoke call...")
        try:
            agent_result = agent.invoke(inputs)
            print("\n=== Final Agent Result (fallback invoke) ===")
            print("Agent Result:", agent_result)
            last_event = None
        except Exception as e2:
            print("Fallback invoke also failed:", repr(e2))
            raise

    # If stream completed normally, derive final agent result from last_event
    if last_event is not None:
        # model messages first, then any tool messages appended by tools node
        final_messages = []
        model_part = last_event.get("model")
        if isinstance(model_part, dict):
            final_messages.extend(model_part.get("messages", []))
        tools_part = last_event.get("tools")
        if isinstance(tools_part, dict):
            final_messages.extend(tools_part.get("messages", []))

        agent_result = {"messages": final_messages}
        print("\n=== Final Agent Result (from stream) ===")
        print("Agent Result:", agent_result)
    print()
    print("=== Final Agent Messages ===")
    messages = agent_result.get("messages", []) if isinstance(agent_result, dict) else []
    if messages:
        last_message = messages[-1]
        content = getattr(last_message, "content", str(last_message))
        # Remove any internal <think>...</think> sections inserted by the agent
        import re
        content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.S)
        content = content.strip()
        print(content)
    print()

