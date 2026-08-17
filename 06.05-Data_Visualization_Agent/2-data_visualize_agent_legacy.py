## Half done attempt to create a data visualization agent using langchain and MiniMax API. The code below is a legacy version that may not work as intended due to changes in the langchain library and MiniMax API. It is provided for reference purposes only.
def warn(*args, **kwargs):
    pass
from typing import Final
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

from dotenv import dotenv_values
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent


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


if __name__ == "__main__":
    print()

    # Get the data
    df = pd.read_csv(
        "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/ZNoKMJ9rssJn-QbJ49kOzA/student-mat.csv"
    )
    print("DataFrame Head: ", df.head(5))
    print()
    print("DataFrame Info: ", df.info())
    print()
    agent = create_pandas_dataframe_agent(
        call_llm(),
        df,
        verbose=True,
        return_intermediate_steps=True,  # set return_intermediate_steps=True so that model could return code that it comes up with to generate the chart
        handle_parsing_errors=True,
        allow_dangerous_code=True,
        prefix="You are a pandas agent. Always respond using Action/Action Input/Final Answer format. Never output raw data directly."
    )
    print()
    print("Number of rows in DataFrame: ", len(df))
    print()
    # 395
    #response = agent.invoke("how many rows of data are in this file?")
    #print("Row of Data, Agent Result: ", response)
    print()
    #print("Agent Result (intermediate_steps): ", response['intermediate_steps'][-1][0].tool_input.replace('; ', '\n'))
    print()
    # 16
    #response = agent.invoke("Give me all the data where student's age is over 18 years old.")
    #print("Students Over 18, Agent Result: ", response)
    print()
    #print("Agent Result (intermediate_steps): ", response['intermediate_steps'][-1][0].tool_input.replace('; ', '\n'))
    print()
    
    ## Thought: The bar chart was generated successfully. The gender count shows F (Female) = 208 and M (Male) = 187.
    ## **Final Answer**: A bar chart has been generated successfully showing the gender count in the dataframe. The chart displays:
    ## **Female (F)**: 208
    ## **Male (M)**: 187
    ## The bar chart was created using matplotlib with appropriate labels, title, and gridlines, and saved as 'gender_count.png'.
    
    #response = agent.invoke("Generate a bar chart to plot the gender count.")
    #print("Bar Chart, Agent Result: ", response)
    print()
    #print("Agent Result (intermediate_steps): ", response['intermediate_steps'][-1][0].tool_input.replace('; ', '\n'))
    print()
    
    ## Final Answer: I generated a pie chart displaying the average value of Walc for each Gender. The chart shows:
    ## **Female (F)**: Average Walc ≈ 2.09
    ## **Male (M)**: Average Walc ≈ 2.65
    ## Males have a higher average weekend alcohol consumption (Walc) than females. The pie chart visualizes the proportion of these averages, with each slice labeled by gender and showing the percentage of the total.
    
    response = agent.invoke("Generate a pie chart to display average value of Walc for each Gender.")
    print("Pie Chart, Agent Result: ", response)
    print()
    #print("Agent Result (intermediate_steps): ", response['intermediate_steps'][-1][0].tool_input.replace('; ', '\n'))
    print()
    
    response = agent.invoke("Create box plots to analyze the relationship between 'freetime' (amount of free time) and 'G3' (final grade) across different levels of free time.")
    print("Box Polts, Agent Result: ", response)
    print()
    #print("Agent Result (intermediate_steps): ", response['intermediate_steps'][-1][0].tool_input.replace('; ', '\n'))
    print()
    response = agent.invoke("Generate scatter plots to examine the correlation between 'Dalc' (daily alcohol consumption) and 'G3', and between 'Walc' (weekend alcohol consumption) and 'G3'.")
    print("Scatter Plots, Agent Result: ", response)
    print()
    #print("Agent Result (intermediate_steps): ", response['intermediate_steps'][-1][0].tool_input.replace('; ', '\n'))
    print()
