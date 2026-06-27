import json
import requests
from dotenv import dotenv_values

from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableLambda
from langchain_core.messages import HumanMessage, SystemMessage


key_location = "C:\\Users\\user\\Desktop\\Coding\\minimax_Key.txt"
model_name = "MiniMax-M2.7"
model_base_url = "https://api.minimaxi.chat/v1/text/chatcompletion_v2"


def load_credentials(key_file: str) -> tuple[str, str]:
    values = dotenv_values(key_file)
    api_key = values.get("MINIMAX_API_KEY")
    group_id = values.get("MINIMAX_GROUP_ID")

    api_key = api_key.strip() if isinstance(api_key, str) else ""
    group_id = group_id.strip() if isinstance(group_id, str) else ""

    if not api_key:
        raise ValueError("MINIMAX_API_KEY is missing or empty.")
    if not group_id:
        raise ValueError("MINIMAX_GROUP_ID is missing or empty.")

    return api_key, group_id


api_key, group_id = load_credentials(key_location)


def _masked_headers(headers: dict) -> dict:
    masked = dict(headers)
    auth = masked.get("Authorization")
    if isinstance(auth, str) and len(auth) > 12:
        masked["Authorization"] = f"{auth[:10]}...{auth[-4:]}"
    return masked


def llm_model(prompt_txt: str, params: dict | None = None, raise_on_error: bool = True):
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt_txt}],
        "max_new_tokens": 256,
        "min_new_tokens": 0,
        "temperature": 0.5,
        "top_p": 0.2,
        "top_k": 1
    }

    if params:
        payload.update(params)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "GroupId": group_id,
        "Content-Type": "application/json",
    }

    response = requests.post(model_base_url, headers=headers, json=payload, timeout=120)
    parsed = None
    try:
        parsed = response.json()
    except ValueError:
        parsed = None
        print("Response body is not valid JSON.")

    ##print()
    ##print(f"Request payload: {json.dumps(payload, indent=2)}")
    ##print(f"Request headers (masked): {json.dumps(_masked_headers(headers), indent=2)}")
    ##print(f"Response object: {response}")
    ##print(f"Status code: {response.status_code}")
    ##print(f"Reason: {response.reason}")
    ##print(f"URL: {response.url}")
    ##print(f"Response headers: {json.dumps(dict(response.headers), indent=2)}")
    ##print(f"Raw response text: {response.text}")
    ##print()
    ##if parsed is not None:
    ##    print("Response JSON (all fields):")
    ##    print(json.dumps(parsed, indent=2, ensure_ascii=False))
    ##else:
    ##    print("Response body is not valid JSON.")
    ##print()

    if raise_on_error:
        response.raise_for_status()
    return parsed if parsed is not None else {"status_code": response.status_code, "text": response.text}


# Wrap the HTTP LLM call in a small runnable function that returns text
def llm_runner(prompt_text: str) -> str:
    result = llm_model(prompt_text)
    # If llm_model returned an error dict, raise so the runnable errors visibly
    if isinstance(result, dict) and result.get("status_code") and not result.get("choices"):
        raise RuntimeError(f"LLM request failed: {result.get('status_code')} - {result.get('text')}")
    # Extract text from the provider response and return plain string
    ##print()
    ##print("extract response text:")
    ##print(extract_text(result))
    ##print("response text:")
    ##print(result)
    ##print()
    if isinstance(result, dict):
        return extract_text(result)
    return str(result)


def extract_text(result: dict) -> str:
    # Handle common provider response shapes safely.
    choices = result.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
            text = first.get("text")
            if isinstance(text, str):
                return text
    reply = result.get("reply")
    if isinstance(reply, str):
        return reply
    return "No text field found in response; inspect full JSON output."


# Define a function to ensure proper formatting
def format_prompt(variables):
    return prompt.format(**variables)

################################################################################
##template = """Tell me a {adjective} joke about {content}.
##"""
##prompt = PromptTemplate.from_template(template)
##
##joke_chain = (
##    RunnableLambda(format_prompt)
##    | RunnableLambda(llm_runner)
##    | StrOutputParser()
##)
##
##vars = {"adjective": "sad", "content": "fish"}
##response = joke_chain.invoke(vars)
##print(f"prompt: {format_prompt(vars)}\n")
##print("joke:")
##print(response)
################################################################################
##content = """
##    The rapid advancement of technology in the 21st century has transformed various industries, including healthcare, education, and transportation. 
##    Innovations such as artificial intelligence, machine learning, and the Internet of Things have revolutionized how we approach everyday tasks and complex problems. 
##    For instance, AI-powered diagnostic tools are improving the accuracy and speed of medical diagnoses, while smart transportation systems are making cities more efficient and reducing traffic congestion. 
##    Moreover, online learning platforms are making education more accessible to people around the world, breaking down geographical and financial barriers. 
##    These technological developments are not only enhancing productivity but also contributing to a more interconnected and informed society.
##"""

##template = """Summarize the {content} in one sentence.
##"""

##prompt = PromptTemplate.from_template(template)

##summarize_chain = (
##    RunnableLambda(format_prompt)
##    | RunnableLambda(llm_runner)
##    | StrOutputParser()
##)

##vars = {"content": content}
##response = summarize_chain.invoke(vars)
##print(f"prompt: {format_prompt(vars)}\n")
##print("summary:")
##print(response)
################################################################################
##content = """
##    The solar system consists of the Sun, eight planets, their moons, dwarf planets, and smaller objects like asteroids and comets. 
##    The inner planets—Mercury, Venus, Earth, and Mars—are rocky and solid. 
##    The outer planets—Jupiter, Saturn, Uranus, and Neptune—are much larger and gaseous.
##"""

##question = "Which planets in the solar system are rocky and solid?"

##template = """
##    Answer the {question} based on the {content}.
##    Respond "Unsure about answer" if not sure about the answer.
    
##    Answer:
    
##"""

##prompt = PromptTemplate.from_template(template)

##qa_chain = (
##    RunnableLambda(format_prompt)
##    | RunnableLambda(llm_runner)
##    | StrOutputParser()
##)

##vars = {"question": question, "content": content}
##response = qa_chain.invoke(vars)
##print(f"prompt: {format_prompt(vars)}\n")
##print("summary:")
##print(response)
################################################################################
##text = """
##    The concert last night was an exhilarating experience with outstanding performances by all artists.
##"""

##categories = "Entertainment, Food and Dining, Technology, Literature, Music."

##template = """
##    Classify the {text} into one of the {categories}.
    
##    Category:
    
##"""

##prompt = PromptTemplate.from_template(template)

##qa_chain = (
##    RunnableLambda(format_prompt)
##    | RunnableLambda(llm_runner)
##    | StrOutputParser()
##)

##vars = {"text": text, "categories": categories}
##response = qa_chain.invoke(vars)
##print(f"prompt: {format_prompt(vars)}\n")
##print("summary:")
##print(response)
################################################################################
##description = """
##    Retrieve the names and email addresses of all customers from the 'customers' table who have made a purchase in the last 30 days. 
##    The table 'purchases' contains a column 'purchase_date'
##"""

##template = """
##    Generate an SQL query based on the {description}
    
##    SQL Query:
    
##"""

##prompt = PromptTemplate.from_template(template)

##qa_chain = (
##    RunnableLambda(format_prompt)
##    | RunnableLambda(llm_runner)
##    | StrOutputParser()
##)

##vars = {"description": description}
##response = qa_chain.invoke(vars)
##print(f"prompt: {format_prompt(vars)}\n")
##print("summary:")
##print(response)
################################################################################
##role = """
##    Dungeon & Dragons game master
##"""

##tone = "engaging and immersive"

##template = """
##    You are an expert {role}. I have this question {question}. I would like our conversation to be {tone}.
    
##    Answer:
    
##"""

##prompt = PromptTemplate.from_template(template)

##qa_chain = (
##    RunnableLambda(format_prompt)
##    | RunnableLambda(llm_runner)
##    | StrOutputParser()
##)

# Create an interactive chat loop
##while True:
##    query = input("Question: ")

##    if query.lower() in ["quit", "exit", "bye"]:
##        print("Answer: Goodbye!")
##        break

##    vars = {"role": role, "question": query, "tone": tone}

##    response = qa_chain.invoke(vars)
##    print("Answer: ", response)
################################################################################
template = """
Analyze the following product review:
"{review}"

Provide your analysis in the following format:
- Sentiment: (positive, negative, or neutral)
- Key Features Mentioned: (list the product features mentioned)
- Summary: (one-sentence summary)
"""

reviews = [
    "I love this smartphone! The camera quality is exceptional and the battery lasts all day. The only downside is that it heats up a bit during gaming.",
    "This laptop is terrible. It's slow, crashes frequently, and the keyboard stopped working after just two months. Customer service was unhelpful."
]

prompt = PromptTemplate.from_template(template)

qa_chain = (
    RunnableLambda(format_prompt)
    | RunnableLambda(llm_runner)
    | StrOutputParser()
)

for review in reviews:
    vars = {"review": review}
    response = qa_chain.invoke(vars)
    print("Answer: ", response)
    pass



