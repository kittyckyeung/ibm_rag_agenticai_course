from dotenv import dotenv_values
import requests
import json
from typing import Any, Iterable
from langchain_core.messages import convert_to_openai_messages
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field


llm_key_location = "C:\\Users\\user\\Desktop\\Coding\\minimax_Key.txt"
llm_model_name = "MiniMax-M3"
llm_model_base_url = "https://api.minimaxi.chat/v1/text/chatcompletion_v2"

##################################################################
# Define JSON output structure
class AIResponse(BaseModel):
    summary: str = Field(description="Summary of the user's message")
    sentiment: int = Field(description="Sentiment score from 0 (negative) to 100 (positive)")
    response: str = Field(description="Suggested response to the user")

json_parser = JsonOutputParser(pydantic_object=AIResponse)


def _load_llm_credentials(key_file: str) -> tuple[str, str]:
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


def _llm_runner(prompt_txt: str | Iterable[Any], raise_on_error: bool = True):
    llm_api_key, llm_group_id = _load_llm_credentials(llm_key_location)
    
    print()
    print(f"Raw prompt messages: {prompt_txt}")
    print()

    oai_messages = convert_to_openai_messages(prompt_txt, text_format="string")

    if isinstance(oai_messages, dict):
        messages = [oai_messages]
    else:
        messages = oai_messages

    print()
    print(f"Converted OpenAI messages: {oai_messages}")
    print()

    # More reliable answer generation with a lower temperature and higher top_p/top_k values
    payload = {
        "model": llm_model_name,
        "messages": messages,
        "max_new_tokens": 256,
        "min_new_tokens": 0,
        "temperature": 0.2,
        "top_p": 0.95,
        "top_k": 40,
    }

    headers = {
        "Authorization": f"Bearer {llm_api_key}",
        "GroupId": llm_group_id,
        "Content-Type": "application/json",
    }

    response = requests.post(llm_model_base_url, headers=headers, json=payload, timeout=120)

    print()
    print(f"Response: {response}")
    print()

    parsed = None
    try:
        parsed = response.json()
    except ValueError:
        print("Response body is not valid JSON")

    if raise_on_error:
        response.raise_for_status()

    return parsed if parsed is not None else {"status_code": response.status_code, "text": response.text}


def _format_prompt(variables: dict):
    sys_prompt = variables.get("system_prompt", "")
    usr_prompt = variables.get("user_prompt", "")

    # include JSON format instructions so the model returns structured output
    try:
        format_instructions = json_parser.get_format_instructions()
        full_content = f"{usr_prompt}\n\n{format_instructions}"
    except Exception:
        full_content = usr_prompt

    return [{"role": "system", "content": sys_prompt}, {"role": "user", "content": full_content}]


def invoke_with_prompts(system_p: dict | str, user_p: PromptTemplate):
    """Invoke the runnable chain with given system and user prompts and the country variable.

    Returns the parsed Pydantic object on success, otherwise returns a dict with raw text and raw response.
    """

    chain = RunnableLambda(_format_prompt) | RunnableLambda(_llm_runner)
    
    variables = {"system_prompt": system_p, "user_prompt": user_p}
    resp = chain.invoke(variables)

    # extract assistant text
    if isinstance(resp, dict) and resp.get("choices"):
        try:
            assistant_text = resp["choices"][0]["message"]["content"]
        except Exception:
            assistant_text = json.dumps(resp, ensure_ascii=False)
    else:
        assistant_text = str(resp)

    # try to parse into AIResponse
    try:
        parsed = json_parser.parse(assistant_text)
        return parsed
    except Exception:
        return {"raw_text": assistant_text, "raw_response": resp}


##################################################################
# Create JSON output parser (pass the Pydantic *class*) before calling the LLM
#####json_parser = JsonOutputParser(pydantic_object=AIResponse)
#####llm_api_key, llm_group_id = _load_llm_credentials(llm_key_location)

# Create prompt template and formatting function
#system_prompt = "You are an expert assistant who provides concise and accurate answers."
#user_prompt = "What is the capital of Canada?"

# Build a Runnable chain: format -> call LLM
#####chain = RunnableLambda(_format_prompt) | RunnableLambda(_llm_runner)

# Invoke the chain with variables
#####variables = {"system_prompt": system_prompt, "user_prompt": user_prompt}
#####response = chain.invoke(variables)

#####print()
#####print(json.dumps(response, indent=2, ensure_ascii=False))
#####print()

#####if isinstance(response, dict) and response.get("choices"):
#####    try:
#####        print()
#####        print(response["choices"][0]["message"]["content"])
#####        print()
#####    except Exception:
#####        pass

# Get and show the format instructions used in the prompt
#####formatted_response = json_parser.get_format_instructions()
#####print()
#####print("Format instructions for model:")
#####print(formatted_response)

# Extract assistant text from the chain response and parse it
#####assistant_text = None
#####if isinstance(response, dict):
    # typical LLM reply shape: {'choices': [{'message': {'content': '...'}}]}
#####    try:
#####        assistant_text = response["choices"][0]["message"]["content"]
#####    except Exception:
#####        assistant_text = json.dumps(response, ensure_ascii=False)
#####else:
#####    assistant_text = str(response)

#####print()
#####print("Assistant raw text:")
#####print(assistant_text)

#####try:
#####    parsed_obj = json_parser.parse(assistant_text)
#####    print()
#####    print("Parsed object:")
#####    print(parsed_obj)
#####except Exception as e:
#####    print()
#####    print("JSON parser error:", e)
