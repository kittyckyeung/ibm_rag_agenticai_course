from dotenv import dotenv_values
import requests
import json
from typing import Any, Iterable
from langchain_core.messages import convert_to_openai_messages


llm_key_location = "C:\\Users\\user\\Desktop\\Coding\\minimax_Key.txt"
llm_model_name = "MiniMax-M3"
llm_model_base_url = "https://api.minimaxi.chat/v1/text/chatcompletion_v2"

##################################################################
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


##################################################################
llm_api_key, llm_group_id = _load_llm_credentials(llm_key_location)

messages = [
    {"role":"system","content":"You are an expert assistant who provides concise and accurate answers."},
    {"role":"user","content":"What is the capital of Canada?"},
]

response = _llm_runner(messages)

print()
print(response)
print()
print()
print(json.dumps(response, indent=2, ensure_ascii=False))
print()
print()
print(response['choices'][0]['message']['content'])
print()
print()
