import json
import requests
from dotenv import dotenv_values


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

################################################################################
##prompt = "The wind is "
##### Zero-shot prompt example
##prompt = """Classify the following statement as true or false: 
##            'The Eiffel Tower is located in Berlin.'
##
##            Answer:
##"""
##prompt = """Translate the following statement into French: 
##            'Good morning, how are you?'
##
##            Answer:
##"""
##### Few-shot prompt example
##prompt = """Here is an example of translating a sentence from English to French:
##
##            English: “How is the weather today?”
##            French: “Comment est le temps aujourd'hui?”
##            
##            Now, translate the following sentence from English to French:
##            
##            English: “Where is the nearest supermarket?”
##            
##"""
##prompt = """Here are few examples of classifying emotions in statements:
##
##            Statement: 'I just won my first marathon!'
##            Emotion: Joy
##           
##            Statement: 'I can't believe I lost my keys again.'
##            Emotion: Frustration
##            
##            Statement: 'My best friend is moving to another country.'
##            Emotion: Sadness
##            
##            Now, classify the emotion in the following statement:
##            Statement: 'That movie was so scary I had to cover my eyes.’
##
##"""
##### Chain of thought prompt example
##prompt = """Consider the problem: 'A store had 22 apples. They sold 15 apples today and got a new delivery of 8 apples. 
##            How many apples are there now?’
##
##            Break down each step of your calculation
##
##"""
##prompt = """The laptop is break down with blue screen. How to fix it?
##
##            Break down each step of your solution, and provide if there is any references
##
##"""
##### Self-consistency prompt example
prompt = """When I was 6, my sister was half of my age. Now I am 70, what age is my sister?

            Provide three independent calculations and explanations, then determine the most consistent result.

"""

# Getting a response from the model with the provided prompt and parameters
response = llm_model(prompt)
print(f"prompt: {prompt}\n")
##print("full response text:")
##print(response)
print()
print("response text:")
print(extract_text(response))
print()







