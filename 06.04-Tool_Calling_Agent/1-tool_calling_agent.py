# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Suppress pytube errors
import logging
pytube_logger = logging.getLogger('pytube')
pytube_logger.setLevel(logging.ERROR)

# Suppress yt-dlp warnings
yt_dpl_logger = logging.getLogger('yt_dlp')
yt_dpl_logger.setLevel(logging.ERROR)

from dotenv import dotenv_values
from langchain_openai import ChatOpenAI
from youtube_transcript_api import YouTubeTranscriptApi
import re
from pytube import YouTube, Search
from langchain_core.tools import tool
from IPython.display import display, JSON
import yt_dlp
from typing import List, Dict
from langchain_core.messages import HumanMessage
from langchain_core.messages import ToolMessage
import json
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


@tool
def extract_video_id(url: str) -> str:
    """
    Extracts the 11-character YouTube video ID from a URL.
    
    Args:
        url (str): A YouTube URL containing a video ID.

    Returns:
        str: Extracted video ID or error message if parsing fails.
    """
    
    # Regex pattern to match video IDs
    pattern = r'(?:v=|be/|embed/)([a-zA-Z0-9_-]{11})'
    match = re.search(pattern, url)
    return match.group(1) if match else "Error: Invalid YouTube URL"


@tool
def fetch_transcript(video_id: str, language: str = "en") -> str:
    """
    Fetches the transcript of a YouTube video.
    
    Args:
        video_id (str): The YouTube video ID (e.g., "dQw4w9WgXcQ").
        language (str): Language code for the transcript (e.g., "en", "es").
    
    Returns:
        str: The transcript text or an error message.
    """
    
    try:
        ytt_api = YouTubeTranscriptApi()
        transcript = ytt_api.fetch(video_id, languages=[language])
        return " ".join([snippet.text for snippet in transcript.snippets])
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def search_youtube_details(query: str, max_results: int = 5):
    '''
    Searches YouTube for videos matching the query and returns a list of video details.
    
    Args:
        query (str): The search query string.
        max_results (int): Maximum number of results to return.
    
    Returns:
        List[Dict]: A list of dictionaries containing video title, ID, and URL.
    '''
    
    ydl_opts = {"quiet": True, "skip_download": True, 'logger': yt_dpl_logger}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(f"ytsearch{max_results}:{query}", download=False)
    entries = info.get("entries", []) or []
    return [
        {"title": e.get("title"), "video_id": e.get("id"), "url": e.get("webpage_url"), "views": e.get("view_count"), "duration": e.get("duration"), "channel": e.get("uploader"), "likes": e.get("like_count"), "comments": e.get("comment_count"), "chapters": e.get("chapters", [])}
        for e in entries
    ]           


@tool
def get_thumbnails(url: str) -> List[Dict]:
    """
    Get available thumbnails for a YouTube video using its URL.
    
    Args:
        url (str): YouTube video URL (any format)
        
    Returns:
        List of dictionaries with thumbnail URLs and resolutions in YouTube's native order
    """
    
    try:
        with yt_dlp.YoutubeDL({'quiet': True, 'logger': yt_dpl_logger}) as ydl:
            info = ydl.extract_info(url, download=False)
            
            thumbnails = []
            for t in info.get('thumbnails', []):
                if 'url' in t:
                    thumbnails.append({
                        "url": t['url'],
                        "width": t.get('width'),
                        "height": t.get('height'),
                        "resolution": f"{t.get('width', '')}x{t.get('height', '')}".strip('x')
                    })
            
            return thumbnails

    except Exception as e:
        return [{"error": f"Failed to get thumbnails: {str(e)}"}]


if __name__ == "__main__":
    print()
    print(extract_video_id.name)
    print("----------------------------")
    print(extract_video_id.description)
    print("----------------------------")
    print(extract_video_id.func)
    print()
    youtube_id=extract_video_id.run("https://www.youtube.com/watch?v=hfIUstzHs9A")
    print("Extracted YouTube Video ID: ", youtube_id)
    print()
    youtube_transcript=fetch_transcript.run(youtube_id)
    print("Fetched YouTube Transcript: ", youtube_transcript)
    print()
    print("Searching YouTube for 'Generative AI' and getting details of top 5 results...")
    search_out=search_youtube_details.run("Generative AI")
    # Print the raw result so missing output is visible in consoles
    try:
        print(json.dumps(search_out, indent=2, ensure_ascii=False))
    except Exception:
        # Fallback if the returned object isn't JSON-serializable
        print(repr(search_out))
    print()
    print("Getting thumbnails for a YouTube video...")
    thumbnails=get_thumbnails.run("https://www.youtube.com/watch?v=qWHaMrR5WHQ")
    try:
        print(json.dumps(thumbnails, indent=2, ensure_ascii=False))
    except Exception:
        print(repr(thumbnails))
    print()
    print("Demonstrating LLM with tools...")
    agent = create_agent(
        model=call_llm(),
        tools=[extract_video_id, fetch_transcript, search_youtube_details, get_thumbnails],
        system_prompt="You are a helpful assistant that can extract YouTube video IDs, fetch transcripts, search for videos, and get thumbnails using the provided tools.",
    )
    agent_result = agent.invoke(
        {"messages": [{"role": "user", "content": "I want to summarize youtube video: https://www.youtube.com/watch?v=T-D1OfcDW1M in english"}]}
    )
    print("Agent Result: ", agent_result)
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
    tool_mapping = {
    "get_thumbnails" : get_thumbnails,
    "extract_video_id": extract_video_id,
    "fetch_transcript": fetch_transcript,
    "search_youtube": search_youtube_details
    }
    print()
    print("=== Tool Call Details ===")
    results = []
    # message objects may have attributes like `type` and `tool_calls`
    tool_call = None
    for message in messages:
        if getattr(message, "type", None) == "ai" and getattr(message, "tool_calls", None):
            tool_calls = message.tool_calls
            break
    print("Tool Call Results:", json.dumps(tool_calls, indent=2))
    print()
    tool_name=tool_calls[0]['name']
    print(tool_name)
    tool_call_id =tool_calls[0]['id']
    print(tool_call_id)
    args=tool_calls[0]['args']
    print(args)
    my_tool=tool_mapping[tool_calls[0]['name']]
    video_id =my_tool.invoke(tool_calls[0]['args'])
    print(video_id)
    message={
        "messages": [{"role": "user", "content": "I want to summarize youtube video: https://www.youtube.com/watch?v=T-D1OfcDW1M in english"}]
    }
    message["messages"].append({'role': 'assistant', 'content': None, 'tool_calls': [{'id': tool_call_id, 'type': 'function', 'function': {'name': tool_name, 'arguments': args}}]})
    message["messages"].append({"role": "tool", "tool_call_id": tool_call_id, "content": str(video_id)})
    print("Messages: ", message)
    response_2=agent.invoke(message)
    print("response_2: ", response_2)
    print()

