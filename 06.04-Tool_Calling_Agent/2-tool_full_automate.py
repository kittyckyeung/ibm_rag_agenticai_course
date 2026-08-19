## Convert the input prompt to a HumanMessage
## Pass the message to LLM with tools
## Extract tool calls from LLM response
## Update message history with tool results
## Send updated messages back to LLM
## Repeat steps 3-5 as needed
## Finally, extract just the content from the final message using RunnableLambda

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
from langchain_core.runnables import RunnablePassthrough, RunnableLambda


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


tool_mapping = {
    "get_thumbnails" : get_thumbnails,
    "extract_video_id": extract_video_id,
    "fetch_transcript": fetch_transcript,
    "search_youtube_details": search_youtube_details
}

# Define the processing steps
def execute_tool(tool_call):
    """Execute single tool call and return ToolMessage"""
    try:
        # Support both dict-style and object-style tool_call
        if isinstance(tool_call, dict):
            name = tool_call.get("name")
            args = tool_call.get("args", {})
            call_id = tool_call.get("id")
        else:
            name = getattr(tool_call, "name", None)
            args = getattr(tool_call, "args", {})
            call_id = getattr(tool_call, "id", None)

        fn = tool_mapping.get(name)
        if fn is None:
            raise ValueError(f"Unknown tool: {name}")

        # Call tool with dict or seq args
        if isinstance(args, dict):
            result = fn(**args)
        elif isinstance(args, (list, tuple)):
            result = fn(*args)
        else:
            result = fn(args)

        # Return a serializable dict representing the tool result
        return {
            "role": "tool",
            "name": name,
            "content": str(result),
            "tool_call_id": call_id,
        }
    except Exception as e:
        return {
            "role": "tool",
            "name": None,
            "content": f"Error: {str(e)}",
            "tool_call_id": (tool_call.get("id") if isinstance(tool_call, dict) else getattr(tool_call, "id", None)),
        }





if __name__ == "__main__":
    print()
    # Usage
    agent = create_agent(
        model=call_llm(),
        tools=[extract_video_id, fetch_transcript, search_youtube_details, get_thumbnails],
        system_prompt="You are a helpful assistant that can extract YouTube video IDs, fetch transcripts, search for videos, and get thumbnails using the provided tools.",
    )
    # Keep a RunnablePassthrough-based chain, but use serializable dict messages
    summarization_chain = (
        RunnablePassthrough.assign(
            messages=lambda x: [{"role": "user", "content": x["query"]}]
        )
        | RunnablePassthrough.assign(
            ai_response=lambda x: agent.invoke({"messages": x["messages"]})
        )
        | RunnablePassthrough.assign(
            ai_dict=lambda x: {"role": "assistant", "content": getattr(x["ai_response"], "content", str(x["ai_response"]))}
        )
        | RunnablePassthrough.assign(
            tool_messages=lambda x: [execute_tool(tc) for tc in getattr(x["ai_response"], "tool_calls", [])]
        )
        | RunnablePassthrough.assign(
            messages=lambda x: x["messages"] + [x["ai_dict"]] + x["tool_messages"]
        )
        | RunnablePassthrough.assign(
            ai_response2=lambda x: agent.invoke({"messages": x["messages"]})
        )
        | RunnablePassthrough.assign(
            ai_dict2=lambda x: {"role": "assistant", "content": getattr(x["ai_response2"], "content", str(x["ai_response2"]))}
        )
        | RunnablePassthrough.assign(
            tool_messages2=lambda x: [execute_tool(tc) for tc in getattr(x["ai_response2"], "tool_calls", [])]
        )
        | RunnablePassthrough.assign(
            messages=lambda x: x["messages"] + [x["ai_dict2"]] + x["tool_messages2"]
        )
        | RunnablePassthrough.assign(
            final_response=lambda x: agent.invoke({"messages": x["messages"]})
        )
        | RunnablePassthrough.assign(
            summary=lambda x: getattr(x["final_response"], "content", str(x["final_response"]))
        )
        | RunnableLambda(lambda x: x["summary"])
    )

    result = summarization_chain.invoke({
        "query": "Summarize this YouTube video: https://www.youtube.com/watch?v=1bUy-1hGZpI"
    })
    print("Video Summary:\n", result)

